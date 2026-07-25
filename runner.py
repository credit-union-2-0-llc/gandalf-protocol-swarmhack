"""Gandalf Protocol — The Orchestrator [OWNER: Kirk]

Connects all agents, judges, and coaches. The loop runs experiments with configurable
conditions (solo, swarm, ablation). Automatically selects local JSON or Azure Blob Storage
based on config.USE_AZURE.
"""
from concurrent.futures import ThreadPoolExecutor
from contracts import Playbook, Episode, CurriculumBatch, to_json
from agents.swarm import Swarm
from agents import world, signal
import config, llm

# [Rig perf-glue] Fan out the independent propose→react→judge chains WITHIN a round. Rounds
# stay sequential (round N+1 must read round N's distilled playbook) and distill is still a
# post-round barrier, so learning semantics are identical — only wall-clock drops (~Wx).
# Forced to 1 in MOCK so the deterministic mock curve / never-cut trio is unchanged.
_WORKERS = 1 if config.MOCK else int(__import__("os").environ.get("PARALLEL_WORKERS", "8"))


def _single_episode(persona, agent, playbook, condition, round_index, is_validation=False):
    """One independent chain: propose → react → judge → Episode. Pure/no shared writes,
    so many run concurrently; the caller saves results sequentially."""
    proposal = agent.propose(persona.profile, playbook, config.GIFTS_PER_PROPOSAL)
    reaction = world.react(persona, proposal)
    score = signal.judge(persona, proposal, reaction, playbook.version)
    return Episode(persona_id=persona.persona_id, agent_id=agent.agent_id,
                   strategy_tag=agent.strategy_tag, proposal=to_json(proposal),
                   reaction=to_json(reaction), score=to_json(score),
                   playbook_version=playbook.version, is_validation=is_validation,
                   condition=condition, occasion=persona.occasion, round_index=round_index)


# [Round-2 Phase 1] Best-of-N + judge-as-verifier. `signal.judge` IS the verifier — and since
# Phase 2 it's automatically a majority-vote PANEL when JUDGE_PANEL_SIZE>1, so we get a stronger
# verifier for free just by calling judge(). Because judge() needs a Reaction, each candidate's
# score comes from a FULL propose→react→judge chain; we keep the candidate whose
# Score.thoughtfulness is highest (ties → the first, matching Python max()).
def _best_of_n_episode(persona, agent, playbook, condition, round_index, is_validation=False):
    """Build N candidate chains (config.BEST_OF_N) and return the max-thoughtfulness Episode.
    Cost is bounded: exactly N × the single-candidate chain (N propose + N react + N judge[panel]
    calls), all counted through llm.calls_made(). Ties resolve to the first candidate. The
    returned Episode has the SAME shape/fields as _single_episode's, so store.save + the
    dashboard are unaffected — Best-of-N only changes WHICH candidate is kept, not the contract."""
    n = max(1, int(getattr(config, "BEST_OF_N", 1) or 1))
    candidates = [_single_episode(persona, agent, playbook, condition, round_index, is_validation)
                  for _ in range(n)]
    # max() returns the first element on a tie, so equal-scoring candidates → the first proposed.
    return max(candidates, key=lambda ep: ep.score["thoughtfulness"])


def _episode_for(persona, agent, playbook, condition, round_index, is_validation=False):
    """The per-(recipient, agent) chain the runner fans out. BEST_OF_N=1 (default) is exactly
    today's single propose→react→judge Episode, byte-for-byte; BEST_OF_N=N>1 verifies N
    candidates with signal.judge and keeps the best. Every caller (run_round, validate) is
    unchanged — the Best-of-N switch lives entirely behind this one entrypoint."""
    if max(1, int(getattr(config, "BEST_OF_N", 1) or 1)) == 1:
        return _single_episode(persona, agent, playbook, condition, round_index, is_validation)
    return _best_of_n_episode(persona, agent, playbook, condition, round_index, is_validation)


def _run_chains(tasks):
    """tasks: list of no-arg thunks returning an Episode. Concurrent, order preserved."""
    if _WORKERS <= 1:
        return [t() for t in tasks]
    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        return list(ex.map(lambda t: t(), tasks))

# Auto-select store backend based on environment
if config.USE_AZURE:
    from store_azure import Store
else:
    from store import Store


def run_round(swarm, playbook, store, batch, condition, round_index, distill=True):
    personas = world.generate_personas(batch)
    # All (persona, agent) chains in a round read the SAME playbook (vN-1) and don't write
    # shared state → safe to run concurrently. Bind loop vars via default args.
    tasks = [(lambda p=p, a=a: _episode_for(p, a, playbook, condition, round_index))
             for p in personas for a in swarm.agents]
    for ep in _run_chains(tasks):
        store.save(ep)   # saved sequentially in the main thread — no store.json race
    # distill AFTER the round → shared playbook grows (skipped in solo).
    # [Rig test harness] The `distill` flag (or NO_DISTILL=1 env) skips distillation so we
    # can A/B a 4-agent swarm WITH vs WITHOUT the shared playbook (same strategy mix) — the
    # only clean test of whether collective learning actually helps, isolated from strategy
    # composition. The OFF arm runs as condition "swarm_nodistill" so the two are separable.
    import os as _os
    if distill and not _os.environ.get("NO_DISTILL"):
        top = store.top_episodes(n=5, condition=condition)
        # [Round-2 Phase 3] When ACE_CURATION is on, also feed the round's worst episodes so
        # the Reflector can emit GEPA-style "avoid" lessons. Flag OFF => bottom stays None =>
        # distill() runs its byte-identical legacy path. held_out_eval stays None here (the
        # real per-lesson gate needs the live LLM judge and only runs deployed).
        bottom = store.bottom_episodes(n=5, condition=condition) if config.ACE_CURATION else None
        swarm.distill(top, playbook, bottom_episodes=bottom)


def validate(swarm, playbook, store, held_out, round_index):
    """[Rig fix P0-3] Snapshot the CURRENT policy against the held-out set every round,
    tagged with round_index, so the validation line rises alongside training instead of
    being a single end-of-run dot. This periodic curve is also our reward-hacking canary:
    if training climbs while this stays flat, the swarm is gaming the simulator."""
    agent = swarm.agents[0]
    tasks = [(lambda p=p: _episode_for(p, agent, playbook, "validation", round_index,
                                       is_validation=True))
             for p in held_out]
    for ep in _run_chains(tasks):
        store.save(ep)


def run_experiment(store, rounds, condition, personas_per_round, held_out=None, on_round=None):
    llm.reset_calls()   # [P1] cost guard is per-RUN, not per-process
    solo = condition.startswith("solo")
    # "swarm_nodistill" = full swarm, but the shared playbook never grows (A/B OFF arm).
    distill = not condition.endswith("nodistill")
    swarm = Swarm(size=config.SWARM_SIZE, solo=solo)
    playbook = Playbook()
    for r in range(rounds):
        batch = signal.next_curriculum(store, personas_per_round)
        run_round(swarm, playbook, store, batch, condition, round_index=r, distill=distill)
        # validate against the held-out set after every round (periodic, per P0-3)
        if held_out:
            validate(swarm, playbook, store, held_out, round_index=r)
        print(f"  [{condition}] round {r+1}/{rounds} done — "
              f"playbook v{playbook.version}, calls={llm.calls_made()}")
        # Checkpoint AFTER each round (charts + playbook), so the dashboard curve grows live
        # and a run that dies mid-way still leaves a real, published result (fixes #38's
        # "nothing publishes because the run never reaches the end"). Never let a checkpoint
        # failure break the round loop.
        if on_round:
            try:
                on_round(r, playbook)
            except Exception as e:
                print(f"  [checkpoint] round {r} skipped: {e}")
    return playbook
