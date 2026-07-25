"""Gandalf Protocol — Best-of-N + judge-as-verifier tests (Round-2 Phase 1) [OWNER: Kirk]

Deterministic and MOCK-safe. Two layers of coverage:

  1. Selection logic — monkeypatch runner._single_episode with CONTROLLED per-candidate scores
     and prove _episode_for/_best_of_n_episode builds exactly N candidates and keeps the
     max-thoughtfulness one (ties → the first). No API key, no cost, no randomness.
  2. Real MOCK chain — with a live mock propose→react→judge chain, prove the call count is
     bounded to exactly N × the single-candidate chain (verifier = the panel judge), and that
     the integrated pipeline keeps the highest-scoring candidate when the verifier is controlled.

Run from repo root:  python3 -m pytest test_best_of_n.py -q
"""
import config, llm, runner
from contracts import Persona, Episode, Playbook, Score
from agents.swarm import GiftAgent
import pytest


# ── helpers ──

def _persona():
    return Persona(
        profile={"interests": [{"value": "trail running", "polarity": "loves"}],
                 "already_owns": [], "constraints": [], "personality": {}},
        hidden_truth={"would_love": ["trail running gear"], "would_return": [],
                      "already_has": []},
        occasion="birthday", budget={"min": 20, "max": 100})


def _ep(thoughtfulness, tag="c"):
    """A minimal Episode with a controllable score dict (same shape store.save persists)."""
    return Episode(persona_id="p", agent_id="a", strategy_tag=tag,
                   proposal={"gifts": []}, reaction={},
                   score={"thoughtfulness": thoughtfulness},
                   playbook_version=0, condition="swarm", occasion="birthday", round_index=0)


def _seq_single(monkeypatch, episodes):
    """Replace runner._single_episode so _best_of_n_episode consumes a fixed, ordered sequence
    of pre-scored candidate Episodes. Returns a call-counter dict."""
    it = iter(episodes)
    calls = {"n": 0}

    def fake(persona, agent, playbook, condition, round_index, is_validation=False):
        calls["n"] += 1
        return next(it)

    monkeypatch.setattr(runner, "_single_episode", fake)
    return calls


# ── (1) selection logic (no MOCK/LLM needed — _single_episode is fully stubbed) ──

def test_best_of_n_keeps_highest(monkeypatch):
    monkeypatch.setattr(config, "BEST_OF_N", 3)
    calls = _seq_single(monkeypatch, [_ep(0.2, "c1"), _ep(0.8, "c2"), _ep(0.5, "c3")])
    ep = runner._episode_for(_persona(), None, Playbook(), "swarm", 0)
    assert calls["n"] == 3                       # exactly N candidates built
    assert ep.score["thoughtfulness"] == 0.8     # the max is kept
    assert ep.strategy_tag == "c2"               # ...and it's the right candidate object


def test_kept_score_equals_max_over_candidates(monkeypatch):
    vals = [0.3, 0.6, 0.6, 0.1]
    monkeypatch.setattr(config, "BEST_OF_N", len(vals))
    _seq_single(monkeypatch, [_ep(v) for v in vals])
    ep = runner._episode_for(_persona(), None, Playbook(), "swarm", 0)
    assert ep.score["thoughtfulness"] == max(vals)


def test_ties_resolve_to_first(monkeypatch):
    monkeypatch.setattr(config, "BEST_OF_N", 3)
    _seq_single(monkeypatch, [_ep(0.6, "first"), _ep(0.6, "second"), _ep(0.4, "third")])
    ep = runner._episode_for(_persona(), None, Playbook(), "swarm", 0)
    assert ep.strategy_tag == "first"            # two-way tie at 0.6 → the earliest candidate


def test_n1_is_exactly_the_single_chain(monkeypatch):
    monkeypatch.setattr(config, "BEST_OF_N", 1)
    sentinel = _ep(0.42, "solo")
    calls = _seq_single(monkeypatch, [sentinel])
    ep = runner._episode_for(_persona(), None, Playbook(), "swarm", 0)
    assert calls["n"] == 1                        # a single chain, no extra candidates
    assert ep is sentinel                         # returned unchanged (same object, same shape)


def test_n1_returned_episode_shape_unchanged(monkeypatch):
    # BEST_OF_N=1 must leave the Episode contract byte-for-byte: same fields store.save reads.
    monkeypatch.setattr(config, "BEST_OF_N", 1)
    sentinel = _ep(0.5, "shape")
    _seq_single(monkeypatch, [sentinel])
    ep = runner._episode_for(_persona(), None, Playbook(), "swarm", 0, is_validation=True)
    for f in ("persona_id", "agent_id", "strategy_tag", "proposal", "reaction", "score",
              "playbook_version", "condition", "occasion", "round_index", "episode_id"):
        assert hasattr(ep, f)


# ── (2) real MOCK chain — cost bound + integrated selection ──

def test_call_count_is_bounded_to_n_times_chain(monkeypatch):
    if not config.MOCK:
        pytest.skip("cost-bound assertion is deterministic only in MOCK mode")
    monkeypatch.setattr(config, "JUDGE_PANEL_SIZE", 1)     # single-judge → fixed per-chain cost
    agent, p, pb = GiftAgent("bold-experiences"), _persona(), Playbook()

    monkeypatch.setattr(config, "BEST_OF_N", 1)
    llm.reset_calls()
    runner._episode_for(p, agent, pb, "swarm", 0)
    per_chain = llm.calls_made()
    assert per_chain >= 3        # propose + react + judge(panel=1) at minimum

    monkeypatch.setattr(config, "BEST_OF_N", 3)
    llm.reset_calls()
    runner._episode_for(p, agent, pb, "swarm", 0)
    n3 = llm.calls_made()
    assert n3 == 3 * per_chain    # exactly N× — counted through llm.calls_made()
    assert n3 <= 3 * per_chain    # bounded ≤ N× the single-candidate chain (DoD)


def test_integration_picks_max_via_real_verifier(monkeypatch):
    if not config.MOCK:
        pytest.skip("real propose/react chain needs MOCK to stay deterministic and free")
    monkeypatch.setattr(config, "BEST_OF_N", 3)
    # Everything (propose, react, Episode assembly) is the REAL mock chain; only the verifier's
    # returned thoughtfulness is controlled so the kept candidate is deterministic.
    scores = iter([0.1, 0.9, 0.4])

    def fake_judge(persona, proposal, reaction, playbook_len=0):
        return Score(proposal_id=proposal.proposal_id, dims={},
                     thoughtfulness=next(scores), rationale="x")

    monkeypatch.setattr(runner.signal, "judge", fake_judge)
    ep = runner._episode_for(_persona(), GiftAgent("bold-experiences"), Playbook(), "swarm", 0)
    assert ep.score["thoughtfulness"] == 0.9      # the best of the three candidates
