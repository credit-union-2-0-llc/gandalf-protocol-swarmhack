"""Gandalf Protocol — Regression guard: training episodes must persist [OWNER: Kirk]

Run from repo root: `python test_training_persists.py`

Guards issue #17. A run must land TRAINING episodes (not only held-out validation),
populate score_history, and produce a per-round training series. When training episodes
go missing (`training_episodes == 0`, `score_history == {}`) the learning curve is flat,
the coach gets no signal, and the swarm never distills — the exact live symptom.

The core loop is correct on HEAD (this test passes), so a live flat curve points at a
stale deployed image, not the logic — this guard fails CI the moment either regresses.

Runs in MOCK (zero cost) and exercises BOTH the serial and real-mode parallel
(`WORKERS > 1`) save paths, since real runs fan out chains.
"""
import os
os.environ.pop("ANTHROPIC_API_KEY", None)  # force MOCK — deterministic, zero cost
import sys
import config, runner
from store import Store
from agents import world

FAILURES = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


def _run(workers):
    runner._WORKERS = workers
    st = Store()
    st.episodes = []
    held = world.load_seed_personas()
    for p in held:
        p.is_validation = True  # mirrors app.py _run_worker
    runner.run_experiment(st, rounds=3, condition="swarm",
                          personas_per_round=config.PERSONAS_PER_ROUND, held_out=held)
    return st


for workers in (1, 8):
    print(f"\n-- run_experiment(swarm, 3 rounds), WORKERS={workers} --")
    st = _run(workers)
    train, val = st.training_episodes(), st.validation_episodes()
    expected_train = config.PERSONAS_PER_ROUND * config.SWARM_SIZE * 3
    check(len(train) > 0, f"training episodes persist (got {len(train)})")
    check(len(train) == expected_train,
          f"training count == personas*agents*rounds ({len(train)} == {expected_train})")
    check(len(val) > 0, f"held-out validation episodes persist (got {len(val)})")
    check(len(st.episodes) == len(train) + len(val),
          f"total == training + validation ({len(st.episodes)} == {len(train)}+{len(val)})")
    check(bool(st.score_history_summary()),
          f"score_history populated (got {st.score_history_summary()})")
    check(len(st.round_series("swarm")) == 3,
          f"training series spans all 3 rounds (got {st.round_series('swarm')})")
    check(len(st.validation_series_by_round()) == 3,
          "validation series spans all 3 rounds")

if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S) — issue #17 invariant broken:")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("\nAll training-persistence invariants hold (issue #17 guard).")
