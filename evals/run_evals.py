"""Gandalf Protocol — eval harness [OWNER: Kirk]

Tracks whether the system actually learns and catches regressions / reward-hacking as
the team iterates. Run after an experiment (reads store.json) + a gold set for the judge.

    python run.py --ablation --rounds 6      # produce a store.json
    python evals/run_evals.py                # grade it

Checks (grounded in what we hit + the research):
  1. calibration   — judge agreement (Cohen's κ) vs a hand-labeled gold set   [needs real judge]
  2. constraint    — bad gift (already_owns / constraint / dup) must score < 0.4  [needs real judge]
  3. learning      — swarm final-round mean > solo final-round mean (does it learn?)
  4. canary        — validation rises with training, not flat-while-train-climbs (reward-hack)
Deterministic checks (3,4) gate CI. LLM-judged checks (1,2) log a trend, per the
"eval as tripwire, not selector" lesson. Exit non-zero if any GATING check fails.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from store import Store  # noqa: E402


def _cohens_kappa(pairs):
    """pairs: list of (predicted_good: bool, labeled_good: bool). Binary κ."""
    n = len(pairs)
    if not n:
        return None
    agree = sum(1 for p, l in pairs if p == l) / n
    pp = sum(p for p, _ in pairs) / n
    pl = sum(l for _, l in pairs) / n
    chance = pp * pl + (1 - pp) * (1 - pl)
    return (agree - chance) / (1 - chance) if chance < 1 else 1.0


def check_learning(store):
    """Noise-robust: compare OVERALL means (per-round finals bounce hard at small N) and
    report the swarm learning SLOPE. Gates on 'swarm not materially worse than solo' — that
    catches real regressions without false-failing on a noisy final round."""
    import statistics
    solo, swarm = store.round_series("solo"), store.round_series("swarm")
    if not solo or not swarm:
        return ("learning", None, "need both solo+swarm rounds (run --ablation)")
    solo_m = statistics.mean(y for _, y in solo)
    swarm_m = statistics.mean(y for _, y in swarm)
    ys = [y for _, y in swarm]; xs = list(range(len(ys)))
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / (sum((x-mx)**2 for x in xs) or 1)
    delta = swarm_m - solo_m
    trend = "← FLAT: no learning trend yet (needs more/harder personas)" if abs(slope) < 0.02 else \
            ("← climbing" if slope > 0 else "← DECLINING")
    return ("learning", delta >= -0.03,
            f"swarm mean {swarm_m:.2f} vs solo {solo_m:.2f} (Δ{delta:+.2f}); slope {slope:+.3f}/round {trend}")


def check_canary(store):
    val = store.validation_series_by_round()
    swarm = store.round_series("swarm")
    if len(val) < 2 or len(swarm) < 2:
        return ("canary", None, "need ≥2 rounds of validation+swarm")
    train_rise = swarm[-1][1] - swarm[0][1]
    val_rise = val[-1][1] - val[0][1]
    # reward-hacking smell: training climbs hard while held-out stays flat/drops
    hacking = train_rise > 0.1 and val_rise < 0.02
    return ("canary", not hacking, f"train Δ{train_rise:+.2f}, held-out Δ{val_rise:+.2f}"
            + ("  ← REWARD-HACKING SMELL" if hacking else ""))


def check_calibration(gold_path):
    """κ of the real judge vs hand labels. Skips gracefully if no key / empty gold set."""
    if not os.path.exists(gold_path):
        return ("calibration", None, "no gold_set.json yet (Jasper: fill it — see #5)")
    gold = json.load(open(gold_path)).get("examples", [])
    if not gold:
        return ("calibration", None, "gold_set.json is empty (Jasper: add labeled examples)")
    import config
    if config.MOCK:
        return ("calibration", None, "MOCK mode — set the gateway key to grade the real judge")
    from contracts import Persona, GiftProposal, Reaction
    from agents import signal
    pairs = []
    for ex in gold:
        persona = Persona(profile=ex.get("profile", {}), hidden_truth=ex["hidden_truth"],
                          occasion=ex.get("occasion", "birthday"), budget=ex.get("budget", {"min": 0, "max": 100}))
        prop = GiftProposal(agent_id="eval", strategy_tag="eval", gifts=ex["gifts"])
        react = Reaction(proposal_id=prop.proposal_id, persona_id=persona.persona_id,
                         verdict=ex.get("verdict", "meh"), quote="")
        score = signal.judge(persona, prop, react)
        pairs.append((score.thoughtfulness >= 0.6, bool(ex["label_good"])))
    k = _cohens_kappa(pairs)
    return ("calibration", (k is not None and k >= 0.6), f"Cohen's κ = {k:.2f} (want ≥0.60)")


def check_constraint(gold_path):
    """Every gold example labeled bad must score < 0.4 (safety gate)."""
    if not os.path.exists(gold_path):
        return ("constraint", None, "no gold_set.json yet")
    import config
    if config.MOCK:
        return ("constraint", None, "MOCK mode — real judge required")
    gold = [e for e in json.load(open(gold_path)).get("examples", []) if not e["label_good"]]
    if not gold:
        return ("constraint", None, "no 'bad' examples in gold set yet")
    from contracts import Persona, GiftProposal, Reaction
    from agents import signal
    worst = 0.0
    for ex in gold:
        persona = Persona(profile=ex.get("profile", {}), hidden_truth=ex["hidden_truth"],
                          occasion=ex.get("occasion", "birthday"), budget=ex.get("budget", {"min": 0, "max": 100}))
        prop = GiftProposal(agent_id="eval", strategy_tag="eval", gifts=ex["gifts"])
        react = Reaction(proposal_id=prop.proposal_id, persona_id=persona.persona_id, verdict="return", quote="")
        worst = max(worst, signal.judge(persona, prop, react).thoughtfulness)
    return ("constraint", worst < 0.4, f"worst bad-gift score = {worst:.2f} (want <0.40)")


GATING = {"learning", "canary", "constraint"}  # these fail CI; calibration logs a trend

def main():
    store = Store(os.path.join(ROOT, "store.json"))
    gold = os.path.join(os.path.dirname(__file__), "gold_set.json")
    results = [check_calibration(gold), check_constraint(gold),
               check_learning(store), check_canary(store)]
    print("\n=== Gandalf eval report ===")
    failed = 0
    for name, ok, detail in results:
        mark = "· SKIP" if ok is None else ("✓ PASS" if ok else "✗ FAIL")
        gate = " [gates CI]" if name in GATING else ""
        print(f"  {mark}  {name}{gate}: {detail}")
        if ok is False and name in GATING:
            failed += 1
    print(f"=== {failed} gating failure(s) ===\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
