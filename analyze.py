#!/usr/bin/env python3
"""Gandalf Protocol — results verdict (standalone, stdlib-only, read-only).

Turns store.json into the judge-facing answer so we don't eyeball a PNG:
  - Does the swarm curve CLIMB, and climb FASTER than solo?
  - Does the held-out VALIDATION line rise with training (generalization, not memorization)?
  - Is training climbing while validation stays flat? (reward-hacking canary — we say so.)

Usage: python3 analyze.py [store.json]
Independent of the run modules on purpose — works even mid-refactor.
"""
import json, sys
from collections import defaultdict


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _by_round(eps):
    """{round_index: [thoughtfulness,...]} -> sorted [(r, mean, n)]."""
    d = defaultdict(list)
    for e in eps:
        r = e.get("round_index", -1)
        if r is not None and r >= 0:
            d[r].append(e["score"]["thoughtfulness"])
    return [(r, _mean(v), len(v)) for r, v in sorted(d.items())]


def _slope(series):
    """Least-squares regression slope over (round, mean). Robust to a single noisy round —
    unlike endpoint (last-first), which one bad final round can flip. >0 = climbing."""
    n = len(series)
    if n < 2:
        return 0.0
    xs = [r for r, _, _ in series]
    ys = [m for _, m, _ in series]
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def _avg_gap(swarm, solo):
    """Mean (swarm - solo) over rounds present in BOTH — the robust 'swarm is better' metric,
    independent of round-to-round climb noise."""
    sm = {r: m for r, m, _ in swarm}
    so = {r: m for r, m, _ in solo}
    common = sorted(set(sm) & set(so))
    if not common:
        return None
    return sum(sm[r] - so[r] for r in common) / len(common)


def main(path="store.json"):
    eps = json.load(open(path)).get("episodes", [])
    if not eps:
        print("no episodes in", path); return

    train = [e for e in eps if not e.get("is_validation")]
    solo = _by_round([e for e in train if e.get("condition") == "solo"])
    swarm = _by_round([e for e in train if e.get("condition") == "swarm"])
    val = _by_round([e for e in eps if e.get("is_validation")])

    print(f"=== Gandalf results — {path} ===")
    print(f"episodes: {len(eps)}  (train={len(train)}, validation={len(eps)-len(train)})\n")

    def show(name, s):
        if not s:
            print(f"{name:10} (no rounds)"); return
        curve = "  ".join(f"r{r}:{m:.3f}" for r, m, _ in s)
        print(f"{name:10} {curve}   slope={_slope(s):+.4f}")
    show("solo", solo)
    show("swarm", swarm)
    show("validation", val)
    print()

    # ── verdicts ──
    ok = True
    if swarm:
        climbs = _slope(swarm) > 0.002
        print(f"[{'PASS' if climbs else 'FAIL'}] swarm curve climbs (slope {_slope(swarm):+.4f})")
        ok &= climbs
    if swarm and solo:
        gap = _avg_gap(swarm, solo)
        above = gap is not None and gap > 0.02
        print(f"[{'PASS' if above else 'FAIL'}] swarm scores ABOVE solo on average "
              f"(mean per-round gap {gap:+.3f}); "
              f"slopes: swarm {_slope(swarm):+.4f} vs solo {_slope(solo):+.4f}")
        ok &= above
    if val:
        vrise = _slope(val) > 0.002
        print(f"[{'PASS' if vrise else 'FAIL'}] validation line rises (slope {_slope(val):+.4f}) "
              f"— generalizes, not memorizes")
        # reward-hacking canary
        if swarm and _slope(swarm) > 0.01 and _slope(val) <= 0.0:
            print("[WARN] training climbs while validation is FLAT/FALLING — possible "
                  "reward-hacking; the swarm may be gaming the simulator. Say so honestly.")
        ok &= vrise

    print("\nVERDICT:", "STORY HOLDS — ship it." if ok else
          "NOT YET — curve/gap/validation weak; needs more rounds or sharper judge/distill.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "store.json")
