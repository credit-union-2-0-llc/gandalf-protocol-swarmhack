"""Gandalf Protocol — structural guard for the judge gold set [#13]

Run from repo root: `python test_gold_set.py`

Validates the SHAPE + balance of evals/gold_set.json on every CI run (no API key,
zero cost). The real judge-calibration gate (Cohen's kappa >= 0.60, bad examples
< 0.40) runs separately in run_evals.py when the gateway key is present — see
ci.yml. This just keeps the gold set well-formed and meaningfully balanced so a
malformed edit can't silently defang calibration.
"""
import json, sys

REQUIRED = {"label_good", "profile", "hidden_truth", "occasion", "budget", "gifts", "verdict"}
GIFT_KEYS = {"name", "category", "price", "reasoning"}
FAILURES = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


gold = json.load(open("evals/gold_set.json"))
ex = gold.get("examples", [])

check(len(ex) >= 10, f">= 10 examples (got {len(ex)})")
good = [e for e in ex if e.get("label_good") is True]
bad = [e for e in ex if e.get("label_good") is False]
check(len(good) >= 4, f">= 4 good examples (got {len(good)})")
check(len(bad) >= 4, f">= 4 bad examples (got {len(bad)})")
check(abs(len(good) - len(bad)) <= 2, f"good/bad roughly balanced ({len(good)}/{len(bad)})")

for i, e in enumerate(ex):
    miss = REQUIRED - set(e)
    check(not miss, f"example[{i}] has all required keys" + (f" (missing {miss})" if miss else ""))
    check(isinstance(e.get("label_good"), bool), f"example[{i}].label_good is a bool")
    gifts = e.get("gifts", [])
    check(bool(gifts) and all(GIFT_KEYS <= set(g) for g in gifts),
          f"example[{i}].gifts each have {sorted(GIFT_KEYS)}")
    b = e.get("budget", {})
    check(isinstance(b.get("min"), (int, float)) and isinstance(b.get("max"), (int, float))
          and b["max"] >= b["min"], f"example[{i}].budget has sane min<=max")

# Every bad example needs a hidden_truth signal that makes it *provably* bad
# (would_return / already_has) OR a constraint — otherwise "bad" is unlabelable
# and calibration can't learn from it.
for i, e in enumerate(ex):
    if e.get("label_good") is False:
        ht = e.get("hidden_truth", {})
        prof = e.get("profile", {})
        has_signal = bool(ht.get("would_return") or ht.get("already_has") or prof.get("constraints")
                          or any(g.get("price", 0) > e.get("budget", {}).get("max", 1e9) for g in e.get("gifts", [])))
        check(has_signal, f"bad example[{i}] carries a concrete bad-signal (return/owns/constraint/over-budget)")

if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S) in gold_set.json:")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print(f"\nGold set OK — {len(ex)} examples ({len(good)} good / {len(bad)} bad).")
