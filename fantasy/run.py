"""Gandalf Protocol — FANTASY demo: the self-evolving swarm learns to draft a winning team.

The Gandalf thesis on a domain where the judge is GROUND TRUTH (real NFL points):
  • A ROOKIE GM (drafts "who's hot", never learns) finishes ~6-8 and misses the playoffs.
  • The swarm reviews its own mistakes and distills a PLAYBOOK; a VETERAN GM re-drafts the same
    pool with it and finishes ~8-6 — clinching the playoffs the rookie missed.

Proven NOT memorized: leave-one-season-out (playbook never sees the season it's graded on),
walk-forward (drafts read weeks 1-3, graded weeks 4-14), and it lifts EVERY season.

Run:  python fantasy/run.py   (or:  python run.py --domain fantasy)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fantasy.season as S                        # noqa: E402
from fantasy.export_demo import main as export    # noqa: E402


def main():
    print("Gandalf · Fantasy — self-improvement demo (leave-one-season-out, real NFL points)\n")
    print(f"  {'season':>6}  {'rookie (one agent)':<26}{'veteran (learned playbook)':<28}{'playoff seeds':>14}")
    print("  " + "-" * 76)
    for yr in S.YEARS:
        pb = S.fit_playbook([y for y in S.YEARS if y != yr])
        L = S.run_league(S._load(yr), pb)
        r, v = S.side_summary(L, "rookie"), S.side_summary(L, "veteran")
        star = " <=" if yr == 2025 else ""
        rk = "{}  {:.1f} ppw".format(r["record"], r["ppw"])
        vt = "{}  {:.1f} ppw  (+{:.1f})".format(v["record"], v["ppw"], v["ppw"] - r["ppw"])
        seeds = "{}/{}".format(v["playoff_teams"], S.PLAYOFF_SPOTS)
        print("  {:>6}  {:<26}{:<28}{:>14}{}".format(yr, rk, vt, seeds, star))
    print("  " + "-" * 76)

    pb = S.fit_playbook([y for y in S.YEARS if y != 2025])
    print("\n  The playbook the swarm distilled (share of how it values a pick):")
    tot = sum(abs(w) for w in pb["std_weights"].values()) or 1.0
    for s, w in sorted(pb["std_weights"].items(), key=lambda t: -abs(t[1])):
        naive = "  (one agent alone — demoted)" if s == S.NAIVE else ""
        print(f"    {s:<10} {abs(w)/tot:>4.0%}   {S.LESSON[s]}{naive}")

    print(f"\n  Trained on {pb['n']:,} walk-forward (player, week) decisions. Regenerating demo_data.json…\n")
    export()


if __name__ == "__main__":
    main()
