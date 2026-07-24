"""Gandalf Protocol — FANTASY demo: the self-evolving swarm learns against real outcomes.

The Gandalf thesis on a domain where the judge is GROUND TRUTH (real NFL points):
  • A SOLO agent (just "who scored recently", no learning) stays FLAT.
  • The SWARM of arguing agents self-evolves — each round it adopts the agent whose signal most
    improves its fit to REAL outcomes — and its decisions CLIMB.

Headline metric (intuitive + honest): the ACTUAL rest-of-season points scored by the top-10
players the swarm would roster, measured on a HELD-OUT season (2025) it never trained on. If the
swarm weren't really learning, this line would be flat — nothing here is tuned to force a climb.

Run:  python fantasy/run.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fantasy.domain import load_examples                          # noqa: E402
from fantasy.swarm import FantasySwarm, decision_quality, mae, AGENTS, TOPK  # noqa: E402


def main():
    train = load_examples(2022) + load_examples(2023) + load_examples(2024)  # 3 seasons, full pool
    held = load_examples(2025)                          # held-out judge (never trained on)

    swarm = FantasySwarm(learn=True)
    solo = FantasySwarm(learn=False)                    # naive: recency only, never evolves
    swarm.bootstrap(train)
    solo.bootstrap(train)

    solo_q, solo_mae = decision_quality(solo, held), mae(solo, held)
    print(f"Fantasy self-evolving swarm — {len(train)} training decisions, {len(held)} held-out "
          f"(2025). Judge = real rest-of-season points.\n")
    print(f"  {'round':>5}  {'adopted agents':<34}{'top-10 real PPG':>16}{'MAE':>8}")
    print("  " + "-" * 66)

    curve = [decision_quality(swarm, held)]
    maes = [mae(swarm, held)]
    print(f"  {0:>5}  {'Recency (naive start)':<34}{curve[0]:>16.1f}{maes[0]:>8.2f}")
    r = 0
    while True:
        added = swarm.distill(train)                    # self-evolve on real outcomes
        if not added:
            break
        r += 1
        curve.append(decision_quality(swarm, held))
        maes.append(mae(swarm, held))
        print(f"  {r:>5}  {'+' + added + '  (' + ', '.join(swarm.adopted) + ')':<34}"
              f"{curve[-1]:>16.1f}{maes[-1]:>8.2f}")

    print("  " + "-" * 66)
    print(f"\n  SOLO (naive recency, no learning): {solo_q:.1f} PPG  (MAE {solo_mae:.2f}) — flat")
    print(f"  SWARM (self-evolved): {curve[0]:.1f} → {curve[-1]:.1f} PPG  (+{curve[-1]-curve[0]:.1f}), "
          f"MAE {maes[0]:.2f} → {maes[-1]:.2f}")
    print(f"  Each top-10 starter scores +{curve[-1]-solo_q:.1f} more real points/wk than the naive "
          f"agent — learned against ground truth, out-of-sample.")

    # the arguing: show the single agents disagree AND each is worse than the learned swarm
    print("\n  The agents argue (each's own top-10, real PPG) — none matches the swarm:")
    for a in AGENTS:
        s = FantasySwarm(learn=False); s.adopted = [a]; s.bootstrap(train)
        print(f"    {a:<9} {decision_quality(s, held):>5.1f}")

    _chart(curve, solo_q, swarm.adopted)


def _chart(curve, solo_q, adopted):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n  (chart skipped: {e})"); return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xs = list(range(len(curve)))
    ax.plot(xs, curve, "-o", color="#2a7d5f", lw=2.6, ms=7, label="Gandalf swarm (self-evolving)")
    ax.axhline(solo_q, color="#b0453c", ls="--", lw=2, label="solo agent (naive, no learning)")
    labels = ["naive\n(Recency)"] + [f"+{a}" for a in adopted[1:]]
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("distillation round — signals the swarm validated against real outcomes")
    ax.set_ylabel("top-10 starters' ACTUAL points/wk (held-out 2025)")
    ax.set_title("Gandalf on fantasy: the swarm learns from real outcomes, the solo agent can't")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    for x, y in zip(xs, curve):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=9,
                    ha="center", color="#2a7d5f")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart_fantasy_learning.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\n  Chart → {out}")


if __name__ == "__main__":
    main()
