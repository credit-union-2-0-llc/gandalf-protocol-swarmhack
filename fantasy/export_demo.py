"""Run the fantasy swarm and export everything the UI needs into fantasy/demo_data.json:
  - climb        : the learning curve (round → top-10 real PPG + MAE) + solo baseline
  - agents       : each agent's own top picks (they argue / disagree)
  - draft        : the team the self-evolved swarm rosters vs the naive agent's team, with the
                   players' ACTUAL rest-of-season points (ground truth) — the tangible payoff
All numbers come straight from the real run; the UI only renders them.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fantasy.domain import load_examples                              # noqa: E402
from fantasy.swarm import FantasySwarm, decision_quality, mae, AGENTS, TOPK  # noqa: E402

TEAM_N = 12


def _dedup(pool):
    """One representative example per player (a draftable pool, each player once)."""
    seen, out = set(), []
    for e in pool:
        if e["name"] not in seen:
            seen.add(e["name"]); out.append(e)
    return out


def _team(swarm, pool, n=TEAM_N):
    order = np.argsort(-swarm.score(pool))
    picks = [pool[i] for i in order[:n]]
    return [{"name": p["name"], "pos": p["pos"], "actual": round(p["outcome"], 1)} for p in picks]


def main():
    train = load_examples(2022) + load_examples(2023) + load_examples(2024)
    held = load_examples(2025)
    swarm = FantasySwarm(learn=True); solo = FantasySwarm(learn=False)
    swarm.bootstrap(train); solo.bootstrap(train)

    climb = [{"label": "naive\n(Recency)", "ppg": round(decision_quality(swarm, held), 1),
              "mae": round(mae(swarm, held), 2), "adopted": list(swarm.adopted)}]
    while True:
        added = swarm.distill(train)
        if not added:
            break
        climb.append({"label": f"+{added}", "ppg": round(decision_quality(swarm, held), 1),
                      "mae": round(mae(swarm, held), 2), "adopted": list(swarm.adopted)})

    pool = _dedup(held)
    # agents argue: each single-agent's own top team
    agents = {}
    for a in AGENTS:
        s = FantasySwarm(learn=False); s.adopted = [a]; s.bootstrap(train)
        agents[a] = {"team": _team(s, pool, 5),
                     "ppg": round(decision_quality(s, held), 1)}

    data = {
        "meta": {"train_decisions": len(train), "held_decisions": len(held),
                 "seasons_train": "2022-2024", "season_held": 2025, "roster_size": TEAM_N},
        "climb": climb,
        "solo_ppg": round(decision_quality(solo, held), 1),
        "solo_mae": round(mae(solo, held), 2),
        "swarm_ppg": climb[-1]["ppg"], "swarm_mae": climb[-1]["mae"],
        "agents": agents,
        "swarm_team": _team(swarm, pool),
        "naive_team": _team(solo, pool),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"demo data → {out}")
    print(f"  climb: {' → '.join(str(c['ppg']) for c in climb)} PPG | "
          f"swarm {data['swarm_ppg']} vs naive {data['solo_ppg']}")
    print(f"  swarm team total actual PPG: {sum(p['actual'] for p in data['swarm_team']):.0f}  "
          f"vs naive team: {sum(p['actual'] for p in data['naive_team']):.0f}")


if __name__ == "__main__":
    main()
