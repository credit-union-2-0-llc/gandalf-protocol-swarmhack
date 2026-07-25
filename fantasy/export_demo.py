"""Run the fantasy self-improvement simulation and export everything the UI renders into
fantasy/demo_data.json. Every number here comes straight from fantasy/season.py — the UI only
draws them. Nothing is hand-typed or tuned.

The arc: rookie GM drafts on hot hands → swarm distills a playbook → veteran GM re-drafts the
same pool → makes the playoffs the rookie missed. Proven not-memorized by leave-one-season-out.
"""
import os
import sys
import json
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fantasy.season as S     # noqa: E402

FOCUS = 2025


def _side(L, kind):
    return S.side_summary(L, kind)


def _standings(L):
    rows = []
    for i in range(S.N_TEAMS):
        wp = L["winpct"][i]
        wins = round(wp * S.N_TEAMS)      # a rough per-team W-L within the 12-team, ~11-week slate
        rows.append({"place": L["place"][i], "kind": L["kinds"][i],
                     "win_pct": round(wp, 3), "ppw": round(L["totals"][i] / len(L["weeks"]), 1)})
    rows.sort(key=lambda r: r["place"])
    return rows


def _lessons(pb):
    tot = sum(abs(v) for v in pb["std_weights"].values()) or 1.0
    out = [{"signal": s, "weight": round(w, 2), "share": round(abs(w) / tot, 3),
            "rule": S.LESSON[s], "is_naive": s == S.NAIVE}
           for s, w in pb["std_weights"].items()]
    out.sort(key=lambda d: -abs(d["weight"]))
    return out


def main():
    # ── headline + weekly series + standings for the focus season (leave-one-out) ──
    pb_focus = S.fit_playbook([y for y in S.YEARS if y != FOCUS])
    L = S.run_league(S._load(FOCUS), pb_focus)
    rook, vet = _side(L, "rookie"), _side(L, "veteran")

    # per-week average points for each side (the two diverging lines)
    series = []
    for wi, W in enumerate(L["weeks"]):
        r = mean(L["wk_scores"][wi][i] for i in range(S.N_TEAMS) if L["kinds"][i] == "rookie")
        v = mean(L["wk_scores"][wi][i] for i in range(S.N_TEAMS) if L["kinds"][i] == "veteran")
        series.append({"week": W, "rookie": round(r, 1), "veteran": round(v, 1)})

    contrast = S.draft_contrast(S._load(FOCUS), pb_focus)

    # self-evolution: how each agent's share of the playbook shifts week to week as the swarm
    # re-fits on each week's real results (walk-forward within the focus season)
    evolution = [{"week": t["through_week"],
                  "shares": {k: round(v, 3) for k, v in t["shares"].items()}}
                 for t in S.weight_trajectory(FOCUS)]

    # ── generalization proof: every leave-one-out season ──
    generalization = []
    for yr in S.YEARS:
        pb = S.fit_playbook([y for y in S.YEARS if y != yr])
        Ly = S.run_league(S._load(yr), pb)
        r, v = _side(Ly, "rookie"), _side(Ly, "veteran")
        generalization.append({
            "year": yr, "is_focus": yr == FOCUS,
            "rookie_ppw": round(r["ppw"], 1), "veteran_ppw": round(v["ppw"], 1),
            "lift_ppw": round(v["ppw"] - r["ppw"], 1),
            "rookie_record": r["record"], "veteran_record": v["record"],
            "veteran_playoffs": v["playoff_teams"], "playoff_spots": S.PLAYOFF_SPOTS,
        })

    # ── the arguing agents, and how much of the playbook each earned ──
    lessons = _lessons(pb_focus)
    share = {d["signal"]: d["share"] for d in lessons}
    claims = {"Recency": "Start whoever's hot — trust the last three weeks.",
              "Season": "Trust the full-season body of work, not a hot streak.",
              "Volume": "Opportunity is real — chase targets and carries.",
              "Trend": "Rising usage is a breakout before the box score shows it."}
    sig_of = {"Recency": "recency", "Season": "season", "Volume": "volume", "Trend": "opp_trend"}
    agents = [{"name": a, "claim": claims[a], "signal": sig_of[a],
               "share": share[sig_of[a]], "is_naive": sig_of[a] == S.NAIVE}
              for a in ("Recency", "Season", "Volume", "Trend")]

    data = {
        "meta": {"focus_season": FOCUS, "train": "leave-one-season-out",
                 "n_teams": S.N_TEAMS, "playoff_spots": S.PLAYOFF_SPOTS, "games": S.GAMES,
                 "train_decisions": pb_focus["n"], "scoring": S.SCORING,
                 "roster": "16-man roster; start 1 QB / 2 RB / 2 WR / 1 TE / 1 FLEX / 1 K / 1 D/ST",
                 "draft_note": "drafted on weeks 1-3 form; graded weeks 4-14 (never seen at draft)"},
        "headline": {
            "rookie": {"record": rook["record"], "ppw": round(rook["ppw"], 1),
                       "avg_place": round(rook["avg_place"], 1), "made_playoffs": rook["made_playoffs"]},
            "veteran": {"record": vet["record"], "ppw": round(vet["ppw"], 1),
                        "avg_place": round(vet["avg_place"], 1), "made_playoffs": vet["made_playoffs"]},
            "lift_ppw": round(vet["ppw"] - rook["ppw"], 1),
        },
        "playbook": {"naive": S.NAIVE, "lessons": lessons},
        "evolution": evolution,
        "series": series,
        "standings": _standings(L),
        "generalization": generalization,
        "contrast": contrast,
        "agents": agents,
    }

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"demo data → {out}")
    print(f"  {FOCUS}: rookie {rook['record']} ({rook['ppw']:.1f} ppw) → "
          f"veteran {vet['record']} ({vet['ppw']:.1f} ppw)  +{data['headline']['lift_ppw']}/wk")
    print(f"  playoffs: rookie {'IN' if rook['made_playoffs'] else 'OUT'} / "
          f"veteran {'IN' if vet['made_playoffs'] else 'OUT'}")
    print("  generalization (lift/wk):", " ".join(f"{g['year']}:{g['lift_ppw']:+.1f}" for g in generalization))


if __name__ == "__main__":
    main()
