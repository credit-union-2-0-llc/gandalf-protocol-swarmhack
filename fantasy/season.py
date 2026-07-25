"""Gandalf Protocol — FANTASY season simulator (the self-improvement demo).

The story, honestly:
  1. A rookie GM drafts by "who's hot" — total points over the first 3 weeks.
  2. The swarm reviews the outcomes and distills a PLAYBOOK: recent points are noise (TD-driven);
     opportunity (volume) and full-season production are what's real. A few signal weights.
  3. A veteran GM re-drafts the SAME player pool with the playbook.
  4. Same season, same field, same schedule — the veteran makes the playoffs, the rookie misses.

Three anti-memorization guarantees, by construction:
  • Walk-forward: the draft reads only weeks 1-3; the season it's graded on is weeks 4..reg. The
    graded weeks are never visible at decision time — no lookahead.
  • Leave-one-season-out: the playbook is fit on the OTHER seasons; it never sees the season it's
    tested on. It is ~4 numbers — it cannot store a player's weekly box score.
  • Generalization: the same playbook lifts EVERY season, not just one. Memory wouldn't transfer.
"""
import os
import json
from statistics import mean
import numpy as np

HERE = os.path.dirname(__file__)
YEARS = [2022, 2023, 2024, 2025]
DRAFT_UPTO = 3               # roster is drafted from weeks 1..3 form (early-season redraft)
FIRST_WEEK = 4               # the season we grade: weeks 4..reg_weeks (strictly after the draft)
N_TEAMS = 12                 # a full league: 6 rookie GMs vs 6 veteran GMs, alternating draft slots
PLAYOFF_SPOTS = 6
GAMES = 14                   # standard fantasy regular season, for projecting all-play win% → record
# Full PPR starting lineup: 1 QB / 2 RB / 2 WR / 1 TE / 1 FLEX / 1 K / 1 D/ST = 9 starters.
SLOTS = [("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1), ("K", 1), ("D/ST", 1)]
FLEX_ELIG = ("RB", "WR", "TE")
ROSTER_CAPS = {"QB": 2, "RB": 5, "WR": 5, "TE": 2, "K": 1, "D/ST": 1}   # 16-man rosters
SKILL = ("QB", "RB", "WR", "TE")   # the influence share is learned on skill players only
SCORING = "PPR (1 pt / reception)"

SIGNALS = ["recency", "season", "volume", "opp_trend"]
NAIVE = "recency"
# plain-English lesson each signal encodes (shown in the "what they learned" panel)
LESSON = {
    "season":    "Trust full-season production over a hot three weeks.",
    "volume":    "Chase opportunity — targets + carries — not last week's points.",
    "recency":   "Recent form still matters, but weight it, don't worship it.",
    "opp_trend": "Rising usage is a breakout signal before the box score catches up.",
}


def _opp(row):
    return row.get("targets", 0.0) + row.get("carries", 0.0) + 0.25 * row.get("pass_att", 0.0)


def _load(year):
    with open(os.path.join(HERE, "data", f"players_{year}.json")) as f:
        return json.load(f)


def _weeks_upto(p, hi):
    return [p["weeks"][str(w)] for w in range(1, hi + 1) if str(w) in p["weeks"]]


def _features_asof(p, W):
    """Features from weeks 1..W-1 ONLY (strictly the past). None if too little history."""
    wk = _weeks_upto(p, W - 1)
    if len(wk) < 2:
        return None
    pts = [r["pts"] for r in wk]
    opp = [_opp(r) for r in wk]
    return {"recency": mean(pts[-3:]), "season": mean(pts),
            "volume": mean(opp), "opp_trend": mean(opp[-3:]) - mean(opp)}


def _actual(p, W):
    r = p["weeks"].get(str(W))
    return None if r is None else r["pts"]


def _draft_pts(p):
    return sum(r["pts"] for r in _weeks_upto(p, DRAFT_UPTO)[:DRAFT_UPTO])


def _rest_ppg(p, reg):
    wk = [p["weeks"][str(w)] for w in range(FIRST_WEEK, reg + 1) if str(w) in p["weeks"]]
    return mean(r["pts"] for r in wk) if wk else 0.0


# ─────────────────────────── the learned playbook ───────────────────────────
def fit_playbook(train_years):
    """OLS over the signals → predict a player's weekly points, from walk-forward (player, week)
    decisions in `train_years`. Returns weights + standardization + human-readable std weights."""
    rows = []
    for yr in train_years:
        d = _load(yr)
        for p in d["players"].values():
            if p["pos"] not in SKILL:          # K / D/ST have no opportunity signal — keep the lessons clean
                continue
            for W in range(FIRST_WEEK, d["reg_weeks"] + 1):
                f = _features_asof(p, W)
                a = _actual(p, W)
                if f is not None and a is not None:
                    rows.append((f, a))
    X = np.array([[r[0][s] for s in SIGNALS] for r in rows], float)
    y = np.array([r[1] for r in rows], float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    A = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"w": w, "mu": mu, "sd": sd, "n": len(rows),
            "std_weights": {s: float(w[1 + i]) for i, s in enumerate(SIGNALS)}}


def _value(pb, f):
    x = np.array([f[s] for s in SIGNALS], float)
    z = np.hstack([[1.0], (x - pb["mu"]) / pb["sd"]])
    return float(z @ pb["w"])


def _shares_from(rows):
    """Standardized |weight| share per signal from (features, outcome) rows — 'how much of the
    playbook each agent earns'. Shares sum to 1."""
    X = np.array([[r[0][s] for s in SIGNALS] for r in rows], float)
    y = np.array([r[1] for r in rows], float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    A = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    aw = {s: abs(w[1 + i]) for i, s in enumerate(SIGNALS)}
    tot = sum(aw.values()) or 1.0
    return {s: aw[s] / tot for s in SIGNALS}


def weight_trajectory(year, min_rows=800):
    """Self-evolution, made visible: walk forward through `year` and, after each week's real
    results land, RE-FIT the playbook on everything seen so far (this season). Returns the share
    each agent earns as the evidence accumulates — the weighting is not fixed, the swarm keeps
    re-deriving it. Walk-forward: 'through week N' uses only games in weeks ≤ N."""
    d = _load(year)
    acc, out = [], []
    for w in range(FIRST_WEEK, d["reg_weeks"] + 1):
        for p in d["players"].values():
            if p["pos"] not in SKILL:
                continue
            f = _features_asof(p, w)
            a = _actual(p, w)
            if f is not None and a is not None:
                acc.append((f, a))
        if len(acc) >= min_rows:
            out.append({"through_week": w, "n": len(acc), "shares": _shares_from(acc)})
    return out


# ─────────────────────────── draft, lineup, league ───────────────────────────
def _draftable(d):
    return [p for p in d["players"].values()
            if p["pos"] in ROSTER_CAPS and _weeks_upto(p, DRAFT_UPTO)
            and _features_asof(p, FIRST_WEEK) is not None]


def draft_league(d, pb):
    """Snake draft 12 teams from ONE pool. Even slots = rookie GMs (rank by weeks-1-3 points),
    odd slots = veteran GMs (rank by the learned playbook value as-of week 4). Fair head-to-head:
    same pool, alternating slots, identical roster limits."""
    pool = _draftable(d)
    rookie_key = _draft_pts
    # veteran uses the learned influence for SKILL players; K/D-ST are a wash — ranked by per-game
    # early points on the SAME per-week scale as _value, so they don't distort skill drafting.
    veteran_key = lambda p: (_value(pb, _features_asof(p, FIRST_WEEK))
                             if p["pos"] in SKILL else _draft_pts(p) / DRAFT_UPTO)
    kinds = ["rookie" if i % 2 == 0 else "veteran" for i in range(N_TEAMS)]
    rosters = [[] for _ in range(N_TEAMS)]
    caps = [dict(ROSTER_CAPS) for _ in range(N_TEAMS)]
    size = sum(ROSTER_CAPS.values())
    taken = set()
    for pk in range(size * N_TEAMS):
        rnd, slot = divmod(pk, N_TEAMS)
        ti = slot if rnd % 2 == 0 else N_TEAMS - 1 - slot
        if len(rosters[ti]) >= size:
            continue
        key = veteran_key if kinds[ti] == "veteran" else rookie_key
        avail = [p for p in pool if id(p) not in taken and caps[ti][p["pos"]] > 0]
        if not avail:
            continue
        p = max(avail, key=key)
        rosters[ti].append(p); caps[ti][p["pos"]] -= 1; taken.add(id(p))
    return kinds, rosters


def _set_lineup(roster, W, ranker):
    cand = [(ranker(f), p) for p in roster
            for f in [_features_asof(p, W)] if f is not None and _actual(p, W) is not None]
    started, used = [], set()
    for pos, n in SLOTS:
        for _, p in sorted([c for c in cand if c[1]["pos"] == pos and id(c[1]) not in used],
                           key=lambda t: -t[0])[:n]:
            started.append(p); used.add(id(p))
    flex = sorted([c for c in cand if c[1]["pos"] in FLEX_ELIG and id(c[1]) not in used],
                  key=lambda t: -t[0])
    if flex:
        started.append(flex[0][1])
    return started


def _score(started, W):
    return sum(_actual(p, W) for p in started)


def run_league(d, pb):
    """Draft, then play weeks 4..reg with a NEUTRAL start/sit (season-to-date avg) for everyone,
    so the standings reflect DRAFT quality, not start/sit tricks. Returns everything the demo needs."""
    reg = d["reg_weeks"]
    weeks = list(range(FIRST_WEEK, reg + 1))
    kinds, rosters = draft_league(d, pb)
    neutral = lambda f: f["season"]
    # weekly team scores
    wk_scores = [[_score(_set_lineup(r, W, neutral), W) for r in rosters] for W in weeks]
    # all-play record: fraction of all other teams you outscore each week (removes schedule luck)
    n = N_TEAMS
    aw = [0.0] * n
    for row in wk_scores:
        for i in range(n):
            aw[i] += sum(1 for j in range(n) if j != i and row[i] > row[j])
    per_game = len(weeks) * (n - 1)
    winpct = [w / per_game for w in aw]
    totals = [sum(wk_scores[t][i] for t in range(len(weeks))) for i in range(n)]
    order = sorted(range(n), key=lambda i: -winpct[i])
    place = {ti: r + 1 for r, ti in enumerate(order)}
    return {"year": d["year"], "reg": reg, "weeks": weeks, "kinds": kinds, "rosters": rosters,
            "wk_scores": wk_scores, "winpct": winpct, "totals": totals, "place": place}


def side_summary(L, kind):
    idx = [i for i in range(N_TEAMS) if L["kinds"][i] == kind]
    wp = mean(L["winpct"][i] for i in idx)
    nwk = len(L["weeks"])
    wins = round(wp * GAMES)
    return {"kind": kind, "win_pct": wp, "record": f"{wins}-{GAMES - wins}",
            "avg_place": mean(L["place"][i] for i in idx),
            "ppw": mean(L["totals"][i] for i in idx) / nwk,
            "playoff_teams": sum(1 for i in idx if L["place"][i] <= PLAYOFF_SPOTS),
            "made_playoffs": mean(L["place"][i] for i in idx) <= PLAYOFF_SPOTS}


def draft_contrast(d, pb, top=5, starter_rank=60, gap=10):
    """The tangible payoff, from the players the two strategies most DISAGREE on. Each player has a
    rookie rank (by early points) and a veteran rank (by playbook value). Judged by ACTUAL
    rest-of-season PPG — reality settles the argument.
      • rookie_reaches: rookie ranks them a startable pick, veteran fades them (gap+) — and they FADED.
      • veteran_values: veteran ranks them a startable pick, rookie passes (gap+) — and they PRODUCED."""
    reg = d["reg_weeks"]
    pool = _draftable(d)
    rr = {id(p): i + 1 for i, p in enumerate(sorted(pool, key=lambda p: -_draft_pts(p)))}
    vr = {id(p): i + 1 for i, p in enumerate(
        sorted(pool, key=lambda p: -_value(pb, _features_asof(p, FIRST_WEEK))))}
    def fmt(p):
        early = _draft_pts(p) / max(1, len(_weeks_upto(p, DRAFT_UPTO)[:DRAFT_UPTO]))
        return {"name": p["name"], "pos": p["pos"], "early_ppg": round(early, 1),
                "rest_ppg": round(_rest_ppg(p, reg), 1),
                "rookie_rank": rr[id(p)], "veteran_rank": vr[id(p)]}
    reaches = [fmt(p) for p in pool if rr[id(p)] <= starter_rank
               and vr[id(p)] - rr[id(p)] >= gap and _rest_ppg(p, reg) < _draft_pts(p) / 3]
    values = [fmt(p) for p in pool if vr[id(p)] <= starter_rank
              and rr[id(p)] - vr[id(p)] >= gap and _rest_ppg(p, reg) > _draft_pts(p) / 3]
    reaches.sort(key=lambda x: x["rest_ppg"])          # worst faders first
    values.sort(key=lambda x: -x["rest_ppg"])          # best producers first
    return {"rookie_reaches": reaches[:top], "veteran_values": values[:top]}


if __name__ == "__main__":
    print("Leave-one-season-out — playbook fit WITHOUT the tested season.\n")
    for yr in YEARS:
        pb = fit_playbook([y for y in YEARS if y != yr])
        L = run_league(_load(yr), pb)
        r, v = side_summary(L, "rookie"), side_summary(L, "veteran")
        print(f"{yr}: rookie {r['ppw']:5.1f} ppw {r['record']} (place {r['avg_place']:.1f}) "
              f"→ veteran {v['ppw']:5.1f} ppw {v['record']} (place {v['avg_place']:.1f}) "
              f"| veterans in playoffs: {v['playoff_teams']}/6")
