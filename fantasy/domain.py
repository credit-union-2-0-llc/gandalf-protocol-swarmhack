"""Gandalf Protocol — FANTASY domain data layer.

The fantasy domain is a demo of the Gandalf self-evolving swarm on a problem with a REAL
reward signal — unlike the gift domain, the judge here is GROUND TRUTH (actual NFL points),
not an LLM's opinion. This module turns cached real player data into decisions-with-outcomes.

A decision example, as-of week W (walk-forward — features use only weeks 1..W):
  features : recency, season-ppg, opportunity/game (targets+carries), opportunity trend
  outcome  : the player's ACTUAL rest-of-season PPG (weeks W+1..end)  ← the ground-truth judge
"""
import os
import json
from statistics import mean

HERE = os.path.dirname(__file__)
ASOF_WEEKS = [4, 5, 6, 7, 8, 9]


def _opp(row):
    return row.get("targets", 0.0) + row.get("carries", 0.0) + 0.25 * row.get("pass_att", 0.0)


def _played(pdata, lo, hi):
    return [pdata["weeks"][str(w)] for w in range(lo, hi + 1) if str(w) in pdata["weeks"]]


def _features(pdata, W):
    wk = _played(pdata, 1, W)
    if len(wk) < 2:
        return None
    pts = [r["pts"] for r in wk]
    opp = [_opp(r) for r in wk]
    return {
        "recency": mean(pts[-3:]),          # naive signal: recent fantasy points
        "season": mean(pts),                # season-to-date ppg
        "volume": mean(opp),                # opportunity per game (targets + carries)
        "opp_trend": mean(opp[-3:]) - mean(opp),  # rising/falling usage
    }


def _outcome(pdata, W, reg):
    wk = _played(pdata, W + 1, reg)
    if len(wk) < 3:
        return None
    return mean(r["pts"] for r in wk)        # GROUND TRUTH: actual rest-of-season PPG


def load_examples(year):
    """[{name, pos, features:{...}, outcome: float}] for every (player, decision-week)."""
    with open(os.path.join(HERE, "data", f"players_{year}.json")) as f:
        d = json.load(f)
    reg = d["reg_weeks"]
    out = []
    for p in d["players"].values():
        for W in ASOF_WEEKS:
            f = _features(p, W)
            o = _outcome(p, W, reg)
            if f is not None and o is not None:
                out.append({"name": p["name"], "pos": p["pos"], "features": f, "outcome": o})
    return out
