"""Gandalf Protocol — Local JSON Storage [OWNER: Kirk]

Dead-simple: one JSON file, episodes accumulate. Do NOT build a database.

Later: swap method bodies to call ops-platform microservice, keeping signatures
identical so nothing else changes.
"""
import json, os
from contracts import to_json


class Store:
    def __init__(self, path="store.json"):
        self.path = path
        self.episodes: list[dict] = []
        self.load()   # [Rig fix P0-2] reload prior episodes so `python dashboard.py`
                      # (or any fresh process) can replay from store.json.

    def load(self):
        """Pull episodes off disk if the file exists (cold start => empty).
        Mirrors store_azure.Store.load() so both backends behave identically."""
        if not os.path.exists(self.path):
            self.episodes = []
            return
        with open(self.path) as f:
            self.episodes = json.load(f).get("episodes", [])

    def save(self, episode):
        self.episodes.append(to_json(episode))
        self._flush()

    def _flush(self):
        with open(self.path, "w") as f:
            json.dump({"episodes": self.episodes}, f, indent=2)

    # ── read helpers used by the loop, coach, and dashboard ──
    def training_episodes(self, condition=None):
        eps = [e for e in self.episodes if not e["is_validation"]]
        if condition:
            eps = [e for e in eps if e.get("condition") == condition]
        return eps

    def validation_episodes(self):
        return [e for e in self.episodes if e["is_validation"]]

    def score_history_summary(self):
        """Compact summary the coach reads to find weak spots."""
        by_occasion = {}
        for e in self.training_episodes():
            occ = e.get("occasion", "unknown")
            by_occasion.setdefault(occ, []).append(e["score"]["thoughtfulness"])
        return {k: round(sum(v)/len(v), 3) for k, v in by_occasion.items()}

    def top_episodes(self, n=5, condition=None):
        eps = self.training_episodes(condition)
        return sorted(eps, key=lambda e: e["score"]["thoughtfulness"], reverse=True)[:n]

    def bottom_episodes(self, n=5, condition=None):
        """[Round-2 Phase 3] The mirror of top_episodes: the LOWEST-thoughtfulness training
        episodes, so the ACE Reflector can extract 'avoid' lessons from failures (GEPA-style).
        Additive — nothing existing calls this; only the ACE_CURATION distill path does."""
        eps = self.training_episodes(condition)
        return sorted(eps, key=lambda e: e["score"]["thoughtfulness"])[:n]

    def thoughtfulness_series(self, condition=None, validation=False, window=10):
        src = self.validation_episodes() if validation else self.training_episodes(condition)
        vals = [e["score"]["thoughtfulness"] for e in src]
        # rolling average for a readable curve
        out = []
        for i in range(len(vals)):
            lo = max(0, i - window + 1)
            out.append(sum(vals[lo:i+1]) / (i - lo + 1))
        return out

    # ── per-round aggregation [Rig fix P0-1 / P0-3] ──
    # The ablation and validation charts must share a ROUND x-axis, not raw episode
    # index — otherwise solo (few episodes) and swarm (4x episodes) span different
    # widths and the "flat vs climbing" story is invisible.
    def _mean_by_round(self, episodes):
        """{round_index: mean thoughtfulness} → sorted [(round, mean), ...]."""
        by_round = {}
        for e in episodes:
            r = e.get("round_index", -1)
            by_round.setdefault(r, []).append(e["score"]["thoughtfulness"])
        return [(r, sum(v) / len(v)) for r, v in sorted(by_round.items()) if r >= 0]

    def round_series(self, condition):
        """Mean thoughtfulness per round for a training condition (solo|swarm)."""
        return self._mean_by_round(self.training_episodes(condition))

    def validation_series_by_round(self):
        """Mean held-out thoughtfulness per round — the rising generalization line."""
        return self._mean_by_round(self.validation_episodes())
