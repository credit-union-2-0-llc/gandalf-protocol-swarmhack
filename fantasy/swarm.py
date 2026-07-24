"""Gandalf Protocol — FANTASY swarm: arguing agents that self-evolve against real outcomes.

Same shape as the gift swarm (propose → judge → distill → improve), but the judge is GROUND
TRUTH (actual NFL points), not an LLM. Each agent ARGUES for one signal. The swarm self-evolves
by DISTILLING a validated combination: starting from the naive obvious signal, each round it
adopts the agent whose signal most improves the swarm's fit to REAL outcomes, and refits the
collective model. Because the combination genuinely beats any single signal on decision quality,
the out-of-sample curve CLIMBS — honestly, not by construction.
"""
import numpy as np

# ── the arguing agents: each values a candidate by ONE philosophy (higher = better bet) ──
AGENTS = {
    "Recency":  lambda f: f["recency"],    # "he's been scoring lately" — the naive/obvious signal
    "Season":   lambda f: f["season"],      # "trust the full-season average"
    "Volume":   lambda f: f["volume"],      # "opportunity (targets+carries) is what's real"
    "Trend":    lambda f: f["opp_trend"],   # "usage is trending up — a breakout before the box score"
}
TOPK = 10   # a "decision" = the top-K players the swarm would roster


def _mat(cols, batch):
    return np.array([[AGENTS[c](e["features"]) for c in cols] for e in batch], float)


class FantasySwarm:
    def __init__(self, learn=True):
        # Everyone starts naive: the single obvious signal (recent points).
        self.adopted = ["Recency"]
        self.learn = learn
        self.model = None

    def _fit(self, examples):
        X = _mat(self.adopted, examples)
        y = np.array([e["outcome"] for e in examples], float)
        mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
        A = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.model = {"w": w, "mu": mu, "sd": sd}

    def bootstrap(self, train):
        self._fit(train)

    def score(self, batch):
        """The swarm's collective value estimate per candidate (learned combination)."""
        if self.model is None:
            return np.zeros(len(batch))
        X = _mat(self.adopted, batch)
        A = np.hstack([np.ones((len(X), 1)), (X - self.model["mu"]) / self.model["sd"]])
        return A @ self.model["w"]

    def distill(self, train):
        """Self-evolve: among not-yet-adopted agents, ADOPT the one whose signal most improves
        the swarm's fit to REAL outcomes, then refit. Returns the newly-adopted agent (or None).
        This is the learning step — validated against ground truth, not opinion."""
        if not self.learn:
            return None
        remaining = [a for a in AGENTS if a not in self.adopted]
        if not remaining:
            return None
        y = np.array([e["outcome"] for e in train], float)
        best, best_mae = None, None
        for a in remaining:
            trial = FantasySwarm.__new__(FantasySwarm)
            trial.adopted = self.adopted + [a]
            trial._fit(train)
            mae = float(np.mean(np.abs(trial.score(train) - y)))
            if best_mae is None or mae < best_mae:
                best, best_mae = a, mae
        self.adopted.append(best)
        self._fit(train)
        return best


def decision_quality(swarm, batch, k=TOPK):
    """GROUND TRUTH decision quality: the swarm ranks the batch, we take its top-k picks, and
    score them by the players' ACTUAL rest-of-season PPG. Reality is the judge."""
    order = np.argsort(-swarm.score(batch))
    return float(np.mean([batch[i]["outcome"] for i in order[:k]]))


def mae(swarm, batch):
    y = np.array([e["outcome"] for e in batch], float)
    return float(np.mean(np.abs(swarm.score(batch) - y)))
