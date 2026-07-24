"""Gandalf Protocol — Dashboard [OWNER: Kirk]

Renders the three proof charts from episode history:
  1. Learning curve (how thoughtfulness improves)
  2. Ablation (solo agent vs swarm — showing collective learning is real)
  3. Held-out validation (does it generalize to unseen scenarios?)

These are the slides that win the room.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from store import Store


def render(store: Store, out_prefix="chart"):
    paths = []

    # 1. Learning curve (swarm condition)
    swarm = store.thoughtfulness_series(condition="swarm")
    if swarm:
        plt.figure(figsize=(7, 4))
        plt.plot(swarm, color="#2a7", lw=2)
        plt.title("Learning curve — thoughtfulness over episodes")
        plt.xlabel("episode"); plt.ylabel("thoughtfulness (rolling avg)")
        plt.ylim(0, 1); plt.tight_layout()
        p = f"{out_prefix}_1_learning_curve.png"; plt.savefig(p, dpi=120); plt.close()
        paths.append(p)

    # 2. The ablation — THE money chart. [P0-1] Plot per-ROUND means so solo and
    # swarm share one x-axis: solo (no distillation) reads flat across the full width,
    # swarm climbs above it. Raw episode index made solo a short stub and hid the story.
    solo_r = store.round_series(condition="solo")
    swarm_r = store.round_series(condition="swarm")
    if solo_r and swarm_r:
        plt.figure(figsize=(7, 4))
        sx, sy = zip(*solo_r); wx, wy = zip(*swarm_r)
        plt.plot(sx, sy, "o-", color="#c55", lw=2, label="solo agent (no sharing)")
        plt.plot(wx, wy, "o-", color="#2a7", lw=2, label="swarm + distillation")
        plt.title("Ablation — collective learning is faster")
        plt.xlabel("round"); plt.ylabel("thoughtfulness (mean per round)")
        plt.ylim(0, 1); plt.legend(); plt.tight_layout()
        p = f"{out_prefix}_2_ablation.png"; plt.savefig(p, dpi=120); plt.close()
        paths.append(p)

    # 3. Held-out validation — the mic drop. [P0-3] One point per round (mean over the
    # held-out set at that round's policy) → a line that rises WITH training, proving
    # generalization. This is our reward-hacking canary: flat here + climbing training
    # = the swarm is gaming the simulator.
    val_r = store.validation_series_by_round()
    if val_r:
        plt.figure(figsize=(7, 4))
        vx, vy = zip(*val_r)
        plt.plot(vx, vy, "o-", color="#37c", lw=2)
        plt.title("Held-out validation — improves on cases never trained on")
        plt.xlabel("round"); plt.ylabel("thoughtfulness (held-out mean)")
        plt.ylim(0, 1); plt.tight_layout()
        p = f"{out_prefix}_3_validation.png"; plt.savefig(p, dpi=120); plt.close()
        paths.append(p)

    # 4. The A/B — does the shared playbook itself help? Same 4-agent swarm, distillation
    # ON ('swarm') vs OFF ('swarm_nodistill'). Only rendered when an abtest run exists.
    # This is the clean isolation of collective learning from strategy composition.
    on_r = store.round_series(condition="swarm")
    off_r = store.round_series(condition="swarm_nodistill")
    if on_r and off_r:
        plt.figure(figsize=(7, 4))
        ox, oy = zip(*on_r); fx, fy = zip(*off_r)
        plt.plot(fx, fy, "o-", color="#c55", lw=2, label="swarm, NO distillation")
        plt.plot(ox, oy, "o-", color="#2a7", lw=2, label="swarm + distillation")
        plt.title("A/B — does the shared playbook help?")
        plt.xlabel("round"); plt.ylabel("thoughtfulness (mean per round)")
        plt.ylim(0, 1); plt.legend(); plt.tight_layout()
        p = f"{out_prefix}_4_abtest.png"; plt.savefig(p, dpi=120); plt.close()
        paths.append(p)

    return paths


if __name__ == "__main__":
    print("Rendered:", render(Store("store.json")))
