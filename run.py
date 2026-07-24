"""Gandalf Protocol — Entrypoint [OWNER: Kirk]

Examples:
  python run.py --smoke                 # tiny mock run, prove the wiring
  python run.py --ablation --rounds 6   # solo vs swarm + validation, then charts
  ANTHROPIC_API_KEY=sk-... python run.py --smoke   # first REAL run — check the bill!

Automatically uses local JSON (--smoke, dev) or Azure Blob (production).
"""
import argparse, os
import config, llm
from agents import world
import runner, dashboard

# NOTE: Store.__init__ now auto-loads store.json (so `python dashboard.py` can replay a
# prior run). A run.py invocation is a NEW experiment, so we start from a clean file
# unless --resume is passed. dashboard.py never deletes — it only reads.

# Auto-select store backend
if config.USE_AZURE:
    from store_azure import Store
else:
    from store import Store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--ablation", action="store_true", help="run solo AND swarm")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify wiring")
    ap.add_argument("--resume", action="store_true",
                    help="append to the existing store.json instead of starting fresh")
    ap.add_argument("--domain", default="gift", choices=["gift", "fantasy"],
                    help="which Gandalf demo domain to run (gift = LLM judge; fantasy = "
                         "ground-truth judge, self-evolving swarm on real NFL outcomes)")
    args = ap.parse_args()

    # Fantasy is a self-contained domain (deterministic ground-truth judge, no LLM/gateway) —
    # dispatch to it and leave the gift/deployed path entirely untouched below.
    if args.domain == "fantasy":
        from fantasy.run import main as fantasy_main
        fantasy_main()
        return

    mode = "MOCK (no cost)" if config.MOCK else "REAL Claude"
    print(f"=== Gandalf Protocol === mode: {mode}")
    if args.smoke:
        args.rounds = 2
        per_round = config.SMOKE_PERSONAS_PER_ROUND
    else:
        per_round = config.PERSONAS_PER_ROUND

    # Fresh experiment: clear the local store first so we don't append to a stale run.
    if not args.resume and not config.USE_AZURE and os.path.exists(config.STORE_PATH):
        os.remove(config.STORE_PATH)

    store = Store(config.STORE_PATH) if not config.USE_AZURE else Store()
    if config.USE_AZURE:
        store.load()
    held_out = world.load_seed_personas()   # reuse seed set as a stand-in held-out set
    for p in held_out:
        p.is_validation = True

    conditions = ["solo", "swarm"] if args.ablation else ["swarm"]
    for cond in conditions:
        print(f"-- running condition: {cond}")
        try:
            runner.run_experiment(store, args.rounds, cond, per_round,
                                  held_out=held_out if cond == "swarm" else None)
        except RuntimeError as e:
            # [P1] park-not-kill: a tripped cost guard stops this condition but we still
            # render charts from whatever was stored, rather than losing the whole run.
            print(f"  !! {cond} halted: {e}")

    print(f"Total LLM calls (last condition): {llm.calls_made()}")
    charts = dashboard.render(store)
    print("Charts:", charts)


if __name__ == "__main__":
    main()
