# Multi-domain: one engine, many learning loops

Gandalf's loop is **domain-neutral**. The gift-recommender is just the first domain. To add
your own (fantasy-football, travel agent, date-night planner), you implement **one plugin** — you
do NOT fork the system.

## What's shared (never copy these) vs what's per-domain

| Shared core (the engine) | Per-domain plugin (you build) |
|---|---|
| `runner.py` — the loop (propose→react→judge→save→distill), parallel, periodic validation | **Scenario generator** — the "personas" of your domain |
| `store.py` / `store_azure.py` — episode persistence | **Agent prompt + strategies** — the policy that proposes |
| `dashboard.py` — the 3 charts | **Judge rubric** — how "good" is scored (dims + gold set) |
| `retrieval.py` — playbook RAG | **Reactor** — the honest reaction against hidden truth |
| `contracts.py`, `llm.py`, `app.py`, `evals/`, CI, deploy | **Seed scenarios** + **eval gold set** |

Only the four bold pieces are gift-specific today (they live in `agents/world.py`,
`agents/swarm.py`, `agents/signal.py`, `seed_personas.json`).

## The Domain interface (target shape)

We extract a `Domain` protocol so the core loop calls the domain, not gift-specific functions:

```python
# domains/base.py  (the contract — Kirk owns, extracts from the gift loop)
class Domain(Protocol):
    name: str
    def generate_scenarios(self, batch) -> list[Persona]: ...     # was world.generate_personas
    def build_agent_prompt(self, strategy, playbook_block) -> str: ...  # was swarm.build_gift_system
    strategies: list[str]                                          # diverse strategy tags
    def react(self, scenario, proposal) -> Reaction: ...          # was world.react
    def judge(self, scenario, proposal, reaction) -> Score: ...   # was signal.judge
    def seed_scenarios(self) -> list[Persona]: ...                # held-out set
    gold_set_path: str                                            # evals/<domain>_gold.json
```

`Persona`/`GiftProposal`/`Reaction`/`Score`/`Episode` in `contracts.py` stay the same shapes — a
"gift" is just the proposal payload; for fantasy football the payload is a lineup, for travel an
itinerary, for date-night a plan. The judge dims (`fit/surprise/effort/budget_respect`) are generic
enough to reuse or lightly rename per domain.

## Adding your domain (the recipe)

1. `domains/<yourname>/` implementing the `Domain` protocol (start by copying the gift domain).
2. Define **3–4 strategies that genuinely disagree** (the swarm only learns if they explore differently).
3. Write a **judge rubric** with binary sub-checks (bad proposal → <0.4) + an **`evals/<domain>_gold.json`**.
4. 8–12 **seed scenarios** with honest `hidden_truth`.
5. Run it: `python run.py --ablation --rounds 6 --domain <yourname>` → your own climbing curve.
6. `python evals/run_evals.py --domain <yourname>` → your regression net.

## The domains (owners)

| Domain | Owner | "Proposal" is… | "Scenario" is… | Judge rewards… |
|---|---|---|---|---|
| gifts (reference) | team | a gift set | a recipient persona | thoughtfulness |
| **fantasy-football** | **Warren** | a weekly lineup | a matchup + roster + constraints | projected-vs-actual edge, not chalk |
| **travel-agent** | **Jasper** | an itinerary | a traveler + budget + constraints | fit to taste, surprise, budget, feasibility |
| **date-night-planner** | **Kirk** | a date plan | a couple + vibe + constraints | delight, effort, respects constraints/budget |

> **Sequence (important):** get the **gift domain's curve actually climbing first** (Jasper's judge
> calibration, #5) — it's the reference implementation. Cloning the pattern before one domain
> demonstrably learns just produces N flat loops. Prove one, then fan out.

> **Violet:** the persona/scenario realism skill is now a *shared* concern every domain needs — own
> the scenario-generation quality bar across domains, or take a domain of your own. TBD with the team.
