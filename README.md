# Gandalf Protocol

A self-play learning orchestrator for multi-agent systems. Agents propose solutions to simulated
scenarios, a judge scores outcomes, a swarm shares what works via a growing knowledge base, and
a coach targets weak spots. Charts prove it learns — and learns *faster together*.

First use case: Broflo gift recommendations. But Gandalf works for any domain where you can
simulate counterparties and measure success.

## Deployment options

- **Local (tonight, 5 hours):** `python run.py --ablation` on your laptop. Uses local `store.json`.
- **Azure (full deploy):** See `DEPLOY.md`. Flask API + Blob Storage. Hit `/api/run` endpoints remotely.

---

## Local: It already runs (in mock mode, zero cost)

```bash
pip install -r requirements.txt
python run.py --ablation --rounds 6      # runs offline, renders 3 charts
```

## Azure: Containerized + Cloud Storage

```bash
# Prerequisites: Azure CLI, storage account, container registry (see DEPLOY.md)
chmod +x deploy.sh
./deploy.sh

# Once deployed, trigger a run via HTTP:
curl -X POST "http://gandalf-protocol.westus2.azurecontainers.io:5000/api/run?rounds=6&condition=ablation"
```

See **`DEPLOY.md`** for full setup instructions, cost notes, and integration with `ops-platform`.

No API key = **MOCK mode**: deterministic fake responses so you can verify all the wiring and
see the dashboard render before spending anything. The mock is rigged so the swarm curve
out-climbs the solo curve — that's a sanity check of the *charts*, not real learning.

## Flip to real Claude

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python run.py --smoke                     # TINY real run FIRST — then check the bill
python run.py --ablation --rounds 6       # the real experiment
```

> **Cost gate:** always `--smoke` first on real Claude. `config.MAX_CALLS_PER_RUN` hard-stops
> a runaway run. Raise it only when you mean to.

## Layout / ownership

```
config.py             run sizes, models, cost cap, mock toggle
contracts.py          LOCKED shared types (file 00)          [everyone reads]
llm.py                Claude wrapper + mock engine
store.py              JSON storage                            [Kirk]  (swap to ops-platform later)
runner.py             the loop + ablation + validation        [Kirk]
dashboard.py          the 3 charts                            [Kirk]
run.py                entrypoint                              [Kirk]
agents/swarm.py       gift agents, diversity, distillation    [Warren]
agents/world.py       persona gen, reactor, perception        [Violet]
agents/signal.py      judge + coach                           [Jasper]
seed_personas.json    bootstrap + stand-in held-out set       [Violet]
```

Every module has a **working version** so the loop runs today. Search `TODO(name)` for the
real-build hooks each owner fills in.

---

## TONIGHT — 3 people, ~5 hours, Warren driving

Warren owns the Swarm, but it's already stubbed with a working single-agent + 4-agent swarm +
basic distillation — **so his absence blocks nobody.** He sharpens it when he's back. Tonight's
job is to make the loop *real* and get a genuine curve.

**Hour 0 (all three, together):** `pip install -r requirements.txt`, run
`python run.py --ablation` in mock, open the 3 PNGs. Confirm everyone sees the same charts.
Lock the contracts by reading `contracts.py` out loud. This is your integration gate — passed
before you write a line.

**Then split for ~4 hours:**

- **Violet — `agents/world.py`:** replace the persona prompt with real Broflo Dossier fields;
  make the reactor react *honestly* against `hidden_truth`. Add 5–10 real seed personas to
  `seed_personas.json`. *Test:* generated personas should feel like specific humans.
- **Jasper — `agents/signal.py`:** calibrate the judge (feed it a gift that's in `already_has`
  and confirm it scores LOW — do this explicitly), then make the coach pick the real weakest
  category. *Test:* obviously-bad gift → thoughtfulness < 0.4.
- **Kirk — `runner.py` + `dashboard.py` + `store.py`:** do the first **real** `--smoke` run
  (check the bill), confirm the curve is real not mock, prettify the 3 charts for the demo,
  and start the `store.py` → `ops-platform` swap (keep the method signatures).

**Last 30 min (together):** one real `--ablation --rounds 6`. If the swarm curve beats solo on
*real* data, you have the entry. Commit. Leave the Perception grounding + Warren's swarm polish
for tomorrow.

**Tonight's win condition:** a real (non-mock) learning curve rendering from `store.json`.
Everything else is upside.

## The never-cut trio
1. the loop runs end-to-end  2. the solo-vs-swarm ablation renders  3. the held-out validation line renders.
