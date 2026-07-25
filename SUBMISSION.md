# Gandalf Protocol — SwarmHack submission

**A self-play learning engine for multi-agent systems.** Agents propose solutions to simulated
scenarios, a judge scores outcomes, the wins distill into a shared **playbook** every agent reads
next round, and a coach targets the weakest area. It gets better without fine-tuning — the policy
*is* the playbook. First domain: gift recommendations (Broflo). Second, already live: fantasy
football judged by real NFL outcomes.

- **Live app:** https://ca-gandalf-protocol.wittyflower-1831f2a2.westus2.azurecontainerapps.io
- **Demo video:** https://share.descript.com/view/VgZ78jr5pmP
- **Demo script:** [`DEMO-SCRIPT.md`](./DEMO-SCRIPT.md) · **How it works:** `/guide` on the live app

## The loop

```
Coach (find the weak spot) → World (generate personas / draw real scenarios)
   → Swarm (4 diverse agents propose) → Judge (score thoughtfulness / ground truth)
   → Distill (winning lessons → shared Playbook) ↺   [the Playbook is the learned policy]
```

Three charts are the whole proof, and one keeps us honest:
1. **Learning curve** — thoughtfulness rises round over round.
2. **Ablation** — a solo agent that can't share stays flat; the sharing swarm scores above it.
3. **Held-out validation** — rises on scenarios never trained on. This is our **anti-reward-hacking
   canary**: if training climbed while this stayed flat, we'd be gaming our own judge — and we'd say so.

## Sponsor usage (honest)

| Sponsor | How we use it |
|---|---|
| **Actian VectorAI DB** | Playbook lessons stored as MiniLM embeddings; **cosine semantic top-k retrieval** per persona picks the most relevant lessons to inject into the agent prompt. See the live **before/after** at `/actian`. Fails safe to a lexical ranker so a run never breaks. |
| **Guild** | Operators start and monitor live runs from a **Guild control plane** (OpenAPI + agent in [`guild/`](./guild)) driving Gandalf's real REST API. |
| **DeepMind** | Lineage/inspiration — **self-play** in the AlphaGo tradition (explore → evaluate → distill → re-play), applied to a real product. No DeepMind product is integrated; the paradigm is the point. |

## Multiple use cases (one engine)

- **Gifts** (`/demo`, `/`) — swarm scores a consistent **+0.138** thoughtfulness above solo.
- **Fantasy football** (`/fantasy`) — judged by **real 2025 NFL points** (ground truth, not an LLM):
  the naive drafter (**95.9** pts/wk) misses the playoffs; the swarm that learned (**100.4**)
  clinches — **+4.5 pts/week**, same players and schedule, the only variable is what it learned.
- **Roadmap** ([`DOMAINS.md`](./DOMAINS.md)) — travel, date-night: same loop, swap the world + judge.

## Architecture

- Python **Flask** app on **Azure Container Apps**; episodes/charts/playbook in **Azure Blob**.
- All LLM calls route through a single gateway chokepoint (budget-capped, usage-logged) — no raw
  keys in code (verified: history is clean).
- Retrieval seam: `retrieval.py` (`Playbook.as_prompt(profile)` → Actian semantic top-k → lexical
  fallback). Distillation transfers, doesn't fossilize: incremental delta lessons, cap+prune,
  Wilson-confidence, distiller ≠ judge (no self-confirmation).

## Where we are / next steps

- **Done:** end-to-end loop live on Azure; 3 charts + server-side A/B verdict; Actian semantic
  retrieval (before/after visible); two domains (gifts + fantasy); Guild control plane.
- **Next:** lower the gift judge's ceiling + add harder personas so gifts show a *climb* (not just
  the gap); expand Actian-backed cross-episode memory; add the travel/date-night worlds.

## Run it

```bash
pip install -r requirements.txt
python run.py --ablation --rounds 6     # offline mock mode, renders the 3 charts, zero cost
```

## Team

- **Kirk Drake** — The Rig (loop, storage, dashboard, integration, demo)
- **Warren** — The Swarm (diverse agents + distillation) & the fantasy-football domain
- **Jasper** — The Signal (judge calibration, curriculum) & the Guild control plane
- **Violet** — The World (personas, held-out validation set, Actian retrieval spec)
