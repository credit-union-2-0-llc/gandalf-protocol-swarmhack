# Gandalf Protocol — <3-minute demo script

**Live app:** https://ca-gandalf-protocol.wittyflower-1831f2a2.westus2.azurecontainerapps.io

Everything below the fold replays from **banked data** (charts, verdict, playbook) — nothing has
to train during the demo. The only live-LLM moment is the flywheel in Beat 3 (pre-warm it once
before recording). Total target: **~2:40**.

> **Pre-flight (do once, right before recording):** open each URL and confirm it renders; click
> `/demo`'s first recipient once so the model is warm; confirm `/actian` shows the Actian badge.

---

### Beat 1 — `/guide` · 30s · *the thesis (DeepMind lineage)*
> "Every team at this hackathon hand-tunes a frozen model. **Ours manufactures its own training
> data — and gets better overnight.** Four agents explore different strategies, a judge scores
> them, the wins distill into a shared playbook every agent reads next round. It's self-play —
> the AlphaGo idea — but the *policy is a text playbook*, so there's **zero fine-tuning**."

Scroll the loop diagram (Coach → World → Swarm → Judge → Distill ↺) and the "three charts, one is
honest about cheating" section.

### Beat 2 — `/` · 25s · *the proof (banked)*
> "It's running live on Azure. Here are the charts. **Ablation:** a solo agent that can't share
> stays flat; the swarm that distills scores consistently above it. **Held-out validation** rises
> on cases we never trained on — that's our **anti-reward-hacking canary**: if it stayed flat
> while training climbed, we'd be gaming our own judge. And the **A/B verdict** — distillation on
> vs off — is computed server-side."

Point at the "distillation …" verdict card and the 2×2 chart grid.

### Beat 3 — `/demo` then `/actian` · 45s · *self-learning + Actian, before/after*
> "Same gift agent. **Left**, empty playbook. **Right**, the *same agent* with the lessons the
> swarm learned — better, more thoughtful gifts. That's the flywheel a product like Broflo gets for
> free." *(click one recipient — pre-warmed)*
>
> "How does it pick the right lessons? **Actian VectorAI DB.**" *(switch to `/actian`)* "Left, a
> keyword ranker. Right, **Actian semantic search** over embeddings — it surfaces the lesson that
> *means* the same thing even with no shared words. The starred ones are lessons keyword-match
> misses entirely. That's the retrieval quality Actian buys us."

### Beat 4 — `/fantasy` · 40s · *multiple use cases + honest eval*
> "Same engine, different world: **fantasy football**, judged by **real 2025 NFL points** — hard
> ground truth, no LLM opinion. Watch the swarm climb from a naive 17.0 to 20.6 points per starter,
> and beat the naive roster that chased busts. **One loop; swap the world and the judge, and it
> learns a new domain.** Gifts today; travel and date-night next."

Let the SVG learning curve animate; point at the swarm-vs-naive roster.

### Beat 5 — Guild + close · 20s · *control plane + the close*
> "Operators start and monitor these runs from a **Guild control plane** — this OpenAPI + agent
> driving our live API." *(show Jasper's Guild dashboard)*
>
> "We didn't build a gift recommender in a weekend. We spent the weekend teaching it to **learn** —
> self-play, an honest held-out eval, semantic memory in Actian, and it's already portable across
> domains. That's Gandalf Protocol."

---

### Fallback if the gateway is flaky at record time
`/guide → / → /fantasy` are 100% static/banked. Skip the live click in Beat 3 and *describe* the
flywheel from the already-rendered `/demo` page. Every sponsor + the learning story still land.
