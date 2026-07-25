# Gandalf Protocol — <3-minute demo script

**Live app:** https://ca-gandalf-protocol.wittyflower-1831f2a2.westus2.azurecontainerapps.io

Everything below the fold replays from **banked data** (charts, verdict, playbook) — nothing has
to train during the demo. The only live-LLM moment is the flywheel in Beat 3 (pre-warm it once
before recording). Total target: **~2:40**.

> **Pre-flight (do once, right before recording):** open each URL and confirm it renders; click
> `/demo`'s first recipient once so the model is warm; confirm `/actian` shows the Actian badge.

---

### Beat 1 — `/guide` · 25s · *the thesis (DeepMind lineage)*
> "Every team at this hackathon hand-tunes a frozen model. **Ours manufactures its own training
> data — and gets better overnight.** Four agents explore different strategies, a judge scores
> them, the wins distill into a shared playbook every agent reads next round. It's self-play —
> the AlphaGo idea — but the *policy is a text playbook*, so there's **zero fine-tuning**."

Scroll the loop diagram (Coach → World → Swarm → Judge → Distill ↺).

### Beat 2 — `/fantasy` · 40s · *the learning proof (ground truth) — LEAD HERE*
> "Here's the proof, on the hardest possible judge: **real NFL outcomes.** Same engine, playing
> fantasy football. The naive drafter — **95.9 points a week — misses the playoffs.** The swarm
> that learned — **100.4 — clinches.** Same players, same schedule; the only variable is *what the
> swarm learned* — worth **+4.5 pts/week**, the gap between a playoff seed and watching from home.
> No LLM grading its own homework; the scoreboard is ground truth."

Point at the 95.9-vs-100.4 head-to-head and the four agents' learned influence shares.

### Beat 3 — `/` · 25s · *the honest instruments (banked)*
> "The same loop on our first product — gifts. **Ablation:** a solo agent that can't share stays
> flat; the sharing swarm scores above it. **Held-out validation** rises on cases we never trained
> on — our **anti-reward-hacking canary**: if it stayed flat while training climbed, we'd be gaming
> our own judge, and we'd see it. And the server-side **A/B verdict: distillation helps.**"

Point at the 2×2 chart grid + the "distillation HELPS ✅" verdict card.

### Beat 4 — `/demo` then `/actian` · 45s · *the flywheel + Actian as memory*
> "The payoff for a product: same gift agent, **left** with an empty playbook, **right** the *same
> agent* with the swarm's learned lessons — better gifts." *(click one recipient — pre-warmed)*
>
> "And its memory is **Actian VectorAI DB.**" *(switch to `/actian`)* "Left, keyword match. Right,
> **Actian semantic search** — it recalls the lesson that *means* the same thing even with no shared
> words. The starred ones keyword search misses entirely. Four agents do the learning; **Actian is
> their memory.**"

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
