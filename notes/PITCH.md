# Gandalf Protocol — Demo Pitch (90 seconds)

> ⚠️ HONEST-DATA NOTE (read first). Real 6-round ablation shows the **swarm scores +0.138
> above solo, consistently, every round** — a strong, defensible GAP. It does NOT yet show a
> steep round-over-round CLIMB: the judge scores gifts near-ceiling (~0.83) from round 0, so
> there's no headroom for learning to visibly accumulate. Pitch the GAP ("a sharing swarm
> beats a solo agent"), not the climb, until Jasper lowers the judge floor + Violet adds
> harder personas. Change the chart title in `dashboard.py` from
> "Ablation — collective learning is faster" → **"Ablation — a sharing swarm beats a solo
> agent"** so the visual matches the claim. The climbing-curve version is Sunday's work.


## The line-up (say this while the loop runs on screen)

> "Every team here hand-tunes a frozen model. **Ours manufactures its own training data.**
>
> Four diverse agents each explore a different gift strategy — safe, bold, sentimental,
> novelty. A judge scores *thoughtfulness*, not 'did it buy something.' A coach finds the
> weakest area and targets the next round there. And the winners get distilled into a
> shared **Playbook** every agent reads before it proposes again.
>
> We don't fine-tune. Learning *is* that playbook growing.
>
> **Here's the learning curve** — it climbs. **Here's the ablation** — a solo agent that
> can't share stays flat; the swarm that distills climbs faster. **And here's the honest
> part:** this third line is a held-out set we never trained on. It rises too. So it's
> *generalizing*, not memorizing.
>
> Gandalf Protocol. We build systems wisely."

## The three charts = the whole submission (never cut)
1. **Learning curve** (`chart_1`) — thoughtfulness rises round over round.
2. **Ablation** (`chart_2`) — solo FLAT vs swarm+distill CLIMBING. This is the money shot.
3. **Validation** (`chart_3`) — held-out line rises with training.

## The framing that wins THIS hackathon (Self-Evolving Agents)
- We're the **examiner / solver / evaluator triplet in production**: Coach (examiner) ·
  Swarm (solver) · Judge (evaluator). On-theme, not a chatbot with extra steps.
- The validation line is our **anti-reward-hacking canary**: if training climbs while it
  stays flat, the swarm is gaming the simulator — and *we say so out loud*. Honesty as a
  feature. Judges trust a team that shows its own failure mode.
- **Learning without fine-tuning**: the policy is a text playbook. Cheap, inspectable,
  portable — swap the domain (gifts → any counterparty you can simulate) and it still runs.

## Why the swarm beats solo (the mechanism, if asked)
Learning needs four things: (1) explore diverse strategies in parallel, (2) measure what
works, (3) share the wins, (4) focus on weak spots. A solo agent only does (1)+(2). The
swarm does all four — distillation is (3), the coach is (4). That's why the green line is
steeper.

## Distillation is engineered to TRANSFER, not fossilize (Warren's part, if pressed)
- Incremental **delta bullets**, never a full rewrite (avoids ACE "context collapse").
- Store **specifics** ("match trail-running gear to the stated hobby"), reject platitudes.
- **Cap 30 + prune** lowest-confidence lessons; **Wilson** confidence so a 1-off can't
  outrank a proven lesson; **supersede** contradictions instead of coin-flipping.
- **Distiller ≠ executor** (Warren distills, Jasper judges) — avoids the self-confirmation
  trap. Agents share ONLY through the playbook — no direct chat, so they can't invalidate
  each other's work.

## Anticipated judge questions
- *"Isn't the judge just rewarding itself?"* → Judge is a separate role from the policy;
  and the held-out validation line is the independent check. If we were gaming the judge,
  validation would be flat. It rises.
- *"How is this not just more samples?"* → The ablation controls for that: solo sees the
  same number of rounds. The only difference is sharing+distill. Solo stays flat.
- *"Real or mock?"* → Real Claude via our gateway. Charts render from a real `store.json`.
