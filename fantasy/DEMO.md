# Gandalf on Fantasy Football — the self-improvement demo

**One line:** Gandalf's arguing agents draft a fantasy team, study their own mistakes, distill a
playbook, and replay the *same* season with what they learned — the rookie misses the playoffs, the
veteran clinches. The judge is ground truth (actual NFL points), so the learning is unfakeable.

## Run it
```
python fantasy/season.py        # prints the leave-one-season-out results
python fantasy/export_demo.py    # regenerates fantasy/demo_data.json (what the UI renders)
```
UI: `fantasy/ui.html` (self-contained) — served live at `/fantasy`, also an Artifact.

## The arc (what the page walks through)
1. **The scoreboard.** Same season, same player pool, same schedule, full **PPR** lineup (incl. K + D/ST).
   One agent (drafts the hot hand) scores **95.9 pts/wk — misses the playoffs (8th)**. The swarm's
   veteran (learned influence) scores **100.4 — makes them (5th)**: **+4.5 pts/wk is the difference
   between a playoff seed and watching from home.**
2. **The film room.** Four agents argue — Recency, Season, Volume, Trend. The swarm distills an
   influence share and reality *reweights* the argument: full-season production earns ~**50%**, the
   one-agent "start whoever's hot" instinct gets demoted to ~22%.
3. **The tape.** Replaying 2025 week-by-week, the veteran teams pull away and claim the playoff seeds.
4. **The draft board.** The picks they disagreed on: one-agent *traps* (Michael Pittman, Courtland
   Sutton — hot early on TDs, then faded) vs veteran *gems* (Josh Jacobs, Kyren Williams — volume
   backs the rookie undervalued, then produced). Reality settled every argument.
5. **The integrity check** (say this if a judge probes "is this real?").

## Why it's honest — the three guarantees (all true by construction)
- **Walk-forward.** Every draft reads only weeks 1-3; it's graded on weeks 4-14 — games that hadn't
  happened at decision time. The outcome is never visible when the choice is made. No lookahead.
- **Held-out influence.** The influence share is four signal weights, fit on the *other* seasons via
  leave-one-season-out — it never sees the season it's scored on. Four numbers can't memorize a
  player's weekly box score.
- **It generalizes.** The identical weights lift *every* season (lift/wk): 2022 **+10.3**, 2023
  **+5.7**, 2024 **+11.7**, 2025 **+4.5** — and in 2022/2024 that's a clean 6-8 → 8-6. Memorizing one
  season would do nothing for the others.

The full PPR lineup adds a ~13-pt/wk K+D/ST "wash" to both sides, so the *records* compress (7-7 in
two seasons) while the **playoff outcome still flips** — that's the honest floor story: consistency,
not ceiling, gets you in.

## The numbers behind it
- Influence fit on **~12,900 walk-forward (player, week) decisions** across three seasons (skill players).
- 12-team league (6 one-agent GMs vs 6 veteran GMs), 16-man rosters, start 1 QB / 2 RB / 2 WR / 1 TE /
  1 FLEX / 1 K / 1 D/ST. **PPR** scoring (confirmed from the data: points = yards + 1/reception).
- Standings by **all-play win%** (fraction of the field you outscore each week) — removes schedule luck.
- **My Team** connects a user's real ESPN league and auto-configures *their* scoring + lineup slots.

## How it maps to Gandalf
Same loop as the gift domain — propose → argue → judge → distill → improve — on a domain with a
**real** reward signal instead of an LLM's opinion. The swarm/judge/distill machinery is the product;
fantasy is one demo domain (Kirk/Jasper/Violet are building others).
