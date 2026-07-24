# Wiring Gandalf into CU2 apps — integration spec (the flywheel)

**Status:** design / G2. **Author:** Dev Manager. **Depends on:** the gift loop demonstrably
climbing (judge calibration, #5) before any app integration goes live.

## Thesis
Gandalf is a **domain-agnostic learning engine**, not a gift app. Any product with (a) a prompt that
*proposes* something and (b) an *outcome signal* can plug in and get a compounding **data flywheel**:

```
        ┌──────────────────────── the flywheel ────────────────────────┐
        │                                                               │
   App proposes  ──▶  real users react (rating / return / click)  ──▶  outcomes
        ▲                                                               │
        │                                                               ▼
   better prompts  ◀──  Gandalf distills a Playbook  ◀──  Gandalf learns on those outcomes
```

The pitch: *"we didn't build a recommender this weekend — we had one in production, and we taught it
to learn from its own outcomes."*

## First integration: Broflo (gifts) — the natural fit
Broflo already has the two hooks:
- a suggestion prompt (`services/ai/app/prompt.py::build_system_prompt`) with a place for learned lessons,
- an outcome signal (`GiftRecord.rating` 1–5, `never_again`, `dismissed`).

### Direction A — Gandalf → Broflo (inject the Playbook)
1. Gandalf **publishes a versioned Playbook artifact** (JSON) to a shared store (blob / ops-platform):
   ```json
   {"version": 18, "generated_at": "...", "judge_kappa": 0.71, "validation_score": 0.62,
    "lessons": [{"text": "...", "confidence": 0.83, "source_episode_ids": [...]}]}
   ```
2. Broflo's `/suggest` fetches the current Playbook and injects the **top-k relevant lessons**
   (same retrieval approach as `retrieval.py`) into its "LEARNED LESSONS" prompt slot.
3. **Behind a feature flag** (`PLAYBOOK_ENABLED`, default off) + a **pinned version** so it's
   rollback-safe. Ship as A/B (playbook-on vs off) to *measure lift* on Broflo's own metrics.

### Direction B — Broflo → Gandalf (ground it in reality)
1. Broflo **exports anonymized outcomes** (nightly): `{profile, gift_given, rating, returned, never_again}`.
2. Gandalf maps them to **real personas + hidden_truth** (rating high → `would_love`; returned /
   never_again → `would_return`) and holds a slice out as the **real validation set** — so the
   held-out line is *real outcomes*, not fiction. This is the honest-learning claim made true.

## The contract (what the two apps agree on)
Two small, versioned schemas own the boundary (mirrors the TrendForge/Above-It-All pattern of a
canonical envelope):
- **`PlaybookArtifact`** (Gandalf → App): the JSON above. Published to `gandalf/playbook/vN.json`.
- **`OutcomeExport`** (App → Gandalf): `{anon_profile, proposal, outcome_signal, occasion, ts}`.
Transport: shared blob or an ops-platform endpoint; both apps already reach both.

## Other apps that plug into the same pattern
| App | "Proposal" | "Outcome signal" | Notes |
|---|---|---|---|
| **Broflo** (first) | gift suggestions | rating / return / never_again | best fit; hooks already exist |
| Connected Brand | clip/post variants | engagement, /eval taste score | it already has a taste-eval loop to feed outcomes |
| Greenvisor / RedVisor | AP/recon suggestions | analyst accept/override | high-value, but member/financial data → strict PII gating |
| Date-night / Travel / Fantasy (our new domains) | plans / itineraries / lineups | user pick / actual result | these are native Gandalf domains, not external apps |

## Safety & governance (non-negotiable for member-facing apps)
- **Never inject an uncalibrated Playbook into live suggestions.** Gate publish on: judge **κ ≥ 0.6**,
  held-out validation **rising**, per-lesson confidence. A flat/uncalibrated learner must not touch prod.
- **PII:** Broflo dossiers/outcomes carry PII → route exports through **Presidio (tokens-out, ADR-028)**
  before Gandalf sees them; Gandalf re-joins tokens only inside the CU2 boundary. Never log cleartext.
- **Rollout as canary** (the practitioner rule): shadow → small % traffic → full, promote only when the
  playbook-on arm wins on Broflo's own metric for a sustained window. Feature-flag + instant rollback.
- **Audit:** log which Playbook version influenced which suggestion (ops-platform audit).
- **Gateway:** all LLM calls stay on `ca-litellm-gateway` (shared budget, logging) — no new key surface.

## Phasing (do NOT skip Phase 0)
0. **Prove the loop learns** — gift curve climbs on the calibrated judge (#5) + evals green. *Gate.*
1. **Contracts** — lock `PlaybookArtifact` + `OutcomeExport` (this spec → schemas).
2. **Publish** — Gandalf writes a versioned Playbook artifact after each run.
3. **Shadow** — Broflo reads it behind `PLAYBOOK_ENABLED=false`, logs what it *would* change (no serving).
4. **A/B** — flag on for a slice; measure suggestion-quality lift vs control.
5. **Ground** — Broflo exports anonymized outcomes → Gandalf's real validation set → full flywheel,
   monitored in ops-platform / Mission Control.

## Reuse (don't rebuild)
Gateway (done), Presidio (PII), ops-platform (registry/audit/storage), `retrieval.py` (playbook top-k),
the Broflo prompt slot + dossier/outcome schema we already vendored in `reference/broflo/`.

## Open questions for Kirk / the team
- Playbook store: blob artifact vs an ops-platform endpoint vs Actian VectorAI (pending Violet's #19)?
- Broflo A/B metric: which existing Broflo quality signal is the ground truth for "did the playbook help"?
- Who owns the Broflo-side change — Broflo team, or a Gandalf PR into Broflo?
