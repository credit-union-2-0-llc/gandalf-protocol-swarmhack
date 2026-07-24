"""Gandalf Protocol — The Signal [OWNER: Jasper]

Judge + Coach. The judge decides what "good" means (your success metric). The coach watches
the learning curve and targets weak spots. Together they direct all learning.

Judge design (per EMAIL-JASPER-AMENDED.md): binary sub-checks, not 1-5/0-1 scales — LLMs
can't hold numeric intervals consistently, and one team reported Cohen's kappa going
0.47 -> 0.78 just from switching to yes/no checks. `budget_ok` is computed in code (pure
arithmetic, mirrors Broflo's postprocess.filter_budget 10% tolerance) rather than asked of
the model — one less thing that needs calibrating and can't be talked into a wrong answer.
`fit_ok` combines a deterministic already_has veto with the LLM's semantic judgment, so a
duplicate gift can't be argued past by clever reasoning — a veto, not a vote.
"""
from statistics import pstdev

from contracts import Score, CurriculumBatch
from llm import call_llm

# ── Context isolation ──
# The judge sees ONLY hidden_truth + the gifts + the reaction. NEVER pass proposal.agent_id
# or proposal.strategy_tag, and never persona.profile, into this prompt — the giver's own
# reasoning about *why* it picked a strategy must not reach the judge, or it grades intent
# instead of outcome. A separate model *instance* is not independence; this is.
JUDGE_SYSTEM = """You are a STRICT, impartial judge of gift THOUGHTFULNESS — not "was it bought"
and not "how persuasive did the reasoning sound." You see the recipient's TRUE preferences
(hidden_truth), the gifts proposed, and how they reacted. Judge honestly against hidden_truth;
a well-argued case for a bad gift is still a bad gift.

Real-run finding (fix this): early judges here clustered almost everything ~0.8 with no
headroom — generic, plausible-but-unremarkable gifts were passing checks they shouldn't. Most
gifts, especially from a cold/untrained agent, ARE generic. Default to false; only pass a check
when the evidence is specific and clear, not merely plausible.

Answer each as true/false (binary, not a 1-5 or 0-1 scale):
- fit_ok: does at least one gift specifically match a would_love item (or its clear spirit),
  with none violating a stated constraint or duplicating something in already_has? A gift that
  only loosely relates to a stated top-level interest — without hitting a specific would_love —
  does NOT pass. Default false unless the match is concrete.
- surprise_ok: is any gift non-obvious — shows insight beyond the literal profile, not just a
  generic restatement of a stated interest (e.g. a gift card "because they like running")?
- effort_ok: does the reasoning cite a SPECIFIC, non-obvious detail about this person — not just
  the same interest category the gift itself is already about? "They like running, so running
  shoes" does NOT pass — that's circular, not evidence of attention. Reasoning that could be
  copy-pasted onto any recipient who shares the interest does NOT pass.

Output JSON: {"fit_ok": true, "surprise_ok": true, "effort_ok": true, "rationale": "..."}"""

COACH_SYSTEM = """You design the next training round for a swarm of gift agents.
Given their thoughtfulness stats by occasion (mean, sample count n, stdev), identify the
1-2 WEAKEST areas and specify the next batch to target them. Do not target mastered areas.
A high stdev at low n is noise, not a real weak spot — prefer categories with enough samples
to trust the mean. Keep synthetic scenarios grounded in realistic gift-giving, not contrived
edge cases the real product would never see.
Output JSON: {"target_weakness":"...","rationale":"...","n_personas":6}"""

# thoughtfulness = weighted count of passed checks — keeps the 0-1 curve the dashboard needs,
# fit stays dominant so a failed fit_ok can't be outvoted by the other three passing.
CHECK_WEIGHTS = {"fit_ok": 0.4, "surprise_ok": 0.2, "effort_ok": 0.2, "budget_ok": 0.2}

# A failed fit_ok is a HARD gate, not just a heavy weight. fit_ok=False means the gift
# violates a constraint, duplicates something owned, or matches nothing in would_love — issue
# #5's DoD names "bad gift < 0.4" as the bar, and pure weighting let such a gift reach 0.6 when
# the other three checks passed (thanks @Warren). Cap below the bar so it can never clear it,
# no matter how surprising/effortful/on-budget the rest of the proposal looks.
FIT_FAIL_CEILING = 0.3

# Fuzzy-match tolerance for the already_has veto — mirrors Broflo's never_again/gift_history
# dedup (reference/broflo/services/ai/app/postprocess.py: _fuzzy_match, threshold=3).
DUPLICATE_EDIT_DISTANCE = 3
BUDGET_TOLERANCE = 1.1  # 10% over — mirrors postprocess.filter_budget's ceiling
GROSS_BUDGET_MULT = 1.5  # beyond this x the ceiling, over-budget is a HARD fail, not a soft ding

# Below this many samples, a category's mean is too noisy for the coach to chase.
MIN_SAMPLES_PER_CATEGORY = 3


def _levenshtein(a: str, b: str) -> int:
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def _duplicate_veto(gifts: list, already_has: list) -> bool:
    """True if ANY gift in the proposal fuzzy-matches something already owned — a hard fail the
    LLM can't be talked out of, no matter how the proposal's reasoning is worded.

    'any', not 'all': Violet's world (#6) stamps a proposal owns_it/return the moment ONE gift
    matches, so a 2-gift proposal (1 dupe + 1 great) already gets a bad reaction — the fit veto
    must agree, or the reward signal is inconsistent across the two files (thanks @Warren)."""
    if not already_has or not gifts:
        return False
    names = [str(item.get("value", item)) if isinstance(item, dict) else str(item)
              for item in already_has]
    return any(_levenshtein(g.get("name", ""), n) < DUPLICATE_EDIT_DISTANCE
               for g in gifts for n in names)


def _budget_ok(gifts: list, budget: dict) -> bool:
    """Deterministic — this is arithmetic, not judgment, so there's nothing to calibrate."""
    ceiling = float(budget.get("max", float("inf"))) * BUDGET_TOLERANCE
    return any(float(g.get("price", 0)) <= ceiling for g in gifts)


def _grossly_over_budget(gifts: list, budget: dict) -> bool:
    """Every gift is FAR beyond budget (> GROSS_BUDGET_MULT x ceiling). A gift you can't afford
    isn't thoughtful no matter how well it fits, so this hard-gates the score like a fit failure
    — otherwise a great-fitting but wildly over-budget gift (e.g. a $900 body against a $60 cap)
    only loses the 0.2 budget weight and still clears 0.4. Gross-only (1.5x) so a mild overage
    stays a soft ding via budget_ok, not a hard fail."""
    if not gifts:
        return False
    ceiling = float(budget.get("max", float("inf"))) * GROSS_BUDGET_MULT
    return all(float(g.get("price", 0)) > ceiling for g in gifts)


def _compute_thoughtfulness(checks: dict, gross_over_budget: bool = False) -> float:
    raw = round(sum(CHECK_WEIGHTS[k] for k, passed in checks.items() if passed), 3)
    # A bad fit OR a wildly-unaffordable gift can never clear the DoD's <0.4 bar.
    if not checks.get("fit_ok") or gross_over_budget:
        return min(raw, FIT_FAIL_CEILING)
    return raw


def judge(persona, proposal, reaction, playbook_len: int = 0) -> Score:
    already_has = persona.hidden_truth.get("already_has", [])
    gifts = proposal.gifts  # never proposal.strategy_tag/agent_id — see module docstring
    user = (f"TRUE PREFERENCES: {persona.hidden_truth}\n"
            f"GIFTS: {gifts}\n"
            f"REACTION: {reaction.verdict} — {reaction.quote}")
    out = call_llm("judge", JUDGE_SYSTEM, user, ctx={"playbook_len": playbook_len},
                   temperature=0.2)

    checks = {
        "fit_ok": bool(out.get("fit_ok", False)) and not _duplicate_veto(gifts, already_has),
        "surprise_ok": bool(out.get("surprise_ok", False)),
        "effort_ok": bool(out.get("effort_ok", False)),
        "budget_ok": _budget_ok(gifts, persona.budget),
    }
    return Score(proposal_id=proposal.proposal_id,
                 dims={k: (1.0 if v else 0.0) for k, v in checks.items()},
                 thoughtfulness=_compute_thoughtfulness(
                     checks, gross_over_budget=_grossly_over_budget(gifts, persona.budget)),
                 rationale=out.get("rationale", ""))


def _category_stats(store) -> dict:
    by_occasion = {}
    for e in store.training_episodes():
        occ = e.get("occasion", "unknown")
        by_occasion.setdefault(occ, []).append(e["score"]["thoughtfulness"])
    return {occ: {"mean": round(sum(vals) / len(vals), 3), "n": len(vals),
                  "stdev": round(pstdev(vals), 3) if len(vals) > 1 else 0.0}
            for occ, vals in by_occasion.items()}


def next_curriculum(store, default_n: int) -> CurriculumBatch:
    stats = _category_stats(store)
    eligible = {occ: s for occ, s in stats.items() if s["n"] >= MIN_SAMPLES_PER_CATEGORY}
    if not eligible:
        # Cold start, or every category is still too thin to trust its mean — round-robin
        # toward whichever occasion we've seen least, instead of chasing one noisy episode.
        under_sampled = sorted(stats.items(), key=lambda kv: kv[1]["n"])
        target = under_sampled[0][0] if under_sampled else "general gift-giving"
        reason = ("cold start" if not stats else
                  f"still building sample size for '{target}' (n={stats[target]['n']})")
        return CurriculumBatch(target, reason, default_n)
    summary = "\n".join(f"- {occ}: mean={s['mean']} n={s['n']} stdev={s['stdev']}"
                        for occ, s in sorted(eligible.items(), key=lambda kv: kv[1]["mean"]))
    out = call_llm("coach", COACH_SYSTEM,
                   f"THOUGHTFULNESS BY OCCASION (>= {MIN_SAMPLES_PER_CATEGORY} samples):\n{summary}",
                   ctx={"n": default_n})
    # Cap synthetic curriculum size — don't over-index on synthetic edge cases the held-out
    # set doesn't represent (spec guidance: keep synthetic modest relative to real cases).
    n_personas = max(1, min(int(out.get("n_personas", default_n)), default_n))
    return CurriculumBatch(out.get("target_weakness", "general"),
                           out.get("rationale", ""), n_personas)
