"""Gandalf Protocol — Tests for The Signal [OWNER: Jasper]

Run from repo root: `python test_signal.py`

Two tiers:
  - Deterministic tests (always run, no API key/cost): the duplicate veto, the budget
    check, the aggregate weighting, and the coach's sample-size gating are all plain
    Python, tested directly against constructed inputs.
  - Live calibration (skips automatically in MOCK mode): a hand-labeled gold set of 8
    scenarios (good / bad / obvious / surprising / constraint-violating / reward-hacking),
    scored against the real judge(), with agreement tracked via Cohen's kappa per
    EMAIL-JASPER-AMENDED.md's guidance (~0.55 floor, not raw % agreement). MOCK mode's
    fake judge response ignores JUDGE_SYSTEM entirely, so it can't prove the prompt is
    calibrated — this needs a real ANTHROPIC_API_KEY to mean anything.
"""
import config
from contracts import Persona, GiftProposal, Reaction
from agents.signal import (
    judge, next_curriculum, _compute_thoughtfulness, _category_stats, _duplicate_veto,
    _budget_ok, MIN_SAMPLES_PER_CATEGORY,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def cohens_kappa(gold: list, pred: list) -> float:
    """Two-rater agreement on a binary label, correcting for chance agreement — the metric
    EMAIL-JASPER-AMENDED.md asks for instead of raw % (raw % looks fine even when both
    raters just say the same thing most of the time by base rate alone)."""
    n = len(gold)
    po = sum(g == p for g, p in zip(gold, pred)) / n
    g_true, p_true = sum(gold) / n, sum(pred) / n
    pe = g_true * p_true + (1 - g_true) * (1 - p_true)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


class FakeStore:
    def __init__(self, episodes):
        self._episodes = episodes

    def training_episodes(self):
        return self._episodes


# ── Deterministic: the duplicate veto (fit_ok can't be talked past) ──

def test_duplicate_veto_catches_fuzzy_match():
    already_has = [{"value": "GPS running watch"}]
    gifts = [{"name": "GPS Running Watch", "price": 120}]  # case-different, same item
    check("fuzzy-duplicate is vetoed", _duplicate_veto(gifts, already_has))


def test_duplicate_veto_lets_distinct_gift_through():
    already_has = [{"value": "GPS running watch"}]
    gifts = [{"name": "trail running vest", "price": 60}]
    check("distinct gift is not vetoed", not _duplicate_veto(gifts, already_has))


def test_duplicate_veto_fires_on_any_gift_not_only_all():
    # A mixed proposal: one duplicate + one genuinely good gift. Violet's world stamps this
    # owns_it on the ANY-match, so the fit veto must fire too (Warren's cross-file note).
    already_has = [{"value": "GPS running watch"}]
    gifts = [{"name": "GPS running watch", "price": 120},
             {"name": "trail running vest", "price": 60}]
    check("veto fires when ANY gift is a dupe (not only when all are)",
          _duplicate_veto(gifts, already_has))


# ── Deterministic: budget is arithmetic, not judgment ──

def test_budget_ok_within_tolerance():
    check("gift 5% over budget still passes (10% tolerance)",
          _budget_ok([{"name": "x", "price": 105}], {"max": 100}))


def test_budget_ok_rejects_large_overage():
    check("gift 50% over budget fails",
          not _budget_ok([{"name": "x", "price": 150}], {"max": 100}))


# ── Deterministic: aggregate weighting ──

def test_aggregate_all_pass_is_high():
    t = _compute_thoughtfulness({"fit_ok": True, "surprise_ok": True, "effort_ok": True,
                                 "budget_ok": True})
    check("all checks passing scores 1.0", t == 1.0, f"got {t}")


def test_aggregate_fit_fail_stays_below_dod_bar():
    # Worst case for the bug Warren caught: fit_ok=False but the other three all pass. Pure
    # weighting put this at 0.6 — above issue #5's "bad gift < 0.4" DoD. Must now be gated under.
    t = _compute_thoughtfulness({"fit_ok": False, "surprise_ok": True, "effort_ok": True,
                                 "budget_ok": True})
    check("failed fit_ok keeps aggregate below the DoD's 0.4 bar", t < 0.4, f"got {t}")


# ── Deterministic: coach sample-size gating ──

def test_category_stats_low_sample_excluded_from_eligible():
    store = FakeStore([
        {"occasion": "birthday", "score": {"thoughtfulness": 0.9}},
        {"occasion": "birthday", "score": {"thoughtfulness": 0.85}},
        {"occasion": "birthday", "score": {"thoughtfulness": 0.88}},
        {"occasion": "anniversary", "score": {"thoughtfulness": 0.1}},
    ])
    stats = _category_stats(store)
    eligible = {occ: s for occ, s in stats.items() if s["n"] >= MIN_SAMPLES_PER_CATEGORY}
    check("low-sample category excluded from eligible set", "anniversary" not in eligible,
          f"eligible={eligible}")
    check("well-sampled category included", "birthday" in eligible, f"eligible={eligible}")


def test_coach_round_robins_when_all_categories_undersampled():
    store = FakeStore([
        {"occasion": "birthday", "score": {"thoughtfulness": 0.9}},
        {"occasion": "birthday", "score": {"thoughtfulness": 0.85}},
        {"occasion": "anniversary", "score": {"thoughtfulness": 0.1}},
    ])
    batch = next_curriculum(store, default_n=6)
    check("targets least-seen category instead of chasing a noisy low score",
          batch.target_weakness == "anniversary", f"got {batch.target_weakness}")


def test_coach_caps_n_personas_at_default(monkeypatch=None):
    # MOCK coach ignores the summary content but always returns n_personas=ctx["n"], so this
    # exercises the clamp path directly against next_curriculum's own cap logic.
    store = FakeStore([
        {"occasion": "birthday", "score": {"thoughtfulness": 0.9}},
        {"occasion": "birthday", "score": {"thoughtfulness": 0.85}},
        {"occasion": "birthday", "score": {"thoughtfulness": 0.8}},
    ])
    batch = next_curriculum(store, default_n=6)
    check("n_personas never exceeds default_n", batch.n_personas <= 6, f"got {batch.n_personas}")


# ── Live: gold-set calibration against the real judge, scored via Cohen's kappa ──

def _persona(would_love, already_has, budget_max=150):
    return Persona(
        profile={"interests": [], "already_owns": [], "constraints": [], "personality": {}},
        hidden_truth={"would_love": would_love, "would_return": [], "already_has": already_has},
        occasion="birthday", budget={"min": 20, "max": budget_max},
    )


def _proposal(gifts):
    return GiftProposal(agent_id="gold", strategy_tag="gold", gifts=gifts)


def _reaction(proposal, persona, verdict, quote):
    return Reaction(proposal_id=proposal.proposal_id, persona_id=persona.persona_id,
                    verdict=verdict, quote=quote)


# Each case: (persona, proposal, reaction, gold labels for fit_ok/surprise_ok/effort_ok).
# budget_ok is excluded from kappa — it's deterministic, not an LLM judgment, so it's not a
# meaningful test of calibration.
def _gold_set():
    cases = []

    p = _persona(["trail running gear"], [{"value": "GPS running watch"}])
    prop = _proposal([{"name": "GPS Running Watch", "category": "electronics", "price": 120,
                       "reasoning": "they love trail running so a GPS watch is perfect"}])
    react = _reaction(prop, p, "owns_it", "I already have this exact watch")
    cases.append((p, prop, react, {"fit_ok": False, "surprise_ok": False, "effort_ok": False}))

    p = _persona(["trail running gear"], [])
    prop = _proposal([{"name": "$50 gift card", "category": "gift_card", "price": 50,
                       "reasoning": "since they like trail running, something running-related"}])
    react = _reaction(prop, p, "meh", "thanks, I guess")
    cases.append((p, prop, react, {"fit_ok": True, "surprise_ok": False, "effort_ok": False}))

    p = _persona(["vinyl records, especially 70s jazz"], [])
    prop = _proposal([{"name": "first pressing of a Bill Evans LP", "category": "music",
                       "price": 45,
                       "reasoning": "their collection skews 70s jazz vinyl, and this specific "
                                    "pressing fills a known gap based on what they already own"}])
    react = _reaction(prop, p, "delight", "how did you even find this")
    cases.append((p, prop, react, {"fit_ok": True, "surprise_ok": True, "effort_ok": True}))

    p = _persona(["marathon training"], [], budget_max=100)
    prop = _proposal([{"name": "running shoes", "category": "apparel", "price": 90,
                       "reasoning": "they run, so running shoes"}])
    react = _reaction(prop, p, "like", "these are fine")
    cases.append((p, prop, react, {"fit_ok": True, "surprise_ok": False, "effort_ok": False}))

    p = _persona(["baking", "no scented products (fragrance allergy)"], [])
    prop = _proposal([{"name": "scented candle gift set", "category": "home", "price": 35,
                       "reasoning": "cozy gift for someone who likes being home baking"}])
    react = _reaction(prop, p, "return", "I'm allergic, had to return it")
    cases.append((p, prop, react, {"fit_ok": False, "surprise_ok": False, "effort_ok": False}))

    p = _persona(["woodworking"], [])
    prop = _proposal([{"name": "a scarf", "category": "apparel", "price": 25,
                       "reasoning": "everyone likes a nice scarf"}])
    react = _reaction(prop, p, "meh", "oh, thanks")
    cases.append((p, prop, react, {"fit_ok": False, "surprise_ok": False, "effort_ok": False}))

    p = _persona(["specialty coffee, has a home espresso setup"], [])
    prop = _proposal([{"name": "single-origin beans from the roaster they mentioned wanting "
                              "to try", "category": "food", "price": 28,
                       "reasoning": "they specifically mentioned wanting to try this roaster "
                                    "but hadn't ordered from them yet"}])
    react = _reaction(prop, p, "delight", "exactly the beans I wanted to try!")
    cases.append((p, prop, react, {"fit_ok": True, "surprise_ok": True, "effort_ok": True}))

    p = _persona(["photography"], [])
    prop = _proposal([{"name": "wireless earbuds", "category": "electronics", "price": 80,
                       "reasoning": "popular gift, good reviews"}])
    react = _reaction(prop, p, "meh", "sure, I can use these I guess")
    cases.append((p, prop, react, {"fit_ok": False, "surprise_ok": False, "effort_ok": False}))

    return cases


def test_judge_gold_set_calibration_live():
    if config.MOCK:
        print("[SKIP] gold-set calibration — MOCK mode ignores the real prompt, "
              "set ANTHROPIC_API_KEY to actually exercise this")
        return

    gold_set = _gold_set()
    dims = ["fit_ok", "surprise_ok", "effort_ok"]
    gold_by_dim = {d: [] for d in dims}
    pred_by_dim = {d: [] for d in dims}

    for persona, proposal, reaction, expected in gold_set:
        score = judge(persona, proposal, reaction)
        for d in dims:
            gold_by_dim[d].append(1 if expected[d] else 0)
            pred_by_dim[d].append(1 if score.dims.get(d, 0.0) >= 0.5 else 0)

    for d in dims:
        kappa = cohens_kappa(gold_by_dim[d], pred_by_dim[d])
        check(f"judge '{d}' agrees with gold labels (Cohen's kappa >= 0.55)", kappa >= 0.55,
              f"kappa={kappa:.2f}, gold={gold_by_dim[d]}, pred={pred_by_dim[d]}")

    all_gold = [v for d in dims for v in gold_by_dim[d]]
    all_pred = [v for d in dims for v in pred_by_dim[d]]
    overall = cohens_kappa(all_gold, all_pred)
    check("judge overall agreement with gold labels (Cohen's kappa >= 0.55)", overall >= 0.55,
          f"kappa={overall:.2f}")


if __name__ == "__main__":
    test_duplicate_veto_catches_fuzzy_match()
    test_duplicate_veto_lets_distinct_gift_through()
    test_duplicate_veto_fires_on_any_gift_not_only_all()
    test_budget_ok_within_tolerance()
    test_budget_ok_rejects_large_overage()
    test_aggregate_all_pass_is_high()
    test_aggregate_fit_fail_stays_below_dod_bar()
    test_category_stats_low_sample_excluded_from_eligible()
    test_coach_round_robins_when_all_categories_undersampled()
    test_coach_caps_n_personas_at_default()
    test_judge_gold_set_calibration_live()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        raise SystemExit(1)
    print("All tests passed.")
