"""Gandalf Protocol — The Swarm [OWNER: Warren]

Multiple diverse agents that explore different strategies. Each proposes solutions to
simulated scenarios. Winners get distilled into a shared Playbook that all agents read.

Three things make the swarm curve steeper than solo:
  1. A battle-tested gift prompt (ported from Broflo) so proposals are grounded, not generic.
  2. Genuinely disagreeing strategies (a safe↔bold axis + distinct lenses) so the swarm
     covers the strategy space in parallel instead of converging on the obvious pick.
  3. Distillation that TRANSFERS instead of fossilizes: specific delta-bullets, capped +
     deduped, with confidence/provenance so a 1-off lesson can't outrank a proven one.

Contract-faithful: only this file changes. `GiftAgent.propose(profile, playbook, count)`
and `Swarm.distill(top_episodes, playbook)` keep their signatures; gifts keep the locked
{"name","category","price","reasoning"} shape (extra score keys pass through harmlessly).
Refs: ACE (2510.04618) · EDV (2606.24428) · playbookd.
"""
import difflib
import re

from contracts import GiftProposal
from llm import call_llm

# ─────────────────────────────────────────────────────────────────────────────
# TODO(Warren) #1 — the gift agent prompt (ported from Broflo's build_system_prompt)
# ─────────────────────────────────────────────────────────────────────────────
# Broflo's prompt is battle-tested. We port its bones — persona framing, HARD RULES,
# surprise axis, sparse-dossier strategy — and adapt the OUTPUT shape to the Gandalf
# contract (name/category/price/reasoning). We additionally ask for delight_score /
# novelty_score so the deterministic re-rank has real signal in REAL mode; they default
# to neutral in MOCK where the stub doesn't emit them.

# Lifted from Broflo services/ai/app/prompt.py (SURPRISE_INSTRUCTIONS).
SURPRISE_INSTRUCTIONS = {
    "safe": ("Prioritize crowd-pleasing, proven choices that are likely to delight. "
             "Weight reliability over surprise."),
    "bold": ("Prioritize unique, unexpected, experience-based gifts. Be adventurous. "
             "Weight novelty and surprise over safe choices."),
}

# Lifted from Broflo prompt.py (BLOCKLIST_CATEGORIES) — used both in-prompt and as a
# deterministic post-filter so blocked content can never reach the judge.
BLOCKLIST_CATEGORIES = [
    "weapons", "firearms", "ammunition", "adult content", "pornography",
    "prescription medication", "controlled substances", "live animals",
]


def build_gift_system(strategy_tag: str, count: int, playbook_block: str) -> str:
    """Assemble the gift-agent system prompt for one strategy.

    Built as a function (not str.format) to avoid escaping the JSON braces, and so the
    strategy persona + surprise instruction + learned playbook all compose cleanly.
    """
    spec = STRATEGY_SPECS[strategy_tag]
    surprise = SURPRISE_INSTRUCTIONS[spec["surprise"]]
    return f"""You are Broflo's Gift Brain — a brilliant, discreet gift concierge. You produce \
personalized gift suggestions that feel like they came from someone who truly knows the \
recipient. You are NOT a generic gift-list generator.

STRATEGY — {strategy_tag}: {spec['lens']}
{surprise}
Apply your strategy's lens hard. Do NOT converge on the safe obvious pick — a swarm only \
learns if different strategies explore different corners of the idea space.

HARD RULES (never violate — a violation is a failed proposal):
1. NEVER propose anything the recipient ALREADY OWNS (see already_owns), or anything that \
closely resembles it.
2. HONOR every CONSTRAINT. Dietary and allergy constraints are safety-critical — never \
propose a gift that contains or relates to a listed allergen or violates a diet. Values \
constraints (e.g. "hates clutter", "minimalist") are hard too.
3. NEVER assume gender, religion, or cultural background unless explicitly stated.
4. NEVER suggest: {", ".join(BLOCKLIST_CATEGORIES)}.
5. Tie EVERY reasoning to a SPECIFIC profile detail (a named interest, its evidence, or the \
gift_style_hint) — not a generic category. Vague reasoning is a weak proposal.

SPARSE DOSSIER STRATEGY: if the profile is thin, do NOT fall back to generic gifts (candles, \
gift cards, spa sets, jewelry). Infer lifestyle from the interests and personality hint, and \
lean toward distinctive, memorable, experience/artisan picks that feel like a specific find.

LEARNED LESSONS (apply these — they are what the swarm has learned so far):
{playbook_block}

Return EXACTLY {count} gifts as JSON, no prose, no markdown:
{{"gifts":[{{"name": str, "category": str, "price": number,
  "reasoning": str tied to a specific profile detail,
  "delight_score": number 0..1, "novelty_score": number 0..1}}]}}"""


# ─────────────────────────────────────────────────────────────────────────────
# TODO(Warren) #2 — diversity strategies that genuinely disagree
# ─────────────────────────────────────────────────────────────────────────────
# Each tag maps onto Broflo's proven safe↔bold surprise axis PLUS a distinct lens so two
# agents propose genuinely different gifts for the same person. safe-practical/sentimental
# lean "safe"; bold-experiences/novelty-delight lean "bold".
STRATEGY_SPECS = {
    "safe-practical": {
        "surprise": "safe",
        "lens": "Useful, low-risk, definitely-used gifts. Solve a real friction in their "
                "daily life or upgrade something they already rely on.",
    },
    "bold-experiences": {
        "surprise": "bold",
        "lens": "Memorable experiences over objects — classes, trips, events, once-in-a-"
                "while adventures tied to a passion. Nothing that sits on a shelf.",
    },
    "sentimental": {
        "surprise": "safe",
        "lens": "Personal meaning and shared history — gifts that reference a specific "
                "relationship, story, or milestone. Emotion over utility.",
    },
    "novelty-delight": {
        "surprise": "bold",
        "lens": "Surprising, hard-to-find, 'where did you even find this?' delight. Weird-"
                "but-perfect artisan or niche finds that match a specific interest.",
    },
}
# Kept as a list so Swarm's STRATEGIES[:size] slicing is unchanged.
STRATEGIES = list(STRATEGY_SPECS)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic post-filter — a constraint / already_owns / blocklist violation must
# never reach the judge. Ported from Broflo postprocess.py, but zero-dep: stdlib difflib
# replaces Levenshtein so the never-cut trio never depends on an optional install.
# ─────────────────────────────────────────────────────────────────────────────
def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower().strip())


def _content_tokens(s: str) -> set[str]:
    """Significant word tokens (drop ≤2-char noise so 'a'/'of' can't force a subset match)."""
    return {t for t in re.findall(r"[a-z0-9]+", _norm(s)) if len(t) > 2}


def _fuzzy_eq(a: str, b: str) -> bool:
    """True if two item names are the same thing, catching word-insertion which BOTH a raw
    char-ratio and Broflo's Levenshtein-<3 miss ("GPS watch" vs "GPS running watch" is edit
    distance 8). Rule: token-subset (one item's content words ⊆ the other's) OR high difflib
    ratio. Token-subset catches insertion without the "book"⊆"cookbook" false positive a raw
    substring check would cause. Zero-dep (stdlib difflib)."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    ta, tb = _content_tokens(a), _content_tokens(b)
    if ta and tb and (ta <= tb or tb <= ta):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.82


def _values(profile: dict, key: str) -> list[str]:
    """Pull the string values out of a profile list-of-dicts field (interests/
    already_owns/constraints use {"value": ...}); tolerate bare strings too."""
    out = []
    for item in profile.get(key, []) or []:
        out.append(item.get("value", "") if isinstance(item, dict) else str(item))
    return [v for v in out if v]


def _gift_text(g: dict) -> str:
    return f"{g.get('name','')} {g.get('category','')} {g.get('reasoning','')}".lower()


def _filter_already_owns(gifts: list[dict], profile: dict) -> list[dict]:
    owns = _values(profile, "already_owns")
    if not owns:
        return gifts
    return [g for g in gifts if not any(_fuzzy_eq(g.get("name", ""), o) or
                                        _fuzzy_eq(g.get("category", ""), o) for o in owns)]


# NOTE: constraints (dietary/allergy/values) are enforced in the PROMPT HARD RULES, not by
# a deterministic post-filter — the semantics are inverted for token matching. A gift that
# RESPECTS "vegan" literally says "vegan" ("vegan-catered retreat"); a gift that VIOLATES it
# ("leather wallet") never contains the word. Token-dropping on the constraint value would
# remove the compliant gifts and keep the violations. Broflo makes the same choice (allergens
# in-prompt, not in postprocess); the judge also tanks 'fit' on any violation as a backstop.


def _filter_blocklist(gifts: list[dict]) -> list[dict]:
    return [g for g in gifts if not any(cat in _gift_text(g) for cat in BLOCKLIST_CATEGORIES)]


def _rerank(gifts: list[dict], bold: bool) -> list[dict]:
    """Broflo's composite: bold weights novelty, safe weights delight. Missing scores
    default to neutral so MOCK (no scores) keeps a stable order."""
    def score(g):
        d = float(g.get("delight_score", 0.5) or 0.5)
        n = float(g.get("novelty_score", 0.5) or 0.5)
        return (d * 0.3 + n * 0.7) if bold else (d * 0.6 + n * 0.4)
    return sorted(gifts, key=score, reverse=True)


class GiftAgent:
    def __init__(self, strategy_tag: str):
        self.agent_id = f"agent-{strategy_tag}"
        self.strategy_tag = strategy_tag

    def propose(self, profile: dict, playbook, count: int) -> GiftProposal:
        # Pass the profile so the playbook returns the top-k RELEVANT lessons for THIS
        # recipient (RAG seam in retrieval.py) rather than dumping the whole growing list
        # — avoids "context collapse" as the playbook grows. [Rig seam]
        sys = build_gift_system(self.strategy_tag, count, playbook.as_prompt(profile))
        user = f"RECIPIENT PROFILE:\n{profile}\n\nPropose {count} gifts."
        out = call_llm("gift_agent", sys, user, ctx={"count": count})
        gifts = out.get("gifts", [])

        # Deterministic post-filter: a hard violation never reaches the judge. Only filters
        # that are SAFE to token-match — already_owns dedup + blocklist. Constraints are
        # prompt-enforced (see note above). Order mirrors Broflo: dedup → blocklist → re-rank.
        gifts = _filter_already_owns(gifts, profile)
        gifts = _filter_blocklist(gifts)
        bold = STRATEGY_SPECS[self.strategy_tag]["surprise"] == "bold"
        gifts = _rerank(gifts, bold)

        return GiftProposal(agent_id=self.agent_id, strategy_tag=self.strategy_tag,
                            gifts=gifts)


# ─────────────────────────────────────────────────────────────────────────────
# TODO(Warren) #3 — distillation that TRANSFERS, not fossilizes
# ─────────────────────────────────────────────────────────────────────────────
DISTILL_SYSTEM = """You distill lessons from the highest-scoring gift episodes this round so \
that ANY agent gives better gifts to SIMILAR recipients next round.

Output 1-3 lessons as itemized DELTAS (small, standalone rules) — never a rewrite of prior \
lessons. Each lesson MUST be:
- SPECIFIC and transferable: "match trail-running gear to the stated hobby, not generic \
fitness gear" — NOT a platitude like "be more thoughtful" or "personalize the gift".
- A PATTERN, not a memorized answer: capture WHY the winning gift landed (which profile \
signal it matched), not the exact product name. Do not just name the top gift.
- Tied to an observable signal: an interest, its evidence, a gift_style_hint, or the \
recipient's reaction.

Output JSON: {"lessons":[{"text": "<one specific, transferable rule>"}]}"""

PLAYBOOK_CAP = 30          # unbounded playbooks fossilize bad habits (TerminalBench result)
WIN_THRESHOLD = 0.6        # an episode "won" if the judge scored it >= this
SUPERSEDE_RATIO = 0.80     # a new lesson this similar to an old one REFINES it, not appends

# Platitudes the distiller sometimes emits; a lesson that is ONLY these is noise, not signal.
_PLATITUDE_RE = re.compile(
    r"^(be |being |try to |remember to )?(more )?"
    r"(thoughtful|personal|personalized|considerate|creative|generic|specific|attentive)\b",
    re.I,
)


def _is_specific(text: str) -> bool:
    """Reject platitudes and one-word fluff; keep transferable specifics. Deliberately
    lenient (short-circuits only obvious noise) so real lessons are never dropped."""
    t = _norm(text)
    if len(t) < 20:
        return False
    if _PLATITUDE_RE.match(t) and len(t.split()) < 6:
        return False
    return True


def _entry_conf(entry: dict) -> float:
    """Wilson lower bound on wins/trials (playbookd pattern) so a 1-success lesson can't
    outrank a proven one. Neutral 0.5 when a lesson has no provenance yet."""
    wins, trials = entry.get("wins"), entry.get("trials")
    if isinstance(wins, (int, float)) and isinstance(trials, (int, float)) and trials > 0:
        p = wins / trials
        z2 = 3.8416
        denom = 1 + z2 / trials
        centre = p + z2 / (2 * trials)
        margin = 1.96 * ((p * (1 - p) + z2 / (4 * trials)) / trials) ** 0.5
        return (centre - margin) / denom
    return 0.5


class Swarm:
    def __init__(self, size: int = 4, solo: bool = False):
        # solo=True is the ablation baseline: a single agent, no distillation.
        tags = STRATEGIES[:1] if solo else STRATEGIES[:size]
        self.agents = [GiftAgent(t) for t in tags]
        self.solo = solo

    def distill(self, top_episodes: list[dict], playbook,
                bottom_episodes: list | None = None, held_out_eval=None):
        """The 'learn together' moment: fold winning PATTERNS into the shared playbook.
        Skipped in solo mode so the ablation shows the difference.

        EDV (2606.24428): the distiller reads Jasper's SCORES, not its own opinion of its
        work — decoupling distiller from executor avoids the Self-Confirmation Trap. Our
        Warren-distills / Jasper-judges split already enforces this; we keep it by ranking
        episodes purely on the judge's thoughtfulness.

        Round-2 Phase 3 (ACE 2510.04618) — behind config.ACE_CURATION (default OFF, so this
        signature stays 100% backward-compatible and the legacy path below is byte-identical):
        when ON, run an explicit Reflector→Curator with incremental DELTAS + a per-lesson
        held-out gate instead of the monolithic re-distill. `bottom_episodes` feeds GEPA-style
        failure reflection ("avoid" lessons); `held_out_eval(playbook)->float` is the pluggable
        Voyager/RSEA gate (None => pass-through no-op; the REAL gate only runs deployed where a
        live LLM judge exists). ACE's thesis: monolithic rewrites cause "context collapse" —
        the playbook erodes into shorter, vaguer text round over round; localized deltas +
        confidence-gated prune preserve the accumulated detail. See _distill_ace()."""
        if self.solo or not top_episodes:
            return

        import config
        if config.ACE_CURATION:
            return self._distill_ace(top_episodes, playbook, bottom_episodes, held_out_eval)

        # ── Legacy path (config.ACE_CURATION OFF, the default) — UNCHANGED from Round-1 ──
        # Ground the distiller in WHY each winner landed: strategy, the gift + its reasoning,
        # the recipient's reaction, and the judge's per-dimension rationale — not just a
        # gift list. Specific inputs → specific, transferable lessons.
        lines = []
        for e in top_episodes:
            gifts = e.get("proposal", {}).get("gifts", [])
            g = gifts[0] if gifts else {}
            score = e.get("score", {})
            react = e.get("reaction", {})
            lines.append(
                f"- strategy={e.get('strategy_tag','?')} "
                f"gift={g.get('name','?')} ({g.get('category','?')}) "
                f"why={g.get('reasoning','')!r} "
                f"reaction={react.get('verdict','?')} "
                f"thoughtfulness={score.get('thoughtfulness','?')} "
                f"judge={score.get('rationale','')!r}")
        out = call_llm("distiller", DISTILL_SYSTEM, "TOP EPISODES THIS ROUND:\n" + "\n".join(lines),
                       ctx={"playbook_len": playbook.version})

        # Provenance for confidence ranking: how many of the source episodes actually won.
        source_ids = [e["episode_id"] for e in top_episodes if e.get("episode_id")]
        trials = len(top_episodes)
        wins = sum(1 for e in top_episodes
                   if float(e.get("score", {}).get("thoughtfulness", 0.0)) >= WIN_THRESHOLD)

        for lesson in out.get("lessons", []):
            text = (lesson.get("text") or "").strip()
            if not _is_specific(text):
                continue  # drop platitudes rather than fossilize them
            self._merge_or_add(playbook, text, source_ids, wins, trials)

        self._prune(playbook)

    # ── playbook maintenance: dedup/supersede + confidence + cap ──
    def _merge_or_add(self, playbook, text, source_ids, wins, trials):
        """Add a delta bullet — but if it REFINES an existing lesson (near-duplicate),
        supersede in place instead of appending two near-copies. Superseding still counts
        as a learning event (version bump + accumulated wins/trials): reinforcement raises
        confidence, and it keeps the round-over-round learning signal intact for the
        dashboard. Conflict resolution by supersede avoids the classic "I like Python day-1
        vs I only use Rust day-30" coin-flip failure."""
        best, best_ratio = None, 0.0
        nt = _norm(text)
        for e in playbook.entries:
            r = difflib.SequenceMatcher(None, nt, _norm(e.get("text", ""))).ratio()
            if r > best_ratio:
                best, best_ratio = e, r

        if best is not None and best_ratio >= SUPERSEDE_RATIO:
            # Refine: keep the newer phrasing, merge provenance, accumulate confidence.
            best["text"] = text
            merged = set(best.get("source_episode_ids", [])) | set(source_ids)
            best["source_episode_ids"] = list(merged)
            best["wins"] = best.get("wins", 0) + wins
            best["trials"] = best.get("trials", 0) + trials
            playbook.version += 1  # reinforcement is a learning event; keeps the curve moving
            return

        # Net-new lesson: use the contract's add() (handles version + timestamp), then
        # attach confidence provenance the retrieval ranker reads.
        playbook.add(text, source_ids)
        entry = playbook.entries[-1]
        entry["wins"], entry["trials"] = wins, trials

    def _prune(self, playbook):
        """Cap the playbook (~30) and drop the lowest-confidence entries so bad habits can't
        fossilize. retrieval.py already returns top-k per persona; pairing that with a hard
        write-cap is what took one team ~30%→~90% on TerminalBench. version is monotonic
        (a learning counter), so pruning entries never rewinds the learning curve."""
        if len(playbook.entries) <= PLAYBOOK_CAP:
            return
        # Keep highest confidence; tie-break toward newer (later index) to avoid thrashing.
        ranked = sorted(enumerate(playbook.entries),
                        key=lambda ie: (_entry_conf(ie[1]), ie[0]), reverse=True)
        playbook.entries = [e for _, e in ranked[:PLAYBOOK_CAP]]

    # ═════════════════════════════════════════════════════════════════════════════
    # Round-2 Phase 3 — ACE delta-curation (config.ACE_CURATION). Everything below is
    # NEW and only reachable when the flag is ON; the legacy distill path never enters
    # here, so flag-OFF behavior is untouched. Kept BELOW the distillation region so the
    # parallel Phase-1 (GiftAgent.propose) merge stays conflict-free.
    #
    # Grounding — ACE (Agentic Context Engineering, arXiv 2510.04618): monolithic
    # "rewrite the whole memory each step" distillation suffers CONTEXT COLLAPSE — the
    # curated context degrades into shorter, blander summaries that shed the very
    # specifics that made lessons transferable. ACE's fix is a Reflector→Curator split
    # that emits SMALL, LOCALIZED deltas (add / refine-in-place / remove) against a
    # structured, itemized playbook, plus dedup + confidence-gated bounded size. We
    # already had delta-add + supersede + Wilson prune; Phase 3 completes the loop with
    # GEPA-style failure reflection, an explicit remove delta, and a per-lesson gate.
    # ═════════════════════════════════════════════════════════════════════════════

    # Reflector prompt for the FAILURE half (GEPA 2507.19457): read the LOWEST-scoring
    # episodes and extract WHY they fell short as "avoid" rules — the negative signal a
    # winners-only distiller never sees.
    _REFLECT_FAIL_SYSTEM = """You diagnose why the LOWEST-scoring gift episodes this round \
FELL SHORT, so that ANY agent AVOIDS the same mistake for SIMILAR recipients next round.

Output 1-3 lessons as itemized DELTAS (small, standalone AVOID rules) — never a rewrite of \
prior lessons. Each lesson MUST be:
- An AVOID rule naming the FAILURE MODE: "avoid generic gift cards when the profile names a \
concrete hobby — they read as low-effort" — NOT a platitude like "be more thoughtful".
- Tied to an observable signal: the mismatch between the gift and a stated interest/constraint, \
the recipient's negative reaction, or the judge's rationale for the low score.
- A PATTERN, not the one bad product: capture WHY it flopped, not just the item name.

Output JSON: {"lessons":[{"text": "<one specific, transferable avoid rule>"}]}"""

    # Remove-delta thresholds: a lesson with enough trials to be sure AND zero wins has a
    # Wilson lower bound ~0 → it is a PROVEN loser, not merely new. Drop it (ACE remove delta)
    # rather than let it linger and mislead. Conservative so a young lesson is never culled.
    _REMOVE_MIN_TRIALS = 3
    _REMOVE_CONF_FLOOR = 0.15

    def _distill_ace(self, top_episodes, playbook, bottom_episodes, held_out_eval):
        """ACE path: Reflector (winners→'do', failures→'avoid') → Curator (add/refine/remove,
        each behind the per-lesson gate) → confidence-gated prune. Returns None."""
        candidates = []

        # Reflector #1 — winners → "do" lessons (same inputs/prompt as the legacy distiller,
        # so the "do" half of an ACE run matches the Round-1 lessons).
        do_lessons, top_sids, top_wins, top_trials = self._reflect(top_episodes, playbook, "win")
        candidates += [{"text": t, "polarity": "do", "source_ids": top_sids,
                        "wins": top_wins, "trials": top_trials} for t in do_lessons]

        # Reflector #2 — failures → "avoid" lessons (GEPA). Only when bottoms are supplied;
        # keeping this optional preserves the frozen distill(top, playbook) call.
        if bottom_episodes:
            avoid_lessons, bot_sids, _, bot_trials = self._reflect(bottom_episodes, playbook, "fail")
            candidates += [{"text": t, "polarity": "avoid", "source_ids": bot_sids,
                            "wins": 0, "trials": bot_trials} for t in avoid_lessons]

        # Curator — apply each candidate as an incremental delta, gated per-lesson.
        for cand in candidates:
            if not _is_specific(cand["text"]):
                continue  # drop platitudes rather than fossilize them
            self._curate_one(playbook, cand, held_out_eval)

        # Remove-delta pass: cull proven-losing lessons, then bound size (unchanged _prune).
        self._curate_removals(playbook)
        self._prune(playbook)

    def _reflect(self, episodes, playbook, mode):
        """Build the grounded episode digest and call the distiller for one reflection mode
        ('win' → DISTILL_SYSTEM, 'fail' → _REFLECT_FAIL_SYSTEM). Returns
        (lesson_texts, source_ids, wins, trials)."""
        lines = []
        for e in episodes:
            gifts = e.get("proposal", {}).get("gifts", [])
            g = gifts[0] if gifts else {}
            score = e.get("score", {})
            react = e.get("reaction", {})
            lines.append(
                f"- strategy={e.get('strategy_tag','?')} "
                f"gift={g.get('name','?')} ({g.get('category','?')}) "
                f"why={g.get('reasoning','')!r} "
                f"reaction={react.get('verdict','?')} "
                f"thoughtfulness={score.get('thoughtfulness','?')} "
                f"judge={score.get('rationale','')!r}")
        source_ids = [e["episode_id"] for e in episodes if e.get("episode_id")]
        trials = len(episodes)
        if mode == "win":
            system, header = DISTILL_SYSTEM, "TOP EPISODES THIS ROUND:\n"
            wins = sum(1 for e in episodes
                       if float(e.get("score", {}).get("thoughtfulness", 0.0)) >= WIN_THRESHOLD)
        else:
            system, header = self._REFLECT_FAIL_SYSTEM, "LOWEST-SCORING (FAILED) EPISODES THIS ROUND:\n"
            wins = 0  # failures don't reinforce; provenance still tracked for audit
        out = call_llm("distiller", system, header + "\n".join(lines),
                       ctx={"playbook_len": playbook.version})
        texts = [(l.get("text") or "").strip() for l in out.get("lessons", [])]
        return [t for t in texts if t], source_ids, wins, trials

    def _curate_one(self, playbook, cand, held_out_eval):
        """Apply ONE candidate as an add / refine-in-place / remove-then-add delta, admitting
        it only through the per-lesson gate. Returns 'applied' or 'rejected'.

        Delta decision (vs the closest existing lesson by difflib ratio):
          • near-dup (ratio ≥ SUPERSEDE_RATIO), SAME polarity → refine-in-place (accumulate).
          • near-dup, OPPOSITE polarity → CONTRADICTION: remove the stale lesson, add the new
            one (an 'avoid' rule negates a near-identical 'do' rule, and vice-versa). This is
            the ACE remove delta that plain supersede can't express.
          • otherwise → net-new append."""
        text = cand["text"]
        polarity = cand.get("polarity", "do")
        sids, wins, trials = cand.get("source_ids", []), cand.get("wins", 0), cand.get("trials", 0)
        i, ratio = self._best_match(playbook, text)

        if i >= 0 and ratio >= SUPERSEDE_RATIO:
            target_pol = playbook.entries[i].get("polarity", "do")
            if polarity != target_pol:
                def op():
                    playbook.remove(i)
                    self._append(playbook, text, sids, wins, trials, polarity)
            else:
                def op():
                    playbook.refine_in_place(i, text, sids, wins, trials)
                    playbook.entries[i]["polarity"] = polarity
        else:
            def op():
                self._append(playbook, text, sids, wins, trials, polarity)

        # Per-lesson gate (Voyager/RSEA). None => pass-through no-op (MOCK / default): apply
        # unconditionally and deterministically. A real held_out_eval (deployed only, needs the
        # live LLM judge) applies-measure-rollback so a lesson that LOWERS held-out is rejected.
        if held_out_eval is None:
            op()
            return "applied"
        import copy
        before = held_out_eval(playbook)
        snap_entries = copy.deepcopy(playbook.entries)
        snap_version = playbook.version
        op()
        if held_out_eval(playbook) < before:
            playbook.entries = snap_entries      # reject: this lesson didn't help held-out
            playbook.version = snap_version
            return "rejected"
        return "applied"

    def _curate_removals(self, playbook):
        """Remove-delta pass: drop PROVEN-losing 'do' recommendations (enough trials, zero wins
        → Wilson bound below the floor) — a "do X" rule tried repeatedly that never rode a
        winning episode is a bad recommendation. 'avoid' lessons are EXEMPT: they encode
        negative knowledge and have wins==0 by construction, so culling them on zero-wins would
        erase every failure lesson. Contradiction removals already happened in _curate_one;
        this catches do-rules that simply never win. version is monotonic, so removal never
        rewinds the learning curve."""
        playbook.remove(
            lambda e: e.get("polarity", "do") != "avoid"
            and e.get("trials", 0) >= self._REMOVE_MIN_TRIALS
            and e.get("wins", 0) == 0
            and _entry_conf(e) < self._REMOVE_CONF_FLOOR)

    @staticmethod
    def _best_match(playbook, text):
        """Index + difflib ratio of the existing entry most similar to `text` (or (-1, 0.0))."""
        best_i, best_ratio, nt = -1, 0.0, _norm(text)
        for idx, e in enumerate(playbook.entries):
            r = difflib.SequenceMatcher(None, nt, _norm(e.get("text", ""))).ratio()
            if r > best_ratio:
                best_i, best_ratio = idx, r
        return best_i, best_ratio

    @staticmethod
    def _append(playbook, text, sids, wins, trials, polarity):
        """Net-new lesson via the contract's add() (handles version + timestamp), then attach
        the confidence provenance + polarity the retrieval ranker / curator read. Mirrors the
        legacy net-new branch in _merge_or_add, plus the ACE polarity tag."""
        playbook.add(text, sids)
        e = playbook.entries[-1]
        e["wins"], e["trials"], e["polarity"] = wins, trials, polarity
