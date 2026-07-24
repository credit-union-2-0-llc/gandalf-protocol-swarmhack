"""Gandalf Protocol — The World [OWNER: Violet]

Generates simulated scenarios and reacts to agent proposals. The quality of your scenarios
directly determines the quality of the learning.

TODOs:
  1. Build realistic scenario generation for your domain.
  2. Implement honest reaction logic (agents should *feel* the difference between good/bad).
  3. (Stretch) ground scenarios in real user data.
Search "TODO(Violet)".
"""
import json, os, re
from contracts import Persona, Reaction
from llm import call_llm

PERSONA_SYSTEM = """Invent {n} realistic gift RECIPIENTS for category: {target}.
For each, produce a PUBLIC profile with:
  interests, already_owns, constraints, personality,
  hobbies, music_taste, favorite_brands, food_preferences,
  clothing_sizes (top/bottom/shoe), pronouns, allergens, dietary_restrictions,
  tags, wishlist_items
(lifted from a real gifting dossier schema — hobbies/music_taste/favorite_brands/
food_preferences/clothing sizes/pronouns/allergens/dietary_restrictions/tags/wishlist_items
are all real fields real gift-givers would know).
Also produce a PRIVATE hidden_truth (3-5 things they'd genuinely love, 2-3 they'd quietly
return or never want again, what they already have).
Make them specific and human — contradictions, a hard-to-shop-for streak, real constraints
(allergens, dietary, values). Avoid stereotypes: keep the 20-30% synthetic-edge-case cap in
mind — most personas should be realistic and representative, not hardest-case gauntlets.
Output JSON: {{"personas":[{{"profile":{{...}},"hidden_truth":{{...}},
"occasion":"...","budget":{{"min":0,"max":0}}}}]}}"""

REACT_SYSTEM = """You ARE this recipient. Someone gave you the gift below. React honestly
and IN CHARACTER against your true preferences — do NOT be polite for its own sake.
Score against your hidden_truth, not vibes: something in already_has is never a delight,
a constraint violation (allergen/dietary/values) is always a return, sarcasm is allowed
("I'd *love* another water bottle" said about something you already own).
If it misses, say so. Output JSON: {"verdict":"delight|like|meh|owns_it|return","quote":"..."}"""

_STOPWORDS = {"a", "an", "the", "of", "to", "for", "with", "and", "another", "generic"}


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOPWORDS}


def _gift_text(gift) -> str:
    """Flatten a gift to searchable text, tolerating BOTH shapes the gift agent emits:
    a dict {name,category,reasoning,...} or a bare string (a real LLM drift). Assuming
    dict-only here crashed a whole real run with 'str' object has no attribute get."""
    if isinstance(gift, dict):
        return " ".join(str(gift.get(k, "")) for k in ("name", "category", "reasoning"))
    return str(gift)


def _constraint_value(constraint) -> str:
    """Constraints may arrive as dicts {value,type,..} or bare strings; normalize to the value."""
    return (constraint.get("value", "") if isinstance(constraint, dict) else str(constraint))


def _gift_matches(gift, phrase: str) -> bool:
    """Loose token-overlap match between a gift (name/category/reasoning) and a
    hidden_truth phrase — good enough to gate an honest verdict, not exact NLP."""
    gift_toks, phrase_toks = _tokens(_gift_text(gift)), _tokens(phrase)
    if not gift_toks or not phrase_toks:
        return False
    overlap = gift_toks & phrase_toks
    return len(overlap) / len(phrase_toks) >= 0.5


def _violates_constraint(gift, constraint) -> bool:
    """Very rough allergen/dietary/values violation check: does the gift text mention
    something incompatible with the stated constraint value (e.g. vegan -> leather/meat)."""
    bad_terms = {
        "vegan": {"leather", "wool", "down", "meat", "cheese", "honey"},
        "vegetarian": {"meat", "leather"},
        "gluten-free": {"gluten", "wheat", "bread", "beer"},
        "kosher": {"pork", "shellfish", "bacon"},
        "nut allergy": {"nut", "nuts", "peanut", "almond"},
        "shellfish allergy": {"shellfish", "shrimp", "crab", "lobster"},
    }.get(_constraint_value(constraint).lower(), set())
    return bool(_tokens(_gift_text(gift)) & bad_terms)


def _enforce_hidden_truth(persona: Persona, proposal, verdict: str, quote: str) -> tuple[str, str]:
    """Deterministic guardrail over the LLM's reaction so the reward signal stays honest
    (the anti-collusion defense: force it to react against hidden_truth, not vibes).
    Tolerant of malformed gift/constraint shapes so one bad LLM response degrades this
    reaction rather than killing the whole run."""
    hidden = persona.hidden_truth if isinstance(persona.hidden_truth, dict) else {}
    profile = persona.profile if isinstance(persona.profile, dict) else {}
    gifts = proposal.gifts if isinstance(proposal.gifts, list) else [proposal.gifts]
    for gift in gifts:
        for owned in hidden.get("already_has", []):
            if _gift_matches(gift, owned):
                return "owns_it", f"I already have {owned}. {quote}".strip()
        for constraint in profile.get("constraints", []):
            if _violates_constraint(gift, constraint):
                return "return", f"That doesn't work for me ({_constraint_value(constraint)}). {quote}".strip()
        for unwanted in hidden.get("would_return", []):
            if _gift_matches(gift, unwanted):
                return "return", quote or "Not for me, but thanks."
        for loved in hidden.get("would_love", []):
            if _gift_matches(gift, loved) and verdict in ("meh", "return", "owns_it"):
                return "like", quote or "Actually, this is closer to what I wanted."
    return verdict, quote


# Generate in SMALL chunks: the dossier-rich persona JSON is large, so asking for a whole
# round's batch in one call overruns MAX_TOKENS, truncates mid-JSON, fails to parse, and
# call_llm returns {} → generate_personas returns [] → run_round saves ZERO training
# episodes → flat/empty learning curve (silent, no error). Small chunks keep each response
# well under the token cap. Overridable if MAX_TOKENS is raised.
_PERSONA_CHUNK = int(os.environ.get("PERSONA_GEN_CHUNK", "2"))


def _persona_from_raw(p: dict) -> Persona | None:
    """Build a Persona from one raw dict, tolerating partial/odd output. None if unusable
    (missing the profile/hidden_truth split) so a malformed entry is skipped, not crashed on."""
    if not isinstance(p, dict):
        return None
    profile, hidden = p.get("profile"), p.get("hidden_truth")
    if not isinstance(profile, dict) or not isinstance(hidden, dict):
        return None
    return Persona(profile=profile, hidden_truth=hidden,
                   occasion=p.get("occasion", "birthday"),
                   budget=p.get("budget", {"min": 20, "max": 100}))


def _fallback_personas(target: str, n: int) -> list[Persona]:
    """Last-resort TRAINING personas if generation yields nothing. Deliberately generic and
    DISTINCT from the held-out seed set (load_seed_personas) so training never contaminates
    the validation set, while guaranteeing run_round always has data to learn from — a
    degraded-but-nonempty training set beats a flat curve. Varied enough to give the judge
    some gradient. Only fires when real generation fully fails."""
    templates = [
        {"style": "practical", "loves": [f"a well-chosen {target} essential"],
         "returns": ["a generic gift card"], "occasion": "birthday"},
        {"style": "experiences", "loves": [f"an experience related to {target}"],
         "returns": ["another mug"], "occasion": "holiday"},
        {"style": "sentimental", "loves": [f"a thoughtful, personal take on {target}"],
         "returns": ["anything obviously last-minute"], "occasion": "anniversary"},
    ]
    out = []
    for i in range(n):
        t = templates[i % len(templates)]
        out.append(Persona(
            profile={"interests": [{"value": target, "confidence": 0.6, "polarity": "loves"}],
                     "already_owns": [], "constraints": [],
                     "personality": {"gift_style_hint": t["style"], "confidence": 0.5},
                     "tags": ["fallback-generated", f"target:{target}"]},
            hidden_truth={"would_love": t["loves"], "would_return": t["returns"], "already_has": []},
            occasion=t["occasion"], budget={"min": 20, "max": 100}))
    return out


def generate_personas(batch, n: int | None = None) -> list[Persona]:
    """Generate n TRAINING personas for the coach's target category. Chunked so rich JSON
    never truncates; tolerant of partial output; falls back to distinct inline personas so a
    round is never starved of training data (which silently flattens the learning curve)."""
    n = n or batch.n_personas
    personas: list[Persona] = []
    attempts, max_attempts = 0, n + 3
    while len(personas) < n and attempts < max_attempts:
        attempts += 1
        want = min(_PERSONA_CHUNK, n - len(personas))
        out = call_llm("persona_gen",
                       PERSONA_SYSTEM.format(n=want, target=batch.target_weakness),
                       f"Generate {want} personas for: {batch.target_weakness}",
                       ctx={"n": want, "target": batch.target_weakness})
        for p in out.get("personas", []):
            persona = _persona_from_raw(p)
            if persona:
                personas.append(persona)
    if not personas:
        print(f"  [world] persona_gen yielded nothing for '{batch.target_weakness}'; "
              f"using {n} fallback personas so training isn't starved")
        personas = _fallback_personas(batch.target_weakness, n)
    return personas[:n]


def react(persona: Persona, proposal) -> Reaction:
    user = (f"YOUR TRUE SELF: {persona.hidden_truth}\n"
            f"YOUR PUBLIC PROFILE: {persona.profile}\n"
            f"GIFT YOU RECEIVED: {proposal.gifts}")
    out = call_llm("reactor", REACT_SYSTEM, user)
    verdict, quote = _enforce_hidden_truth(persona, proposal,
                                            out.get("verdict", "meh"), out.get("quote", ""))
    return Reaction(proposal_id=proposal.proposal_id, persona_id=persona.persona_id,
                    verdict=verdict, quote=quote)


def load_seed_personas(path="seed_personas.json") -> list[Persona]:
    """Bootstrap set so round 1 has something before the coach takes over."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        raw = json.load(f)
    return [Persona(**p) for p in raw]


def perception_extract(media) -> dict:
    """TODO(Violet): real build per perception-agent-spec.md. Image/chat -> profile.
    Stub returns an empty profile so the interface exists and nothing blocks on it."""
    return {"interests": [], "already_owns": [], "constraints": [], "personality": {}}
