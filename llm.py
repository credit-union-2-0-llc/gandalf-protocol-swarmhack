"""Gandalf Protocol — LLM Wrapper

One place all Claude calls go through. In MOCK mode (no API key) it returns deterministic
fake JSON so the whole loop runs offline at zero cost — you can verify wiring and see
charts render before spending anything.

The mock is rigged so that:
  - judge scores RISE with playbook size → learning curve climbs
  - distillation only happens in swarm condition → ablation shows swarm steeper than solo
This lets you sanity-check the dashboard before a single real call.
"""
import json, random, hashlib, os, threading
import config

_call_count = 0
_count_lock = threading.Lock()   # runner fans out calls across threads; guard the counter

def calls_made() -> int:
    return _call_count

def reset_calls() -> None:
    """[P1] Reset the per-run call counter. run_experiment() calls this so the
    MAX_CALLS_PER_RUN guard bounds a single condition's run, not the whole process
    (an --ablation runs solo + swarm in one process and must not accumulate)."""
    global _call_count
    with _count_lock:
        _call_count = 0


def _client():
    import anthropic
    return anthropic.Anthropic()


# The CU2 LiteLLM gateway's cu2_policy_plugin 400s any /v1/messages call whose
# metadata.user_id isn't a valid UUID. Harmless for direct-Anthropic (metadata is an
# accepted param), so we always send it. Overridable via env. [local run-glue for gateway]
_GATEWAY_USER_ID = os.environ.get("GATEWAY_USER_ID", "a7f3c2e1-9b4d-4e6a-8c2f-1d5e7b9a0c34")


def parse_json(text: str):
    """Strip ```json fences and parse. Raises on failure (caller handles)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    return json.loads(t.strip())


# Roles whose object legitimately wraps a list; a model that returns the BARE list gets
# re-wrapped under the expected key rather than crashing the caller.
_LIST_WRAP_KEY = {"persona_gen": "personas", "gift_agent": "gifts", "distiller": "lessons"}


def _coerce(role: str, obj):
    """Best-effort: guarantee call_llm returns the dict its caller expects, even when the
    model emits a bare list (e.g. reactor returns [{"verdict":..}] or gift_agent returns a
    raw [ ... ]). A malformed single response then degrades to a default-scored episode
    instead of killing the whole run. [integration glue — flag to Rig]"""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        if role in _LIST_WRAP_KEY:                       # bare collection → wrap it
            return {_LIST_WRAP_KEY[role]: obj}
        if len(obj) == 1 and isinstance(obj[0], dict):   # object wrapped in a 1-elem array
            return obj[0]
    return {}  # unusable shape → empty dict; callers use .get(..., default) and move on


def call_llm(role: str, system: str, user: str, ctx: dict | None = None,
             temperature: float | None = None) -> dict:
    """Return parsed JSON from the model for a given role. ctx carries hints the
    mock uses to fake plausible learning (e.g. playbook_len, condition). temperature is
    optional and omitted from the request when None (API default) — the judge role passes
    a low value so the held-out validation line (the reward-hacking canary) isn't noisy
    from run to run for reasons unrelated to the policy actually changing."""
    global _call_count
    with _count_lock:                       # atomic under the runner's thread pool
        _call_count += 1
        count = _call_count
    if config.MAX_CALLS_PER_RUN and count > config.MAX_CALLS_PER_RUN:
        raise RuntimeError(f"Cost guard tripped: >{config.MAX_CALLS_PER_RUN} LLM calls. "
                           "Raise MAX_CALLS_PER_RUN in config.py if intentional.")
    if config.MOCK:
        return _mock(role, system, user, ctx or {})

    # Prompt-cache the system block: the high-volume roles (judge/reactor/persona_gen)
    # reuse a fixed system prompt across hundreds of calls per run, so caching it cuts
    # input cost dramatically (Broflo pattern; practitioners report up to ~90% savings).
    system_block = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    kwargs = {"temperature": temperature} if temperature is not None else {}
    resp = _client().messages.create(
        model=config.MODELS[role], max_tokens=config.MAX_TOKENS,
        system=system_block, messages=[{"role": "user", "content": user}],
        metadata={"user_id": _GATEWAY_USER_ID}, **kwargs,
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    try:
        return _coerce(role, parse_json(text))
    except Exception:
        # One retry with an explicit nudge; then surface the raw text for debugging.
        resp2 = _client().messages.create(
            model=config.MODELS[role], max_tokens=config.MAX_TOKENS, system=system_block,
            messages=[{"role": "user", "content": user},
                      {"role": "assistant", "content": text},
                      {"role": "user", "content": "Return ONLY valid JSON, nothing else."}],
            metadata={"user_id": _GATEWAY_USER_ID}, **kwargs,
        )
        text2 = "".join(b.text for b in resp2.content if b.type == "text")
        try:
            return _coerce(role, parse_json(text2))
        except Exception:
            # Two malformed responses in a row: cost this ONE episode, not the whole run.
            # Callers use .get(..., default) so an empty dict degrades gracefully.
            print(f"  [llm] {role}: unparseable JSON after retry; skipping (empty result)")
            return {}


# ────────────────────────────── MOCK ENGINE ──────────────────────────────
_RNG = random.Random(7)

def _seeded(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 1000 / 1000.0

def _mock(role, system, user, ctx):
    if role == "persona_gen":
        n = ctx.get("n", 3)
        return {"personas": [_mock_persona(i, ctx) for i in range(n)]}
    if role == "gift_agent":
        return {"gifts": [
            {"name": f"gift-{_RNG.randint(1,999)}", "category": "misc",
             "price": round(_RNG.uniform(20, 120), 2),
             "reasoning": "mock reasoning tied to a profile detail"}
            for _ in range(ctx.get("count", 3))]}
    if role == "reactor":
        v = _RNG.choice(["delight", "like", "meh", "owns_it", "return"])
        return {"verdict": v, "quote": f"(mock) {v} reaction"}
    if role == "judge":
        # [Jasper — binary sub-checks] Each check's pass probability climbs with playbook
        # size, noisily — this is what makes the mock learning curve move. budget_ok isn't
        # asked here at all; the real judge() computes it deterministically from price vs.
        # budget, not from the model.
        pb = ctx.get("playbook_len", 0)
        p = max(0.0, min(0.95, 0.35 + 0.05 * pb + _RNG.uniform(-0.08, 0.08)))
        return {"fit_ok": _RNG.random() < p, "surprise_ok": _RNG.random() < p * 0.8,
                "effort_ok": _RNG.random() < p, "rationale": "(mock) score"}
    if role == "distiller":
        return {"lessons": [{"text": f"(mock) lesson v{ctx.get('playbook_len',0)+1}: "
                             "match specific hobbies over generic categories"}]}
    if role == "coach":
        w = _RNG.choice(["coworker gifts", "teens", "under-$20", "hard-to-shop-for dads"])
        return {"target_weakness": w, "rationale": "(mock) lowest avg score here",
                "n_personas": ctx.get("n", config.PERSONAS_PER_ROUND)}
    return {}

def _mock_persona(i, ctx):
    hobbies = _RNG.choice([["trail running"], ["vinyl records"], ["baking"], ["woodworking"]])
    return {
        "profile": {"interests": [{"value": hobbies[0], "confidence": 0.8,
                                   "evidence": "mock", "polarity": "loves"}],
                    "already_owns": [], "constraints": [],
                    "personality": {"gift_style_hint": "experiences", "confidence": 0.6}},
        "hidden_truth": {"would_love": [hobbies[0] + " gear"], "would_return": ["gift cards"],
                         "already_has": []},
        "occasion": "birthday", "budget": {"min": 20, "max": 100},
    }
