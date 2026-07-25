"""Gandalf Protocol — Tests for ACE delta-curation [Round-2 Phase 3]

Run from repo root: `python3 test_ace_curation.py`  (also collected by pytest)

Deterministic, MOCK-safe (no API key / zero cost). Exercises the curation MECHANICS behind
config.ACE_CURATION — the parts that are pure Python and don't need a live judge:

  • refine-in-place vs net-new append, decided by SUPERSEDE_RATIO
  • wins/trials accumulate across a refine; source_episode_ids provenance merges
  • a contradicted lesson (opposite polarity, near-identical text) is REMOVED then replaced
  • a proven-losing lesson (trials≥N, zero wins) is remove-delta'd
  • _prune caps the playbook at PLAYBOOK_CAP
  • the per-lesson gate is a PASS-THROUGH no-op when held_out_eval=None (deterministic in
    MOCK), and a pluggable rollback when a real eval callback is supplied (the deployed path)
  • flag-OFF distill ignores the new params entirely → byte-identical legacy behavior

The REAL held-out gate needs the live LLM judge and only runs deployed; here it's exercised
with a synthetic eval callback so the pluggability + rollback are provable offline.
"""
import agents.swarm as swarm_mod
from agents.swarm import Swarm, PLAYBOOK_CAP, SUPERSEDE_RATIO, WIN_THRESHOLD
from contracts import Playbook
import config

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


# ── helpers ──────────────────────────────────────────────────────────────────
def _ep(eid, tf, gift="trail running vest", reasoning="matches their stated trail-running hobby",
        verdict="delight", strategy="safe-practical"):
    """A synthetic stored-episode dict shaped like store.top/bottom_episodes returns."""
    return {
        "episode_id": eid, "strategy_tag": strategy,
        "proposal": {"gifts": [{"name": gift, "category": "apparel", "reasoning": reasoning}]},
        "reaction": {"verdict": verdict},
        "score": {"thoughtfulness": tf, "rationale": f"(fake) score {tf}"},
    }


class _FakeLLM:
    """Scripted distiller: returns an 'avoid' lesson for the failure-reflection system prompt
    (identified by 'FELL SHORT'), a 'do' lesson otherwise. Deterministic — no randomness."""
    def __init__(self, do_text, avoid_text):
        self.do_text, self.avoid_text = do_text, avoid_text
        self.calls = []

    def __call__(self, role, system, user, ctx=None, temperature=None):
        self.calls.append((role, "FELL SHORT" in system))
        text = self.avoid_text if "FELL SHORT" in system else self.do_text
        return {"lessons": [{"text": text}]}


class _patched:
    """Context manager: swap agents.swarm.call_llm + config.ACE_CURATION, restore on exit."""
    def __init__(self, call_llm=None, ace=None):
        self.call_llm, self.ace = call_llm, ace

    def __enter__(self):
        self._llm, self._ace = swarm_mod.call_llm, config.ACE_CURATION
        if self.call_llm is not None:
            swarm_mod.call_llm = self.call_llm
        if self.ace is not None:
            config.ACE_CURATION = self.ace
        return self

    def __exit__(self, *a):
        swarm_mod.call_llm, config.ACE_CURATION = self._llm, self._ace


DO = "match trail-running gear to the stated running hobby, not generic fitness equipment"
AVOID = "avoid generic gift cards when the profile names a concrete hobby — they read as low-effort"


# ── refine-in-place vs append, accumulation, provenance ───────────────────────
def test_refine_in_place_accumulates_and_merges_provenance():
    sw, pb = Swarm(size=1), Playbook()
    sw._curate_one(pb, {"text": DO, "polarity": "do", "source_ids": ["ep1"],
                        "wins": 2, "trials": 3}, held_out_eval=None)
    check("first lesson appends (net-new)", len(pb.entries) == 1, f"len={len(pb.entries)}")
    e = pb.entries[0]
    check("net-new carries wins/trials", e["wins"] == 2 and e["trials"] == 3, str(e))
    check("net-new carries provenance", e["source_episode_ids"] == ["ep1"], str(e))

    # near-identical text (ratio 1.0 ≥ SUPERSEDE_RATIO) → refine-in-place, not a 2nd entry
    sw._curate_one(pb, {"text": DO, "polarity": "do", "source_ids": ["ep2"],
                        "wins": 1, "trials": 2}, held_out_eval=None)
    check("near-dup refines in place (no append)", len(pb.entries) == 1, f"len={len(pb.entries)}")
    e = pb.entries[0]
    check("wins accumulate on refine", e["wins"] == 3, f"wins={e['wins']}")
    check("trials accumulate on refine", e["trials"] == 5, f"trials={e['trials']}")
    check("provenance merges on refine", set(e["source_episode_ids"]) == {"ep1", "ep2"}, str(e))

    # clearly different text (ratio < SUPERSEDE_RATIO) → net-new append
    sw._curate_one(pb, {"text": "prefer experience gifts for people who dislike clutter",
                        "polarity": "do", "source_ids": ["ep3"], "wins": 1, "trials": 1},
                   held_out_eval=None)
    check("distinct lesson appends as net-new", len(pb.entries) == 2, f"len={len(pb.entries)}")


# ── contradiction removes then replaces ───────────────────────────────────────
def test_contradiction_removes_stale_lesson():
    sw, pb = Swarm(size=1), Playbook()
    sw._append(pb, DO, ["ep1"], 2, 3, "do")
    check("seed 'do' lesson present", len(pb.entries) == 1 and pb.entries[0]["polarity"] == "do")
    # an 'avoid' lesson with near-identical text NEGATES the 'do' rule → remove + add
    sw._curate_one(pb, {"text": DO, "polarity": "avoid", "source_ids": ["ep9"],
                        "wins": 0, "trials": 2}, held_out_eval=None)
    check("contradiction keeps a single entry (remove+replace)", len(pb.entries) == 1,
          f"len={len(pb.entries)}")
    check("surviving entry flipped to 'avoid'", pb.entries[0]["polarity"] == "avoid",
          str(pb.entries[0]))


# ── proven-loser remove-delta ─────────────────────────────────────────────────
def test_curate_removals_drops_proven_losers():
    sw, pb = Swarm(size=1), Playbook()
    sw._append(pb, DO, ["ep1"], 5, 5, "do")                 # 5/5 winning 'do' — keep
    sw._append(pb, "always recommend spa sets for anyone at all", ["ep2"], 0, 5, "do")  # 0/5 loser 'do'
    sw._append(pb, "brand new untested lesson about niche hobbies", ["ep3"], 0, 1, "do")  # young — keep
    sw._append(pb, AVOID, ["ep4"], 0, 5, "avoid")           # 0/5 'avoid' — EXEMPT (negative knowledge)
    sw._curate_removals(pb)
    texts = [e["text"] for e in pb.entries]
    check("proven-losing 'do' rule removed", "always recommend spa sets for anyone at all" not in texts,
          str(texts))
    check("proven winner kept", DO in texts, str(texts))
    check("young untested lesson kept (not enough trials to condemn)",
          "brand new untested lesson about niche hobbies" in texts, str(texts))
    check("'avoid' lesson exempt from zero-wins cull (negative knowledge)", AVOID in texts,
          str(texts))


# ── prune caps at PLAYBOOK_CAP ────────────────────────────────────────────────
def test_prune_caps_at_playbook_cap():
    sw, pb = Swarm(size=1), Playbook()
    for i in range(PLAYBOOK_CAP + 12):
        # varied confidence so prune has a real ranking to apply
        sw._append(pb, f"lesson number {i} about a specific distinct hobby detail",
                   [f"ep{i}"], wins=i % 5, trials=5, polarity="do")
    check("playbook over cap before prune", len(pb.entries) == PLAYBOOK_CAP + 12)
    sw._prune(pb)
    check("prune caps at PLAYBOOK_CAP", len(pb.entries) == PLAYBOOK_CAP, f"len={len(pb.entries)}")


# ── per-lesson gate: no-op vs pluggable rollback ──────────────────────────────
def test_gate_noop_when_eval_none():
    sw, pb = Swarm(size=1), Playbook()
    sw._curate_one(pb, {"text": DO, "polarity": "do", "source_ids": ["ep1"],
                        "wins": 1, "trials": 1}, held_out_eval=None)
    check("gate=None admits unconditionally (pass-through)", len(pb.entries) == 1)


def test_gate_rejects_when_held_out_drops():
    sw, pb = Swarm(size=1), Playbook()
    # eval that gets WORSE the moment we add anything → candidate must be rolled back
    worse = lambda p: -len(p.entries)
    v0 = pb.version
    sw._curate_one(pb, {"text": DO, "polarity": "do", "source_ids": ["ep1"],
                        "wins": 1, "trials": 1}, held_out_eval=worse)
    check("gate rejects a lesson that lowers held-out", len(pb.entries) == 0, f"len={len(pb.entries)}")
    check("rejected lesson rolls back version too", pb.version == v0, f"v={pb.version}")

    # eval that IMPROVES with a bigger playbook → candidate admitted
    better = lambda p: len(p.entries)
    sw._curate_one(pb, {"text": DO, "polarity": "do", "source_ids": ["ep1"],
                        "wins": 1, "trials": 1}, held_out_eval=better)
    check("gate admits a lesson that doesn't lower held-out", len(pb.entries) == 1)


# ── full ACE distill path (flag ON) with synthetic episodes ───────────────────
def test_distill_ace_emits_do_and_avoid():
    sw = Swarm(size=1)
    pb = Playbook()
    tops = [_ep("t1", 0.9), _ep("t2", 0.85)]
    bottoms = [_ep("b1", 0.1, gift="$50 gift card", verdict="meh"),
               _ep("b2", 0.15, gift="scented candle", verdict="return")]
    fake = _FakeLLM(DO, AVOID)
    with _patched(call_llm=fake, ace=True):
        sw.distill(tops, pb, bottom_episodes=bottoms, held_out_eval=None)
    texts = [e["text"] for e in pb.entries]
    check("ACE distill emits the 'do' lesson", DO in texts, str(texts))
    check("ACE distill emits the GEPA 'avoid' lesson", AVOID in texts, str(texts))
    do_e = next(e for e in pb.entries if e["text"] == DO)
    check("'do' lesson wins == count of winners (tf≥threshold)", do_e["wins"] == 2,
          f"wins={do_e['wins']}, thr={WIN_THRESHOLD}")
    check("'do' provenance from top episodes", set(do_e["source_episode_ids"]) == {"t1", "t2"},
          str(do_e))
    avoid_e = next(e for e in pb.entries if e["text"] == AVOID)
    check("'avoid' lesson has zero wins", avoid_e["wins"] == 0, str(avoid_e))
    check("'avoid' provenance from bottom episodes",
          set(avoid_e["source_episode_ids"]) == {"b1", "b2"}, str(avoid_e))
    check("distiller was called for BOTH win and fail reflection",
          any(fail for _, fail in fake.calls) and any(not fail for _, fail in fake.calls),
          str(fake.calls))


# ── flag-OFF == legacy behavior on the same inputs ────────────────────────────
def test_flag_off_ignores_new_params_and_matches_legacy():
    tops = [_ep("t1", 0.9), _ep("t2", 0.85)]
    bottoms = [_ep("b1", 0.1, gift="$50 gift card")]

    # Flag OFF: distill must ignore bottom_episodes + held_out_eval entirely (legacy path).
    sw = Swarm(size=1)
    pb_off = Playbook()
    fake = _FakeLLM(DO, AVOID)
    with _patched(call_llm=fake, ace=False):
        sw.distill(tops, pb_off, bottom_episodes=bottoms, held_out_eval=lambda p: -999)
    off_texts = [e["text"] for e in pb_off.entries]
    check("flag OFF ignores bottoms (no 'avoid' lesson)", AVOID not in off_texts, str(off_texts))
    check("flag OFF still learns the 'do' lesson", DO in off_texts, str(off_texts))
    check("flag OFF never tagged polarity (legacy entry shape)",
          all("polarity" not in e for e in pb_off.entries), str(pb_off.entries))
    check("flag OFF was NOT rolled back by the ignored held_out_eval",
          len(pb_off.entries) == 1, f"len={len(pb_off.entries)}")
    check("flag OFF only ran WIN reflection (no failure reflection call)",
          all(not fail for _, fail in fake.calls), str(fake.calls))

    # Flag ON on identical inputs behaves DIFFERENTLY (proves the flag is the only lever).
    sw2 = Swarm(size=1)
    pb_on = Playbook()
    with _patched(call_llm=_FakeLLM(DO, AVOID), ace=True):
        sw2.distill(tops, pb_on, bottom_episodes=bottoms, held_out_eval=None)
    on_texts = [e["text"] for e in pb_on.entries]
    check("flag ON adds the 'avoid' lesson the OFF path skipped", AVOID in on_texts, str(on_texts))


ALL = [
    test_refine_in_place_accumulates_and_merges_provenance,
    test_contradiction_removes_stale_lesson,
    test_curate_removals_drops_proven_losers,
    test_prune_caps_at_playbook_cap,
    test_gate_noop_when_eval_none,
    test_gate_rejects_when_held_out_drops,
    test_distill_ace_emits_do_and_avoid,
    test_flag_off_ignores_new_params_and_matches_legacy,
]

if __name__ == "__main__":
    for t in ALL:
        t()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        raise SystemExit(1)
    print("All ACE-curation tests passed.")
