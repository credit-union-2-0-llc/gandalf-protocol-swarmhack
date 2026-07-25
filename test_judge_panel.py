"""Gandalf Protocol — Judge panel tests (Round-2 Phase 2) [OWNER: Jasper]

Deterministic and MOCK-safe: every test monkeypatches either the sub-judge helper
(`signal._judge_once`) or the underlying LLM call (`signal.call_llm`) with CONTROLLED votes,
so the majority-vote logic and the deterministic hard-gates are proven without any API key
or cost. This is the offline green bar for the panel change.

Run from repo root:  python3 -m pytest test_judge_panel.py -q

What's covered:
  - K=3 majority per check: [T,T,F]→pass, [T,F,F]→fail, and per-check independence.
  - K=1 reproduces the pre-panel single-call result (exactly one call at temperature 0.2).
  - K>1 makes K genuinely-independent calls (distinct temperatures + varied lens prompt).
  - The deterministic guards STILL hard-gate even when the whole panel votes True:
    a duplicate/already_has gift and a grossly-over-budget gift both stay < 0.4.
"""
import config
from contracts import Persona, GiftProposal, Reaction
from agents import signal


# ── helpers ──

def _persona(would_love=None, already_has=None, budget_max=100):
    return Persona(
        profile={"interests": [], "already_owns": [], "constraints": [], "personality": {}},
        hidden_truth={"would_love": would_love or ["trail running gear"],
                      "would_return": [], "already_has": already_has or []},
        occasion="birthday", budget={"min": 20, "max": budget_max})


def _proposal(gifts):
    return GiftProposal(agent_id="test", strategy_tag="test", gifts=gifts)


def _reaction(prop, p, verdict="like"):
    return Reaction(proposal_id=prop.proposal_id, persona_id=p.persona_id,
                    verdict=verdict, quote="")


def _stub_panel(monkeypatch, per_call_votes):
    """Replace signal._judge_once so judge() sees a fixed, ordered sequence of sub-judge votes.
    per_call_votes: list of {fit_ok,surprise_ok,effort_ok[,rationale]}, one per sub-judge."""
    state = {"i": 0}

    def fake(persona, proposal, reaction, playbook_len, temperature=0.2, lens=""):
        v = per_call_votes[state["i"]]
        state["i"] += 1
        return {"fit_ok": bool(v.get("fit_ok", False)),
                "surprise_ok": bool(v.get("surprise_ok", False)),
                "effort_ok": bool(v.get("effort_ok", False)),
                "rationale": v.get("rationale", "")}

    monkeypatch.setattr(signal, "_judge_once", fake)
    return state


_ALL_TRUE = {"fit_ok": True, "surprise_ok": True, "effort_ok": True}
_ALL_FALSE = {"fit_ok": False, "surprise_ok": False, "effort_ok": False}


# ── (a) majority logic ──

def test_panel_fit_passes_on_2_of_3(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PANEL_SIZE", 3)
    _stub_panel(monkeypatch, [_ALL_TRUE, _ALL_TRUE, _ALL_FALSE])  # [T,T,F]
    p = _persona()
    prop = _proposal([{"name": "trail running vest", "price": 60}])
    score = signal.judge(p, prop, _reaction(prop, p))
    assert score.dims["fit_ok"] == 1.0
    assert score.dims["surprise_ok"] == 1.0
    assert score.dims["effort_ok"] == 1.0
    assert score.thoughtfulness == 1.0  # all four checks pass (budget deterministic-true)


def test_panel_fit_fails_on_1_of_3(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PANEL_SIZE", 3)
    _stub_panel(monkeypatch, [_ALL_TRUE, _ALL_FALSE, _ALL_FALSE])  # [T,F,F]
    p = _persona()
    prop = _proposal([{"name": "trail running vest", "price": 60}])
    score = signal.judge(p, prop, _reaction(prop, p))
    assert score.dims["fit_ok"] == 0.0
    # failed fit_ok is hard-gated below the DoD's < 0.4 bar
    assert score.thoughtfulness < 0.4


def test_panel_votes_each_check_independently(monkeypatch):
    # fit [T,T,F]→pass, surprise [T,F,F]→fail, effort [F,T,T]→pass — proves each sub-check
    # is tallied on its own, not as a single "did this judge pass" bundle.
    monkeypatch.setattr(config, "JUDGE_PANEL_SIZE", 3)
    _stub_panel(monkeypatch, [
        {"fit_ok": True, "surprise_ok": True, "effort_ok": False},
        {"fit_ok": True, "surprise_ok": False, "effort_ok": True},
        {"fit_ok": False, "surprise_ok": False, "effort_ok": True},
    ])
    p = _persona()
    prop = _proposal([{"name": "trail running vest", "price": 60}])
    score = signal.judge(p, prop, _reaction(prop, p))
    assert score.dims["fit_ok"] == 1.0       # 2/3
    assert score.dims["surprise_ok"] == 0.0  # 1/3
    assert score.dims["effort_ok"] == 1.0    # 2/3


# ── (b) K=1 reproduces the pre-panel single-call result ──

def test_k1_reproduces_single_call(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PANEL_SIZE", 1)
    fixed = {"fit_ok": True, "surprise_ok": False, "effort_ok": True, "rationale": "single"}
    seen = {"n": 0, "temps": [], "users": []}

    def fake_call(role, system, user, ctx=None, temperature=None):
        seen["n"] += 1
        seen["temps"].append(temperature)
        seen["users"].append(user)
        return dict(fixed)

    monkeypatch.setattr(signal, "call_llm", fake_call)
    p = _persona()
    prop = _proposal([{"name": "trail running vest", "price": 60}])
    score = signal.judge(p, prop, _reaction(prop, p))

    # exactly ONE judge call, canonical temperature, no lens appended to the prompt
    assert seen["n"] == 1
    assert seen["temps"] == [0.2]
    assert "LENS" not in seen["users"][0]
    # same checks + aggregate the pre-panel judge() produced: fit+effort+budget pass, surprise fails
    assert score.dims == {"fit_ok": 1.0, "surprise_ok": 0.0, "effort_ok": 1.0, "budget_ok": 1.0}
    assert score.thoughtfulness == 0.8  # 0.4 (fit) + 0.2 (effort) + 0.2 (budget)
    assert score.rationale == "single"


def test_kgt1_makes_k_independent_calls(monkeypatch):
    # Real _judge_once + judge(); only the LLM boundary is stubbed. Proves the panel actually
    # fans out K calls that differ (distinct temperatures + a varied lens hint in the prompt).
    monkeypatch.setattr(config, "JUDGE_PANEL_SIZE", 3)
    seen = {"temps": [], "users": []}

    def fake_call(role, system, user, ctx=None, temperature=None):
        seen["temps"].append(temperature)
        seen["users"].append(user)
        return {"fit_ok": True, "surprise_ok": True, "effort_ok": True, "rationale": "x"}

    monkeypatch.setattr(signal, "call_llm", fake_call)
    p = _persona()
    prop = _proposal([{"name": "trail running vest", "price": 60}])
    signal.judge(p, prop, _reaction(prop, p))

    assert len(seen["temps"]) == 3
    assert len(set(seen["temps"])) == 3         # each sub-judge at a distinct temperature
    assert seen["temps"][0] == 0.2              # first sub-judge keeps the canonical temp
    assert len(set(seen["users"])) >= 2         # lens hint varies the prompt across sub-judges


# ── (c) deterministic hard-gates override a unanimous panel ──

def test_duplicate_veto_overrides_unanimous_panel(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PANEL_SIZE", 3)
    _stub_panel(monkeypatch, [_ALL_TRUE, _ALL_TRUE, _ALL_TRUE])  # every sub-judge says fit=True
    p = _persona(already_has=[{"value": "GPS running watch"}])
    prop = _proposal([{"name": "GPS Running Watch", "price": 60}])  # duplicates already_has
    score = signal.judge(p, prop, _reaction(prop, p, verdict="owns_it"))
    assert score.dims["fit_ok"] == 0.0     # veto beats the vote
    assert score.thoughtfulness < 0.4      # fit-fail hard gate holds


def test_gross_over_budget_overrides_unanimous_panel(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PANEL_SIZE", 3)
    _stub_panel(monkeypatch, [_ALL_TRUE, _ALL_TRUE, _ALL_TRUE])
    p = _persona(budget_max=60)
    prop = _proposal([{"name": "trail running vest", "price": 900}])  # ~15x the ceiling
    score = signal.judge(p, prop, _reaction(prop, p))
    assert score.dims["fit_ok"] == 1.0     # panel voted fit, and it's not a duplicate
    assert score.dims["budget_ok"] == 0.0  # deterministic budget check fails
    assert score.thoughtfulness < 0.4      # grossly-over-budget hard gate still fires
