"""Gandalf Protocol — Shared Contracts

LOCKED. Everyone builds to these shapes. These dataclasses are the single source of truth
for the JSON that flows between agents, judges, and the orchestrator. If you need to change
a shape, change it HERE and tell the team.
"""
from dataclasses import dataclass, field, asdict
from typing import Any
import uuid, time


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ── The recipient (Violet owns generation; schema mirrors Broflo's Dossier) ──
@dataclass
class Persona:
    profile: dict          # PUBLIC: what the gift agent sees (RecipientProfile shape)
    hidden_truth: dict     # PRIVATE: answer key for the judge; agent NEVER sees this
    occasion: str
    budget: dict           # {"min": float, "max": float}
    persona_id: str = field(default_factory=lambda: _id("persona"))
    is_validation: bool = False   # True => held-out real case, never trained on


# ── The gift agent's output (Warren) ──
@dataclass
class GiftProposal:
    agent_id: str
    strategy_tag: str
    gifts: list            # [{"name","category","price","reasoning"}]
    proposal_id: str = field(default_factory=lambda: _id("prop"))


# ── The simulator's reaction (Violet) ──
@dataclass
class Reaction:
    proposal_id: str
    persona_id: str
    verdict: str           # delight | like | meh | owns_it | return
    quote: str


# ── The judge's score (Jasper) ──
@dataclass
class Score:
    proposal_id: str
    dims: dict             # {"fit","surprise","effort","budget_respect"} each 0..1
    thoughtfulness: float  # 0..1 aggregate — the headline metric
    rationale: str


# ── The stored unit (Kirk) ──
@dataclass
class Episode:
    persona_id: str
    agent_id: str
    strategy_tag: str
    proposal: dict
    reaction: dict
    score: dict
    playbook_version: int
    # condition ("solo"|"swarm"|"validation") + occasion tag the ablation & coach;
    # round_index lets the dashboard plot both conditions on a shared round axis and
    # draw the validation line rising per-round. These used to be injected as loose
    # dict keys AFTER to_json() (bypassing store.save); they now live on the contract
    # so store.save(Episode) preserves them. [Rig fix — announce to team]
    condition: str = "swarm"
    occasion: str = "unknown"
    round_index: int = -1
    is_validation: bool = False
    episode_id: str = field(default_factory=lambda: _id("ep"))
    timestamp: float = field(default_factory=time.time)


# ── The learning substrate (Warren writes via distill; every agent reads) ──
class Playbook:
    """The 'policy'. We don't fine-tune — learning = this list of lessons growing,
    injected into the gift agents' prompts each round.

    ── FROZEN CONTRACT (Round-2) ──
    These are the STABLE public surfaces every caller relies on — do NOT change their
    signatures or remove them:
      • .entries : list[dict]  — each dict has at least {"text","source_episode_ids",
                    "added_at"}; swarm.distill also attaches {"wins","trials"} and
                    retrieval._confidence / playbook_artifact._wilson read those keys.
      • .version : int         — monotonic learning counter (never rewind it).
      • add(text: str, source_ids: list[str]) -> None
      • as_prompt(profile: dict | None = None) -> str
    Round-2 Phase 3 (ACE delta-curation) may ONLY ADD new methods (e.g. refine_in_place,
    remove, merge_or_add, prune, per-lesson gate helpers) and may enrich entry dicts with
    NEW optional keys. It must not break the four surfaces above or the entry keys listed.
    """
    def __init__(self):
        self.entries: list[dict] = []   # [{"text","source_episode_ids","added_at"}]
        self.version: int = 0

    def add(self, text: str, source_ids: list[str]):
        self.entries.append({"text": text, "source_episode_ids": source_ids,
                             "added_at": time.time()})
        self.version += 1

    # ── Round-2 Phase 3 (ACE delta-curation) — ADD-only, additive to the frozen surface ──
    def refine_in_place(self, index: int, text: str, source_ids: list[str] | None = None,
                        wins: int = 0, trials: int = 0) -> dict:
        """Supersede the lesson at `index` IN PLACE (an ACE 'refine' delta): keep the newer
        phrasing, MERGE provenance (source_episode_ids), and ACCUMULATE wins/trials so
        reinforcement raises confidence. Bumps version (reinforcement is a learning event).
        Entry-dict keys stay stable — text/source_episode_ids/added_at/wins/trials."""
        e = self.entries[index]
        e["text"] = text
        if source_ids:
            e["source_episode_ids"] = list(set(e.get("source_episode_ids", [])) | set(source_ids))
        e["wins"] = e.get("wins", 0) + wins
        e["trials"] = e.get("trials", 0) + trials
        self.version += 1
        return e

    def remove(self, index_or_predicate) -> int:
        """Delete lessons (an ACE 'remove' delta) — pass an int index, or a
        predicate(entry) -> bool to drop every matching entry. Returns the count removed.
        version is monotonic (a learning counter): a removal never rewinds it, matching the
        prune contract in swarm.py."""
        before = len(self.entries)
        if callable(index_or_predicate):
            self.entries = [e for e in self.entries if not index_or_predicate(e)]
        else:
            del self.entries[index_or_predicate]
        return before - len(self.entries)

    def as_prompt(self, profile: dict | None = None) -> str:
        """Render lessons for injection into a gift-agent prompt.
        Backward-compatible: `as_prompt()` with no args behaves as before (all lessons).
        Pass the recipient `profile` to get the top-k MOST RELEVANT lessons instead
        (RAG over the playbook via retrieval.py — Actian VectorAI DB when configured,
        local lexical fallback otherwise). [Rig seam — announce to team]"""
        if not self.entries:
            return "(no lessons learned yet)"
        if profile is not None:
            try:
                import retrieval
                return retrieval.lessons_for(self, profile)
            except Exception:
                pass  # never let retrieval break the loop; fall through to dump-all
        return "\n".join(f"- {e['text']}" for e in self.entries)


# ── Coach output (Jasper) ──
@dataclass
class CurriculumBatch:
    target_weakness: str
    rationale: str
    n_personas: int

def to_json(obj) -> dict:
    return asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
