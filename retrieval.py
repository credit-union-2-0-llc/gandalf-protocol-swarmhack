"""Gandalf Protocol — Playbook Retrieval Seam [OWNER: Kirk]

The Playbook is our "policy" (we don't fine-tune — learning = the lesson list growing).
Naively we dump ALL lessons into every gift-agent prompt. That has two known failure
modes the research is loud about:

  1. "Context collapse" — as the playbook grows, dumping everything drowns the relevant
     lesson and craters accuracy (Stanford ACE, ICLR 2026).
  2. Cost — every lesson is re-sent on every call.

This module is the seam that fixes both: given the CURRENT recipient profile, return the
top-k MOST RELEVANT lessons instead of all of them (RAG over the playbook).

Backends, auto-selected, all behind ONE function `lessons_for(playbook, profile, k)`:
  • Actian VectorAI DB  — if ACTIAN_* env is set + SDK present. Portable vector DB
    (edge→cloud, same API); we embed each lesson once and do cosine top-k per persona.
    This is our on-theme SwarmHack sponsor integration.
  • Local lexical fallback — token-overlap ranking, zero deps, works offline / in MOCK
    mode so the never-cut trio never depends on Actian being reachable.
  • Recency fallback — if a profile can't be summarized, return the last-k lessons.

Nothing here is required for the loop to run; if everything fails we return the classic
"dump all" string, so behavior is never worse than the original scaffold.
"""
import os, re, hashlib
from functools import lru_cache

# ── config ──
ACTIAN_ENABLED = bool(os.environ.get("ACTIAN_CONNECTION_STRING") or os.environ.get("ACTIAN_DB_URL"))
DEFAULT_K = int(os.environ.get("PLAYBOOK_TOPK", "8"))


def lessons_for(playbook, profile: dict | None = None, k: int = DEFAULT_K) -> str:
    """Return the k most relevant playbook lessons for `profile`, as a prompt block.
    Falls back gracefully: Actian → local lexical → recency → dump-all."""
    entries = getattr(playbook, "entries", []) or []
    if not entries:
        return "(no lessons learned yet)"
    if profile is None:
        return _format(entries[-k:])

    query = _profile_text(profile)
    if not query.strip():
        return _format(entries[-k:])

    if ACTIAN_ENABLED:
        try:
            return _format(_actian_topk(entries, query, k))
        except Exception as e:  # never let retrieval break a run
            print(f"  [retrieval] Actian unavailable ({e}); using local ranker")

    return _format(_lexical_topk(entries, query, k))


# ── prompt formatting (ACE-style itemized bullets, highest-confidence first) ──
def _format(entries: list[dict]) -> str:
    if not entries:
        return "(no lessons learned yet)"
    ordered = sorted(entries, key=_confidence, reverse=True)
    return "\n".join(f"- {e['text']}" for e in ordered)


def _confidence(entry: dict) -> float:
    """Wilson-style confidence if present (playbookd pattern); else neutral 0.5.
    A lesson with 1 success shouldn't outrank one proven 95/100 — Warren's distiller
    can populate `wins`/`trials` on entries to make this real."""
    wins, trials = entry.get("wins"), entry.get("trials")
    if isinstance(wins, (int, float)) and isinstance(trials, (int, float)) and trials > 0:
        p = wins / trials
        # lower bound of a simple Wilson interval (z≈1.96)
        z2 = 3.8416
        denom = 1 + z2 / trials
        centre = p + z2 / (2 * trials)
        margin = 1.96 * ((p * (1 - p) + z2 / (4 * trials)) / trials) ** 0.5
        return (centre - margin) / denom
    return 0.5


# ── local lexical ranker (default; zero deps) ──
_WORD = re.compile(r"[a-z0-9]+")

def _tokens(s: str) -> set[str]:
    return set(_WORD.findall(s.lower()))

def _lexical_topk(entries: list[dict], query: str, k: int) -> list[dict]:
    q = _tokens(query)
    if not q:
        return entries[-k:]
    scored = []
    for e in entries:
        overlap = len(_tokens(e.get("text", "")) & q)
        scored.append((overlap, e))
    scored.sort(key=lambda t: (t[0], _confidence(t[1])), reverse=True)
    # keep only entries with some overlap; if none overlap, fall back to recency
    hits = [e for ov, e in scored if ov > 0]
    return hits[:k] if hits else entries[-k:]


def _profile_text(profile: dict) -> str:
    """Flatten a recipient profile into a query string for retrieval."""
    parts = []
    for key in ("interests", "already_owns", "constraints"):
        for item in profile.get(key, []) or []:
            parts.append(item.get("value", "") if isinstance(item, dict) else str(item))
    pers = profile.get("personality", {}) or {}
    if isinstance(pers, dict):
        parts.append(str(pers.get("gift_style_hint", "")))
    return " ".join(p for p in parts if p)


# ── Actian VectorAI DB adapter (the SwarmHack sponsor integration) ──
# Per Violet's ACTIAN-INTEGRATION-SPEC: local MiniLM embeddings + the real (Qdrant-style)
# Actian SDK. Fails closed — any error here is caught in lessons_for() and falls back to
# lexical, so the never-cut trio never depends on Actian being reachable.
EMBED_DIM = 384                       # all-MiniLM-L6-v2 — MUST match the collection VectorParams size
_ACTIAN_COLLECTION = "gandalf_playbook"
_actian_ready = False


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _embed(text: str) -> list[float]:
    """text → 384-dim unit vector via a small local model (offline, deterministic, CPU-fine)."""
    return _embedder().encode(text, normalize_embeddings=True).tolist()


def _stable_id(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:12], 16)  # stable across restarts (unlike hash())


@lru_cache(maxsize=1)
def _actian_client():
    # The SDK client is lazy — constructing it does NOT open the connection; collections/
    # points calls raise "Client is not connected" until .connect() runs. Cache the connected
    # client so we pay the gRPC handshake once, not per retrieval call.
    from actian_vectorai import VectorAIClient
    c = VectorAIClient(os.environ.get("ACTIAN_CONNECTION_STRING") or os.environ["ACTIAN_DB_URL"])
    c.connect()
    return c


def _actian_topk(entries: list[dict], query: str, k: int) -> list[dict]:
    from actian_vectorai import VectorParams, Distance, PointStruct
    global _actian_ready
    client = _actian_client()
    if not _actian_ready:
        try:
            client.collections.create(_ACTIAN_COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.Cosine))
        except Exception:
            pass  # already exists
        _actian_ready = True
    # re-upsert lessons every call (idempotent by stable id) — cheap; keeps ephemeral DB filled
    points = [PointStruct(id=_stable_id(e["text"]), vector=_embed(e["text"]),
                          payload={"text": e["text"]}) for e in entries]
    client.points.upsert(_ACTIAN_COLLECTION, points)
    hits = client.points.search(_ACTIAN_COLLECTION, vector=_embed(query), limit=k)
    texts = {h.payload["text"] for h in hits}
    return [e for e in entries if e["text"] in texts]
