"""Gandalf Protocol — Central config. Change run sizes and models here — not scattered in code."""
import os

# Load .env (zero-dep) so `cp .env.example .env` + fill-in-the-key just works — the app
# reads os.environ, and nothing else loads the file. Real env vars win over .env; lines are
# KEY=VALUE, '#' comments and blanks ignored, surrounding quotes stripped. [Rig auth-glue]
def _load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:      # don't override an already-set real env var
                os.environ[k] = v
_load_dotenv()

# Real Claude calls happen only when a key is present. No key => MOCK mode
# (deterministic fake responses, zero cost) so you can verify the whole loop
# end-to-end before spending anything. Flip to real by setting ANTHROPIC_API_KEY.
_key = os.environ.get("ANTHROPIC_API_KEY", "")
# An uncustomized .env leaves a placeholder ("<...>") or the sample sk-ant stub — that is NOT
# a real key. Treat it as unset so we fall back to MOCK cleanly instead of firing a bogus 401
# that looks like "it's asking for the key." [Rig — stop the shared-key confusion]
if _key.startswith("<") or _key in ("", "sk-ant-...your-key-here..."):
    os.environ.pop("ANTHROPIC_API_KEY", None)
    _key = ""
MOCK = not bool(_key)

# Use Azure Blob Storage instead of local JSON (set automatically if connection string is present)
USE_AZURE = bool(os.environ.get("AZURE_STORAGE_CONNECTION_STRING"))

# Model per role. Cheap models for the high-volume roles (simulator/judge),
# a stronger one for the policy (gift agent). Strings are the real API ids.
# NOTE: names are gateway ALIASES (CU2 LiteLLM gateway rejects dated ids like
# claude-haiku-4-5-20251001 — key allowlists only carry the alias). Overridable via env
# so a direct-Anthropic run can pin dated ids without editing this file. [local run-glue]
MODELS = {
    "gift_agent": os.environ.get("MODEL_GIFT_AGENT", "claude-sonnet-5"),
    "distiller":  os.environ.get("MODEL_DISTILLER",  "claude-sonnet-5"),
    "persona_gen":os.environ.get("MODEL_PERSONA_GEN","claude-haiku-4-5"),
    "reactor":    os.environ.get("MODEL_REACTOR",    "claude-haiku-4-5"),
    "judge":      os.environ.get("MODEL_JUDGE",      "claude-haiku-4-5"),
    "coach":      os.environ.get("MODEL_COACH",      "claude-sonnet-5"),
}
# 1200 truncates persona_gen (3 full personas) + rich gift reasoning mid-JSON, and a
# parse failure kills the whole run (llm.parse_json raises; loop only catches the cost
# guard). 4096 gives headroom. Env-overridable. [local run-glue — flag to Rig]
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))

# Run sizes. START SMALL. Prove the loop on a tiny run, check the bill, then scale.
# Env-overridable so we can dial run cost/latency without a code change — a trimmed run
# (fewer personas) banks demo-ready charts + A/B verdict in minutes instead of ~40.
SMOKE_PERSONAS_PER_ROUND = 3     # for the first real run — keep tiny
PERSONAS_PER_ROUND = int(os.environ.get("PERSONAS_PER_ROUND", "6"))
GIFTS_PER_PROPOSAL = int(os.environ.get("GIFTS_PER_PROPOSAL", "3"))
SWARM_SIZE = int(os.environ.get("SWARM_SIZE", "4"))   # diverse agents in the swarm condition

# Cost guardrail: hard stop if a single run would exceed this many LLM calls.
MAX_CALLS_PER_RUN = int(os.environ.get("MAX_CALLS_PER_RUN", "3000"))  # was 800 — too low; 6-round 30-persona swarm cut off at round 5

STORE_PATH = "store.json"
