# Broflo drop-in: consume the Gandalf Playbook (for the Broflo team)

This is a **reviewed module to add to Broflo** — not a change we push into Broflo's prod. It's
**off by default** and **refuses to inject an uncalibrated playbook** (checks `ready_for_prod`).

## 1. The client (add as `services/ai/app/gandalf_playbook.py`)

```python
import os, json, urllib.request

GANDALF_URL = os.environ.get("GANDALF_URL", "https://ca-gandalf-protocol.wittyflower-1831f2a2.westus2.azurecontainerapps.io")
GANDALF_DOMAIN = os.environ.get("GANDALF_DOMAIN", "gifts")   # Broflo=gifts, fantasy app=fantasy-football, travel app=travel
PLAYBOOK_ENABLED = os.environ.get("PLAYBOOK_ENABLED", "false").lower() == "true"

def fetch_playbook() -> dict | None:
    """Fetch the current Gandalf playbook for THIS app's domain. Returns None on any failure."""
    if not PLAYBOOK_ENABLED:
        return None
    try:
        with urllib.request.urlopen(f"{GANDALF_URL}/api/playbook?domain={GANDALF_DOMAIN}", timeout=3) as r:
            pb = json.load(r)
        return pb if pb.get("ready_for_prod") else None   # HARD gate — uncalibrated => ignore
    except Exception:
        return None

def lessons_block(profile: dict, k: int = 6) -> str:
    """Top-k relevant lessons as a prompt block; '' if disabled/ungated/unavailable."""
    pb = fetch_playbook()
    if not pb or not pb.get("lessons"):
        return ""
    # simple lexical relevance; swap for embeddings later
    words = {w.lower() for v in profile.values() if isinstance(v, str) for w in v.split()}
    ranked = sorted(pb["lessons"],
                    key=lambda l: (len(words & set(l["text"].lower().split())), l.get("confidence", 0)),
                    reverse=True)
    return "\n".join(f"- {l['text']}" for l in ranked[:k])
```

## 2. Wire into the prompt (one line in `prompt.py::build_system_prompt`)

```python
from .gandalf_playbook import lessons_block
# ...inside build_system_prompt, where the "LEARNED LESSONS" section goes:
learned = lessons_block(dossier_dict)
if learned:
    prompt += f"\n\nLEARNED LESSONS (apply these — proven across many recipients):\n{learned}\n"
```

## 2.5 The SAME drop-in works for every app/domain

This module is generic — the only per-app difference is the `GANDALF_DOMAIN` env var. One mechanism,
many apps:

| App | `GANDALF_DOMAIN` | Reads | Injects lessons into |
|---|---|---|---|
| **Broflo** | `gifts` | `playbook/gifts/latest.json` | its `/suggest` prompt |
| **Warren's fantasy-football app** | `fantasy-football` | `playbook/fantasy-football/latest.json` | its lineup-suggestion prompt |
| **Jasper's travel-planner app** | `travel` | `playbook/travel/latest.json` | its itinerary prompt |

Each domain's Gandalf loop publishes its own gated playbook (namespaced by domain); each app fetches
*its* domain and injects the top-k relevant lessons the same way. Nothing app-specific in the client
except the domain name and where the lessons string gets spliced into that app's prompt.

## 3. Rollout (safe by construction)
- Ship with `PLAYBOOK_ENABLED=false` → **zero behavior change** (shadow: log `lessons_block()` output, don't inject).
- Flip `true` for a **small A/B slice**; compare Broflo's own suggestion-quality metric vs control.
- Promote only if the playbook arm wins for a sustained window. Instant rollback = flip the env var.
- The client already **fails closed** on ungated playbooks, network errors, or the flag being off.

## 4. What Gandalf provides (this repo)
- `integration/playbook_artifact.py` — builds + gates + publishes the artifact.
- **Next Gandalf step (small):** `GET /api/playbook` endpoint + call `publish` after each run so the
  artifact is always current. (Tracked as the integration build task.)

## 5. Still gated on
- Phase 0: the gift curve must climb (judge calibration #5) → only then does `ready_for_prod` flip true.
- PII: Broflo→Gandalf outcome export (Direction B) routes through Presidio before Gandalf sees it.
