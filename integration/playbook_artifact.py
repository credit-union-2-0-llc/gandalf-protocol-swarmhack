"""Gandalf → apps: the Playbook artifact [OWNER: Kirk]

The ONE thing an app (Broflo first) consumes from Gandalf: a versioned, gated Playbook.
Safe to build ahead of the Phase-0 gate because the artifact carries `ready_for_prod` —
an app MUST refuse to use a playbook that isn't gated true. See docs/INTEGRATION-SPEC.md.
"""
import json, time
from contracts import to_json

# Phase-0 gate thresholds — an app may only inject a playbook that clears ALL of these.
KAPPA_MIN = 0.60          # judge must agree with the gold set (calibrated)
VALIDATION_MIN = 0.55     # held-out score must be respectable (it generalizes)
MIN_LESSONS = 3


def build_artifact(store, playbook, judge_kappa=None, domain="gifts") -> dict:
    """Serialize the current Playbook into the cross-app contract shape.
    `judge_kappa` comes from evals/run_evals.py (None if the gold set isn't filled yet)."""
    val = store.validation_series_by_round()
    validation_score = round(val[-1][1], 3) if val else None
    lessons = [{"text": e["text"],
                "confidence": round(_wilson(e), 3),
                "source_episode_ids": e.get("source_episode_ids", [])}
               for e in getattr(playbook, "entries", [])]
    ready, reasons = _gate(judge_kappa, validation_score, lessons)
    return {
        "schema": "gandalf.playbook/v1",
        "domain": domain,
        "version": getattr(playbook, "version", 0),
        "generated_at": time.time(),
        "judge_kappa": judge_kappa,
        "validation_score": validation_score,
        "ready_for_prod": ready,          # apps MUST check this before injecting
        "gate_reasons": reasons,          # why it is / isn't ready
        "lessons": lessons,
    }


def _wilson(entry) -> float:
    wins, trials = entry.get("wins"), entry.get("trials")
    if isinstance(wins, (int, float)) and isinstance(trials, (int, float)) and trials > 0:
        p = wins / trials
        z2 = 3.8416
        return (p + z2 / (2 * trials) - 1.96 * ((p * (1 - p) + z2 / (4 * trials)) / trials) ** 0.5) / (1 + z2 / trials)
    return 0.5


def _gate(kappa, validation, lessons):
    reasons = []
    if kappa is None:
        reasons.append("judge κ unknown (fill the gold set + run evals)")
    elif kappa < KAPPA_MIN:
        reasons.append(f"judge κ {kappa:.2f} < {KAPPA_MIN} (not calibrated)")
    if validation is None:
        reasons.append("no held-out validation score yet")
    elif validation < VALIDATION_MIN:
        reasons.append(f"validation {validation:.2f} < {VALIDATION_MIN} (doesn't generalize yet)")
    if len(lessons) < MIN_LESSONS:
        reasons.append(f"only {len(lessons)} lessons (< {MIN_LESSONS})")
    return (len(reasons) == 0), reasons


def publish_to_blob(store, artifact) -> str:
    """Write the artifact namespaced BY DOMAIN so every domain/app has its own playbook:
    playbook/<domain>/vN.json + playbook/<domain>/latest.json. Returns the version key.
    Broflo reads playbook/gifts/latest.json; Warren's app reads playbook/fantasy-football/...;
    Jasper's reads playbook/travel/... — one mechanism, many domains."""
    d = artifact.get("domain", "gifts")
    key = f"playbook/{d}/v{artifact['version']}.json"
    data = json.dumps(artifact, indent=2)
    for name in (key, f"playbook/{d}/latest.json"):
        blob = store.client.get_blob_client(container=store.container, blob=name)
        blob.upload_blob(data, overwrite=True)
    return key
