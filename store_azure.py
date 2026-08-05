"""Gandalf Protocol — Azure Blob Storage Backend [OWNER: Kirk]

Same method signatures as store.py — swap one import and nothing else changes.
Connects via AZURE_STORAGE_CONNECTION_STRING env var.

Episodes are stored as JSON objects in blob storage:
  - gandalf/episodes/ ... the individual episode JSONs
  - gandalf/episodes.json ... the full episodes list (downloaded for analysis)
"""
import json, os
from azure.storage.blob import BlobServiceClient, BlobClient
import logging

logger = logging.getLogger(__name__)

def _get_blob_service():
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")
    return BlobServiceClient.from_connection_string(conn_str)

class Store:
    def __init__(self, container="gandalf"):
        self.client = _get_blob_service()
        self.container = container
        self.episodes = []
        # Ensure container exists
        try:
            self.client.create_container(name=container)
        except:
            pass  # theater-ok: Azure blob container may already exist; idempotent create
        # Parity with store.py: reload prior episodes so a fresh process
        # (`python dashboard.py`, or any worker that didn't run the loop) can replay.
        self.load()

    def save(self, episode):
        """Save an episode to blob storage and sync local list."""
        from contracts import to_json
        ep_dict = to_json(episode)
        self.episodes.append(ep_dict)
        self._flush()

    def _flush(self):
        """Write the full episodes list to blob."""
        blob_client = self.client.get_blob_client(container=self.container, 
                                                   blob="episodes.json")
        data = json.dumps({"episodes": self.episodes}, indent=2)
        blob_client.upload_blob(data, overwrite=True)
        logger.info(f"Flushed {len(self.episodes)} episodes to blob")

    def load(self):
        """Pull episodes from blob on startup."""
        blob_client = self.client.get_blob_client(container=self.container,
                                                   blob="episodes.json")
        try:
            data = blob_client.download_blob().readall().decode()
            self.episodes = json.loads(data).get("episodes", [])
            logger.info(f"Loaded {len(self.episodes)} episodes from blob")
        except:
            logger.info("No episodes.json in blob yet (cold start)")
            self.episodes = []

    # ── read helpers (same as store.py) ──
    def training_episodes(self, condition=None):
        eps = [e for e in self.episodes if not e.get("is_validation")]
        if condition:
            eps = [e for e in eps if e.get("condition") == condition]
        return eps

    def validation_episodes(self):
        return [e for e in self.episodes if e.get("is_validation")]

    def score_history_summary(self):
        by_occasion = {}
        for e in self.training_episodes():
            occ = e.get("occasion", "unknown")
            by_occasion.setdefault(occ, []).append(e["score"]["thoughtfulness"])
        return {k: round(sum(v)/len(v), 3) for k, v in by_occasion.items()}

    def top_episodes(self, n=5, condition=None):
        eps = self.training_episodes(condition)
        return sorted(eps, key=lambda e: e["score"]["thoughtfulness"], reverse=True)[:n]

    def bottom_episodes(self, n=5, condition=None):
        """[Round-2 Phase 3] Lowest-thoughtfulness training episodes (parity with store.py)
        so the ACE Reflector can mine failures for 'avoid' lessons. Additive."""
        eps = self.training_episodes(condition)
        return sorted(eps, key=lambda e: e["score"]["thoughtfulness"])[:n]

    def thoughtfulness_series(self, condition=None, validation=False, window=10):
        src = self.validation_episodes() if validation else self.training_episodes(condition)
        vals = [e["score"]["thoughtfulness"] for e in src]
        out = []
        for i in range(len(vals)):
            lo = max(0, i - window + 1)
            out.append(sum(vals[lo:i+1]) / (i - lo + 1))
        return out

    # ── per-round aggregation ──
    # Parity with store.py [Rig fix P0-1 / P0-3]. These were added to the local
    # backend but never ported here, so dashboard.render() — which calls round_series()
    # and validation_series_by_round() — crashed against the Azure backend with
    # AttributeError. app.py always imports THIS Store, so every real /api/run died at
    # chart render. Keep both backends in lockstep.
    def _mean_by_round(self, episodes):
        """{round_index: mean thoughtfulness} → sorted [(round, mean), ...]."""
        by_round = {}
        for e in episodes:
            r = e.get("round_index", -1)
            by_round.setdefault(r, []).append(e["score"]["thoughtfulness"])
        return [(r, sum(v) / len(v)) for r, v in sorted(by_round.items()) if r >= 0]

    def round_series(self, condition):
        """Mean thoughtfulness per round for a training condition (solo|swarm)."""
        return self._mean_by_round(self.training_episodes(condition))

    def validation_series_by_round(self):
        """Mean held-out thoughtfulness per round — the rising generalization line."""
        return self._mean_by_round(self.validation_episodes())

    def save_chart(self, filename: str, png_bytes: bytes):
        """Upload a rendered chart PNG to blob."""
        blob_client = self.client.get_blob_client(container=self.container,
                                                   blob=f"charts/{filename}")
        blob_client.upload_blob(png_bytes, overwrite=True)
        logger.info(f"Saved chart: {filename}")

    def get_chart_url(self, filename: str) -> str:
        """Return a SAS URL for a chart."""
        blob_client = self.client.get_blob_client(container=self.container,
                                                   blob=f"charts/{filename}")
        return blob_client.url
