# Azure Deployment Guide for Gandalf Protocol

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Azure                                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Container Registry (gandalfacr)                            │
│      ↓                                                       │
│  Container Instances (gandalf-protocol)                     │
│      ├── API port 5000                                      │
│      └── ANTHROPIC_API_KEY, AZURE_STORAGE_CONNECTION_STRING │
│          (from environment variables)                       │
│      ↓                                                       │
│  Blob Storage (gandalfstorage)                              │
│      ├── /gandalf/episodes.json                             │
│      └── /gandalf/charts/*.png                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Azure subscription** — use `az login` to authenticate.
2. **Azure CLI** — `az --version` to confirm installed.
3. **Docker** (local testing only) — `docker --version`.

## Step 1 — Create Azure Resources

```bash
# Create resource group
az group create -n gandalf-rg -l westus2

# Create container registry (stores Docker images)
az acr create -g gandalf-rg -n gandalfacr --sku Basic

# Create storage account (stores episodes + charts)
az storage account create \
  -g gandalf-rg \
  -n gandalfstorage \
  --sku Standard_LRS \
  -l westus2

# Create a blob container
STORAGE_KEY=$(az storage account keys list -n gandalfstorage -g gandalf-rg -o tsv --query '[0].value')
az storage container create \
  -n gandalf \
  --account-name gandalfstorage \
  --account-key "$STORAGE_KEY"
```

## Step 2 — Set Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
# Your Anthropic API key (used by the Claude models)
ANTHROPIC_API_KEY=sk-ant-...

# Azure Storage connection string (auto-generated if you use deploy.sh)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
```

## Step 3 — Deploy

**Option A: Using `deploy.sh` (automated)**

```bash
chmod +x deploy.sh
./deploy.sh
```

This will:
1. Build the Docker image locally.
2. Push to Azure Container Registry.
3. Deploy to Azure Container Instances.
4. Output the public endpoint.

**Option B: Manual steps (if deploy.sh fails)**

```bash
# Build and push to ACR
az acr build -r giftgymacr -t giftgym:latest .

# Get storage connection string
STORAGE_KEY=$(az storage account keys list -n giftgymstorage -g giftgym-rg -o tsv --query '[0].value')
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=giftgymstorage;AccountKey=${STORAGE_KEY};EndpointSuffix=core.windows.net"

# Deploy to ACI
az container create \
  --resource-group giftgym-rg \
  --name giftgym-run \
  --image giftgymacr.azurecr.io/giftgym:latest \
  --registry-login-server giftgymacr.azurecr.io \
  --registry-username $(az acr credential show -n giftgymacr -o tsv --query 'username') \
  --registry-password $(az acr credential show -n giftgymacr -o tsv --query 'passwords[0].value') \
  --environment-variables \
    ANTHROPIC_API_KEY="sk-ant-..." \
    AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING" \
  --port 5000 \
  --cpu 2 --memory 2
```

## Step 4 — Get the Endpoint

```bash
az container show -g giftgym-rg -n giftgym-run --query ipAddress.fqdn -o tsv
```

Returns something like: `giftgym-run.westus2.azurecontainers.io`

## Step 5 — Test Locally (Before Deploying)

```bash
# Install dependencies
pip install -r requirements.txt

# Set env vars (optional — mock mode if ANTHROPIC_API_KEY is missing)
export ANTHROPIC_API_KEY=sk-ant-...
export AZURE_STORAGE_CONNECTION_STRING=...

# Run the Flask app
python app.py
# Check health: curl http://localhost:5000/health
```

Or with Docker:
```bash
docker build -t giftgym:local .
docker run -p 5000:5000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e AZURE_STORAGE_CONNECTION_STRING=... \
  giftgym:local
```

## Step 6 — Trigger a Run

Once deployed, hit the `/api/run` endpoint:

```bash
ENDPOINT="http://giftgym-run.westus2.azurecontainers.io:5000"

# Full ablation (solo vs swarm) — 6 rounds each
curl -X POST "${ENDPOINT}/api/run?rounds=6&condition=ablation"

# Returns:
# {
#   "status": "ok",
#   "rounds": 6,
#   "condition": "ablation",
#   "total_episodes": 432,
#   "llm_calls": 578,
#   "charts": ["https://giftgymstorage.blob.core.windows.net/giftgym/charts/chart_1_learning_curve.png", ...]
# }
```

## Step 7 — Retrieve Results

```bash
# Get charts
curl "${ENDPOINT}/api/charts"

# Get episodes
curl "${ENDPOINT}/api/results" > episodes.json

# Get summary
curl "${ENDPOINT}/api/summary"
```

## Monitoring & Logs

```bash
# View container logs
az container logs -g giftgym-rg -n giftgym-run --tail 100

# Follow logs in real-time
az container logs -g giftgym-rg -n giftgym-run -f
```

## Cost Notes

- **Container Registry** — ~$5/month (Basic tier).
- **Blob Storage** — ~$0.50/month for typical usage.
- **Container Instances** — ~$0.50 per run (2 CPU, 2GB mem, ~5 min).
- **Anthropic API** — ~$0.50–$2 per experiment depending on rounds/personas.

Total: ~$10/month + API costs.

## Cleanup

```bash
# Delete the resource group (removes everything)
az group delete -n gandalf-rg
```

## Integration with ops-platform

Once this is running, Kirk can swap the store to use `ops-platform` for job scheduling + audit logging:

```python
# Instead of:
from store import Store  # or store_azure.Store

# Use:
from ops_platform import Store

# Everything else stays identical — same method signatures.
```

The Gandalf Protocol stays the same; just the persistence layer changes.
