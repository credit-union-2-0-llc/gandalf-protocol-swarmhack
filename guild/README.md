# Guild control panel for Gandalf

Guild is the **dashboard + control panel** over the Gandalf self-learning engine. It does
not run Gandalf's Python — it drives Gandalf's live REST API from the outside, so you start
runs, watch them, track spend, and govern the agent from Guild instead of curling Azure.

```
Operator / schedule / API  ─trigger─▶  Guild agent  ─tools─▶  Guild integration proxy  ─HTTPS─▶  Gandalf API (Azure)
```

## What's in this folder
- `openapi.yaml` — the Gandalf REST API described as OpenAPI 3.1. Guild imports this to build
  the integration; each operation becomes an agent tool (`startRun`, `getRunStatus`,
  `getSummary`, `getResults`, `listCharts`, `healthCheck`).
- `agent.ts` — the control-panel agent (`@guildai/agents-sdk` `llmAgent`) that starts a run,
  polls to completion, and reports whether it learned.

## Prerequisites (only you can do these)
1. A **Guild.ai account** with an org + workspace.
2. The **Guild CLI** installed and logged in (`guild login`).
3. A **trigger API key** (org settings) if you want to start runs programmatically.

Everything below is inert until these exist — that's the one part I can't build for you.

## Setup (once you have the account)
Org `cu2` / workspace `gandalf` are created. Do these **in order** — the integration must
exist before the agent (the agent imports the integration's generated tools), and a trigger
must pick an existing agent.

0. **CLI login + select the workspace**:
   ```bash
   npm i -g @guildai/cli
   guild auth login && guild auth status
   guild workspace select      # pick cu2/gandalf
   ```
1. **Create the integration + import the spec + build/publish a version FIRST** (public
   Azure URL, no tunnel needed). `--base-url` and `--auth-scheme` are both required, and
   `--auth-scheme` only accepts `api-key` or `oauth` (there is **no** `none` — see Auth):
   ```bash
   guild integration create gandalf \
     --base-url https://ca-gandalf-protocol.wittyflower-1831f2a2.westus2.azurecontainerapps.io \
     --auth-scheme api-key \
     --description "Gandalf control panel API"
   guild integration operation create cu2~gandalf --openapi ./guild/openapi.yaml
   guild integration version build   cu2~gandalf --version-number 1.0.0
   guild integration version publish cu2~gandalf --version-number 1.0.0
   ```
   The build+publish steps are what make `@guildai-services/cu2~gandalf` resolve for the agent.
2. **Scaffold, save, and publish the agent** (`agent.ts` already targets
   `@guildai-services/cu2~gandalf`):
   ```bash
   guild agent init --name gandalf --template LLM     # skip if already scaffolded
   guild agent save --message "First version" --wait --publish
   ```
   Verify it appears in the `gandalf` workspace's Agents tab.
3. **Add a trigger** (only after the agent is published — a trigger targets a specific agent):
   - *Scheduled nightly run*:
     ```bash
     guild trigger create --type time --frequency DAILY --time 02:00 --agent cu2~gandalf
     ```
   - *On-demand via API*: create a trigger API key in the web UI, then start a run with
     `POST /api/workspaces/cu2/gandalf/sessions` (Basic Auth `key_id:key_secret`),
     body `{"session_type":"api_trigger","agent_input":{"type":"text","text":"run swarm, 3 rounds"}}`.
4. **Drive it**: open a **Session** with the agent in the Guild UI and say "start a swarm run,
   3 rounds." Watch progress; Guild's Insights tab shows token spend; Audit logs record actions.

## Auth (required decision — `none` is not an option)
`--auth-scheme` must be `api-key` or `oauth`, so the integration always sends *some* key.
Gandalf currently validates none, so two paths:
- **Demo shortcut (0 code):** use `--auth-scheme api-key` with any placeholder key. Gandalf
  ignores it, the integration works immediately — but the endpoint stays open to the world.
- **Real control point (small code change):** add an `X-API-Key` header check to Gandalf's
  `app.py` (env-gated so it's backward-compatible), set the key as a Container App secret,
  uncomment the `securitySchemes` block in `openapi.yaml`, and register the key as a Guild
  **credential** with a **credential policy** limiting the agent to read/run ops
  (`startRun`/`getRunStatus`/`getSummary`).

## Long-run note
A run takes minutes; Guild caps synchronous steps per task. The `multi-turn` agent here polls
across turns, which is fine for short runs (rounds ≤ 3). For longer runs, prefer the
**split pattern**: one agent calls `startRun` and returns; a **scheduled trigger** checks
`getRunStatus` every few minutes and reports when done. Convert to a self-managed state agent
if you need explicit resumption across a long job.

## Verify
- Spec is valid: `python -c "import yaml,sys; yaml.safe_load(open('guild/openapi.yaml')); print('openapi ok')"`.
- Gandalf reachable: `curl -s $BASE/health` → `{"status":"ok"}`.
- End-to-end (after Guild setup): open a session, "start a swarm run 3 rounds", confirm the
  agent reports climbing `training_episodes` and a non-empty `score_history`.
