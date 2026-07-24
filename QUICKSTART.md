# Gandalf Protocol — Quick Start

**A wizard never builds systems late. And Gandalf Protocol does not fail.**

---

## What is Gandalf Protocol?

A self-play learning orchestrator for multi-agent systems. You build:
- **Agents** that propose solutions
- **A World** that simulates scenarios and reacts
- **A Judge** that scores outcomes
- **A Coach** that identifies weak spots

Gandalf coordinates all four, orchestrating learning loops. Agents explore in parallel, share what works, and get smarter together. Overnight.

**First use case:** gift recommendations (Broflo). But Gandalf works for any domain: hiring, content generation, design iteration, customer service, anything where you can simulate and measure success.

---

## Tonight (5 hours)

```bash
# Setup
unzip gandalf-protocol.zip
cd gandalf-protocol
pip install -r requirements.txt

# Verify the loop runs (mock mode, zero cost)
python run.py --ablation --rounds 6

# Charts render. You see three PNGs: learning curve, ablation, validation.
# The swarm curve beats solo. That's the story.
```

Then Violet, Jasper, and Kirk each sharpen their component (see README.md).

---

## For Azure deployment (post-hackathon)

```bash
# Prerequisites: Azure CLI, az login, resource group created
chmod +x deploy.sh
./deploy.sh

# Once deployed, trigger remotely:
curl -X POST "http://gandalf-protocol.westus2.azurecontainers.io:5000/api/run?rounds=6&condition=ablation"
```

Full instructions in `DEPLOY.md`.

---

## The Three Files You Need to Read

1. **`README.md`** — architecture, run commands, who builds what
2. **`DEPLOY.md`** — Azure setup, step-by-step (do this later)
3. **`contracts.py`** — the locked JSON shapes (read out loud in hour 0)

---

## The Core Idea

One solo agent learning is flat. Four agents exploring different strategies, sharing wins, with a coach targeting weak spots? That's a curve that climbs.

The ablation chart proves it. That's why Gandalf wins.

```
solo agent (red)    ← flat, no learning from others
swarm + distill (green) ← climbs, agents share knowledge
```

---

## The Wizard's Guidance

> "A Gandalf is never late, nor is it early. It arrives precisely when the learning is complete."

You have 36 hours to prove it learns. Go.
