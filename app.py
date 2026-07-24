"""Gandalf Protocol — Flask API + browsable dashboard [OWNER: Kirk]

Routes:
  GET  /                       → landing page (status, Run buttons, live charts)
  POST /api/run?condition=&rounds=  → START a run ASYNC (returns immediately); poll /api/status
  GET  /api/status             → run state + episode counts (poll this)
  GET  /api/summary            → score history + stats
  GET  /api/results            → episodes.json
  GET  /api/charts             → chart image URLs (served by THIS app, not raw blob)
  GET  /chart/<name>           → stream a chart PNG (from blob), browser-viewable
  GET  /fantasy                → self-contained fantasy-domain demo UI (the learning-curve climb)
  GET  /health                 → liveness

Runs are ASYNC: a synchronous run exceeds the Container Apps ~4-min ingress timeout, so
POST /api/run spawns a background thread and returns a job id. The UI polls /api/status.
"""
from flask import Flask, jsonify, request, Response, render_template_string
import os, threading, io, json
import config, llm
from store_azure import Store
from agents import world, signal
from agents.swarm import GiftAgent
from contracts import Playbook
import runner, dashboard

app = Flask(__name__)
store = None
_lock = threading.Lock()
# single-run-at-a-time state the UI polls
RUN = {"running": False, "condition": None, "rounds": 0, "phase": "idle",
       "started_calls": 0, "error": None, "done_at": None}


def _get_store():
    global store
    if store is None:
        store = Store()
        store.load()
    return store


@app.before_request
def _init():
    _get_store()


# ────────────────────────────── the run worker ──────────────────────────────
def _run_worker(condition, rounds):
    """Runs in a daemon thread so the HTTP request can return immediately."""
    try:
        st = _get_store()
        st.episodes = []          # fresh experiment (overwrites blob on first flush)
        last_playbook = None
        # "ablation" = solo vs swarm (is a swarm better?). "abtest" = swarm WITH vs WITHOUT
        # distillation (does the shared playbook itself help? — Warren's clean test, run
        # server-side so laptop sleep can't kill it).
        # "full" banks every chart in one run: solo (ablation baseline) + swarm (learning +
        # playbook) + swarm_nodistill (A/B OFF arm). One run → charts 1–4 + A/B verdict + playbook.
        conditions = {"ablation": ["solo", "swarm"],
                      "abtest": ["swarm", "swarm_nodistill"],
                      "full": ["solo", "swarm", "swarm_nodistill"]}.get(condition, [condition])
        for cond in conditions:
            RUN["phase"] = f"running {cond}"
            held = world.load_seed_personas()
            for p in held:
                p.is_validation = True
            pb = runner.run_experiment(st, rounds, cond, config.PERSONAS_PER_ROUND,
                                       held_out=held if cond in ("swarm", "solo", "swarm_nodistill") else None)
            if cond == "swarm":      # only the distilled swarm playbook is worth publishing
                last_playbook = pb
        RUN["phase"] = "rendering charts"
        paths = dashboard.render(st, out_prefix="/tmp/chart")
        for path in paths:
            with open(path, "rb") as f:
                st.save_chart(os.path.basename(path), f.read())
        # Publish the learned Playbook artifact (gated — apps only inject it when ready_for_prod).
        if last_playbook is not None:
            try:
                from integration import playbook_artifact as pa
                art = pa.build_artifact(st, last_playbook, judge_kappa=_grade_kappa()[0], domain="gifts")
                pa.publish_to_blob(st, art)
                RUN["playbook_version"], RUN["playbook_ready"] = art["version"], art["ready_for_prod"]
            except Exception as e:   # never let publish break a run
                print(f"[playbook publish] skipped: {e}")
        RUN["phase"] = "done"
    except Exception as e:            # noqa: BLE001 — surface any run error to the UI
        RUN["error"] = str(e)
        RUN["phase"] = "error"
    finally:
        RUN["running"] = False
        RUN["done_at"] = llm.calls_made()


@app.route("/api/run", methods=["POST"])
def start_run():
    condition = request.args.get("condition", default="swarm")
    rounds = request.args.get("rounds", default=6, type=int)
    with _lock:
        if RUN["running"]:
            return jsonify({"status": "busy", "phase": RUN["phase"],
                            "message": "a run is already in progress"}), 409
        RUN.update({"running": True, "condition": condition, "rounds": rounds,
                    "phase": "starting", "error": None, "done_at": None})
    threading.Thread(target=_run_worker, args=(condition, rounds), daemon=True).start()
    return jsonify({"status": "started", "condition": condition, "rounds": rounds,
                    "poll": "/api/status"}), 202


@app.route("/api/playbook", methods=["GET"])
def playbook():
    """The cross-app contract: return the gated Playbook artifact for a domain.
    Broflo/Warren/Jasper apps fetch this (see integration/BROFLO-DROPIN.md). Fails closed
    with a safe shape so a consuming app never breaks if none is published yet."""
    domain = request.args.get("domain", "gifts")
    st = _get_store()
    try:
        blob = st.client.get_blob_client(container=st.container, blob=f"playbook/{domain}/latest.json")
        return Response(blob.download_blob().readall(), mimetype="application/json")
    except Exception:
        return jsonify({"schema": "gandalf.playbook/v1", "domain": domain,
                        "ready_for_prod": False, "lessons": [],
                        "gate_reasons": [f"no playbook published for '{domain}' yet"]}), 404


@app.route("/api/status", methods=["GET"])
def status():
    st = _get_store()
    return jsonify({
        "running": RUN["running"], "phase": RUN["phase"],
        "condition": RUN["condition"], "rounds": RUN["rounds"],
        "error": RUN["error"], "llm_calls": llm.calls_made(),
        "total_episodes": len(st.episodes),
        "training_episodes": len(st.training_episodes()),
        "validation_episodes": len(st.validation_episodes()),
    })


@app.route("/api/summary", methods=["GET"])
def summary():
    st = _get_store()
    return jsonify({
        "total_episodes": len(st.episodes),
        "training_episodes": len(st.training_episodes()),
        "validation_episodes": len(st.validation_episodes()),
        "score_history": st.score_history_summary(),
    })


@app.route("/api/results", methods=["GET"])
def results():
    st = _get_store()
    return jsonify({"episodes": st.episodes, "count": len(st.episodes)})


@app.route("/api/charts", methods=["GET"])
def charts():
    # Served by THIS app (browser-viewable) — not raw private-blob URLs.
    names = ["1_learning_curve", "2_ablation", "3_validation", "4_abtest"]
    return jsonify({"charts": [{"name": n, "url": f"/chart/{n}"} for n in names]})


@app.route("/chart/<name>", methods=["GET"])
def chart(name):
    """Stream a chart PNG from blob so it's viewable without SAS / public access."""
    fname = f"chart_{name}.png" if not name.endswith(".png") else name
    st = _get_store()
    try:
        blob = st.client.get_blob_client(container=st.container, blob=f"charts/{fname}")
        data = blob.download_blob().readall()
        return Response(data, mimetype="image/png")
    except Exception:
        return jsonify({"error": f"chart '{fname}' not found — run an experiment first"}), 404


# ── flywheel demo: the SAME gift agent, WITH vs WITHOUT the learned playbook ──
def _current_playbook():
    """Load the published playbook lessons into a Playbook (empty if none published yet)."""
    pb = Playbook()
    st = _get_store()
    try:
        blob = st.client.get_blob_client(container=st.container, blob="playbook/gifts/latest.json")
        for l in json.loads(blob.download_blob().readall()).get("lessons", []):
            pb.add(l["text"], l.get("source_episode_ids", []))
    except Exception:
        pass
    return pb


@app.route("/api/suggest", methods=["GET"])
def suggest():
    """One recipient, two proposals: WITHOUT vs WITH the learned playbook. The flywheel, visible.

    Each pair is TWO live Claude calls (~1 min), so results are cached to blob keyed by the
    current playbook version — the demo loads instantly after the first view, and a new
    published playbook (new version) transparently invalidates the cache. `?fresh=1` bypasses.
    """
    seeds = world.load_seed_personas()
    if not seeds:
        return jsonify({"error": "no seed personas"}), 404
    idx = request.args.get("i", default=0, type=int) % len(seeds)
    persona, pb = seeds[idx], _current_playbook()
    st = _get_store()
    cache_blob = f"suggest_cache/v{pb.version}/{idx}.json"

    if request.args.get("fresh") != "1":
        try:
            cached = st.client.get_blob_client(
                container=st.container, blob=cache_blob).download_blob().readall()
            return Response(cached, mimetype="application/json")
        except Exception:
            pass  # cold cache — generate below

    agent = GiftAgent("bold-experiences")
    try:
        without = agent.propose(persona.profile, Playbook(), config.GIFTS_PER_PROPOSAL)
        withpb = agent.propose(persona.profile, pb, config.GIFTS_PER_PROPOSAL)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    payload = {"recipient_index": idx, "recipient_count": len(seeds),
               "profile": persona.profile, "occasion": persona.occasion, "budget": persona.budget,
               "playbook_version": pb.version, "lessons": [e["text"] for e in pb.entries],
               "without_playbook": without.gifts, "with_playbook": withpb.gifts}
    data = json.dumps(payload)
    try:
        st.client.get_blob_client(
            container=st.container, blob=cache_blob).upload_blob(data, overwrite=True)
    except Exception:
        pass  # cache write is best-effort; never fail the request on it
    return Response(data, mimetype="application/json")


_DEMO = """<!doctype html><meta charset=utf-8><title>Gandalf — the flywheel</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{--ink:#0e1116;--panel:#151b23;--line:#252d38;--fg:#e7ebf1;--muted:#9aa6b6;--swarm:#33cc88;--solo:#e0645f}
 @media(prefers-color-scheme:light){:root{--ink:#f7f8fa;--panel:#fff;--line:#e2e6ec;--fg:#141a22;--muted:#4d5766;--swarm:#1c9c63;--solo:#c94a45}}
 body{margin:0;background:var(--ink);color:var(--fg);font:16px/1.6 system-ui,-apple-system,sans-serif}
 .wrap{max-width:1000px;margin:0 auto;padding:28px}
 a{color:var(--swarm);text-decoration:none}
 h1{font-size:26px;letter-spacing:-.02em;margin:0 0 4px} .sub{color:var(--muted);margin:0 0 18px}
 button{background:var(--swarm);color:#04120b;border:0;border-radius:8px;padding:10px 16px;font-weight:700;cursor:pointer}
 .recipient{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:14px} @media(max-width:720px){.cols{grid-template-columns:1fr}}
 .col{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
 .col.with{border-color:color-mix(in srgb,var(--swarm) 45%,var(--line))}
 .col h3{margin:0 0 10px;font-size:15px;display:flex;gap:8px;align-items:center}
 .tag{font:600 11px/1 ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;padding:3px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
 .tag.g{color:var(--swarm);border-color:color-mix(in srgb,var(--swarm) 45%,var(--line))}
 .gift{border-top:1px solid var(--line);padding:10px 0} .gift:first-of-type{border-top:0}
 .gift b{display:block} .gift small{color:var(--muted)}
 .lessons{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:14px}
 .lessons li{color:var(--muted)} .muted{color:var(--muted)} .mono{font-family:ui-monospace,monospace}
</style>
<div class=wrap>
 <p class=mono style="margin:0"><a href="/">← dashboard</a> &nbsp; <a href="/guide">how it works</a></p>
 <h1>🎁 The flywheel — the same agent, but it learned</h1>
 <p class=sub>One recipient. Left: the gift agent with an empty playbook. Right: the <b>same agent</b> with the lessons the swarm distilled. This is exactly what a live product (Broflo) gets by wiring in Gandalf.</p>
 <button id=prev onclick="go(cur-1)" style="background:transparent;color:var(--fg);border:1px solid var(--line)">← Previous</button>
 <button id=cur onclick="go(cur)">Suggest / re-run</button>
 <button id=next onclick="go(cur+1)" style="background:transparent;color:var(--fg);border:1px solid var(--line)">Next →</button>
 <div id=out></div>
</div>
<script>
let cur=0;
function gifts(list){return (list||[]).map(g=>`<div class=gift><b>${g.name||'—'}</b><small>${g.category||''} · $${g.price||'?'} — ${g.reasoning||''}</small></div>`).join('')||'<p class=muted>—</p>';}
function busy(b){['prev','cur','next'].forEach(id=>document.getElementById(id).disabled=b);}
async function go(i){
  const out=document.getElementById('out'); busy(true);
  out.innerHTML='<p class=muted>thinking… (two real proposals)</p>';
  try{
    const d=await (await fetch('/api/suggest?i='+i)).json();
    if(d.error){out.innerHTML='<p class=muted>'+d.error+'</p>';busy(false);return;}
    cur=d.recipient_index;   // normalize (server wraps negative/overflow) so prev/next stay in range
    const p=d.profile||{}; const ints=(p.interests||[]).map(x=>x.value||x).join(', ');
    const cons=(p.constraints||[]).map(x=>x.value||x).join(', ');
    out.innerHTML=`
      <div class=recipient><b>Recipient ${d.recipient_index+1}/${d.recipient_count}</b> · ${d.occasion} · $${(d.budget||{}).min}–$${(d.budget||{}).max}
        <div class=muted>likes: ${ints||'—'} ${cons?(' · constraints: '+cons):''}</div></div>
      <div class=cols>
        <div class=col><h3><span class=tag>without playbook</span></h3>${gifts(d.without_playbook)}</div>
        <div class="col with"><h3><span class="tag g">with playbook · v${d.playbook_version}</span></h3>${gifts(d.with_playbook)}</div>
      </div>
      <div class=lessons><b>Lessons the swarm learned (injected on the right):</b>
        ${d.lessons.length?('<ul>'+d.lessons.map(l=>'<li>'+l+'</li>').join('')+'</ul>'):'<p class=muted>No lessons yet — run an experiment on the dashboard, then this side gets smarter.</p>'}</div>`;
    busy(false);
  }catch(e){out.innerHTML='<p class=muted>error: '+e+'</p>';busy(false);}
}
// Deep-link support (?i=N); default to recipient 2 — the hero contrast where the
// empty-playbook agent goes solo and the learned agent makes it shareable-with-friends.
const _qi = new URLSearchParams(location.search).get('i');
go(_qi !== null ? (parseInt(_qi, 10) || 0) : 2);
</script>"""


@app.route("/demo", methods=["GET"])
def demo():
    return render_template_string(_DEMO)


# ── before/after Actian: which playbook lessons keyword-match picks vs Actian semantic picks ──
def _semantic_local(entries, query, k):
    """Honest fallback if the Actian DB is unreachable: rank the SAME lessons by cosine over
    the SAME MiniLM embeddings locally. Still shows semantic-vs-lexical; labeled accurately."""
    import retrieval
    qv = retrieval._embed(query)
    scored = sorted(entries,
                    key=lambda e: sum(x * y for x, y in zip(retrieval._embed(e["text"]), qv)),
                    reverse=True)   # unit vectors → dot product == cosine
    return scored[:k]


@app.route("/api/retrieval-compare", methods=["GET"])
def retrieval_compare():
    """BEFORE/AFTER ACTIAN, no LLM (instant, safe to show live). For one recipient, contrast
    the lessons a keyword ranker (BEFORE) surfaces vs Actian VectorAI semantic top-k (AFTER).
    `semantic_only` = lessons Actian finds that share no keywords with the persona — the RAG win."""
    import retrieval
    seeds = world.load_seed_personas()
    if not seeds:
        return jsonify({"error": "no seed personas"}), 404
    idx = request.args.get("i", default=0, type=int) % len(seeds)
    k = request.args.get("k", default=5, type=int)
    persona = seeds[idx]
    entries = list(_current_playbook().entries or [])
    query = retrieval._profile_text(persona.profile)

    before = [e["text"] for e in retrieval._lexical_topk(entries, query, k)]
    actian_live = True
    try:
        after = [e["text"] for e in retrieval._actian_topk(entries, query, k)]
    except Exception:            # DB unreachable → honest local semantic fallback
        actian_live = False
        after = [e["text"] for e in _semantic_local(entries, query, k)]

    before_set = set(before)
    return jsonify({
        "recipient_index": idx, "recipient_count": len(seeds),
        "occasion": persona.occasion, "query": query, "lesson_count": len(entries),
        "before_lexical": before, "after_actian": after,
        "semantic_only": [t for t in after if t not in before_set],
        "actian_live": actian_live,
    })


_ACTIAN_PAGE = """<!doctype html><meta charset=utf-8><title>Gandalf — before/after Actian</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{--ink:#0e1116;--panel:#151b23;--line:#252d38;--fg:#e7ebf1;--muted:#9aa6b6;--swarm:#33cc88;--solo:#e0645f;--warn:#e0a95f}
 @media(prefers-color-scheme:light){:root{--ink:#f7f8fa;--panel:#fff;--line:#e2e6ec;--fg:#141a22;--muted:#4d5766;--swarm:#1c9c63;--solo:#c94a45}}
 body{margin:0;background:var(--ink);color:var(--fg);font:16px/1.6 system-ui,-apple-system,sans-serif}
 .wrap{max-width:1000px;margin:0 auto;padding:28px}
 a{color:var(--swarm);text-decoration:none}
 h1{font-size:26px;letter-spacing:-.02em;margin:0 0 4px} .sub{color:var(--muted);margin:0 0 18px}
 button{background:var(--swarm);color:#04120b;border:0;border-radius:8px;padding:10px 16px;font-weight:700;cursor:pointer}
 .q{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:720px){.cols{grid-template-columns:1fr}}
 .col{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
 .col.after{border-color:color-mix(in srgb,var(--swarm) 45%,var(--line))}
 .col h3{margin:0 0 10px;font-size:15px} .lesson{border-top:1px solid var(--line);padding:9px 0} .lesson:first-of-type{border-top:0}
 .badge{font:600 11px/1 ui-monospace,monospace;letter-spacing:.06em;text-transform:uppercase;padding:3px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
 .badge.g{color:var(--swarm);border-color:color-mix(in srgb,var(--swarm) 45%,var(--line))}
 .badge.w{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 45%,var(--line))}
 .hit{color:var(--swarm);font-weight:700} .muted{color:var(--muted)} .mono{font-family:ui-monospace,monospace}
</style>
<div class=wrap>
 <p class=mono style="margin:0"><a href="/">← dashboard</a> &nbsp; <a href="/demo">flywheel</a> &nbsp; <a href="/guide">how it works</a></p>
 <h1>🧭 Before / after Actian</h1>
 <p class=sub>Same recipient, same learned lessons. <b>Left</b>: a keyword ranker picks lessons.
 <b>Right</b>: <b>Actian VectorAI DB</b> does semantic top-k over MiniLM embeddings — it surfaces
 lessons that <i>mean</i> the same thing even with zero shared words. Highlighted = Actian-only finds.</p>
 <span id=live class=badge>checking Actian…</span>
 <div style="margin:12px 0">
   <button id=prev onclick="go(cur-1)" style="background:transparent;color:var(--fg);border:1px solid var(--line)">← Previous</button>
   <button id=next onclick="go(cur+1)" style="background:transparent;color:var(--fg);border:1px solid var(--line)">Next recipient →</button>
 </div>
 <div id=out></div>
</div>
<script>
let cur=0;
function items(list,hi){return (list||[]).map(t=>`<div class="lesson ${hi&&hi.has(t)?'hit':''}">${hi&&hi.has(t)?'★ ':''}${t}</div>`).join('')||'<p class=muted>—</p>';}
function busy(b){['prev','next'].forEach(id=>document.getElementById(id).disabled=b);}
async function go(i){
  busy(true); const out=document.getElementById('out'); out.innerHTML='<p class=muted>retrieving…</p>';
  try{
    const d=await (await fetch('/api/retrieval-compare?i='+i)).json();
    if(d.error){out.innerHTML='<p class=muted>'+d.error+'</p>';busy(false);return;}
    cur=d.recipient_index;
    const lv=document.getElementById('live');
    if(d.actian_live){lv.className='badge g';lv.textContent='Actian VectorAI DB — LIVE ✅';}
    else{lv.className='badge w';lv.textContent='Actian unreachable — local MiniLM semantic fallback';}
    const hi=new Set(d.semantic_only||[]);
    out.innerHTML=`
      <div class=q><b>Recipient ${d.recipient_index+1}/${d.recipient_count}</b> · ${d.occasion}
        · ${d.lesson_count} lessons in the playbook
        <div class=muted>query built from profile: <span class=mono>${d.query||'—'}</span></div></div>
      <div class=cols>
        <div class=col><h3><span class=badge>before · keyword match</span></h3>${items(d.before_lexical)}</div>
        <div class="col after"><h3><span class="badge g">after · Actian semantic</span></h3>${items(d.after_actian,hi)}</div>
      </div>
      <p class=muted style="margin-top:12px">★ = surfaced by Actian's semantic search but missed by keyword overlap.
      This is the lesson the learned agent gets to use that it otherwise wouldn't.</p>`;
    busy(false);
  }catch(e){out.innerHTML='<p class=muted>error: '+e+'</p>';busy(false);}
}
go(0);
</script>"""


@app.route("/actian", methods=["GET"])
def actian_view():
    return render_template_string(_ACTIAN_PAGE)


def _grade_kappa():
    """Grade the judge server-side: Cohen's κ vs the gold set + worst bad-gift score.
    Runs on the container's gateway key — NO local runs, no key handed out. (None, None)
    if MOCK or no gold set."""
    gp = os.path.join(os.path.dirname(__file__), "evals", "gold_set.json")
    try:
        gold = json.load(open(gp)).get("examples", [])
    except Exception:
        return (None, None)
    if config.MOCK or not gold:
        return (None, None)
    from contracts import Persona, GiftProposal, Reaction
    pairs, worst_bad = [], 0.0
    for ex in gold:
        persona = Persona(profile=ex.get("profile", {}), hidden_truth=ex["hidden_truth"],
                          occasion=ex.get("occasion", "birthday"), budget=ex.get("budget", {"min": 0, "max": 100}))
        prop = GiftProposal(agent_id="eval", strategy_tag="eval", gifts=ex["gifts"])
        react = Reaction(proposal_id=prop.proposal_id, persona_id=persona.persona_id,
                         verdict=ex.get("verdict", "meh"), quote="")
        t = signal.judge(persona, prop, react).thoughtfulness
        pairs.append((t >= 0.6, bool(ex["label_good"])))
        if not ex["label_good"]:
            worst_bad = max(worst_bad, t)
    n = len(pairs)
    agree = sum(1 for p, l in pairs if p == l) / n
    pp = sum(p for p, _ in pairs) / n; pl = sum(l for _, l in pairs) / n
    chance = pp * pl + (1 - pp) * (1 - pl)
    k = (agree - chance) / (1 - chance) if chance < 1 else 1.0
    return (round(k, 3), round(worst_bad, 3))


@app.route("/api/eval", methods=["GET"])
def api_eval():
    """Grade the judge on the server (no local runs). Returns κ + worst bad-gift + gates."""
    if config.MOCK:
        return jsonify({"error": "server has no gateway key (MOCK) — eval needs the deployed key"}), 400
    k, worst = _grade_kappa()
    if k is None:
        return jsonify({"error": "no gold set to grade against"}), 404
    return jsonify({"kappa": k, "worst_bad_gift": worst,
                    "kappa_ok": k >= 0.6, "constraint_ok": worst < 0.4,
                    "verdict": "judge calibrated ✓" if (k >= 0.6 and worst < 0.4) else "needs tuning"})


@app.route("/api/actian", methods=["GET"])
def api_actian():
    """Server-side probe: can the app actually reach the Actian VectorAI DB? Does a real
    connect → create collection → upsert 1 → search 1 round-trip and reports ok/error, so
    we don't have to burn a full run to discover whether retrieval is live or falling back
    to the lexical ranker. Also confirms the gRPC-over-ACA-ingress connection string."""
    import retrieval
    conn = os.environ.get("ACTIAN_CONNECTION_STRING") or os.environ.get("ACTIAN_DB_URL")
    if not conn:
        return jsonify({"enabled": False, "reason": "ACTIAN_CONNECTION_STRING not set"}), 200
    try:
        from actian_vectorai import VectorParams, Distance, PointStruct
        client = retrieval._actian_client()
        try:
            client.collections.create("gandalf_probe",
                vectors_config=VectorParams(size=retrieval.EMBED_DIM, distance=Distance.Cosine))
        except Exception:
            pass  # already exists
        vec = retrieval._embed("a thoughtful birthday gift for a coffee lover")
        client.points.upsert("gandalf_probe",
            [PointStruct(id=1, vector=vec, payload={"text": "probe"})])
        hits = client.points.search("gandalf_probe", vector=vec, limit=1)
        return jsonify({"enabled": True, "connection": conn, "reachable": True,
                        "round_trip": "connect→create→upsert→search OK",
                        "hits": len(list(hits)),
                        "verdict": "Actian semantic retrieval is LIVE ✅"}), 200
    except Exception as e:
        return jsonify({"enabled": True, "connection": conn, "reachable": False,
                        "error": f"{type(e).__name__}: {e}",
                        "verdict": "unreachable — app falls back to lexical ranker (safe)"}), 200


@app.route("/api/abtest", methods=["GET"])
def api_abtest():
    """The distillation proof, server-side. Compares the swarm WITH the shared playbook
    (condition 'swarm') vs the same swarm WITHOUT it ('swarm_nodistill'), per round.
    Reports overall Δ and a later-rounds Δ (where learning should have accumulated) so a
    noisy early round can't swamp the signal. Run POST /api/run?condition=abtest first."""
    st = _get_store()
    on = st.round_series("swarm")            # [(round, mean_thoughtfulness), ...]
    off = st.round_series("swarm_nodistill")
    if not on or not off:
        return jsonify({"error": "no A/B data — POST /api/run?condition=abtest first",
                        "have_on": bool(on), "have_off": bool(off)}), 404

    def _mean(series):
        return round(sum(v for _, v in series) / len(series), 4)

    def _late(series):                        # second half of rounds
        half = series[len(series) // 2:] or series
        return round(sum(v for _, v in half) / len(half), 4)

    on_all, off_all = _mean(on), _mean(off)
    on_late, off_late = _late(on), _late(off)
    d_all, d_late = round(on_all - off_all, 4), round(on_late - off_late, 4)
    # verdict keys off the later-rounds delta (the accumulated-learning window)
    if d_late > 0.03:
        verdict = "distillation HELPS ✅ — the shared playbook lifts the swarm"
    elif d_late < -0.03:
        verdict = "distillation HURTS ❌ — the playbook is dragging the swarm down"
    else:
        verdict = "NEUTRAL — within noise; no clear distillation effect"
    return jsonify({
        "distill_on":  {"overall": on_all,  "later_rounds": on_late,  "series": on},
        "distill_off": {"overall": off_all, "later_rounds": off_late, "series": off},
        "delta_overall": d_all, "delta_later_rounds": d_late,
        "verdict": verdict,
    })


@app.route("/guide", methods=["GET"])
def guide():
    """The human explainer — lives in production so it's always findable at <app>/guide."""
    try:
        with open(os.path.join(os.path.dirname(__file__), "guide.html"), encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "guide unavailable", 404


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ────────────────────────── fantasy demo (Warren) ──────────────────────────
# The fantasy domain's demo UI is a self-contained page (data inlined by
# fantasy/export_demo.py). It's authored in Artifact-content form — no
# <head>/<body> of its own — so we wrap it in a minimal skeleton to serve it
# standalone. Read at request time so a redeploy of the file needs no restart.
_FANTASY_UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fantasy", "ui.html")


@app.route("/fantasy", methods=["GET"])
def fantasy_demo():
    try:
        with open(_FANTASY_UI, encoding="utf-8") as f:
            body = f.read()
    except FileNotFoundError:
        return jsonify({"error": "fantasy demo UI not found — fantasy/ui.html is missing"}), 404
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "</head><body>" + body + "</body></html>")
    return Response(page, mimetype="text/html")


# ────────────────────────────── landing page ──────────────────────────────
_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Gandalf Protocol</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1116;color:#e6e8ef}
 .wrap{max-width:920px;margin:0 auto;padding:28px}
 h1{font-size:22px;margin:0 0 4px} .sub{color:#8b90a0;margin:0 0 20px}
 .row{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}
 button{background:#2a7;color:#04120b;border:0;border-radius:8px;padding:10px 16px;font-weight:600;cursor:pointer}
 button.alt{background:#37c;color:#fff} button:disabled{opacity:.5;cursor:default}
 .card{background:#171a22;border:1px solid #232838;border-radius:12px;padding:16px;margin:14px 0}
 .status{font-weight:600} .muted{color:#8b90a0}
 img{width:100%;border-radius:8px;background:#fff;margin-top:8px}
 code{background:#232838;padding:2px 6px;border-radius:5px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
</style></head><body><div class=wrap>
 <h1>🧙 Gandalf Protocol</h1>
 <p class=sub>Self-evolving multi-agent gift-learning loop — live, gateway-wired, server-side.</p>
 <p style="margin:-8px 0 16px;font-weight:600"><a href="/demo" style="color:var(--swarm,#2a7)">🎁 Flywheel demo (with vs without learning)</a> &nbsp;·&nbsp; <a href="/guide" style="color:var(--signal,#37c)">📖 How it works</a> &nbsp;·&nbsp; <a href="/fantasy" style="color:#31b57f">🏈 Fantasy domain — the climb</a> &nbsp;·&nbsp; <a href="/actian" style="color:#e0a95f">🧭 Before/after Actian</a></p>
 <div class=row>
   <button id=swarm onclick="run('swarm',3)">Run swarm (3 rounds)</button>
   <button id=abl class=alt onclick="run('ablation',3)">Run ablation (solo vs swarm)</button>
   <button id=abt class=alt onclick="run('abtest',6)">Run A/B (distill on vs off)</button>
   <span id=st class="status muted" style="align-self:center">loading…</span>
 </div>
 <div class=card><b>Status</b><div id=detail class=muted>—</div></div>
 <div class=card><b>Score history</b> <span class=muted>(by occasion)</span><div id=scores class=muted>run an experiment to populate</div></div>
 <div class=card><b>Distillation A/B verdict</b> <span class=muted>(swarm with vs without the shared playbook)</span><div id=abv class=muted>run an A/B to populate</div></div>
 <div class=card><b>The charts</b>
   <div class=grid>
     <div><div class=muted>Learning curve</div><img src="/chart/1_learning_curve" onerror="this.style.display='none'"></div>
     <div><div class=muted>Ablation (solo vs swarm)</div><img src="/chart/2_ablation" onerror="this.style.display='none'"></div>
     <div><div class=muted>Held-out validation</div><img src="/chart/3_validation" onerror="this.style.display='none'"></div>
     <div><div class=muted>A/B (distill on vs off)</div><img src="/chart/4_abtest" onerror="this.style.display='none'"></div>
   </div>
 </div>
 <p class=muted>API: <code>POST /api/run?condition=swarm&rounds=3</code> · <code>GET /api/status</code> · <code>GET /api/summary</code></p>
<script>
let busy=false;
async function tick(){
  try{
    const s=await (await fetch('/api/status')).json();
    document.getElementById('st').textContent = s.running? ('● '+s.phase) : (s.error? '⚠ '+s.error : 'idle');
    document.getElementById('detail').innerHTML =
      `episodes: <b>${s.total_episodes}</b> (train ${s.training_episodes} / val ${s.validation_episodes}) · llm calls: ${s.llm_calls} · phase: ${s.phase}`;
    busy=s.running;
    ['swarm','abl','abt'].forEach(id=>{const b=document.getElementById(id); if(b) b.disabled=busy;});
    try{
      const ab=await (await fetch('/api/abtest')).json();
      document.getElementById('abv').innerHTML = ab.verdict
        ? `<b>${ab.verdict}</b><br>later-rounds Δ(on−off): <b>${ab.delta_later_rounds}</b> `
          +`(on ${ab.distill_on.later_rounds} vs off ${ab.distill_off.later_rounds}) · overall Δ ${ab.delta_overall}`
        : 'run an A/B to populate';
    }catch(e){}
    const sum=await (await fetch('/api/summary')).json();
    const sh=sum.score_history||{};
    document.getElementById('scores').innerHTML = Object.keys(sh).length
      ? Object.entries(sh).map(([k,v])=>`${k}: <b>${v.toFixed(3)}</b>`).join(' &nbsp;·&nbsp; ')
      : 'run an experiment to populate';
    if(!s.running){ // refresh charts (cache-bust) when idle
      document.querySelectorAll('img').forEach(im=>{const u=im.src.split('?')[0]; im.src=u+'?t='+Date.now(); im.style.display='';});
    }
  }catch(e){ document.getElementById('st').textContent='(status unavailable)'; }
}
async function run(cond,rounds){
  if(busy) return;
  const r=await fetch('/api/run?condition='+cond+'&rounds='+rounds,{method:'POST'});
  await tick();
}
tick(); setInterval(tick, 4000);
</script>
</div></body></html>"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(_PAGE)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
