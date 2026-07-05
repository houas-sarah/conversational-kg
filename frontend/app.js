(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const state = {
    graph: { nodes: [], edges: [] },
    freshNodeIds: new Set(),
    freshEdgeIds: new Set(),
    turn: 0,
    ws: null,
    wsReady: false,
    pending: [],   // FIFO queue of { userEl, botEl } awaiting a server turn
  };

  const transcript = $("transcript");
  const calloutsList = $("callouts-list");
  const canvasEl = document.querySelector(".canvas");

  // ── Session identity ─────────────────────────────────────────────
  // A per-browser id kept in localStorage. It scopes this visitor's graph on
  // the server, so the shared live demo gives everyone their own memory.
  function sessionId() {
    let s = localStorage.getItem("kg_sid");
    if (!s) {
      s = (window.crypto && crypto.randomUUID)
        ? crypto.randomUUID()
        : Date.now().toString(36) + Math.random().toString(16).slice(2);
      localStorage.setItem("kg_sid", s);
    }
    return s;
  }
  state.sid = sessionId();
  const sidHeaders = (extra) => Object.assign({ "X-Session-Id": state.sid }, extra || {});
  function adoptSid(p) {
    if (p && p.sid && p.sid !== state.sid) {
      state.sid = p.sid;
      localStorage.setItem("kg_sid", p.sid);
    }
  }

  // ── Helpers ──────────────────────────────────────────────────────
  const norm = (s) => String(s).trim().toLowerCase();
  const cap = (s) => { s = String(s || ""); return s ? s[0].toUpperCase() + s.slice(1) : s; };
  const prettyEntity = (s) => (norm(s) === "user" ? "You" : cap(s));
  const nodeLabel = (d) => (norm(d.label) === "user" ? "You" : cap(d.label));
  const shortMethod = (m) => (/llm|groq/i.test(String(m || "")) ? "hybrid · llm" : "rules");

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }
  function scrollDown() { transcript.scrollTop = transcript.scrollHeight; }

  // ── WebSocket ────────────────────────────────────────────────────
  function setStatus(live) {
    const el = $("status");
    el.textContent = live ? "live" : "offline";
    el.className = "meta__v status " + (live ? "status--live" : "status--off");
  }

  function connectWS() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws?sid=${encodeURIComponent(state.sid)}`);
    state.ws = ws;

    ws.onopen = () => {
      state.wsReady = true;
      $("t-backend").textContent = "connected";
      setStatus(true);
    };
    ws.onclose = () => {
      state.wsReady = false;
      $("t-backend").textContent = "disconnected";
      setStatus(false);
      setTimeout(connectWS, 1500);
    };
    ws.onerror = () => { $("t-backend").textContent = "error"; };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === "hello") {
        adoptSid(msg.payload);
        applyGraph(msg.payload.graph);
        applyStats(msg.payload.stats);
        renderGraph();
      } else if (msg.type === "turn") {
        onTurn(msg.payload);
      }
    };
  }

  // ── Status fetch ─────────────────────────────────────────────────
  async function fetchStatus() {
    try {
      const r = await fetch("/api/status", { headers: sidHeaders() });
      const s = await r.json();
      adoptSid(s);
      $("method").textContent = shortMethod(s.extraction_method);
      applyStats(s.stats);
    } catch {}
  }

  // ── Send / receive ───────────────────────────────────────────────
  async function sendMessage(text) {
    if (!text || !text.trim()) return;
    const slot = { userEl: addUserMsg(text), botEl: addPending() };
    state.pending.push(slot);

    if (state.wsReady) {
      state.ws.send(JSON.stringify({ type: "chat", payload: { text } }));
    } else {
      try {
        const r = await fetch("/api/chat", {
          method: "POST",
          headers: sidHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify({ text }),
        });
        onTurn(await r.json());
      } catch {
        const i = state.pending.indexOf(slot);
        if (i >= 0) state.pending.splice(i, 1);
        fillBotMsg(slot.botEl, "Connection error — is the backend running?");
      }
    }
  }

  function onTurn(p) {
    adoptSid(p);
    const slot = state.pending.shift() || {};
    state.turn = p.turn;
    $("session-id").textContent = String(p.turn).padStart(2, "0");
    if (p.extraction_method) $("method").textContent = shortMethod(p.extraction_method);

    // Make coreference visible: when the bot resolved a pronoun, show the
    // rewritten sentence right under this turn's user message.
    if (slot.userEl && p.resolved_text && norm(p.resolved_text) !== norm(p.user || "")) {
      const capEl = document.createElement("div");
      capEl.className = "msg__resolved";
      capEl.innerHTML =
        `<span class="msg__resolved-arrow">↳</span>` +
        `<span>read as&nbsp; <b>${escapeHtml(p.resolved_text)}</b></span>`;
      slot.userEl.appendChild(capEl);
    }

    state.freshNodeIds = new Set();
    state.freshEdgeIds = new Set();
    for (const f of p.new_facts || []) {
      state.freshEdgeIds.add(f.id);
      state.freshNodeIds.add(norm(f.subject));
      state.freshNodeIds.add(norm(f.object));
    }
    applyGraph(p.graph);
    applyStats(p.stats);
    $("t-turn").textContent = "#" + String(p.turn).padStart(2, "0");
    $("t-latency").textContent = (p.elapsed_ms != null ? p.elapsed_ms : "—") + " ms";

    renderCallouts(p.new_facts, p.superseded, p.retracted);
    fillBotMsg(slot.botEl, p.reply, { newFacts: p.new_facts, retracted: p.retracted });
    renderGraph();

    setTimeout(() => {
      state.freshNodeIds.clear();
      state.freshEdgeIds.clear();
      renderGraph();
    }, 2400);
  }

  function applyGraph(g) { state.graph = g || { nodes: [], edges: [] }; }
  function applyStats(s) {
    if (!s) return;
    $("t-nodes").textContent = s.nodes;
    $("t-edges").textContent = s.edges_active;
    $("t-conflicts").textContent = s.conflicts;
  }

  // ── Transcript ────────────────────────────────────────────────────
  function killWelcome() { const w = $("welcome"); if (w) w.remove(); }

  function addUserMsg(text) {
    killWelcome();
    const el = document.createElement("div");
    el.className = "msg msg--user";
    el.innerHTML =
      `<div class="msg__role">you</div>` +
      `<div class="msg__bubble">${escapeHtml(text)}</div>`;
    transcript.appendChild(el);
    scrollDown();
    return el;
  }

  function addPending() {
    const el = document.createElement("div");
    el.className = "msg msg--bot msg--pending";
    el.innerHTML =
      `<div class="msg__role">bot</div>` +
      `<div class="msg__bubble"><span class="dots"><i></i><i></i><i></i></span></div>`;
    transcript.appendChild(el);
    scrollDown();
    return el;
  }

  // Turn a pending "thinking…" bubble into the real reply, in place — so a
  // reply always lands in its own slot even when several turns are in flight.
  function fillBotMsg(el, text, meta = {}) {
    if (!el) {
      el = document.createElement("div");
      el.className = "msg msg--bot";
      transcript.appendChild(el);
    }
    el.classList.remove("msg--pending");
    const n = (meta.newFacts || []).length;
    const r = (meta.retracted || []).length;
    let tag = "";
    if (n || r) {
      const bits = [];
      if (n) bits.push(`stored <b>${n}</b> new`);
      if (r) bits.push(`retracted <b>${r}</b>`);
      const total = n + r;
      tag = `<div class="msg__tag">${bits.join(" · ")} fact${total === 1 ? "" : "s"} in the graph</div>`;
    }
    el.innerHTML =
      `<div class="msg__role">bot</div>` +
      `<div class="msg__bubble">${escapeHtml(text)}${tag}</div>`;
    scrollDown();
    return el;
  }

  // ── Callouts ─────────────────────────────────────────────────────
  function calloutRow(f, cls) {
    const li = document.createElement("li");
    if (cls) li.className = cls;
    li.innerHTML =
      `<span class="callout__ent">${escapeHtml(prettyEntity(f.subject))}</span>` +
      `<span class="callout__pred">${escapeHtml(f.predicate)}</span>` +
      `<span class="callout__ent">${escapeHtml(prettyEntity(f.object))}</span>`;
    return li;
  }
  function renderCallouts(newFacts, superseded, retracted) {
    calloutsList.innerHTML = "";
    const nf = newFacts || [], sp = superseded || [], rt = retracted || [];
    if (!nf.length && !sp.length && !rt.length) {
      const li = document.createElement("li");
      li.className = "callouts__empty";
      li.textContent = "Nothing extracted this turn — mention a name, place or topic.";
      calloutsList.appendChild(li);
      return;
    }
    for (const f of nf) calloutsList.appendChild(calloutRow(f, ""));
    for (const f of sp) calloutsList.appendChild(calloutRow(f, "superseded"));
    for (const f of rt) calloutsList.appendChild(calloutRow(f, "superseded"));
  }

  // ── Graph rendering (D3 force) ───────────────────────────────────
  const svg = d3.select("#graph");
  const gRoot = svg.append("g").attr("class", "root");
  const gLinks = gRoot.append("g").attr("class", "links");
  const gLinkLabels = gRoot.append("g").attr("class", "link-labels");
  const gNodes = gRoot.append("g").attr("class", "nodes");

  const defs = svg.append("defs");
  function marker(id, color, w) {
    defs.append("marker")
      .attr("id", id)
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 18).attr("refY", 0)
      .attr("markerWidth", w).attr("markerHeight", w)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", color);
  }
  marker("arrow", "#c4c0d8", 7.5);
  marker("arrow-fresh", "#d946ef", 9);

  let simulation = null;

  function nodeKind(kind) {
    const k = String(kind || "").toLowerCase();
    if (k === "person") return "person";
    if (["location", "gpe", "loc"].includes(k)) return "location";
    if (["organization", "org"].includes(k)) return "organization";
    if (["time", "date"].includes(k)) return "time";
    if (k === "feeling") return "feeling";
    if (["subject", "topic"].includes(k)) return "subject";
    return "concept";
  }

  svg.call(d3.zoom().scaleExtent([0.4, 3]).on("zoom", (event) => {
    gRoot.attr("transform", event.transform);
  }));

  function size() {
    const el = svg.node();
    return { w: el.clientWidth || 800, h: el.clientHeight || 600 };
  }

  function linkClass(d) {
    return `link ${state.freshEdgeIds.has(d.id) ? "fresh" : ""} ${d.superseded ? "superseded" : ""}`;
  }
  function linkMarker(d) {
    return state.freshEdgeIds.has(d.id) ? "url(#arrow-fresh)" : "url(#arrow)";
  }
  function nodeClass(d) {
    return `node ${nodeKind(d.kind)} ${state.freshNodeIds.has(d.id) ? "fresh" : ""}`;
  }

  function renderGraph() {
    const { w, h } = size();
    const nodes = state.graph.nodes.map((n) => ({ ...n }));
    const nodeIndex = new Map(nodes.map((n) => [n.id, n]));
    const edges = state.graph.edges
      .filter((e) => nodeIndex.has(e.source) && nodeIndex.has(e.target))
      .map((e) => ({ ...e }));

    // Fan out parallel edges between the same node pair: each edge gets its
    // own "lane" (curvature offset) so paths and labels never overlap.
    // `dir` keeps the bend side consistent regardless of edge direction.
    const pairGroups = new Map();
    for (const e of edges) {
      const k = e.source < e.target ? `${e.source}|${e.target}` : `${e.target}|${e.source}`;
      e.dir = e.source < e.target ? 1 : -1;
      if (!pairGroups.has(k)) pairGroups.set(k, []);
      pairGroups.get(k).push(e);
    }
    for (const group of pairGroups.values()) {
      group.forEach((e, i) => { e.lane = i - (group.length - 1) / 2; });
    }

    canvasEl.classList.toggle("has-nodes", nodes.length > 0);

    if (simulation) simulation.stop();

    simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(edges).id((d) => d.id).distance(96).strength(0.6))
      .force("charge", d3.forceManyBody().strength(-280))
      .force("center", d3.forceCenter(w / 2, h / 2))
      .force("collision", d3.forceCollide().radius(30))
      .alpha(0.9).alphaDecay(0.045);

    const link = gLinks.selectAll("path.link")
      .data(edges, (d) => d.id)
      .join(
        (enter) => enter.append("path").attr("class", linkClass).attr("marker-end", linkMarker),
        (update) => update.attr("class", linkClass).attr("marker-end", linkMarker),
        (exit) => exit.remove()
      );

    const labelClass = (d) => `link-label${d.superseded ? " superseded" : ""}`;
    const linkLabel = gLinkLabels.selectAll("text.link-label")
      .data(edges, (d) => d.id)
      .join(
        (enter) => enter.append("text").attr("class", labelClass).attr("text-anchor", "middle"),
        (update) => update.attr("class", labelClass),
        (exit) => exit.remove()
      )
      .text((d) => d.predicate);

    const node = gNodes.selectAll("g.node")
      .data(nodes, (d) => d.id)
      .join(
        (enter) => {
          const g = enter.append("g").attr("class", nodeClass).call(drag());
          g.append("circle").attr("r", 9);
          g.append("text").attr("x", 13).attr("y", 4).text(nodeLabel);
          g.append("title").text((d) =>
            `${nodeLabel(d)} · ${d.kind} · ${d.mentions} mention${d.mentions === 1 ? "" : "s"}`);
          return g;
        },
        (update) => {
          update.attr("class", nodeClass);
          update.select("text").text(nodeLabel);
          return update;
        },
        (exit) => exit.remove()
      );

    // Perpendicular bend for an edge: single edges get a gentle arc,
    // parallel edges fan out on both sides.
    function bendOf(d, dist) {
      const mag = d.lane ? 0.14 + 0.24 * Math.abs(d.lane) : 0.10;
      const side = (d.lane ? Math.sign(d.lane) : 1) * (d.dir || 1);
      return dist * mag * side;
    }

    simulation.on("tick", () => {
      link.attr("d", (d) => {
        const sx = d.source.x, sy = d.source.y, tx = d.target.x, ty = d.target.y;
        const dx = tx - sx, dy = ty - sy;
        const dist = Math.hypot(dx, dy) || 1;
        const b = bendOf(d, dist);
        // control point = midpoint + 2b along the normal → curve peak at +b
        const cx = (sx + tx) / 2 + (-dy / dist) * 2 * b;
        const cy = (sy + ty) / 2 + (dx / dist) * 2 * b;
        return `M${sx},${sy} Q${cx},${cy} ${tx},${ty}`;
      });
      linkLabel
        .attr("x", (d) => {
          const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
          const dist = Math.hypot(dx, dy) || 1;
          return (d.source.x + d.target.x) / 2 + (-dy / dist) * bendOf(d, dist);
        })
        .attr("y", (d) => {
          const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
          const dist = Math.hypot(dx, dy) || 1;
          return (d.source.y + d.target.y) / 2 + (dx / dist) * bendOf(d, dist) - 3;
        });
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });
  }

  function drag() {
    function started(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    }
    function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
    function ended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null; d.fy = null;
    }
    return d3.drag().on("start", started).on("drag", dragged).on("end", ended);
  }

  window.addEventListener("resize", () => {
    if (simulation) {
      const { w, h } = size();
      simulation.force("center", d3.forceCenter(w / 2, h / 2));
      simulation.alpha(0.4).restart();
    }
  });

  // ── Event wiring ─────────────────────────────────────────────────
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = $("input");
    const text = input.value;
    input.value = "";
    sendMessage(text);
  });

  document.addEventListener("click", (e) => {
    const t = e.target;
    if (t.classList && t.classList.contains("try")) {
      sendMessage(t.textContent.trim());
    }
  });

  $("btn-reset").addEventListener("click", async () => {
    if (!confirm("Clear the entire knowledge graph and conversation?")) return;
    try { await fetch("/api/reset", { method: "POST", headers: sidHeaders() }); } catch {}
    location.reload();
  });

  // ── Boot ─────────────────────────────────────────────────────────
  fetchStatus();
  connectWS();
})();
