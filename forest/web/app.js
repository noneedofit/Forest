const MAX_ROWS = 25;
const $ = (id) => document.getElementById(id);
const inr = (n) => "₹" + Math.round(n).toLocaleString("en-IN");

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function addRow(e) {
  const tr = document.createElement("tr");
  tr.className = "new";
  const top = e.reasons.length ? e.reasons[0].en : "";
  tr.innerHTML =
    `<td>${esc(e.ts.slice(5))}</td><td>${esc(e.from)}</td><td>${esc(e.to)}</td>` +
    `<td class="num">${inr(e.amount_inr)}</td>` +
    `<td><span class="bar"><i style="width:${Math.min(100, e.score * 100)}%"></i></span></td>` +
    `<td><span class="chip ${e.decision}">${e.decision}</span></td><td class="why">${esc(top)}</td>`;
  const body = $("rows");
  body.prepend(tr);
  while (body.children.length > MAX_ROWS) body.lastChild.remove();
}

// At 50 payments/s the table would be unreadable, so rows are shown at ~8/s. When a backlog
// builds up, ordinary allowed payments are dropped from the view first (they are still scored).
const queue = [];
function drain() {
  if (queue.length > 40) {
    const keep = queue.filter((e) => e.decision !== "allow" || e.victim);
    queue.length = 0;
    queue.push(...keep);
  }
  const e = queue.shift();
  if (!e) return;
  addRow(e);
  showPhone(e);
}

function showPhone(e) {
  if (!e.victim || e.decision === "allow") return;
  const lines = e.reasons.slice(0, 2).map((r) => `<p>${esc(r.hi)}<br><i>${esc(r.en)}</i></p>`).join("");
  const title = e.decision === "hold" ? "Ruk jaiye! / Wait!" : "Ek minute / One moment";
  $("phone").innerHTML =
    `<div class="alert"><h3>${title}</h3><p>${inr(e.amount_inr)} to ${esc(e.to)} was stopped.</p>${lines}</div>` +
    `<p class="ok">Family contact notified ✓</p>`;
}

function renderIntel(rows) {
  const ul = $("intel");
  if (!rows.length) return;
  ul.innerHTML = rows
    .map(
      (r) =>
        `<li><span>${esc(r.display)} <small>${esc(r.scam_type || r.kind)} · ${esc(r.first_seen.slice(11, 16))}</small></span>` +
        `<span class="hits">${r.hits} held</span></li>`
    )
    .join("");
}

function chatNote(cls, html) {
  const box = $("chat");
  if (box.querySelector(".empty")) box.innerHTML = "";
  const div = document.createElement("div");
  div.className = cls;
  div.innerHTML = html;
  box.append(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

let current = null; // session shown in the chat panel

function newSession(e) {
  current = e.session;
  $("chat").innerHTML = "";
  chatNote("note", `Reported number · ${esc(e.caller)}`);
}

// Granite takes a few seconds per reply, so show that Kamla is "typing" meanwhile.
function addChat(e) {
  current = e.session;
  const typing = $("chat").querySelector(".typing");
  if (typing) typing.remove();
  let html = esc(e.text);
  for (const ent of e.entities || []) html = html.replace(esc(ent), `<mark>${esc(ent)}</mark>`);
  chatNote("msg " + (e.role === "agent" ? "agent" : "scammer"), html);
  if (e.role === "scammer") chatNote("msg agent typing", "Kamla is typing…");
}

const modelName = (f) => (f || "").replace(/-be\.\w+\.gguf$|\.gguf$/, "");

// The 2B summary arrives a minute or two after the chat ends; until then the card says so,
// and the final report replaces it in place.
function showReport(r) {
  const ids = r.ids.map((i) => esc(i.display)).join(", ") || "none";
  const by = {
    pending: `${esc(modelName(r.summary_model))} is writing the summary…`,
    granite: `written by ${esc(modelName(r.summary_model) || "Granite")}`,
    template: "template",
  }[r.summary_by];
  const body = r.summary_by === "pending" ? "<p><i>Summary pending…</i></p>" : `<p>${esc(r.summary)}</p>`;
  const html =
    `<b>Report for the 1930 helpline</b> <small>${by}</small>${body}` +
    `<small>${esc(r.scam_type.replace("_", " "))} · ids: ${ids} · ` +
    `${r.turns} turns · ${r.scammer_minutes} scammer min</small>`;
  const card = $("chat").querySelector(`[data-report="${r.session}"]`);
  if (card) card.innerHTML = html;
  else if (r.session === current) chatNote("report", html).dataset.report = r.session;
}

function showStats(s) {
  $("b-scored").textContent = `${s.scored.toLocaleString()} scored`;
  $("b-p99").textContent = `p99 ${s.p99_ms.toFixed(2)} ms`;
  $("f-held").textContent = s.held.toLocaleString();
  $("f-paused").textContent = s.paused.toLocaleString();
  $("f-inr").textContent = inr(s.inr_protected);
  $("f-min").textContent = s.scammer_minutes.toFixed(1);
}

async function refreshIntel() {
  renderIntel(await (await fetch("/intel")).json());
}

async function init() {
  const h = await (await fetch("/health")).json();
  const badge = $("b-platform");
  const nnpa = h.nnpa && h.tree_format.startsWith("zdnn");
  badge.textContent = `${h.machine} · ${nnpa ? "Snap ML on Telum II NNPA (zDNN)" : "Snap ML"}` +
    (h.granite_loaded ? ` · ${modelName(h.granite_model)}` : "") +
    (h.summary_model ? ` + ${modelName(h.summary_model)}` : "");
  badge.classList.add("ok");
  showStats(await (await fetch("/stats")).json());
  await refreshIntel();

  const es = new EventSource("/events");
  es.addEventListener("decision", (m) => {
    const e = JSON.parse(m.data);
    // the victim's payment is the point of the demo, so it skips the display backlog
    if (e.victim) queue.unshift(e);
    else queue.push(e);
  });
  setInterval(drain, 120);
  es.addEventListener("stats", (m) => showStats(JSON.parse(m.data)));
  es.addEventListener("intel", () => refreshIntel());
  es.addEventListener("chat", (m) => addChat(JSON.parse(m.data)));
  es.addEventListener("session", (m) => newSession(JSON.parse(m.data)));
  es.addEventListener("report", (m) => showReport(JSON.parse(m.data)));
  setInterval(refreshIntel, 5000);
}

init();
