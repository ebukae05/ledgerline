// Ledgerline web UI: a client of the same public API anyone else would call.

const $ = (selector) => document.querySelector(selector);
const tooltip = $("#tooltip");

async function api(path, options = {}) {
  const started = performance.now();
  const response = await fetch(path, {
    ...options,
    headers: { "content-type": "application/json", ...(options.headers || {}) },
  });
  let body = null;
  try { body = await response.json(); } catch { /* empty body */ }
  return { ok: response.ok, status: response.status, body, ms: performance.now() - started };
}

const escape = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function formatScore(score) {
  if (score >= 0.001) return score.toPrecision(3);
  const [mantissa, exponent] = score.toExponential(1).split("e");
  const sup = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  return `${mantissa} × 10${[...String(Number(exponent))].map((c) => sup[c]).join("")}`;
}

const ICON_FLAG = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
const ICON_CLEAR = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

// ---------------------------------------------------------------- health
async function refreshHealth() {
  const nav = $("#nav-status");
  try {
    const { status, body } = await api("/health");
    const ok = status === 200;
    nav.classList.toggle("ok", ok);
    nav.querySelector(".label").textContent = ok ? "live" : "degraded";
    $("#hero-version").textContent = body.model_version;
    $("#h-model").textContent = body.model_version;
    $("#h-db").textContent = body.database ? "Reachable" : "Unreachable";
    $("#h-llm").textContent = body.merchant_llm ? "Gemini, with TF-IDF fallback" : "Off (TF-IDF only)";
    const recent = body.merchant_last_hour || {};
    const parts = Object.entries(recent).map(([method, n]) => `${n} ${method === "llm" ? "LLM" : "TF-IDF"}`);
    $("#h-merchant").textContent = parts.length ? parts.join(" · ") : "None yet";
  } catch {
    nav.classList.remove("ok");
    nav.querySelector(".label").textContent = "offline";
  }
}

// ---------------------------------------------------------------- score
const KIND_LABELS = {
  normal: "Known label: legitimate",
  fraud_flagged: "Known label: fraud",
  fraud_missed: "Known label: fraud",
  broken: "",
};
let samples = null;
const cursor = { normal: 0, fraud_flagged: 0, fraud_missed: 0, broken: 0 };
let currentKind = null;

async function loadSamples() {
  const { ok, body } = await api("/samples");
  if (ok) {
    samples = body;
    return;
  }
  document.querySelectorAll("[data-kind]").forEach((b) => (b.disabled = b.dataset.kind !== "broken"));
  $("#score-result").innerHTML =
    `<p class="placeholder">No example transactions yet. Run <code>python -m ledgerline.api.samples</code>, or paste your own JSON into the request body.</p>`;
}

function pick(kind) {
  const pool = samples?.[kind === "broken" ? "normal" : kind] || [];
  if (!pool.length) return null;
  const tx = structuredClone(pool[cursor[kind] % pool.length]);
  cursor[kind] += 1;
  if (kind === "broken") {
    tx.TransactionAmt = -25;
    tx.Amout = 10; // misspelled field
    delete tx.card1;
  }
  return tx;
}

function setBody(tx) {
  $("#score-body").value = JSON.stringify(tx, null, 2);
  const fields = Object.keys(tx).length;
  $("#body-summary").textContent = `· transaction ${tx.TransactionID ?? "?"} · ${fields} fields`;
}

async function sendScore() {
  let tx;
  try {
    tx = JSON.parse($("#score-body").value);
  } catch (error) {
    renderErrors(["Request body isn't valid JSON: " + error.message], null);
    return;
  }
  const target = $("#score-result");
  target.setAttribute("aria-busy", "true");
  const { status, body, ms } = await api("/score", { method: "POST", body: JSON.stringify(tx) });
  target.removeAttribute("aria-busy");
  if (status === 200) renderScore(body, ms);
  else renderErrors(body?.errors || [body?.detail || `HTTP ${status}`], status, ms);
  refreshAudit();
}

function renderErrors(errors, status, ms) {
  $("#score-result").innerHTML = `
    <div class="result-head">
      <span class="hero-num">${status ?? "—"}</span>
      <span class="status flagged">${ICON_FLAG}Rejected</span>
    </div>
    <div class="meta mono"><span>${status === 422 ? "Invalid request, nothing scored or audited" : "Request failed"}</span>${ms ? `<span>${Math.round(ms)} ms</span>` : ""}</div>
    <ul class="errors">${errors.map((e) => `<li>${escape(e)}</li>`).join("")}</ul>`;
}

function renderScore(result, ms) {
  const flagged = result.flagged;
  const label = KIND_LABELS[currentKind] || "";
  const reasons = result.reasons || [];
  const maxContribution = Math.max(...reasons.map((r) => r.contribution), 1e-9);
  $("#score-result").innerHTML = `
    <div class="result-head">
      <span class="hero-num" aria-label="Fraud score ${result.score}">${formatScore(result.score)}</span>
      <span class="status ${flagged ? "flagged" : "clear"}">${flagged ? ICON_FLAG + "Flagged" : ICON_CLEAR + "Clear"}</span>
    </div>
    <div class="meta mono">
      <span>Transaction ${result.transaction_id}</span>
      <span>Threshold ${result.threshold.toPrecision(3)}</span>
      <span>${escape(result.model_version)}</span>
      <span>${Math.round(ms)} ms round trip</span>
    </div>
    ${label ? `<p class="truth">${label}, from the test set's labels. The API never sees it.</p>` : ""}
    <figure class="meter" id="meter" style="margin-inline:0"></figure>
    ${flagged ? `
      <p class="reasons-title mono">Why it was flagged</p>
      <ol class="reasons">
        ${reasons.map((r) => `
          <li title="${escape(r.feature)}: contribution +${r.contribution.toFixed(3)} log-odds">
            <span class="text">${escape(r.text)}</span>
            <span class="val">+${r.contribution.toFixed(2)}</span>
            <span class="bar" style="width:${(100 * r.contribution) / maxContribution}%"></span>
          </li>`).join("")}
      </ol>` : `<p class="caption">Unflagged transactions skip the reasons step: computing exact per-feature contributions costs about 0.45 s, so it's reserved for flags.</p>`}`;
  drawMeter($("#meter"), result.score, result.threshold, flagged);
}

// Log-scale meter: scores span 1e-6 to 1, and the threshold sits near 0.03,
// so a linear 0-1 bar would crowd almost everything into the first pixel.
function drawMeter(figure, score, threshold, flagged) {
  const lo = -6, hi = 0;
  const pos = (v) => (Math.log10(Math.min(Math.max(v, 10 ** lo), 1)) - lo) / (hi - lo);
  const W = 1000, trackY = 34, trackH = 8;
  const sx = pos(score) * W, tx = pos(threshold) * W;
  const ticks = [-6, -4, -2, 0].map((e) => {
    const x = ((e - lo) / (hi - lo)) * W;
    const text = e === 0 ? "1" : `10${e === -6 ? "⁻⁶" : e === -4 ? "⁻⁴" : "⁻²"}`;
    return `<text class="tick" x="${x}" y="70" text-anchor="${e === 0 ? "end" : e === lo ? "start" : "middle"}">${text}</text>`;
  }).join("");
  figure.innerHTML = `
    <svg viewBox="0 0 ${W} 76" preserveAspectRatio="none" role="img"
         aria-label="Score ${score.toPrecision(3)} on a log scale from one in a million to 1; threshold ${threshold.toPrecision(3)}; ${flagged ? "above" : "below"} threshold">
      <rect class="track" x="0" y="${trackY}" width="${W}" height="${trackH}" rx="4"/>
      <rect class="fill ${flagged ? "flagged" : ""}" x="0" y="${trackY}" width="${Math.max(sx, 4)}" height="${trackH}" rx="4"/>
      <line class="threshold" x1="${tx}" x2="${tx}" y1="${trackY - 14}" y2="${trackY + trackH + 8}"/>
      <text class="label" x="${tx}" y="${trackY - 20}" text-anchor="middle">THRESHOLD ${threshold.toPrecision(2)}</text>
      <circle class="marker" cx="${sx}" cy="${trackY + trackH / 2}" r="7"/>
      ${ticks}
      <rect class="hit" x="0" y="0" width="${W}" height="76"/>
    </svg>
    <figcaption class="caption">Fraud score on a log scale. Flag when the score reaches the threshold picked on validation data (at most 1% of legitimate transactions flagged).</figcaption>`;
  const hit = figure.querySelector(".hit");
  hit.addEventListener("pointermove", (event) => {
    const rect = figure.querySelector("svg").getBoundingClientRect();
    const frac = Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1);
    const value = 10 ** (lo + frac * (hi - lo));
    tooltip.hidden = false;
    tooltip.style.left = `${event.clientX}px`;
    tooltip.style.top = `${rect.top + 22}px`;
    tooltip.innerHTML = `here: ${formatScore(value)}<br>this score: ${formatScore(score)}`;
  });
  hit.addEventListener("pointerleave", () => (tooltip.hidden = true));
}

document.querySelectorAll("[data-kind]").forEach((button) => {
  button.addEventListener("click", () => {
    const tx = pick(button.dataset.kind);
    if (!tx) return;
    currentKind = button.dataset.kind;
    document.querySelectorAll("[data-kind]").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    setBody(tx);
    sendScore();
  });
});
$("#score-send").addEventListener("click", () => {
  currentKind = null;
  document.querySelectorAll("[data-kind]").forEach((b) => b.setAttribute("aria-pressed", "false"));
  sendScore();
});

// ---------------------------------------------------------------- merchant
async function categorize(description, direction) {
  const target = $("#cat-result");
  if (!description.trim()) return;
  target.innerHTML = `<p class="placeholder">Categorizing…</p>`;
  const { status, body, ms } = await api("/merchant", {
    method: "POST",
    body: JSON.stringify({ description, direction }),
  });
  if (status !== 200) {
    target.innerHTML = `<ul class="errors">${(body?.errors || [body?.detail || `HTTP ${status}`]).map((e) => `<li>${escape(e)}</li>`).join("")}</ul>`;
    return;
  }
  const via = body.method === "llm"
    ? `<span class="badge">LLM</span> Gemini answered`
    : `<span class="badge">TF-IDF</span> local model answered, ${(body.confidence * 100).toFixed(0)}% confident`;
  target.innerHTML = `
    <p class="category">${escape(body.category)}</p>
    <div class="how">${via}<span class="muted">· ${Math.round(ms)} ms</span></div>`;
  refreshAudit();
  refreshHealth();
}

$("#cat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  categorize($("#cat-input").value, new FormData(event.target).get("direction"));
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("#cat-input").value = chip.textContent;
    const direction = chip.dataset.direction || "debit";
    document.querySelector(`input[name=direction][value=${direction}]`).checked = true;
    categorize(chip.textContent, direction);
  });
});

// ---------------------------------------------------------------- audit
const seen = new Set();
let firstAuditLoad = true;

async function refreshAudit() {
  const { ok, body } = await api("/decisions?limit=12");
  const rows = $("#audit-rows");
  if (!ok) {
    rows.innerHTML = `<tr><td colspan="7" class="muted">Audit log unavailable.</td></tr>`;
    return;
  }
  if (!body.length) {
    rows.innerHTML = `<tr><td colspan="7" class="muted">No decisions yet.</td></tr>`;
    return;
  }
  rows.innerHTML = body.map((d) => {
    const key = `${d.created_at}|${d.input_hash}`;
    const fresh = !firstAuditLoad && !seen.has(key);
    seen.add(key);
    const time = new Date(d.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    let decision = "";
    if (d.endpoint === "/score") decision = d.flagged ? '<span class="flag-yes">✕ flagged</span>' : '<span class="flag-no">✓ clear</span>';
    else if (d.output?.category) decision = escape(d.output.category);
    return `<tr class="${fresh ? "fresh" : ""}">
      <td>${time}</td>
      <td>${escape(d.endpoint)}</td>
      <td>${d.transaction_id ?? ""}</td>
      <td class="r">${d.score == null ? "" : formatScore(d.score)}</td>
      <td>${decision}</td>
      <td>${escape(d.model_version)}</td>
      <td class="muted">${escape(d.input_hash.slice(0, 12))}</td>
    </tr>`;
  }).join("");
  firstAuditLoad = false;
}

// ---------------------------------------------------------------- start
refreshHealth();
refreshAudit();
loadSamples();
setInterval(refreshHealth, 15000);
setInterval(() => { if (!document.hidden) refreshAudit(); }, 5000);
