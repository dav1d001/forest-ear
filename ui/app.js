"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const nice = s => String(s).replace(/_/g, " ");
const pct = (x, d = 0) => x == null ? "–" : (x * 100).toFixed(d) + "%";
const TEAL = "#1F6F66", ALERT = "#D4452B", INK = "#14211F";
let ALERTS = new Set(["chainsaw", "crackling_fire", "helicopter"]);
let STATUS = null, MODEL = null;

async function api(path, opts) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch (_) {}
  if (!r.ok) throw new Error((body && (body.detail?.[0]?.msg || body.detail)) || `Request failed (${r.status})`);
  return body;
}
function dur(s) {
  s = Math.floor(s); const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
  return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`;
}

/* ------------------------------------------------------------------ tabs */
function showTab(name) {
  $$(".tabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.tab === name));
  $$(".tab").forEach(t => t.classList.toggle("on", t.id === "tab-" + name));
  if (location.hash.slice(1) !== name) history.replaceState(null, "", "#" + name);
  if (name === "insights") loadInsights();
  if (name === "teach") { refreshUploads(); refreshRetrain(); }
  if (name === "monitor") { refreshStatus(); loadModel(); }
}
$$(".tabs button").forEach(b => b.onclick = () => showTab(b.dataset.tab));

/* ----------------------------------------------------------- spectrogram */
const LUT = (() => {
  const stops = [[0, [14, 24, 48]], [.35, [31, 95, 122]], [.6, [47, 163, 154]], [.82, [216, 210, 106]], [1, [255, 244, 194]]];
  const out = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255; let k = 1; while (k < stops.length - 1 && t > stops[k][0]) k++;
    const [t0, c0] = stops[k - 1], [t1, c1] = stops[k], u = (t - t0) / (t1 - t0);
    out.push(c0.map((v, j) => Math.round(v + (c1[j] - v) * u)));
  }
  return out;
})();
const canvas = $("#spec"), cx = canvas.getContext("2d");
function drawIdle() {
  cx.fillStyle = "#101C1A"; cx.fillRect(0, 0, canvas.width, canvas.height);
  cx.strokeStyle = "rgba(159,216,204,.10)"; cx.lineWidth = 1;
  for (let y = 20; y < canvas.height; y += 28) { cx.beginPath(); cx.moveTo(0, y); cx.lineTo(canvas.width, y); cx.stroke(); }
}
function drawSpec(grid) {
  const h = grid.length, w = grid[0].length;
  const off = document.createElement("canvas"); off.width = w; off.height = h;
  const img = off.getContext("2d").createImageData(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const c = LUT[grid[h - 1 - y][x]], i = (y * w + x) * 4;
    img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = 255;
  }
  off.getContext("2d").putImageData(img, 0, 0);
  cx.imageSmoothingEnabled = true; cx.imageSmoothingQuality = "high";
  cx.drawImage(off, 0, 0, canvas.width, canvas.height);
  cx.fillStyle = "rgba(255,255,255,.75)"; cx.font = "12px system-ui";
  cx.fillText("low pitch", 8, canvas.height - 8); cx.fillText("high pitch", 8, 16);
  cx.fillText("5 s →", canvas.width - 44, canvas.height - 8);
}
drawIdle();

/* --------------------------------------------------------------- predict */
const drop = $("#drop"), fileIn = $("#file"), player = $("#player");
$("#pick").onclick = () => fileIn.click();
drop.onclick = () => fileIn.click();
drop.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileIn.click(); } };
fileIn.onchange = () => fileIn.files[0] && predictBlob(fileIn.files[0], fileIn.files[0].name);
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) predictBlob(f, f.name); });

async function predictBlob(blob, name, expected) {
  $("#verdictEmpty").hidden = false; $("#verdictBody").hidden = true;
  $("#verdictEmpty").innerHTML = "<h2>Listening…</h2><p>Analysing " + esc(name) + "</p>";
  player.hidden = false; player.src = URL.createObjectURL(blob);
  const fd = new FormData(); fd.append("file", blob, name);
  try {
    const r = await api("/api/predict", { method: "POST", body: fd });
    drop.classList.add("has"); drawSpec(r.spectrogram); renderVerdict(r, expected); refreshStatus();
  } catch (e) {
    $("#verdictEmpty").innerHTML = `<h2>Could not read that clip</h2><p>${esc(e.message)}</p><p>Try a WAV, FLAC or OGG file with some sound in it.</p>`;
  }
}
function renderVerdict(r, expected) {
  $("#verdictEmpty").hidden = true; $("#verdictBody").hidden = false;
  $("#vLabel").textContent = nice(r.label);
  $("#vConf").textContent = pct(r.confidence); $("#vMs").textContent = r.latency_ms; $("#vVer").textContent = r.model_version;
  const fill = $("#vFill"); fill.style.width = (r.confidence * 100) + "%"; fill.classList.toggle("threat", r.alert);
  $("#alertBanner").hidden = !r.alert; $("#lowBanner").hidden = !r.low_confidence;
  $("#vTop").innerHTML = r.top.slice(1).map(t => `<li><span>${esc(nice(t.label))}</span><div class="bar2"><i style="width:${t.probability * 100}%"></i></div><span>${pct(t.probability)}</span></li>`).join("");
  const t = $("#vTruth");
  if (expected) { const ok = expected === r.label; t.hidden = false; t.className = "truth" + (ok ? "" : " bad"); t.textContent = ok ? `Correct: this test clip is labelled “${nice(expected)}”.` : `Wrong: this test clip is labelled “${nice(expected)}”.`; }
  else t.hidden = true;
}
async function loadSamples() {
  try {
    const list = await api("/api/samples");
    $("#samples").innerHTML = list.map(s => `<button class="chip ${ALERTS.has(s.label) ? "threat" : ""}" data-url="${esc(s.url)}" data-label="${esc(s.label)}">${esc(nice(s.label))}</button>`).join("");
    $$("#samples .chip").forEach(b => b.onclick = async () => {
      const blob = await (await fetch(b.dataset.url)).blob();
      predictBlob(blob, b.dataset.url.split("/").pop(), b.dataset.label);
    });
  } catch (_) { $("#samples").textContent = "Samples unavailable."; }
}

/* ---------------------------------------------------- record 5 s (WAV) */
let recording = false;
$("#rec").onclick = async () => {
  if (recording) return; const btn = $("#rec");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } });
    const ac = new (window.AudioContext || window.webkitAudioContext)(), src = ac.createMediaStreamSource(stream);
    const proc = ac.createScriptProcessor(4096, 1, 1), chunks = []; let n = 0; recording = true;
    btn.classList.add("recording"); btn.textContent = "Recording…";
    proc.onaudioprocess = e => { const d = new Float32Array(e.inputBuffer.getChannelData(0)); chunks.push(d); n += d.length;
      if (n >= ac.sampleRate * 5) { finish(); } };
    const finish = () => {
      proc.disconnect(); src.disconnect(); stream.getTracks().forEach(t => t.stop()); recording = false;
      btn.classList.remove("recording"); btn.textContent = "Record 5 s";
      const pcm = new Float32Array(n); let o = 0; chunks.forEach(c => { pcm.set(c, o); o += c.length; });
      const rate = ac.sampleRate; ac.close(); predictBlob(wavBlob(pcm, rate), "recording.wav");
    };
    src.connect(proc); proc.connect(ac.destination);
  } catch (e) { recording = false; $("#verdictEmpty").hidden = false; $("#verdictBody").hidden = true;
    $("#verdictEmpty").innerHTML = `<h2>Microphone unavailable</h2><p>${esc(e.message)}. Browsers only allow recording on localhost or HTTPS.</p>`; }
};
function wavBlob(f32, rate) {
  const buf = new ArrayBuffer(44 + f32.length * 2), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + f32.length * 2, true); w(8, "WAVEfmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, f32.length * 2, true);
  f32.forEach((s, i) => v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 32767, true));
  return new Blob([buf], { type: "audio/wav" });
}

/* --------------------------------------------------------------- insights */
let insightsLoaded = false;
async function loadInsights() {
  if (insightsLoaded) return;
  let d; try { d = await api("/api/insights"); } catch (e) { $("#insightBody").innerHTML = `<p class="muted">${esc(e.message)}</p>`; return; }
  insightsLoaded = true; ALERTS = new Set(d.alert_classes);
  const names = { centroid_hz: ["Spectral centroid (Hz)", v => Math.round(v)], active_fraction: ["Share of clip that is 'on'", v => (v * 100).toFixed(0) + "%"], flatness: ["Spectral flatness (0 tonal, 1 noisy)", v => v.toFixed(3)] };
  let html = "";
  d.stories.forEach((s, i) => {
    html += `<article class="story"><div id="chart${i}"></div><div><h3>${esc(s.title)}</h3><p>${esc(s.text)}</p></div></article>`;
  });
  html += `<article class="fullcard"><h3>Average spectrum of each class</h3><p class="muted">Each row is a class; brighter means more energy at that pitch (left = low, right = high). Helicopter keeps its energy low; chainsaw, rain and sea waves spread theirs across many pitches.</p><div id="heat"></div></article>`;
  $("#insightBody").innerHTML = html;
  d.stories.forEach((s, i) => $("#chart" + i).innerHTML = rangeChart(d, s.feature, ...names[s.feature]));
  $("#heat").innerHTML = heatmap(d);
}
function rangeChart(d, key, title, fmt) {
  const rows = d.classes.map(c => ({ c, m: d.per_class[c][key].mean, s: d.per_class[c][key].std })).sort((a, b) => a.m - b.m);
  const W = 520, rowH = 26, L = 110, R = 64, T = 28, H = T + rows.length * rowH + 14;
  const lo = Math.min(...rows.map(r => r.m - r.s), key === "active_fraction" ? 0 : Infinity), hi = Math.max(...rows.map(r => r.m + r.s), key === "active_fraction" ? 1 : -Infinity);
  const lo2 = key === "centroid_hz" ? Math.max(0, lo) : Math.max(0, lo), x = v => L + (Math.max(lo2, Math.min(hi, v)) - lo2) / (hi - lo2) * (W - L - R);
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)} by class"><text x="${L}" y="14" style="font-weight:600">${esc(title)}</text>`;
  rows.forEach((r, i) => {
    const y = T + i * rowH + 8, col = d.alert_classes.includes(r.c) ? ALERT : TEAL;
    g += `<text class="lbl ${d.alert_classes.includes(r.c) ? "threat" : ""}" x="${L - 8}" y="${y + 4}" text-anchor="end">${esc(nice(r.c))}</text>`;
    g += `<line x1="${x(r.m - r.s)}" x2="${x(r.m + r.s)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="3" stroke-linecap="round" opacity=".35"/>`;
    g += `<circle cx="${x(r.m)}" cy="${y}" r="5" fill="${col}"/><text x="${x(r.m + r.s) + 6}" y="${y + 4}" style="fill:#5C6E69">${fmt(r.m)}</text>`;
  });
  return g + "</svg>";
}
function heatmap(d) {
  const order = [...d.classes].sort((a, b) => d.per_class[a].centroid_hz.mean - d.per_class[b].centroid_hz.mean);
  const L = 110, W = 700, cw = (W - L) / 64, rh = 22, T = 6, H = T + order.length * rh + 34;
  let all = []; order.forEach(c => all.push(...d.per_class[c].mel_profile));
  all.sort((a, b) => a - b); const lo = all[Math.floor(all.length * .03)], hi = all[Math.floor(all.length * .99)];
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Average mel spectrum per class">`;
  order.forEach((c, r) => {
    g += `<text class="lbl ${d.alert_classes.includes(c) ? "threat" : ""}" x="${L - 8}" y="${T + r * rh + 15}" text-anchor="end">${esc(nice(c))}</text>`;
    d.per_class[c].mel_profile.forEach((v, k) => {
      const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo))), col = LUT[Math.round(t * 255)];
      g += `<rect x="${L + k * cw}" y="${T + r * rh}" width="${cw + .5}" height="${rh - 2}" fill="rgb(${col})"/>`;
    });
  });
  [[0, 0], [16, 16], [32, 32], [48, 48], [63, 63]].forEach(([k]) => {
    g += `<text x="${L + k * cw}" y="${H - 12}" text-anchor="${k === 63 ? "end" : "middle"}" style="fill:#5C6E69">${d.mel_freqs_hz[k] >= 1000 ? (d.mel_freqs_hz[k] / 1000).toFixed(1) + " kHz" : Math.round(d.mel_freqs_hz[k]) + " Hz"}</text>`;
  });
  return g + "</svg>";
}

/* ------------------------------------------------------------------ teach */
const bulkIn = $("#bulkFiles"), bulkDrop = $("#bulkDrop"); let bulk = [];
bulkDrop.onclick = () => bulkIn.click();
bulkDrop.onkeydown = e => { if (e.key === "Enter") bulkIn.click(); };
bulkIn.onchange = () => setBulk([...bulkIn.files]);
["dragenter", "dragover"].forEach(ev => bulkDrop.addEventListener(ev, e => { e.preventDefault(); bulkDrop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => bulkDrop.addEventListener(ev, e => { e.preventDefault(); bulkDrop.classList.remove("over"); }));
bulkDrop.addEventListener("drop", e => setBulk([...e.dataTransfer.files]));
$("#upLabel").oninput = syncUp;
function setBulk(files) { bulk = files.filter(f => /\.(wav|flac|ogg)$/i.test(f.name) || f.type.startsWith("audio/")); $("#bulkText").textContent = bulk.length ? `${bulk.length} clip${bulk.length > 1 ? "s" : ""} ready` : "No audio files found. Try WAV, FLAC or OGG."; syncUp(); }
function syncUp() { $("#upBtn").disabled = !(bulk.length && $("#upLabel").value.trim().length >= 2); }
$("#upBtn").onclick = async () => {
  const msg = $("#upMsg"); msg.className = "msg"; msg.textContent = `Uploading ${bulk.length} clips…`; $("#upBtn").disabled = true;
  const fd = new FormData(); fd.append("label", $("#upLabel").value);
  bulk.forEach(f => fd.append("files", f, f.name));
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    msg.className = "msg ok";
    msg.textContent = `Saved ${r.accepted} clip${r.accepted === 1 ? "" : "s"} as “${nice(r.label)}”.` + (r.rejected.length ? ` ${r.rejected.length} rejected: ${r.rejected[0].reason}` : "") + (r.auto_retrain_started ? " Enough new data: retraining started automatically." : "");
    bulk = []; bulkIn.value = ""; $("#bulkText").textContent = "Drop several clips here or click to choose";
    refreshUploads(); refreshRetrain(true);
  } catch (e) { msg.className = "msg err"; msg.textContent = e.message; syncUp(); }
};
async function refreshUploads() {
  try {
    const u = await api("/api/uploads"), classes = new Set((STATUS?.model.classes) || []); u.classes.forEach(c => classes.add(c.label));
    $("#classList").innerHTML = [...classes].sort().map(c => `<option value="${esc(c)}">`).join("");
    const max = Math.max(1, ...u.classes.map(c => c.n));
    $("#pendingList").innerHTML = u.classes.length ? u.classes.map(c => `<div class="pending-row"><span>${esc(nice(c.label))}</span><div class="conf"><div class="conf-fill" style="width:${c.n / max * 100}%"></div></div><span>${c.pending} new</span></div>`).join("") + `<p class="muted">${u.pending} clip${u.pending === 1 ? "" : "s"} not yet used for training.</p>` : "Nothing uploaded yet.";
  } catch (_) {}
}
const STAGES = [["collecting", "Collect uploads + original data"], ["preprocessing", "Preprocess audio into spectrograms"], ["tuning", "Train (fine-tune the current model)"], ["evaluating", "Evaluate on the held-out test clips"], ["saving", "Save and decide whether to promote"]];
function stageIndex(stage) {
  if (!stage) return -1; if (stage.startsWith("collecting")) return 0; if (stage.startsWith("preprocessing")) return 1;
  if (stage.startsWith("training") || stage.startsWith("fine-tuning")) return 2; if (stage.startsWith("evaluating")) return 3;
  if (stage.startsWith("saving")) return 4; return stage === "finished" ? 5 : -1;
}
let rtTimer = null;
async function refreshRetrain(poll) {
  let r; try { r = await api("/api/retrain/status"); } catch (_) { return; }
  $("#autoNote").textContent = r.auto_retrain ? `Retraining also starts by itself once ${r.auto_retrain_min_clips} new clips are waiting, or when live confidence drops and new clips exist.` : "Automatic retraining is off.";
  const rec = $("#reco"); rec.hidden = !r.recommendation.recommended; rec.textContent = r.recommendation.reason || "";
  const run = r.latest, running = r.running;
  $("#retrainBtn").disabled = running; $("#retrainBtn").textContent = running ? "Training…" : "Retrain now";
  const idx = run ? (run.status === "done" ? 5 : stageIndex(run.stage)) : -1;
  $("#stages").innerHTML = run && (running || run.status !== "done" || true) ? STAGES.map((s, i) => `<li class="${i < idx ? "done" : i === idx && running ? "now" : i <= idx ? "done" : ""}">${esc(i === 2 && run.mode === "scratch" ? "Train a new model from scratch" : s[1])}</li>`).join("") : "";
  if (run && !running) { $("#rtMsg").className = "msg " + (run.status === "failed" ? "err" : "ok"); $("#rtMsg").textContent = run.message || ""; }
  else if (running && run) { $("#rtMsg").className = "msg"; $("#rtMsg").textContent = run.stage; }
  $("#runs").innerHTML = r.history.length ? `<table><tr><th>#</th><th>Trigger</th><th>Result</th><th>Test acc.</th></tr>${r.history.map(h => `<tr><td>${h.id}</td><td>${esc(h.trigger)}<br><span class="muted">${esc(h.mode)}</span></td><td>${h.status === "running" ? '<span class="tag run">running</span>' : h.status === "failed" ? '<span class="tag bad">failed</span>' : `v${h.from_version ?? "–"} → v${h.to_version} ${h.promoted ? '<span class="tag">promoted</span>' : '<span class="tag bad">kept old</span>'}`}</td><td>${h.metrics ? pct(h.metrics.test.accuracy, 1) : "–"}</td></tr>`).join("")}</table>` : "No runs yet.";
  clearTimeout(rtTimer); if (running || poll) rtTimer = setTimeout(() => refreshRetrain(running), 2000);
}
$("#retrainBtn").onclick = async () => {
  $("#rtMsg").className = "msg"; $("#rtMsg").textContent = "Starting…";
  try { await api("/api/retrain?mode=" + $("#mode").value, { method: "POST" }); refreshRetrain(true); }
  catch (e) { $("#rtMsg").className = "msg err"; $("#rtMsg").textContent = e.message; }
};

/* ---------------------------------------------------------------- monitor */
async function refreshStatus() {
  try { STATUS = await api("/api/status"); } catch (e) {
    $("#pulse").innerHTML = `<span class="dot off"></span>Service unreachable`; return;
  }
  const s = STATUS, t = s.traffic, m = s.model;
  $("#pulse").innerHTML = `<span class="dot"></span><b>Online</b> · up ${dur(s.service.uptime_s)} · ${m.loaded ? `model v${m.version} (${m.classes.length} sounds)` : "<b>no model yet</b>"} · ${t.total} predictions`;
  if ($("#tab-monitor").classList.contains("on")) renderMonitor();
}
function tile(v, l) { return `<div class="tile"><b>${v}</b><span>${l}</span></div>`; }
function renderMonitor() {
  const s = STATUS, t = s.traffic, m = s.model, ms = v => v == null ? "–" : v.toFixed(0) + " ms";
  $("#tiles").innerHTML = tile(dur(s.service.uptime_s), "service up-time") + tile(t.total, "predictions served") + tile(t.last_minute, "in the last minute") +
    tile(ms(t.latency_p50_ms), "median response") + tile(ms(t.latency_p95_ms), "95th percentile") + tile(m.loaded ? pct(m.metrics.accuracy, 1) : "–", "test accuracy");
  const c = t.recent_mean_confidence, thr = s.thresholds.low_confidence, mix = Object.entries(t.label_mix), mx = Math.max(1, ...mix.map(x => x[1]));
  $("#traffic").innerHTML = `<p class="muted">Average confidence over the last ${t.recent_n} predictions. A falling line is an early sign that the sounds out there no longer match what the model learned.</p>
    <div class="gauge"><i style="width:${(c ?? 0) * 100}%;${c != null && c < thr ? "background:#D4452B" : ""}"></i><u style="left:${thr * 100}%" title="low-confidence line"></u></div>
    <p><b>${c == null ? "No traffic yet" : pct(c)}</b> <span class="muted">· red line = ${pct(thr)}</span></p>
    <h3>What it has been hearing</h3>${mix.length ? mix.map(([k, n]) => `<div class="pending-row"><span>${esc(nice(k))}</span><div class="conf"><div class="conf-fill ${ALERTS.has(k) ? "threat" : ""}" style="width:${n / mx * 100}%"></div></div><span>${n}</span></div>`).join("") : '<p class="muted">Predictions will appear here.</p>'}`;
  const mt = m.metrics || {}, row = (k, v) => `<dt>${k}</dt><dd>${v}</dd>`;
  $("#modelInfo").innerHTML = m.loaded ? `<dl class="kv">${row("Version", "v" + m.version + " · " + esc(m.file))}${row("Trained by", esc(m.trigger))}${row("Created", esc(m.created))}${row("Sounds it knows", m.classes.map(nice).map(esc).join(", "))}${row("Accuracy", pct(mt.accuracy, 1))}${row("Precision / Recall (macro)", pct(mt.precision_macro, 1) + " / " + pct(mt.recall_macro, 1))}${row("F1 macro / weighted", pct(mt.f1_macro, 1) + " / " + pct(mt.f1_weighted, 1))}${row("Test loss", mt.loss?.toFixed(3))}${row("Served by", esc(s.service.host))}</dl>` : '<p class="muted">No model is loaded. Open Teach &amp; retrain and train one from scratch.</p>';
}
async function loadModel() {
  try { MODEL = await api("/api/model"); } catch (_) { return; }
  const cur = MODEL.versions.find(v => v.version === MODEL.current);
  $("#versions").innerHTML = `<table class="t"><tr><th>Ver.</th><th>Source</th><th>Acc.</th><th>F1</th><th>Sounds</th><th></th></tr>${MODEL.versions.map(v => `<tr><td>v${v.version}</td><td>${esc(v.trigger)}</td><td>${pct(v.metrics.accuracy, 1)}</td><td>${pct(v.metrics.f1_macro, 1)}</td><td>${v.classes.length}</td><td>${v.version === MODEL.current ? '<span class="tag">live</span>' : v.promoted ? "" : '<span class="tag bad">rejected</span>'}</td></tr>`).join("")}</table>`;
  if (!cur || !cur.metrics.confusion_matrix) { $("#cm").innerHTML = '<p class="muted">No evaluation yet.</p>'; return; }
  const cl = cur.metrics.classes || cur.classes, cm = cur.metrics.confusion_matrix, n = cl.length;
  const mx = Math.max(1, ...cm.flat());
  let h = `<div class="cmgrid" style="grid-template-columns:auto repeat(${n},1fr)"><div></div>${cl.map(c => `<div class="h" title="${esc(nice(c))}">${esc(c.slice(0, 4))}</div>`).join("")}`;
  cm.forEach((r, i) => { h += `<div class="rowl h">${esc(nice(cl[i]))}</div>` + r.map((v, j) => { const a = v / mx, bad = i !== j && v > 0; return `<div title="true ${esc(cl[i])} → said ${esc(cl[j])}: ${v}" style="background:${bad ? `rgba(212,69,43,${.15 + a * .85})` : `rgba(31,111,102,${.08 + a * .92})`};color:${a > .5 ? "#fff" : INK}">${v || ""}</div>`; }).join(""); });
  $("#cm").innerHTML = h + "</div>";
}

/* ------------------------------------------------------------------- boot */
(async function () {
  await refreshStatus(); loadSamples();
  const start = location.hash.slice(1); showTab(["listen", "insights", "teach", "monitor"].includes(start) ? start : "listen");
  setInterval(refreshStatus, 4000);
})();
