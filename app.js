// E-nose Predictor (static site): the model runs here in the browser (enose.js); the Live tab reads
// the e-nose over USB with Web Serial (Chrome / Edge).
import { Engine, Model, Player, parseUpload, livePipeline } from "./enose.js";

// ---------- state ----------
const S = {
  mode: "file", phase: "idle", items: [], categories: [], window: 50,
  samples: [],          // current burst: {t, Sensor1..4, Temperature, Humidity}
  burstNo: 1, device: null, nWindows: 0,
  bursts: [], liveBurst: null, sample: null, playback: null,
  bands: [],            // analysed windows to flash on the charts: {t0, born}
  photoItem: null,
};
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (p) => `${Math.round(p * 100)}%`;
const nice = (s) => (s || "").replace(/-/g, " ");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
// air (nothing in the chamber) is grey and last, so the other categories keep their colours
const catColor = (cat) => {
  const cats = S.categories.filter((c) => c !== "air").concat("air");
  return css(`--c${Math.max(0, cats.indexOf(cat)) % 7}`);
};

// ---------- charts ----------
const CHANNELS = [
  { key: "Sensor1", label: "Sensor 1", unit: "ADC", color: "--s1", gas: true },
  { key: "Sensor2", label: "Sensor 2", unit: "ADC", color: "--s2", gas: true },
  { key: "Sensor3", label: "Sensor 3", unit: "ADC", color: "--s3", gas: true },
  { key: "Sensor4", label: "Sensor 4", unit: "ADC", color: "--s4", gas: true },
  { key: "Temperature", label: "Temperature", unit: "°C", color: "--temp" },
  { key: "Humidity", label: "Humidity", unit: "%RH", color: "--rh" },
];
const charts = CHANNELS.map((ch) => {
  const el = document.createElement("div");
  el.className = "chart";
  el.innerHTML = `<div class="chart-top"><span class="name" style="color:var(${ch.color})">${ch.label}</span>
    <span><span class="delta"></span> <span class="value">—</span> <span class="muted small">${ch.unit}</span></span></div>
    <canvas></canvas>`;
  $("charts").appendChild(el);
  return { ...ch, canvas: el.querySelector("canvas"), value: el.querySelector(".value"), delta: el.querySelector(".delta") };
});

function fitCanvas(c) {
  const r = c.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  if (c.width !== Math.round(r.width * d) || c.height !== Math.round(r.height * d)) {
    c.width = Math.round(r.width * d); c.height = Math.round(r.height * d);
  }
  const ctx = c.getContext("2d");
  ctx.setTransform(d, 0, 0, d, 0, 0);
  return [ctx, r.width, r.height];
}

function baseline(key) {
  const first = S.samples.filter((s) => s.t < 1);
  if (!first.length) return null;
  return first.reduce((a, s) => a + s[key], 0) / first.length;
}

function drawCharts(now) {
  const tMax = Math.max(60, S.samples.length ? S.samples[S.samples.length - 1].t : 0);
  for (const ch of charts) {
    const [ctx, w, h] = fitCanvas(ch.canvas);
    ctx.clearRect(0, 0, w, h);
    const padL = 4, padR = 4, padT = 6, padB = 14;
    const X = (t) => padL + (t / tMax) * (w - padL - padR);
    // grid + time axis
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1; ctx.fillStyle = css("--muted");
    ctx.font = "10px -apple-system, sans-serif"; ctx.textAlign = "center";
    for (let t = 0; t <= tMax; t += 10) {
      ctx.beginPath(); ctx.moveTo(X(t), padT); ctx.lineTo(X(t), h - padB); ctx.stroke();
      ctx.fillText(`${t}s`, Math.min(Math.max(X(t), 10), w - 12), h - 2);
    }
    // windows the model just analysed
    for (const b of S.bands) {
      const age = (now - b.born) / 1000;
      if (age > 3) continue;
      ctx.fillStyle = css("--accent");
      ctx.globalAlpha = 0.22 * (1 - age / 3);
      ctx.fillRect(X(b.t0), padT, X(b.t0 + 5) - X(b.t0), h - padT - padB);
      ctx.globalAlpha = 1;
    }
    const pts = S.samples;
    if (!pts.length) continue;
    let lo = Infinity, hi = -Infinity;
    for (const p of pts) { lo = Math.min(lo, p[ch.key]); hi = Math.max(hi, p[ch.key]); }
    const m = (hi - lo) * 0.12 || Math.max(1, Math.abs(hi) * 0.01);
    lo -= m; hi += m;
    const Y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (h - padT - padB);
    ctx.strokeStyle = css(ch.color); ctx.lineWidth = 1.6; ctx.lineJoin = "round";
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(X(p.t), Y(p[ch.key])) : ctx.moveTo(X(p.t), Y(p[ch.key]))));
    ctx.stroke();
    const last = pts[pts.length - 1];
    ctx.fillStyle = css(ch.color);
    ctx.beginPath(); ctx.arc(X(last.t), Y(last[ch.key]), 3, 0, 7); ctx.fill();
  }
}

function updateChartLabels() {
  const last = S.samples[S.samples.length - 1];
  for (const ch of charts) {
    if (!last) { ch.value.textContent = "—"; ch.delta.textContent = ""; continue; }
    const v = last[ch.key];
    ch.value.textContent = ch.gas ? Math.round(v).toLocaleString() : v.toFixed(1);
    const b = ch.gas ? baseline(ch.key) : null;
    ch.delta.textContent = b ? `${(100 * (v - b) / b >= 0 ? "+" : "")}${(100 * (v - b) / b).toFixed(1)}%` : "";
  }
}

// ---------- chamber animation ----------
const chamber = { particles: [], signals: [], glow: [0, 0, 0, 0] };

function sensorResponse() {
  // |% change| of each gas sensor from burst start, for the chip glow
  const last = S.samples[S.samples.length - 1];
  return CHANNELS.slice(0, 4).map((ch) => {
    const b = last ? baseline(ch.key) : null;
    if (!b || S.device === 1 && ch.key === "Sensor1") return 0;
    return Math.min(1, Math.abs(100 * (last[ch.key] - b) / b) / 25);
  });
}

function drawChamber(now, dt) {
  const [ctx, w, h] = fitCanvas($("chamber"));
  ctx.clearRect(0, 0, w, h);
  const cx0 = w * 0.16, cx1 = w * 0.84, cy0 = 22, cy1 = h - 34;
  const pipeY = (cy0 + cy1) / 2 - 10, pipeH = 18;
  const phase = S.phase;
  const hl = headline();
  const odor = hl && hl.res.decided ? catColor(hl.res.category) : css("--s3");

  // pipes
  ctx.fillStyle = css("--chamber"); ctx.strokeStyle = css("--chamber-edge"); ctx.lineWidth = 1.5;
  for (const [x0, x1] of [[0, cx0], [cx1, w]]) {
    ctx.fillRect(x0, pipeY - pipeH / 2, x1 - x0, pipeH);
    ctx.beginPath(); ctx.moveTo(x0, pipeY - pipeH / 2); ctx.lineTo(x1, pipeY - pipeH / 2);
    ctx.moveTo(x0, pipeY + pipeH / 2); ctx.lineTo(x1, pipeY + pipeH / 2); ctx.stroke();
  }
  // chamber body
  ctx.beginPath(); ctx.roundRect(cx0, cy0, cx1 - cx0, cy1 - cy0, 18);
  ctx.fillStyle = css("--chamber"); ctx.fill(); ctx.stroke();
  ctx.fillStyle = css("--muted"); ctx.font = "11px -apple-system, sans-serif"; ctx.textAlign = "left";
  ctx.fillText("sample in", 6, pipeY - 16);
  ctx.textAlign = "right"; ctx.fillText("exhaust", w - 6, pipeY - 16);

  // spawn particles by phase
  const P = chamber.particles;
  const spawn = (n, color, fromX, vx) => {
    for (let i = 0; i < n; i++) P.push({ x: fromX, y: pipeY + (Math.random() - 0.5) * (pipeH - 6),
      vx: vx * (0.7 + Math.random() * 0.6), vy: 0, life: 1, color, r: 2 + Math.random() * 2.2, inside: false });
  };
  if ((phase === "filling" || phase === "filled") && P.length < 160) spawn(dt * 50 > Math.random() ? 2 : 0, odor, 0, 140);
  if (phase === "flushing" && Math.random() < dt * 30) spawn(1, css("--rh"), 0, 260);
  if (phase === "measuring" && P.length < 90) {   // page opened mid-measurement: gas is already inside
    P.push({ x: cx0 + 14 + Math.random() * (cx1 - cx0 - 28), y: cy0 + 30 + Math.random() * (cy1 - cy0 - 62),
      vx: 0, vy: 0, life: 1, color: odor, r: 2 + Math.random() * 2.2, inside: true });
  }

  const swirl = phase === "measuring" || phase === "filled" || phase === "filling";
  for (const p of P) {
    if (!p.inside && p.x > cx0 + 8) p.inside = true;
    if (phase === "flushing") {
      p.vx += 220 * dt; p.vy += (pipeY - p.y) * 2.5 * dt;
    } else if (p.inside && swirl) {
      // drift towards a private target inside the chamber, pick a new one on arrival -> even diffusion
      if (p.tx === undefined || Math.hypot(p.tx - p.x, p.ty - p.y) < 12) {
        p.tx = cx0 + 14 + Math.random() * (cx1 - cx0 - 28);
        p.ty = cy0 + 30 + Math.random() * (cy1 - cy0 - 62);
      }
      p.vx += ((p.tx - p.x) * 1.4 + (Math.random() - 0.5) * 160) * dt;
      p.vy += ((p.ty - p.y) * 1.4 + (Math.random() - 0.5) * 160) * dt;
      const sp = Math.hypot(p.vx, p.vy), max = phase === "measuring" ? 55 : 90;
      if (sp > max) { p.vx *= max / sp; p.vy *= max / sp; }
    } else if (p.inside) {
      p.vx *= 0.97; p.vy *= 0.97; p.life -= dt * 0.25;
    }
    p.x += p.vx * dt; p.y += p.vy * dt;
    if (p.inside && phase !== "flushing") {
      if (p.x < cx0 + 6) { p.x = cx0 + 6; p.vx = Math.abs(p.vx); }
      if (p.x > cx1 - 6) { p.x = cx1 - 6; p.vx = -Math.abs(p.vx); }
      if (p.y < cy0 + 6) { p.y = cy0 + 6; p.vy = Math.abs(p.vy); }
      if (p.y > cy1 - 26) { p.y = cy1 - 26; p.vy = -Math.abs(p.vy); }
    }
    if (p.x > w + 10) p.life = 0;
  }
  chamber.particles = P.filter((p) => p.life > 0);
  for (const p of chamber.particles) {
    ctx.globalAlpha = Math.max(0, Math.min(1, p.life)) * 0.85;
    ctx.fillStyle = p.color; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 7); ctx.fill();
  }
  ctx.globalAlpha = 1;

  // sensor chips on the floor
  const resp = sensorResponse();
  const chipW = Math.min(54, (cx1 - cx0) / 6), gap = ((cx1 - cx0) - chipW * 4) / 5;
  const chipY = cy1 - 20;
  CHANNELS.slice(0, 4).forEach((ch, i) => {
    chamber.glow[i] += ((phase === "measuring" ? resp[i] : 0) - chamber.glow[i]) * Math.min(1, dt * 4);
    const x = cx0 + gap + i * (chipW + gap);
    const g = chamber.glow[i], col = css(ch.color);
    if (g > 0.02) {
      const r = 14 + 26 * g;
      const grd = ctx.createRadialGradient(x + chipW / 2, chipY, 2, x + chipW / 2, chipY, r);
      grd.addColorStop(0, col); grd.addColorStop(1, col + "00");
      ctx.globalAlpha = 0.2 + 0.45 * g; ctx.fillStyle = grd;
      ctx.beginPath(); ctx.arc(x + chipW / 2, chipY, r, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
    }
    ctx.fillStyle = css("--card"); ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.roundRect(x, chipY - 7, chipW, 14, 4); ctx.fill(); ctx.stroke();
    ctx.fillStyle = col; ctx.font = "600 10px -apple-system, sans-serif"; ctx.textAlign = "center";
    ctx.fillText(`S${i + 1}`, x + chipW / 2, chipY + 4);
  });
  // signal pulses run down a wire to the model each time a window is analysed
  const wireY = cy1 + 14;
  ctx.strokeStyle = css("--line"); ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
  ctx.beginPath(); ctx.moveTo(cx0 + gap + chipW / 2, wireY); ctx.lineTo(w - 4, wireY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = css("--muted"); ctx.font = "10px -apple-system, sans-serif"; ctx.textAlign = "right";
  ctx.fillText("to model →", w - 4, wireY + 12);
  for (const s of chamber.signals) {
    s.k += dt * 0.9;
    if (s.k < 0) continue;
    const x0 = cx0 + gap + s.i * (chipW + gap) + chipW / 2, drop = 0.15;
    const x = s.k < drop ? x0 : x0 + (w - 4 - x0) * (s.k - drop) / (1 - drop);
    const y = s.k < drop ? chipY + 7 + (wireY - chipY - 7) * (s.k / drop) : wireY;
    ctx.fillStyle = css(CHANNELS[s.i].color); ctx.globalAlpha = Math.min(1, 3 * (1 - s.k));
    ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
  }
  chamber.signals = chamber.signals.filter((s) => s.k < 1);

  // phase label
  const label = { idle: "WAITING FOR DEVICE", filling: "FILLING", filled: "CHAMBER FILLED", measuring: "MEASURING",
    finished: "MEASURED", flushing: "FLUSHING", ready: "CLEAN · READY" }[phase] || phase.toUpperCase();
  ctx.textAlign = "left"; ctx.font = "600 11px ui-monospace, Menlo, monospace";
  ctx.fillStyle = phase === "measuring" ? css("--accent") : css("--muted");
  const t = S.samples.length ? S.samples[S.samples.length - 1].t : 0;
  ctx.fillText(phase === "measuring" ? `● ${label}  ${t.toFixed(1)} s` : label, cx0 + 14, cy0 + 20);
  if (phase === "measuring") {
    const narrow = cx1 - cx0 < 360;
    ctx.textAlign = narrow ? "left" : "right"; ctx.fillStyle = css("--muted");
    ctx.fillText(`${S.samples.length} rows · ${S.nWindows} windows`, narrow ? cx0 + 14 : cx1 - 14, cy0 + (narrow ? 36 : 20));
  }
}

// ---------- model "thinking" overlay ----------
const brain = { pulses: [], nodes: null, scan: 0 };
function drawBrain(now, dt) {
  const [ctx, w, h] = fitCanvas($("brain"));
  ctx.clearRect(0, 0, w, h);
  const thinking = S.phase === "measuring";
  const hasPhoto = !$("photo").hidden;
  if (!brain.nodes) {
    const layers = [4, 6, 6, 3];
    brain.nodes = layers.map((n, li) => Array.from({ length: n }, (_, k) => ({
      x: (li + 0.5) / layers.length, y: (k + 1) / (n + 1) })));
  }
  if (hasPhoto && !thinking) return;
  const alpha = hasPhoto ? 0.55 : 1;
  if (hasPhoto) { ctx.fillStyle = css("--card"); ctx.globalAlpha = 0.45; ctx.fillRect(0, 0, w, h); }
  ctx.globalAlpha = alpha;
  const N = brain.nodes, acc = css("--accent");
  ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
  for (let l = 0; l < N.length - 1; l++)
    for (const a of N[l]) for (const b of N[l + 1]) {
      ctx.beginPath(); ctx.moveTo(a.x * w, a.y * h); ctx.lineTo(b.x * w, b.y * h); ctx.stroke();
    }
  // pulses run layer by layer
  for (const p of brain.pulses) {
    p.k += dt * 1.8;
    const l = Math.floor(p.k * (N.length - 1));
    if (l < 0 || l >= N.length - 1) continue;
    const f = p.k * (N.length - 1) - l;
    const a = N[l][p.path[l] % N[l].length], b = N[l + 1][p.path[l + 1] % N[l + 1].length];
    ctx.fillStyle = acc; ctx.beginPath();
    ctx.arc((a.x + (b.x - a.x) * f) * w, (a.y + (b.y - a.y) * f) * h, 3, 0, 7); ctx.fill();
  }
  brain.pulses = brain.pulses.filter((p) => p.k < 1);
  const idleGlow = thinking ? 0.5 + 0.5 * Math.sin(now / 300) : 0.2;
  N.forEach((layer, li) => layer.forEach((n, k) => {
    const lit = brain.pulses.some((p) => Math.abs(p.k * (N.length - 1) - li) < 0.25 && p.path[li] % layer.length === k);
    ctx.fillStyle = lit ? acc : css("--card");
    ctx.strokeStyle = acc; ctx.lineWidth = 1.5;
    ctx.globalAlpha = alpha * (lit ? 1 : 0.5 + 0.5 * idleGlow);
    ctx.beginPath(); ctx.arc(n.x * w, n.y * h, lit ? 6 : 4.5, 0, 7); ctx.fill(); ctx.stroke();
  }));
  if (thinking) {   // scanning line
    brain.scan = (brain.scan + dt * 0.6) % 1;
    const y = brain.scan * h;
    const grd = ctx.createLinearGradient(0, y - 24, 0, y);
    grd.addColorStop(0, "transparent"); grd.addColorStop(1, acc);
    ctx.globalAlpha = 0.35; ctx.fillStyle = grd; ctx.fillRect(0, y - 24, w, 24);
    ctx.globalAlpha = 1; ctx.fillStyle = acc; ctx.fillRect(0, y, w, 1.5);
  }
  ctx.globalAlpha = 1;
}

function firePulses() {
  for (let i = 0; i < 6; i++)
    brain.pulses.push({ k: -i * 0.08, path: Array.from({ length: 4 }, () => Math.floor(Math.random() * 6)) });
  for (let i = 0; i < 4; i++) chamber.signals.push({ i, k: -i * 0.05 });
}

// ---------- prediction panel ----------
// What the big result shows: the burst being measured, else (file mode) the sample = all bursts
// averaged, else the last burst.
function headline() {
  if (S.phase === "measuring" && S.liveBurst) return { kind: "live", res: S.liveBurst };
  if (S.mode === "file" && S.sample) return { kind: "sample", res: S.sample };
  const b = S.bursts.at(-1);
  return b ? { kind: "burst", res: b.result, burst: b } : null;
}

function renderBars(el, rows, nameKey, colorOf) {
  el.innerHTML = rows.map((r) => `<div class="bar"><span class="name">${nice(r[nameKey])}</span>
    <span class="track"><span class="fill" style="background:${colorOf(r)}" data-w="${(r.p * 100).toFixed(1)}"></span></span>
    <span class="val">${pct(r.p)}</span></div>`).join("");
  requestAnimationFrame(() => el.querySelectorAll(".fill").forEach((f) => (f.style.width = f.dataset.w + "%")));
}

function setPhoto(item) {
  const img = $("photo");
  if (!item) { img.hidden = true; S.photoItem = null; return; }
  if (S.photoItem === item) return;
  S.photoItem = item;
  img.onload = () => (img.hidden = false);
  img.onerror = () => (img.hidden = true);
  img.src = `photos/${encodeURIComponent(item)}.jpg`;
}

// the three most likely items, and how often the right item was among them on unseen data
function renderTop3(r) {
  const list = $("v-top3"), note = $("v-top3-note"), rel = r.reliability;
  list.innerHTML = r.top.slice(0, 3).map((t) => `<li><span class="dot3" style="background:${catColor(t.category)}"></span>` +
    `<b>${nice(t.item)}</b> <span class="muted">${nice(t.category)}</span><span class="p3">${pct(t.p)}</span></li>`).join("");
  list.hidden = false;
  if (rel && rel.top3 != null) {
    note.textContent = `the right item was in the top 3 for ${pct(rel.top3)} of ${rel.level === "sample" ? "files" : "bursts"} ` +
      `at this confidence on unseen data`;
    note.hidden = false;
  }
}

// how often predictions at this confidence were right on data the model never saw
function relText(rel) {
  return `on unseen data, ${rel.level === "sample" ? "files" : "bursts"} at this confidence were right ` +
    `${pct(rel.right)} of the time (${rel.n} cases)`;
}

function truthLine(truth, decided, correct) {
  if (!truth) return null;
  if (!decided) return { cls: "na", text: `not sure · actual: ${nice(truth.item)}` };
  return correct ? { cls: "ok", text: "✓ correct" } : { cls: "no", text: `✗ wrong · actual: ${nice(truth.item)}` };
}

function renderVerdict() {
  const h = headline(), measuring = S.phase === "measuring";
  $("model-card").classList.toggle("thinking", measuring);
  const status = $("model-status"), truthEl = $("v-truth"), ring = $("ring");
  truthEl.hidden = true; $("v-top3").hidden = true; $("v-top3-note").hidden = true;
  if (!h) {
    $("result-title").textContent = "Prediction";
    status.textContent = measuring ? "analysing" : "idle"; status.className = "badge" + (measuring ? " live" : "");
    $("v-item").textContent = measuring ? "Analysing…" : "—"; $("v-item").className = "v-item unsure";
    $("v-category").textContent = "—"; $("v-category").style.color = "";
    $("v-conf").textContent = "—";
    $("v-basis").textContent = measuring
      ? `collecting the first 5 s window (${Math.min(S.samples.length, S.window)}/${S.window} rows)` : "no data yet";
    ring.style.strokeDashoffset = 97.4;
    $("item-bars").innerHTML = ""; $("cat-bars").innerHTML = "";
    setPhoto(null);
    return;
  }
  const r = h.res;
  $("result-title").textContent = h.kind === "live" ? `Burst ${r.burst} · analysing`
    : h.kind === "sample" ? (r.n_bursts === 1 ? "Result · 1 burst"
      : r.bursts_used && r.bursts_used < r.n_bursts ? `Sample result · ${r.bursts_used} of ${r.n_bursts} bursts used (others unknown data)`
      : `Sample result · ${r.n_bursts} bursts averaged`) : `Burst ${r.burst} result`;
  if (measuring) { status.textContent = "analysing"; status.className = "badge live"; }
  else if (r.decided) { status.textContent = "predicted"; status.className = "badge done"; }
  else { status.textContent = "unknown data"; status.className = "badge unsure"; }

  if (r.decided) {
    $("v-item").textContent = nice(r.item); $("v-item").className = "v-item";
    $("v-category").textContent = nice(r.category); $("v-category").style.color = catColor(r.category);
    const rel = r.reliability;
    ring.style.stroke = !rel ? css("--good") : rel.right >= 0.8 ? css("--good") : rel.right >= 0.55 ? css("--warn") : css("--bad");
    $("v-basis").textContent = rel ? relText(rel) : "confidence of the top item";
    renderTop3(r);
    setPhoto(r.item);
  } else {
    $("v-item").textContent = "Not sure"; $("v-item").className = "v-item unsure";
    $("v-category").textContent = "—"; $("v-category").style.color = "";
    ring.style.stroke = css("--warn");
    $("v-basis").textContent = "this data is unlike anything the model was trained on";
    setPhoto(null);
  }
  $("v-conf").textContent = pct(r.confidence);
  ring.style.strokeDashoffset = 97.4 * (1 - r.confidence);
  renderBars($("item-bars"), r.top, "item", (x) => catColor(x.category));
  renderBars($("cat-bars"), r.categories, "category", (x) => catColor(x.category));

  const t = h.kind === "sample" ? truthLine(r.truth, r.decided, r.correct)
    : h.kind === "burst" ? truthLine(h.burst.truth, r.decided, h.burst.correct) : null;
  if (t && !measuring) { truthEl.hidden = false; truthEl.className = `v-truth ${t.cls}`; truthEl.textContent = t.text; }
}

// ---------- phase / steps ----------
const PHASE_STEP = { idle: -1, filling: 0, filled: 0, measuring: 1, finished: 2, flushing: 2, ready: 4 };
function renderPhase() {
  const step = PHASE_STEP[S.phase] ?? -1;
  const activeIdx = { filling: 0, filled: 0, measuring: 1, flushing: 2 }[S.phase];
  document.querySelectorAll("#steps li").forEach((li, i) => {
    li.classList.toggle("active", i === activeIdx);
    li.classList.toggle("done", i < step || (S.phase === "ready" && i <= 3) || (S.phase === "finished" && i < 2));
  });
  const t = S.samples.length ? S.samples[S.samples.length - 1].t : 0;
  const prog = { idle: 0, filling: 12, filled: 20, measuring: 20 + 65 * Math.min(1, t / 60),
    finished: 85, flushing: 92, ready: 100 }[S.phase] ?? 0;
  $("progress-bar").style.width = prog + "%";
  const b = S.burstNo, file = S.mode === "file", p = S.playback;
  let text = {
    idle: file ? "Upload a CSV file to predict it." : port ? "Waiting for the e-nose. Press the start button on the device."
      : "Connect the e-nose (USB) with the button above, then press its start button.",
    filling: `Burst ${b}: filling the chamber with the sample's headspace…`,
    filled: `Burst ${b}: chamber filled, measurement starting…`,
    measuring: `Burst ${b}: collecting sensor data (${t.toFixed(1)} s) — the model scores every 5 s window as it arrives.`,
    finished: `Burst ${b}: measurement finished.`,
    flushing: `Flushing gas out of the chamber…`,
    ready: file ? "Chamber clean." : "Chamber clean. Press the device button for the next burst.",
  }[S.phase] || S.phase;
  if (file && p && !p.playing && (p.done || p.stopped))
    text = `${p.stopped ? "Stopped" : "Finished"}: ${esc(p.label ? p.label.name : "sample")} · ${S.bursts.length} burst${S.bursts.length === 1 ? "" : "s"} played.`;
  $("phase-text").textContent = text;
  if (file && p && p.label) {
    $("burst-label").textContent = p.playing ? `${p.label.name} · burst ${p.burst} of ${p.n_bursts}` : p.label.name;
  } else {
    $("burst-label").textContent = S.phase !== "idle" || S.bursts.length ? `burst ${b}` : "burst —";
  }
}

// ---------- bursts ----------
function resultLines(r) {
  return r.decided
    ? `<div class="item">${nice(r.item)}</div><div class="cat" style="color:${catColor(r.category)}">${nice(r.category)} · ${pct(r.confidence)}</div>` +
      `<div class="rel">top 3: ${r.top.slice(0, 3).map((t) => `${nice(t.item)} ${pct(t.p)}`).join(" · ")}</div>` +
      (r.reliability ? `<div class="rel">right ${pct(r.reliability.right)} of the time at this confidence` +
        (r.reliability.top3 != null ? `; in the top 3 ${pct(r.reliability.top3)}` : "") + `</div>` : "")
    : `<div class="item unsure">Not sure</div><div class="cat note">unlike the training data</div>`;
}
function renderBursts() {
  const cards = S.bursts.map((b) => {
    const t = truthLine(b.truth, b.result.decided, b.correct);
    const pl = b.pipeline;
    const steps = pl && pl.steps ? `<details class="steps-detail"><summary>saved, then processed like the training data</summary>
      <ol>${pl.steps.map((x) => `<li>${esc(x)}</li>`).join("")}</ol></details>`
      : pl && pl.error ? `<div class="truth no">processing the saved file failed: ${esc(pl.error)}</div>` : "";
    return `<div class="burst ${t && t.cls !== "na" ? t.cls : ""}"><div class="head"><span>Burst ${b.number} · device ${b.device}</span>
      <span>${b.n_windows} windows</span></div>${resultLines(b.result)}${t ? `<div class="truth ${t.cls}">${t.text}</div>` : ""}${steps}</div>`;
  });
  if (S.phase === "measuring") {
    const r = S.liveBurst;
    cards.push(`<div class="burst live"><div class="head"><span>Burst ${S.burstNo} · analysing</span><span>${S.nWindows} windows</span></div>
      ${r ? resultLines(r) : `<div class="item unsure">Measuring…</div><div class="cat note">waiting for the first 5 s window</div>`}</div>`);
  }
  $("bursts").innerHTML = cards.length ? cards.join("") : `<p class="muted empty">No bursts yet.</p>`;
  const n = S.bursts.length, answered = S.bursts.filter((b) => b.result.decided).length;
  const scored = S.bursts.filter((b) => b.truth);
  $("score").textContent = !n ? "every burst is predicted on its own"
    : scored.length ? `${scored.filter((b) => b.correct).length} of ${n} right` + (n - answered ? ` · ${n - answered} not sure (unknown data)` : "")
    : n - answered ? `${n - answered} of ${n} not sure (unknown data)` : `${n} burst${n > 1 ? "s" : ""} predicted`;
}

// ---------- device lines, appended in batches (the device sends ~10 lines/s) ----------
let rawQueue = [], rawCount = 0;
const rawTimes = [];
function flushRaw() {
  const el = $("raw");
  if (rawQueue.length) {
    const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
    const frag = document.createDocumentFragment();
    for (const r of rawQueue) {
      const div = document.createElement("div");
      const isRow = /Sensor4\s*:/i.test(r.text);
      div.className = isRow ? "" : /^[A-Z .]+$/.test(r.text) ? "status" : "bad";
      div.textContent = `${r.time}  ${r.text}`;
      frag.appendChild(div);
      rawTimes.push(performance.now());
    }
    rawQueue = [];
    el.appendChild(frag);
    while (el.children.length > 200) el.firstChild.remove();
    if (stick) el.scrollTop = el.scrollHeight;
  }
  const now = performance.now();
  while (rawTimes.length && now - rawTimes[0] > 3000) rawTimes.shift();
  $("raw-rate").textContent = `${rawCount} lines · ${(rawTimes.length / 3).toFixed(1)} lines/s`;
}
setInterval(flushRaw, 250);

function renderConnection(c) {
  S.conn = c;
  const pill = $("conn");
  if (S.mode === "file") {
    pill.className = "pill rec"; $("conn-text").textContent = "model loaded · runs in your browser"; pill.title = "";
    return;
  }
  pill.className = "pill " + ({ connected: "ok", error: "err", waiting: "wait" }[c.state] || "");
  $("conn-text").textContent = c.detail || c.state;
  pill.title = c.detail || c.state;
}

function renderDevice() {
  $("device").textContent = S.device ? `device ${S.device}${S.device === 1 ? " · Sensor1 dead" : ""}` : "device —";
}

function renderMode() {
  const file = S.mode === "file";
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.mode === S.mode));
  $("file-controls").hidden = !file; $("live-controls").hidden = file;
  $("raw-title").textContent = file ? "Device lines sent to the model (replayed)" : "Incoming serial data — exact lines from the device";
  renderLiveButtons();
  const playing = !!(S.playback && S.playback.playing);
  $("stop").hidden = !playing;
}

// ---------- events ----------
function applyState(st) {
  Object.assign(S, { mode: st.mode, phase: st.phase, items: st.items, categories: st.categories, window: st.window,
    bursts: st.bursts, sample: st.sample, playback: st.playback });
  S.samples = st.current ? st.recent : [];
  S.burstNo = st.current ? st.current.number : (st.bursts.at(-1)?.number || 0) + 1;
  S.nWindows = st.current ? st.current.n_windows : 0;
  S.liveBurst = st.current ? st.current.result : null;
  S.device = st.current?.device || st.bursts.at(-1)?.device || null;
  $("raw").innerHTML = ""; rawQueue = st.raw; rawCount = st.n_lines; flushRaw();
  if (S.phase === "idle") { chamber.particles = []; S.bands = []; }
  renderMode(); renderConnection(st.connection);
  renderAll();
}
function renderAll() { renderPhase(); renderVerdict(); renderBursts(); renderDevice(); updateChartLabels(); }

// events, by kind (the server numbers them; the page polls for the ones it has not seen)
let labelTimer = 0;
const handlers = {
  state: (d) => applyState(d),
  connection: (d) => renderConnection(d),
  phase: (d) => {
    S.phase = d.phase; S.burstNo = d.burst;
    if (d.phase === "measuring") { S.samples = []; S.nWindows = 0; S.liveBurst = null; S.bands = []; }
    if (d.phase === "filling") chamber.particles = chamber.particles.filter((p) => !p.inside);
    renderAll();
  },
  sample: (d) => {
    S.samples.push(d); S.device = d.device; S.burstNo = d.burst;
    if (S.phase !== "measuring") { S.phase = "measuring"; renderAll(); }
    $("rows-label").textContent = `${d.n} rows · burst ${d.burst}`;
    const now = performance.now();
    if (now - labelTimer > 200) {
      labelTimer = now; updateChartLabels(); renderPhase(); renderDevice();
      if (!S.liveBurst) renderVerdict();
    }
  },
  window: (d) => {
    const r = d.burst_result;
    S.nWindows = d.window; S.liveBurst = r;
    S.bands.push({ t0: d.t_start, born: performance.now() });
    $("window-ticker").textContent = `${S.mode === "live" ? "Preview — " : ""}window ${d.window} of burst ${d.burst} scored · so far: ` +
      (r.decided ? `${nice(r.item)} ${pct(r.confidence)}` : "not sure (unlike the training data)");
    firePulses(); renderVerdict(); renderBursts();
  },
  burst: (d) => {
    const r = d.result;
    S.bursts.push(d); S.bursts = S.bursts.slice(-20); S.sample = d.sample; S.liveBurst = null; S.device = d.device;
    $("window-ticker").textContent = (d.pipeline && d.pipeline.saved_to
      ? `Burst ${d.number} saved to ${d.pipeline.saved_to} and processed like the training data: `
      : `Burst ${d.number} finished (${d.n_windows} windows, ${d.duration_s} s): `) +
      (r.decided ? `${nice(r.item)} (${nice(r.category)}) ${pct(r.confidence)}` : "not sure (unlike the training data)");
    renderAll();
  },
  short: (d) => { $("window-ticker").textContent = `Burst ${d.burst} had only ${d.rows} rows (needs ${d.need}) — not scored.`; },
  raw: (d) => { rawQueue.push(d); rawCount = d.n; },
  playback: (d) => { S.playback = d; renderMode(); renderPhase(); renderVerdict(); },
};

// ---------- engines: one for uploads, one for the live sensor; both run in this page ----------
const io = {
  json: async (p) => { const r = await fetch(p); if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); },
  buffer: async (p) => { const r = await fetch(p); if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.arrayBuffer(); },
};
let META = null, LIBRARY = null, fileEngine = null, liveEngine = null, player = null;
const foldModels = {};
const engineFor = (mode) => (mode === "live" ? liveEngine : fileEngine);
const route = (kind, data, eng) => {
  if (eng.mode !== S.mode) return;
  try { handlers[kind]?.(data); } catch (err) { console.error(err); }
};

function connect(mode) {
  S.mode = mode; renderMode();
  const eng = engineFor(mode);
  if (eng) applyState(eng.snapshot());
}

async function boot() {
  const pill = $("conn");
  pill.className = "pill wait"; $("conn-text").textContent = "loading the model…";
  try {
    [META, LIBRARY] = await Promise.all([io.json("model/meta.json"), io.json("library.json")]);
    const full = await Model.load("model/full", META, io, true);
    fileEngine = new Engine({ model: full, foldModels, meta: META, mode: "file", onEvent: route });
    liveEngine = new Engine({ model: full, meta: META, mode: "live", onEvent: route,
      finalize: livePipeline({ model: full, meta: META, library: LIBRARY, save: saveBurstCsv }) });
    liveEngine.setConnection("idle", "not connected");
    player = new Player(fileEngine, META.play_speed);
    setInterval(() => liveEngine.tick(performance.now() / 1000), 500);
    connect(location.hash === "#live" ? "live" : "file");
  } catch (err) {
    pill.className = "pill err"; $("conn-text").textContent = "could not load the model";
    console.error(err);
  }
}

// ---------- controls ----------
document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => {
  if (t.dataset.mode === S.mode) return;
  history.replaceState(null, "", t.dataset.mode === "live" ? "#live" : location.pathname);
  connect(t.dataset.mode);
}));

function setNote(html, err = false) {
  const el = $("file-note");
  el.innerHTML = html; el.classList.toggle("err", err);
}
$("stop").onclick = () => {
  if (!player) return;
  player.stop();
  if (fileEngine.playback) fileEngine.setPlayback({ ...fileEngine.playback, playing: false, stopped: true });
};
$("file").onchange = async () => {
  const f = $("file").files[0];
  if (!f || !fileEngine) return;
  setNote(`Reading ${esc(f.name)}…`);
  try {
    const dev = $("dev").value;
    let parsed;
    try { parsed = parseUpload(await f.text(), LIBRARY, META, dev === "1" ? 1 : dev === "2" ? 2 : "auto"); }
    catch (err) { setNote(`<b>${esc(f.name)}</b>: ${esc(err.message)}.`, true); return; }
    const { bursts, report: r } = parsed;
    const parts = [`${r.rows ?? 0} rows`];
    if (r.skipped_lines) parts.push(`${r.skipped_lines} unreadable lines skipped`);
    if (r.outside_measurement) parts.push(`${r.outside_measurement} rows outside a measurement (e.g. purging) skipped`);
    if (r.short_bursts) parts.push(`${r.short_bursts} burst${r.short_bursts > 1 ? "s" : ""} under ${META.min_rows} rows skipped`);
    if (!bursts.length) { setNote(`<b>${esc(f.name)}</b>: no burst with at least ${META.min_rows} rows found (${parts.join(", ")}).`, true); return; }
    // collected bursts are predicted by the fold model that never saw them: fetch the ones needed
    const need = [...new Set(bursts.map((b) => b.info.fold).filter((k) => k && !foldModels[k]))];
    if (need.length) {
      setNote(`<b>${esc(f.name)}</b>: loading the held-out model${need.length > 1 ? "s" : ""}…`);
      await Promise.all(need.map(async (k) => (foldModels[k] = await Model.load(`model/fold${k}`, META, io))));
    }
    setNote(`<b>${esc(f.name)}</b>: ${r.bursts} burst${r.bursts > 1 ? "s" : ""}, ${parts.join(", ")}.`);
    player.play(bursts, { name: f.name, kind: "upload" });
  } catch (err) {
    setNote(`Upload failed: ${esc(err.message || err)}`, true);
    console.error(err);
  } finally {
    $("file").value = "";
  }
};

// ---------- live sensor over Web Serial (same reading as serial_plotter.py: 115200 baud, line by line) ----------
let port = null, reader = null, liveLog = [], saveDir = null, linesSinceConnect = 0, silenceTimer = null;
// common USB-to-serial chips on Arduino-type boards (for the connection label)
const USB_CHIPS = { 0x1a86: "CH340 USB serial", 0x0403: "FTDI USB serial", 0x10c4: "CP210x USB serial",
  0x2341: "Arduino", 0x2a03: "Arduino", 0x067b: "Prolific USB serial", 0x239a: "Adafruit", 0x303a: "ESP32" };

// live bursts are saved before they are processed: into a folder the user picks (Chrome / Edge),
// otherwise as a download
function saveBurstCsv(name, text) {
  if (saveDir) {
    (async () => {
      const fh = await saveDir.getFileHandle(name, { create: true });
      const w = await fh.createWritable(); await w.write(text); await w.close();
    })().catch((err) => console.error(err));
    return `${saveDir.name}/${name}`;
  }
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([text], { type: "text/csv" })), download: name });
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  return `Downloads/${name}`;
}
function renderLiveButtons() {
  const supported = "serial" in navigator;
  $("serial-unsupported").hidden = supported;
  $("connect").hidden = !supported || !!port;
  $("disconnect").hidden = !port;
  $("download-log").disabled = !liveLog.length;
  $("save-folder").hidden = !("showDirectoryPicker" in window);
  $("save-folder").textContent = saveDir ? `Saving to: ${saveDir.name}` : "Choose save folder";
}
$("save-folder").onclick = async () => {
  try { saveDir = await window.showDirectoryPicker({ mode: "readwrite" }); } catch {}
  renderLiveButtons();
};
async function readLines(p) {
  const decoder = new TextDecoderStream();
  const closed = p.readable.pipeTo(decoder.writable).catch(() => {});
  reader = decoder.readable.getReader();
  let buf = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += value;
      let i;
      while ((i = buf.search(/\r?\n/)) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(buf[i] === "\r" ? i + 2 : i + 1);
        if (!line) continue;
        liveLog.push(`${(Date.now() / 1000).toFixed(3)}\t${line}`);
        if (!linesSinceConnect++ && liveEngine.connection.state === "waiting")
          liveEngine.setConnection("connected", "e-nose connected · receiving data");
        liveEngine.onLine(line, performance.now() / 1000);
      }
    }
  } finally {
    reader.releaseLock();
    await closed;
  }
}
$("connect").onclick = async () => {
  try {
    port = await navigator.serial.requestPort();
    await port.open({ baudRate: Number($("baud").value) });
    const info = port.getInfo(), baud = $("baud").value;
    linesSinceConnect = 0;
    if (!info.usbVendorId) {
      // Bluetooth ports (earbuds, speakers) and built-in ports have no USB ids: they never send e-nose data
      liveEngine.setConnection("error", "this port is not a USB device (Bluetooth?) — Disconnect and pick the e-nose's USB port");
    } else {
      const chip = USB_CHIPS[info.usbVendorId] || `USB ${info.usbVendorId.toString(16)}:${(info.usbProductId || 0).toString(16)}`;
      liveEngine.setConnection("connected", `e-nose connected (${chip}) @ ${baud} baud`);
    }
    renderLiveButtons();
    clearTimeout(silenceTimer);
    silenceTimer = setTimeout(() => {
      if (port && !linesSinceConnect)
        liveEngine.setConnection("waiting", "connected, but nothing received yet: press the device's start button " +
          "(it only sends data while measuring), or check the port and baud rate");
    }, 20000);
    await readLines(port);
  } catch (err) {
    if (err.name !== "NotFoundError") liveEngine.setConnection("error", err.message);   // NotFoundError: picker closed
  } finally {
    if (port) { try { await port.close(); } catch {} }
    port = null; reader = null; clearTimeout(silenceTimer);
    if (liveEngine.connection.state === "connected") liveEngine.setConnection("idle", "disconnected");
    renderLiveButtons();
  }
};
$("disconnect").onclick = async () => { if (reader) await reader.cancel().catch(() => {}); };
$("download-log").onclick = () => {
  const blob = new Blob([liveLog.join("\n") + "\n"], { type: "text/plain" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob),
    download: `enose-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.log` });
  a.click(); URL.revokeObjectURL(a.href);
};
renderLiveButtons();
setInterval(() => { $("download-log").disabled = !liveLog.length; }, 2000);
boot();

// ---------- loop ----------
let last = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  for (const draw of [drawChamber, drawBrain, drawCharts]) {
    try { draw(now, dt); } catch (err) { console.error(err); }
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
