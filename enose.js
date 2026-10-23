// E-nose prediction in the browser: the same steps as app/sniffer.py and app/server.py.
//
//   parse     device lines "Temprature: 38.48 Humidity: 35.52 ; Sensor1 : 6464.00 ... Sensor4 : 1648.00"
//             and status lines (FILLING CHAMBER... / MEASUREMENT STARTED / ... / FLUSH FINISHED)
//   burst     rows between MEASUREMENT STARTED and MEASUREMENT FINISHED (or a gap > 2 s)
//   windows   50 rows every 25 rows; gas as % change from the first 1 s of the burst
//   model     gas readings corrected to 38 °C / 55 %RH; random forest on 39 features + 1D-CNN on 8 x 50
//             gas series (no temperature / humidity inputs), probabilities averaged
//   result    the most likely item + how often that confidence was right on unseen data;
//             "not sure" only for data unlike anything in the training data (novelty)
//
// Works in the browser and in Node (web/test/parity.mjs checks it against the Python app).

export const GAS = ["Sensor1", "Sensor2", "Sensor3", "Sensor4"];
export const ENV = ["Temperature", "Humidity"];
export const CHANNELS = [...GAS, ...ENV];

const LINE_RE = new RegExp(
  "Temp\\w*\\s*:\\s*([\\d.+-]+).+?Humidity\\s*:\\s*([\\d.+-]+).+?" +
  "Sensor1\\s*:\\s*([\\d.+-]+).+?Sensor2\\s*:\\s*([\\d.+-]+).+?" +
  "Sensor3\\s*:\\s*([\\d.+-]+).+?Sensor4\\s*:\\s*([\\d.+-]+)", "i");
const NO_GAS_RISE = 0.5;   // %: if no gas sensor (Sensor2-4) rises this much in a burst, no gas reached them
const NO_GAS_MIN_S = 10;   // seconds of a burst needed before that check is made (sniffer.NO_GAS_*)
const NOT_CLEAN_AIR_X = 1.5;    // chamber check: a new sample starting at >= 1.5x the clean-air S4/S3 ratio ...
const NOT_CLEAN_SAME = 0.25;    // ... and within 25 % of the previous sample's ratio (sniffer.NOT_CLEAN_*)
const RUN_GAP_S = 120;  // live: a fill this long after the last finished cycle is a new start (sniffer.RUN_GAP_S)
const STATUS = [["GAS CYCLE COMPLETE", "ready"], ["PURGING STOPPED", "ready"], ["PURGING", "flushing"],
  ["FLUSH FINISHED", "ready"], ["FLUSHING", "flushing"], ["MEASUREMENT FINISHED", "finished"],
  ["MEASUREMENT STARTED", "measuring"], ["CHAMBER FILLED", "filled"], ["FILLING", "filling"]];

export function parseRow(line) {
  const m = LINE_RE.exec(line);
  if (!m) return null;
  const v = m.slice(1).map(Number);
  if (v.some((x) => Number.isNaN(x))) return null;
  return { Temperature: v[0], Humidity: v[1], Sensor1: v[2], Sensor2: v[3], Sensor3: v[4], Sensor4: v[5] };
}

export function parseStatus(line) {
  const up = line.toUpperCase();
  for (const [key, phase] of STATUS) if (up.includes(key)) return phase;
  return null;
}

export function deviceLine(v) {
  return `Temprature: ${v.Temperature} Humidity: ${v.Humidity} ; Sensor1 : ${v.Sensor1} ` +
    `Sensor2 : ${v.Sensor2} Sensor3 : ${v.Sensor3} Sensor4 : ${v.Sensor4}`;
}

// ---------- numerics: numpy semantics, including numpy's pairwise summation order ----------
// Sums are added in numpy's exact order so results match bit for bit; this matters where a
// spread is exactly zero (device 1 moves in steps of 16, so a window can be flat).
function npSum(a, lo = 0, hi = a.length) {
  const n = hi - lo;
  if (n < 8) { let r = 0; for (let i = lo; i < hi; i++) r += a[i]; return r; }
  if (n <= 128) {
    const r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]];
    let i = lo + 8;
    for (; i < hi - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < hi; i++) res += a[i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return npSum(a, lo, lo + n2) + npSum(a, lo + n2, hi);
}
const mean = (a) => npSum(a) / a.length;
const allFinite = (a) => a.every((x) => Number.isFinite(x));
function std(a) { const m = mean(a); return Math.sqrt(npSum(a.map((x) => (x - m) * (x - m))) / a.length); }
const minOf = (a) => (a.some(Number.isNaN) ? NaN : Math.min(...a));
const maxOf = (a) => (a.some(Number.isNaN) ? NaN : Math.max(...a));
function nanmean(a) {           // np.nanmean: NaN -> 0, numpy sum, divided by the count of numbers
  const n = a.filter((x) => !Number.isNaN(x)).length;
  return n ? npSum(a.map((x) => (Number.isNaN(x) ? 0 : x))) / n : NaN;
}

function slope(t, y) {
  const tm = mean(t), ym = mean(y);
  const tc = t.map((v) => v - tm);
  const d = npSum(tc.map((v) => v * v));
  return d ? npSum(tc.map((v, i) => v * (y[i] - ym))) / d : 0;
}

function corr(a, b) {
  if (!allFinite(a) || !allFinite(b) || std(a) === 0 || std(b) === 0) return NaN;
  const am = mean(a), bm = mean(b), k = a.length - 1;
  let cab = 0, caa = 0, cbb = 0;
  for (let i = 0; i < a.length; i++) { const x = a[i] - am, y = b[i] - bm; cab += x * y; caa += x * x; cbb += y * y; }
  const r = (cab / k) / Math.sqrt(caa / k) / Math.sqrt(cbb / k);
  return Math.max(-1, Math.min(1, r));
}

// ---------- features (modeling/train_classifier.py window_features) ----------
export function windowFeatures(b, start, window, baselineS) {
  const end = start + window;
  const t = b.t.slice(start, end);
  const f = {}, pct = {};
  for (const ch of GAS) {
    const series = b[ch + "_cal"];
    const base = series.some(Number.isFinite) ? nanmean(series.filter((_, i) => b.t[i] < baselineS)) : NaN;
    const w = series.slice(start, end);
    const p = Number.isFinite(base) && base ? w.map((v) => (100 * (v - base)) / base) : w.map(() => NaN);
    pct[ch] = p;
    const s = ch.replace("Sensor", "S");
    f[`${s}_pct_mean`] = mean(p);
    f[`${s}_pct_std`] = std(p);
    f[`${s}_pct_min`] = minOf(p);
    f[`${s}_pct_max`] = maxOf(p);
    f[`${s}_pct_slope`] = allFinite(p) ? slope(t, p) : NaN;
    f[`${s}_pct_delta`] = p[p.length - 1] - p[0];
    f[`${s}_level`] = mean(w);
  }
  f.ratio_S2_S3 = f.S2_level / f.S3_level;
  f.ratio_S4_S3 = f.S4_level / f.S3_level;
  f.ratio_S1_S4 = f.S1_level / f.S4_level;
  for (let i = 0; i < GAS.length; i++)
    for (let j = i + 1; j < GAS.length; j++)
      f[`corr_${GAS[i].at(-1)}${GAS[j].at(-1)}`] = corr(pct[GAS[i]], pct[GAS[j]]);
  for (const [ch, s] of [["Temperature", "T"], ["Humidity", "RH"]]) {
    const v = b[ch + "_cal"].slice(start, end);
    f[`${s}_mean`] = mean(v);
    f[`${s}_slope`] = slope(t, v);
    f[`${s}_delta`] = v[v.length - 1] - v[0];
  }
  f.t_start_s = t[0];
  return f;
}

// ---------- random forest (median imputation + 500 trees, scikit-learn semantics) ----------
export class Forest {
  constructor(header, buffer) {
    const arr = (name, Type) => {
      const a = header.arrays[name];
      return new Type(buffer, a.offset, a.length);
    };
    this.h = header;
    this.threshold = arr("threshold", Float64Array);
    this.leafValue = arr("leaf_value", Float64Array);
    this.feature = arr("feature", Int32Array);
    this.left = arr("left", Int32Array);
    this.right = arr("right", Int32Array);
    this.leafClass = arr("leaf_class", Uint8Array);
  }

  predictProba(x) {
    const { medians, tree_start: starts, n_classes: nc } = this.h;
    const xf = x.map((v, i) => Math.fround(Number.isNaN(v) ? medians[i] : v));   // impute, then float32 like sklearn
    const acc = new Float64Array(nc);
    for (const s of starts) {
      let n = s;
      while (this.feature[n] !== -1) n = xf[this.feature[n]] <= this.threshold[n] ? this.left[n] : this.right[n];
      for (let k = this.left[n], e = k + this.right[n]; k < e; k++) acc[this.leafClass[k]] += this.leafValue[k];
    }
    return Array.from(acc, (v) => v / starts.length);
  }
}

// ---------- 1D-CNN (modeling/improve.py SniffCNN, eval mode) ----------
export class CNN {
  constructor(spec) {
    this.mean = spec.mean.map(Math.fround);
    this.std = spec.std.map(Math.fround);
    this.eps = spec.eps;
    this.L = spec.layers;
    this.compensation = spec.compensation || null;   // this model's temperature / humidity correction
  }

  static conv(x, w, b, pad) {
    const [co, ci, k] = w.shape, L = x[0].length, out = [];
    for (let o = 0; o < co; o++) {
      const row = new Float64Array(L).fill(b.data[o]);
      for (let c = 0; c < ci; c++) {
        const xc = x[c];
        for (let j = 0; j < k; j++) {
          const wv = w.data[(o * ci + c) * k + j];
          for (let l = 0; l < L; l++) {
            const p = l + j - pad;
            if (p >= 0 && p < L) row[l] += wv * xc[p];
          }
        }
      }
      out.push(row);
    }
    return out;
  }

  bnRelu(x, name) {
    const g = this.L[`${name}.weight`].data, bt = this.L[`${name}.bias`].data;
    const rm = this.L[`${name}.running_mean`].data, rv = this.L[`${name}.running_var`].data;
    return x.map((row, c) => row.map((v) => Math.max(0, ((v - rm[c]) / Math.sqrt(rv[c] + this.eps)) * g[c] + bt[c])));
  }

  forward(z) {
    const L = this.L;
    let x = this.bnRelu(CNN.conv(z, L["net.0.weight"], L["net.0.bias"], 2), "net.1");
    x = this.bnRelu(CNN.conv(x, L["net.3.weight"], L["net.3.bias"], 2), "net.4");
    x = x.map((row) => Array.from({ length: Math.floor(row.length / 2) }, (_, i) => Math.max(row[2 * i], row[2 * i + 1])));
    x = this.bnRelu(CNN.conv(x, L["net.7.weight"], L["net.7.bias"], 1), "net.8");
    const pooled = x.map((row) => mean(row));
    const W = L["net.13.weight"], bias = L["net.13.bias"].data, [nc, nf] = W.shape;
    const logits = Array.from({ length: nc }, (_, o) => {
      let s = bias[o];
      for (let i = 0; i < nf; i++) s += W.data[o * nf + i] * pooled[i];
      return s;
    });
    const m = Math.max(...logits), e = logits.map((v) => Math.exp(v - m)), sum = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / sum);
  }

  predict(b, start, window, baselineS) {
    const end = start + window, f32 = Math.fround, s = [];
    for (const ch of GAS) {
      const v = b[ch + "_cal"];
      const base = mean(v.filter((_, i) => b.t[i] < baselineS));
      s.push(v.slice(start, end).map((x) => f32((100 * (x - base)) / base)));
    }
    for (const ch of GAS) s.push(b[ch + "_cal"].slice(start, end).map((x) => f32(Math.log(Math.max(x, 1)))));
    for (const ch of ENV) s.push(b[ch + "_cal"].slice(start, end).map(f32));
    s.length = this.mean.length;                         // 8 gas channels, or 10 for a model that uses T / RH
    const MAXF = 3.4028234663852886e38;
    const z = s.map((row, c) => row.map((v) => {
      const r = f32(f32(v - this.mean[c]) / this.std[c]);
      return Number.isNaN(r) ? 0 : r === Infinity ? MAXF : r === -Infinity ? -MAXF : r;
    }));
    return this.forward(z);
  }
}

// ---------- the model: forest + CNN, calibration, result summary ----------
export class Model {
  constructor(meta, forest, cnn, novelty = null) {
    Object.assign(this, { meta, forest, cnn, nov: novelty });
    this.items = meta.items;
    this.categories = meta.categories;
    this.catOf = this.items.map((it) => this.categories.indexOf(meta.item_category[it]));
    this.window = meta.window;
    this.stride = meta.stride;
  }

  static async load(dir, meta, io, withNovelty = false) {
    const [h, buf, spec] = await Promise.all([io.json(`${dir}/rf.json`), io.buffer(`${dir}/rf.bin`), io.json(`${dir}/cnn.json`)]);
    let nov = null;
    if (withNovelty) {
      const [nh, nb] = await Promise.all([io.json(`${dir}/novelty.json`), io.buffer(`${dir}/novelty.bin`)]);
      nov = { ...nh, Z: new Float32Array(nb) };
    }
    return new Model(meta, new Forest(h, buf), new CNN(spec), nov);
  }

  // distance to the k-th nearest training window (39 features, imputed, standardised)
  novelty(x) {
    const { Z, mu, sd, k, n_features: nf } = this.nov, med = this.forest.h.medians;
    const z = x.map((v, i) => ((Number.isNaN(v) ? med[i] : v) - mu[i]) / sd[i]);
    const n = Z.length / nf, d = new Float64Array(n), sq = new Array(nf);
    for (let r = 0; r < n; r++) {
      for (let j = 0; j < nf; j++) { const e = Z[r * nf + j] - z[j]; sq[j] = e * e; }
      d[r] = Math.sqrt(npSum(sq));
    }
    return d.sort()[k - 1];
  }

  reliability(conf, level) {
    for (const b of this.meta.reliability[level])
      if ((b.lo <= conf && conf < b.hi) || (conf >= 1 && b.hi >= 1))
        return { right: b.right, top3: b.top3 ?? null, n: b.n, lo: b.lo, hi: b.hi, level };
    return null;
  }

  calibrate(raw, device) {
    const out = {};
    for (const ch of CHANNELS) {
      let v = raw[ch].slice();
      if (device === 1) {
        if (ch === "Sensor1") v = v.map(() => NaN);
        else if (this.meta.calibration[ch]) {
          const { slope: a, intercept: c } = this.meta.calibration[ch];
          v = v.map((x) => Number((a * x + c).toFixed(2)));   // 2 decimals, as calibrate_devices.py
        }
      }
      out[ch + "_cal"] = v;
    }
    return out;
  }

  // Gas readings corrected to the reference temperature / humidity (sniffer.py Model.compensate), so a
  // warming device does not look like another smell.
  compensate(b) {
    const comp = this.cnn.compensation;
    if (!comp) return b;
    const t0 = comp.reference.Temperature, rh0 = comp.reference.Humidity, T = b.Temperature_cal, H = b.Humidity_cal;
    const out = { ...b };
    for (const [ch, c] of Object.entries(comp.sensors))
      out[ch + "_cal"] = b[ch + "_cal"].map((v, i) => v * Math.exp(-(c.per_degC * (T[i] - t0) + c.per_pctRH * (H[i] - rh0))));
    return out;
  }

  // [probabilities, novelty score]; the score always uses the final model's reference
  predictWindow(b, start, device, noveltyModel = null) {
    b = this.compensate(b);
    const f = windowFeatures(b, start, this.window, this.meta.baseline_s);
    f.device = device;
    const x = this.meta.rf_features.map((n) => f[n]);
    const pRf = this.forest.predictProba(x);
    const pCnn = this.cnn.predict(b, start, this.window, this.meta.baseline_s);
    return [pRf.map((v, i) => (v + pCnn[i]) / 2), (noveltyModel || this).novelty(x)];
  }

  summary(p, novel = false, top = 5, level = "burst", noGas = false) {
    const cat = new Array(this.categories.length).fill(0);
    p.forEach((v, i) => (cat[this.catOf[i]] += v));
    const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]).slice(0, top);
    const best = order[0], ic = this.meta.item_category;
    const r4 = (v) => Math.round(v * 1e4) / 1e4;
    const undecided = novel || noGas;
    const res = {
      decided: !undecided,
      item: undecided ? null : this.items[best],
      category: undecided ? null : ic[this.items[best]],
      confidence: r4(p[best]),
      top: order.map((i) => ({ item: this.items[i], category: ic[this.items[i]], p: r4(p[i]) })),
      categories: this.categories.map((c, i) => ({ category: c, p: r4(cat[i]) })).sort((a, b) => b.p - a.p),
    };
    if (noGas) Object.assign(res, { no_gas: true, reason: "no gas detected: the sensors did not respond" });
    else if (novel) res.reason = "unlike anything in the training data";
    res.reliability = this.reliability(res.confidence, level);
    return res;
  }
}

const meanRows = (rows) => rows[0].map((_, i) => mean(rows.map((r) => r[i])));
function median(a) { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
const clock = () => new Date().toTimeString().slice(0, 8);

// ---------- one measurement ----------
class Burst {
  constructor(number, deviceSetting, info, deadMax) {
    Object.assign(this, { number, deviceSetting, info: info || {}, deadMax });
    this.t0 = null; this.t = []; this.raw = Object.fromEntries(CHANNELS.map((c) => [c, []]));
    this.windowProbs = []; this.windowNovelty = []; this.nextStart = 0; this.lastRow = null; this.wall = [];
  }

  get device() {
    if (this.info.device === 1 || this.info.device === 2) return this.info.device;
    if (this.deviceSetting === 1 || this.deviceSetting === 2) return this.deviceSetting;
    const s1 = this.raw.Sensor1.slice(0, 20);
    return s1.length && Math.max(...s1) < this.deadMax ? 1 : 2;
  }

  add(row, now) {
    this.wall.push(Date.now());
    if (this.t0 === null) this.t0 = now;
    this.t.push(now - this.t0);
    for (const c of CHANNELS) this.raw[c].push(row[c]);
    this.lastRow = now;
  }

  proba() { return this.windowProbs.length ? meanRows(this.windowProbs) : null; }
  novelty() { return this.windowNovelty.length ? median(this.windowNovelty) : null; }

  // Largest % rise of Sensor2-4 above their level in the first second; null before NO_GAS_MIN_S.
  gasRise(baselineS) {
    if (!this.t.length || this.t[this.t.length - 1] < NO_GAS_MIN_S) return null;
    const rise = [];
    for (const c of ["Sensor2", "Sensor3", "Sensor4"]) {
      const v = this.raw[c], base = mean(v.filter((_, i) => this.t[i] < baselineS));
      if (Number.isFinite(base) && base) rise.push(100 * (Math.max(...v) - base) / base);
    }
    return rise.length ? Math.max(...rise) : null;
  }

  // No gas sensor responded (e.g. the e-nose is powered over USB but not started: no gas is pumped in).
  noGas(baselineS) { const r = this.gasRise(baselineS); return r !== null && r < NO_GAS_RISE; }
}

// ---------- engine: lines in, phases / samples / predictions out (app/sniffer.py Engine) ----------
export class Engine {
  constructor({ model, foldModels = {}, meta, mode = "file", deviceSetting = "auto", onEvent = () => {}, finalize = null }) {
    Object.assign(this, { model, foldModels, meta, mode, deviceSetting, onEvent, finalize });
    this.playback = null; this.connection = { state: "idle", detail: "" };
    this.prevRunRatio = null;                         // live: S4/S3 start ratio of the last burst of the previous run
    this.reset(false);
  }

  emit(kind, data) { this.onEvent(kind, data, this); }

  reset(broadcast = true) {
    this.phase = "idle"; this.burst = null; this.bursts = []; this.recent = [];
    this.raw = []; this.nLines = 0; this.burstCount = 0; this.nextInfo = null; this.markersSeen = false;
    this.lastReady = null;
    this.runLabel = null;                             // live: what the user said was in the chamber for this run                            // when the device last reported a finished cycle
    if (broadcast) this.emit("state", this.snapshot());
  }

  expect(info) { this.nextInfo = info; }
  setPlayback(p) { this.playback = p; this.emit("playback", p); }
  setConnection(state, detail = "") { this.connection = { state, detail }; this.emit("connection", this.connection); }

  burstResult(b) {
    const score = b.novelty();
    const rise = b.gasRise(this.meta.baseline_s);
    return { ...this.model.summary(b.proba(), score > this.model.nov.threshold, 5, "burst", b.noGas(this.meta.baseline_s)),
      burst: b.number, n_windows: b.windowProbs.length, novelty: Math.round(score * 100) / 100,
      gas_rise: rise === null ? null : Math.round(rise * 100) / 100 };
  }

  // the bursts that resemble the training data, averaged: the file (uploads) or the run (live)
  sampleResult() {
    if (!this.bursts.length) return null;
    const known = this.bursts.filter((b) => b.result.decided), use = known.length ? known : this.bursts;
    const noGas = !known.length && this.bursts.every((b) => b.result.no_gas);
    const res = this.model.summary(meanRows(use.map((b) => b.proba)), !known.length && !noGas, 5, use.length > 1 ? "sample" : "burst", noGas);
    Object.assign(res, { n_bursts: this.bursts.length, bursts_used: known.length });
    const truth = this.bursts[0].truth;
    if (truth && this.bursts.every((b) => b.truth && b.truth.item === truth.item))
      Object.assign(res, { truth, correct: res.decided && res.item === truth.item });
    return res;
  }

  snapshot() {
    const b = this.burst;
    const cur = b && { number: b.number, device: b.t.length ? b.device : null, n_rows: b.t.length,
      n_windows: b.windowProbs.length, result: b.windowProbs.length ? this.burstResult(b) : null };
    return {
      mode: this.mode, connection: this.connection, phase: this.phase,
      novelty_threshold: this.model.nov.threshold, window: this.model.window,
      items: this.model.items, item_category: this.meta.item_category, categories: this.model.categories,
      current: cur || null, recent: this.recent.slice(-700),
      bursts: this.bursts.map(({ proba, ...rest }) => rest), sample: this.sampleResult(),
      playback: this.playback, raw: this.raw.slice(-200), n_lines: this.nLines, run_label: this.runLabel,
    };
  }

  setPhase(phase) {
    this.phase = phase;
    this.emit("phase", { phase, burst: this.burst ? this.burst.number : this.burstCount + 1 });
  }

  onLine(line, now) {
    line = line.trim();
    if (!line) return;
    this.nLines += 1;
    const raw = { time: clock(), text: line };
    this.raw.push(raw); if (this.raw.length > 200) this.raw.shift();
    this.emit("raw", { ...raw, n: this.nLines });
    const row = parseRow(line);
    if (row) return this.onRow(row, now);
    const phase = parseStatus(line);
    if (phase === "measuring") { this.markersSeen = true; this.startBurst(); }
    else if (phase === "finished") this.finishBurst();
    else if (phase) {
      if (this.burst) this.finishBurst();
      if (phase === "filling" && this.mode === "live" && !(this.lastReady !== null && now - this.lastReady < RUN_GAP_S)) this.newRun();
      if (phase === "ready") this.lastReady = now;
      this.setPhase(phase);
    }
  }

  // Live: the device's start button was pressed (a fill that does not follow a finished cycle, e.g. after
  // start-up or a pause). The page starts over; the device lines stay.
  // The page's Clear button: forget the results and device lines shown so far. A burst being measured is
  // dropped too; the next one starts at the device's next MEASUREMENT STARTED.
  clear() {
    if (this.bursts.length && this.bursts.at(-1).start_ratio != null) this.prevRunRatio = this.bursts.at(-1).start_ratio;
    this.burst = null; this.bursts = []; this.recent = []; this.burstCount = 0; this.raw = []; this.nLines = 0;
    this.runLabel = null;
    this.emit("state", this.snapshot());
  }

  newRun() {
    if (this.bursts.length && this.bursts.at(-1).start_ratio != null) this.prevRunRatio = this.bursts.at(-1).start_ratio;
    this.burst = null; this.bursts = []; this.recent = []; this.burstCount = 0; this.runLabel = null;
    this.emit("state", this.snapshot());
  }

  // live: the user's answer to "What was in the chamber?" and where the labelled run was saved
  setRunLabel(label) { this.runLabel = label; this.emit("label", label); }

  tick(now) {
    const b = this.burst;
    if (b && b.lastRow !== null && now - b.lastRow > this.meta.burst_gap_s) this.finishBurst();
  }

  startBurst() {
    if (this.burst) this.finishBurst();
    this.burstCount += 1;
    this.burst = new Burst(this.burstCount, this.deviceSetting, this.nextInfo, this.meta.dead_sensor1_max);
    this.nextInfo = null; this.recent = [];
    this.setPhase("measuring");
  }

  onRow(row, now) {
    let b = this.burst;
    if (!b && this.markersSeen) return;              // outside a measurement (e.g. purging): shown, not predicted
    if (b && b.lastRow !== null && now - b.lastRow > this.meta.burst_gap_s) { this.finishBurst(); b = null; }
    if (!b) { this.startBurst(); b = this.burst; }
    b.add(row, now);
    const r2 = (v) => Math.round(v * 100) / 100;
    const sample = { t: r2(b.t[b.t.length - 1]), ...Object.fromEntries(CHANNELS.map((c) => [c, r2(row[c])])) };
    this.recent.push(sample); if (this.recent.length > 700) this.recent.shift();
    this.emit("sample", { ...sample, n: b.t.length, burst: b.number, device: b.device });
    const W = this.model.window;
    while (b.nextStart + W <= b.t.length) { this.predict(b, b.nextStart); b.nextStart += this.model.stride; }
  }

  predict(b, start) {
    const model = this.foldModels[b.info.fold] || this.model;
    const n = start + model.window;
    const raw = Object.fromEntries(CHANNELS.map((c) => [c, b.raw[c].slice(0, n)]));
    let cal;
    if (b.info.calibrated) {
      cal = Object.fromEntries(CHANNELS.map((c) => [c + "_cal", raw[c]]));
      if (b.info.no_sensor1) cal.Sensor1_cal = cal.Sensor1_cal.map(() => NaN);
    } else cal = model.calibrate(raw, b.device);
    cal.t = b.t.slice(0, n);
    const [p, score] = model.predictWindow(cal, start, b.device, this.model);
    b.windowProbs.push(p); b.windowNovelty.push(score);
    this.emit("window", { burst: b.number, window: b.windowProbs.length, t_start: Math.round(b.t[start] * 100) / 100,
      burst_result: this.burstResult(b) });
  }

  // Sensor4 / Sensor3 in the burst's first second: what the chamber held when the measurement started. The
  // sensors drop together as the device warms, so the ratio needs no temperature / humidity (sniffer.py).
  startRatio(b) {
    const cal = this.model.calibrate(b.raw, b.device), first = (_, i) => b.t[i] < this.meta.baseline_s;
    const s4 = cal.Sensor4_cal.filter(first), s3 = cal.Sensor3_cal.filter(first);
    return s4.length ? Math.round((mean(s4) / mean(s3)) * 1e4) / 1e4 : null;
  }

  // First burst of a new sample: is the chamber still at the previous sample's level, well above clean air?
  chamberCheck(level) {
    const air = (this.meta.clean_air || {}).S4_S3, prev = this.prevRunRatio;
    if (level === null || !air || prev === null) return null;
    const notClean = level >= NOT_CLEAN_AIR_X * air && Math.abs(level - prev) <= NOT_CLEAN_SAME * prev;
    return { not_clean: notClean, start_ratio: level, previous_ratio: prev, clean_air_ratio: air,
      times_clean_air: Math.round((level / air) * 10) / 10 };
  }

  finishBurst() {
    const b = this.burst;
    this.burst = null;
    if (!b) return;
    if (!b.windowProbs.length) {
      this.emit("short", { burst: b.number, rows: b.t.length, need: this.model.window });
      return this.setPhase("finished");
    }
    const rec = { number: b.number, device: b.device, n_rows: b.t.length,
      duration_s: Math.round(b.t[b.t.length - 1] * 10) / 10, n_windows: b.windowProbs.length,
      result: this.burstResult(b), proba: b.proba(), fold: b.info.fold ?? null };
    if (this.finalize) {
      let final;
      try { final = this.finalize(b); } catch (err) { final = { error: String(err.message || err) }; }
      if (final.proba) {
        const res = this.model.summary(final.proba, final.novelty > this.model.nov.threshold, 5, "burst", b.noGas(this.meta.baseline_s));
        Object.assign(res, { burst: b.number, n_windows: final.n_windows, novelty: Math.round(final.novelty * 100) / 100,
          gas_rise: rec.result.gas_rise });
        Object.assign(rec, { result: res, proba: final.proba, n_windows: final.n_windows });
      }
      const { proba, novelty, ...pipe } = final;
      rec.pipeline = pipe;
    }
    if (this.mode === "live") {
      rec.start_ratio = this.startRatio(b);
      if (b.number === 1) rec.chamber = this.chamberCheck(rec.start_ratio);
    }
    if (b.info.truth) Object.assign(rec, { truth: b.info.truth, correct: rec.result.decided && rec.result.item === b.info.truth.item });
    this.bursts.push(rec);
    if (this.bursts.length > 20) this.bursts.shift();
    const { proba, ...pub } = rec;
    this.emit("burst", { ...pub, sample: this.sampleResult() });
    this.setPhase("finished");
  }
}

// ---------- uploads (app/server.py parse_upload) ----------
const FP_COLS = ["Sensor2", "Sensor3", "Sensor4", "Temperature", "Humidity"];
const fixed = (v, d) => Number(v.toFixed(d));
const fingerprint = (rows) => rows.slice(0, 5).map((r) => FP_COLS.map((c) => r[c].toFixed(2)).join(",")).join(",");

// labeled-data/label_data.py: bursts split at gaps > 2 s; rows sharing a second spread evenly across it
function splitBursts(secs) {
  const out = [];
  let s = 0;
  for (let i = 1; i < secs.length; i++) if (secs[i] - secs[i - 1] > 2) { out.push([s, i]); s = i; }
  out.push([s, secs.length]);
  return out;
}
function burstOffsets(secs) {
  const count = new Map(), seen = new Map();
  for (const t of secs) count.set(t, (count.get(t) || 0) + 1);
  return secs.map((t) => {
    const k = seen.get(t) || 0;
    seen.set(t, k + 1);
    return t - secs[0] + k / count.get(t);
  });
}

function recognisedBurst(entry, rows, offs, meta) {
  const cal = meta.calibration;
  const values = rows.map((r, i) => {
    if (entry.device === 2) return { ...r };
    const v = { Sensor1: entry.sensor1 ? entry.sensor1[i] : 0 };
    for (const c of ["Sensor2", "Sensor3", "Sensor4", "Temperature", "Humidity"])
      v[c] = fixed(cal[c].slope * r[c] + cal[c].intercept, 2);
    return v;
  });
  return {
    rows: values.map((v, i) => [fixed(offs[i], 3), v]),
    info: { device: entry.device, fold: Math.min(entry.burst, 4), calibrated: true, no_sensor1: entry.device === 1 && !entry.sensor1,
      truth: { item: entry.item, category: entry.category, recording: entry.recording, burst: entry.burst } },
  };
}

export function parseUpload(text, library, meta, device) {
  const lines = text.split(/\r\n|\r|\n/);
  const report = { format: null, rows: 0, skipped_lines: 0, outside_measurement: 0, bursts: 0, short_bursts: 0, recognised: 0 };
  const header = lines.find((l) => l.trim()) || "";
  const rawBursts = [];                                  // [secs (CSV) | offsets | null, rows, kind]
  if (header.includes(",") && /sensor1/i.test(header) && !parseRow(header)) {
    report.format = "csv";
    const cols = header.split(",").map((c) => c.trim());
    const canon = cols.map((c) => (c.toLowerCase().startsWith("temp") ? "Temperature" : CHANNELS.find((k) => k.toLowerCase() === c.toLowerCase()) || null));
    const missing = CHANNELS.filter((k) => !canon.includes(k));
    if (missing.length) throw new Error(`CSV is missing columns: ${missing.sort().join(", ")}`);
    const TIME = ["day", "month", "year", "hour", "minute", "second"];
    const tIdx = TIME.map((t) => cols.findIndex((c) => c.toLowerCase() === t));
    const hasTime = tIdx.every((i) => i >= 0);
    const rows = [], secs = [];
    for (const line of lines.slice(lines.indexOf(header) + 1)) {
      const parts = line.split(",").map((p) => p.trim().replace(/^"|"$/g, ""));
      if (!parts.some((p) => p)) continue;
      if (parts.length !== cols.length) { report.skipped_lines++; continue; }
      const num = (p) => (p === "" || !/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(p) ? NaN : Number(p));
      const vals = {};
      let bad = false;
      canon.forEach((k, i) => { if (k) { vals[k] = num(parts[i]); if (Number.isNaN(vals[k])) bad = true; } });
      let sec = null;
      if (hasTime && !bad) {
        const [d, mo, y, h, mi, s] = tIdx.map((i) => Math.trunc(num(parts[i])));
        const ok = [d, mo, y, h, mi, s].every(Number.isFinite) && mo >= 1 && mo <= 12 && d >= 1 &&
          d <= new Date(Date.UTC(y, mo, 0)).getUTCDate() && h >= 0 && h < 24 && mi >= 0 && mi < 60 && s >= 0 && s < 60 && y >= 1;
        if (!ok) bad = true; else sec = Date.UTC(y, mo - 1, d, h, mi, s) / 1000;
      }
      if (bad) { report.skipped_lines++; continue; }
      rows.push(Object.fromEntries(CHANNELS.map((k) => [k, fixed(vals[k], 2)])));
      if (hasTime) secs.push(sec);
    }
    report.rows = rows.length;
    if (hasTime && rows.length) for (const [s, e] of splitBursts(secs)) rawBursts.push([secs.slice(s, e), rows.slice(s, e), "csv"]);
    else if (rows.length) rawBursts.push([null, rows, "none"]);
  } else {
    report.format = "device lines";
    const markers = lines.some((l) => l.toUpperCase().includes("MEASUREMENT STARTED"));
    let inside = !markers, cur = [], curT = [], lastT = null;
    const close = () => {
      if (cur.length) rawBursts.push([curT.every((x) => x !== null) ? curT.map((x) => x - curT[0]) : null, cur, curT.every((x) => x !== null) ? "offsets" : "none"]);
      cur = []; curT = [];
    };
    for (const l of lines) {
      const tab = l.indexOf("\t");
      const ts = tab >= 0 ? l.slice(0, tab) : "", body = (tab >= 0 ? l.slice(tab + 1) : l).trim();
      const t = ts && /^[+-]?(\d+\.?\d*|\.\d+)$/.test(ts.trim()) ? Number(ts) : null;
      const row = parseRow(body), status = row ? null : parseStatus(body);
      if (status === "measuring" || (row && cur.length && t !== null && lastT !== null && t - lastT > 2)) {
        close();
        inside = inside || status === "measuring";
      } else if (["finished", "flushing", "ready"].includes(status)) { close(); inside = !markers; }
      if (row && inside) {
        cur.push(Object.fromEntries(CHANNELS.map((k) => [k, fixed(row[k], 2)])));
        curT.push(t); lastT = t; report.rows++;
      } else if (row) report.outside_measurement++;
      else if (body && !status) report.skipped_lines++;
    }
    close();
  }

  const bursts = [];
  for (const [times, rows, kind] of rawBursts) {
    if (rows.length < meta.min_rows) { report.short_bursts++; continue; }
    const offs = kind === "csv" ? burstOffsets(times).map((o) => fixed(o, 3)) : kind === "offsets" ? times : rows.map((_, i) => i * 0.1);
    const entry = rows.length >= 5 ? library[fingerprint(rows)] : null;
    if (entry) { report.recognised++; bursts.push(recognisedBurst(entry, rows, offs, meta)); continue; }
    bursts.push({ rows: rows.map((r, i) => [offs[i], r]), info: device === 1 || device === 2 ? { device } : {} });
  }
  report.bursts = bursts.length;
  return { bursts, report };
}

// ---------- player: plays bursts as the device would (fill, measure, flush) ----------
export class Player {
  constructor(engine, speed) { this.engine = engine; this.speed = speed; this.token = 0; }

  stop() { this.token++; }

  async play(bursts, label) {
    const token = ++this.token, eng = this.engine, speed = this.speed;
    const alive = () => token === this.token;
    const sleep = (s) => new Promise((r) => setTimeout(r, (s / speed) * 1000));
    eng.reset();
    const state = { playing: true, label, n_bursts: bursts.length };
    for (let k = 0; k < bursts.length; k++) {
      const b = bursts[k];
      eng.setPlayback({ ...state, burst: k + 1 });
      eng.expect(b.info);
      for (const [msg, s] of [["FILLING CHAMBER...", 6], ["CHAMBER FILLED", 0.8], ["MEASUREMENT STARTED", 0]]) {
        eng.onLine(msg, 0); await sleep(s); if (!alive()) return;
      }
      const t0 = performance.now();
      for (const [off, values] of b.rows) {
        const wait = t0 + (off / speed) * 1000 - performance.now();
        if (wait > 4) { await new Promise((r) => setTimeout(r, wait)); if (!alive()) return; }
        eng.onLine(deviceLine(values), off);
      }
      for (const [msg, s] of [["MEASUREMENT FINISHED", 0.6], ["FLUSHING GAS...", 5], ["FLUSH FINISHED", 1]]) {
        eng.onLine(msg, 0); await sleep(s); if (!alive()) return;
      }
    }
    eng.setPlayback({ ...state, playing: false, done: true, burst: bursts.length });
  }
}

// ---------- live: save the burst, then run the saved file through the training preprocessing ----------
const CSV_HEADER = "Sensor1,Sensor2,Sensor3,Sensor4,Temperature,Humidity,day,month,year,Hour,Minute,second";
export function livePipeline({ model, meta, library, save }) {
  return (b) => {
    const t0 = new Date(b.wall[0]), p2 = (n) => String(n).padStart(2, "0");
    const name = `${t0.getFullYear()}-${p2(t0.getMonth() + 1)}-${p2(t0.getDate())}_${p2(t0.getHours())}${p2(t0.getMinutes())}${p2(t0.getSeconds())}-burst${b.number}.csv`;
    const lines = [CSV_HEADER];
    b.wall.forEach((w, i) => {
      const t = new Date(w);
      lines.push(CHANNELS.map((c) => b.raw[c][i].toFixed(2)).join(",") +
        `,${t.getDate()},${t.getMonth() + 1},${t.getFullYear()},${t.getHours()},${t.getMinutes()},${t.getSeconds()}`);
    });
    const text = lines.join("\n") + "\n";
    const savedTo = save(name, text);

    const { bursts, report } = parseUpload(text, library, meta, b.deviceSetting);
    if (!bursts.length) throw new Error(`saved burst has no window of ${meta.min_rows} rows`);
    const burst = bursts.reduce((x, y) => (y.rows.length > x.rows.length ? y : x));
    const eng = new Engine({ model, meta, mode: "file", deviceSetting: b.deviceSetting });
    eng.expect(burst.info);
    eng.onLine("MEASUREMENT STARTED", 0);
    for (const [off, v] of burst.rows) eng.onLine(deviceLine(v), off);
    eng.onLine("MEASUREMENT FINISHED", 0);
    const r = eng.bursts[eng.bursts.length - 1];
    return {
      proba: r.proba, novelty: r.result.novelty, n_windows: r.n_windows, saved_to: savedTo, bursts_in_file: report.bursts,
      csv_text: text,                                   // kept for the run's "What was in the chamber?" label
      steps: [`saved ${b.wall.length} rows to ${savedTo}`,
        "split into bursts at time gaps > 2 s; rows spread across each second (label_data.py)",
        r.device === 1 ? "device 1 mapped to device 2's scale (device_calibration.json); Sensor1 dead, left empty"
          : "device 2: already on the reference scale (calibration not needed)",
        model.cnn.compensation
          ? "gas readings corrected to 38 °C / 55 %RH (per-sensor effect measured within the training recordings), so the device's temperature and humidity are not read as smell"
          : "temperature and humidity not used: the model works on the gas sensors only",
        "gas as % change from the burst's first 1 s; 5 s windows every 2.5 s (prepare_windows.py)",
        "39 features per window: response, levels, ratios, sensor correlations (no temperature / humidity); missing values filled with the training medians",
        "CNN input (8 gas channels) standardised with the training mean / std",
        `random forest + CNN, ${r.n_windows} windows averaged`],
    };
  };
}
