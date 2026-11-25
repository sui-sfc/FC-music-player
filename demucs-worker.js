// 音源分離 + AI 解析を動かす Web Worker (「音源分離」モード)
//   AudioWorklet から入力音声を受け取り、一定間隔で Demucs をかけて 歌・ベース・ドラム・その他 に分け、
//   歌 → RMVPE, その他の楽器・ベース → Basic Pitch, ベース → RMVPE (単音) で解析して、
//   分離した音と解析結果を AudioWorklet に返す。区切り方は worklet/stemstream.js (Node のシミュレーターと共通)
//
// モデル: 'htdemucs' (1 つで 4 音) / 'ft' (htdemucs_ft: 音ごとに専用のモデル 4 つ。分離の精度が高いが 4 倍重い)
import * as ort from './node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs';
import { demucsSpec, demucsISpec, DEMUCS_SR, DEMUCS_L, DEMUCS_T, resample } from './worklet/demucs.js';
import { melSpectrogram, decodePitch } from './worklet/rmvpe.js';
import { BP_SR, BP_HOP, BP_OUTPUTS, basicPitchFull } from './worklet/basicpitch.js';
import { StemStreamer } from './worklet/stemstream.js';

ort.env.wasm.wasmPaths = new URL('./node_modules/onnxruntime-web/dist/', import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 2) : 1;

let demucs = [], rmvpe = null, bp = null, ep = '', model = 'htdemucs';
let streamer = null, busy = false, port = null, cfg = null;

async function fetchModel(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${label} のモデルが見つかりません (${res.status})。README の手順で models/ に置いてください`);
  const total = +res.headers.get('Content-Length') || 0;
  const reader = res.body.getReader();
  const buf = new Uint8Array(total || 400e6);
  let loaded = 0, lastPost = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, loaded);
    loaded += value.length;
    if (loaded - lastPost > 8e6) { lastPost = loaded; postMessage({ type: 'progress', label, loaded, total }); }
  }
  return buf.subarray(0, loaded);
}

async function load(want) {
  const wasm = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  bp = await ort.InferenceSession.create(await fetchModel('models/basic-pitch.onnx', 'Basic Pitch'), wasm);
  const rmBuf = await fetchModel('models/rmvpe.onnx', 'RMVPE');
  const gpu = !!navigator.gpu;
  // RMVPE も可能なら WebGPU を試し、ダメなら WASM にフォールバック
  rmvpe = null;
  if (gpu) { try { rmvpe = await ort.InferenceSession.create(rmBuf, { executionProviders: ['webgpu'] }); } catch { rmvpe = null; } }
  if (!rmvpe) rmvpe = await ort.InferenceSession.create(rmBuf, wasm);

  // GPU (WebGPU) が使えればそちらで (約 10 倍速い)。ft は GPU のときだけ (CPU では間に合わない)
  model = want === 'ft' && gpu ? 'ft' : 'htdemucs';
  const files = model === 'ft' ? [0, 1, 2, 3].map((i) => [`models/htdemucs_ft_${i}.onnx`, `Demucs ft ${i + 1}/4`]) : [['models/htdemucs.onnx', 'Demucs']];
  demucs = [];
  ep = gpu ? 'webgpu' : 'wasm';
  for (const [f, label] of files) {
    const buf = await fetchModel(f, label);
    postMessage({ type: 'progress', compiling: true, label });
    let s = null;
    if (gpu) { try { s = await ort.InferenceSession.create(buf, { executionProviders: ['webgpu'] }); } catch { s = null; } }
    if (!s) { s = await ort.InferenceSession.create(buf, wasm); ep = 'wasm'; }
    demucs.push(s);
  }
  if (ep === 'wasm' && model === 'ft') { model = 'htdemucs'; demucs = [await ort.InferenceSession.create(await fetchModel('models/htdemucs.onnx', 'Demucs'), wasm)]; }
  // 間隔: GPU で htdemucs なら 1 秒ごと、ft なら 2 秒ごと、CPU なら 3 秒ごと。右の余白は 1 秒
  cfg = { hop: ep === 'wasm' ? 3 : model === 'ft' ? 2 : 1, margin: 1, ctx: 1.5 };
  // 試しに 1 回動かしておく (GPU は初回だけシェーダーの準備で遅く、そのせいで出力の遅れが延びてしまうため)
  postMessage({ type: 'progress', compiling: true, label: '準備運転' });
  const z = new Float32Array(DEMUCS_L);
  await sep(z, z);
  await runRmvpe(new Float32Array(16000 * 4));
  await runBp(new Float32Array(22050 * 4));
  postMessage({ type: 'ready', ep, model, threads: ort.env.wasm.numThreads, delay: cfg.hop + cfg.margin });
}

const sep = async (l, r) => {
  const mix = new Float32Array(2 * DEMUCS_L);
  mix.set(l); mix.set(r, DEMUCS_L);
  const feeds = { mix: new ort.Tensor('float32', mix, [1, 2, DEMUCS_L]), spec: new ort.Tensor('float32', demucsSpec(l, r), [1, 4, 2048, DEMUCS_T]) };
  if (demucs.length === 1) { const o = await demucs[0].run(feeds); return demucsISpec(o.zs.data, o.xt.data); }
  const st = [];
  for (let i = 0; i < 4; i++) { const o = await demucs[i].run(feeds); st.push(demucsISpec(o.zs.data, o.xt.data, [i])[i]); }
  return st;
};
const runRmvpe = async (x) => {
  const { data, T, Tpad } = melSpectrogram(x);
  const o = await rmvpe.run({ [rmvpe.inputNames[0]]: new ort.Tensor('float32', data, [1, 128, Tpad]) });
  return decodePitch(o[rmvpe.outputNames[0]].data, T);
};
// 長尺音声用: 20 秒窓・2 秒重なりで RMVPE を分割実行しメモリ上限を避けつつ端の精度劣化を防ぐ
async function runRmvpeFull(x16, onProg) {
  const WIN = 16000 * 20, PAD = 16000 * 2, HOP = WIN - 2 * PAD;
  const totalT = Math.ceil(x16.length / 160) + 1;
  const f0 = new Float32Array(totalT), conf = new Float32Array(totalT);
  const buf = new Float32Array(WIN);
  const steps = Math.max(1, Math.ceil(x16.length / HOP));
  for (let s = 0, step = 0; s < x16.length; s += HOP, step++) {
    if (onProg) onProg(step / steps);
    buf.fill(0);
    const s0 = s - PAD;
    for (let i = 0; i < WIN; i++) { const k = s0 + i; if (k >= 0 && k < x16.length) buf[i] = x16[k]; }
    const r = await runRmvpe(buf);
    const keepLo = s === 0 ? 0 : Math.round(PAD / 160);
    const keepHi = s + HOP >= x16.length ? r.f0.length : Math.round((PAD + HOP) / 160);
    for (let j = keepLo; j < keepHi && j < r.f0.length; j++) {
      const g = Math.round(s0 / 160) + j;
      if (g >= 0 && g < totalT) { f0[g] = r.f0[j]; conf[g] = r.conf[j]; }
    }
  }
  return { f0, conf, step: 0.01 };
}
const runBp = (x) => basicPitchFull(x, async (buf) => {
  const o = await bp.run({ [bp.inputNames[0]]: new ort.Tensor('float32', buf, [1, buf.length, 1]) });
  return { note: o[BP_OUTPUTS.note].data, onset: o[BP_OUTPUTS.onset].data };
});

// ---------------- フル曲・一括高精度分離 + AI 解析 (WebGPU 向け) ----------------
async function runFullSong({ L, R, sr, overlap = 0.5, shifts = 1 }) {
  const l44 = resample(L, sr, DEMUCS_SR), r44 = resample(R, sr, DEMUCS_SR);
  const len = l44.length;
  // 4 音源 [drums, bass, other, vocals] × 2ch のオーバーラップ加算バッファ
  const acc = Array.from({ length: 4 }, () => [new Float32Array(len), new Float32Array(len)]);
  const wSum = new Float32Array(len);
  const stride = Math.max(1, Math.round(DEMUCS_L * (1 - Math.min(0.75, Math.max(0.25, overlap)))));
  // 三角窓でつなぎ目を完全平滑化 (窓の端の重みを 0 にしてエッジ誤差を排除)
  const tri = new Float32Array(DEMUCS_L);
  const half = DEMUCS_L / 2;
  for (let i = 0; i < DEMUCS_L; i++) tri[i] = Math.max(1e-3, 1 - Math.abs(i - half) / half);

  const nShifts = Math.max(1, Math.min(4, Math.round(shifts)));
  const maxShift = Math.round(0.5 * DEMUCS_SR);
  const chunksPerShift = Math.max(1, Math.ceil(Math.max(1, len - DEMUCS_L) / stride) + 1);
  let doneChunks = 0, totalChunks = chunksPerShift * nShifts;
  const wL = new Float32Array(DEMUCS_L), wR = new Float32Array(DEMUCS_L);

  for (let sh = 0; sh < nShifts; sh++) {
    const shiftOff = sh === 0 ? 0 : Math.round((sh / nShifts) * maxShift);
    for (let pos = -shiftOff; pos < len; pos += stride) {
      postMessage({ type: 'fullProgress', p: 0.05 + 0.65 * (doneChunks / totalChunks), label: `GPU 音源分離中 (${doneChunks + 1}/${totalChunks})` });
      wL.fill(0); wR.fill(0);
      for (let i = 0; i < DEMUCS_L; i++) {
        const k = pos + i;
        if (k >= 0 && k < len) { wL[i] = l44[k]; wR[i] = r44[k]; }
      }
      const st = await sep(wL, wR);
      for (let i = 0; i < DEMUCS_L; i++) {
        const k = pos + i;
        if (k < 0 || k >= len) continue;
        const w = tri[i];
        wSum[k] += w;
        for (let s = 0; s < 4; s++) { acc[s][0][k] += st[s][0][i] * w; acc[s][1][k] += st[s][1][i] * w; }
      }
      doneChunks++;
      if (pos + DEMUCS_L >= len) break;
    }
  }
  for (let k = 0; k < len; k++) {
    const inv = wSum[k] > 0 ? 1 / wSum[k] : 1;
    for (let s = 0; s < 4; s++) { acc[s][0][k] *= inv; acc[s][1][k] *= inv; }
  }
  const mono44 = (s) => {
    const x = new Float32Array(len), a = acc[s][0], b = acc[s][1];
    for (let i = 0; i < len; i++) x[i] = (a[i] + b[i]) * 0.5;
    return x;
  };

  postMessage({ type: 'fullProgress', p: 0.72, label: 'RMVPE で歌の音程を高精度解析中…' });
  const rm = await runRmvpeFull(resample(mono44(3), DEMUCS_SR, 16000), (p) =>
    postMessage({ type: 'fullProgress', p: 0.72 + 0.08 * p, label: 'RMVPE (歌) 解析中…' }),
  );
  postMessage({ type: 'fullProgress', p: 0.80, label: 'RMVPE でベースの音程を解析中…' });
  const bf = await runRmvpeFull(resample(mono44(1), DEMUCS_SR, 16000), (p) =>
    postMessage({ type: 'fullProgress', p: 0.80 + 0.06 * p, label: 'RMVPE (ベース) 解析中…' }),
  );
  postMessage({ type: 'fullProgress', p: 0.87, label: 'Basic Pitch で楽器・ベースを採譜中…' });
  const bo = await runBp(resample(mono44(2), DEMUCS_SR, BP_SR));
  const bb = await runBp(resample(mono44(1), DEMUCS_SR, BP_SR));
  const note = new Float32Array(bo.note.length), onset = new Float32Array(bo.onset.length);
  for (let i = 0; i < note.length; i++) { note[i] = Math.max(bo.note[i], bb.note[i] || 0); onset[i] = Math.max(bo.onset[i], bb.onset[i] || 0); }

  const vL = resample(acc[3][0], DEMUCS_SR, sr);
  const vR = resample(acc[3][1], DEMUCS_SR, sr);
  const d = resample(mono44(0), DEMUCS_SR, sr);
  const bpRes = { frames: bo.frames, step: BP_HOP / BP_SR, note, onset, inst: bo.note, instOn: bo.onset, bass: bb.note, bassOn: bb.onset };
  postMessage(
    { type: 'fullResult', vL, vR, d, rm, bf, bp: bpRes },
    [vL.buffer, vR.buffer, d.buffer, rm.f0.buffer, rm.conf.buffer, bf.f0.buffer, bf.conf.buffer, note.buffer, onset.buffer, bo.note.buffer, bo.onset.buffer, bb.note.buffer, bb.onset.buffer],
  );
}

async function pump() {
  if (busy) return;
  busy = true;
  while (streamer && streamer.ready()) {
    const t0 = performance.now();
    try {
      const msg = await streamer.step();
      port.postMessage(msg, [msg.vL.buffer, msg.vR.buffer, msg.d.buffer]);
      postMessage({ type: 'stat', ms: performance.now() - t0, sec: cfg.hop });
    } catch (err) {
      postMessage({ type: 'error', message: String(err.message || err) });
      streamer.next += streamer.hop;
    }
  }
  busy = false;
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') await load(m.model);
    else if (m.type === 'fullSong') await runFullSong(m);
    else if (m.type === 'port') {
      port = m.port;
      port.onmessage = (ev) => {
        const d = ev.data;
        if (d.type === 'reset') streamer = new StemStreamer({ sr: d.sr, ...cfg, sep, rmvpe: runRmvpe, bp: runBp });
        else if (d.type === 'mix' && streamer) { streamer.push(d.L, d.R); pump(); }
      };
    }
  } catch (err) {
    postMessage({ type: 'error', message: String(err.message || err) });
  }
};
