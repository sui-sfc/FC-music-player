// AI 解析を動かす Web Worker
//   RMVPE (歌の音程) と Basic Pitch (鳴っている全ての音符) をブラウザで実行する
//  - メインスレッドから { type: 'load' } でモデルを読み込む
//  - { type: 'port', port } で AudioWorklet と直結し、音声の窓を受け取って結果を返す
import * as ort from './node_modules/onnxruntime-web/dist/ort.wasm.min.mjs';
import { melSpectrogram, decodePitch } from './worklet/rmvpe.js';
import { BP_SR, BP_WIN, BP_HOP, BP_OUTPUTS, basicPitchFull } from './worklet/basicpitch.js';
import { resample } from './worklet/demucs.js';

ort.env.wasm.wasmPaths = new URL('./node_modules/onnxruntime-web/dist/', import.meta.url).href;
// マルチスレッドは cross-origin isolation (server.js のヘッダー) が有効なときだけ使える
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

let rmvpe = null, bp = null;
let debug = false;

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
    if (loaded - lastPost > 8e6) { lastPost = loaded; postMessage({ type: 'progress', loaded, total }); }
  }
  return buf.subarray(0, loaded);
}

async function load() {
  const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  bp = await ort.InferenceSession.create(await fetchModel('models/basic-pitch.onnx', 'Basic Pitch'), opts);
  const rm = await fetchModel('models/rmvpe.onnx', 'RMVPE');
  postMessage({ type: 'progress', compiling: true });
  rmvpe = await ort.InferenceSession.create(rm, opts);
  postMessage({ type: 'ready', threads: ort.env.wasm.numThreads });
}

async function infer(port, { a16, s16, a22, s22 }) {
  const msg = { rm: null, bp: null }, transfer = [];
  const t0 = performance.now();
  try {
    // RMVPE: 歌の音程
    const { data, T, Tpad } = melSpectrogram(a16);
    const o1 = await rmvpe.run({ [rmvpe.inputNames[0]]: new ort.Tensor('float32', data, [1, 128, Tpad]) });
    const { f0, conf } = decodePitch(o1[rmvpe.outputNames[0]].data, T);
    if (debug) postMessage({ type: 'debug', start: s16, f0: Array.from(f0) }); // 開発用
    msg.rm = { start: s16, f0, conf };
    transfer.push(f0.buffer, conf.buffer);
    // Basic Pitch: 全ての音符
    const o2 = await bp.run({ [bp.inputNames[0]]: new ort.Tensor('float32', a22, [1, BP_WIN, 1]) });
    const note = o2[BP_OUTPUTS.note].data.slice(), onset = o2[BP_OUTPUTS.onset].data.slice();
    msg.bp = { start: s22, note, onset };
    transfer.push(note.buffer, onset.buffer);
  } catch (err) {
    postMessage({ type: 'error', message: String(err.message || err) });
  }
  port.postMessage(msg, transfer);
  postMessage({ type: 'stat', ms: performance.now() - t0, sec: a16.length / 16000 });
}

async function runFullSong({ L, R, sr }) {
  const mono = new Float32Array(L.length);
  for (let i = 0; i < L.length; i++) mono[i] = (L[i] + R[i]) * 0.5;
  postMessage({ type: 'fullProgress', p: 0.1, label: 'RMVPE で全曲の歌を解析中…' });
  const x16 = resample(mono, sr, 16000);
  const WIN = 16000 * 20, PAD = 16000 * 2, HOP = WIN - 2 * PAD;
  const totalT = Math.ceil(x16.length / 160) + 1;
  const f0 = new Float32Array(totalT), conf = new Float32Array(totalT), buf = new Float32Array(WIN);
  const steps = Math.max(1, Math.ceil(x16.length / HOP));
  for (let s = 0, step = 0; s < x16.length; s += HOP, step++) {
    postMessage({ type: 'fullProgress', p: 0.1 + 0.4 * (step / steps), label: 'RMVPE 解析中…' });
    buf.fill(0);
    const s0 = s - PAD;
    for (let i = 0; i < WIN; i++) { const k = s0 + i; if (k >= 0 && k < x16.length) buf[i] = x16[k]; }
    const { data, T, Tpad } = melSpectrogram(buf);
    const o1 = await rmvpe.run({ [rmvpe.inputNames[0]]: new ort.Tensor('float32', data, [1, 128, Tpad]) });
    const r = decodePitch(o1[rmvpe.outputNames[0]].data, T);
    const keepLo = s === 0 ? 0 : Math.round(PAD / 160);
    const keepHi = s + HOP >= x16.length ? r.f0.length : Math.round((PAD + HOP) / 160);
    for (let j = keepLo; j < keepHi && j < r.f0.length; j++) {
      const g = Math.round(s0 / 160) + j;
      if (g >= 0 && g < totalT) { f0[g] = r.f0[j]; conf[g] = r.conf[j]; }
    }
  }
  postMessage({ type: 'fullProgress', p: 0.55, label: 'Basic Pitch で全曲の音符を採譜中…' });
  const x22 = resample(mono, sr, BP_SR);
  const b = await basicPitchFull(x22, async (w) => {
    const o = await bp.run({ [bp.inputNames[0]]: new ort.Tensor('float32', w, [1, w.length, 1]) });
    return { note: o[BP_OUTPUTS.note].data, onset: o[BP_OUTPUTS.onset].data };
  });
  const rm = { f0, conf, step: 0.01 };
  const bpRes = { frames: b.frames, step: BP_HOP / BP_SR, note: b.note, onset: b.onset };
  postMessage({ type: 'fullResult', rm, bp: bpRes }, [f0.buffer, conf.buffer, b.note.buffer, b.onset.buffer]);
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') await load();
    else if (m.type === 'fullSong') await runFullSong(m);
    else if (m.type === 'port') m.port.onmessage = (ev) => infer(m.port, ev.data);
    else if (m.type === 'debug') debug = m.on;
  } catch (err) {
    postMessage({ type: 'error', message: String(err.message || err) });
  }
};
