// AI 解析を動かす Web Worker
//   RMVPE (歌の音程) と Basic Pitch (鳴っている全ての音符) をブラウザで実行する
//  - メインスレッドから { type: 'load' } でモデルを読み込む
//  - { type: 'port', port } で AudioWorklet と直結し、音声の窓を受け取って結果を返す
import * as ort from './node_modules/onnxruntime-web/dist/ort.wasm.min.mjs';
import { melSpectrogram, decodePitch } from './worklet/rmvpe.js';
import { BP_WIN, BP_OUTPUTS } from './worklet/basicpitch.js';

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

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') await load();
    else if (m.type === 'port') m.port.onmessage = (ev) => infer(m.port, ev.data);
    else if (m.type === 'debug') debug = m.on;
  } catch (err) {
    postMessage({ type: 'error', message: String(err.message || err) });
  }
};
