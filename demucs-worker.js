// 音源分離 + AI 解析を動かす Web Worker (「音源分離」モード)
//   AudioWorklet から入力音声を受け取り、一定間隔で Demucs をかけて 歌・ベース・ドラム・その他 に分け、
//   歌 → RMVPE, その他の楽器・ベース → Basic Pitch, ベース → RMVPE (単音) で解析して、
//   分離した音と解析結果を AudioWorklet に返す。区切り方は worklet/stemstream.js (Node のシミュレーターと共通)
//
// モデル: 'htdemucs' (1 つで 4 音) / 'ft' (htdemucs_ft: 音ごとに専用のモデル 4 つ。分離の精度が高いが 4 倍重い)
import * as ort from './node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs';
import { demucsSpec, demucsISpec, DEMUCS_L, DEMUCS_T } from './worklet/demucs.js';
import { melSpectrogram, decodePitch } from './worklet/rmvpe.js';
import { BP_OUTPUTS, basicPitchFull } from './worklet/basicpitch.js';
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
  rmvpe = await ort.InferenceSession.create(await fetchModel('models/rmvpe.onnx', 'RMVPE'), wasm);
  // GPU (WebGPU) が使えればそちらで (約 10 倍速い)。ft は GPU のときだけ (CPU では間に合わない)
  const gpu = !!navigator.gpu;
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
const runBp = (x) => basicPitchFull(x, async (buf) => {
  const o = await bp.run({ [bp.inputNames[0]]: new ort.Tensor('float32', buf, [1, buf.length, 1]) });
  return { note: o[BP_OUTPUTS.note].data, onset: o[BP_OUTPUTS.onset].data };
});

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
