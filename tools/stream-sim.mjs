// 開発用: ブラウザの「音源分離 + AI 解析」モードを Node で再現する (リアルタイムと同じ区切り方で分離・解析)
//   node tools/stream-sim.mjs <音声ファイル> [--dur 70] [--model htdemucs|ft] [--hop 1] [--margin 1] [--ctx 1.5] [--out dir] [--set k=v]
// 出力: <out>/frames.json (devtest と同じ形式), chip.wav
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import ort from 'onnxruntime-node';
import { Engine } from '../worklet/engine.js';
import { StemStreamer } from '../worklet/stemstream.js';
import { demucsSpec, demucsISpec, DEMUCS_L, DEMUCS_T } from '../worklet/demucs.js';
import { melSpectrogram, decodePitch } from '../worklet/rmvpe.js';
import { BP_OUTPUTS, basicPitchFull } from '../worklet/basicpitch.js';

const args = process.argv.slice(2);
const opt = { dur: 70, model: 'htdemucs', hop: 1, margin: 1, ctx: 1.5, out: 'dev-out/sim', set: [] };
let input = null;
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a.startsWith('--') && a !== '--set') opt[a.slice(2)] = isNaN(+args[i + 1]) ? args[++i] : +args[++i];
  else if (a === '--set') opt.set.push(args[++i]);
  else input = a;
}
fs.mkdirSync(opt.out, { recursive: true });
const SR = 48000;
const d = spawnSync('ffmpeg', ['-v', 'error', '-t', String(opt.dur), '-i', input, '-f', 'f32le', '-ac', '2', '-ar', String(SR), '-'], { maxBuffer: 1 << 30 });
const pcm = new Float32Array(d.stdout.buffer.slice(d.stdout.byteOffset, d.stdout.byteOffset + d.stdout.byteLength));
const n = pcm.length / 2;
const L = new Float32Array(n), R = new Float32Array(n);
for (let i = 0; i < n; i++) { L[i] = pcm[2 * i]; R[i] = pcm[2 * i + 1]; }

const so = { intraOpNumThreads: 16 };
const files = opt.model === 'ft' ? [0, 1, 2, 3].map((i) => `models/htdemucs_ft_${i}.onnx`) : ['models/htdemucs.onnx'];
const dm = [];
for (const f of files) dm.push(await ort.InferenceSession.create(f, so));
const rm = await ort.InferenceSession.create('models/rmvpe.onnx', so);
const bpS = await ort.InferenceSession.create('models/basic-pitch.onnx', so);
let tSep = 0, nSep = 0;
const sep = async (l, r) => {
  const t0 = performance.now();
  const mix = new Float32Array(2 * DEMUCS_L); mix.set(l); mix.set(r, DEMUCS_L);
  const feeds = { mix: new ort.Tensor('float32', mix, [1, 2, DEMUCS_L]), spec: new ort.Tensor('float32', demucsSpec(l, r), [1, 4, 2048, DEMUCS_T]) };
  let st;
  if (dm.length === 1) { const o = await dm[0].run(feeds); st = demucsISpec(o.zs.data, o.xt.data); }
  else {
    st = [];
    for (let i = 0; i < 4; i++) { const o = await dm[i].run(feeds); st.push(demucsISpec(o.zs.data, o.xt.data, [i])[i]); }
  }
  tSep += performance.now() - t0; nSep++;
  return st;
};
const rmvpe = async (x) => {
  const { data, T, Tpad } = melSpectrogram(x);
  const o = await rm.run({ [rm.inputNames[0]]: new ort.Tensor('float32', data, [1, 128, Tpad]) });
  return decodePitch(o[rm.outputNames[0]].data, T);
};
const bp = (x) => basicPitchFull(x, async (buf) => {
  const o = await bpS.run({ [bpS.inputNames[0]]: new ort.Tensor('float32', buf, [1, buf.length, 1]) });
  return { note: o[BP_OUTPUTS.note].data, onset: o[BP_OUTPUTS.onset].data };
});

const streamer = new StemStreamer({ sr: SR, hop: opt.hop, margin: opt.margin, ctx: opt.ctx, sep, rmvpe, bp });
streamer.skipBehind = false;
const frames = [];
const engine = new Engine(SR, (info) => frames.push(info));
for (const kv of opt.set) {
  const [k, v] = kv.split('=');
  engine.driver.params[k] = v.includes(',') ? v.split(',').map(Number) : v === 'true' ? true : v === 'false' ? false : isNaN(Number(v)) ? v : Number(v);
}
const lookahead = engine.driver.params.lookahead = Math.max(60, engine.driver.params.lookahead);
// AudioWorklet との間の MessagePort の代わり
const fakePort = { onmessage: null, postMessage(m) { if (m.type === 'mix') streamer.push(m.L, m.R); } };
engine.setStemPort(fakePort, 30);
const out = new Float32Array(n + SR * 30);
const pad = SR * 6; // 最後の区間まで出し切るための無音
for (let i = 0; i < n + pad; i += 128) {
  const m = 128;
  const l = i < n ? L.subarray(i, Math.min(n, i + m)) : new Float32Array(m), r = i < n ? R.subarray(i, Math.min(n, i + m)) : new Float32Array(m);
  const o = new Float32Array(l.length);
  engine.processLive(l, r, o, l.length);
  while (streamer.ready()) fakePort.onmessage({ data: await streamer.step() });
}
// 残りを変換しきる
for (let k = 0; k < 400; k++) engine.processLive(new Float32Array(128), new Float32Array(128), new Float32Array(128), 128);
console.log(`分離: ${nSep} 回, 1 回 ${(tSep / nSep).toFixed(0)}ms (${opt.model}, hop ${opt.hop}s, margin ${opt.margin}s)`);
frames.splice(0, lookahead);
fs.writeFileSync(path.join(opt.out, 'frames.json'), JSON.stringify(frames.filter((f) => f.p1).map((f) => ({ t: f.time, p1: f.p1, lead: f.lead, p2: f.p2.note, tri: f.tri.note, drum: f.noise.vol ? f.noise.drum : '' }))));
console.log('出力:', opt.out);
process.exit(0);
