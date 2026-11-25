// 開発用: 音声ファイルの一部をオフラインで変換し、結果を画像と WAV で出力する
//
//   node tools/devtest.mjs <音声ファイル> [--start 0] [--dur 15] [--out dev-out] [--set key=value ...] [--ai] [--stems <dir>]
//   --stems: tools/separate.py で分離した vocals/drums/bass/other.wav (曲の頭から分離したもの) を使う
//
// 出力 (dev-out/):
//   roll.png  上段 = 中央成分スペクトル (短窓) + メロディと倍音のはしご (ピンク), 下段 = 全体スペクトル + 全チャンネル
//   chip.wav  ファミコン音
//   orig.wav  元音声の同じ区間
//   frames.json  フレームごとの検出結果
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { Engine } from '../worklet/engine.js';
import { NOTE_LO } from '../worklet/driver.js';

const args = process.argv.slice(2);
const opt = { start: 0, dur: 15, out: 'dev-out', set: [], rmvpe: false };
let input = null;
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === '--start') opt.start = +args[++i];
  else if (a === '--dur') opt.dur = +args[++i];
  else if (a === '--out') opt.out = args[++i];
  else if (a === '--set') opt.set.push(args[++i]);
  else if (a === '--rmvpe' || a === '--ai') opt.rmvpe = true; // RMVPE + Basic Pitch
  else if (a === '--stems') opt.stems = args[++i];
  else input = a;
}
if (!input) { console.error('使い方: node tools/devtest.mjs <file> [--start s] [--dur s] [--set key=value]'); process.exit(1); }
fs.mkdirSync(opt.out, { recursive: true });

const SR = 48000;
// 音声を読み込む (ch チャンネル, sr Hz)。区間は --start/--dur
function decode(file, ch, sr) {
  const d = spawnSync('ffmpeg', ['-v', 'error', '-ss', String(opt.start), '-t', String(opt.dur), '-i', file,
    '-f', 'f32le', '-ac', String(ch), '-ar', String(sr), '-'], { maxBuffer: 1 << 30 });
  if (d.status !== 0) { console.error(d.stderr.toString()); process.exit(1); }
  return new Float32Array(d.stdout.buffer.slice(d.stdout.byteOffset, d.stdout.byteOffset + d.stdout.byteLength));
}
const stem = (name) => path.join(opt.stems, name + '.wav');
const dec = spawnSync('ffmpeg', ['-v', 'error', '-ss', String(opt.start), '-t', String(opt.dur), '-i', input,
  '-f', 'f32le', '-ac', '2', '-ar', String(SR), '-'], { maxBuffer: 1 << 30 });
if (dec.status !== 0) { console.error(dec.stderr.toString()); process.exit(1); }
const pcm = new Float32Array(dec.stdout.buffer, dec.stdout.byteOffset, dec.stdout.byteLength / 4);
const n = pcm.length / 2;
const L = new Float32Array(n), R = new Float32Array(n);
for (let i = 0; i < n; i++) { L[i] = pcm[2 * i]; R[i] = pcm[2 * i + 1]; }

// --- RMVPE (機械学習による歌の音程推定) ---
let ext = null, bassExt = null;
if (opt.rmvpe) {
  const { runRmvpe } = await import('./rmvpe-node.mjs');
  const a16 = decode(opt.stems ? stem('vocals') : input, 1, 16000); // 分離済みなら歌だけを渡す
  ext = await runRmvpe(a16);
  if (opt.stems && !opt.set.includes('bassF0=false')) {
    const b16 = decode(stem('bass'), 1, 16000);
    bassExt = await runRmvpe(b16);
  }
  console.log(`RMVPE: ${opt.dur} 秒を ${(ext.ms / 1000).toFixed(2)} 秒で推論`);
}
// --- Basic Pitch (機械学習による多声の音符推定) ---
let bp = null;
if (opt.rmvpe) {
  const { runBasicPitch } = await import('./basicpitch-node.mjs');
  if (opt.stems) {
    // 分離済み: 「その他の楽器」とベースを別々に採譜し、全体 (コード認識用) はその合成にする
    const bo = await runBasicPitch(decode(stem('other'), 1, 22050));
    const bb = await runBasicPitch(decode(stem('bass'), 1, 22050));
    const note = new Float32Array(bo.note.length), onset = new Float32Array(bo.onset.length);
    for (let i = 0; i < note.length; i++) { note[i] = Math.max(bo.note[i], bb.note[i] || 0); onset[i] = Math.max(bo.onset[i], bb.onset[i] || 0); }
    bp = { frames: bo.frames, note, onset, inst: bo.note, instOn: bo.onset, bass: bb.note, bassOn: bb.onset, ms: bo.ms + bb.ms };
  } else bp = await runBasicPitch(decode(input, 1, 22050));
  console.log(`Basic Pitch: ${opt.dur} 秒を ${(bp.ms / 1000).toFixed(2)} 秒で推論`);
}

// --- スペクトログラム用 (ノート単位, 1 半音 = 4px) ---
const NOTE_HI_VIEW = 100, PX = 4;
const H = (NOTE_HI_VIEW - NOTE_LO) * PX;
const colsC = [], colsA = [], frames = [];
function sampleSpec(amp, binHz) {
  const col = new Float32Array(H);
  for (let y = 0; y < H; y++) {
    const note = NOTE_HI_VIEW - y / PX;
    const k = (440 * Math.pow(2, (note - 69) / 12)) / binHz;
    const k0 = Math.floor(k), t = k - k0;
    col[y] = amp[k0] * (1 - t) + amp[k0 + 1] * t;
  }
  return col;
}

const out = new Float32Array(n);
const engine = new Engine(SR, (info, eng) => {
  colsC.push(sampleSpec(eng.ampV, SR / eng.NV));
  colsA.push(sampleSpec(eng.amp, SR / eng.N));
  frames.push(info);
});
for (const kv of opt.set) {
  const [k, v] = kv.split('=');
  engine.driver.params[k] = v.includes(',') ? v.split(',').map(Number) : v === 'true' ? true : v === 'false' ? false : isNaN(Number(v)) ? v : Number(v);
}
if (ext) {
  engine.driver.extLookup = (t) => {
    const k = Math.round(t * 100);
    return k >= 0 && k < ext.f0.length ? { f0: ext.f0[k], conf: ext.conf[k] } : null;
  };
}
if (bassExt) {
  engine.driver.bassLookup = (t) => {
    const k = Math.round(t * 100);
    return k >= 0 && k < bassExt.f0.length ? { f0: bassExt.f0[k], conf: bassExt.conf[k] } : null;
  };
}
if (bp) {
  engine.driver.bpLookup = (t) => {
    const g = Math.round((t * 22050) / 256);
    if (g < 0 || g >= bp.frames) return null;
    const sl = (a) => (a ? a.subarray(g * 88, g * 88 + 88) : undefined);
    return { note: sl(bp.note), onset: sl(bp.onset), inst: sl(bp.inst), instOn: sl(bp.instOn), bass: sl(bp.bass), bassOn: sl(bp.bassOn) };
  };
}
const t0 = Date.now();
let st = null;
if (opt.stems) {
  const v = decode(stem('vocals'), 2, SR), d = decode(stem('drums'), 1, SR);
  st = { vL: new Float32Array(n), vR: new Float32Array(n), d: new Float32Array(n) };
  for (let i = 0; i < n; i++) { st.vL[i] = v[2 * i] || 0; st.vR[i] = v[2 * i + 1] || 0; st.d[i] = d[i] || 0; }
}
for (let i = 0; i < n; i += 128) {
  const m = Math.min(128, n - i);
  if (st) engine.processStems(L.subarray(i, i + m), R.subarray(i, i + m), { vL: st.vL.subarray(i, i + m), vR: st.vR.subarray(i, i + m), d: st.d.subarray(i, i + m) }, out.subarray(i, i + m), m);
  else engine.process(L.subarray(i, i + m), R.subarray(i, i + m), out.subarray(i, i + m), m);
}
const elapsed = (Date.now() - t0) / 1000;
// 出力は先読みぶん遅れているので、解析フレームの位置に合わせて詰める
const lag = Math.round(engine.driver.params.lookahead);
frames.splice(0, lag);
while (frames.length < colsC.length) frames.push({ p1: { note: -1, pitch: -1, vol: 0 }, p2: { note: -1 }, tri: { note: -1 }, noise: { vol: 0 } });

// --- 画像 ---
const W = frames.length, GAP = 6, IH = H * 2 + GAP + 20;
const img = Buffer.alloc(W * IH * 3);
const put = (x, y, r, g, b) => {
  if (x < 0 || x >= W || y < 0 || y >= IH) return;
  const o = (y * W + x) * 3; img[o] = r; img[o + 1] = g; img[o + 2] = b;
};
function drawSpec(cols, y0) {
  let mx = 1e-9;
  for (const c of cols) for (const v of c) if (v > mx) mx = v;
  cols.forEach((c, x) => {
    for (let y = 0; y < H; y++) {
      const db = 20 * Math.log10(c[y] / mx + 1e-9);
      const v = Math.max(0, Math.min(1, (db + 60) / 60));
      const g = Math.round(v * v * 200);
      put(x, y0 + y, g, g, Math.round(g * 1.1 > 255 ? 255 : g * 1.1));
    }
    // オクターブ線 (C)
    for (let note = 36; note < NOTE_HI_VIEW; note += 12) put(x, y0 + (NOTE_HI_VIEW - note) * PX, 60, 60, 90);
  });
}
drawSpec(colsC, 0);
drawSpec(colsA, H + GAP);
const yOf = (note, y0) => y0 + Math.round((NOTE_HI_VIEW - note) * PX);
frames.forEach((f, x) => {
  if (f.p1.pitch >= 0) {
    for (const y0 of [0, H + GAP]) for (let d = -1; d <= 1; d++) put(x, yOf(f.p1.pitch, y0) + d, 255, 60, 60);
    // 倍音のはしご (上段): 歌の倍音の線と重なっていれば検出は正しい
    for (let h = 2; h <= 8; h++) if (x % 2 === 0) put(x, yOf(f.p1.pitch + 12 * Math.log2(h), 0), 255, 120, 200);
  }
  if (f.p2.note >= 0) for (let d = -1; d <= 0; d++) put(x, yOf(f.p2.note, H + GAP) + d, 255, 170, 40);
  if (f.tri.note >= 0) for (let d = -1; d <= 0; d++) put(x, yOf(f.tri.note, H + GAP) + d, 60, 200, 255);
  if (f.noise.vol > 0) {
    const row = { kick: 0, snare: 1, hat: 2 }[f.noise.drum] ?? 1;
    for (let y = 0; y < 5; y++) put(x, H * 2 + GAP + 3 + row * 6 + y, 180, 240, 90);
  }
});
for (let s = 0; s < opt.dur; s++) { // 1 秒ごとの目盛り
  const x = Math.round(s * 60.0988);
  for (let y = H * 2 + GAP; y < IH; y++) put(x, y, 120, 120, 120);
}
const raw = path.join(opt.out, 'roll.rgb');
fs.writeFileSync(raw, img);
spawnSync('ffmpeg', ['-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${W}x${IH}`, '-i', raw, path.join(opt.out, 'roll.png')]);
fs.unlinkSync(raw);

// --- WAV ---
function writeWav(file, chans) {
  const len = chans[0].length, nc = chans.length;
  const b = Buffer.alloc(44 + len * nc * 2);
  b.write('RIFF', 0); b.writeUInt32LE(36 + len * nc * 2, 4); b.write('WAVEfmt ', 8);
  b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(nc, 22); b.writeUInt32LE(SR, 24);
  b.writeUInt32LE(SR * nc * 2, 28); b.writeUInt16LE(nc * 2, 32); b.writeUInt16LE(16, 34);
  b.write('data', 36); b.writeUInt32LE(len * nc * 2, 40);
  let o = 44;
  for (let i = 0; i < len; i++) for (let c = 0; c < nc; c++) {
    b.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(chans[c][i] * 32767))), o); o += 2;
  }
  fs.writeFileSync(file, b);
}
// 出力は「先読み + 解析窓の半分」だけ遅れているので、元の音と時刻が揃うように詰める
const delaySamples = Math.round(lag * (SR / 60.0988) + engine.N / 2);
const aligned = new Float32Array(n);
aligned.set(out.subarray(delaySamples));
writeWav(path.join(opt.out, 'chip.wav'), [aligned]);
writeWav(path.join(opt.out, 'orig.wav'), [L, R]);
fs.writeFileSync(path.join(opt.out, 'frames.json'), JSON.stringify(frames.map((f) => ({
  t: f.time, e: f.energy, edb: Math.round(f.edb * 10) / 10, vdb: Math.round(f.vdb * 10) / 10, bpm: f.bpm ? Math.round(f.bpm * 10) / 10 : 0, p1: f.p1, lead: f.lead, p2: f.p2.note, tri: f.tri.note, drum: f.noise.vol ? f.noise.drum : '',
}))));

// --- 統計 ---
const NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const nn = (x) => (x < 0 ? '.' : NAMES[Math.round(x) % 12] + (Math.floor(Math.round(x) / 12) - 1));
let voiced = 0, jumps = 0, octJumps = 0, prev = -1;
for (const f of frames) {
  const p = f.p1.pitch;
  if (p >= 0) {
    voiced++;
    if (prev >= 0) {
      const d = Math.abs(p - prev);
      if (d > 7) jumps++;
      if (Math.abs(d - 12) < 0.7) octJumps++;
    }
  }
  prev = p;
}
const drums = { kick: 0, snare: 0, hat: 0 };
let prevD = '';
for (const f of frames) { const d = f.noise.vol ? f.noise.drum : ''; if (d && d !== prevD) drums[d]++; prevD = d; }
console.log(`${W} フレーム (${(W / 60.0988).toFixed(1)} 秒) を ${elapsed.toFixed(1)} 秒で処理`);
console.log(`メロディ発声率 ${((voiced / W) * 100).toFixed(0)}%, 7半音超の跳躍 ${jumps} 回, オクターブ跳躍 ${octJumps} 回`);
console.log(`ドラム: kick ${drums.kick}, snare ${drums.snare}, hat ${drums.hat}`);
const line = [];
for (let i = 0; i < W; i += 15) line.push(`${(i / 60.0988).toFixed(2)}:${nn(frames[i].p1.pitch)}`);
console.log('メロディ (0.25 秒ごと):\n' + line.join(' '));
const cl = [], bl = [];
for (let i = 0; i < W; i += 30) { cl.push(`${(i / 60.0988).toFixed(1)}:${frames[i].chord ?? '-'}`); bl.push(nn(frames[i].tri.note)); }
console.log('コード (0.5 秒ごと):\n' + cl.join(' '));
console.log('ベース (0.5 秒ごと):\n' + bl.join(' '));
console.log(`出力: ${opt.out}/roll.png, chip.wav, orig.wav`);
process.exit(0); // onnxruntime のスレッドが残ってプロセスが終わらないことがあるため
