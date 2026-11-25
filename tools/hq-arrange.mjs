// ============================================================
//  GPU 最高精度・フル曲 MIDI ファミコンアレンジ CLI
//  ---------------------------------------------------------
//  使い方:
//    node tools/hq-arrange.mjs <音声ファイル> [オプション]
//  オプション:
//    --out <dir>         出力先 (既定: dev-out/hq)
//    --start <sec>       開始秒 (既定: 0)
//    --dur <sec>         長さ秒 (省略で全曲)
//    --stems <dir>       分離済みフォルダを使う (vocals/bass/drums/other.wav)
//    --model <name>      Demucs モデル (既定: htdemucs_ft。6 音源は htdemucs_6s)
//    --shifts <N>        Demucs シフト平均回数 (既定: GPU=4, CPU=1)
//    --overlap <F>       Demucs 窓の重なり率 (既定: GPU=0.5, CPU=0.25)
//    --roformer          BS-RoFormer でボーカルを最高精度分離 (要 audio-separator)
//    --no-sep            音源分離をスキップ (AI 解析のみ / 内蔵解析のみ)
//    --no-ai             RMVPE / Basic Pitch を使わず内蔵解析のみで全曲 MIDI 化
//    --set key=value     パラメータ上書き (例: --set melOctave=auto --set bassPattern=oct)
//
//  出力 (<out>/):
//    chip.wav          2A03 APU 48kHz 変換音声
//    arrangement.mid   4 トラック Standard MIDI File (DAW / FamiTracker 向け)
//    fc-regs.bin       ファミコン実機用 21 バイト/フレーム レジスタログ
//    dpcm-c000.bin     ファミコン実機カートリッジ $C000 用 DPCM バンク
//    summary.json      BPM・調・小節ごとの構成とコード進行
// ============================================================
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { arrangeFullSong } from '../worklet/midi-arranger.js';
import { exportMidi } from '../worklet/midi-export.js';
import { buildDrumBank } from '../worklet/dpcm.js';

const args = process.argv.slice(2);
const opt = { out: 'dev-out/hq', start: 0, dur: 99999, model: 'htdemucs_ft', shifts: 0, overlap: 0, roformer: false, noSep: false, noAi: false, stems: null, set: [] };
let input = null;
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === '--out') opt.out = args[++i];
  else if (a === '--start') opt.start = +args[++i];
  else if (a === '--dur') opt.dur = +args[++i];
  else if (a === '--stems') opt.stems = args[++i];
  else if (a === '--model') opt.model = args[++i];
  else if (a === '--shifts') opt.shifts = +args[++i];
  else if (a === '--overlap') opt.overlap = +args[++i];
  else if (a === '--roformer') opt.roformer = true;
  else if (a === '--no-sep') opt.noSep = true;
  else if (a === '--no-ai') opt.noAi = true;
  else if (a === '--set') opt.set.push(args[++i]);
  else if (a === '-h' || a === '--help') {
    console.log('使い方: node tools/hq-arrange.mjs <音声ファイル> [--out dir] [--start sec] [--dur sec] [--model htdemucs_ft|htdemucs_6s] [--shifts 4] [--overlap 0.5] [--roformer] [--stems dir] [--no-sep] [--no-ai] [--set k=v]');
    process.exit(0);
  }
  else input = a;
}
if (!input) {
  console.error('使い方: node tools/hq-arrange.mjs <音声ファイル> [--out dir] [--model htdemucs_ft|htdemucs_6s] [--shifts 4] [--roformer]');
  process.exit(1);
}
fs.mkdirSync(opt.out, { recursive: true });

const SR = 48000;
function decode(file, ch, sr, useSlice = true) {
  const ffArgs = ['-v', 'error'];
  if (useSlice) ffArgs.push('-ss', String(opt.start), '-t', String(opt.dur));
  ffArgs.push('-i', file, '-f', 'f32le', '-ac', String(ch), '-ar', String(sr), '-');
  const d = spawnSync('ffmpeg', ffArgs, { maxBuffer: 1 << 30 });
  if (d.error) { console.error(`ffmpeg の起動に失敗しました: ${d.error.message}`); process.exit(1); }
  if (d.status !== 0) { console.error(d.stderr ? d.stderr.toString() : `ffmpeg failed (code ${d.status})`); process.exit(1); }
  return new Float32Array(d.stdout.buffer.slice(d.stdout.byteOffset, d.stdout.byteOffset + d.stdout.byteLength));
}

function findPython() {
  for (const cand of ['.tools/venv/Scripts/python.exe', '.tools/venv/bin/python', 'python3', 'python']) {
    if (cand.includes('/') && !fs.existsSync(cand)) continue;
    const r = spawnSync(cand, ['-c', 'import torch, demucs'], { stdio: 'ignore' });
    if (r.status === 0) return cand;
  }
  return null;
}

// 1. 音源分離 (GPU 優先)
let stemsDir = opt.stems;
let stemsSliced = false;
if (!stemsDir && !opt.noSep) {
  const py = findPython();
  if (py) {
    stemsDir = path.join(opt.out, 'stems');
    const pyArgs = ['tools/separate.py', input, stemsDir, String(opt.start), String(opt.dur), opt.model];
    if (opt.shifts > 0) pyArgs.push('--shifts', String(opt.shifts));
    if (opt.overlap > 0) pyArgs.push('--overlap', String(opt.overlap));
    if (opt.roformer) pyArgs.push('--roformer');
    console.log('[1/4] GPU 高精度音源分離を実行中…');
    const r = spawnSync(py, pyArgs, { stdio: 'inherit' });
    if (r.status !== 0) { console.error('音源分離に失敗しました (--no-sep でスキップ可能)'); process.exit(1); }
    stemsSliced = true;
  } else {
    console.log('[注意] Python (torch + demucs) が見つからないため、音源分離をスキップします');
  }
}

const mixPcm = decode(input, 2, SR, true);
const n = mixPcm.length / 2;
const L = new Float32Array(n), R = new Float32Array(n);
for (let i = 0; i < n; i++) { L[i] = mixPcm[2 * i]; R[i] = mixPcm[2 * i + 1]; }

let stems = null;
const stemPath = (name) => path.join(stemsDir, name + '.wav');
if (stemsDir && fs.existsSync(stemPath('vocals'))) {
  const v = decode(stemPath('vocals'), 2, SR, !stemsSliced);
  const d = decode(stemPath('drums'), 1, SR, !stemsSliced);
  stems = { vL: new Float32Array(n), vR: new Float32Array(n), d: new Float32Array(n) };
  for (let i = 0; i < n; i++) { stems.vL[i] = v[2 * i] || 0; stems.vR[i] = v[2 * i + 1] || 0; stems.d[i] = d[i] || 0; }
}

// 2. AI 採譜 (RMVPE + Basic Pitch, GPU 対応)
let rmvpe = null, bassRmvpe = null, bp = null;
if (!opt.noAi && fs.existsSync('models/rmvpe.onnx') && fs.existsSync('models/basic-pitch.onnx')) {
  console.log('[2/4] AI 解析 (RMVPE + Basic Pitch) を実行中…');
  const { runRmvpe } = await import('./rmvpe-node.mjs');
  const { runBasicPitch } = await import('./basicpitch-node.mjs');
  const voc16 = decode(stemsDir ? stemPath('vocals') : input, 1, 16000, !stemsSliced);
  rmvpe = await runRmvpe(voc16);
  console.log(`  RMVPE (歌): ${(rmvpe.ms / 1000).toFixed(2)} 秒`);
  if (stemsDir && fs.existsSync(stemPath('bass'))) {
    const bass16 = decode(stemPath('bass'), 1, 16000, !stemsSliced);
    bassRmvpe = await runRmvpe(bass16);
    console.log(`  RMVPE (ベース): ${(bassRmvpe.ms / 1000).toFixed(2)} 秒`);
  }
  if (stemsDir && fs.existsSync(stemPath('other')) && fs.existsSync(stemPath('bass'))) {
    const bo = await runBasicPitch(decode(stemPath('other'), 1, 22050, !stemsSliced));
    const bb = await runBasicPitch(decode(stemPath('bass'), 1, 22050, !stemsSliced));
    const note = new Float32Array(bo.note.length), onset = new Float32Array(bo.onset.length);
    for (let i = 0; i < note.length; i++) { note[i] = Math.max(bo.note[i], bb.note[i] || 0); onset[i] = Math.max(bo.onset[i], bb.onset[i] || 0); }
    bp = { frames: bo.frames, note, onset, inst: bo.note, instOn: bo.onset, bass: bb.note, bassOn: bb.onset };
    console.log(`  Basic Pitch (楽器 + ベース): ${((bo.ms + bb.ms) / 1000).toFixed(2)} 秒`);
  } else {
    bp = await runBasicPitch(decode(input, 1, 22050, true));
    console.log(`  Basic Pitch: ${(bp.ms / 1000).toFixed(2)} 秒`);
  }
} else if (!opt.noAi) {
  console.log('[注意] models/rmvpe.onnx または basic-pitch.onnx が見つからないため、内蔵解析で全曲 MIDI 化します');
}

// 3. フル曲 2 パス高精度 MIDI 編曲 + 2A03 APU 合成
console.log('[3/4] 全曲 MIDI アレンジ＆2A03 合成を実行中…');
const params = {};
for (const kv of opt.set) {
  const [k, v] = kv.split('=');
  params[k] = v.includes(',') ? v.split(',').map(Number) : v === 'true' ? true : v === 'false' ? false : isNaN(Number(v)) ? v : Number(v);
}
const res = arrangeFullSong({
  sr: SR, L, R, stems, rmvpe, bassRmvpe, bp, params,
  onProgress: (p, msg) => process.stdout.write(`\r  ${Math.round(p * 100)}% ${msg}          `),
});
console.log('');

// 4. 書き出し (.wav / .mid / .bin / summary.json)
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

console.log('[4/4] ファイルを書き出し中…');
writeWav(path.join(opt.out, 'chip.wav'), [res.pcm]);
fs.writeFileSync(path.join(opt.out, 'arrangement.mid'), exportMidi(res));
fs.writeFileSync(path.join(opt.out, 'fc-regs.bin'), res.regLog);
fs.writeFileSync(path.join(opt.out, 'dpcm-c000.bin'), buildDrumBank().bank);
fs.writeFileSync(path.join(opt.out, 'summary.json'), JSON.stringify({
  bpm: Math.round(res.bpm * 100) / 100,
  key: res.keyName,
  bars: res.numBars,
  sections: res.barSections,
  chords: res.barChords,
  counts: { p1: res.tracks.p1.length, p2: res.tracks.p2.length, tri: res.tracks.tri.length, drums: res.tracks.drums.length },
}, null, 2));

console.log(`完了! BPM=${res.bpm.toFixed(1)}, Key=${res.keyName}, ${res.numBars} 小節`);
console.log(`コード進行: ${res.barChords.slice(0, 16).join(' | ')}${res.barChords.length > 16 ? ' ...' : ''}`);
console.log(`出力: ${opt.out}/chip.wav, arrangement.mid, fc-regs.bin, dpcm-c000.bin, summary.json`);
process.exit(0);
