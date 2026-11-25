// 開発用: 参考アレンジと自分の変換結果を Basic Pitch で採譜し、パートごとの一致率を出す
//   node tools/compare-ref.mjs <参考音声> <変換結果> [開始秒] [長さ秒]
//   - 最上声 (C4 以上で最も高い音) = メロディ/リード
//   - 最低音 = ベース
// を 0.1 秒ごとに比べる (音程そのもの / 音名だけ)
import { spawnSync } from 'node:child_process';
import { runBasicPitch } from './basicpitch-node.mjs';

const [refFile, oursFile, start = '0', dur = '60'] = process.argv.slice(2);
async function load(f) {
  const d = spawnSync('ffmpeg', ['-v', 'error', '-ss', start, '-t', dur, '-i', f, '-f', 'f32le', '-ac', '1', '-ar', '22050', '-'], { maxBuffer: 1 << 30 });
  return runBasicPitch(new Float32Array(d.stdout.buffer, d.stdout.byteOffset, d.stdout.byteLength / 4));
}
const ref = await load(refFile), ours = await load(oursFile);
const fps = 22050 / 256;

// 参考と変換結果の時間のずれを、和音の響き (クロマ) の相関で推定して合わせる (±3 秒)
function chroma(r) {
  const out = [];
  for (let g = 0; g < r.frames; g++) {
    const c = new Float64Array(12);
    for (let m = 21; m < 109; m++) c[m % 12] += r.note[g * 88 + m - 21];
    const s = Math.hypot(...c) || 1;
    out.push(c.map((v) => v / s));
  }
  return out;
}
const ca = chroma(ref), cb = chroma(ours);
let lag = 0, bestC = -1;
for (let l = -Math.round(3 * fps); l <= Math.round(3 * fps); l++) {
  let s = 0, n = 0;
  for (let g = 0; g < ca.length; g++) {
    const h = g + l;
    if (h < 0 || h >= cb.length) continue;
    let d = 0;
    for (let k = 0; k < 12; k++) d += ca[g][k] * cb[h][k];
    s += d; n++;
  }
  if (s / n > bestC) { bestC = s / n; lag = l; }
}
console.log(`時間のずれ: 変換結果が参考より ${(lag / fps).toFixed(2)} 秒ずれている → 補正して比較`);
// 0.1 秒の区間で最も強く鳴っていた音の集合
function active(r, g0, g1) {
  const out = [];
  for (let m = 28; m <= 100; m++) {
    let p = 0;
    for (let g = g0; g < g1 && g < r.frames; g++) p = Math.max(p, r.note[g * 88 + m - 21]);
    if (p > 0.4) out.push(m);
  }
  return out;
}
const S = { top: [0, 0, 0], bass: [0, 0, 0] }; // [比較数, 音程一致, 音名一致]
for (let t = 0; t < +dur; t += 0.1) {
  const g0 = Math.round(t * fps), g1 = Math.round((t + 0.1) * fps);
  if (g0 + lag < 0) continue;
  const a = active(ref, g0, g1), b = active(ours, g0 + lag, g1 + lag);
  const topA = a.filter((m) => m >= 60).pop(), topB = b.filter((m) => m >= 60).pop();
  if (topA !== undefined && topB !== undefined) { S.top[0]++; if (topA === topB) S.top[1]++; if (topA % 12 === topB % 12) S.top[2]++; }
  const lowA = a.find((m) => m < 60), lowB = b.find((m) => m < 60);
  if (lowA !== undefined && lowB !== undefined) { S.bass[0]++; if (lowA === lowB) S.bass[1]++; if (lowA % 12 === lowB % 12) S.bass[2]++; }
}
const pct = (x, n) => (n ? ((x / n) * 100).toFixed(0) + '%' : '-');
console.log(`メロディ/リード: 音程一致 ${pct(S.top[1], S.top[0])}, 音名一致 ${pct(S.top[2], S.top[0])} (${S.top[0]} 点)`);
console.log(`ベース:          音程一致 ${pct(S.bass[1], S.bass[0])}, 音名一致 ${pct(S.bass[2], S.bass[0])} (${S.bass[0]} 点)`);
process.exit(0);
