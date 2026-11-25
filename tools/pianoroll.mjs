// 開発用: 複数の音声を Basic Pitch で採譜し、ピアノロールを縦に並べた画像にする
//   node tools/pianoroll.mjs out.png <file>@<開始秒>:<長さ秒> [<file>@<開始>:<長さ> ...]
// 例: 参考アレンジと自分の変換結果を同じ物差しで見比べる
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import { runBasicPitch } from './basicpitch-node.mjs';

const [outPng, ...specs] = process.argv.slice(2);
const LO = 28, HI = 100, PX = 3, XS = 2; // 表示する音域, 1 半音の高さ, 時間の間引き
const rolls = [];
for (const spec of specs) {
  const m = spec.match(/^(.*)@([\d.]+):([\d.]+)$/);
  const [file, start, dur] = m ? [m[1], m[2], m[3]] : [spec, '0', '30'];
  const d = spawnSync('ffmpeg', ['-v', 'error', '-ss', start, '-t', dur, '-i', file, '-f', 'f32le', '-ac', '1', '-ar', '22050', '-'], { maxBuffer: 1 << 30 });
  const a = new Float32Array(d.stdout.buffer, d.stdout.byteOffset, d.stdout.byteLength / 4);
  rolls.push({ file, ...(await runBasicPitch(a)) });
}
const W = Math.ceil(Math.max(...rolls.map((r) => r.frames)) / XS);
const RH = (HI - LO + 1) * PX, GAP = 8, IH = rolls.length * (RH + GAP);
const img = Buffer.alloc(W * IH * 3);
const put = (x, y, c) => { if (x >= 0 && x < W && y >= 0 && y < IH) { const o = (y * W + x) * 3; img[o] = c[0]; img[o + 1] = c[1]; img[o + 2] = c[2]; } };
rolls.forEach((r, ri) => {
  const y0 = ri * (RH + GAP);
  for (let x = 0; x < W; x++) {
    for (let n = LO; n <= HI; n++) {
      let p = 0, on = 0;
      for (let k = 0; k < XS; k++) { const g = x * XS + k; if (g < r.frames) { p = Math.max(p, r.note[g * 88 + n - 21]); on = Math.max(on, r.onset[g * 88 + n - 21]); } }
      const y = y0 + (HI - n) * PX;
      const v = Math.min(255, Math.round(p * 300));
      // 黒鍵の段は少し色を変える / C の線
      const base = [1, 3, 6, 8, 10].includes(n % 12) ? 14 : 22;
      for (let dy = 0; dy < PX; dy++) put(x, y + dy, p > 0.15 ? [v, Math.round(v * 0.85), on > 0.5 ? 60 : 255 - v] : [base, base, base + 6]);
      if (n % 12 === 0) put(x, y + PX - 1, [70, 70, 110]);
    }
    if (Math.round(x * XS * 256 / 22050) !== Math.round((x - 1) * XS * 256 / 22050)) for (let y = y0 + RH - 6; y < y0 + RH; y++) put(x, y, [140, 140, 140]); // 1 秒目盛り
  }
});
fs.writeFileSync(outPng + '.rgb', img);
spawnSync('ffmpeg', ['-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${W}x${IH}`, '-i', outPng + '.rgb', outPng]);
fs.unlinkSync(outPng + '.rgb');
console.log(`${outPng}: ${W}x${IH}, ${rolls.map((r) => r.file.split(/[\\/]/).pop()).join(' / ')} (1 半音 = ${PX}px, C ごとに線, 下端の目盛り = 1 秒)`);
process.exit(0);
