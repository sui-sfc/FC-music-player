// 開発用: 音声の帯域ごとの音量バランスを比べる
//   node tools/balance.mjs <file>@<開始>:<長さ> [<file>@<開始>:<長さ> ...]
// 全体の音量をそろえた上で、帯域ごとの割合 (dB) を表示する
import { spawnSync } from 'node:child_process';
import { FFT } from '../worklet/fft.js';

const SR = 48000, N = 4096;
const BANDS = [[30, 150, '低音 (三角波・キック)'], [150, 500, '中低音'], [500, 2000, '中音 (矩形波)'], [2000, 6000, '中高音'], [6000, 16000, '高音 (ノイズ・ハイハット)']];
const fft = new FFT(N);
const win = new Float64Array(N).map((_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)));

for (const spec of process.argv.slice(2)) {
  const m = spec.match(/^(.*)@([\d.]+):([\d.]+)$/);
  const [file, start, dur] = m ? [m[1], m[2], m[3]] : [spec, '0', '60'];
  const d = spawnSync('ffmpeg', ['-v', 'error', '-ss', start, '-t', dur, '-i', file, '-f', 'f32le', '-ac', '1', '-ar', String(SR), '-'], { maxBuffer: 1 << 30 });
  const a = new Float32Array(d.stdout.buffer, d.stdout.byteOffset, d.stdout.byteLength / 4);
  const e = new Float64Array(BANDS.length);
  const re = new Float64Array(N), im = new Float64Array(N);
  let total = 0;
  for (let s = 0; s + N <= a.length; s += N / 2) {
    for (let i = 0; i < N; i++) { re[i] = a[s + i] * win[i]; im[i] = 0; }
    fft.transform(re, im);
    for (let k = 1; k < N / 2; k++) {
      const f = (k * SR) / N, p = re[k] * re[k] + im[k] * im[k];
      total += p;
      BANDS.forEach(([lo, hi], b) => { if (f >= lo && f < hi) e[b] += p; });
    }
  }
  let rms = 0;
  for (const x of a) rms += x * x;
  rms = Math.sqrt(rms / a.length);
  console.log(`${file.split(/[\\/]/).pop()}  (RMS ${(20 * Math.log10(rms)).toFixed(1)} dBFS)`);
  BANDS.forEach(([, , name], b) => console.log(`  ${name.padEnd(20)} ${(10 * Math.log10(e[b] / total)).toFixed(1).padStart(6)} dB`));
}
process.exit(0);
