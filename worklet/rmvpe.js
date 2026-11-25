// ============================================================
//  RMVPE (伴奏入りの曲から歌の音程を推定する機械学習モデル) の前後処理
//  ---------------------------------------------------------
//  モデル本体 (rmvpe.onnx) の実行は環境ごとに違う (Node / ブラウザ / ラズパイ) ので、
//  ここには純粋な計算だけを置く:
//    melSpectrogram(): 16kHz モノラル音声 → log メルスペクトログラム (1, 128, T)
//    decodePitch():    モデル出力 (T, 360) → 10ms ごとの音程 (Hz) と確信度
//  RVC (Retrieval-based Voice Conversion) の実装と同じ仕様:
//    sr 16000, n_fft = win = 1024, hop 160 (10ms), メル 128 本 (30〜8000Hz, HTK 式),
//    振幅 (パワーではない) にメル行列を掛けて log(max(x, 1e-5))
// ============================================================
import { FFT } from './fft.js';

export const RMVPE_SR = 16000;
export const RMVPE_HOP = 160;
const N_FFT = 1024;
const N_MELS = 128;
const FMIN = 30;
const FMAX = 8000;

const hzToMel = (f) => 2595 * Math.log10(1 + f / 700);
const melToHz = (m) => 700 * (Math.pow(10, m / 2595) - 1);

// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=True, norm='slaney') と同じ
function melFilterbank() {
  const nb = N_FFT / 2 + 1;
  const pts = new Float64Array(N_MELS + 2);
  const m0 = hzToMel(FMIN), m1 = hzToMel(FMAX);
  for (let i = 0; i < N_MELS + 2; i++) pts[i] = melToHz(m0 + ((m1 - m0) * i) / (N_MELS + 1));
  const W = [];
  for (let i = 0; i < N_MELS; i++) {
    const lo = pts[i], c = pts[i + 1], hi = pts[i + 2];
    const enorm = 2 / (hi - lo);
    const row = new Float32Array(nb);
    let k0 = nb, k1 = 0;
    for (let k = 0; k < nb; k++) {
      const f = (k * RMVPE_SR) / N_FFT;
      const w = Math.max(0, Math.min((f - lo) / (c - lo), (hi - f) / (hi - c)));
      if (w > 0) { row[k] = w * enorm; if (k < k0) k0 = k; k1 = k; }
    }
    W.push({ row, k0, k1 });
  }
  return W;
}

let FB = null, fft = null, hann = null;

// audio: Float32Array (16kHz モノラル)。戻り値: { data: Float32Array(128 * Tpad), T, Tpad }
// Tpad はモデルの都合で 32 の倍数に 0 埋めしたフレーム数
export function melSpectrogram(audio) {
  if (!FB) {
    FB = melFilterbank();
    fft = new FFT(N_FFT);
    hann = new Float32Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N_FFT); // periodic
  }
  const pad = N_FFT / 2, len = audio.length;
  const T = 1 + Math.floor(len / RMVPE_HOP);
  const Tpad = 32 * Math.ceil(T / 32);
  const data = new Float32Array(N_MELS * Tpad);
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT), mag = new Float32Array(N_FFT / 2 + 1);
  // center=True: 両端を反射で延長
  const at = (i) => {
    if (i < 0) i = -i;
    if (i >= len) i = 2 * (len - 1) - i;
    return i >= 0 && i < len ? audio[i] : 0;
  };
  for (let t = 0; t < T; t++) {
    const s = t * RMVPE_HOP - pad;
    for (let i = 0; i < N_FFT; i++) { re[i] = at(s + i) * hann[i]; im[i] = 0; }
    fft.transform(re, im);
    for (let k = 0; k <= N_FFT / 2; k++) mag[k] = Math.hypot(re[k], im[k]);
    for (let m = 0; m < N_MELS; m++) {
      const { row, k0, k1 } = FB[m];
      let acc = 0;
      for (let k = k0; k <= k1; k++) acc += row[k] * mag[k];
      data[m * Tpad + t] = Math.log(Math.max(acc, 1e-5));
    }
  }
  for (let m = 0; m < N_MELS; m++) for (let t = T; t < Tpad; t++) data[m * Tpad + t] = 0;
  return { data, T, Tpad };
}

// hidden: Float32Array (Tpad * 360), 戻り値: T フレームぶんの { f0: Hz (0 = 無声), conf }
const CENTS = new Float64Array(360);
for (let i = 0; i < 360; i++) CENTS[i] = 20 * i + 1997.3794084376191;
export function decodePitch(hidden, T, threshold = 0.03) {
  const f0 = new Float32Array(T), conf = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    const o = t * 360;
    let c = 0, mx = -Infinity;
    for (let i = 0; i < 360; i++) if (hidden[o + i] > mx) { mx = hidden[o + i]; c = i; }
    conf[t] = mx;
    if (mx <= threshold) continue;
    // 最大値の周り ±4 ビンの重み付き平均 (local average cents)
    let ps = 0, ws = 0;
    for (let i = Math.max(0, c - 4); i <= Math.min(359, c + 4); i++) { ps += hidden[o + i] * CENTS[i]; ws += hidden[o + i]; }
    f0[t] = 10 * Math.pow(2, ps / ws / 1200);
  }
  return { f0, conf };
}
