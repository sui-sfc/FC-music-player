// 音源分離 (Demucs htdemucs) の前後処理。ネットワーク本体は ONNX (models/htdemucs.onnx, tools/export-demucs.py で作成)
//   入力: 44.1kHz ステレオ 7.8 秒 (DEMUCS_L サンプル) → 出力: drums / bass / other / vocals (各ステレオ)
// STFT は n_fft 4096, hop 1024, 周期的ハン窓, 正規化あり (PyTorch 版と同じ並び・同じ端の処理)
import { FFT } from './fft.js';

export const DEMUCS_SR = 44100;
export const DEMUCS_L = 343980; // 学習時の長さ (7.8 秒)
export const DEMUCS_SOURCES = ['drums', 'bass', 'other', 'vocals'];
const NFFT = 4096, HL = 1024, FQ = 2048, PAD = (HL / 2) * 3;
export const DEMUCS_T = Math.ceil(DEMUCS_L / HL); // 336 フレーム

let fft = null, win = null;
function init() {
  if (fft) return;
  fft = new FFT(NFFT);
  win = new Float64Array(NFFT);
  for (let i = 0; i < NFFT; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / NFFT); // 周期的ハン窓
}
// 反射で端を延ばしたときの値 (x[-i] = x[i], x[L-1+k] = x[L-1-k])
function at(x, i) {
  const L = x.length;
  if (i < 0) i = -i;
  if (i >= L) i = 2 * (L - 1) - i;
  return x[i];
}

// ネットワークへの入力 spec [1, 4, 2048, 336] (チャンネル = L実, L虚, R実, R虚)
export function demucsSpec(l, r) {
  init();
  const T = DEMUCS_T, out = new Float32Array(4 * FQ * T);
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT), sc = 1 / Math.sqrt(NFFT);
  for (let t = 0; t < T; t++) {
    const s0 = t * HL - PAD;
    // L を実部・R を虚部に入れて 1 回の FFT で両方求める
    for (let n = 0; n < NFFT; n++) { re[n] = at(l, s0 + n) * win[n]; im[n] = at(r, s0 + n) * win[n]; }
    fft.transform(re, im);
    for (let k = 0; k < FQ; k++) {
      const j = (NFFT - k) % NFFT;
      const lr = (re[k] + re[j]) / 2, li = (im[k] - im[j]) / 2;
      const rr = (im[k] + im[j]) / 2, ri = (re[j] - re[k]) / 2;
      const o = k * T + t;
      out[o] = lr * sc; out[FQ * T + o] = li * sc;
      out[2 * FQ * T + o] = rr * sc; out[3 * FQ * T + o] = ri * sc;
    }
  }
  return out;
}

// ネットワークの出力 → 音ごとのステレオ波形。want = 求める音の番号 (省略で全部)
//   zs [1, 4, 4, 2048, 336], xt [1, 4, 2, L]。戻り値: [音][ch] = Float32Array(L)
export function demucsISpec(zs, xt, want = [0, 1, 2, 3]) {
  init();
  const T = DEMUCS_T, L = DEMUCS_L, res = [];
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT), sc = 1 / Math.sqrt(NFFT) / 1.5; // 1.5 = 窓の二乗和
  for (let s = 0; s < 4; s++) {
    if (!want.includes(s)) { res.push(null); continue; }
    const ch = [new Float32Array(L), new Float32Array(L)];
    // 2 チャンネルを 1 回の逆 FFT で: (L + i·R) のスペクトルを作って逆変換すると実部 = L, 虚部 = R
    for (let t = 0; t < T; t++) {
      re.fill(0); im.fill(0);
      const base = s * 4 * FQ * T;
      for (let k = 0; k < FQ; k++) {
        const o = k * T + t;
        const lr = zs[base + o], li = zs[base + FQ * T + o];
        const rr = zs[base + 2 * FQ * T + o], ri = zs[base + 3 * FQ * T + o];
        // X[k] = Lk + i·Rk, X[N-k] = conj(Lk) + i·conj(Rk)
        re[k] += lr - ri; im[k] += li + rr;
        if (k > 0) { const j = NFFT - k; re[j] += lr + ri; im[j] += -li + rr; }
      }
      // 逆 FFT = 共役 → 順 FFT → 共役
      for (let n = 0; n < NFFT; n++) im[n] = -im[n];
      fft.transform(re, im);
      const s0 = t * HL - PAD;
      for (let n = 0; n < NFFT; n++) {
        const i = s0 + n;
        if (i < 0 || i >= L) continue;
        const w = win[n] * sc;
        ch[0][i] += re[n] * w;
        ch[1][i] += -im[n] * w;
      }
    }
    for (let c = 0; c < 2; c++) {
      const o = (s * 2 + c) * L;
      for (let i = 0; i < L; i++) ch[c][i] += xt[o + i];
    }
    res.push(ch);
  }
  return res;
}

// ---- サンプリングレート変換 (音源分離の前後で使う) ----
// 2 次ローパス 2 段で折り返しを抑えてから線形補間。連続したチャンクを順に渡す (状態を持つ)
function lpCoef(fc, sr, q) {
  const w = (2 * Math.PI * fc) / sr, cs = Math.cos(w), al = Math.sin(w) / (2 * q), a0 = 1 + al;
  return { b0: (1 - cs) / 2 / a0, b1: (1 - cs) / a0, b2: (1 - cs) / 2 / a0, a1: (-2 * cs) / a0, a2: (1 - al) / a0, x1: 0, x2: 0, y1: 0, y2: 0 };
}
function lp(f, x) {
  const y = f.b0 * x + f.b1 * f.x1 + f.b2 * f.x2 - f.a1 * f.y1 - f.a2 * f.y2;
  f.x2 = f.x1; f.x1 = x; f.y2 = f.y1; f.y1 = y;
  return y;
}
export class StreamResampler {
  constructor(inRate, outRate) {
    const fc = 0.45 * Math.min(inRate, outRate);
    this.f = [lpCoef(fc, inRate, 0.54), lpCoef(fc, inRate, 1.31)];
    this.step = outRate / inRate;
    this.ph = 0; this.prev = 0;
  }
  process(x) {
    const out = new Float32Array(Math.ceil(x.length * this.step) + 2);
    let n = 0, ph = this.ph, prev = this.prev;
    for (let i = 0; i < x.length; i++) {
      const y = lp(this.f[1], lp(this.f[0], x[i]));
      ph += this.step;
      while (ph >= 1) { ph -= 1; out[n++] = prev + (y - prev) * (1 - ph / this.step); }
      prev = y;
    }
    this.ph = ph; this.prev = prev;
    return out.subarray(0, n);
  }
}
// 一度だけの変換 (区間の頭で少し乱れるが、解析用なので問題ない)
export function resample(x, inRate, outRate) { return new StreamResampler(inRate, outRate).process(x).slice(); }
