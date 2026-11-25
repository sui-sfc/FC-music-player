// ============================================================
//  フル曲・高精度 2 パス MIDI アレンジャー (ブラウザ / Node 共通)
//  ---------------------------------------------------------
//  リアルタイム制約を外し、曲全体の音源分離・AI 解析結果を一望してから
//   1. 全曲 DP ビート＆小節頭 (Downbeat) 推定
//   2. 小節単位の全曲ビタビ調 (転調含む)・場面 (静か/普通/サビ) 判定
//   3. 7th / sus4 対応の全曲ビタビコード進行 (機能和声 + ベース一致)
//   4. 全曲ビタビ採譜 → 16 分音符グリッドの離散 MIDI ノート化 (4 トラック)
//   5. 2A03 APU 合成 (PCM + 21 バイト/フレーム レジスタログ + .mid)
//  を行う。
// ============================================================

import { APU, CPU_CLOCK, FRAME_RATE } from './apu.js';
import { FFT } from './fft.js';
import { Salience, NOTE_LO, NOTE_HI, noteFreq, pulsePeriod, triPeriod, DEFAULT_PARAMS } from './driver.js';
import { BP_MIDI0 } from './basicpitch.js';
import { buildDrumBank, DPCM_START_LEVEL } from './dpcm.js';

const VOCAL_LO = 45; // A2
const VOCAL_HI = 84; // C6
const freqToMidi = (f) => 69 + 12 * Math.log2(f / 440);
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);

// ---------------- 調 (Krumhansl-Schmuckler 24 調) ----------------
const KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const normProf = (p) => {
  const m = p.reduce((a, b) => a + b, 0) / 12;
  const d = p.map((v) => v - m);
  const s = Math.hypot(...d) || 1;
  return d.map((v) => v / s);
};
const PROF_MAJ = normProf([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]);
const PROF_MIN = normProf([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]);
const SCALE_MAJ = new Set([0, 2, 4, 5, 7, 9, 11]);
const SCALE_MIN = new Set([0, 2, 3, 5, 7, 8, 10, 11]);

// ---------------- リッチコード語彙 (6 種 × 12 ルート = 72 種 + N) ----------------
const RICH_CHORD_TYPES = [
  { name: '',     iv: [0, 4, 7],     bias: 0.03 }, // メジャー
  { name: 'm',    iv: [0, 3, 7],     bias: 0.03 }, // マイナー
  { name: '7',    iv: [0, 4, 7, 10], bias: -0.02 }, // ドミナント 7th
  { name: 'm7',   iv: [0, 3, 7, 10], bias: -0.02 }, // マイナー 7th
  { name: 'M7',   iv: [0, 4, 7, 11], bias: -0.03 }, // メジャー 7th
  { name: 'sus4', iv: [0, 5, 7],     bias: -0.04 }, // sus4
];
const N_RICH = RICH_CHORD_TYPES.length * 12; // 72 (72 = N)
const richChordName = (id) => (id < 0 || id >= N_RICH ? 'N' : KEY_NAMES[id % 12] + RICH_CHORD_TYPES[Math.floor(id / 12)].name);
const richChordRoot = (id) => id % 12;
const richChordIv = (id) => RICH_CHORD_TYPES[Math.floor(id / 12)].iv;

// 全曲ビタビ (任意状態列)
function viterbiFull(states, emits, trans) {
  const T = states.length;
  if (!T) return [];
  const back = new Array(T);
  let prev = Float64Array.from(emits[0]);
  for (let t = 1; t < T; t++) {
    const S = states[t], Sp = states[t - 1], E = emits[t];
    const cur = new Float64Array(S.length), bp = new Int32Array(S.length);
    for (let j = 0; j < S.length; j++) {
      let best = -Infinity, bi = 0;
      for (let i = 0; i < Sp.length; i++) {
        const v = prev[i] + trans(Sp[i], S[j], t);
        if (v > best) { best = v; bi = i; }
      }
      cur[j] = best + E[j];
      bp[j] = bi;
    }
    back[t] = bp;
    prev = cur;
  }
  const idx = new Int32Array(T);
  let j = 0;
  for (let k = 1; k < prev.length; k++) if (prev[k] > prev[j]) j = k;
  idx[T - 1] = j;
  for (let t = T - 1; t > 0; t--) idx[t - 1] = back[t][idx[t]];
  return idx;
}

// 均等遷移の高速全曲ビタビ
function viterbiFullUniform(emits, costFn) {
  const T = emits.length;
  if (!T) return new Int32Array(0);
  const S = emits[0].length;
  const back = new Array(T);
  let prev = Float64Array.from(emits[0]);
  for (let t = 1; t < T; t++) {
    let mi = 0;
    for (let i = 1; i < S; i++) if (prev[i] > prev[mi]) mi = i;
    const cost = typeof costFn === 'function' ? costFn(t) : costFn;
    const cur = new Float64Array(S), bp = new Int32Array(S);
    const move = prev[mi] - cost;
    for (let j = 0; j < S; j++) {
      if (prev[j] >= move) { cur[j] = prev[j] + emits[t][j]; bp[j] = j; }
      else { cur[j] = move + emits[t][j]; bp[j] = mi; }
    }
    back[t] = bp;
    prev = cur;
  }
  const out = new Int32Array(T);
  let j = 0;
  for (let k = 1; k < S; k++) if (prev[k] > prev[j]) j = k;
  out[T - 1] = j;
  for (let t = T - 1; t > 0; t--) out[t - 1] = back[t][out[t]];
  return out;
}

function percentile(arr, p) {
  if (!arr.length) return 0;
  const s = Float64Array.from(arr).sort();
  return s[clamp(Math.floor(s.length * p), 0, s.length - 1)];
}

function median9(a) {
  const b = a.slice().sort((x, y) => x - y);
  return b[b.length >> 1];
}

// ============================================================
//  1. 特徴抽出 (全曲 60.0988Hz, 窓中心を完全に一致)
// ============================================================
function extractFeatures({ sr, L, R, stems, rmvpe, bassRmvpe, bp, onProgress }) {
  const len = L.length;
  const N = sr > 60000 ? 16384 : 8192;
  const NV = N / 2;
  const frameLen = sr / FRAME_RATE;
  const nFrames = Math.max(1, Math.floor(len / frameLen));

  const fft = new FFT(N), fftV = new FFT(NV);
  const win = new Float32Array(N), winV = new Float32Array(NV);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1));
  for (let i = 0; i < NV; i++) winV[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (NV - 1));
  const re = new Float64Array(N), im = new Float64Array(N);
  const reV = new Float64Array(NV), imV = new Float64Array(NV);
  const amp = new Float32Array(N / 2), ampV = new Float32Array(NV / 2), ampD = new Float32Array(NV / 2);

  const A = new Salience({ binHz: sr / N, nb: N / 2, lo: NOTE_LO, hi: NOTE_HI, weights: [1, 0.8, 0.64, 0.5, 0.4, 0.32], maxFreq: Math.min(sr / 2, 9000) });
  const V = new Salience({ binHz: sr / NV, nb: NV / 2, lo: VOCAL_LO - 1, hi: VOCAL_HI + 1, weights: [1, 0.9, 0.81, 0.73, 0.66, 0.59, 0.53, 0.48, 0.43, 0.39], maxFreq: 5000, oddCheck: true, f0Penalty: 0.9, lowCut: 300 });

  // ドラム用 4 帯域 (低 / 胴 / 響き / 高)
  const binHzV = sr / NV;
  const NB = Math.min(NV / 2, Math.round(12000 / binHzV));
  const band = (lo, hi) => [Math.max(1, Math.round(lo / binHzV)), Math.min(NB - 1, Math.round(hi / binHzV))];
  const bands = [band(35, 150), band(180, 1200), band(1500, 5000), band(6000, 12000)];
  const drumSpecs = new Array(nFrames);

  const specAt = (center, n, fObj, w, rA, iA, srcL, srcR, mixOut, centerOut) => {
    const s0 = Math.round(center - n / 2);
    const lenL = srcL.length, lenR = srcR.length;
    for (let i = 0; i < n; i++) {
      const k = s0 + i;
      const l = k >= 0 && k < lenL ? srcL[k] : 0;
      const r = k >= 0 && k < lenR ? srcR[k] : 0;
      rA[i] = l * w[i]; iA[i] = r * w[i];
    }
    fObj.transform(rA, iA);
    const sc = 2 / n;
    for (let k = 0; k < n / 2; k++) {
      const j = (n - k) % n;
      const lr = (rA[k] + rA[j]) / 2, li = (iA[k] - iA[j]) / 2;
      const rr = (iA[k] + iA[j]) / 2, ri = (rA[j] - rA[k]) / 2;
      const mix = Math.hypot(lr + rr, li + ri) * sc;
      if (mixOut) mixOut[k] = mix;
      if (!centerOut) continue;
      const lm = Math.hypot(lr, li), rm = Math.hypot(rr, ri), pw = lm * lm + rm * rm;
      let mask = 0;
      if (pw > 1e-20) { const bal = (2 * lm * rm) / pw, coh = (lr * rr + li * ri) / (lm * rm + 1e-20); mask = coh > 0 ? bal * coh : 0; mask *= mask; mask *= mask; }
      centerOut[k] = mix * mask;
    }
  };

  // ドラム抜きミックス (分離済みなら)
  let specL = L, specR = R;
  if (stems && stems.d) {
    specL = new Float32Array(len); specR = new Float32Array(len);
    const dLen = stems.d.length;
    for (let i = 0; i < len; i++) { const dv = i < dLen ? stems.d[i] : 0; specL[i] = L[i] - dv; specR[i] = R[i] - dv; }
  }

  const feats = new Array(nFrames);
  const fl = Math.max(1, Math.round(frameLen));
  let maxSmax = 1e-6, maxVSal = 1e-6;

  for (let t = 0; t < nFrames; t++) {
    if (onProgress && t % 300 === 0) onProgress(0.05 + 0.35 * (t / nFrames), '特徴抽出中…');
    const center = Math.round((t + 0.5) * frameLen);
    const time = center / sr;

    // 中心 1 フレームの RMS
    let sq = 0;
    const r0 = center - (fl >> 1);
    for (let i = 0; i < fl; i++) { const k = r0 + i; if (k >= 0 && k < len) { const x = (L[k] + R[k]) * 0.5; sq += x * x; } }
    const rms = Math.sqrt(sq / fl);

    // 長窓 & 短窓スペクトル (完全に同じ中心時刻)
    specAt(center, N, fft, win, re, im, specL, specR, amp, null);
    if (stems && stems.vL && stems.d) {
      specAt(center, NV, fftV, winV, reV, imV, stems.vL, stems.vR, ampV, null);
      specAt(center, NV, fftV, winV, reV, imV, stems.d, stems.d, ampD, null);
    } else {
      specAt(center, NV, fftV, winV, reV, imV, L, R, ampD, ampV);
    }
    drumSpecs[t] = ampD.subarray(0, NB).slice();

    const smax = A.analyze(amp);
    if (smax > maxSmax) maxSmax = smax;
    const vsmax = V.analyze(ampV);
    if (vsmax > maxVSal) maxVSal = vsmax;

    let ve = 0;
    for (let k = 0; k < ampV.length; k++) ve += ampV[k] * ampV[k];

    // 外部 AI 結果の参照
    let rm = null, brm = null, bpf = null;
    if (rmvpe) {
      const k = Math.round(time / (rmvpe.step || 0.01));
      if (k >= 0 && k < rmvpe.f0.length) rm = { f0: rmvpe.f0[k], conf: rmvpe.conf[k] };
    }
    if (bassRmvpe) {
      const k = Math.round(time / (bassRmvpe.step || 0.01));
      if (k >= 0 && k < bassRmvpe.f0.length) brm = { f0: bassRmvpe.f0[k], conf: bassRmvpe.conf[k] };
    }
    if (bp) {
      const g = Math.round(time / (bp.step || 256 / 22050));
      if (g >= 0 && g < bp.frames) {
        const sl = (a) => (a ? a.subarray(g * 88, g * 88 + 88) : null);
        bpf = { note: sl(bp.note), onset: sl(bp.onset), inst: sl(bp.inst), instOn: sl(bp.instOn), bass: sl(bp.bass), bassOn: sl(bp.bassOn) };
      }
    }

    // クロマ
    const chroma = new Float64Array(12);
    let chromaTotal = 0;
    if (bpf) {
      const note = bpf.note;
      for (let m = 33; m <= 84; m++) {
        const p = note[m - BP_MIDI0];
        if (p <= 0.15) continue;
        const w = m < 48 ? 1.3 : 1;
        chroma[m % 12] += p * w; chromaTotal += p * w;
      }
    } else {
      for (let m = NOTE_LO; m <= Math.min(84, NOTE_HI); m++) {
        const s = A.s(m);
        if (s <= 0) continue;
        const w = m < 48 ? 1.3 : 1;
        chroma[m % 12] += s * w; chromaTotal += s * w;
      }
    }

    // 歌候補
    const vCands = [];
    let rmConf = 0;
    if (rm) {
      rmConf = rm.f0 > 0 ? rm.conf : 0;
      if (rm.f0 > 0 && rm.conf >= 0.05) {
        const m = freqToMidi(rm.f0), n = Math.round(m);
        const vs = (k) => (k >= V.lo && k <= V.hi ? V.sal[k - V.lo] : 0);
        const rawV = Math.max(vs(n), vs(n - 1), vs(n + 1), vs(n + 12));
        vCands.push({ m, s: rawV, conf: rm.conf });
      }
    } else {
      for (let n = VOCAL_LO; n <= VOCAL_HI; n++) {
        const s = V.s(n);
        if (s > 0 && s >= V.s(n - 1) && s > V.s(n + 1)) {
          let m = freqToMidi(V.refine(n));
          if (!(Math.abs(m - n) <= 0.6)) m = n;
          vCands.push({ m, s, conf: 0 });
        }
      }
      vCands.sort((a, b) => b.s - a.s);
      vCands.length = Math.min(vCands.length, 5);
    }

    // リード楽器候補 & 伴奏楽器候補
    const leadCands = [], instCands = [];
    if (bpf) {
      const note = bpf.inst || bpf.note, onset = bpf.instOn || bpf.onset;
      for (let m = 55; m <= 86; m++) {
        const p = note[m - BP_MIDI0];
        if (p >= 0.3 && m >= 60) leadCands.push({ m, p, on: onset ? onset[m - BP_MIDI0] : 0 });
        if (p >= 0.2 && m <= 79) instCands.push({ m, p, on: onset ? onset[m - BP_MIDI0] : 0 });
      }
      leadCands.sort((a, b) => (b.p + 0.004 * (b.m - 60)) - (a.p + 0.004 * (a.m - 60)));
      leadCands.length = Math.min(leadCands.length, 3);
      instCands.sort((a, b) => b.p - a.p);
      instCands.length = Math.min(instCands.length, 4);
    } else {
      for (let n = 60; n <= 86; n++) {
        const s = A.s(n);
        if (s > 0.2 * smax && s >= A.s(n - 1) && s > A.s(n + 1)) leadCands.push({ m: n, p: s / (smax + 1e-9), on: 0 });
      }
      leadCands.sort((a, b) => b.p - a.p);
      leadCands.length = Math.min(leadCands.length, 3);
    }

    // ベース候補
    const bassCands = [];
    if (bpf) {
      const note = bpf.bass || bpf.note, onset = bpf.bassOn || bpf.onset;
      let mx = 0;
      for (let m = 28; m <= 55; m++) mx = Math.max(mx, note[m - BP_MIDI0]);
      for (let m = 28; m <= 55; m++) {
        const p = note[m - BP_MIDI0];
        if (p >= 0.22 && p >= 0.38 * mx) bassCands.push({ m, e: Math.log(p), on: onset ? onset[m - BP_MIDI0] : 0 });
      }
    } else {
      let bmax = 0;
      for (let n = NOTE_LO; n <= 55; n++) bmax = Math.max(bmax, A.s(n));
      if (bmax >= 0.1 * smax) {
        for (let n = NOTE_LO; n <= 55; n++) {
          const s = A.s(n);
          if (s >= 0.42 * bmax && s >= 0.07 * smax && s >= A.s(n - 1) && s >= A.s(n + 1)) {
            bassCands.push({ m: n, e: Math.log(Math.max(0.05, 0.7 * (s / bmax))), on: 0 });
          }
        }
      }
    }
    if (brm && brm.f0 > 0 && brm.conf >= 0.25) {
      const m = Math.round(freqToMidi(brm.f0));
      if (m >= 24 && m <= 60) {
        const e = Math.log(Math.max(brm.conf, 0.05)) + 0.55;
        const c = bassCands.find((o) => o.m === m);
        if (c) c.e = Math.max(c.e, e) + 0.25;
        else bassCands.unshift({ m, e, on: 0 });
      }
    }
    bassCands.sort((a, b) => a.m - b.m);
    if (bassCands.length > 4) bassCands.sort((a, b) => b.e - a.e), bassCands.length = 4, bassCands.sort((a, b) => a.m - b.m);
    if (bassCands.length) bassCands[0].e += 0.25;

    feats[t] = { time, rms, smax, ve, rmConf, chroma, chromaTotal, vCands, leadCands, instCands, bassCands, hasBp: !!bpf, hasRm: !!rm };
  }

  // 全曲基準の正規化 (ゲート・歌の相対音量)
  const rmsRef = percentile(feats.map((f) => f.rms), 0.9) || 1e-6;
  const smaxRef = percentile(feats.map((f) => f.smax), 0.9) || 1e-6;
  const vsalRef = maxVSal || 1e-6;
  for (const f of feats) {
    f.gate = f.rms > Math.max(0.0015, 0.03 * rmsRef) && f.smax > 0.03 * smaxRef;
    for (const c of f.vCands) {
      c.s = clamp(c.s / (vsalRef * 0.45 + 1e-9), 0.15, 1);
    }
  }

  // 全曲 HPSS によるドラム 4 帯域エネルギー & フラックス (ウォームアップ誤差ゼロ)
  const KT = 9, KF = 8, halfT = KT >> 1;
  const drumE = Array.from({ length: nFrames }, () => new Float64Array(4));
  const drumF = Array.from({ length: nFrames }, () => new Float64Array(4));
  const tmpT = new Float32Array(KT), tmpF = new Float32Array(2 * KF + 1);
  for (let t = 0; t < nFrames; t++) {
    if (onProgress && t % 400 === 0) onProgress(0.4 + 0.15 * (t / nFrames), 'ドラム打撃成分を解析中…');
    const center = drumSpecs[t];
    for (let b = 0; b < 4; b++) {
      const [lo, hi] = bands[b];
      let e = 0;
      for (let k = lo; k <= hi; k++) {
        for (let i = 0; i < KT; i++) {
          const ti = clamp(t - halfT + i, 0, nFrames - 1);
          tmpT[i] = drumSpecs[ti][k];
        }
        const H = median9(tmpT);
        let m = 0;
        for (let d = -KF; d <= KF; d++) { const kk = k + d; tmpF[m++] = kk > 0 && kk < NB ? center[kk] : 0; }
        const Pm = median9(tmpF.subarray(0, m));
        const p2 = Pm * Pm, h2 = H * H;
        const val = center[k] * (p2 / (p2 + h2 + 1e-20));
        e += val * val;
      }
      drumE[t][b] = e;
      const prev = t > 0 ? drumE[t - 1][b] : 0;
      drumF[t][b] = Math.max(0, Math.log(e + 1e-12) - Math.log(prev + 1e-12)) * Math.sqrt(e);
    }
  }
  // 帯域ごとに曲全体の平均フラックスで正規化 (無音帯域暴走を防ぐ下限あり)
  const globalE = [0, 0, 0, 0], globalF = [0, 0, 0, 0];
  for (let t = 0; t < nFrames; t++) for (let b = 0; b < 4; b++) { globalE[b] += drumE[t][b]; globalF[b] += drumF[t][b]; }
  const meanTotE = (globalE[0] + globalE[1] + globalE[2] + globalE[3]) / nFrames;
  const fFloor = Math.sqrt(meanTotE) * 0.03 + 1e-7;
  const drumO = Array.from({ length: nFrames }, () => new Float64Array(4));
  const onsetCurve = new Float64Array(nFrames);
  for (let t = 0; t < nFrames; t++) {
    for (let b = 0; b < 4; b++) {
      const avgF = Math.max(globalF[b] / nFrames, fFloor);
      drumO[t][b] = Math.min(12, drumF[t][b] / avgF);
    }
    onsetCurve[t] = 1.8 * drumO[t][0] + 1.0 * drumO[t][1] + 0.9 * drumO[t][2] + 0.35 * drumO[t][3];
  }

  return { feats, nFrames, drumE, drumO, onsetCurve, rmsRef };
}

// ============================================================
//  2. 全曲 DP ビート＆小節頭 (Downbeat) トラッカー
// ============================================================
function trackGlobalBeats(onset, drumE, feats) {
  const T = onset.length;
  // 1) 全曲自己相関で正確な BPM (1 拍のフレーム数 P) を推定
  let mean = 0;
  for (let t = 0; t < T; t++) mean += onset[t];
  mean /= Math.max(1, T);
  const maxLag = Math.min(T - 1, 290);
  const ac = new Float64Array(maxLag + 1);
  for (let L = 0; L <= maxLag; L++) {
    let s = 0;
    for (let t = L; t < T; t++) s += (onset[t] - mean) * (onset[t - L] - mean);
    ac[L] = s / Math.max(1, T - L);
  }
  let best = 30, bv = -Infinity;
  for (let L = 18; L <= Math.min(72, maxLag); L++) {
    const prior = Math.exp(-0.5 * (Math.log2(L / 30) / 0.72) ** 2);
    const h2 = 2 * L <= maxLag ? ac[2 * L] : 0;
    const h4 = 4 * L <= maxLag ? ac[4 * L] : 0;
    const v = (ac[L] + 0.5 * h2 + 0.25 * h4) * prior;
    if (v > bv) { bv = v; best = L; }
  }
  const a = ac[best - 1] || 0, b = ac[best] || 0, c = ac[best + 1] || 0, den = a - 2 * b + c;
  const P = clamp(best + (den < 0 ? (0.5 * (a - c)) / den : 0), 18, 72);
  const bpm = (FRAME_RATE * 60) / P;

  // 2) Ellis (2007) DP ビートトラッキング (わずかなテンポ揺れにも追従)
  const score = new Float64Array(T);
  const back = new Int32Array(T).fill(-1);
  const loD = Math.max(8, Math.round(P * 0.75)), hiD = Math.min(T - 1, Math.round(P * 1.3));
  const lam = 6.0; // 周期一貫性の強さ
  for (let t = 0; t < T; t++) {
    let bestPrev = 0, bi = -1;
    for (let d = loD; d <= hiD; d++) {
      const p = t - d;
      if (p < 0) break;
      const pen = -lam * Math.log(d / P) ** 2;
      const v = score[p] + pen;
      if (bi < 0 || v > bestPrev) { bestPrev = v; bi = p; }
    }
    score[t] = onset[t] + (bi >= 0 ? bestPrev : 0);
    back[t] = bi;
  }
  let end = T - 1;
  for (let t = Math.max(0, T - Math.round(P * 2)); t < T; t++) if (score[t] > score[end]) end = t;
  const beats = [];
  for (let t = end; t >= 0; t = back[t]) beats.push(t);
  beats.reverse();
  // 曲頭・曲末まで周期 P で外挿して途切れを無くす
  while (beats.length && beats[0] > P * 0.5) beats.unshift(beats[0] - P);
  while (beats.length && beats[beats.length - 1] < T + P) beats.push(beats[beats.length - 1] + P);
  if (beats.length < 4) {
    beats.length = 0;
    for (let f = 0; f <= T + P; f += P) beats.push(f);
  }

  // 3) 4/4 小節頭 (Downbeat) 位相を全曲から決定
  // キックは 1・3 拍、スネアは 2・4 拍、コード変化 (クロマ変化) は 1 拍目に集中する
  const phaseScore = [0, 0, 0, 0];
  const beatChroma = beats.map((bf) => {
    const c = new Float64Array(12);
    const f0 = clamp(Math.round(bf), 0, T - 1), f1 = clamp(Math.round(bf + P), 0, T);
    for (let t = f0; t < f1; t++) for (let i = 0; i < 12; i++) c[i] += feats[t].chroma[i];
    const n = Math.hypot(...c) || 1;
    for (let i = 0; i < 12; i++) c[i] /= n;
    return c;
  });
  for (let bi = 1; bi < beats.length - 1; bi++) {
    const f = clamp(Math.round(beats[bi]), 0, T - 1);
    const kE = Math.sqrt(drumE[f][0]), sE = Math.sqrt(drumE[f][2]);
    let dot = 0;
    for (let i = 0; i < 12; i++) dot += beatChroma[bi][i] * beatChroma[bi - 1][i];
    const chChange = 1 - dot; // コード変化が大きいほど 1 に近い
    const mod = bi % 4;
    for (let p = 0; p < 4; p++) {
      const pos = (mod - p + 4) % 4;
      if (pos === 0) phaseScore[p] += 1.4 * kE + 2.5 * chChange;
      else if (pos === 2) phaseScore[p] += 0.9 * kE + 0.8 * chChange;
      else phaseScore[p] += 1.1 * sE;
    }
  }
  let downPhase = 0;
  for (let p = 1; p < 4; p++) if (phaseScore[p] > phaseScore[downPhase]) downPhase = p;
  // 先頭に位相ぶんの拍を補って beat 0 = 小節 1 の 1 拍目にする
  if (downPhase > 0) {
    const shift = 4 - downPhase;
    for (let k = 0; k < shift; k++) beats.unshift(beats[0] - P);
  }

  // 各フレームの連続拍位置 beatOf[t] と、各 16 分音符スロットのフレーム範囲 slotRange[s]
  const beatOf = new Float64Array(T);
  let bi = 0;
  for (let t = 0; t < T; t++) {
    while (bi + 1 < beats.length - 1 && beats[bi + 1] <= t) bi++;
    const b0 = beats[bi], b1 = beats[bi + 1] ?? (b0 + P);
    beatOf[t] = bi + (t - b0) / Math.max(1, b1 - b0);
  }
  const totalBeats = Math.ceil(beatOf[T - 1]) + 1;
  const numSlots = totalBeats * 4;
  const slotFrame = new Float64Array(numSlots + 1);
  for (let s = 0; s <= numSlots; s++) {
    const b = Math.floor(s / 4), frac = (s % 4) / 4;
    const f0 = beats[Math.min(b, beats.length - 1)] ?? (b * P);
    const f1 = beats[Math.min(b + 1, beats.length - 1)] ?? (f0 + P);
    slotFrame[s] = f0 + frac * (f1 - f0);
  }
  return { bpm, period: P, beats, beatOf, totalBeats, numSlots, slotFrame, numBars: Math.ceil(totalBeats / 4) };
}

// ============================================================
//  3. 小節単位の全曲ビタビ調 (転調含む) & 場面判定
// ============================================================
function analyzeStructure(feats, drumO, grid) {
  const { numBars, beatOf } = grid;
  const T = feats.length;
  const barChroma = Array.from({ length: numBars }, () => new Float64Array(12));
  const barMix = new Float64Array(numBars), barVoc = new Float64Array(numBars), barDrum = new Float64Array(numBars), barCnt = new Int32Array(numBars);

  for (let t = 0; t < T; t++) {
    const b = clamp(Math.floor(beatOf[t] / 4), 0, numBars - 1);
    barCnt[b]++;
    barMix[b] += feats[t].rms * feats[t].rms;
    barVoc[b] += feats[t].ve;
    barDrum[b] += drumO[t][0] + drumO[t][2];
    if (feats[t].gate && feats[t].chromaTotal > 0) {
      for (let i = 0; i < 12; i++) barChroma[b][i] += feats[t].chroma[i] / feats[t].chromaTotal;
    }
  }
  for (let b = 0; b < numBars; b++) {
    const n = Math.max(1, barCnt[b]);
    barMix[b] = Math.sqrt(barMix[b] / n);
    barVoc[b] = Math.sqrt(barVoc[b] / n);
    barDrum[b] /= n;
  }

  // 全曲ビタビ調推定 (24 調, 小節単位。転調コスト高めで誤切替を防止)
  const keyEmits = [];
  for (let b = 0; b < numBars; b++) {
    const e = new Float64Array(24);
    const ch = barChroma[b];
    let mean = 0; for (let i = 0; i < 12; i++) mean += ch[i]; mean /= 12;
    let norm = 0; const h = new Float64Array(12);
    for (let i = 0; i < 12; i++) { h[i] = ch[i] - mean; norm += h[i] * h[i]; }
    norm = Math.sqrt(norm);
    if (norm > 1e-6) {
      for (let k = 0; k < 24; k++) {
        const r = k % 12, prof = k < 12 ? PROF_MAJ : PROF_MIN;
        let dot = 0; for (let i = 0; i < 12; i++) dot += h[(r + i) % 12] * prof[i];
        e[k] = 3.0 * (dot / norm);
      }
    }
    keyEmits.push(e);
  }
  const barKey = viterbiFullUniform(keyEmits, 4.5);

  // 全曲ビタビ場面判定 (0=静か / 1=普通 / 2=サビ) — 曲全体の分布から小節境界で綺麗に切る
  const mixP90 = percentile(barMix, 0.9) || 1e-6;
  const vocP90 = percentile(barVoc, 0.9) || 1e-6;
  const drumP85 = percentile(barDrum, 0.85) || 1e-6;
  const secEmits = [];
  for (let b = 0; b < numBars; b++) {
    const m = barMix[b] / mixP90, v = barVoc[b] / vocP90, d = barDrum[b] / drumP85;
    const e = new Float64Array(3);
    // 静か: ドラムが薄い、または全体が小さい
    e[0] = (d < 0.35 ? 1.8 : -1.0) + (m < 0.55 ? 1.4 : -0.6);
    // 普通: ドラムがあり中程度の盛り上がり
    e[1] = (d >= 0.3 ? 1.0 : -0.4) + (v >= 0.25 && v < 0.82 ? 0.8 : 0.2);
    // サビ: ドラムがあり、歌・全体とも曲の上位水準 (0 小節目からのサビ始まりにも対応!)
    e[2] = (d >= 0.45 ? 1.0 : -2.5) + (v >= 0.78 ? 1.6 : (v - 0.75) * 3.0) + (m >= 0.78 ? 0.9 : -0.5);
    secEmits.push(e);
  }
  const barSec = viterbiFullUniform(secEmits, (b) => (b % 4 === 0 ? 1.8 : 3.2)); // 4 小節単位での切替を優先

  // 曲の代表キー (最頻出)
  const kc = new Int32Array(24);
  for (let b = 0; b < numBars; b++) kc[barKey[b]]++;
  let mainKey = 0; for (let k = 1; k < 24; k++) if (kc[k] > kc[mainKey]) mainKey = k;
  return { barKey, barSec, mainKey };
}

function keyNoteBias(midi, keyId) {
  const root = keyId % 12, minor = keyId >= 12;
  const pc = ((Math.round(midi) - root) % 12 + 12) % 12;
  const sc = minor ? SCALE_MIN : SCALE_MAJ;
  if (pc === 0 || pc === 7) return 0.4;
  return sc.has(pc) ? 0.22 : -0.45;
}

function keyChordBias(chordId, keyId) {
  if (chordId < 0 || chordId >= N_RICH) return 0;
  const root = keyId % 12, minor = keyId >= 12;
  const r = ((chordId % 12) - root + 12) % 12;
  const t = Math.floor(chordId / 12); // 0=maj, 1=min, 2=7, 3=m7, 4=M7, 5=sus4
  if (!minor) {
    if ((r === 0 && (t === 0 || t === 4)) || (r === 5 && (t === 0 || t === 4)) || (r === 7 && (t === 0 || t === 2 || t === 5))) return 0.75;
    if ((r === 2 || r === 4 || r === 9) && (t === 1 || t === 3)) return 0.65;
    return -0.25;
  } else {
    if ((r === 0 && (t === 1 || t === 3)) || (r === 5 && (t === 1 || t === 3)) || (r === 7 && (t === 0 || t === 1 || t === 2 || t === 5))) return 0.75;
    if ((r === 3 || r === 8 || r === 10) && (t === 0 || t === 4 || t === 2)) return 0.65;
    return -0.25;
  }
}

// ============================================================
//  4. 全曲ビタビ・リッチコード進行 (1 拍単位, 機能和声 + ベース一致)
// ============================================================
function decodeGlobalChords(feats, grid, structure) {
  const { totalBeats, beatOf } = grid;
  const { barKey } = structure;
  const T = feats.length;
  const beatChroma = Array.from({ length: totalBeats }, () => new Float64Array(12));
  const beatTotal = new Float64Array(totalBeats);
  const beatBass = Array.from({ length: totalBeats }, () => new Float64Array(12));

  for (let t = 0; t < T; t++) {
    const b = clamp(Math.floor(beatOf[t]), 0, totalBeats - 1);
    if (!feats[t].gate) continue;
    for (let i = 0; i < 12; i++) beatChroma[b][i] += feats[t].chroma[i];
    beatTotal[b] += feats[t].chromaTotal;
    if (feats[t].bassCands.length) {
      const topB = feats[t].bassCands[0];
      beatBass[b][topB.m % 12] += Math.exp(Math.min(0, topB.e));
    }
  }

  const emits = [];
  for (let b = 0; b < totalBeats; b++) {
    const e = new Float64Array(N_RICH + 1);
    const bar = Math.min(Math.floor(b / 4), barKey.length - 1);
    const keyId = barKey[bar];
    if (beatTotal[b] < 0.3) {
      for (let i = 0; i < N_RICH; i++) e[i] = -2;
      e[N_RICH] = 0;
      emits.push(e);
      continue;
    }
    const c = beatChroma[b].map((v) => v / beatTotal[b]);
    let bMax = 0, bRoot = -1;
    for (let i = 0; i < 12; i++) if (beatBass[b][i] > bMax) { bMax = beatBass[b][i]; bRoot = i; }
    for (let ti = 0; ti < RICH_CHORD_TYPES.length; ti++) {
      const { iv, bias } = RICH_CHORD_TYPES[ti];
      const invLen = iv.length === 4 ? 0.92 : 1.0; // 4 和音は 7th が明確なときだけ選ばれるように
      for (let r = 0; r < 12; r++) {
        let inC = 0;
        for (const d of iv) inC += c[(r + d) % 12];
        const s = inC * invLen - 0.55 * (1 - inC) + bias + (c[r] > 0.12 ? 0.08 : 0) + (r === bRoot ? 0.16 : 0);
        const id = ti * 12 + r;
        e[id] = 7.5 * s + keyChordBias(id, keyId);
      }
    }
    e[N_RICH] = 1.2;
    emits.push(e);
  }

  // 小節頭 (beat % 4 === 0) が最もコード変化しやすく、3 拍目が次点、2・4 拍目は変化しにくい
  const states = Array.from({ length: totalBeats }, () => Array.from({ length: N_RICH + 1 }, (_, i) => i));
  const trans = (a, b, t) => {
    if (a === b) return 0;
    const beatPos = t % 4;
    let cost = beatPos === 0 ? 1.8 : beatPos === 2 ? 2.6 : 4.2;
    if (a < N_RICH && b < N_RICH) {
      // 機能和声 (五度圏の強進行 V→I, ii→V, IV→V, vi→ii 等に軽い加点)
      const diff = ((b % 12) - (a % 12) + 12) % 12;
      if (diff === 5 || diff === 7 || diff === 2) cost -= 0.35;
    }
    return -cost;
  };
  return viterbiFull(states, emits, trans);
}

// ============================================================
//  5. 全曲ビタビ採譜 → 16 分音符 MIDI ノート化 (4 トラック)
// ============================================================
function transcribeToMidi(feats, drumE, drumO, grid, structure, beatChords, P) {
  const { numSlots, slotFrame, beatOf, totalBeats, numBars } = grid;
  const { barKey, barSec } = structure;
  const T = feats.length;

  // ---------- Track 0: メロディ (Pulse 1) ----------
  const melTrans = (a, b) => {
    if (a < 0 && b < 0) return 0;
    if (a < 0 || b < 0) return -1.3;
    const d = Math.abs(a - b);
    return d <= 1 ? -0.75 * d * d : -(P.noteCost + 0.08 * d);
  };
  const vStates = new Array(T), vEmits = new Array(T);
  const lStates = new Array(T), lEmits = new Array(T);
  for (let t = 0; t < T; t++) {
    const f = feats[t];
    const bar = clamp(Math.floor(beatOf[t] / 4), 0, numBars - 1);
    const keyId = barKey[bar];
    const bIdx = clamp(Math.floor(beatOf[t]), 0, totalBeats - 1);
    const cid = beatChords[bIdx];
    const chordToneBonus = (m) => {
      if (cid < 0 || cid >= N_RICH) return 0;
      const pc = ((Math.round(m) - richChordRoot(cid)) % 12 + 12) % 12;
      return richChordIv(cid).includes(pc) ? 0.12 : 0;
    };
    const vs = [], ve = [];
    if (f.gate) {
      for (const c of f.vCands) {
        const sc = f.hasRm ? (c.conf / (P.extThr || 0.1)) * (P.voiceThr || 0.2) : c.s;
        vs.push(c.m);
        ve.push(Math.log(Math.max(sc, 1e-4)) + keyNoteBias(c.m, keyId) + chordToneBonus(c.m));
      }
    }
    vs.push(-1); ve.push(Math.log(P.voiceThr || 0.2));
    vStates[t] = vs; vEmits[t] = ve;

    const ls = [], le = [];
    if (f.gate) {
      for (const c of f.leadCands) {
        ls.push(c.m);
        le.push(Math.log(Math.max(c.p, 1e-4)) + keyNoteBias(c.m, keyId) + chordToneBonus(c.m));
      }
    }
    ls.push(-1); le.push(Math.log(0.32));
    lStates[t] = ls; lEmits[t] = le;
  }
  const vPathIdx = viterbiFull(vStates, vEmits, melTrans);
  const lPathIdx = viterbiFull(lStates, lEmits, melTrans);

  // 小節単位で「本当に歌がある小節か」を判定 (RMVPE が楽器を拾う誤認を抑え、無歌区間はリード楽器へ)
  const barHasVocal = new Uint8Array(numBars);
  for (let b = 0; b < numBars; b++) {
    let cs = 0, cn = 0, vn = 0, tot = 0;
    const s0 = Math.max(0, Math.floor(slotFrame[b * 16] || 0));
    const s1 = Math.min(T, Math.ceil(slotFrame[Math.min(numSlots, (b + 1) * 16)] || T));
    for (let t = s0; t < s1; t++) {
      tot++;
      if (vStates[t][vPathIdx[t]] >= 0) vn++;
      if (feats[t].rmConf > 0.03) { cs += feats[t].rmConf; cn++; }
    }
    const hasRm = feats[Math.min(s0, T - 1)]?.hasRm;
    barHasVocal[b] = hasRm ? (cn > 4 && cs / cn >= (P.vocalPresence || 0.25) ? 1 : 0) : (vn > tot * 0.1 ? 1 : 0);
  }

  // 各 16 分音符スロットのメロディ音程と再発音 (同音連打) を決定
  const melSlotNote = new Int16Array(numSlots).fill(-1);
  const melSlotVel = new Float32Array(numSlots);
  const melSlotReArtic = new Uint8Array(numSlots);
  const melSlotVocal = new Uint8Array(numSlots);
  const melShift = P.melOctave === 'auto' ? null : 12 * (Number(P.melOctave) || 0);

  // オクターブ自動時の代表シフト
  let autoShift = 0;
  if (melShift === null) {
    const allM = [];
    for (let t = 0; t < T; t++) { const m = vStates[t][vPathIdx[t]]; if (m >= 0) allM.push(m); }
    allM.sort((a, b) => a - b);
    const med = allM.length ? allM[allM.length >> 1] : 64;
    autoShift = Math.abs(med + 12 - 72) < Math.abs(med - 72) ? 12 : 0;
  }

  let prevSlotAmp = 0, prevSlotMin = 1e9, prevSlotPeak = 0;
  for (let s = 0; s < numSlots; s++) {
    const f0 = clamp(Math.round(slotFrame[s]), 0, T - 1);
    const f1 = clamp(Math.round(slotFrame[s + 1]), f0 + 1, T);
    const bar = Math.min(Math.floor(s / 16), numBars - 1);
    const useVocal = !!barHasVocal[bar];
    const counts = new Map();
    let vSum = 0, ampSum = 0, ampMin = Infinity, ampMax = 0, onsetHit = false;
    for (let t = f0; t < f1; t++) {
      let m = useVocal ? vStates[t][vPathIdx[t]] : -1;
      let isV = m >= 0;
      if (m < 0 && !useVocal && P.lead) m = lStates[t][lPathIdx[t]];
      if (m >= 0) {
        const n = Math.round(m) + (isV ? (melShift ?? autoShift) : 0);
        counts.set(n, (counts.get(n) || 0) + 1);
        const sal = isV ? (feats[t].vCands[vPathIdx[t]]?.s ?? 0.6) : (feats[t].leadCands[lPathIdx[t]]?.p ?? 0.5);
        vSum += Math.sqrt(sal);
        const a = Math.sqrt(feats[t].ve || 0);
        ampSum += a; if (a < ampMin) ampMin = a; if (a > ampMax) ampMax = a;
        if (!isV && feats[t].leadCands[lPathIdx[t]]?.on > 0.45) onsetHit = true;
      }
    }
    let bestN = -1, bestC = 0;
    for (const [n, c] of counts) if (c > bestC) { bestC = c; bestN = n; }
    if (bestN >= 0 && bestC >= Math.max(1, (f1 - f0) * 0.45)) {
      melSlotNote[s] = bestN;
      melSlotVel[s] = clamp(Math.round((vSum / bestC) * 115), 48, 124);
      melSlotVocal[s] = useVocal ? 1 : 0;
      const avgAmp = ampSum / bestC;
      if (s > 0 && melSlotNote[s - 1] === bestN) {
        if (onsetHit || (prevSlotPeak > 0 && prevSlotMin < prevSlotPeak * 0.68 && avgAmp > prevSlotMin * 1.32 && avgAmp > prevSlotAmp * 1.08)) {
          melSlotReArtic[s] = 1;
          prevSlotPeak = avgAmp; prevSlotMin = avgAmp;
        } else {
          prevSlotPeak = Math.max(prevSlotPeak, avgAmp);
          prevSlotMin = Math.min(prevSlotMin, ampMin);
        }
      } else {
        prevSlotPeak = avgAmp; prevSlotMin = avgAmp;
      }
      prevSlotAmp = avgAmp;
    } else {
      prevSlotAmp = 0; prevSlotPeak = 0; prevSlotMin = 1e9;
    }
  }
  // 1 スロットだけの不自然なヒゲ (A - B - A で半音ズレ等) を除去して打ち込み感を高める
  for (let s = 1; s < numSlots - 1; s++) {
    if (melSlotNote[s] >= 0 && melSlotNote[s - 1] >= 0 && melSlotNote[s - 1] === melSlotNote[s + 1] && melSlotNote[s] !== melSlotNote[s - 1]) {
      if (Math.abs(melSlotNote[s] - melSlotNote[s - 1]) <= 2) melSlotNote[s] = melSlotNote[s - 1];
    }
  }
  const p1Notes = [];
  for (let s = 0; s < numSlots; ) {
    const n = melSlotNote[s];
    if (n < 0) { s++; continue; }
    let e = s + 1, vel = melSlotVel[s];
    while (e < numSlots && melSlotNote[e] === n && !melSlotReArtic[e]) { vel = Math.max(vel, melSlotVel[e]); e++; }
    p1Notes.push({ startSlot: s, endSlot: e, startBeat: s / 4, endBeat: e / 4, note: n, vel, isVocal: !!melSlotVocal[s] });
    s = e;
  }

  // ---------- Track 2: ベース (Triangle) ----------
  const bStates = new Array(T), bEmits = new Array(T);
  for (let t = 0; t < T; t++) {
    const f = feats[t];
    const bar = clamp(Math.floor(beatOf[t] / 4), 0, numBars - 1);
    const keyId = barKey[bar];
    const bIdx = clamp(Math.floor(beatOf[t]), 0, totalBeats - 1);
    const cid = beatChords[bIdx];
    const cRoot = cid >= 0 && cid < N_RICH ? richChordRoot(cid) : -1;
    const bs = [], be = [];
    if (f.gate) {
      for (const c of f.bassCands) {
        const rootBonus = c.m % 12 === cRoot ? 0.65 : 0;
        bs.push(c.m);
        be.push(c.e + keyNoteBias(c.m, keyId) + rootBonus);
      }
    }
    bs.push(-1); be.push(Math.log(0.22));
    bStates[t] = bs; bEmits[t] = be;
  }
  const bPathIdx = viterbiFull(bStates, bEmits, (a, b) => (a === b ? 0 : a < 0 || b < 0 ? -0.9 : -(P.bassCost || 1.5)));

  // 8 分音符スロットごとにベース音程を確定し、場面・bassPattern に応じて MIDI ノート化
  const num8 = Math.floor(numSlots / 2);
  const triNotes = [];
  for (let e8 = 0; e8 < num8; e8++) {
    const f0 = clamp(Math.round(slotFrame[e8 * 2]), 0, T - 1);
    const f1 = clamp(Math.round(slotFrame[e8 * 2 + 2]), f0 + 1, T);
    const counts = new Map();
    for (let t = f0; t < f1; t++) {
      const m = bStates[t][bPathIdx[t]];
      if (m >= 0) counts.set(m, (counts.get(m) || 0) + 1);
    }
    let bestB = -1, bestC = 0;
    for (const [m, c] of counts) if (c > bestC) { bestC = c; bestB = m; }
    if (bestB < 0 || bestC < (f1 - f0) * 0.35) continue;
    const bar = Math.min(Math.floor(e8 / 8), numBars - 1);
    const sec = barSec[bar];
    let note = bestB + 12 * (P.bassOctave || 0);
    const doOct = P.bassPattern === 'oct' && sec >= 1;
    const do8th = P.bassPattern !== 'follow' && sec >= 1;
    if (doOct && (e8 & 1) && note <= 48) note += 12;
    if (note < (P.bassFloor || 0)) note += 12;
    if (do8th) {
      triNotes.push({ startSlot: e8 * 2, endSlot: e8 * 2 + 2, startBeat: e8 / 2, endBeat: (e8 + 1) / 2, note, vel: (e8 & 1) ? 92 : 108 });
    } else {
      const prev = triNotes[triNotes.length - 1];
      if (prev && prev.note === note && prev.endSlot === e8 * 2) { prev.endSlot += 2; prev.endBeat = (e8 + 1) / 2; }
      else triNotes.push({ startSlot: e8 * 2, endSlot: e8 * 2 + 2, startBeat: e8 / 2, endBeat: (e8 + 1) / 2, note, vel: 100 });
    }
  }

  // ---------- Track 1: 伴奏 / カウンターメロディ (Pulse 2) ----------
  // 分離済み 'other' の旋律候補も全曲ビタビで 1 本の線に整理
  const iStates = new Array(T), iEmits = new Array(T);
  for (let t = 0; t < T; t++) {
    const f = feats[t];
    const bar = clamp(Math.floor(beatOf[t] / 4), 0, numBars - 1);
    const keyId = barKey[bar];
    const melNow = melSlotNote[clamp(Math.floor(beatOf[t] * 4), 0, numSlots - 1)];
    const is = [], ie = [];
    if (f.gate && f.hasBp) {
      for (const c of f.instCands) {
        if (melNow >= 0 && (Math.abs(c.m - melNow) <= 1 || (c.m - melNow) % 12 === 0)) continue;
        is.push(c.m);
        ie.push(Math.log(Math.max(c.p, 1e-4)) + keyNoteBias(c.m, keyId) - 0.01 * Math.abs(c.m - 65));
      }
    }
    is.push(-1); ie.push(Math.log((P.instThr || 0.2) * 0.9));
    iStates[t] = is; iEmits[t] = ie;
  }
  const iPathIdx = viterbiFull(iStates, iEmits, (a, b) => (a < 0 && b < 0 ? 0 : a < 0 || b < 0 ? -1.5 : a === b ? 0 : -(P.instCost || 2.5) - 0.08 * Math.abs(a - b)));

  const p2Notes = [];
  const ARP_PAT = { up: [0, 1, 2], down: [2, 1, 0], updown: [0, 1, 2, 1], up8: [0, 1, 2, 3], '1535': [0, 2, 1, 2] };

  for (let bar = 0; bar < numBars; bar++) {
    const sec = barSec[bar];
    const sBar0 = bar * 16, sBar1 = Math.min(numSlots, (bar + 1) * 16);
    // この小節のメロディ平均音高 (伴奏がメロディより上に出ないよう調整するため)
    let melSum = 0, melCnt = 0;
    for (let s = sBar0; s < sBar1; s++) if (melSlotNote[s] >= 0) { melSum += melSlotNote[s]; melCnt++; }
    const barMelAvg = melCnt ? melSum / melCnt : 72;

    // この小節で AI 楽器カウンターメロディが十分強いなら普通場面で採用
    const instSlot = new Int16Array(sBar1 - sBar0).fill(-1);
    let instFilled = 0;
    if (P.p2mode === 'inst' || (P.p2mode === 'auto' && sec === 1)) {
      for (let s = sBar0; s < sBar1; s++) {
        const f0 = clamp(Math.round(slotFrame[s]), 0, T - 1), f1 = clamp(Math.round(slotFrame[s + 1]), f0 + 1, T);
        const cnt = new Map();
        for (let t = f0; t < f1; t++) { const m = iStates[t][iPathIdx[t]]; if (m >= 0) cnt.set(m, (cnt.get(m) || 0) + 1); }
        let bn = -1, bc = 0;
        for (const [m, c] of cnt) if (c > bc) { bc = c; bn = m; }
        if (bn >= 0 && bc >= Math.max(1, (f1 - f0) * 0.45)) {
          if (bn > barMelAvg - 2 && bn - 12 >= 48) bn -= 12;
          instSlot[s - sBar0] = bn; instFilled++;
        }
      }
    }

    if (P.p2mode === 'echo') continue; // echo はレンダラーで処理

    if (instFilled >= 6 && (P.p2mode === 'inst' || (P.p2mode === 'auto' && sec === 1))) {
      for (let i = 0; i < instSlot.length; ) {
        const n = instSlot[i]; if (n < 0) { i++; continue; }
        let j = i + 1; while (j < instSlot.length && instSlot[j] === n) j++;
        p2Notes.push({ startSlot: sBar0 + i, endSlot: sBar0 + j, startBeat: (sBar0 + i) / 4, endBeat: (sBar0 + j) / 4, note: n, vel: 84 });
        i = j;
      }
      continue;
    }

    // コードに基づく MIDI 伴奏 (静か=穏やかな8分分散 / 普通=8分アルベルティ / サビ=16分アルペジオ)
    const isArp = P.p2mode === 'arp' || (P.p2mode === 'auto' && sec === 2);
    const stepSlots = isArp ? (P.arpRate === '8' ? 2 : 1) : 2; // サビは 16 分 (1 slot), それ以外は 8 分 (2 slots)
    const patKey = isArp ? (P.arpPattern || 'updown') : '1535';
    const baseVel = sec === 0 ? 58 : sec === 2 ? 90 : 74;

    for (let s = sBar0; s < sBar1; s += stepSlots) {
      const bIdx = clamp(Math.floor(s / 4), 0, totalBeats - 1);
      const cid = beatChords[bIdx];
      if (cid < 0 || cid >= N_RICH) continue;
      // ゲート確認
      const fMid = clamp(Math.round((slotFrame[s] + slotFrame[Math.min(numSlots, s + stepSlots)]) * 0.5), 0, T - 1);
      if (!feats[fMid].gate) continue;

      const root = (P.arpLow || 62) + ((richChordRoot(cid) - ((P.arpLow || 62) % 12) + 12) % 12);
      let tones = richChordIv(cid).map((d) => root + d);
      if (tones.length < 4) tones.push(root + 12);
      // メロディより上に出る場合は和音全体を 1 オクターブ下げて綺麗なボイシングを保つ
      if (tones[tones.length - 1] > barMelAvg - 1 && tones[0] - 12 >= 48) tones = tones.map((t) => t - 12);
      const pat = (tones.length === 4 && isArp && patKey === 'up8') ? [0, 1, 2, 3] : (ARP_PAT[patKey] || ARP_PAT.updown);
      const idx = pat[Math.floor((s - sBar0) / stepSlots) % pat.length];
      const note = tones[Math.min(idx, tones.length - 1)];
      const accent = (s % 4 === 0) ? 10 : 0;
      p2Notes.push({ startSlot: s, endSlot: Math.min(sBar1, s + stepSlots), startBeat: s / 4, endBeat: Math.min(sBar1, s + stepSlots) / 4, note, vel: clamp(baseVel + accent, 32, 115) });
    }
  }

  // ---------- Track 3: ドラム (16 分グリッド吸着 + クラッシュ + 8 分ハイハット) ----------
  const thr = (P.drumSens || 1.8) * 1.7;
  const drumEvents = [];
  for (let s = 0; s < numSlots; s++) {
    const f0 = clamp(Math.round(slotFrame[s] - 1), 0, T - 1);
    const f1 = clamp(Math.round(slotFrame[s + 1] - 1), f0 + 1, T);
    const bar = Math.min(Math.floor(s / 16), numBars - 1);
    const sec = barSec[bar];
    const fMid = clamp(Math.round((f0 + f1) * 0.5), 0, T - 1);
    if (!feats[fMid].gate) continue;

    // この 16 分スロット内のピークを調べる
    let kMax = 0, sMax = 0, hMax = 0, b1Max = 0;
    let e0 = 0, e1 = 0, e2 = 0, e3 = 0;
    for (let t = f0; t < f1; t++) {
      if (drumO[t][0] > kMax) { kMax = drumO[t][0]; e0 = drumE[t][0]; }
      if (drumO[t][1] > b1Max) { b1Max = drumO[t][1]; e1 = drumE[t][1]; }
      if (drumO[t][2] > sMax) { sMax = drumO[t][2]; e2 = drumE[t][2]; }
      if (drumO[t][3] > hMax) { hMax = drumO[t][3]; e3 = drumE[t][3]; }
    }
    const eSum = e0 + e1 + e2 + e3 + 1e-12;
    const isKick = kMax > thr && e0 >= 0.22 * eSum && e0 > 1.1 * e2;
    const isSnare = sMax > thr && b1Max > thr * 0.5 && e1 >= 0.04 * eSum && e2 >= 0.04 * eSum && sMax >= 0.4 * hMax;
    const isHat = hMax > thr * 0.9 && e3 >= 0.03 * eSum;

    // サビ頭 / 場面切り替えの 1 拍目にクラッシュシンバル
    const isSectionHead = s % 16 === 0 && (bar === 0 ? sec === 2 : sec > barSec[bar - 1]);
    if (isSectionHead && sec >= 1) {
      drumEvents.push({ slot: s, beat: s / 4, kind: 'crash', vel: 112 });
    }
    if (isKick) drumEvents.push({ slot: s, beat: s / 4, kind: 'kick', vel: clamp(Math.round(75 + 45 * Math.min(1, kMax / (thr * 2.5))), 60, 125) });
    if (isSnare) drumEvents.push({ slot: s, beat: s / 4, kind: 'snare', vel: clamp(Math.round(80 + 45 * Math.min(1, sMax / (thr * 2.5))), 65, 125) });
    if (!isSectionHead && !isSnare) {
      if (isHat) {
        // 強めかつ裏拍ならオープンハット
        const open = (s % 4 === 2) && hMax > thr * 2.2;
        drumEvents.push({ slot: s, beat: s / 4, kind: open ? 'openhat' : 'hat', vel: clamp(Math.round(62 + 45 * Math.min(1, hMax / (thr * 2.5))), 50, 110) });
      } else if (sec >= 1 && s % 2 === 0) {
        // ドラム区間の 8 分ハイハット補完 (表拍を少し強めてグルーヴ感を出す)
        drumEvents.push({ slot: s, beat: s / 4, kind: 'hat', vel: (s % 4 === 0) ? 72 : 52 });
      }
    }
  }

  return { p1: p1Notes, p2: p2Notes, tri: triNotes, drums: drumEvents };
}

// ============================================================
//  6. MIDI トラック → 正確な 2A03 APU 音声 & レジスタログ合成
// ============================================================
const HQ_DRUMS = {
  kick:    { pri: 3, seq: [[13, 15], [14, 11], [14, 7], [15, 4], [15, 2]], tri: [170, 110, 75] },
  snare:   { pri: 2, seq: [[4, 15], [5, 14], [6, 12], [6, 10], [7, 8], [7, 6], [8, 4], [8, 3], [8, 2], [9, 1]] },
  hat:     { pri: 1, seq: [[3, 14], [3, 9], [3, 5], [3, 2]] },
  openhat: { pri: 1, seq: [[3, 14], [3, 12], [3, 10], [3, 8], [3, 6], [3, 5], [3, 3], [3, 2], [3, 1]] },
  crash:   { pri: 2, seq: [[4, 15], [3, 14], [4, 13], [3, 11], [4, 10], [3, 9], [4, 8], [3, 7], [4, 6], [3, 5], [4, 4], [3, 3], [4, 2], [3, 2], [4, 1], [3, 1]] },
};
const HQ_DRUMS_LAYER = {
  kick:  { pri: 3, seq: [[12, 4], [13, 2]] },
  snare: { pri: 2, seq: [[3, 15], [4, 13], [5, 11], [6, 9], [6, 7], [7, 5], [7, 3], [8, 2]] },
};

function renderArrangementToAPU({ sr, nSamples, nFrames, grid, structure, beatChords, tracks, params }) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const apu = new APU(sr);
  const dpcm = buildDrumBank();
  apu.mem.set(dpcm.bank);

  const regs = new Uint8Array(20);
  const hiWritten = [-1, -1, -1, -1];
  let mask = 0;
  const w = (a, v) => { if (a < 0x4014) regs[a - 0x4000] = v; apu.write(a, v); };
  const wHi = (ch, a, v) => { if (hiWritten[ch] !== v) { hiWritten[ch] = v; w(a, v); mask |= 1 << ch; } };
  w(0x4015, 0x0f); w(0x4001, 0x08); w(0x4005, 0x08);

  // スロット番号 → フレーム番号マップ
  const { slotFrame, beatOf, totalBeats, numBars, bpm } = grid;
  const { barKey, barSec } = structure;
  const toFrame = (s) => clamp(Math.round(slotFrame[clamp(s, 0, slotFrame.length - 1)]), 0, nFrames);

  // フレームごとのイベント配列を事前に展開 (高速・完全同期)
  const p1At = new Array(nFrames).fill(null);
  for (const n of tracks.p1) {
    const f0 = toFrame(n.startSlot), f1 = Math.max(f0 + 2, toFrame(n.endSlot));
    // 次の音と隣接している場合は末尾 1 フレームを区切りにして MIDI ゲート感を出す
    const gap = 1;
    const fEnd = Math.max(f0 + 1, f1 - gap);
    for (let t = f0; t < fEnd && t < nFrames; t++) p1At[t] = { note: n.note, vel: n.vel, age: t - f0 + 1, len: fEnd - f0 };
  }
  const p2At = new Array(nFrames).fill(null);
  for (const n of tracks.p2) {
    const f0 = toFrame(n.startSlot), f1 = Math.max(f0 + 1, toFrame(n.endSlot));
    const fEnd = Math.max(f0 + 1, f1 - (f1 - f0 >= 4 ? 1 : 0));
    for (let t = f0; t < fEnd && t < nFrames; t++) p2At[t] = { note: n.note, vel: n.vel, age: t - f0 + 1, len: fEnd - f0 };
  }
  const triAt = new Array(nFrames).fill(null);
  for (const n of tracks.tri) {
    const f0 = toFrame(n.startSlot), f1 = Math.max(f0 + 1, toFrame(n.endSlot));
    const fEnd = Math.max(f0 + 1, f1 - 1);
    for (let t = f0; t < fEnd && t < nFrames; t++) triAt[t] = { note: n.note };
  }
  const drumAt = Array.from({ length: nFrames }, () => []);
  for (const d of tracks.drums) {
    const f0 = toFrame(d.slot);
    if (f0 < nFrames) drumAt[f0].push(d);
  }

  const out = new Float32Array(nSamples);
  const regLog = new Uint8Array(nFrames * 21);
  const frames = new Array(nFrames);
  const frameLen = sr / FRAME_RATE;
  let relLeft = 0, relVol = 0, prevV1 = 0;
  let drumSeq = null, drumPos = 0, drumName = '', drumGain = 1;
  const melHist = [];

  for (let fi = 0; fi < nFrames; fi++) {
    mask = 0;
    const bar = clamp(Math.floor(beatOf[fi] / 4), 0, numBars - 1);
    const bIdx = clamp(Math.floor(beatOf[fi]), 0, totalBeats - 1);
    const sec = barSec[bar];
    const keyId = barKey[bar];
    const cid = beatChords[bIdx];

    // --- Pulse 1 (メロディ) ---
    const m1 = p1At[fi];
    let v1 = 0, duty1 = P.duty1, melPitch = -1, melNote = -1;
    if (m1) {
      relLeft = 0;
      melNote = m1.note;
      const peak = clamp(Math.round((m1.vel / 127) * 14 * (P.melodyVol || 1)), 4, 15);
      const env = 1 - P.decay + P.decay * Math.exp(-(m1.age - 1) / 10);
      v1 = clamp(Math.round(Math.min(15, peak + (P.accent || 0)) * env), 1, 15);
      const a = m1.age - (P.vibDelay || 18);
      const vib = a > 0 ? (P.vibDepth || 0.15) * Math.min(1, a / 15) * Math.sin((2 * Math.PI * a) / (P.vibPeriod || 10)) : 0;
      melPitch = melNote + vib;
      if (P.attackBend && m1.age <= 3) melPitch += P.attackBend * [1, 0.5, 0.25][m1.age - 1];
      if (m1.age <= 2 && P.dutyEnv === 'pluck') duty1 = 2;
      else if (m1.age <= 2 && P.dutyEnv === 'thin') duty1 = 0;
    } else if (P.release > 0) {
      // 次の音がすぐ来ない (フレーズ末尾) ときだけ 12.5% の余韻を残す
      const nextSoon = fi + 1 < nFrames && p1At[fi + 1];
      if (!nextSoon && prevV1 > 0 && relLeft === 0) { relLeft = P.release; relVol = prevV1; }
      if (relLeft > 0) { v1 = Math.round((relVol * relLeft) / (P.release + 2)); duty1 = 0; relLeft--; if (v1 <= 0) relLeft = 0; }
    }
    if (P.mute && P.mute[0]) v1 = 0;
    prevV1 = m1 ? v1 : 0;
    w(0x4000, (duty1 << 6) | 0x30 | v1);
    if (melNote >= 0) { const t = pulsePeriod(noteFreq(melPitch)); w(0x4002, t & 0xff); wHi(0, 0x4003, 0xf8 | (t >> 8)); }
    melHist.push({ pitch: v1 > 0 ? melPitch : -1, vol: v1 });
    if (melHist.length > 64) melHist.shift();

    // --- Pulse 2 (伴奏) ---
    let h2 = -1, v2 = 0;
    if (P.p2mode === 'echo') {
      const e = melHist[melHist.length - 1 - (P.echoDelay || 10)];
      if (e && e.pitch >= 0) { h2 = e.pitch; v2 = Math.round(e.vol * 0.45); }
    } else {
      const m2 = p2At[fi];
      if (m2) {
        h2 = m2.note;
        const frac = (m2.age - 1) / Math.max(1, m2.len);
        const peak = (m2.vel / 127) * 12 * (P.harmVol ?? 0.65);
        v2 = clamp(Math.round(peak * (1 - 0.45 * frac)), 1, 15);
      }
    }
    if (h2 < 0 || (P.mute && P.mute[1])) v2 = 0;
    w(0x4004, (P.duty2 << 6) | 0x30 | v2);
    if (h2 >= 0) { const t = pulsePeriod(noteFreq(h2)); w(0x4006, t & 0xff); wHi(1, 0x4007, 0xf8 | (t >> 8)); }

    // --- Drums (DPCM + Noise) ---
    const evs = drumAt[fi];
    let dmcHit = null;
    if (evs.length && !(P.mute && P.mute[3])) {
      const kick = evs.find((e) => e.kind === 'kick');
      const snare = evs.find((e) => e.kind === 'snare');
      const crash = evs.find((e) => e.kind === 'crash');
      const hat = evs.find((e) => e.kind === 'openhat' || e.kind === 'hat');
      const useDpcm = P.drumMode !== 'noise';
      if (useDpcm) dmcHit = kick ? 'kick' : snare ? 'snare' : null;
      // ノイズ側の優先: クラッシュ > スネア > ハット > キック補助
      let nEv = crash || snare || hat || (!useDpcm ? kick : null);
      if (useDpcm && snare && !crash) {
        drumSeq = HQ_DRUMS_LAYER.snare; drumPos = 0; drumName = 'snare'; drumGain = 0.55 + 0.45 * (snare.vel / 127);
      } else if (nEv) {
        const d = HQ_DRUMS[nEv.kind] || HQ_DRUMS.hat;
        const cur = drumSeq && drumPos < drumSeq.seq.length ? drumSeq : null;
        if (!cur || d.pri >= cur.pri || drumPos >= 2) {
          drumSeq = d; drumPos = 0; drumName = nEv.kind; drumGain = 0.5 + 0.5 * (nEv.vel / 127);
        }
      } else if (useDpcm && kick) {
        const cur = drumSeq && drumPos < drumSeq.seq.length ? drumSeq : null;
        if (!cur) { drumSeq = HQ_DRUMS_LAYER.kick; drumPos = 0; drumName = 'kick'; drumGain = 0.7; }
      }
    }
    let nIdx = regs[14] & 15, nVol = 0, dLabel = '', kickTriFreq = 0;
    if (drumSeq && drumPos < drumSeq.seq.length) {
      [nIdx, nVol] = drumSeq.seq[drumPos];
      nVol = clamp(Math.round(nVol * (P.drumVol ?? 1) * drumGain), 0, 15);
      dLabel = drumName;
      if (drumSeq.tri && P.kickTri && drumPos < drumSeq.tri.length) kickTriFreq = drumSeq.tri[drumPos];
      drumPos++;
    }
    if (P.mute && P.mute[3]) nVol = 0;
    w(0x400c, 0x30 | nVol); w(0x400e, nIdx);
    if (nVol > 0 && drumPos === 1) { w(0x400f, 0xf8); mask |= 8; }
    if (dmcHit) {
      const smp = dpcm.samples[dmcHit];
      w(0x4011, DPCM_START_LEVEL); w(0x4010, dpcm.rate); w(0x4012, smp.reg12); w(0x4013, smp.reg13);
      apu.write(0x4015, 0x0f); apu.write(0x4015, 0x1f); mask |= 16;
      if (!dLabel) dLabel = dmcHit;
    }

    // --- Triangle (ベース / キック) ---
    let triOn = false, triNote = -1;
    if (!(P.mute && P.mute[2])) {
      let f = 0;
      if (kickTriFreq) f = kickTriFreq;
      else if (triAt[fi]) { triNote = triAt[fi].note; f = noteFreq(triNote); }
      if (f > 0) { const t = triPeriod(f); w(0x400a, t & 0xff); wHi(2, 0x400b, 0xf8 | (t >> 8)); triOn = true; }
    }
    w(0x4008, triOn ? 0xff : 0x80);

    // レジスタログ & モニター用フレーム
    regLog.set(regs, fi * 21);
    regLog[fi * 21 + 20] = mask;
    frames[fi] = {
      p1: { note: v1 > 0 ? melNote : -1, pitch: v1 > 0 ? melPitch : -1, vol: v1 },
      p2: { note: v2 > 0 ? h2 : -1, vol: v2 },
      tri: { note: triOn ? (kickTriFreq ? -2 : triNote) : -1 },
      noise: { idx: nIdx, vol: nVol, drum: dLabel === 'openhat' || dLabel === 'crash' ? 'hat' : dLabel },
      chord: richChordName(cid),
      key: KEY_NAMES[keyId % 12] + (keyId >= 12 ? 'm' : ''),
      regs: regs.slice(0, 20),
      mask,
      energy: sec,
      bpm,
      time: fi / FRAME_RATE,
    };

    // 1 フレーム分の 2A03 波形を合成
    const s0 = Math.round(fi * frameLen), s1 = Math.min(nSamples, Math.round((fi + 1) * frameLen));
    for (let i = s0; i < s1; i++) out[i] = apu.sample();
  }

  return { pcm: out, regLog, frames };
}

// ============================================================
//  エントリポイント: フル曲高精度 MIDI アレンジ
// ============================================================
export function arrangeFullSong({ sr, L, R, stems = null, rmvpe = null, bassRmvpe = null, bp = null, params = {}, onProgress = null }) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const { feats, nFrames, drumE, drumO, onsetCurve } = extractFeatures({ sr, L, R, stems, rmvpe, bassRmvpe, bp, onProgress });
  if (onProgress) onProgress(0.6, '全曲ビート・小節頭を推定中…');
  const grid = trackGlobalBeats(onsetCurve, drumE, feats);
  if (onProgress) onProgress(0.7, '全曲キー・転調・構成 (Aメロ/サビ) を解析中…');
  const structure = analyzeStructure(feats, drumO, grid);
  if (onProgress) onProgress(0.78, '全曲コード進行 (7th/sus4) を推定中…');
  const beatChords = decodeGlobalChords(feats, grid, structure);
  if (onProgress) onProgress(0.86, '16 分グリッド MIDI ノートへ編曲中…');
  const tracks = transcribeToMidi(feats, drumE, drumO, grid, structure, beatChords, P);
  if (onProgress) onProgress(0.93, '2A03 APU で波形合成中…');
  const rendered = renderArrangementToAPU({ sr, nSamples: L.length, nFrames, grid, structure, beatChords, tracks, params: P });
  if (onProgress) onProgress(1.0, '完了');

  const keyRoot = structure.mainKey % 12, keyMinor = structure.mainKey >= 12;
  return {
    bpm: grid.bpm,
    keyRoot,
    keyMinor,
    keyName: KEY_NAMES[keyRoot] + (keyMinor ? 'm' : ''),
    numBars: grid.numBars,
    barSections: Array.from(structure.barSec),
    barChords: Array.from({ length: grid.numBars }, (_, b) => richChordName(beatChords[Math.min(b * 4, beatChords.length - 1)])),
    tracks,
    ...rendered,
  };
}
