// ============================================================
//  DPCM ドラムサンプル (DMC チャンネル用)
//  ---------------------------------------------------------
//  ファミコンの DMC は「1 ビットごとに DAC を +2 / -2 する」だけの単純な方式。
//  ここでキック・スネアの波形を合成して 1 ビットデルタに変換し、
//  カートリッジの $C000 からそのまま置ける 16KB のバンクとして並べる。
//    - 開始アドレスは 64 バイト単位 ($4012 = (addr - $C000) / 64)
//    - 長さは 16n+1 バイト ($4013 = n)
//  実機では、このバンクをカートリッジ (ラズパイ側) が $C000-$FFFF に見せればよい。
// ============================================================
import { CPU_CLOCK, DMC_RATES } from './apu.js';

export const DPCM_RATE = 15; // 約 33.1kHz (最高音質)
export const DPCM_START_LEVEL = 64; // 再生前に $4011 に書く値 (DAC の中央)

// 目標の波形 (DAC 単位 0..127) を 1 ビットデルタに変換する。
// 1 ビット先読みで、次のビットまで含めて誤差が小さくなる方を選ぶ
export function encodeDPCM(target, start = DPCM_START_LEVEL) {
  const nBits = target.length;
  const nBytes = Math.ceil(nBits / 8 / 16) * 16 + 1; // 16n+1 バイトに切り上げ
  const out = new Uint8Array(nBytes);
  let level = start;
  const stepOf = (lv, bit) => (bit ? (lv <= 125 ? lv + 2 : lv) : lv >= 2 ? lv - 2 : lv);
  for (let i = 0; i < nBytes * 8; i++) {
    const tgt = i < nBits ? target[i] : start; // 余りは中央に戻す
    const tgt2 = i + 1 < nBits ? target[i + 1] : start;
    let bestBit = 0, bestErr = Infinity;
    for (const b of [0, 1]) {
      const l1 = stepOf(level, b);
      const e = Math.abs(l1 - tgt) + 0.5 * Math.min(Math.abs(stepOf(l1, 0) - tgt2), Math.abs(stepOf(l1, 1) - tgt2));
      if (e < bestErr) { bestErr = e; bestBit = b; }
    }
    level = stepOf(level, bestBit);
    if (bestBit) out[i >> 3] |= 1 << (i & 7); // 下位ビットから再生される
  }
  return out;
}

// 簡単な疑似乱数 (毎回同じサンプルになるように)
function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296) * 2 - 1;
}

function synthKick(fs) {
  const n = Math.round(fs * 0.2), w = new Float32Array(n), rnd = rng(67890);
  let ph = 0;
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const f = 48 + 145 * Math.exp(-t / 0.022); // 高い所から急に下がる
    ph += (2 * Math.PI * f) / fs;
    const env = Math.exp(-t / 0.085) * Math.min(1, t / 0.0015);
    const click = t < 0.004 ? rnd() * 14 * Math.exp(-t / 0.0015) : 0; // ビーターのアタック感
    w[i] = DPCM_START_LEVEL + 46 * env * Math.sin(ph) + click;
  }
  return w;
}

function synthSnare(fs) {
  const n = Math.round(fs * 0.17), w = new Float32Array(n), rnd = rng(12345);
  let ph = 0, lp = 0;
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    ph += (2 * Math.PI * (210 - 60 * Math.min(1, t / 0.05))) / fs;
    lp += 0.3 * (rnd() - lp); // 高域を落としたノイズ (DPCM は ±2 ずつしか動けないので、速すぎる変化は再現できない)
    const body = Math.sin(ph) * Math.exp(-t / 0.03) * 34;
    const noise = lp * Math.exp(-t / 0.05) * 75;
    w[i] = DPCM_START_LEVEL + (body + noise) * Math.min(1, t / 0.001);
  }
  return w;
}

// 16KB ($C000-$FFFF) のバンクを作る。戻り値: { bank, samples: { kick: { reg12, reg13, bytes }, ... } }
export function buildDrumBank(rate = DPCM_RATE) {
  const fs = CPU_CLOCK / DMC_RATES[rate];
  const bank = new Uint8Array(0x4000);
  const samples = {};
  let off = 0;
  for (const [name, wave] of [['kick', synthKick(fs)], ['snare', synthSnare(fs)]]) {
    const data = encodeDPCM(wave);
    bank.set(data, off);
    samples[name] = { reg12: off / 64, reg13: (data.length - 1) / 16, bytes: data.length };
    off = Math.ceil((off + data.length) / 64) * 64;
  }
  return { bank, samples, used: off, rate };
}
