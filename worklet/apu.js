// ============================================================
//  2A03 APU エミュレーター (NTSC ファミコン)
//  ---------------------------------------------------------
//  実機と同じレジスタ ($4000-$4017) を受け取り、音を合成する。
//  ドライバ (driver.js) はこのクラスに write(addr, value) するだけなので、
//  将来ファミコン実機に移植するときは、このファイルが「本物の APU」に
//  置き換わるだけになる。
//
//  簡略化している点:
//   - エンベロープ / 長さカウンタ / スイープの自動変化は未実装
//     (ドライバは常に「定音量モード + 長さカウンタ停止」で使う)
//   - DMC の IRQ は未実装 (再生・ループ・$4011 直接書き込みは実装済み)
// ============================================================

export const CPU_CLOCK = 1789773; // NTSC CPU クロック (Hz)
export const FRAME_RATE = 60.0988; // NTSC フレームレート

const DUTY = [
  [0, 1, 0, 0, 0, 0, 0, 0], // 12.5%
  [0, 1, 1, 0, 0, 0, 0, 0], // 25%
  [0, 1, 1, 1, 1, 0, 0, 0], // 50%
  [1, 0, 0, 1, 1, 1, 1, 1], // 75% (25% の反転)
];

const TRI_SEQ = new Uint8Array(32);
for (let i = 0; i < 16; i++) {
  TRI_SEQ[i] = 15 - i;
  TRI_SEQ[16 + i] = i;
}

// DMC の 1 ビットあたりの CPU サイクル (NTSC)。index 15 = 約 33.1kHz
export const DMC_RATES = [428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 84, 72, 54];

// ノイズのタイマー周期 (CPU サイクル, NTSC)
export const NOISE_PERIODS = [
  4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068,
];

function makePulse() {
  return { duty: 0, vol: 0, constVol: 1, negate: 0, shift: 0, timer: 0, seq: 0, rem: 1 };
}

export class APU {
  constructor(sampleRate) {
    this.sampleRate = sampleRate;
    this.cps = CPU_CLOCK / sampleRate; // 1 サンプルあたりの CPU サイクル
    this.pulse = [makePulse(), makePulse()];
    this.tri = { timer: 0, seq: 0, rem: 1, linear: 0 };
    this.noise = { vol: 0, mode: 0, idx: 0, lfsr: 1, rem: 4 };
    this.enable = 0;
    // DMC: カートリッジの $C000-$FFFF にあるサンプル (1 ビットデルタ) を再生する
    this.dmc = { rate: 0, loop: 0, level: 0, addr: 0xc000, len: 1, cur: 0xc000, remain: 0, buf: -1, shift: 0, bits: 8, silence: true, rem: 428 };
    this.mem = new Uint8Array(0x4000); // $C000-$FFFF (DPCM サンプル置き場)

    // 出力フィルタ (ファミコン本体のアナログ回路を近似)
    const dt = 1 / sampleRate;
    const hpRC = 1 / (2 * Math.PI * 37); // ハイパス 37Hz (DC カット)
    const lpRC = 1 / (2 * Math.PI * 14000); // ローパス 14kHz
    this.hpA = hpRC / (hpRC + dt);
    this.lpB = dt / (lpRC + dt);
    this.hpX = 0;
    this.hpY = 0;
    this.lp = 0;
  }

  write(addr, v) {
    v &= 0xff;
    switch (addr) {
      case 0x4000: case 0x4004: {
        const p = this.pulse[(addr >> 2) & 1];
        p.duty = v >> 6;
        p.constVol = (v >> 4) & 1;
        p.vol = v & 15;
        break;
      }
      case 0x4001: case 0x4005: {
        const p = this.pulse[(addr >> 2) & 1];
        p.negate = (v >> 3) & 1;
        p.shift = v & 7;
        break;
      }
      case 0x4002: case 0x4006: {
        const p = this.pulse[(addr >> 2) & 1];
        p.timer = (p.timer & 0x700) | v;
        break;
      }
      case 0x4003: case 0x4007: {
        const p = this.pulse[(addr >> 2) & 1];
        p.timer = (p.timer & 0xff) | ((v & 7) << 8);
        p.seq = 0; // 実機同様、$4003 書き込みで位相リセット
        break;
      }
      case 0x4008:
        // 制御フラグ=1 前提: 線形カウンタは常にリロード値に保たれる
        this.tri.linear = v & 0x7f;
        break;
      case 0x400a:
        this.tri.timer = (this.tri.timer & 0x700) | v;
        break;
      case 0x400b:
        this.tri.timer = (this.tri.timer & 0xff) | ((v & 7) << 8);
        break;
      case 0x400c:
        this.noise.vol = v & 15;
        break;
      case 0x400e:
        this.noise.mode = v >> 7;
        this.noise.idx = v & 15;
        break;
      case 0x4010:
        this.dmc.rate = v & 15;
        this.dmc.loop = (v >> 6) & 1;
        break;
      case 0x4011:
        this.dmc.level = v & 0x7f; // DAC に直接書き込み
        break;
      case 0x4012:
        this.dmc.addr = 0xc000 + v * 64;
        break;
      case 0x4013:
        this.dmc.len = v * 16 + 1;
        break;
      case 0x4015: {
        this.enable = v & 0x1f;
        const d = this.dmc;
        if (!(v & 0x10)) d.remain = 0; // 停止
        else if (d.remain === 0) { d.cur = d.addr; d.remain = d.len; this._dmcFetch(); } // (再) 開始
        break;
      }
      default:
        break;
    }
  }

  // --- 各チャンネルを 1 サンプル分 (cps サイクル) 進め、平均レベルを返す ---
  // 区間積分 (box filter) なので折り返しノイズがかなり抑えられる

  _pulse(p, bit) {
    const c = this.cps;
    const period = 2 * (p.timer + 1); // パルスのタイマーは APU クロック (CPU/2)
    const muted =
      !(this.enable & bit) ||
      p.timer < 8 ||
      (!p.negate && p.timer + (p.timer >> p.shift) > 0x7ff) ||
      p.vol === 0;
    if (muted) {
      p.rem -= c;
      while (p.rem <= 0) { p.rem += period; p.seq = (p.seq + 1) & 7; }
      return 0;
    }
    const seq = DUTY[p.duty];
    let acc = 0, left = c;
    while (left > 0) {
      const d = left < p.rem ? left : p.rem;
      acc += seq[p.seq] * d;
      left -= d;
      p.rem -= d;
      if (p.rem <= 0) { p.rem += period; p.seq = (p.seq + 1) & 7; }
    }
    return (acc * p.vol) / c;
  }

  _triangle() {
    const t = this.tri;
    const c = this.cps;
    // 線形カウンタ 0 のときはシーケンサが止まり、出力は直前の値を保持する (実機の挙動)
    if (!(this.enable & 4) || t.linear === 0 || t.timer < 2) return TRI_SEQ[t.seq];
    const period = t.timer + 1;
    let acc = 0, left = c;
    while (left > 0) {
      const d = left < t.rem ? left : t.rem;
      acc += TRI_SEQ[t.seq] * d;
      left -= d;
      t.rem -= d;
      if (t.rem <= 0) { t.rem += period; t.seq = (t.seq + 1) & 31; }
    }
    return acc / c;
  }

  _noise() {
    const n = this.noise;
    const c = this.cps;
    const period = NOISE_PERIODS[n.idx];
    const on = (this.enable & 8) && n.vol > 0;
    let acc = 0, left = c;
    const tap = n.mode ? 6 : 1;
    while (left > 0) {
      const d = left < n.rem ? left : n.rem;
      if (on && !(n.lfsr & 1)) acc += d;
      left -= d;
      n.rem -= d;
      if (n.rem <= 0) {
        n.rem += period;
        const fb = (n.lfsr ^ (n.lfsr >> tap)) & 1;
        n.lfsr = (n.lfsr >> 1) | (fb << 14);
      }
    }
    return on ? (acc * n.vol) / c : 0;
  }

  // サンプルバッファが空なら次の 1 バイトを読む (実機では CPU を止めて DMA で読む)
  _dmcFetch() {
    const d = this.dmc;
    if (d.buf >= 0 || d.remain === 0) return;
    d.buf = d.cur >= 0xc000 ? this.mem[d.cur - 0xc000] : 0;
    d.cur = d.cur === 0xffff ? 0x8000 : d.cur + 1;
    if (--d.remain === 0 && d.loop) { d.cur = d.addr; d.remain = d.len; }
  }

  _dmcStep() {
    const d = this.dmc;
    if (!d.silence) {
      if (d.shift & 1) { if (d.level <= 125) d.level += 2; } else if (d.level >= 2) d.level -= 2;
      d.shift >>= 1;
    }
    if (--d.bits === 0) {
      d.bits = 8;
      if (d.buf < 0) d.silence = true;
      else { d.silence = false; d.shift = d.buf; d.buf = -1; this._dmcFetch(); }
    }
  }

  _dmcOut() {
    const d = this.dmc, c = this.cps, period = DMC_RATES[d.rate];
    let acc = 0, left = c;
    while (left > 0) {
      const t = left < d.rem ? left : d.rem;
      acc += d.level * t;
      left -= t;
      d.rem -= t;
      if (d.rem <= 0) { d.rem += period; this._dmcStep(); }
    }
    return acc / c;
  }

  sample() {
    const p1 = this._pulse(this.pulse[0], 1);
    const p2 = this._pulse(this.pulse[1], 2);
    const t = this._triangle();
    const n = this._noise();
    const dm = this._dmcOut();

    // 実機の非線形ミキサー (nesdev wiki の近似式)
    const ps = p1 + p2;
    const pout = ps > 0 ? 95.88 / (8128 / ps + 100) : 0;
    // DMC の出力は三角波・ノイズと同じ非線形ミキサーに入る (DMC が大きいと三角波・ノイズが少し小さくなる = 実機の癖)
    const tnd = t / 8227 + n / 12241 + dm / 22638;
    const tout = tnd > 0 ? 159.79 / (1 / tnd + 100) : 0;
    const x = pout + tout;

    const y = this.hpA * (this.hpY + x - this.hpX);
    this.hpX = x;
    this.hpY = y;
    this.lp += this.lpB * (y - this.lp);
    // 参考アレンジに近い音量に上げ、まれな山はソフトクリップで丸める (0.8 を超えた分だけゆるやかに圧縮)
    let y2 = this.lp * 2.8;
    const ay = Math.abs(y2);
    if (ay > 0.8) y2 = Math.sign(y2) * (0.8 + 0.2 * Math.tanh((ay - 0.8) / 0.2));
    return y2;
  }
}
