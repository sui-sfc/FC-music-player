// 変換エンジン本体 (ブラウザの AudioWorklet と Node の開発ツールで共通)
//   入力 L/R サンプル → 解析 (driver) → APU レジスタ → APU 合成 → 出力サンプル
import { APU, FRAME_RATE } from './apu.js';
import { Driver } from './driver.js';
import { FFT } from './fft.js';
import { BP_SR, BP_WIN, BP_HOP, BP_FRAMES, BP_NOTES, BP_EDGE } from './basicpitch.js';

function biquad(type, f, q, sr) {
  const w = (2 * Math.PI * f) / sr, cs = Math.cos(w), al = Math.sin(w) / (2 * q);
  let b0, b1, b2;
  if (type === 'lp') { b0 = (1 - cs) / 2; b1 = 1 - cs; b2 = b0; }
  else if (type === 'hp') { b0 = (1 + cs) / 2; b1 = -(1 + cs); b2 = b0; }
  else { b0 = al; b1 = 0; b2 = -al; } // bandpass
  const a0 = 1 + al;
  return { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: (-2 * cs) / a0, a2: (1 - al) / a0, x1: 0, x2: 0, y1: 0, y2: 0 };
}
function runBq(f, x) {
  const y = f.b0 * x + f.b1 * f.x1 + f.b2 * f.x2 - f.a1 * f.y1 - f.a2 * f.y2;
  f.x2 = f.x1; f.x1 = x; f.y2 = f.y1; f.y1 = y;
  return y;
}

// ---- サンプリングレート変換 (AI モデル用) ----
// 4 次ローパスで折り返しを防いでから線形補間で間引く。count = 変換後の通し番号
function makeResampler(rate, sr, cutoff) {
  return { rate, step: rate / sr, ph: 0, prev: 0, count: 0, ring: new Float32Array(rate * 8),
    f: [biquad('lp', cutoff, 0.54, sr), biquad('lp', cutoff, 1.31, sr)] };
}
function feedResampler(r, x) {
  const y = runBq(r.f[1], runBq(r.f[0], x));
  let ph = r.ph + r.step;
  while (ph >= 1) {
    ph -= 1;
    const frac = 1 - ph / r.step;
    r.ring[r.count % r.ring.length] = r.prev + (y - r.prev) * frac;
    r.count++;
  }
  r.ph = ph;
  r.prev = y;
}
function readRing(r, n) {
  const out = new Float32Array(n), L = r.ring.length, s0 = r.count - n;
  for (let i = 0; i < n; i++) out[i] = r.ring[(s0 + i) % L];
  return out;
}

export class Engine {
  constructor(sr, onFrame) {
    this.onFrame = onFrame;
    this.sr = sr;
    this.stemPort = null; // 音源分離 (Demucs) の Worker への MessagePort
    this.N = sr > 60000 ? 16384 : 8192;
    this.ringL = new Float32Array(this.N);
    this.ringR = new Float32Array(this.N);
    this.ringMix = new Float32Array(this.N); // 原曲モノラル (解析窓の中心で RMS を測るため)
    // 音源分離 (Demucs) の結果を使うときの入力: 歌 L/R・ドラム (stems = true のとき)
    this.stems = false;
    this.ringVL = new Float32Array(this.N);
    this.ringVR = new Float32Array(this.N);
    this.ringD = new Float32Array(this.N);
    this.pos = 0;
    this.re = new Float64Array(this.N);
    this.im = new Float64Array(this.N);
    this.amp = new Float32Array(this.N / 2);
    // ボーカル用の短い窓 (ビブラートでにじまないように)
    this.NV = this.N / 2;
    this.reV = new Float64Array(this.NV);
    this.imV = new Float64Array(this.NV);
    this.ampV = new Float32Array(this.NV / 2);
    this.ampVmix = new Float32Array(this.NV / 2); // 短窓の全体スペクトル (ドラム検出用)
    this.winV = new Float32Array(this.NV);
    for (let i = 0; i < this.NV; i++) this.winV[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (this.NV - 1));
    this.fftV = new FFT(this.NV);
    this.win = new Float32Array(this.N);
    for (let i = 0; i < this.N; i++) this.win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (this.N - 1));
    this.fft = new FFT(this.N);

    this.apu = new APU(sr);
    this.driver = new Driver(sr, this.N, this.NV, (a, v) => this.apu.write(a, v));
    this.apu.mem.set(this.driver.dpcm.bank); // DPCM サンプルを $C000- に置く (実機ではカートリッジ側)

    this.frameLen = sr / FRAME_RATE;
    this.frameAcc = 0;
    this.bq = [biquad('lp', 120, 0.7, sr), biquad('bp', 2500, 0.8, sr), biquad('hp', 7000, 0.7, sr)];
    this.bandE = [0, 0, 0];
    this.sq = 0;
    this.cnt = 0;
    this.inCount = 0; // 入力サンプル数 (時刻の基準)

    // ---- AI 解析 (Worker で動く機械学習モデル) への入力 ----
    //   16kHz: RMVPE (歌の音程), 22050Hz: Basic Pitch (全ての音符)
    this.rs = [makeResampler(16000, sr, 7000), makeResampler(BP_SR, sr, 10000)];
    this.aiPort = null; // 推論側 (Worker) への MessagePort
    this.busy = false;
    this.lastSend = 0;
    // 推論結果のタイムライン (10ms ごと, 10 秒ぶん)
    this.extF0 = new Float32Array(1000);
    this.extConf = new Float32Array(1000);
    this.extK = new Int32Array(1000).fill(-1);
    this.bpNote = new Float32Array(1000 * BP_NOTES);
    this.bpOnset = new Float32Array(1000 * BP_NOTES);
    this.bpK = new Int32Array(1000).fill(-1);
  }

  // ================= 音源分離モード (リアルタイム) =================
  // 入力は Worker (demucs-worker.js) に送り、分離した音と解析結果が届いた区間から順に変換する。
  // 変換結果は入力の位置で outRing に置き、delay 秒遅らせて出力する (分離の待ち時間ぶん遅れる)
  setStemPort(port, delaySec = 3) {
    if (this.stemPort) this.stemPort.onmessage = null;
    this.stemPort = port;
    this.stems = false;
    if (!port) { this.driver.extLookup = null; this.driver.bpLookup = null; this.driver.bassLookup = null; return; }
    this.setAiPort(null);
    const sr = this.sr, R = sr * 30;
    this.sRing = R;
    for (const k of ['mixL', 'mixR', 'sVL', 'sVR', 'sD', 'outRing']) this[k] = new Float32Array(R);
    this.tIn = 0; this.p = 0; this.stemEnd = 0;
    this.D = Math.round(delaySec * sr);
    this.underruns = 0;
    this.batchL = new Float32Array(2048); this.batchR = new Float32Array(2048); this.batchN = 0;
    this.tmpOut = new Float32Array(1024);
    // 時刻の基準を分離側とそろえるため、解析をやり直す
    this.inCount = 0; this.frameAcc = 0; this.ringL.fill(0); this.ringR.fill(0); this.ringMix.fill(0);
    this.driver.reset();
    this.extK.fill(-1); this.bpK.fill(-1);
    const slot = (k) => ((k % 1000) + 1000) % 1000;
    this.bfF0 = new Float32Array(1000); this.bfConf = new Float32Array(1000); this.bfK = new Int32Array(1000).fill(-1);
    this.driver.bassLookup = (t) => {
      const k = Math.round(t * 100), i = slot(k);
      return k >= 0 && this.bfK[i] === k ? { f0: this.bfF0[i], conf: this.bfConf[i] } : null;
    };
    this.bpInst = new Float32Array(1000 * BP_NOTES); this.bpInstOn = new Float32Array(1000 * BP_NOTES);
    this.bpBass = new Float32Array(1000 * BP_NOTES); this.bpBassOn = new Float32Array(1000 * BP_NOTES);
    port.postMessage({ type: 'reset', sr });
    port.onmessage = (e) => this._onStems(e.data);
    this.driver.extLookup = (t) => {
      const k = Math.round(t * 100), i = slot(k);
      return k >= 0 && this.extK[i] === k ? { f0: this.extF0[i], conf: this.extConf[i] } : null;
    };
    this.driver.bpLookup = (t) => {
      const k0 = Math.round(t * 100);
      for (const k of [k0, k0 - 1, k0 + 1]) {
        const i = slot(k);
        if (k >= 0 && this.bpK[i] === k) {
          const sl = (a) => a.subarray(i * BP_NOTES, (i + 1) * BP_NOTES);
          return { note: sl(this.bpNote), onset: sl(this.bpOnset), inst: sl(this.bpInst), instOn: sl(this.bpInstOn), bass: sl(this.bpBass), bassOn: sl(this.bpBassOn) };
        }
      }
      return null;
    };
  }

  _onStems(m) {
    const R = this.sRing, n = m.vL.length;
    if (m.start > this.stemEnd && this.stemEnd >= this.p) {
      // 分離側が処理しきれず区間を飛ばした: その間は変換せずに進める
      const gap = m.start - Math.max(this.p, this.stemEnd);
      if (gap > 0 && m.start > this.p) { for (let t = this.p; t < m.start; t++) this.outRing[t % R] = 0; this.inCount += m.start - this.p; this.p = m.start; }
    }
    for (let i = 0; i < n; i++) {
      const j = (m.start + i) % R;
      this.sVL[j] = m.vL[i]; this.sVR[j] = m.vR[i]; this.sD[j] = m.d[i];
    }
    this.stemEnd = Math.max(this.stemEnd, m.start + n);
    const slot = (k) => ((k % 1000) + 1000) % 1000;
    const rm = m.rm;
    for (let j = 0; j < rm.f0.length; j++) {
      const rel = j * rm.step;
      if (rel < rm.from) continue;
      const k = Math.round((rm.t0 + rel) * 100), i = slot(k);
      this.extK[i] = k; this.extF0[i] = rm.f0[j]; this.extConf[i] = rm.conf[j];
    }
    if (m.bf) {
      for (let j = 0; j < m.bf.f0.length; j++) {
        const rel = j * m.bf.step;
        if (rel < m.bf.from) continue;
        const k = Math.round((m.bf.t0 + rel) * 100), i = slot(k);
        this.bfK[i] = k; this.bfF0[i] = m.bf.f0[j]; this.bfConf[i] = m.bf.conf[j];
      }
    }
    const bp = m.bp, NB = BP_NOTES;
    for (let g = 0; g < bp.frames; g++) {
      const rel = g * bp.step;
      if (rel < bp.from) continue;
      const k = Math.round((bp.t0 + rel) * 100), i = slot(k);
      this.bpK[i] = k;
      for (let q = 0; q < NB; q++) {
        const a = g * NB + q, b = i * NB + q;
        this.bpInst[b] = bp.note[a]; this.bpInstOn[b] = bp.onset[a]; this.bpBass[b] = bp.bass[a]; this.bpBassOn[b] = bp.bassOn[a];
        this.bpNote[b] = Math.max(bp.note[a], bp.bass[a]);
        this.bpOnset[b] = Math.max(bp.onset[a], bp.bassOn[a]);
      }
    }
  }

  // 音源分離モードの 1 ブロック: 入力を貯めて送り、届いた区間を変換し、遅らせて出力する
  processLive(inL, inR, out, n) {
    const R = this.sRing;
    for (let i = 0; i < n; i++) {
      const l = inL ? inL[i] : 0, r = inR ? inR[i] : l, j = this.tIn % R;
      this.mixL[j] = l; this.mixR[j] = r; this.tIn++;
      this.batchL[this.batchN] = l; this.batchR[this.batchN] = r;
      if (++this.batchN === this.batchL.length) {
        this.stemPort.postMessage({ type: 'mix', L: this.batchL, R: this.batchR }, [this.batchL.buffer, this.batchR.buffer]);
        this.batchL = new Float32Array(2048); this.batchR = new Float32Array(2048); this.batchN = 0;
      }
    }
    // 届いた区間を変換 (1 ブロックあたり最大 1024 サンプル = 実時間の 8 倍速で追いつく)
    let budget = 1024;
    while (budget > 0) {
      const avail = Math.min(this.stemEnd, this.tIn) - this.p;
      if (avail <= 0) break;
      const j = this.p % R, len = Math.min(avail, budget, R - j);
      this.processStems(this.mixL.subarray(j, j + len), this.mixR.subarray(j, j + len),
        { vL: this.sVL.subarray(j, j + len), vR: this.sVR.subarray(j, j + len), d: this.sD.subarray(j, j + len) }, this.tmpOut, len);
      this.outRing.set(this.tmpOut.subarray(0, len), j);
      this.p += len; budget -= len;
    }
    // 出力: delay ぶん前の入力位置の変換結果。まだ無ければ無音 (足りなくなったら遅れを 0.25 秒増やす)
    let under = false;
    for (let i = 0; i < n; i++) {
      const t = this.tIn - n + i - this.D;
      if (t >= 0 && t < this.p) out[i] = this.outRing[t % R];
      else { out[i] = 0; if (t >= 0) under = true; }
    }
    if (under && this.p > 0 && this.D < 10 * this.sr) { this.D += Math.round(0.25 * this.sr); this.underruns++; }
  }

  static WIN16 = 24000; // RMVPE の窓 1.5 秒
  static HOP16 = 4000; // 0.25 秒ごとに推論

  // 推論側と接続する (port = null で切断して内蔵推定に戻す)
  setAiPort(port) {
    if (this.aiPort) this.aiPort.onmessage = null;
    this.aiPort = port;
    this.busy = false;
    this.extK.fill(-1);
    this.bpK.fill(-1);
    // 変換後の時刻を解析側の時刻 (inCount) に合わせる
    for (const r of this.rs) { r.count = Math.floor(this.inCount * r.step); r.ring.fill(0); }
    this.lastSend = this.rs[0].count;
    if (!port) { this.driver.extLookup = null; this.driver.bpLookup = null; return; }
    port.onmessage = (e) => this._onAi(e.data);
    const slot = (k) => ((k % 1000) + 1000) % 1000;
    this.driver.extLookup = (t) => {
      const k = Math.round(t * 100), i = slot(k);
      return k >= 0 && this.extK[i] === k ? { f0: this.extF0[i], conf: this.extConf[i] } : null;
    };
    this.driver.bpLookup = (t) => {
      const k0 = Math.round(t * 100);
      for (const k of [k0, k0 - 1, k0 + 1]) {
        const i = slot(k);
        if (k >= 0 && this.bpK[i] === k) {
          return { note: this.bpNote.subarray(i * BP_NOTES, (i + 1) * BP_NOTES), onset: this.bpOnset.subarray(i * BP_NOTES, (i + 1) * BP_NOTES) };
        }
      }
      return null;
    };
  }

  _onAi({ rm, bp }) {
    this.busy = false;
    if (rm) {
      for (let j = 0; j < rm.f0.length; j++) {
        const k = Math.round((rm.start + j * 160) / 160), i = k % 1000;
        // 窓の先頭 0.25 秒は過去の文脈が足りないので、他に結果が無いときだけ使う
        if (j < 25 && this.extK[i] === k) continue;
        this.extK[i] = k; this.extF0[i] = rm.f0[j]; this.extConf[i] = rm.conf[j];
      }
    }
    if (bp) {
      for (let f = 0; f < BP_FRAMES; f++) {
        const k = Math.round(((bp.start + f * BP_HOP) / BP_SR) * 100), i = k % 1000;
        if (k < 0) continue;
        if (f < BP_EDGE && this.bpK[i] === k) continue;
        this.bpK[i] = k;
        this.bpNote.set(bp.note.subarray(f * BP_NOTES, (f + 1) * BP_NOTES), i * BP_NOTES);
        this.bpOnset.set(bp.onset.subarray(f * BP_NOTES, (f + 1) * BP_NOTES), i * BP_NOTES);
      }
    }
  }

  _feedAi(x) {
    for (const r of this.rs) feedResampler(r, x);
    const r16 = this.rs[0], r22 = this.rs[1];
    if (!this.busy && r16.count >= Engine.WIN16 && r22.count >= BP_WIN && r16.count - this.lastSend >= Engine.HOP16) {
      const a16 = readRing(r16, Engine.WIN16), a22 = readRing(r22, BP_WIN);
      this.busy = true;
      this.lastSend = r16.count;
      this.aiPort.postMessage(
        { a16, s16: r16.count - Engine.WIN16, a22, s22: r22.count - BP_WIN },
        [a16.buffer, a22.buffer],
      );
    }
  }

  // 解析窓の中心 (pos - N/2) を揃えて、全体の振幅 (mixOut) と中央成分の振幅 (centerOut) を求める
  // 長窓 (this.N) と短窓 (this.NV) の中心時刻が一致するので、ドラム・歌・コード・AI 解析・原音遅延がズレない
  _spectrum(N, fft, win, re, im, mixOut, centerOut, rL = this.ringL, rR = this.ringR) {
    const RN = this.N, off = this.pos + RN - ((RN + N) >> 1);
    // L を実部・R を虚部に入れて 1 回の FFT で両チャンネルを求める
    for (let i = 0; i < N; i++) {
      const j = (off + i) % RN;
      re[i] = rL[j] * win[i];
      im[i] = rR[j] * win[i];
    }
    fft.transform(re, im);
    const scale = 2 / N;
    for (let k = 0; k < N / 2; k++) {
      const n = (N - k) % N;
      const lr = (re[k] + re[n]) / 2, li = (im[k] - im[n]) / 2;
      const rr = (im[k] + im[n]) / 2, ri = (re[n] - re[k]) / 2;
      const mix = Math.hypot(lr + rr, li + ri) * scale;
      if (mixOut) mixOut[k] = mix;
      if (!centerOut) continue;
      // 中央成分マスク: 左右の大きさが揃っていて (balance)、位相も揃っている (coh) ほど 1
      const lm = Math.hypot(lr, li), rm = Math.hypot(rr, ri);
      const pw = lm * lm + rm * rm;
      let mask = 0;
      if (pw > 1e-20) {
        const balance = (2 * lm * rm) / pw;
        const coh = (lr * rr + li * ri) / (lm * rm + 1e-20);
        mask = coh > 0 ? balance * coh : 0;
        mask *= mask; mask *= mask; // 4 乗で鋭くする
      }
      centerOut[k] = mix * mask;
    }
  }

  runFrame() {
    this._spectrum(this.N, this.fft, this.win, this.re, this.im, this.amp, null);
    if (this.stems) {
      // 分離済み: 歌のスペクトルは歌の音だけから (中央成分の抽出は不要)、ドラム検出はドラムの音だけから
      this._spectrum(this.NV, this.fftV, this.winV, this.reV, this.imV, this.ampV, null, this.ringVL, this.ringVR);
      this._spectrum(this.NV, this.fftV, this.winV, this.reV, this.imV, this.ampVmix, null, this.ringD, this.ringD);
    } else this._spectrum(this.NV, this.fftV, this.winV, this.reV, this.imV, this.ampVmix, this.ampV);

    // RMS も解析窓の中心 (pos - N/2) の 1 フレームぶんで測り、フレーズ末尾の早切れを防ぐ
    const fl = Math.max(1, Math.round(this.frameLen));
    const RN = this.N, rOff = this.pos + RN - ((RN + fl) >> 1);
    let sq = 0;
    for (let i = 0; i < fl; i++) {
      const s = this.ringMix[(rOff + i) % RN];
      sq += s * s;
    }
    const rms = Math.sqrt(sq / fl);
    this.bandE[0] = this.bandE[1] = this.bandE[2] = 0;
    this.sq = 0;
    this.cnt = 0;

    const time = (this.inCount - this.N / 2) / this.sr;
    const info = this.driver.frame(this.amp, this.ampV, this.ampVmix, rms, time);
    info.rms = rms;
    this.onFrame(info, this);
  }

  // 音源分離済みの入力: mixL/R = 原曲, st = { vL, vR, d } (歌 L/R, ドラム モノラル)。
  // 音程の解析 (amp) はドラムを除いた音で行う
  processStems(mixL, mixR, st, out, n) {
    this.stems = true;
    const pos0 = this.pos;
    const L = new Float32Array(n), R = new Float32Array(n);
    for (let i = 0; i < n; i++) { L[i] = mixL[i] - st.d[i]; R[i] = mixR[i] - st.d[i]; }
    for (let i = 0; i < n; i++) {
      const j = (pos0 + i) % this.N;
      this.ringVL[j] = st.vL[i]; this.ringVR[j] = st.vR[i]; this.ringD[j] = st.d[i];
    }
    // rms・帯域エネルギーは原曲から (無音判定に使う)。ringL/R には後でドラム抜きを入れ直す
    this.process(mixL, mixR, out, n, L, R);
  }

  // inL/inR: 入力 (inR は null 可 = モノラル), out: 出力, n: サンプル数
  process(inL, inR, out, n, specL = null, specR = null) {
    const bq = this.bq;
    for (let i = 0; i < n; i++) {
      const l = inL ? inL[i] : 0;
      const r = inR ? inR[i] : l;
      const x = (l + r) / 2;
      this.ringL[this.pos] = specL ? specL[i] : l;
      this.ringR[this.pos] = specR ? specR[i] : r;
      this.ringMix[this.pos] = x;
      this.pos = (this.pos + 1) % this.N;
      const bl = runBq(bq[0], x), bm = runBq(bq[1], x), bh = runBq(bq[2], x);
      this.bandE[0] += bl * bl; this.bandE[1] += bm * bm; this.bandE[2] += bh * bh;
      this.sq += x * x;
      this.cnt++;
      this.inCount++;
      if (this.aiPort) this._feedAi(x);

      if (++this.frameAcc >= this.frameLen) {
        this.frameAcc -= this.frameLen;
        this.runFrame();
      }
      out[i] = this.apu.sample();
    }
  }
}
