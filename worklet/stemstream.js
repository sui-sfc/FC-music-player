// 音源分離 (Demucs) + AI 解析の「流しながら処理」部分 (ブラウザの Worker と Node のシミュレーターで共通)
//
// 時間の流れ (44.1kHz):  …─[左の文脈]─[今回出す区間 hop]─[右の余白 margin]─| ← 受け取り済みの末尾
//   Demucs の窓 (7.8 秒) の右端を受け取り済みの末尾に合わせ、余白を残した区間だけを使う (窓の端は精度が落ちるため)
//
// 使い方:
//   const s = new StemStreamer({ sr, hop, margin, sep, rmvpe, bp })
//     sep(l, r)  → [drums, bass, other, vocals][ch] (各 Float32Array(DEMUCS_L), 44.1kHz)
//     rmvpe(x16) → { f0, conf } (16kHz, 10ms ごと)
//     bp(x22)    → { note, onset, frames } (22.05kHz, basicPitchFull の結果)
//   s.push(L, R)          入力 (sr Hz) を足す
//   while (s.ready()) msg = await s.step()   分離 + 解析した 1 区間 (Engine._onStems に渡す形)
import { DEMUCS_SR, DEMUCS_L, StreamResampler, resample } from './demucs.js';
import { BP_SR, BP_HOP } from './basicpitch.js';

export class StemStreamer {
  constructor({ sr, hop = 1, margin = 1, ctx = 1.5, xfade = 0.05, sep, rmvpe, bp }) {
    this.sr = sr;
    this.hop = Math.round(hop * DEMUCS_SR);
    this.margin = Math.round(margin * DEMUCS_SR);
    this.ctx = Math.round(ctx * DEMUCS_SR);
    this.xfade = Math.round(xfade * DEMUCS_SR);
    this.sep = sep; this.rmvpe = rmvpe; this.bp = bp;
    this.RING = DEMUCS_SR * 40;
    this.ringL = new Float32Array(this.RING); this.ringR = new Float32Array(this.RING);
    this.rsL = new StreamResampler(sr, DEMUCS_SR); this.rsR = new StreamResampler(sr, DEMUCS_SR);
    this.n44 = 0; // 受け取った音声 (44.1kHz 換算) のサンプル数
    this.next = 0; // 次に出す区間の先頭 (44.1kHz)
    this.prevTail = null;
    this.skipBehind = true; // 処理が追いつかないとき古い区間を飛ばす (シミュレーターでは false)
  }

  get delaySec() { return (this.hop + this.margin) / DEMUCS_SR; }

  push(L, R) {
    const a = this.rsL.process(L), b = this.rsR.process(R || L);
    for (let i = 0; i < a.length; i++) { const j = (this.n44 + i) % this.RING; this.ringL[j] = a[i]; this.ringR[j] = b[i] ?? a[i]; }
    this.n44 += a.length;
  }

  ready() { return this.n44 >= this.next + this.hop + this.margin; }

  _take(ring, s, n) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) { const k = s + i; if (k >= 0 && k > this.n44 - this.RING) out[i] = ring[k % this.RING]; }
    return out;
  }

  async step() {
    const { hop: n, margin, xfade } = this;
    if (this.skipBehind && this.n44 - (this.next + n + margin) > 2 * n) this.next = this.n44 - n - margin;
    const next = this.next;
    const end = next + n + margin, ws = end - DEMUCS_L;
    const st = await this.sep(this._take(this.ringL, ws, DEMUCS_L), this._take(this.ringR, ws, DEMUCS_L));
    const off = next - ws; // 窓の中での今回の区間の位置
    const seg = (s, c, a, len) => st[s][c].subarray(off + a, off + a + len);
    const mono = (s, a, len) => { const x = new Float32Array(len), p = seg(s, 0, a, len), q = seg(s, 1, a, len); for (let i = 0; i < len; i++) x[i] = (p[i] + q[i]) / 2; return x; };
    // 区間を取り出す。前回の余白部分と短く重ねてつなぎ目を滑らかに
    const vL = seg(3, 0, 0, n).slice(), vR = seg(3, 1, 0, n).slice(), d = mono(0, 0, n);
    if (this.prevTail) {
      const p = this.prevTail;
      for (let i = 0; i < xfade; i++) {
        const w = i / xfade;
        vL[i] = p.vL[i] * (1 - w) + vL[i] * w; vR[i] = p.vR[i] * (1 - w) + vR[i] * w; d[i] = p.d[i] * (1 - w) + d[i] * w;
      }
    }
    this.prevTail = { vL: seg(3, 0, n, xfade).slice(), vR: seg(3, 1, n, xfade).slice(), d: mono(0, n, xfade) };

    // AI 解析: 区間の前を文脈として足し、右の余白も含めて推論して、区間から先を使う (余白の部分は次の区間で上書きされる)
    const a0 = -Math.min(this.ctx, off), len = n + margin - a0;
    const t44 = next + a0;
    const { f0, conf } = await this.rmvpe(resample(mono(3, a0, len), DEMUCS_SR, 16000));
    const bo = await this.bp(resample(mono(2, a0, len), DEMUCS_SR, BP_SR));
    const bb = await this.bp(resample(mono(1, a0, len), DEMUCS_SR, BP_SR));
    const bassF0 = await this.rmvpe(resample(mono(1, a0, len), DEMUCS_SR, 16000)); // ベースの単音推定

    const toSr = (x) => resample(x, DEMUCS_SR, this.sr);
    const from = -a0 / DEMUCS_SR;
    const msg = {
      start: Math.round((next * this.sr) / DEMUCS_SR),
      vL: toSr(vL), vR: toSr(vR), d: toSr(d),
      rm: { t0: t44 / DEMUCS_SR, step: 0.01, f0, conf, from },
      bf: { t0: t44 / DEMUCS_SR, step: 0.01, f0: bassF0.f0, conf: bassF0.conf, from },
      bp: { t0: t44 / DEMUCS_SR, step: BP_HOP / BP_SR, frames: bo.frames, note: bo.note, onset: bo.onset, bass: bb.note, bassOn: bb.onset, from },
    };
    this.next += n;
    return msg;
  }
}
