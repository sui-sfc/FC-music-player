// 編曲の判断 (「変換」ではなく「アレンジ」に近づけるための部品)
//   TempoTracker:  ドラムの立ち上がりからテンポと拍の位置を推定 → アルペジオを拍に合わせる
//   EnergyTracker: 曲の盛り上がり (静か / 普通 / サビ) を判定 → 編成を変える
//   arpTone:       アルペジオの音型 (上行・下行・往復・オクターブ付き) と速さ (16 分・32 分・毎フレーム)

const FPS = 60.0988;

// ---------------- テンポ ----------------
// 立ち上がりの強さの列を自己相関してテンポ (1 拍のフレーム数) を求め、
// その周期で櫛形に足し合わせて拍の位置 (位相) を求める。120 BPM 付近をやや優先 (倍・半分の取り違え防止)
export class TempoTracker {
  constructor() {
    this.N = 512; // 約 8.5 秒
    this.buf = new Float32Array(this.N);
    this.n = 0; // 入れたフレーム数 (= 解析フレームの番号)
    this.period = 0; // 1 拍のフレーム数 (0 = 不明)
    this.refFrame = 0; // 基準となる拍のフレーム位置
    this.refBeat = 0; // refFrame での累積拍番号 (連続・単調増加)
    this.hasBeat = false;
    this.conf = 0;
  }
  push(o) {
    this.buf[this.n % this.N] = Math.min(o, 8);
    this.n++;
    if (this.n >= 240 && this.n % 15 === 0) this._estimate();
  }
  _at(i) { return this.buf[((i % this.N) + this.N) % this.N]; }
  _estimate() {
    const M = Math.min(this.n, 480), end = this.n;
    let mean = 0;
    for (let i = end - M; i < end; i++) mean += this._at(i);
    mean /= M;
    const ac = new Float64Array(150);
    for (let L = 0; L < 150; L++) {
      let s = 0;
      for (let i = end - M + L; i < end; i++) s += (this._at(i) - mean) * (this._at(i - L) - mean);
      ac[L] = s / (M - L);
    }
    let best = 0, bv = 0;
    for (let L = 18; L <= 72; L++) { // 50〜200 BPM
      const prior = Math.exp(-0.5 * (Math.log2(L / 30) / 0.7) ** 2);
      const v = (ac[L] + 0.5 * ac[Math.min(149, 2 * L)]) * prior;
      if (v > bv) { bv = v; best = L; }
    }
    if (!best || ac[0] <= 0) { this.conf = 0; return; }
    // 放物線補間で小数のフレーム数に
    const a = ac[best - 1], b = ac[best], c = ac[best + 1], den = a - 2 * b + c;
    const L = best + (den < 0 ? (0.5 * (a - c)) / den : 0);
    this.conf = Math.max(0, b / ac[0]);
    // 前の推定に近ければなめらかに追従。倍・半分になっただけなら前のテンポのまま (拍の取り方が揺れないように)。
    // それ以外に大きく違うときは、同じ結果が 3 回続いたら切り替える
    const oldPeriod = this.period;
    const near = (x, y) => Math.abs(x - y) / y < 0.06;
    if (this.period && near(L, this.period)) this.period = this.period * 0.8 + L * 0.2;
    else if (this.period && (near(L, 2 * this.period) || near(L, this.period / 2))) { /* そのまま */ }
    else if (this.period && !(this.pend && near(L, this.pend) && ++this.pendN >= 3)) { if (!this.pend || !near(L, this.pend)) { this.pend = L; this.pendN = 1; } }
    else { this.period = L; this.pend = 0; this.pendN = 0; }
    // 位相: 直近の拍の位置を、周期ごとの立ち上がりの合計が最大になる所に
    const P = this.period;
    let bp = 0, bs = -1;
    for (let ph = 0; ph < Math.ceil(P); ph++) {
      let s = 0;
      for (let k = 0; k < 8; k++) { const i = Math.round(end - 1 - ph - k * P); if (i < end - this.N) break; s += this._at(i) * (1 - k * 0.08); }
      if (s > bs) { bs = s; bp = ph; }
    }
    const target = end - 1 - bp;
    // 基準拍が毎回 1 拍ぶん飛んでベースやアルペジオが途切れないよう、連続な累積拍番号で緩やかに位相追従する
    if (!this.hasBeat) {
      this.refFrame = target;
      this.refBeat = 0;
      this.hasBeat = true;
    } else {
      const approxBeat = this.refBeat + (target - this.refFrame) / (oldPeriod || P);
      const nearestBeat = Math.round(approxBeat);
      const predicted = this.refFrame + (nearestBeat - this.refBeat) * P;
      let err = target - predicted;
      err -= Math.round(err / P) * P;
      this.refFrame = predicted + err * 0.25;
      this.refBeat = nearestBeat;
    }
  }
  get bpm() { return this.period ? (FPS * 60) / this.period : 0; }
  // フレーム番号 f が何拍目か (小数)。テンポ不明なら null
  beatAt(f) { return this.period && this.hasBeat && this.conf > 0.05 ? this.refBeat + (f - this.refFrame) / this.period : null; }
}

// ---------------- 盛り上がり ----------------
// 最近の J-POP などは音圧がそろっていて、全体の音量ではサビと A メロの差がほとんど出ない (Get Wild で 0〜3dB)。
// そこで「歌の大きさ」(サビは声を張る) と「ドラムが鳴っているか」で判定する。どれも約 3 秒 (過去 2 秒 + 先読み 1 秒) の平均。
//   静か (0): 全体が小さい、またはドラムがほとんど無い
//   サビ (2): 歌が、これまでで一番大きい所に近い (聴き始めて 30 秒たってから)
//   普通 (1): それ以外
// 行ったり来たりしないよう、同じ判定が 2 秒続いたら切り替える
export class EnergyTracker {
  constructor() {
    this.H = 120;
    this.ms = new Float64Array(this.H); this.ve = new Float64Array(this.H); this.hit = new Uint8Array(180);
    this.n = 0;
    this.refMix = 1e-12; this.refVoc = 1e-12;
    this.level = 1; this.cand = 1; this.candAge = 0;
    this.db = 0; this.vdb = 0; this.hits = 0;
  }
  // r: 今のフレーム (rms, ve = 歌の強さ, hit = ドラム), future: 先読み区間のフレーム
  update(r, future, P) {
    const i = this.n % this.H;
    this.ms[i] = r.rms * r.rms; this.ve[i] = r.ve || 0; this.hit[this.n % 180] = r.hit ? 1 : 0;
    this.n++;
    let sm = 0, sv = 0, k = 0, hits = 0;
    for (let j = 0; j < Math.min(this.n, this.H); j++) { sm += this.ms[j]; sv += this.ve[j]; k++; }
    for (let j = 0; j < Math.min(this.n, 180); j++) hits += this.hit[j];
    for (const f of future) { sm += f.rms * f.rms; sv += f.ve || 0; k++; if (f.hit) hits++; }
    const em = sm / k, ev = sv / k;
    this.refMix = Math.max(em, this.refMix * 0.99985); // 約 75 秒で 1/e
    this.refVoc = Math.max(ev, this.refVoc * 0.99985);
    this.db = 10 * Math.log10((em + 1e-12) / this.refMix);
    this.vdb = 10 * Math.log10((ev + 1e-12) / this.refVoc);
    this.hits = hits;
    let want = 1;
    if (this.db < P.quietDb || hits < 3) want = 0;
    else if (this.n > 720 && this.vdb > P.loudDb) want = 2;
    if (want === this.level) { this.cand = want; this.candAge = 0; }
    // サビから抜けるのは 4 秒、それ以外は 1.5 秒続いたら切り替える (サビの中の息継ぎで抜けないように)
    else if (want === this.cand) { if (++this.candAge > (this.level === 2 ? 240 : 90)) { this.level = want; this.candAge = 0; } }
    else { this.cand = want; this.candAge = 0; }
    return this.level;
  }
}

// ---------------- 調 (キー / スケール) ----------------
// クロマ履歴を Krumhansl-Schmuckler プロファイルと照合して長調/短調 24 種を推定する。
// 音程・コード認識に穏やかなスケール内バイアスをかけ、MIDI 打ち込みらしい破綻の少ない音選びにする
const KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const KS_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KS_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
function normProfile(p) {
  const m = p.reduce((a, b) => a + b, 0) / 12;
  const d = p.map((v) => v - m);
  const s = Math.hypot(...d) || 1;
  return d.map((v) => v / s);
}
const PROF_MAJ = normProfile(KS_MAJ);
const PROF_MIN = normProfile(KS_MIN);
const SCALE_MAJ = new Set([0, 2, 4, 5, 7, 9, 11]);
const SCALE_MIN = new Set([0, 2, 3, 5, 7, 8, 10, 11]); // 導音 (+11) も許容
// 調ごとのダイアトニックコード (相対ルートと種別 0=maj, 1=min, 2=5)
const DIA_MAJ = [
  [0, 0], [2, 1], [4, 1], [5, 0], [7, 0], [9, 1],
  [0, 2], [2, 2], [4, 2], [5, 2], [7, 2], [9, 2],
];
const DIA_MIN = [
  [0, 1], [3, 0], [5, 1], [7, 1], [7, 0], [8, 0], [10, 0],
  [0, 2], [3, 2], [5, 2], [7, 2], [8, 2], [10, 2],
];

export class KeyTracker {
  constructor() {
    this.hist = new Float64Array(12);
    this.n = 0;
    this.root = -1; // 0..11 (-1 = 不明)
    this.minor = false;
    this.conf = 0; // 0..1
    this.cand = -1;
    this.candAge = 0;
  }
  push(chroma, total) {
    if (!chroma || total <= 0.1) return;
    // 約 20 秒の指数減衰 (セクション内は安定しつつ、ラスサビの転調にも追従できる)
    const decay = 0.9992;
    for (let i = 0; i < 12; i++) this.hist[i] = this.hist[i] * decay + chroma[i] / total;
    this.n++;
    if (this.n >= 90 && this.n % 15 === 0) this._estimate();
  }
  _estimate() {
    let mean = 0;
    for (let i = 0; i < 12; i++) mean += this.hist[i];
    mean /= 12;
    let norm = 0;
    const h = new Float64Array(12);
    for (let i = 0; i < 12; i++) { h[i] = this.hist[i] - mean; norm += h[i] * h[i]; }
    norm = Math.sqrt(norm);
    if (norm < 1e-6) return;
    let best = -1, bestS = -Infinity, secondS = -Infinity;
    for (let k = 0; k < 24; k++) {
      const r = k % 12, prof = k < 12 ? PROF_MAJ : PROF_MIN;
      let dot = 0;
      for (let i = 0; i < 12; i++) dot += h[(r + i) % 12] * prof[i];
      const sc = dot / norm;
      if (sc > bestS) { secondS = bestS; bestS = sc; best = k; }
      else if (sc > secondS) secondS = sc;
    }
    const c = Math.max(0, Math.min(1, (bestS - 0.35) / 0.45));
    if (c < 0.15) return;
    const cur = this.root < 0 ? -1 : this.root + (this.minor ? 12 : 0);
    if (cur < 0 || best === cur) {
      this.root = best % 12; this.minor = best >= 12; this.conf = c; this.cand = best; this.candAge = 0;
    } else if (best === this.cand) {
      if (++this.candAge >= 8) { this.root = best % 12; this.minor = best >= 12; this.conf = c; this.candAge = 0; }
    } else { this.cand = best; this.candAge = 1; }
  }
  get keyName() { return this.root < 0 ? '--' : KEY_NAMES[this.root] + (this.minor ? 'm' : ''); }
  // MIDI ノートが推定スケール内かどうかの穏やかな対数スコア補正
  noteBias(midi) {
    if (this.root < 0 || this.conf < 0.2) return 0;
    const pc = ((Math.round(midi) - this.root) % 12 + 12) % 12;
    const sc = this.minor ? SCALE_MIN : SCALE_MAJ;
    if (pc === 0 || pc === 7) return 0.35 * this.conf;
    return (sc.has(pc) ? 0.2 : -0.4) * this.conf;
  }
  // コード ID (0..35) がダイアトニックコードかどうかのスコア補正
  chordBias(chordId) {
    if (this.root < 0 || this.conf < 0.2 || chordId < 0 || chordId >= 36) return 0;
    const type = Math.floor(chordId / 12), r = ((chordId % 12) - this.root + 12) % 12;
    const list = this.minor ? DIA_MIN : DIA_MAJ;
    for (const [dr, dt] of list) if (dr === r && dt === type) return 0.65 * this.conf;
    return -0.25 * this.conf;
  }
}

// ---------------- アルペジオ ----------------
export const ARP_PATTERNS = {
  up: [0, 1, 2],
  down: [2, 1, 0],
  updown: [0, 1, 2, 1], // 1-3-5-3
  up8: [0, 1, 2, 3], // 1-3-5-8
  '1535': [0, 2, 1, 2], // 1-5-3-5 (MIDI 伴奏定番のアルベルティ型)
};
// tones: 構成音 (低い順, 3 音以上), beat: 今が何拍目か (null = テンポ不明), frame: フレーム番号
// rate: 'frame' (arpSpeed フレームごと) / '8' / '16' / '32'
// 戻り値: { note, age (この音になってからの割合 0..1) }
export function arpTone(tones, pattern, rate, beat, frame, arpSpeed) {
  const pat = ARP_PATTERNS[pattern] || ARP_PATTERNS.up;
  const t4 = tones.length >= 4 ? tones : [...tones, tones[0] + 12];
  let step, age;
  rate = String(rate);
  if (rate === 'frame' || beat === null) {
    step = Math.floor(frame / arpSpeed); age = 0;
  } else {
    const div = rate === '32' ? 8 : rate === '8' ? 2 : 4;
    const x = beat * div;
    step = Math.floor(x); age = x - step;
  }
  const i = pat[((step % pat.length) + pat.length) % pat.length];
  return { note: t4[Math.min(i, t4.length - 1)], age, step };
}
