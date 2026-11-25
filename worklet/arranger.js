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
    this.beatRef = 0; // 拍の頭だったフレーム番号
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
    this.beatRef = end - 1 - bp;
  }
  get bpm() { return this.period ? (FPS * 60) / this.period : 0; }
  // フレーム番号 f が何拍目か (小数)。テンポ不明なら null
  beatAt(f) { return this.period && this.conf > 0.05 ? (f - this.beatRef) / this.period : null; }
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
    else if (this.n > 1800 && this.vdb > P.loudDb) want = 2;
    if (want === this.level) { this.cand = want; this.candAge = 0; }
    // サビから抜けるのは 4 秒、それ以外は 2 秒続いたら切り替える (サビの中の息継ぎで抜けないように)
    else if (want === this.cand) { if (++this.candAge > (this.level === 2 ? 240 : 120)) { this.level = want; this.candAge = 0; } }
    else { this.cand = want; this.candAge = 0; }
    return this.level;
  }
}

// ---------------- アルペジオ ----------------
export const ARP_PATTERNS = {
  up: [0, 1, 2],
  down: [2, 1, 0],
  updown: [0, 1, 2, 1], // 1-3-5-3
  up8: [0, 1, 2, 3], // 1-3-5-8
};
// tones: 構成音 (低い順, 3 音以上), beat: 今が何拍目か (null = テンポ不明), frame: フレーム番号
// rate: 'frame' (arpSpeed フレームごと) / '16' / '32'
// 戻り値: { note, age (この音になってからの割合 0..1) }
export function arpTone(tones, pattern, rate, beat, frame, arpSpeed) {
  const pat = ARP_PATTERNS[pattern] || ARP_PATTERNS.up;
  const t4 = tones.length >= 4 ? tones : [...tones, tones[0] + 12];
  let step, age;
  rate = String(rate);
  if (rate === 'frame' || beat === null) {
    step = Math.floor(frame / arpSpeed); age = 0;
  } else {
    const div = rate === '32' ? 8 : 4;
    const x = beat * div;
    step = Math.floor(x); age = x - step;
  }
  const i = pat[((step % pat.length) + pat.length) % pat.length];
  return { note: t4[Math.min(i, t4.length - 1)], age, step };
}
