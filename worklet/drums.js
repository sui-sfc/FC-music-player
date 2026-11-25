// ============================================================
//  ドラム検出 (調波・打撃音分離 HPSS + 帯域ごとのオンセット)
//  ---------------------------------------------------------
//  スペクトログラム上で
//    - 時間方向に伸びる成分 (伸ばした音・和音・歌) = 調波成分
//    - 周波数方向に伸びる成分 (瞬間的な打撃) = 打撃成分
//  に分けるため、時間方向・周波数方向それぞれの中央値で比べる (Fitzgerald 2010)。
//  打撃成分だけで帯域ごとの立ち上がりを調べ、キック / スネア / ハイハットを判定する。
//  時間方向の中央値に前後のフレームが必要なので、結果は DELAY フレーム遅れて出る。
// ============================================================

const KT = 9; // 時間方向の中央値の幅 (フレーム)
const KF = 8; // 周波数方向の中央値の片側幅 (ビン)
export const DRUM_DELAY = (KT >> 1) + 1; // 中央値の遅れ + ピーク判定の 1 フレーム

function median(arr, n) {
  // 小さな配列用の挿入ソート
  for (let i = 1; i < n; i++) {
    const v = arr[i];
    let j = i - 1;
    while (j >= 0 && arr[j] > v) { arr[j + 1] = arr[j]; j--; }
    arr[j + 1] = v;
  }
  return arr[n >> 1];
}

export class DrumDetector {
  constructor(binHz, nb) {
    this.NB = Math.min(nb, Math.round(12000 / binHz));
    this.ring = Array.from({ length: KT }, () => new Float32Array(this.NB));
    this.n = 0;
    const band = (lo, hi) => [Math.max(1, Math.round(lo / binHz)), Math.min(this.NB - 1, Math.round(hi / binHz))];
    this.bands = [band(35, 150), band(180, 1200), band(1500, 5000), band(6000, 12000)]; // 低 / 胴 / 響き / 高
    this.prevE = new Float64Array(4);
    this.avg = new Float64Array(4).fill(1e-12);
    this.hist = [new Float64Array(4), new Float64Array(4)]; // 直近 2 フレームのオンセット強度
    this.tmpT = new Float32Array(KT);
    this.tmpF = new Float32Array(2 * KF + 1);
    this.perc = new Float32Array(this.NB);
    this.cool = [0, 0, 0]; // 同じ太鼓を連打と誤判定しないための休止フレーム数
  }

  // mag: 振幅スペクトル。戻り値: DRUM_DELAY フレーム前のフレームの { hit, strength } (無ければ null)
  push(mag, thr) {
    const NB = this.NB, ring = this.ring;
    ring[this.n % KT].set(mag.subarray(0, NB));
    this.n++;
    if (this.n < KT) return null;
    const center = ring[(this.n - 1 - (KT >> 1)) % KT];

    // 打撃成分 = 周波数方向の中央値が時間方向の中央値より大きい部分 (ソフトマスク)
    const perc = this.perc, tT = this.tmpT, tF = this.tmpF;
    for (let k = 1; k < NB; k++) {
      for (let i = 0; i < KT; i++) tT[i] = ring[i][k];
      const H = median(tT, KT);
      let m = 0;
      for (let d = -KF; d <= KF; d++) { const kk = k + d; tF[m++] = kk > 0 && kk < NB ? center[kk] : 0; }
      const Pm = median(tF, m);
      const p2 = Pm * Pm, h2 = H * H;
      perc[k] = center[k] * (p2 / (p2 + h2 + 1e-20));
    }

    // 帯域ごとのエネルギーの立ち上がり (対数差) をゆっくり追従する平均で正規化
    const O = new Float64Array(4);
    for (let b = 0; b < 4; b++) {
      const [lo, hi] = this.bands[b];
      let e = 0;
      for (let k = lo; k <= hi; k++) e += perc[k] * perc[k];
      const flux = Math.max(0, Math.log(e + 1e-12) - Math.log(this.prevE[b] + 1e-12)) * Math.sqrt(e);
      this.prevE[b] = e;
      O[b] = flux / (this.avg[b] + 1e-12);
      this.avg[b] = this.avg[b] * 0.985 + flux * 0.015;
    }

    // 1 フレーム前がピーク (前後より大きい) かを判定。
    // 立ち上がりの強さは裾の重い分布になるので、しきい値は上位 2〜3 割に当たる程度 (T = thr * 2)
    const [pp, p] = this.hist;
    const T = thr * 2;
    const peak = (b) => p[b] > T && p[b] > pp[b] && p[b] >= O[b];
    this.cool = this.cool.map((c) => c - 1);
    // それぞれ自分の帯域が主役のときだけ (広い帯域の立ち上がり = 歌やギターの子音を誤検出しない)
    const kick = peak(0) && p[0] >= 0.8 * Math.max(p[1], p[2]) && this.cool[0] <= 0;
    const snare = peak(2) && p[1] > T * 0.6 && p[2] >= 0.8 * p[3] && this.cool[1] <= 0;
    const hat = peak(3) && p[3] > p[2] && this.cool[2] <= 0;
    let res = null;
    if (kick) { res = { hit: 'kick', strength: p[0] }; this.cool[0] = 5; }
    else if (snare) { res = { hit: 'snare', strength: (p[1] + p[2]) / 2 }; this.cool[1] = 5; }
    else if (hat) { res = { hit: 'hat', strength: p[3] }; this.cool[2] = 3; }
    if (res) res.strength = Math.min(1, res.strength / (T * 3));
    this.hist = [p, O];
    this.onset = O[0] + O[1] + O[2] + 0.5 * O[3]; // 全帯域の立ち上がり (テンポ推定用)
    return res;
  }
}
