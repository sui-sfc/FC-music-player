// ============================================================
//  サウンドドライバ (解析 → APU レジスタ書き込み)
//  ---------------------------------------------------------
//  1 フレーム (1/60 秒) ごとに、入力音声のスペクトルから
//   - 矩形波1 : メロディ (ボーカル)
//   - 矩形波2 : ハーモニー (コード / アルペジオ)
//   - 三角波  : ベース (+ キックのピッチスイープ)
//   - ノイズ  : ドラム (キック / スネア / ハイハット)
//  を推定し、実機と同じ形式の APU レジスタ値を書き込む。
//
//  出力は「1 フレームあたり 16 バイトのレジスタ値」なので、
//  記録したログをそのままファミコン実機の再生ルーチンに渡せる。
// ============================================================

import { CPU_CLOCK } from './apu.js';
import { viterbiFirst } from './viterbi.js';
import { chromaFrom, chordEmits, decideChord, chordName, chordRoot, chordIntervals, N_CHORDS } from './chords.js';
import { DrumDetector, DRUM_DELAY, ONSET_DELAY } from './drums.js';
import { BP_MIDI0 } from './basicpitch.js';
import { buildDrumBank, DPCM_START_LEVEL } from './dpcm.js';
import { TempoTracker, EnergyTracker, arpTone } from './arranger.js';

export const NOTE_LO = 28; // E1
export const NOTE_HI = 100; // E7
const VOCAL_LO = 45; // A2
const VOCAL_HI = 84; // C6

export const noteFreq = (n) => 440 * Math.pow(2, (n - 69) / 12);
const freqToMidi = (f) => 69 + 12 * Math.log2(f / 440);
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);

export function pulsePeriod(freq) {
  return clamp(Math.round(CPU_CLOCK / (16 * freq) - 1), 8, 0x7ff);
}
export function triPeriod(freq) {
  return clamp(Math.round(CPU_CLOCK / (32 * freq) - 1), 2, 0x7ff);
}

// ------------------------------------------------------------
//  倍音加算による「各ノートらしさ (顕著度)」の計算
// ------------------------------------------------------------
export class Salience {
  // binHz: FFT 1 ビンの周波数幅, nb: ビン数, lo/hi: 対象ノート範囲,
  // weights: 倍音ごとの重み, maxFreq: 使う倍音の上限周波数, oddCheck: 奇数倍音チェック
  // lowCut: これより低い倍音は伴奏 (ギター・ベース) と重なりやすいので重みを下げる
  constructor({ binHz, nb, lo, hi, weights, maxFreq, oddCheck = false, f0Penalty = 0.4, lowCut = 0, lowGain = 0.3 }) {
    Object.assign(this, { binHz, nb, lo, hi, weights, oddCheck, f0Penalty, lowCut });
    this.H = weights.length;
    this.count = hi - lo + 1;
    const nk = this.count * this.H;
    this.rlo = new Int32Array(nk);
    this.rhi = new Int32Array(nk);
    this.flo = new Float32Array(nk);
    this.fhi = new Float32Array(nk);
    this.w = new Float32Array(nk); // ノート×倍音ごとの重み
    for (let i = 0; i < this.count; i++) {
      const f0 = noteFreq(i + lo);
      for (let h = 0; h < this.H; h++) {
        const f = f0 * (h + 1), k = i * this.H + h;
        this.w[k] = weights[h] * (f < lowCut ? lowGain : 1);
        if (f > maxFreq) { this.rlo[k] = 1; this.rhi[k] = 0; continue; } // 空範囲
        // ±1/4 音 (= 隣のノートと重ならない) に入るピークだけを数える
        this.flo[k] = (f * Math.pow(2, -1 / 24)) / binHz;
        this.fhi[k] = (f * Math.pow(2, 1 / 24)) / binHz;
        this.rlo[k] = Math.max(1, Math.floor(this.flo[k]) - 1);
        this.rhi[k] = Math.min(nb - 2, Math.ceil(this.fhi[k]) + 1);
      }
    }
    this.pk = new Float32Array(nb);
    this.pkv = new Float32Array(nb); // 局所最大ビンのピーク値 (それ以外 0)
    this.pkf = new Float32Array(nb); // 放物線補間したピーク位置 (ビン単位)
    this.prefix = new Float64Array(nb + 1);
    this.sal = new Float32Array(this.count);
  }

  analyze(amp) {
    const nb = this.nb, pre = this.prefix, pk = this.pk;
    // スペクトルの白色化 (局所平均を引いてピークだけ残す)
    pre[0] = 0;
    for (let k = 0; k < nb; k++) pre[k + 1] = pre[k] + amp[k];
    for (let k = 0; k < nb; k++) {
      const w = Math.max(6, Math.round(k * 0.12));
      const a = Math.max(0, k - w), b = Math.min(nb, k + w + 1);
      const v = amp[k] - (1.5 * (pre[b] - pre[a])) / (b - a);
      pk[k] = v > 0 ? v : 0;
    }
    // 局所最大を探し、放物線補間で正確な周波数を求める
    const pkv = this.pkv, pkf = this.pkf;
    pkv.fill(0);
    for (let k = 2; k < nb - 1; k++) {
      if (pk[k] <= 0 || amp[k] < amp[k - 1] || amp[k] < amp[k + 1]) continue;
      const a = Math.log(amp[k - 1] + 1e-12), b = Math.log(amp[k] + 1e-12), c = Math.log(amp[k + 1] + 1e-12);
      const den = a - 2 * b + c;
      pkv[k] = pk[k];
      pkf[k] = k + (den < 0 ? (0.5 * (a - c)) / den : 0);
    }
    // 倍音加算
    const H = this.H, W = this.w, sal = this.sal;
    let smax = 0;
    for (let i = 0; i < this.count; i++) {
      let s = 0, mh = 0, m1 = 0, odd = 0, even = 0;
      for (let h = 0; h < H; h++) {
        const m = this._harm(i * H + h);
        if (h === 0) m1 = m;
        if (m > mh) mh = m;
        s += W[i * H + h] * m;
        if (h & 1) even += m; else odd += m;
      }
      if (m1 < 0.2 * mh && noteFreq(i + this.lo) >= this.lowCut) s *= this.f0Penalty; // 基音が無い → 1 オクターブ下の誤検出を抑制
      // 奇数倍音 (1,3,5…) がほとんど無い → 本当の音は 1 オクターブ上
      if (this.oddCheck && odd < 0.35 * even) s *= 0.3;
      sal[i] = s;
      if (s > smax) smax = s;
    }
    return smax;
  }

  _harm(k) {
    const fl = this.flo[k], fh = this.fhi[k], pkv = this.pkv, pkf = this.pkf;
    let m = 0;
    for (let b = this.rlo[k]; b <= this.rhi[k]; b++) if (pkv[b] > m && pkf[b] >= fl && pkf[b] < fh) m = pkv[b];
    return m;
  }

  // ノート n の倍音ピーク位置から、半音より細かい実際の基本周波数を推定 (Hz)
  refine(n) {
    const i = n - this.lo;
    let sw = 0, sf = 0;
    for (let h = 0; h < this.H; h++) {
      const k = i * this.H + h;
      const fl = this.flo[k], fh = this.fhi[k];
      let m = 0, best = -1;
      for (let b = this.rlo[k]; b <= this.rhi[k]; b++)
        if (this.pkv[b] > m && this.pkf[b] >= fl && this.pkf[b] < fh) { m = this.pkv[b]; best = b; }
      if (best < 0) continue;
      const w = this.weights[h] * m;
      sf += w * (this.pkf[best] / (h + 1));
      sw += w;
    }
    return sw > 0 ? (sf / sw) * this.binHz : noteFreq(n);
  }

  s(n) { return n >= this.lo && n <= this.hi ? this.sal[n - this.lo] : 0; }
}

// 音程の揺れ (ビブラートや検出ミス) で音がバタつかないためのヒステリシス
class Voice {
  constructor() { this.note = -1; this.cand = -1; this.count = 0; }
  update(n, hold) {
    if (n === this.note) { this.cand = -1; this.count = 0; return this.note; }
    if (n === this.cand) this.count++;
    else { this.cand = n; this.count = 1; }
    if (this.count >= hold) { this.note = n; this.cand = -1; this.count = 0; }
    return this.note;
  }
  reset() { this.note = -1; this.cand = -1; this.count = 0; }
}

// ドラム音色 (1 フレームごとの [ノイズ周期 index, 音量])
const DRUMS = {
  kick:  { pri: 3, seq: [[13, 15], [14, 11], [14, 7], [15, 4], [15, 2]], tri: [170, 110, 75] },
  snare: { pri: 2, seq: [[4, 15], [5, 14], [6, 12], [6, 10], [7, 8], [7, 6], [8, 4], [8, 3], [8, 2], [9, 1]] },
  hat:   { pri: 1, seq: [[3, 15], [3, 11], [3, 8], [3, 5], [3, 3], [3, 1]] }, // index 1 だと音の大半が可聴域の外に出る
};

// DPCM でキック・スネアを鳴らすときに重ねるノイズ (アタックの「チッ」だけ)
const DRUMS_LAYER = {
  kick:  { pri: 3, seq: [[12, 4], [13, 2]] },
  snare: { pri: 2, seq: [[3, 15], [4, 13], [5, 11], [6, 9], [6, 7], [7, 5], [7, 3], [8, 2]] },
};

export const DEFAULT_PARAMS = {
  duty1: 1, // 25% (聴き比べで 50% より歌らしいと判断)
  duty2: 0, // 12.5% (矩形波 1 と音色を分けてメロディを立たせる)
  mute: [false, false, false, false], // P1, P2, Tri, Noise
  p2mode: 'auto', // 矩形波2: 'auto' 盛り上がりで切替 (静か = 楽器の旋律を控えめに, 普通 = 楽器の旋律, サビ = 拍に合わせたアルペジオ) / 'inst' 楽器の旋律 / 'arp' コードのアルペジオ / 'harm' ハモリ / 'echo' メロディのエコー
  instThr: 0.2, // 楽器の音として拾う Basic Pitch の確率の下限
  instPad: true, // 楽器の旋律が無い所はコードの音を静かに伸ばす
  instCost: 2.5, // 楽器の旋律の落ち着き (大きいほど音が変わりにくい)
  arpSpeed: 1, // アルペジオ 1 音あたりのフレーム数 (arpRate = 'frame' のとき)
  arpRate: '32', // アルペジオの速さ: '16' 16 分音符 / '32' 32 分音符 (テンポに合わせる) / 'frame' 毎フレーム (従来)
  arpPattern: 'updown', // アルペジオの音型: 'up' / 'down' / 'updown' (1-3-5-3) / 'up8' (1-3-5-8)
  quietDb: -9, // 盛り上がりの判定: 曲の大きい所よりこれだけ小さいと「静か」
  loudDb: -1.5, // 歌の大きさがこれまでで一番大きい所からこの範囲内なら「サビ」
  attackBend: 0, // 音の出だしで上からしゃくり下ろす幅 (半音, 0 = しない。歌モノでは上ずりを防ぐため既定 0)
  dutyEnv: 'pluck', // 音の出だしのデューティ比: 'pluck' 最初の 2 フレームを 50% (太く) / 'thin' 12.5% (鋭く) / 'off'
  release: 4, // 音の終わりに 12.5% で小さく残す長さ (フレーム, 0 = すぐ切る)
  arpLow: 62, // アルペジオの最低音 (D4)
  echoDelay: 10, // エコーの遅れ (フレーム)
  bassOctave: 0,
  bassPattern: 'oct', // (聴き比べで G を採用) ベースの弾き方: 'follow' 原曲どおり / '8th' 8 分音符で刻む / 'oct' 8 分音符でオクターブを行き来 (テンポが分かるとき)
  melQuant: false, // メロディの音の切り替えを 16 分音符の拍に揃える (テンポが分かるとき)
  bassFloor: 0, // これより低いベース音は 1 オクターブ上げる (MIDI 番号, 0 = しない)
  kickTri: true,
  hold: 2,
  sens: 0.25, // 2 声目以降の相対しきい値 (小さいほど音を拾う)
  drumSens: 1.8, // ドラム判定のしきい値 (小さいほどドラムを拾う)
  drumVol: 1.0,
  drumMode: 'dpcm', // 'dpcm' キック・スネアを DPCM サンプルで / 'noise' ノイズと三角波だけで
  melodyVol: 1.0,
  harmVol: 0.65,
  vocal: true, // ボーカルモード: ステレオ中央成分からメロディを追跡
  melOctave: '0', // (聴き比べで、上げない方が自然と判断)  メロディのオクターブ: 'auto' 歌が低ければ 1 オクターブ上げる / -1 / 0 / 1
  lead: true, // 歌の無い区間はリード楽器を矩形波1で鳴らす
  vocalPresence: 0.25, // RMVPE の確信度の平均がこれ未満なら歌ではなく楽器とみなす
  pitchMode: 'chip', // メロディの音程: 'chip' 半音に固定 + 伸ばした音にだけ一定のビブラート / 'snapvib' 半音に合わせて歌のビブラートは残す / 'snap' 半音のみ / 'raw' そのまま
  minNote: 4, // chip: 新しい音に切り替えるのに必要な長さ (フレーム)
  vibDelay: 18, // chip: 音の出だしからビブラートが始まるまで (フレーム)
  vibDepth: 0.15, // chip: ビブラートの深さ (半音)
  vibPeriod: 10, // chip: ビブラートの周期 (フレーム, 10 = 6Hz)
  voiceThr: 0.2, // 発声とみなす強さ (相対値)
  lookahead: 30, // 先読みフレーム数 (30 = 0.5 秒)。この分だけ出力が遅れる
  extThr: 0.1, // 外部 (RMVPE) の確信度がこれ以上なら歌っているとみなす
  noteCost: 1.5, // メロディの音符を切り替えるコスト。大きいほど音が落ち着く
  chordCost: 3.0, // コードを切り替えるコスト
  bassF0Thr: 0.3, // ベースの単音推定 (RMVPE) を使う確信度の下限
  bassF0W: 0.5, // ベースの単音推定の重み
  bassCost: 1.5, // ベースの音を切り替えるコスト
  accent: 1, // 音の出だしの音量の上乗せ (段)
  decay: 0.5, // 1 音の中での減衰の深さ (0 = 減衰しない, 0.5 = 約 0.3 秒で半分まで)
};

const MAX_LOOKAHEAD = 120;
const SILENT = { gate: false, bassN: -1, picks: [], cands: [], mel: -1, melPitch: -1, melVol: 0, time: 0, ext: true, bpDone: true, chromaS: { chroma: new Float64Array(12), total: 0 } };

export class Driver {
  // fftSize: 全体解析の FFT 長, vocalFftSize: ボーカル解析の FFT 長 (短い窓)
  constructor(sampleRate, fftSize, vocalFftSize, write) {
    this.write = write;
    this.params = { ...DEFAULT_PARAMS };
    // 全体 (楽器・コード・ベース): 長い窓で低音まで分解
    this.A = new Salience({
      binHz: sampleRate / fftSize, nb: fftSize / 2, lo: NOTE_LO, hi: NOTE_HI,
      weights: [1.0, 0.8, 0.64, 0.5, 0.4, 0.32], maxFreq: Math.min(sampleRate / 2, 9000),
    });
    // ボーカル: 短い窓 (ビブラートでにじまない) + 多くの倍音 (伴奏と重なる低域に頼らない)
    this.V = new Salience({
      binHz: sampleRate / vocalFftSize, nb: vocalFftSize / 2, lo: VOCAL_LO - 1, hi: VOCAL_HI + 1,
      weights: [1, 0.9, 0.81, 0.73, 0.66, 0.59, 0.53, 0.48, 0.43, 0.39], maxFreq: 5000,
      oddCheck: true, f0Penalty: 0.9, lowCut: 300,
    });
    this.drums = new DrumDetector(sampleRate / vocalFftSize, vocalFftSize / 2);
    this.drumArgs = [sampleRate / vocalFftSize, vocalFftSize / 2];
    this.tmp = new Float32Array(this.A.count);
    this.mel = new Voice();
    this.har = new Voice();
    this.regs = new Uint8Array(20); // $4000-$4013 の写し
    this.dpcm = buildDrumBank(); // カートリッジの $C000- に置く DPCM サンプル
    this.extLookup = null; // 秒 → { f0, conf } (RMVPE)
    this.bpLookup = null; // 秒 → { note, onset } (Basic Pitch, 各 88 鍵)
    this.reset();
  }

  reset() {
    this.agc = 1e-6;
    this.agcV = 1e-6;
    this.agcChord = 1e-6;
    this.center = 64; // 声域の中心 (E4 から開始)
    this.lastM = new Float32Array(128).fill(NaN);
    this.motion = new Float32Array(128);
    this.queue = []; // 先読み用: 解析済みでまだ出力していないフレーム
    this.drums = new DrumDetector(...(this.drumArgs || [1, 1]));
    this.tempo = new TempoTracker(); // ドラムの立ち上がりからテンポと拍の位置
    this.energy = new EnergyTracker(); // 曲の盛り上がり
    this.level = 1; // 0 = 静か, 1 = 普通, 2 = サビ
    this.prevBeatInt = null;
    this.anaN = 0; // 解析したフレーム数
    this.relLeft = 0; this.relVol = 0; this.prevV1 = 0; // 矩形波1 の音の終わり (リリース)
    // メロディ
    this.vState = -1; // 確定した音程 (-1 = 無音)
    this.vOut = -1; // 平滑化後の音程
    this.noteCenter = -1; // 音符の中心音程 (ビブラートを除いた音程)
    this.melAge = 0; // 今の音符が始まってからのフレーム数
    this.chipNote = -1; // chip モードで鳴らしている音
    this.notePeak = 0;
    this.melSalMin = 1; this.prevMelSal = 0; // 同音連打 (再発音) の検出用
    this.melHist = []; // エコー用の履歴
    this.octUp = false; // 1 オクターブ上げているか (auto のとき)
    this.vocalGap = 999; // 歌が途切れてからのフレーム数
    this.leadState = -1; // リード楽器の確定音程
    this.instState = -1; this.instAge = 0; this.instPeak = 0; this.padNote = -1; this.padChord = -1; // 矩形波2 の楽器の旋律
    // コード・ベース
    this.chordState = -1;
    this.chordId = -1;
    this.bassState = -1;
    this.bassAge = 0;
    this.bassE8 = undefined;
    this.mel.reset(); this.har.reset();
    this.frameNo = 0;
    this.vol = [0, 0];
    this.drum = null; this.drumPos = 0; this.drumGain = 1;
    this.hiWritten = [-1, -1, -1, -1];
    this.mask = 0;
    this.regs.fill(0);
    this.write(0x4015, 0x0f);
    this.write(0x4001, 0x08); // スイープ無効 + negate (低音ミュート回避)
    this.write(0x4005, 0x08);
    this.regs[1] = 0x08; this.regs[5] = 0x08;
  }

  _w(addr, v) {
    if (addr < 0x4014) this.regs[addr - 0x4000] = v;
    this.write(addr, v);
  }

  // $4003/$4007/$400B は位相がリセットされるので上位ビットが変わったときだけ書く
  _wHi(ch, addr, v) {
    if (this.hiWritten[ch] !== v) {
      this.hiWritten[ch] = v;
      this._w(addr, v);
      this.mask |= 1 << ch;
    }
  }

  // ---- ボーカル候補 ----
  // 中央成分のピークから歌の音程候補を作り、「歌らしさ」スコアを付ける:
  //   顕著度 × 音程の揺れ (ビブラート) × 声域の中心への近さ
  _vocalCands(gate, bassN) {
    const V = this.V;
    const cands = [];
    for (let n = VOCAL_LO; n <= VOCAL_HI; n++) {
      let s = V.s(n);
      if (!(s > 0 && s >= V.s(n - 1) && s > V.s(n + 1))) continue;
      // ベースも中央に定位していることが多いので、ベースの倍音にあたる候補は割り引く
      if (bassN >= 0 && [12, 19, 24, 28, 31, 36].includes(n - bassN)) s *= 0.5;
      let m = freqToMidi(V.refine(n));
      if (!(Math.abs(m - n) <= 0.6)) m = n;
      cands.push({ note: n, s, m, motion: 0 });
    }
    cands.sort((a, b) => b.s - a.s);
    const top = cands.length ? cands[0].s : 0;
    this.agcV = Math.max(top, this.agcV * 0.998, 1e-6);

    // 音程の揺れ: 前フレームの近い候補からの移動量を平滑化 (ギターなどはほぼ 0、歌は大きい)
    const lastM = this.lastM, motion = this.motion;
    const newM = new Float32Array(128).fill(NaN), newMotion = motion.map((v) => v * 0.8);
    for (const c of cands) {
      let pm = NaN;
      for (let k = c.note - 1; k <= c.note + 1; k++) {
        const v = lastM[k];
        if (!isNaN(v) && Math.abs(v - c.m) < 0.8 && (isNaN(pm) || Math.abs(v - c.m) < Math.abs(pm - c.m))) pm = v;
      }
      const d = isNaN(pm) ? 0 : Math.min(Math.abs(c.m - pm), 0.5);
      let mot = 0;
      for (let k = c.note - 1; k <= c.note + 1; k++) if (motion[k] > mot) mot = motion[k];
      c.motion = mot * 0.6 + d * 0.4;
      newM[c.note] = c.m;
      newMotion[c.note] = c.motion;
    }
    this.lastM = newM;
    this.motion = newMotion;
    if (!gate) return [];

    // 1 オクターブ下の可能性も候補に入れる (歌の奇数倍音が伴奏に埋もれると 1 オクターブ上に見えるため)
    const pool = cands.slice(0, 6).filter((c) => c.s >= 0.2 * top);
    for (const c of pool.slice(0, 3)) {
      if (c.note - 12 >= VOCAL_LO && !pool.some((p) => p.note === c.note - 12)) {
        pool.push({ note: c.note - 12, s: c.s * 0.7, m: c.m - 12, motion: c.motion });
      }
    }
    return pool.map((c) => {
      const vib = 1 + 1.5 * Math.min(1, c.motion / 0.12);
      const dc = c.m - this.center;
      const sc = (c.s / this.agcV) * vib * Math.exp(-(dc * dc) / 128); // 声域の中心から ±8 半音程度
      return { m: c.m, s: c.s / this.agcV, e: Math.log(Math.max(sc, 1e-4)) };
    });
  }

  // ---- 外部の解析結果 (機械学習モデル) をフレームに差し込む ----
  // まだ結果が届いていないフレームは内蔵の推定のまま使う
  _attach(rec) {
    const P = this.params;
    // RMVPE をベースの音 (分離済み) にかけた結果: ベースは単音なので、和音向けの採譜より音程が確か
    if (!rec.bassExt && this.bassLookup) {
      const b = this.bassLookup(rec.time);
      if (b) { rec.bassExt = true; rec.bassF0 = b.f0; rec.bassConf = b.conf; rec.bassC = null; }
    }
    // RMVPE: 歌の音程
    if (!rec.ext && this.extLookup && rec.vsal) {
      const r = this.extLookup(rec.time);
      if (r) {
        rec.ext = true;
        rec.conf = r.f0 > 0 ? r.conf : 0;
        rec.cands = []; rec.instC = null;
        if (rec.gate && r.f0 > 0) {
          const m = freqToMidi(r.f0), n = Math.round(m), V = this.V;
          // 音量は中央成分の強さから (モデルの確信度は「歌らしさ」で、大きさではないため)
          const vs = (k) => (k >= V.lo && k <= V.hi ? rec.vsal[k - V.lo] : 0);
          const v = Math.max(vs(n), vs(n - 1), vs(n + 1), vs(n + 12)) / rec.agcV;
          rec.cands.push({ m, s: clamp(v, 0.15, 1), e: Math.log(Math.max((r.conf / P.extThr) * P.voiceThr, 1e-4)) });
        }
      }
    }
    // Basic Pitch: 鳴っている全ての音符
    if (!rec.bpDone && this.bpLookup) {
      const b = this.bpLookup(rec.time);
      if (b) { rec.bp = b; rec.bpDone = true; rec.chordE = null; rec.bassC = null; rec.instC = null; }
    }
  }

  // コード認識用のスコア (Basic Pitch があればそれ、無ければスペクトルのピークから)
  _chordEmits(rec) {
    if (rec.chordE) return rec.chordE;
    let c;
    if (rec.bp) {
      const note = rec.bp.note;
      c = chromaFrom((m) => { const p = note[m - BP_MIDI0]; return p > 0.15 ? p : 0; });
      rec.chordTotal = c.total;
      rec.chordE = chordEmits(c.chroma, c.total, 0.6);
    } else {
      c = rec.chromaS;
      rec.chordTotal = c.total;
      rec.chordE = chordEmits(c.chroma, c.total, 0.4);
    }
    return rec.chordE;
  }

  // リード楽器の候補 (歌の無い区間でメロディを担当する、高めの音域で目立つ音)
  _leadCands(rec) {
    if (rec.leadC) return rec.leadC;
    let out = [];
    if (rec.bp) {
      const note = rec.bp.inst || rec.bp.note; // 分離済みなら「その他の楽器」だけから
      for (let m = 60; m <= 86; m++) { // C4〜D6 (それより上はシンバルなどの倍音を拾いやすい)
        const p = note[m - BP_MIDI0];
        if (p >= 0.35) out.push({ m, p, e: Math.log(p) + 0.004 * (m - 60) }); // 同じくらいなら高い音を少し優先
      }
    } else {
      for (const pk of rec.picks) if (pk.note >= 60) out.push({ m: pk.note, p: 0.5, e: Math.log(0.5) + 0.015 * (pk.note - 60) });
    }
    out.sort((a, b) => b.e - a.e);
    out = out.slice(0, 3);
    rec.leadC = out;
    return out;
  }

  // 楽器の旋律の候補 (歌とベース以外で鳴っている中音域の音)
  _instCands(rec) {
    if (rec.instC) return rec.instC;
    const P = this.params;
    const out = [];
    if (rec.bp && rec.gate) {
      const note = rec.bp.inst || rec.bp.note, onset = rec.bp.instOn || rec.bp.onset;
      // 歌の音程 (とそのオクターブ) は除く。歌は矩形波1が鳴らしている
      const vm = rec.cands.length && rec.conf > 0.03 ? Math.round(rec.cands[0].m) : -99;
      for (let m = 55; m <= 79; m++) { // G3〜G5
        const p = note[m - BP_MIDI0];
        if (p < P.instThr) continue;
        if (vm > 0 && (Math.abs(m - vm) <= 1 || (m - vm) % 12 === 0)) continue;
        // 中音域 (C4〜C5 あたり) を少し優先: 矩形波2 は歌より下で鳴る方が落ち着く
        const e = Math.log(p) - 0.01 * Math.abs(m - 66);
        out.push({ m, p, e, on: onset && onset[m - BP_MIDI0] > 0.5 });
      }
      out.sort((a, b) => b.e - a.e);
      out.length = Math.min(out.length, 4);
    }
    rec.instC = out;
    return out;
  }

  // ベースの候補 (低い音域で鳴っている音)
  _bassCands(rec) {
    const P = this.params;
    if (rec.bassC) return rec.bassC;
    const out = [];
    if (rec.bp) {
      const note = rec.bp.bass || rec.bp.note; // 分離済みならベースの音だけから
      let mx = 0;
      for (let m = 28; m <= 55; m++) mx = Math.max(mx, note[m - BP_MIDI0]);
      for (let m = 28; m <= 55 && out.length < 3; m++) {
        const p = note[m - BP_MIDI0];
        if (p >= 0.25 && p >= 0.4 * mx) out.push({ m, e: Math.log(p) });
      }
    } else if (rec.bassPeaks && rec.bassPeaks.length) {
      // 内蔵解析でも複数候補をビタビに渡すことで、キック通過時の 1〜2 フレームの揺れや音欠けを防ぐ
      for (const bp of rec.bassPeaks) out.push({ m: bp.note, e: Math.log(Math.max(bp.p, 0.05)) });
    } else if (rec.bassN >= 0) out.push({ m: rec.bassN, e: Math.log(0.5) });
    if (rec.bassF0 > 0 && rec.bassConf >= P.bassF0Thr) {
      const m = Math.round(freqToMidi(rec.bassF0));
      if (m >= 24 && m <= 60) {
        const c = out.find((o) => o.m === m);
        const e = Math.log(Math.max(rec.bassConf, 0.05)) + P.bassF0W;
        if (c) c.e = Math.max(c.e, e) + 0.2; else out.unshift({ m, e });
      }
    }
    out.sort((a, b) => a.m - b.m);
    // ベースはルート音を弾くことが多い: 一番低い音と、そのフレームのコードのルートに加点
    if (out.length) out[0].e += 0.3;
    const ce = this._chordEmits(rec);
    let best = 0;
    for (let i = 1; i < N_CHORDS; i++) if (ce[i] > ce[best]) best = i;
    if (ce[best] > ce[N_CHORDS]) for (const c of out) if (c.m % 12 === chordRoot(best)) c.e += 0.3;
    rec.bassC = out;
    return out;
  }

  // ============ 1. 解析 (入力フレームごとにすぐ実行) ============
  _analyze(amp, ampV, rms, time) {
    const P = this.params, A = this.A;
    const smax = A.analyze(amp);
    this.agc = Math.max(smax, this.agc * 0.998, 1e-6);
    const gate = rms > 0.0015 && smax > 0.04 * this.agc;

    // ピーク抽出 (倍音関係にある音を抑制しながら最大 5 音)
    const picks = [];
    if (gate) {
      const t = this.tmp, NN = A.count;
      t.set(A.sal);
      for (let n = 0; n < 5; n++) {
        let bi = -1, bv = 0;
        for (let i = 0; i < NN; i++) if (t[i] > bv) { bv = t[i]; bi = i; }
        if (bi < 0) break;
        if (picks.length && bv < P.sens * picks[0].s) break;
        picks.push({ note: bi + NOTE_LO, s: A.sal[bi] });
        t[bi] = 0;
        if (bi > 0) t[bi - 1] = 0;
        if (bi < NN - 1) t[bi + 1] = 0;
        if (bi > 1) t[bi - 2] *= 0.5;
        if (bi < NN - 2) t[bi + 2] *= 0.5;
        for (const d of [12, 19, 24, 28, 31]) if (bi + d < NN) t[bi + d] *= 0.3;
        if (bi >= 12) t[bi - 12] *= 0.5;
      }
    }
    const top = picks.length ? picks[0].s : 0;
    const vol = (s) => clamp(Math.round(15 * Math.sqrt(s / this.agc)), 0, 15);

    // ベース (内蔵推定。Basic Pitch が無いときに使う。低域の上位候補も保持してビタビを安定させる)
    let bassN = -1;
    for (const p of picks) if (p.note <= 55 && p.s >= 0.12 * top && (bassN < 0 || p.note < bassN)) bassN = p.note;
    const bassPeaks = [];
    if (gate) {
      let bmax = 0;
      for (let n = NOTE_LO; n <= 55; n++) if (A.s(n) > bmax) bmax = A.s(n);
      if (bmax >= 0.12 * smax) {
        const raw = [];
        for (let n = NOTE_LO; n <= 55; n++) {
          const s = A.s(n);
          if (s >= 0.45 * bmax && s >= 0.08 * smax && s >= A.s(n - 1) && s >= A.s(n + 1)) raw.push({ note: n, s });
        }
        raw.sort((a, b) => b.s - a.s);
        for (const r of raw) {
          if (bassPeaks.length >= 4) break;
          bassPeaks.push({ note: r.note, p: 0.7 * (r.s / bmax) });
        }
        if (bassN < 0 && bmax >= 0.25 * smax && bassPeaks.length) bassN = bassPeaks[0].note;
      }
    }

    const rec = { gate, bassN, bassPeaks, picks, cands: [], mel: -1, melPitch: -1, melVol: 0, hit: null, time, ext: false, bpDone: false };
    // コード認識用のクロマ (内蔵推定)
    rec.chromaS = gate ? chromaFrom((m) => (m >= NOTE_LO && m <= NOTE_HI ? Math.min(1, A.s(m) / this.agc) : 0)) : { chroma: new Float64Array(12), total: 0 };
    rec.pickVol = new Map(picks.map((p) => [p.note, vol(p.s)]));

    // メロディ候補
    if (P.vocal) {
      this.V.analyze(ampV);
      rec.cands = this._vocalCands(gate, bassN);
      // 外部推定 (RMVPE) を後から差し込めるよう、音量計算に使う中央成分の強さを保存しておく
      rec.vsal = this.V.sal.slice();
      rec.agcV = this.agcV;
    } else {
      let melN = -1, melScore = 0;
      for (const p of picks) {
        if (p.note < 52 || p.note === bassN) continue;
        const sc = p.s * (1 + 0.03 * (p.note - 60));
        if (sc > melScore) { melScore = sc; melN = p.note; }
      }
      const cur = this.mel.note;
      if (melN >= 0 && cur >= 0 && Math.abs(melN - cur) === 1 && A.s(cur) >= 0.7 * A.s(melN)) melN = cur;
      const mel = this.mel.update(melN, P.hold);
      rec.mel = mel;
      rec.melPitch = mel;
      rec.melVol = mel >= 0 ? vol(A.s(mel)) : 0;
      if (mel >= 0) {
        const m = freqToMidi(A.refine(mel));
        if (Math.abs(m - mel) <= 0.5) rec.melPitch = m;
      }
    }
    return rec;
  }

  // ============ 2. 出力 (先読みぶん遅れて実行。レジスタを書く) ============
  _render(r, q) {
    const P = this.params;
    this.mask = 0;
    this.frameNo++;
    for (const rec of q) this._attach(rec);
    // 曲の盛り上がり (静か / 普通 / サビ) と、今が何拍目か
    const beat = r.fi !== undefined ? this.tempo.beatAt(r.fi + ONSET_DELAY) : null;
    const beatInt = beat !== null ? Math.floor(beat) : null;
    if (r.rms !== undefined) {
      const wantLevel = this.energy.update(r, q.slice(1), P);
      // テンポが分かっているときは拍の頭で場面を切り替える (小節の途中で唐突にアルペジオが始まらないように)
      if (wantLevel !== this.level && (beatInt === null || this.prevBeatInt === null || beatInt !== this.prevBeatInt)) {
        this.level = wantLevel;
      }
    }
    this.prevBeatInt = beatInt;

    // ---------------- メロディ ----------------
    let mel = -1, melPitch = -1, target = 0, melSal = 0;
    const trans = (a, b) => {
      if (a < 0 && b < 0) return 0;
      if (a < 0 || b < 0) return -1.2;
      const d = Math.abs(a - b);
      return d <= 1 ? -0.8 * d * d : -(P.noteCost + 0.08 * d);
    };
    if (P.vocal) {
      const states = q.map((x) => [...x.cands.map((c) => c.m), -1]);
      const emits = q.map((x) => [...x.cands.map((c) => c.e), Math.log(P.voiceThr)]);
      const j = viterbiFirst(states, emits, this.vState, trans);
      const m = states[0][j];
      this.vState = m;
      if (m >= 0) {
        // 確定した経路を軽く平滑化 (ビブラートは残し、細かいブレだけ取る)
        this.vOut = this.vOut >= 0 && Math.abs(m - this.vOut) <= 1 ? this.vOut + (m - this.vOut) * 0.7 : m;
        melPitch = this.vOut;
        melSal = r.cands[j].s;
        target = clamp(Math.round(15 * Math.sqrt(melSal) * P.melodyVol), 0, 15);
        this.center += (melPitch - this.center) * 0.008; // 声域の中心をゆっくり追従
      } else this.vOut = -1;
    } else if (r.mel >= 0) {
      melPitch = r.melPitch;
      target = Math.round(r.melVol * P.melodyVol);
      melSal = r.melVol / 15;
    }

    let isLead = false;
    if (P.vocal) {
      // 本当に歌か: RMVPE は「一番目立つ旋律」を拾うので楽器の旋律も拾う。
      // 先読み区間の確信度の平均が低い (楽器らしい) ときは歌として扱わず、リード楽器に任せる
      if (this.extLookup && melPitch >= 0) {
        let cs = 0, n = 0;
        for (const x of q) if (x.conf > 0.03) { cs += x.conf; n++; }
        if (n > 0 && cs / n < P.vocalPresence) melPitch = -1;
      }
      // フレーズの頭で、そのフレーズを何オクターブで鳴らすか決める:
      // 先読み区間の音程の中央値が C5 付近に来る方を選ぶ (歌が低ければ 1 オクターブ上げる = ファミコンアレンジの定番)
      if (melPitch >= 0 && this.vocalGap > 10) {
        const ms = q.filter((x) => x.cands.length).map((x) => x.cands[0].m).sort((a, b) => a - b);
        const med = ms.length ? ms[ms.length >> 1] : melPitch;
        if (P.melOctave === 'auto') {
          const d0 = Math.abs(med - 72), d1 = Math.abs(med + 12 - 72);
          this.phraseShift = d1 < d0 ? 12 : d0 < d1 ? 0 : this.phraseShift || 0;
        } else this.phraseShift = 12 * (Number(P.melOctave) || 0);
      }
      if (melPitch >= 0) { melPitch += this.phraseShift || 0; this.vocalGap = 0; } else this.vocalGap++;

      // 歌の無い区間 (イントロ・間奏) は、リード楽器 (シンセやギターの旋律) を先読みで追って鳴らす
      if (P.lead) {
        const ls = q.map((x) => [...this._leadCands(x).map((c) => c.m), -1]);
        const le = q.map((x) => [...x.leadC.map((c) => c.e), Math.log(0.3)]);
        const lj = viterbiFirst(ls, le, this.leadState, trans);
        this.leadState = ls[0][lj];
        // 息継ぎ程度の短い隙間 (0.3 秒未満) は埋めない
        if (melPitch < 0 && this.vocalGap > 20 && this.leadState >= 0) {
          melPitch = this.leadState;
          melSal = r.leadC[lj].p;
          target = clamp(Math.round(15 * Math.sqrt(melSal) * P.melodyVol * 0.9), 0, 15);
          isLead = true;
        }
      }
    }

    // 音程の整え方: 音符の中心を半音に合わせ、ビブラート (中心からのズレ) は残す
    let newNote = false, reArtic = false;
    // 16 分音符の区切り (メロディの切り替えを拍に揃えるとき用)
    const q16 = beat !== null ? Math.floor(beat * 4) : null;
    if (melPitch >= 0 && P.pitchMode === 'chip') {
      // ファミコン風: 音程は半音にぴったり固定。新しい音は、先読みで数フレーム続くと確認できたときだけ切り替える
      // (しゃくり上げ・音の間のすべり・ビブラートで細かい音が出ないように)。
      // 伸ばした音にだけ、決まった深さ・速さのビブラートを後からかける (チップチューンの定番)
      const n = Math.round(melPitch);
      const shift = isLead ? 0 : this.phraseShift || 0;
      // 先読みフレームに音程 n の候補があるか (cands[0] だけでなくビタビが選び得る上位候補も確認)
      const futureHas = (x) => {
        const list = isLead ? x.leadC : x.cands;
        if (!list || !list.length) return false;
        const bestE = list[0].e;
        for (const c of list) if (c.e >= bestE - 1.5 && Math.round(c.m + shift) === n) return true;
        return false;
      };
      if (this.chipNote < 0) { this.chipNote = n; newNote = true; this.melAge = 0; }
      else if (n !== this.chipNote && !(P.melQuant && this.q16 !== null && q16 === this.q16)) {
        let keep = 0;
        const K = Math.min(P.minNote, q.length - 1);
        for (let i = 1; i <= K; i++) if (futureHas(q[i])) keep++;
        if (keep >= K - 1) { this.chipNote = n; newNote = true; this.melAge = 0; }
      }
      this.q16 = q16;
      this.melAge++;
      // 同音連打 (同じ音程で言葉を刻む歌・連打リフ) を検出して発音を区切る
      if (!newNote) {
        if (isLead && r.bp) {
          const ons = r.bp.instOn || r.bp.onset;
          if (ons && this.chipNote >= BP_MIDI0 && this.chipNote < BP_MIDI0 + 88 && ons[this.chipNote - BP_MIDI0] > 0.45 && this.melAge >= 6) {
            reArtic = true;
          }
        } else if (this.melAge >= 8 && melSal > 0.12 && melSal > this.melSalMin * 1.4 && melSal > this.prevMelSal * 1.05) {
          reArtic = true;
        }
        if (reArtic) this.melAge = 1;
      }
      const a = this.melAge - P.vibDelay;
      const vib = a > 0 ? P.vibDepth * Math.min(1, a / 15) * Math.sin((2 * Math.PI * a) / P.vibPeriod) : 0;
      melPitch = this.chipNote + vib;
      mel = this.chipNote;
      this.noteCenter = this.chipNote;
    } else if (melPitch >= 0) {
      this.chipNote = -1;
      if (this.noteCenter < 0 || Math.abs(melPitch - this.noteCenter) > 0.8) {
        this.noteCenter = melPitch; this.melAge = 0; newNote = true;
      } else this.noteCenter += (melPitch - this.noteCenter) * 0.08;
      const snapped = Math.round(this.noteCenter);
      if (P.pitchMode === 'snapvib') melPitch = snapped + (melPitch - this.noteCenter);
      else if (P.pitchMode === 'snap') melPitch = snapped;
      mel = Math.round(melPitch);
      this.melAge++;
      if (!newNote && this.melAge >= 8 && melSal > 0.12 && melSal > this.melSalMin * 1.4 && melSal > this.prevMelSal * 1.05) {
        reArtic = true;
        this.melAge = 1;
      }
    } else { this.noteCenter = -1; this.chipNote = -1; this.melSalMin = 1; this.prevMelSal = 0; }

    // 同音連打の検出用に、現在の音符中での最小強さを追跡
    if (mel >= 0) {
      if (newNote || reArtic) this.melSalMin = melSal;
      else if (this.melAge >= 2 && melSal < this.melSalMin) this.melSalMin = melSal;
      this.prevMelSal = melSal;
    }

    // 音量: 1 音ごとに「出だしが最大 → 約 0.3 秒かけて減衰」(参考アレンジの音符の平均的な形)。
    // 1 音ずつ鳴らし直すことで、レガートな歌でもファミコンらしい粒立ちになる
    let v1 = 0;
    if (mel >= 0) {
      if (newNote || reArtic) this.notePeak = Math.max(target, 5);
      else if (target > this.notePeak) this.notePeak = Math.max(this.notePeak, target * 0.85);
      else this.notePeak = this.notePeak * 0.97 + target * 0.03; // 伸ばした音の自然なデクレッシェンドに緩やかに追従
      const env = 1 - P.decay + P.decay * Math.exp(-(this.melAge - 1) / 10);
      v1 = clamp(Math.round(Math.min(15, this.notePeak + P.accent) * env), 1, 15);
    }
    // 奏法: 音の出だしは上からしゃくり下ろし (attackBend)、デューティ比も最初の 2 フレームだけ変える。
    // 音の終わりはすぐ切らずに、12.5% で小さく数フレーム残す (release)
    let duty1 = P.duty1;
    if (mel >= 0) {
      this.relLeft = 0;
      if (P.attackBend && !reArtic && this.melAge <= 3) melPitch += P.attackBend * [1, 0.5, 0.25][this.melAge - 1];
      if (this.melAge <= 2 && P.dutyEnv === 'pluck') duty1 = 2;
      else if (this.melAge <= 2 && P.dutyEnv === 'thin') duty1 = 0;
    } else if (P.release > 0) {
      if (this.prevV1 > 0 && this.relLeft === 0) { this.relLeft = P.release; this.relVol = this.prevV1; }
      if (this.relLeft > 0) { v1 = Math.round((this.relVol * this.relLeft) / (P.release + 2)); duty1 = 0; this.relLeft--; if (v1 <= 0) this.relLeft = 0; }
    }
    if (P.mute[0]) v1 = 0;
    this.prevV1 = mel >= 0 ? v1 : 0;
    this.vol[0] = v1;
    this._w(0x4000, (duty1 << 6) | 0x30 | v1);
    if (mel >= 0) {
      const t = pulsePeriod(noteFreq(melPitch));
      this._w(0x4002, t & 0xff);
      this._wHi(0, 0x4003, 0xf8 | (t >> 8));
    }
    this.melHist.push({ pitch: v1 > 0 ? melPitch : -1, vol: v1 });
    if (this.melHist.length > 64) this.melHist.shift();

    // ---------------- コード ----------------
    const chordId = decideChord(q.map((x) => this._chordEmits(x)), this.chordState, P.chordCost);
    this.chordState = chordId;
    const hasChord = chordId < N_CHORDS;
    this.chordId = hasChord ? chordId : -1;
    this.agcChord = Math.max(r.chordTotal || 0, this.agcChord * 0.998, 1e-6);

    // ---------------- 矩形波 2 ----------------
    let h2 = -1, v2 = 0;
    // 'auto': 曲の盛り上がりで役割を変える (静か = 楽器の旋律を控えめに / 普通 = 楽器の旋律 / サビ = 拍に合わせたアルペジオ)
    const p2mode = P.p2mode === 'auto' ? (this.level === 2 && hasChord ? 'arp' : 'inst') : P.p2mode;
    const p2gain = P.p2mode === 'auto' && this.level === 0 ? 0.6 : 1;
    if (p2mode === 'echo') {
      const e = this.melHist[this.melHist.length - 1 - P.echoDelay];
      if (e && e.pitch >= 0) { h2 = e.pitch; v2 = Math.round(e.vol * 0.45); }
    } else if (p2mode === 'inst') {
      // 楽器の旋律: 原曲で歌・ベース以外に鳴っている楽器の音 (ギター・シンセ・ピアノの音) を
      // 先読みで 1 本の線にまとめて鳴らす。音はなるべく変えず、1 音ごとに減衰させて歌の邪魔をしない
      const fr = q.map((x) => this._instCands(x));
      const st = fr.map((c) => [...c.map((o) => o.m), -1]);
      const em = fr.map((c) => [...c.map((o) => o.e), Math.log(P.instThr * 0.8)]);
      const itrans = (a, b) => {
        if (a < 0 && b < 0) return 0;
        if (a < 0 || b < 0) return -1.5;
        const d = Math.abs(a - b);
        return d === 0 ? 0 : -(P.instCost + 0.1 * d);
      };
      const j = viterbiFirst(st, em, this.instState, itrans);
      let m = st[0][j];
      // 歌より上に出るとメロディが埋もれるので、歌より上なら 1 オクターブ下げる
      const below = m >= 0 && mel >= 0 && m > mel - 2 ? m - 12 : m;
      if (below >= 0 && (mel < 0 || (Math.abs(below - mel) > 1 && (below - mel) % 12 !== 0))) {
        // 楽器が弾き直した (Basic Pitch の発音) か、音程が変わったら新しい音
        const on = fr[0][j] && fr[0][j].on;
        if (m !== this.instState || on) { this.instAge = 0; this.instPeak = fr[0][j].p; }
        this.instAge++;
        h2 = below;
        const env = 0.4 + 0.6 * Math.exp(-(this.instAge - 1) / 12);
        v2 = clamp(Math.round((2 + 8 * Math.sqrt(this.instPeak)) * P.harmVol * env), 1, 15);
      }
      this.instState = m;
      if (m < 0) this.instAge = 0;
      // AI 解析 (Basic Pitch) が無いときや楽器が見つからない所は、コードの構成音を静かに伸ばす
      // (静かな場面でも p2gain で控えめになるので、無音にしてスカスカにしない)
      if (h2 < 0 && hasChord && r.gate && P.instPad) {
        const root = 60 + chordRoot(chordId);
        let tones = chordIntervals(chordId).map((d) => root + d);
        // 歌と半音・同じ音でぶつからない構成音だけ使う
        if (mel >= 0) tones = tones.map((t) => (t > mel - 2 ? t - 12 : t));
        const ok = tones.filter((t) => mel < 0 || (Math.abs(t - mel) > 1 && (t - mel) % 12 !== 0));
        if (ok.length) tones = ok;
        let t = tones[1] ?? tones[0];
        if (this.padNote >= 0) t = tones.reduce((a, b) => (Math.abs(b - this.padNote) < Math.abs(a - this.padNote) ? b : a));
        if (t !== this.padNote || chordId !== this.padChord) { this.instAge = 0; this.padNote = t; this.padChord = chordId; }
        this.instAge++;
        h2 = t;
        const e = Math.sqrt(Math.min(1, (r.chordTotal || 0) / this.agcChord));
        v2 = clamp(Math.round((2 + 6 * e) * P.harmVol * (0.5 + 0.5 * Math.exp(-(this.instAge - 1) / 15))), 1, 15);
      } else if (h2 < 0) this.padNote = -1;
      else this.padNote = h2;
    } else if (p2mode === 'arp') {
      if (hasChord && r.gate) {
        // コードの構成音を arpLow 以上の音域に並べ、音型 (arpPattern) に沿って切り替える。
        // 速さはテンポに合わせた 16 分 / 32 分音符 (テンポが分からないときは毎フレーム)
        const root = P.arpLow + ((chordRoot(chordId) - (P.arpLow % 12) + 12) % 12);
        let tones = chordIntervals(chordId).map((d) => root + d);
        if (tones.length === 2) tones.push(root + 12); // パワーコードはオクターブを足して 3 音に
        // 歌より上に出るとメロディが埋もれるので、和音全体を 1 オクターブ下げて転回形の乱れを防ぐ
        if (mel >= 0 && P.p2mode === 'auto' && tones[tones.length - 1] > mel - 2 && tones[0] - 12 >= 48) {
          tones = tones.map((t) => t - 12);
        }
        const a = arpTone(tones, P.arpPattern, P.arpRate, beat, this.frameNo, P.arpSpeed);
        h2 = a.note;
        const e = Math.sqrt(Math.min(1, (r.chordTotal || 0) / this.agcChord));
        // 拍に合わせるときは 1 音ごとに少し減衰させて粒を立てる
        v2 = clamp(Math.round((3 + 8 * e) * P.harmVol * (1 - 0.35 * a.age)), 0, 15);
      }
    } else {
      // ハモリ: メロディ以外で最も強い中音域の音
      let best = -1, bp = 0;
      if (r.bp) {
        for (let m = 52; m <= 84; m++) {
          const p = r.bp.note[m - BP_MIDI0];
          if (p > 0.3 && p > bp && (mel < 0 || (Math.abs(m - mel) > 1 && (m - mel) % 12 !== 0))) { bp = p; best = m; }
        }
      } else {
        const h = r.picks.filter((p) => p.note >= 45 && p.note !== r.bassN && (mel < 0 || (Math.abs(p.note - mel) > 1 && (p.note - mel) % 12 !== 0)));
        if (h.length) { best = h[0].note; bp = 0.6; }
      }
      h2 = this.har.update(best, P.hold);
      if (h2 >= 0 && r.gate) v2 = clamp(Math.round((3 + 8 * Math.sqrt(Math.max(bp, 0.3))) * P.harmVol), 0, 15);
    }
    v2 = Math.round(v2 * p2gain);
    if (h2 < 0 || P.mute[1]) v2 = 0;
    if (p2mode !== 'arp') v2 = Math.max(v2, h2 >= 0 && !P.mute[1] ? this.vol[1] - 1 : 0);
    this.vol[1] = v2;
    this._w(0x4004, (P.duty2 << 6) | 0x30 | v2);
    if (h2 >= 0) {
      const t = pulsePeriod(noteFreq(h2));
      this._w(0x4006, t & 0xff);
      this._wHi(1, 0x4007, 0xf8 | (t >> 8));
    }

    // ---------------- ノイズ (ドラム) ----------------
    let dmcHit = null;
    if (r.hit) {
      const useDpcm = P.drumMode === 'dpcm' && (r.hit === 'kick' || r.hit === 'snare') && !P.mute[3];
      if (useDpcm) dmcHit = r.hit;
      // DPCM キックと同時にハイハットが鳴っているときは、ノイズ側でハイハットを重ねる (実機定番の分担)
      const noiseHit = useDpcm && r.hit === 'kick' && r.hat ? 'hat' : r.hit;
      const d = useDpcm && noiseHit !== 'hat' ? DRUMS_LAYER[r.hit] : DRUMS[noiseHit];
      const curD = this.drum && this.drumPos < this.drum.seq.length ? this.drum : null;
      if (!curD || d.pri >= curD.pri || this.drumPos >= 2) {
        this.drum = d; this.drumPos = 0; this.drumName = noiseHit;
        const st = noiseHit === 'hat' && r.hit === 'kick' ? (r.hatStrength ?? 0.8) : (r.hitStrength ?? 1);
        this.drumGain = 0.55 + 0.45 * st; // 強く叩いたものほど大きく
      }
    }
    let nIdx = this.regs[14] & 15, nVol = 0, drumName = '';
    let kickTriFreq = 0;
    if (this.drum && this.drumPos < this.drum.seq.length) {
      [nIdx, nVol] = this.drum.seq[this.drumPos];
      nVol = clamp(Math.round(nVol * P.drumVol * this.drumGain), 0, 15);
      drumName = this.drumName;
      if (this.drum.tri && P.kickTri && this.drumPos < this.drum.tri.length) kickTriFreq = this.drum.tri[this.drumPos];
      this.drumPos++;
    }
    if (P.mute[3]) nVol = 0;
    this._w(0x400c, 0x30 | nVol);
    this._w(0x400e, nIdx);
    if (nVol > 0 && this.drumPos === 1) { this._w(0x400f, 0xf8); this.mask |= 8; }
    // DPCM: $4011 で DAC を中央に戻し、サンプルの場所と長さを設定して $4015 で (再) 開始
    if (dmcHit) {
      const smp = this.dpcm.samples[dmcHit];
      this._w(0x4011, DPCM_START_LEVEL);
      this._w(0x4010, this.dpcm.rate);
      this._w(0x4012, smp.reg12);
      this._w(0x4013, smp.reg13);
      this.write(0x4015, 0x0f);
      this.write(0x4015, 0x1f);
      this.mask |= 16;
      if (!drumName) drumName = dmcHit;
    }

    // ---------------- 三角波 (ベース / キック) ----------------
    const bstates = q.map((x) => [...this._bassCands(x).map((c) => c.m), -1]);
    const bemits = q.map((x) => [...x.bassC.map((c) => c.e), Math.log(0.25)]);
    const btrans = (a, b) => (a === b ? 0 : a < 0 || b < 0 ? -0.8 : -P.bassCost);
    const bj = viterbiFirst(bstates, bemits, this.bassState, btrans);
    const bass = r.gate ? bstates[0][bj] : -1;
    const use8th = P.bassPattern !== 'follow' && beat !== null && this.level >= 1;
    // 同じ音を弾き直したところ (Basic Pitch の立ち上がり) は 1 フレーム切って発音を区切る (8 分刻み時は二重カットしない)
    let retrig = false;
    const bassOn = r.bp && (r.bp.bassOn || r.bp.onset);
    if (!use8th && bass >= 0 && bass === this.bassState && bassOn && bassOn[bass - BP_MIDI0] > 0.5 && this.bassAge > 6) retrig = true;
    this.bassAge = bass === this.bassState && !retrig ? this.bassAge + 1 : 0;
    this.bassState = bass;

    let triOn = false, triNote = -1;
    if (!P.mute[2]) {
      let f = 0;
      if (kickTriFreq) f = kickTriFreq;
      else if (bass >= 0 && !retrig) {
        triNote = bass + 12 * P.bassOctave;
        // 8 分音符で刻む: 拍の頭で 1 フレーム切って弾き直す。'oct' は裏拍を 1 オクターブ上に (高すぎて歌と被らない範囲で)
        if (use8th) {
          const e8 = Math.floor(beat * 2);
          if (P.bassPattern === 'oct' && (e8 & 1) && triNote <= 48) triNote += 12;
          if (this.bassE8 !== undefined && e8 !== this.bassE8) triNote = -2; // 区切り
          this.bassE8 = e8;
        }
        if (triNote === -2) triNote = -1;
        else {
          if (triNote < P.bassFloor) triNote += 12; // 低すぎる音は三角波だと聞こえにくいので 1 オクターブ上げる
          f = noteFreq(triNote);
        }
      }
      if (f > 0) {
        const t = triPeriod(f);
        this._w(0x400a, t & 0xff);
        this._wHi(2, 0x400b, 0xf8 | (t >> 8));
        triOn = true;
      }
    }
    this._w(0x4008, triOn ? 0xff : 0x80);

    return {
      p1: { note: v1 > 0 ? mel : -1, pitch: v1 > 0 ? melPitch : -1, vol: v1 },
      p2: { note: v2 > 0 ? h2 : -1, vol: v2 },
      tri: { note: triOn ? (kickTriFreq ? -2 : triNote) : -1 },
      noise: { idx: nIdx, vol: nVol, drum: drumName },
      chord: chordName(this.chordId),
      lead: isLead, // 歌ではなくリード楽器を鳴らしているか
      regs: this.regs.slice(0, 20),
      mask: this.mask,
      lag: q.length - 1, // 入力に対する出力の遅れ (フレーム)
      ext: !!r.ext && !!this.extLookup, // 外部推定 (RMVPE) の結果を使ったか
      bp: !!r.bp, // Basic Pitch の結果を使ったか
      time: r.time,
      energy: this.level, // 0 = 静か, 1 = 普通, 2 = サビ
      edb: this.energy.db, // 曲の大きい所と比べた今の音量 (dB)
      vdb: this.energy.vdb, // 歌の一番大きい所と比べた今の歌の大きさ (dB)
      bpm: this.tempo.bpm,
    };
  }

  // amp: 振幅スペクトル (長窓), ampV: 中央成分スペクトル (短窓), ampVmix: 全体スペクトル (短窓),
  // rms: 入力 RMS, time: 解析窓の中心時刻 (秒。外部推定の結果と突き合わせるのに使う)
  // 戻り値: このフレームで出力したチャンネル情報 (先読みが溜まるまでは無音のフレーム)
  frame(amp, ampV, ampVmix, rms, time = 0) {
    const P = this.params;
    const L = clamp(Math.round(P.lookahead), 0, MAX_LOOKAHEAD);
    const rec = this._analyze(amp, ampV, rms, time);
    rec.fi = this.anaN++; rec.rms = rms;
    let ve = 0;
    for (let k = 0; k < ampV.length; k++) ve += ampV[k] * ampV[k];
    rec.ve = ve; // 歌の強さ (分離済みなら歌の音そのもの、そうでなければ中央成分)
    this.queue.push(rec);
    // ドラムは数フレーム遅れて判定が出るので、該当フレームに書き込む
    const d = this.drums.push(ampVmix, P.drumSens);
    this.tempo.push(this.drums.onset || 0); // 立ち上がりの強さ (テンポ推定用。ONSET_DELAY フレーム前のもの)
    const target = this.queue[this.queue.length - 1 - DRUM_DELAY];
    if (d && target) { target.hit = d.hit; target.hitStrength = d.strength; target.hat = d.hat; target.hatStrength = d.hatStrength; }
    // 先読みフレーム数が減らされた場合は溜まった分を出力して追いつく
    while (this.queue.length > L + 1) { this._render(this.queue[0], this.queue); this.queue.shift(); }
    if (this.queue.length <= L) return this._render(SILENT, [SILENT]);
    const info = this._render(this.queue[0], this.queue);
    this.queue.shift();
    return info;
  }
}
