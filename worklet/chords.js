// ============================================================
//  コード認識
//  ---------------------------------------------------------
//  各フレームの「音名ごとの強さ (クロマ, 12 次元)」をコードの型と照合し、
//  先読み区間全体で最もつじつまの合うコード進行をビタビ法で選ぶ。
//  型: メジャー / マイナー / パワーコード (ルート + 5 度, ロックで多用)
// ============================================================
import { viterbiFirstUniform } from './viterbi.js';

export const CHORD_TYPES = [
  { name: '', iv: [0, 4, 7], bias: 0 },
  { name: 'm', iv: [0, 3, 7], bias: 0 },
  { name: '5', iv: [0, 7], bias: 0.04 }, // 3 度が無ければパワーコードを優先
];
export const N_CHORDS = CHORD_TYPES.length * 12; // 36。番号 N_CHORDS = コード無し
const NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export function chordName(id) {
  if (id < 0 || id >= N_CHORDS) return 'N';
  const t = CHORD_TYPES[Math.floor(id / 12)];
  return NAMES[id % 12] + t.name;
}
export function chordRoot(id) { return id % 12; }
export function chordIntervals(id) { return CHORD_TYPES[Math.floor(id / 12)].iv; }

// noteProb(midi) → 0..1 から クロマと全体の強さを作る
export function chromaFrom(noteProb, lo = 33, hi = 84) {
  const c = new Float64Array(12);
  let total = 0;
  for (let m = lo; m <= hi; m++) {
    const p = noteProb(m);
    if (p <= 0) continue;
    const w = m < 48 ? 1.3 : 1; // 低音は和音の土台なので少し重く
    c[m % 12] += p * w;
    total += p * w;
  }
  return { chroma: c, total };
}

// 1 フレームの各コードの対数スコア (N_CHORDS + 1 個。最後が「コード無し」)
export function chordEmits(chroma, total, minTotal) {
  const e = new Float64Array(N_CHORDS + 1);
  const K = 8;
  if (total < minTotal) {
    for (let i = 0; i < N_CHORDS; i++) e[i] = -2;
    e[N_CHORDS] = 0;
    return e;
  }
  const c = chroma.map((v) => v / total);
  for (let ti = 0; ti < CHORD_TYPES.length; ti++) {
    const { iv, bias } = CHORD_TYPES[ti];
    for (let r = 0; r < 12; r++) {
      let inC = 0;
      for (const d of iv) inC += c[(r + d) % 12];
      const s = inC - 0.5 * (1 - inC) + bias + (c[r] > 0.12 ? 0.05 : 0); // ルートが鳴っていると少し加点
      e[ti * 12 + r] = K * s;
    }
  }
  e[N_CHORDS] = K * 0.2;
  return e;
}

// emitsList: フレームごとの chordEmits の結果 (先頭が今回決めるフレーム)
export function decideChord(emitsList, anchor, cost) {
  return viterbiFirstUniform(emitsList, anchor < 0 ? N_CHORDS : anchor, cost);
}
