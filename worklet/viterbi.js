// 先読みつき最尤経路 (ビタビ法) の共通処理
//
//   states[t]: フレーム t の状態の値の配列 (音程, コード番号など)
//   emits[t]:  各状態の「らしさ」(対数スコア)
//   anchor:    直前に確定した状態の値 (フレーム 0 の前)
//   trans(a, b): 状態 a → b に移るときのスコア (0 以下)
// 戻り値: フレーム 0 で選ばれた状態の番号 (未来まで見て最もつじつまの合う経路の先頭)
export function viterbiFirst(states, emits, anchor, trans) {
  const T = states.length;
  let delta = states[0].map((v, j) => trans(anchor, v) + emits[0][j]);
  const back = [];
  for (let t = 1; t < T; t++) {
    const S = states[t], Sp = states[t - 1], E = emits[t];
    const nd = new Float64Array(S.length), bp = new Int32Array(S.length);
    for (let j = 0; j < S.length; j++) {
      let best = -Infinity, bi = 0;
      for (let i = 0; i < Sp.length; i++) {
        const v = delta[i] + trans(Sp[i], S[j]);
        if (v > best) { best = v; bi = i; }
      }
      nd[j] = best + E[j];
      bp[j] = bi;
    }
    back.push(bp);
    delta = nd;
  }
  let j = 0;
  for (let k = 1; k < delta.length; k++) if (delta[k] > delta[j]) j = k;
  for (let t = T - 1; t > 0; t--) j = back[t - 1][j];
  return j;
}

// 状態数が多く「同じ状態に留まる = 0, 別の状態へ移る = 一律 -cost」の場合の高速版 (コード認識用)
//   emits[t]: Float64Array(S)  (全フレーム同じ状態集合)
export function viterbiFirstUniform(emits, anchor, cost) {
  const T = emits.length, S = emits[0].length;
  let delta = new Float64Array(S);
  for (let j = 0; j < S; j++) delta[j] = (j === anchor ? 0 : -cost) + emits[0][j];
  const back = [];
  for (let t = 1; t < T; t++) {
    let mi = 0;
    for (let i = 1; i < S; i++) if (delta[i] > delta[mi]) mi = i;
    const nd = new Float64Array(S), bp = new Int32Array(S);
    for (let j = 0; j < S; j++) {
      const stay = delta[j], move = delta[mi] - cost;
      if (stay >= move) { nd[j] = stay; bp[j] = j; } else { nd[j] = move; bp[j] = mi; }
      nd[j] += emits[t][j];
    }
    back.push(bp);
    delta = nd;
  }
  let j = 0;
  for (let k = 1; k < S; k++) if (delta[k] > delta[j]) j = k;
  for (let t = T - 1; t > 0; t--) j = back[t - 1][j];
  return j;
}
