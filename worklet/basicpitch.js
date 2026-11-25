// ============================================================
//  Basic Pitch (Spotify の多声音符推定モデル) の前後処理
//  ---------------------------------------------------------
//  入力: 22050Hz モノラル 43844 サンプル (約 2 秒) の窓
//  出力: 172 フレーム (256 サンプル = 約 11.6ms ごと) × 88 鍵 (A0〜C8) の
//        「音符が鳴っている確率 (note)」と「音符の立ち上がり確率 (onset)」
//  モデル本体の実行は環境ごとに違うので、ここには定数とつなぎ合わせの処理だけを置く。
// ============================================================

export const BP_SR = 22050;
export const BP_WIN = 43844;
export const BP_HOP = 256;
export const BP_FRAMES = 172;
export const BP_NOTES = 88;
export const BP_MIDI0 = 21; // 88 鍵の先頭 = A0
export const BP_OUTPUTS = { note: 'StatefulPartitionedCall:1', onset: 'StatefulPartitionedCall:2' };
export const BP_EDGE = 15; // 窓の両端で捨てるフレーム数 (公式実装と同じ)

// 曲全体をオフラインで処理 (公式実装と同じ窓の重ね方)
//   runWindow(Float32Array(BP_WIN)) → { note, onset } (各 BP_FRAMES * 88)
//   戻り値: { note, onset, frames } (全体のフレーム列, フレーム g の時刻 = g * BP_HOP / BP_SR)
export async function basicPitchFull(audio, runWindow) {
  const pad = BP_EDGE * BP_HOP; // 3840
  const hop = BP_WIN - 2 * pad; // 36164
  const total = Math.ceil(audio.length / BP_HOP);
  const note = new Float32Array(total * BP_NOTES), onset = new Float32Array(total * BP_NOTES);
  const buf = new Float32Array(BP_WIN);
  for (let s = -pad, g = 0; g < total; s += hop) {
    buf.fill(0);
    for (let i = 0; i < BP_WIN; i++) { const k = s + i; if (k >= 0 && k < audio.length) buf[i] = audio[k]; }
    const r = await runWindow(buf);
    // 窓の中のフレーム f の時刻 = 窓の先頭 s + f * BP_HOP。それに一番近い全体のフレームに置く。
    // (公式実装のように 142 フレームずつ並べると、窓の間隔 141.27 フレームとの差で 1 窓ごとに約 8.5ms ずつ遅れていく)
    let last = g;
    for (let f = BP_EDGE; f < BP_FRAMES - BP_EDGE; f++) {
      const G = Math.round((s + f * BP_HOP) / BP_HOP);
      if (G < 0 || G >= total) continue;
      note.set(r.note.subarray(f * BP_NOTES, (f + 1) * BP_NOTES), G * BP_NOTES);
      onset.set(r.onset.subarray(f * BP_NOTES, (f + 1) * BP_NOTES), G * BP_NOTES);
      last = G + 1;
    }
    g = Math.max(g + 1, last);
  }
  return { note, onset, frames: total };
}
