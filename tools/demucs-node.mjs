// 開発用: Node (onnxruntime-node) で Demucs を動かす + JS の前後処理の検算
//   node tools/demucs-node.mjs --check    (tools/export-demucs.py が作った検算用データと比べる)
import fs from 'node:fs';
import ort from 'onnxruntime-node';
import { demucsSpec, demucsISpec, DEMUCS_L, DEMUCS_T } from '../worklet/demucs.js';

let sess = null;
export async function loadDemucs(file = 'models/htdemucs.onnx') {
  sess = await ort.InferenceSession.create(file, { intraOpNumThreads: 8 });
}
// l, r: 44.1kHz の DEMUCS_L サンプル → [音][ch]
export async function runDemucs(l, r, want) {
  if (!sess) await loadDemucs();
  const mix = new Float32Array(2 * DEMUCS_L);
  mix.set(l, 0); mix.set(r, DEMUCS_L);
  const t0 = performance.now();
  const spec = demucsSpec(l, r);
  const t1 = performance.now();
  const o = await sess.run({ mix: new ort.Tensor('float32', mix, [1, 2, DEMUCS_L]), spec: new ort.Tensor('float32', spec, [1, 4, 2048, DEMUCS_T]) });
  const t2 = performance.now();
  const out = demucsISpec(o.zs.data, o.xt.data, want);
  const t3 = performance.now();
  return { out, ms: { spec: t1 - t0, net: t2 - t1, ispec: t3 - t2 } };
}

function readNpy(file) {
  const b = fs.readFileSync(file);
  const hl = b.readUInt16LE(8);
  const data = b.subarray(10 + hl);
  return new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
}

if (process.argv.includes('--check')) {
  const mix = readNpy('models/htdemucs.onnx.test-mix.npy');
  const ref = readNpy('models/htdemucs.onnx.test-out.npy'); // [1, 4, 2, L]
  const l = mix.subarray(0, DEMUCS_L), r = mix.subarray(DEMUCS_L);
  await loadDemucs();
  for (let k = 0; k < 2; k++) {
    const { out, ms } = await runDemucs(l, r);
    let err = 0, mx = 0;
    for (let s = 0; s < 4; s++) for (let c = 0; c < 2; c++) {
      const o = (s * 2 + c) * DEMUCS_L;
      for (let i = 0; i < DEMUCS_L; i++) { err = Math.max(err, Math.abs(out[s][c][i] - ref[o + i])); mx = Math.max(mx, Math.abs(ref[o + i])); }
    }
    console.log(`誤差 最大 ${err.toExponential(2)} (出力の最大 ${mx.toFixed(3)}), 時間: STFT ${ms.spec.toFixed(0)}ms, ネットワーク ${ms.net.toFixed(0)}ms, 逆 STFT ${ms.ispec.toFixed(0)}ms`);
  }
  process.exit(0);
}
