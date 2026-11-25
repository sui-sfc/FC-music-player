// Node で RMVPE を実行する (onnxruntime-node)
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { melSpectrogram, decodePitch } from '../worklet/rmvpe.js';

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
export const MODEL_PATH = path.join(here, '..', 'models', 'rmvpe.onnx');

let session = null;
export async function loadRmvpe(modelPath = MODEL_PATH) {
  if (session) return session;
  const ort = require('onnxruntime-node');
  session = await ort.InferenceSession.create(modelPath, { graphOptimizationLevel: 'all' });
  session.ort = ort;
  return session;
}

// audio16k: Float32Array (16kHz モノラル) → { f0: Float32Array (Hz, 10ms ごと), conf, ms: 推論時間 }
export async function runRmvpe(audio16k) {
  const s = await loadRmvpe();
  const { data, T, Tpad } = melSpectrogram(audio16k);
  const input = new s.ort.Tensor('float32', data, [1, 128, Tpad]);
  const t0 = performance.now();
  const out = await s.run({ [s.inputNames[0]]: input });
  const ms = performance.now() - t0;
  const hidden = out[s.outputNames[0]].data;
  return { ...decodePitch(hidden, T), ms };
}
