// Node で Basic Pitch を実行する (onnxruntime-node)
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { BP_WIN, BP_OUTPUTS, basicPitchFull } from '../worklet/basicpitch.js';

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
let session = null;

export async function runBasicPitchWindow(win) {
  if (!session) {
    const ort = require('onnxruntime-node');
    const modelPath = path.join(here, '..', 'models', 'basic-pitch.onnx');
    for (const ep of ['cuda', 'dml', 'coreml', 'cpu']) {
      try {
        session = await ort.InferenceSession.create(modelPath, { executionProviders: [ep], graphOptimizationLevel: 'all' });
        session.ort = ort;
        session.ep = ep;
        break;
      } catch {}
    }
    if (!session) {
      session = await ort.InferenceSession.create(modelPath);
      session.ort = ort;
      session.ep = 'cpu';
    }
  }
  const out = await session.run({ [session.inputNames[0]]: new session.ort.Tensor('float32', win, [1, BP_WIN, 1]) });
  return { note: out[BP_OUTPUTS.note].data, onset: out[BP_OUTPUTS.onset].data };
}

// audio22k: Float32Array (22050Hz モノラル) → { note, onset, frames, ms }
export async function runBasicPitch(audio22k) {
  const t0 = performance.now();
  const r = await basicPitchFull(audio22k, runBasicPitchWindow);
  return { ...r, ms: performance.now() - t0 };
}
