// AudioWorklet: 変換エンジン (engine.js) をリアルタイムで動かす
import { Engine } from './engine.js';

class FCProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // 各フレームに「その音が出る時刻 (AudioContext の時計)」を付けて送る。
    // 画面はこの時刻に合わせて表示する (音源分離モードでは結果がまとめて・数秒早く届くため)
    this.n = 128;
    this.engine = new Engine(sampleRate, (info, e) => {
      if (e.stemPort) info.at = currentTime + (e.inCount + e.D - (e.tIn - this.n)) / sampleRate;
      else info.at = currentTime;
      this.port.postMessage(info);
    });
    this.buf = new Float32Array(128);
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'params') Object.assign(this.engine.driver.params, m.params);
      else if (m.type === 'reset') this.engine.driver.reset();
      else if (m.type === 'aiPort') this.engine.setAiPort(m.port); // AI 解析の Worker と直結
      else if (m.type === 'aiOff') this.engine.setAiPort(null);
      else if (m.type === 'stemPort') this.engine.setStemPort(m.port, m.delay); // 音源分離の Worker と直結
      else if (m.type === 'stemOff') this.engine.setStemPort(null);
      else if (m.type === 'stemStat') this.port.postMessage({ type: 'stemStat', delay: this.engine.D / sampleRate, underruns: this.engine.underruns, behind: (this.engine.tIn - this.engine.p) / sampleRate });
    };
  }

  process(inputs, outputs) {
    const inp = inputs[0];
    const out = outputs[0];
    const n = out[0].length;
    this.n = n;
    if (this.buf.length < n) this.buf = new Float32Array(n);
    const nch = inp ? inp.length : 0;
    if (this.engine.stemPort) this.engine.processLive(nch ? inp[0] : null, nch > 1 ? inp[1] : null, this.buf, n);
    else this.engine.process(nch ? inp[0] : null, nch > 1 ? inp[1] : null, this.buf, n);
    for (let c = 0; c < out.length; c++) out[c].set(this.buf.subarray(0, n));
    return true;
  }
}

registerProcessor('fc-processor', FCProcessor);
