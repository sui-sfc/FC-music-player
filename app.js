import { buildDrumBank } from './worklet/dpcm.js';
import { arrangeFullSong } from './worklet/midi-arranger.js';
import { exportMidi } from './worklet/midi-export.js';
// メインスレッド: 入力ソースの管理, UI, 可視化, レジスタログ記録
const $ = (id) => document.getElementById(id);
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const noteName = (n) => (n < 0 ? '--' : NOTE_NAMES[n % 12] + (Math.floor(n / 12) - 1));

let ctx, fcNode, bus, chipGain, origGain, delay;
let sepDelay = 0; // 音源分離モードの出力の遅れ (秒)
let source = null; // { kind, node, stop() }
let fileBuf = null, fileLabel = '', fileStart = 0, fileOffset = 0, filePlaying = false, fileSrc = null;

// フル曲・高精度 MIDI 変換モードの状態
let hqMode = false, hqResult = null, hqAudioBuf = null, hqCache = null;
let hqChipSrc = null, hqOrigSrc = null, hqFrameIdx = -1, hqBusy = false;

async function ensureAudio() {
  if (ctx) { if (ctx.state === 'suspended') await ctx.resume(); return; }
  ctx = new AudioContext({ latencyHint: 'interactive' });
  await ctx.audioWorklet.addModule('worklet/processor.js');
  fcNode = new AudioWorkletNode(ctx, 'fc-processor', {
    numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2],
    // ステレオで受け取る (モノラル入力は左右同じ値に展開される)
    channelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'speakers',
  });
  bus = ctx.createGain();
  chipGain = ctx.createGain();
  origGain = ctx.createGain();
  // 解析窓の中心ぶん原音を遅らせて、ファミコン音とタイミングを合わせる
  delay = ctx.createDelay(15); // 音源分離モードでは数秒遅れる
  updateDelay();
  bus.connect(fcNode);
  bus.connect(delay).connect(origGain).connect(ctx.destination);
  fcNode.connect(chipGain).connect(ctx.destination);
  fcNode.port.onmessage = (e) => onFrame(e.data);
  applyVolumes();
  sendParams();
}

// 原音を「解析窓の中心 + 先読み」ぶん遅らせて、ファミコン音とタイミングを合わせる
function updateDelay() {
  if (!delay) return;
  delay.delayTime.value = (ctx.sampleRate > 60000 ? 8192 : 4096) / ctx.sampleRate + +$('lookahead').value / 60.0988 + sepDelay;
}

function stopAudioNodes() {
  for (const n of [fileSrc, hqChipSrc, hqOrigSrc]) {
    if (!n) continue;
    n.onended = null;
    try { n.stop(); } catch {}
    try { n.disconnect(); } catch {}
  }
  fileSrc = null; hqChipSrc = null; hqOrigSrc = null;
  filePlaying = false;
}

function stopSource() {
  stopAudioNodes();
  fileOffset = 0;
  if (source && source.stop) source.stop();
  source = null;
  $('play').textContent = '▶';
  fcNode && fcNode.port.postMessage({ type: 'reset' });
}

// ---------------- ファイル再生 ----------------
function playFile(offset) {
  stopAudioNodes();
  if (hqMode && hqAudioBuf) {
    // フル曲変換済み: 2A03 音声と原音を遅延 0 で完全同期再生
    hqChipSrc = ctx.createBufferSource();
    hqChipSrc.buffer = hqAudioBuf;
    hqChipSrc.connect(chipGain);
    hqOrigSrc = ctx.createBufferSource();
    hqOrigSrc.buffer = fileBuf;
    hqOrigSrc.connect(origGain);
    const t0 = ctx.currentTime + 0.02;
    hqChipSrc.onended = () => { filePlaying = false; fileOffset = 0; $('play').textContent = '▶'; };
    hqChipSrc.start(t0, offset);
    hqOrigSrc.start(t0, offset);
    fileStart = t0 - offset;
    filePlaying = true;
    hqFrameIdx = -1;
    $('play').textContent = '❚❚';
    return;
  }
  fileSrc = ctx.createBufferSource();
  fileSrc.buffer = fileBuf;
  fileSrc.connect(bus);
  fileSrc.onended = () => { filePlaying = false; fileOffset = 0; $('play').textContent = '▶'; };
  fileSrc.start(0, offset);
  fileStart = ctx.currentTime - offset;
  filePlaying = true;
  $('play').textContent = '❚❚';
}
function pauseFile() {
  if (!filePlaying) return;
  fileOffset = Math.max(0, ctx.currentTime - fileStart);
  stopAudioNodes();
  $('play').textContent = '▶';
}
function updateExportButtons() {
  $('saveMidi').disabled = !hqResult;
  $('saveWav').disabled = !hqResult;
  $('save').disabled = recording || (!recFrames.length && !hqResult);
}
async function loadBuffer(buf, label) {
  stopSource();
  hqMode = false; hqResult = null; hqAudioBuf = null; hqCache = null; hqFrameIdx = -1;
  fileBuf = buf; fileLabel = label; fileOffset = 0;
  source = { kind: 'file' };
  $('play').disabled = false;
  $('hqRun').disabled = false;
  updateExportButtons();
  if (origGain && +$('origVol').value === 0) { /* 原音はユーザーが上げたときだけ */ }
  setStatus(`${label} (${fmt(buf.duration)})`);
  playFile(0);
}

$('file').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  await ensureAudio();
  setStatus('読み込み中…');
  try {
    const buf = await ctx.decodeAudioData(await f.arrayBuffer());
    await loadBuffer(buf, f.name);
  } catch (err) { setStatus('読み込み失敗: ' + err.message); }
  e.target.value = '';
});

// 開発用: URL の音声を読み込んで再生 (コンソールから fcDev.loadUrl('dev-out/orig.wav'))
window.fcDev = {
  async loadUrl(url) {
    await ensureAudio();
    const buf = await ctx.decodeAudioData(await (await fetch(url)).arrayBuffer());
    window.fcFrames = [];
    await loadBuffer(buf, url);
  },
  // 開発用: 画面表示の状態 (表示待ちのフレーム数, 最後に表示したフレームの音の時刻と今の時刻の差)
  mon() { return { waiting: waiting.length, lag: lastInfo && ctx ? ctx.currentTime - (ctx.outputLatency || ctx.baseLatency || 0) - lastInfo.at : null }; },
  debugPitch(on) { window.fcPitch = []; rmvpeWorker?.postMessage({ type: 'debug', on }); },
};

$('demo').addEventListener('click', async () => {
  await ensureAudio();
  setStatus('テスト音源を生成中…');
  await loadBuffer(await makeDemo(ctx.sampleRate), 'テスト音源');
});
$('demoVocal').addEventListener('click', async () => {
  await ensureAudio();
  setStatus('テスト音源(歌)を生成中…');
  await loadBuffer(await makeVocalDemo(ctx.sampleRate), 'テスト音源(歌): 中央=歌, 左=コード, 右=高音のシンセ');
});

$('play').addEventListener('click', async () => {
  await ensureAudio();
  if (!fileBuf) return;
  if (filePlaying) pauseFile();
  else playFile(fileOffset);
});

$('seek').addEventListener('click', (e) => {
  if (!fileBuf) return;
  const r = e.currentTarget.getBoundingClientRect();
  const t = ((e.clientX - r.left) / r.width) * fileBuf.duration;
  fileOffset = t;
  if (filePlaying) playFile(t);
});

// ---------------- マイク / タブ音声 ----------------
async function useStream(stream, kind, label) {
  stopSource();
  hqMode = false; hqResult = null; hqAudioBuf = null; hqCache = null; hqFrameIdx = -1;
  fileBuf = null; $('play').disabled = true; $('hqRun').disabled = true;
  updateExportButtons();
  const node = ctx.createMediaStreamSource(stream);
  node.connect(bus);
  source = {
    kind, node,
    stop() { node.disconnect(); stream.getTracks().forEach((t) => t.stop()); },
  };
  stream.getAudioTracks()[0]?.addEventListener('ended', () => { if (source?.node === node) { stopSource(); setStatus('共有が終了しました'); } });
  setStatus(label);
}

$('mic').addEventListener('click', async () => {
  await ensureAudio();
  try {
    const s = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    $('origVol').value = 0; applyVolumes(); // ハウリング防止
    await useStream(s, 'mic', 'マイク入力中 (ヘッドホン推奨)');
  } catch (err) { setStatus('マイクを使えません: ' + err.message); }
});

$('tab').addEventListener('click', async () => {
  await ensureAudio();
  try {
    const s = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, suppressLocalAudioPlayback: true },
      systemAudio: 'include',
    });
    if (!s.getAudioTracks().length) {
      s.getTracks().forEach((t) => t.stop());
      setStatus('音声が共有されませんでした。「音声を共有する」にチェックを入れてください');
      return;
    }
    await useStream(s, 'tab', 'タブ/システム音声を変換中 (原音スライダーで元の音も聴けます)');
  } catch (err) { setStatus('キャンセルされました: ' + err.message); }
});

$('stop').addEventListener('click', () => { stopSource(); setStatus('停止しました'); });

// ---------------- パラメータ ----------------
function applyVolumes() {
  if (!ctx) return;
  chipGain.gain.value = +$('chipVol').value;
  origGain.gain.value = +$('origVol').value;
}
$('chipVol').addEventListener('input', applyVolumes);
$('origVol').addEventListener('input', applyVolumes);

function getParams() {
  const mute = [0, 1, 2, 3].map((c) => !document.querySelector(`.mute[data-ch="${c}"]`).checked);
  return {
    midiMode: $('midiMode').checked,
    duty1: +$('duty1').value, duty2: +$('duty2').value, bassOctave: +$('bassOctave').value, bassPattern: $('bassPattern').value,
    mute, p2mode: $('p2mode').value, arpSpeed: +$('arpSpeed').value, arpRate: $('arpRate').value, arpPattern: $('arpPattern').value,
    attackBend: +$('attackBend').value, dutyEnv: $('dutyEnv').value, release: +$('release').value, harmVol: +$('harmVol').value, instCost: +$('instCost').value, kickTri: $('kickTri').checked, drumMode: $('drumMode').value,
    melOctave: $('melOctave').value, melQuant: $('melQuant').checked, lead: $('lead').checked,
    pitchMode: $('pitchMode').value, accent: +$('accent').value, decay: +$('decay').value, chordCost: +$('chordCost').value, bassCost: +$('bassCost').value,
    hold: +$('hold').value, vocal: $('vocal').checked, voiceThr: +$('voiceThr').value, lookahead: +$('lookahead').value, noteCost: +$('noteCost').value, sens: +$('sens').value, drumSens: +$('drumSens').value, drumVol: +$('drumVol').value,
  };
}
function sendParams() {
  if (!fcNode) return;
  updateDelay();
  fcNode.port.postMessage({ type: 'params', params: getParams() });
}
document.querySelectorAll('.grid input, .grid select').forEach((el) => {
  if (el.id !== 'chipVol' && el.id !== 'origVol' && el.id !== 'hqShifts') el.addEventListener('input', sendParams);
});

// ---------------- AI 解析 (RMVPE + Basic Pitch) ----------------
// モデルは Web Worker で動かし、AudioWorklet と MessagePort で直結する
// (16kHz 音声の窓 → Worker → 音程 → AudioWorklet。メインスレッドは経由しない)
let rmvpeWorker = null, rmvpeReadyP = null, rmvpeFullPend = null;
const rmvpeStatus = (s) => { $('rmvpeStatus').textContent = s; };
async function startRmvpe() {
  await ensureAudio();
  rmvpeStatus('読み込み中…');
  const w = new Worker('rmvpe-worker.js', { type: 'module' });
  rmvpeWorker = w;
  let resReady, rejReady;
  rmvpeReadyP = new Promise((res, rej) => { resReady = res; rejReady = rej; });
  w.onmessage = (e) => {
    const m = e.data;
    if (w !== rmvpeWorker) return;
    if (m.type === 'progress') {
      rmvpeStatus(m.compiling ? '準備中…' : `読み込み中 ${(m.loaded / 1e6).toFixed(0)}/${(m.total / 1e6).toFixed(0)} MB`);
    } else if (m.type === 'ready') {
      const ch = new MessageChannel();
      w.postMessage({ type: 'port', port: ch.port1 }, [ch.port1]);
      fcNode.port.postMessage({ type: 'aiPort', port: ch.port2 }, [ch.port2]);
      rmvpeStatus(`ON (${m.threads} スレッド)`);
      // 推論窓の端 (未来の文脈が少ない部分) を使わずに済むよう、先読みを 1 秒以上にする
      if (+$('lookahead').value < 60) { $('lookahead').value = 60; sendParams(); }
      resReady && resReady(m); resReady = null;
    } else if (m.type === 'fullProgress') {
      rmvpeFullPend?.onProgress?.(m.p, m.label);
    } else if (m.type === 'fullResult') {
      const p = rmvpeFullPend; rmvpeFullPend = null; p?.resolve(m);
    } else if (m.type === 'stat') {
      rmvpeStatus(`ON: ${m.sec} 秒を ${m.ms.toFixed(0)}ms で推論`);
    } else if (m.type === 'debug') {
      (window.fcPitch ||= []).push(m);
    } else if (m.type === 'error') {
      rmvpeStatus('エラー: ' + m.message);
      rejReady && rejReady(new Error(m.message)); rejReady = null;
      const p = rmvpeFullPend; rmvpeFullPend = null; p?.reject(new Error(m.message));
    }
  };
  w.onerror = (e) => {
    const msg = e.message || 'Worker を起動できません';
    rmvpeStatus('エラー: ' + msg);
    rejReady && rejReady(new Error(msg)); rejReady = null;
    const p = rmvpeFullPend; rmvpeFullPend = null; p?.reject(new Error(msg));
  };
  w.postMessage({ type: 'load', url: 'models/rmvpe.onnx' });
}
function stopRmvpe() {
  if (fcNode) fcNode.port.postMessage({ type: 'aiOff' });
  if (rmvpeWorker) rmvpeWorker.terminate();
  rmvpeWorker = null; rmvpeReadyP = null;
  if (rmvpeFullPend) { rmvpeFullPend.reject(new Error('AI 解析が停止されました')); rmvpeFullPend = null; }
  rmvpeStatus('');
}
$('rmvpe').addEventListener('change', (e) => {
  if (e.target.checked && $('demucs').checked) { $('demucs').checked = false; stopDemucs(); }
  e.target.checked ? startRmvpe() : stopRmvpe();
});

// ---------------- 音源分離 (Demucs) + AI 解析 ----------------
// 曲を 歌・ベース・ドラム・その他 に分けてから解析する。分離を待つぶん出力は数秒遅れる
let demucsWorker = null, demucsReadyP = null, demucsFullPend = null, sepTimer = 0;
const sepStatus = (s) => { $('demucsStatus').textContent = s; };
let sepInfo = '';
async function startDemucs() {
  await ensureAudio();
  sepStatus('読み込み中…');
  const w = new Worker('demucs-worker.js', { type: 'module' });
  demucsWorker = w;
  let resReady, rejReady;
  demucsReadyP = new Promise((res, rej) => { resReady = res; rejReady = rej; });
  w.onmessage = (e) => {
    const m = e.data;
    if (w !== demucsWorker) return;
    if (m.type === 'progress') {
      sepStatus(m.compiling ? '準備中…' : `${m.label} 読み込み中 ${(m.loaded / 1e6).toFixed(0)}/${(m.total / 1e6).toFixed(0)} MB`);
    } else if (m.type === 'ready') {
      const ch = new MessageChannel();
      w.postMessage({ type: 'port', port: ch.port1 }, [ch.port1]);
      // 分離の間隔 + 余白 + 処理時間の余裕 (足りなければ AudioWorklet 側で自動的に延ばす)
      sepDelay = m.delay + (m.ep !== 'webgpu' ? 2.5 : m.model === 'ft' ? 1.5 : 0.8);
      fcNode.port.postMessage({ type: 'stemPort', port: ch.port2, delay: sepDelay }, [ch.port2]);
      sepInfo = (m.model === 'ft' ? '高精度, ' : '') + (m.ep === 'webgpu' ? 'GPU' : `CPU ${m.threads} スレッド`);
      sepStatus(`ON (${sepInfo})`);
      if (+$('lookahead').value < 60) $('lookahead').value = 60;
      sendParams();
      clearInterval(sepTimer);
      sepTimer = setInterval(() => fcNode && fcNode.port.postMessage({ type: 'stemStat' }), 1000);
      resReady && resReady(m); resReady = null;
    } else if (m.type === 'fullProgress') {
      demucsFullPend?.onProgress?.(m.p, m.label);
    } else if (m.type === 'fullResult') {
      const p = demucsFullPend; demucsFullPend = null; p?.resolve(m);
    } else if (m.type === 'stat') {
      window.fcSepStat = m;
    } else if (m.type === 'error') {
      sepStatus('エラー: ' + m.message);
      rejReady && rejReady(new Error(m.message)); rejReady = null;
      const p = demucsFullPend; demucsFullPend = null; p?.reject(new Error(m.message));
    }
  };
  w.onerror = (e) => {
    const msg = e.message || 'Worker を起動できません';
    sepStatus('エラー: ' + msg);
    rejReady && rejReady(new Error(msg)); rejReady = null;
    const p = demucsFullPend; demucsFullPend = null; p?.reject(new Error(msg));
  };
  w.postMessage({ type: 'load', model: $('demucsModel').value });
}
function onStemStat(m) {
  window.fcStemStat = m;
  if (Math.abs(m.delay - sepDelay) > 0.01) { sepDelay = m.delay; updateDelay(); }
  const st = window.fcSepStat;
  sepStatus(`ON (${sepInfo}): 遅れ ${m.delay.toFixed(1)} 秒${st ? `, 分離 ${st.sec.toFixed(0)} 秒ぶんを ${st.ms.toFixed(0)}ms` : ''}`);
}
function stopDemucs() {
  clearInterval(sepTimer);
  if (fcNode) fcNode.port.postMessage({ type: 'stemOff' });
  if (demucsWorker) demucsWorker.terminate();
  demucsWorker = null; demucsReadyP = null;
  if (demucsFullPend) { demucsFullPend.reject(new Error('音源分離が停止されました')); demucsFullPend = null; }
  sepDelay = 0;
  updateDelay();
  sepStatus('');
}
$('demucsModel').addEventListener('change', () => { if ($('demucs').checked) { stopDemucs(); startDemucs(); } });
$('demucs').addEventListener('change', (e) => {
  if (e.target.checked && $('rmvpe').checked) { $('rmvpe').checked = false; stopRmvpe(); }
  e.target.checked ? startDemucs() : stopDemucs();
});

// ---------------- 高精度フル曲 MIDI 変換 (2 パス一括 + GPU) ----------------
async function runHqArrange() {
  if (!fileBuf || hqBusy) return;
  await ensureAudio();
  hqBusy = true;
  $('hqRun').disabled = true;
  pauseFile();
  fcNode && fcNode.port.postMessage({ type: 'reset' });

  const sr = fileBuf.sampleRate;
  const L = fileBuf.getChannelData(0);
  const R = fileBuf.numberOfChannels > 1 ? fileBuf.getChannelData(1) : L;
  const mode = $('demucs').checked ? 'demucs' : $('rmvpe').checked ? 'rmvpe' : 'builtin';
  const model = $('demucsModel').value;
  const shifts = +$('hqShifts').value;
  const t0 = performance.now();

  try {
    let stems = null, rmvpe = null, bassRmvpe = null, bp = null;
    const hit = hqCache && hqCache.buf === fileBuf && hqCache.mode === mode &&
      (mode !== 'demucs' || (hqCache.model === model && hqCache.shifts === shifts));

    if (hit) {
      ({ stems, rmvpe, bassRmvpe, bp } = hqCache);
      setStatus('キャッシュ済みの分離・AI 解析結果を使って再アレンジ中…');
    } else if (mode === 'demucs') {
      if (!demucsWorker) startDemucs();
      setStatus('音源分離モデルの準備完了を待機中…');
      await demucsReadyP;
      const lCopy = L.slice(), rCopy = R.slice();
      const res = await new Promise((resolve, reject) => {
        demucsFullPend = {
          resolve, reject,
          onProgress: (p, label) => setStatus(`★ フル曲変換 (${Math.round(p * 100)}%): ${label}`),
        };
        demucsWorker.postMessage({ type: 'fullSong', L: lCopy, R: rCopy, sr, overlap: 0.5, shifts }, [lCopy.buffer, rCopy.buffer]);
      });
      stems = { vL: res.vL, vR: res.vR, d: res.d };
      rmvpe = res.rm; bassRmvpe = res.bf; bp = res.bp;
      hqCache = { buf: fileBuf, mode, model, shifts, stems, rmvpe, bassRmvpe, bp };
    } else if (mode === 'rmvpe') {
      if (!rmvpeWorker) startRmvpe();
      setStatus('AI モデルの準備完了を待機中…');
      await rmvpeReadyP;
      const lCopy = L.slice(), rCopy = R.slice();
      const res = await new Promise((resolve, reject) => {
        rmvpeFullPend = {
          resolve, reject,
          onProgress: (p, label) => setStatus(`★ フル曲変換 (${Math.round(p * 100)}%): ${label}`),
        };
        rmvpeWorker.postMessage({ type: 'fullSong', L: lCopy, R: rCopy, sr }, [lCopy.buffer, rCopy.buffer]);
      });
      rmvpe = res.rm; bp = res.bp;
      hqCache = { buf: fileBuf, mode, model, shifts, stems, rmvpe, bassRmvpe, bp };
    } else {
      hqCache = { buf: fileBuf, mode, model, shifts, stems, rmvpe, bassRmvpe, bp };
    }

    setStatus('★ 全曲ビート・キー・7thコード・16分MIDI編曲＆2A03合成中…');
    await new Promise((r) => setTimeout(r, 20));
    const res = arrangeFullSong({
      sr, L, R, stems, rmvpe, bassRmvpe, bp,
      params: getParams(),
      onProgress: (p, label) => { $('status').textContent = `★ MIDI 編曲 (${Math.round(p * 100)}%): ${label}`; },
    });

    const outBuf = ctx.createBuffer(2, res.pcm.length, sr);
    outBuf.getChannelData(0).set(res.pcm);
    outBuf.getChannelData(1).set(res.pcm);
    hqResult = res;
    hqAudioBuf = outBuf;
    hqMode = true;
    hqFrameIdx = -1;
    waiting = []; pending = [];
    g.fillStyle = '#07070c'; g.fillRect(0, 0, roll.width, roll.height);
    updateExportButtons();

    const sec = ((performance.now() - t0) / 1000).toFixed(1);
    const modeTag = mode === 'demucs' ? `GPU分離 ${model} shifts=${shifts}` : mode === 'rmvpe' ? 'AI解析' : '内蔵解析';
    setStatus(`★ フル曲MIDI完了 (${sec}s, ${modeTag}): ${fileLabel} | Key=${res.keyName} | BPM=${res.bpm.toFixed(1)} | ${res.numBars}小節 (原音と遅延0同期)`);
    $('recInfo').textContent = `フル曲変換済み: ${res.frames.length} フレーム (${(res.frames.length / 60.1).toFixed(1)} 秒) — MIDI / WAV / .bin 保存可`;
    playFile(0);
  } catch (err) {
    setStatus('フル曲変換エラー: ' + (err.message || err));
  } finally {
    hqBusy = false;
    $('hqRun').disabled = !fileBuf;
  }
}
$('hqRun').addEventListener('click', runHqArrange);

// ---------------- 書き出し / レジスタログ ----------------
// 1 フレーム = $4000-$4013 の 20 バイト + 書き込みマスク 1 バイト (README 参照)
const FRAME_BYTES = 21;
function download(name, data) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([data], { type: 'application/octet-stream' }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function encodeWav16(samples, sr) {
  const n = samples.length, buf = new ArrayBuffer(44 + n * 2), dv = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); dv.setUint32(4, 36 + n * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  str(36, 'data'); dv.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    dv.setInt16(44 + i * 2, s < 0 ? s * 32768 : s * 32767, true);
  }
  return new Uint8Array(buf);
}
const baseFileName = () => (fileLabel || 'fc-arrangement').replace(/\.[^/.]+$/, '').replace(/[\\/:*?"<>|]/g, '_');

$('saveMidi').addEventListener('click', () => {
  if (!hqResult) return;
  download(`${baseFileName()}.mid`, exportMidi(hqResult));
});
$('saveWav').addEventListener('click', () => {
  if (!hqResult || !fileBuf) return;
  download(`${baseFileName()}-fc.wav`, encodeWav16(hqResult.pcm, fileBuf.sampleRate));
});
// DPCM サンプルのバンク ($C000-$FFFF に置く 16KB)。実機ではカートリッジ側に置く
$('saveDpcm').addEventListener('click', () => download('dpcm-c000.bin', buildDrumBank().bank));
let recording = false, recFrames = [];
$('rec').addEventListener('click', () => {
  recording = !recording;
  if (recording) recFrames = [];
  $('rec').textContent = recording ? '■ 記録停止' : '● 記録開始';
  $('rec').classList.toggle('on', recording);
  updateExportButtons();
});
$('save').addEventListener('click', () => {
  if (recFrames.length) {
    const data = new Uint8Array(recFrames.length * FRAME_BYTES);
    recFrames.forEach((f, i) => data.set(f, i * FRAME_BYTES));
    download('fc-regs.bin', data);
  } else if (hqResult) {
    download(`${baseFileName()}-regs.bin`, hqResult.regLog);
  }
});

// ---------------- 可視化 ----------------
const roll = $('roll'), g = roll.getContext('2d');
const COLORS = ['#ff5a5a', '#ffb13b', '#3bc9ff', '#b8f05a'];
const LO = 28, HI = 96;
let pending = []; // 今回の描画で表示するフレーム
let waiting = []; // 届いたが、まだ音が出ていないフレーム (info.at = 音が出る時刻)
let lastInfo = null;

function onFrame(info) {
  if (info.type === 'stemStat') return onStemStat(info);
  if (hqMode) return;
  if (recording) {
    const f = new Uint8Array(FRAME_BYTES);
    f.set(info.regs, 0); // $4000-$4013
    f[20] = info.mask;
    recFrames.push(f);
  }
  waiting.push(info);
  if (waiting.length > 3000) waiting.splice(0, waiting.length - 3000);
  // デバッグ用: 直近 20 秒ぶんのフレームを保持
  (window.fcFrames ||= []).push(info);
  if (window.fcFrames.length > 4000) window.fcFrames.shift();
}

function draw() {
  if (hqMode && hqResult && ctx) {
    // フル曲変換モード: 再生位置に合わせて変換済みフレームを厳密に同期描画
    const lat = filePlaying ? (ctx.outputLatency || ctx.baseLatency || 0) : 0;
    const t = Math.max(0, (filePlaying ? ctx.currentTime - fileStart : fileOffset) - lat);
    const fi = Math.max(0, Math.min(hqResult.frames.length - 1, Math.floor(t * 60.0988)));
    if (fi !== hqFrameIdx) {
      if (fi < hqFrameIdx || fi - hqFrameIdx > 300) {
        g.fillStyle = '#07070c'; g.fillRect(0, 0, roll.width, roll.height);
        const start = Math.max(0, fi - Math.floor(roll.width / 2) + 1);
        for (let k = start; k <= fi; k++) pending.push(hqResult.frames[k]);
      } else {
        for (let k = hqFrameIdx + 1; k <= fi; k++) pending.push(hqResult.frames[k]);
      }
      lastInfo = hqResult.frames[fi];
      hqFrameIdx = fi;
    }
  } else if (ctx && waiting.length) {
    // 音が出た (スピーカーに届いた) フレームだけを表示に回す
    const now = ctx.currentTime - (ctx.outputLatency || ctx.baseLatency || 0);
    let k = 0;
    while (k < waiting.length && !(waiting[k].at > now)) k++;
    if (k) {
      const shown = waiting.splice(0, k);
      // 長く止まっていた (タブが裏にあった等) ときは直近のぶんだけ描く
      for (const f of shown.length > 600 ? shown.slice(-600) : shown) pending.push(f);
      (window.fcDrawCounts ||= []).push(shown.length);
      if (window.fcDrawCounts.length > 2000) window.fcDrawCounts.shift();
      lastInfo = shown[shown.length - 1];
    }
  }
  const W = roll.width, H = roll.height, laneH = H - 24;
  const n = pending.length;
  if (n) {
    const px = 2 * n;
    g.drawImage(roll, -px, 0);
    g.fillStyle = '#07070c';
    g.fillRect(W - px, 0, px, H);
    pending.forEach((f, i) => {
      const x = W - px + i * 2;
      const y = (note) => laneH - ((note - LO) / (HI - LO)) * laneH;
      const notes = [{ note: f.p1.pitch ?? f.p1.note, vol: f.p1.vol }, f.p2, { note: f.tri.note }];
      notes.forEach((v, c) => {
        if (v.note < 0) return;
        if (v.note === -2) { g.fillStyle = COLORS[2]; g.fillRect(x, laneH - 6, 2, 6); return; }
        g.globalAlpha = c === 2 ? 1 : 0.35 + (v.vol / 15) * 0.65;
        g.fillStyle = COLORS[c];
        g.fillRect(x, y(v.note) - 2, 2, 4);
      });
      g.globalAlpha = 1;
      if (f.noise.vol > 0) {
        g.globalAlpha = f.noise.vol / 15;
        g.fillStyle = COLORS[3];
        const row = { kick: 0, snare: 1, hat: 2 }[f.noise.drum] ?? 1;
        g.fillRect(x, laneH + 4 + row * 6, 2, 5);
        g.globalAlpha = 1;
      }
    });
    pending = [];
  }

  if (lastInfo) {
    const f = lastInfo;
    $('n0').textContent = noteName(f.p1.note);
    $('n1').textContent = noteName(f.p2.note);
    $('n2').textContent = f.tri.note === -2 ? 'KICK' : noteName(f.tri.note);
    $('n3').textContent = f.noise.vol ? f.noise.drum.toUpperCase() : '--';
    $('chord').textContent = f.chord && f.chord !== 'N' ? f.chord : '--';
    $('key').textContent = f.key || '--';
    $('bpm').textContent = f.bpm ? Math.round(f.bpm) : '--';
    $('section').textContent = ['静か', '普通', 'サビ'][f.energy] ?? '--';
    $('m0').style.setProperty('--v', (f.p1.vol / 15) * 100 + '%');
    $('m1').style.setProperty('--v', (f.p2.vol / 15) * 100 + '%');
    $('m3').style.setProperty('--v', (f.noise.vol / 15) * 100 + '%');
    const hex = (v) => v.toString(16).toUpperCase().padStart(2, '0');
    const r = f.regs;
    $('regs').textContent =
      `$4000: ${hex(r[0])} ${hex(r[1])} ${hex(r[2])} ${hex(r[3])}   矩形波1\n` +
      `$4004: ${hex(r[4])} ${hex(r[5])} ${hex(r[6])} ${hex(r[7])}   矩形波2\n` +
      `$4008: ${hex(r[8])} ${hex(r[9])} ${hex(r[10])} ${hex(r[11])}   三角波\n` +
      `$400C: ${hex(r[12])} ${hex(r[13])} ${hex(r[14])} ${hex(r[15])}   ノイズ\n` +
      `$4010: ${hex(r[16])} ${hex(r[17])} ${hex(r[18])} ${hex(r[19])}   DPCM`;
  }
  if (recording) $('recInfo').textContent = `${recFrames.length} フレーム (${(recFrames.length / 60.1).toFixed(1)} 秒, ${recFrames.length * FRAME_BYTES} バイト)`;
  else if (recFrames.length) $('recInfo').textContent = `記録済み: ${recFrames.length} フレーム`;

  if (fileBuf) {
    const t = Math.max(0, filePlaying ? ctx.currentTime - fileStart : fileOffset);
    $('seekbar').style.width = Math.min(100, (t / fileBuf.duration) * 100) + '%';
    $('time').textContent = `${fmt(t)} / ${fmt(fileBuf.duration)}`;
  } else $('time').textContent = '--:--';
  requestAnimationFrame(draw);
}
g.fillStyle = '#07070c';
g.fillRect(0, 0, roll.width, roll.height);
requestAnimationFrame(draw);

function setStatus(s) { $('status').textContent = s; }
function fmt(t) { t = Math.max(0, t); return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`; }

// ---------------- テスト音源 (普通の音色で作った曲) ----------------
async function makeDemo(sr) {
  const bpm = 120, beat = 60 / bpm, bars = 8, dur = bars * 4 * beat + 1;
  const oc = new OfflineAudioContext(2, Math.ceil(dur * sr), sr);
  const out = oc.createDynamicsCompressor();
  out.connect(oc.destination);
  const mf = (n) => 440 * Math.pow(2, (n - 69) / 12);

  function tone(note, t, len, type, vol, cutoff) {
    const o = oc.createOscillator(), f = oc.createBiquadFilter(), gn = oc.createGain();
    o.type = type; o.frequency.value = mf(note);
    f.type = 'lowpass'; f.frequency.value = cutoff;
    gn.gain.setValueAtTime(0, t);
    gn.gain.linearRampToValueAtTime(vol, t + 0.01);
    gn.gain.exponentialRampToValueAtTime(vol * 0.5, t + len * 0.8);
    gn.gain.linearRampToValueAtTime(0, t + len);
    o.connect(f).connect(gn).connect(out);
    o.start(t); o.stop(t + len + 0.02);
  }
  const noiseBuf = oc.createBuffer(1, sr, sr);
  const nd = noiseBuf.getChannelData(0);
  for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
  function noise(t, len, type, freq, vol) {
    const s = oc.createBufferSource(), f = oc.createBiquadFilter(), gn = oc.createGain();
    s.buffer = noiseBuf; f.type = type; f.frequency.value = freq;
    gn.gain.setValueAtTime(vol, t);
    gn.gain.exponentialRampToValueAtTime(0.001, t + len);
    s.connect(f).connect(gn).connect(out);
    s.start(t); s.stop(t + len);
  }
  function kick(t) {
    const o = oc.createOscillator(), gn = oc.createGain();
    o.frequency.setValueAtTime(150, t);
    o.frequency.exponentialRampToValueAtTime(45, t + 0.15);
    gn.gain.setValueAtTime(0.9, t);
    gn.gain.exponentialRampToValueAtTime(0.001, t + 0.3);
    o.connect(gn).connect(out);
    o.start(t); o.stop(t + 0.3);
  }

  const chords = [[60, 64, 67], [57, 60, 64], [53, 57, 60], [55, 59, 62]];
  const roots = [36, 33, 41, 43];
  const melody = [
    76, 74, 72, 74, 76, 76, 76, -1, 74, 74, 74, -1, 76, 79, 79, -1,
    76, 74, 72, 74, 76, 76, 76, 76, 74, 74, 76, 74, 72, -1, -1, -1,
  ];
  for (let bar = 0; bar < bars; bar++) {
    const t0 = 0.5 + bar * 4 * beat;
    const ci = bar % 4;
    for (const n of chords[ci]) tone(n, t0, 4 * beat - 0.05, 'triangle', 0.07, 2500);
    for (let e = 0; e < 8; e++) tone(roots[ci], t0 + e * beat / 2, beat / 2 - 0.04, 'sawtooth', 0.18, 600);
    for (let q = 0; q < 4; q++) {
      const m = melody[(bar * 4 + q) % melody.length];
      if (m > 0) tone(m, t0 + q * beat, beat - 0.05, 'sawtooth', 0.16, 3500);
      if (q % 2 === 0) kick(t0 + q * beat);
      else noise(t0 + q * beat, 0.18, 'bandpass', 2500, 0.6);
    }
    for (let e = 0; e < 8; e++) noise(t0 + e * beat / 2, 0.05, 'highpass', 8000, 0.25);
  }
  return oc.startRendering();
}

// ---------------- テスト音源(歌): 中央に歌っぽい声, 左右に伴奏 ----------------
// 声はノコギリ波をフォルマント (母音「あ」) で整形し、ビブラートとしゃくりを付けたもの。
// 右には歌より高く目立つシンセを置き、「一番高い音 = メロディ」では誤る状況を作っている。
async function makeVocalDemo(sr) {
  const bpm = 120, beat = 60 / bpm, bars = 8, dur = bars * 4 * beat + 1;
  const oc = new OfflineAudioContext(2, Math.ceil(dur * sr), sr);
  const out = oc.createDynamicsCompressor();
  out.connect(oc.destination);
  const mf = (n) => 440 * Math.pow(2, (n - 69) / 12);
  const pan = (p) => { const s = oc.createStereoPanner(); s.pan.value = p; s.connect(out); return s; };
  const left = pan(-0.85), right = pan(0.85);

  function tone(note, t, len, type, vol, cutoff, dest) {
    const o = oc.createOscillator(), f = oc.createBiquadFilter(), gn = oc.createGain();
    o.type = type; o.frequency.value = mf(note);
    f.type = 'lowpass'; f.frequency.value = cutoff;
    gn.gain.setValueAtTime(0, t);
    gn.gain.linearRampToValueAtTime(vol, t + 0.01);
    gn.gain.exponentialRampToValueAtTime(vol * 0.5, t + len * 0.8);
    gn.gain.linearRampToValueAtTime(0, t + len);
    o.connect(f).connect(gn).connect(dest);
    o.start(t); o.stop(t + len + 0.02);
  }

  // --- 声 ---
  const melody = [
    64, 64, 67, 69, 67, -1, 64, -1, 62, 62, 64, 67, 64, -1, -1, -1,
    64, 64, 67, 72, 71, 69, 67, -1, 65, 64, 62, 64, 60, -1, -1, -1,
  ];
  const t0 = 0.5;
  const vo = oc.createOscillator();
  vo.type = 'sawtooth';
  const lfo = oc.createOscillator(), lfoGain = oc.createGain();
  lfo.frequency.value = 5.5; lfoGain.gain.value = 35; // ±35 セントのビブラート
  lfo.connect(lfoGain).connect(vo.detune);
  const venv = oc.createGain();
  venv.gain.value = 0;
  for (const [fq, q, g] of [[750, 6, 1.0], [1200, 7, 0.55], [2600, 9, 0.3], [300, 1, 0.25]]) {
    const bp = oc.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = fq; bp.Q.value = q;
    const gg = oc.createGain(); gg.gain.value = g * 1.6;
    vo.connect(bp).connect(gg).connect(venv);
  }
  venv.connect(out);
  vo.frequency.setValueAtTime(mf(64), 0);
  melody.forEach((n, i) => {
    const t = t0 + i * beat;
    if (n < 0) { venv.gain.setTargetAtTime(0, t - 0.05, 0.03); return; }
    vo.frequency.setValueAtTime(mf(n) * 0.97, t); // しゃくり (少し下から入る)
    vo.frequency.setTargetAtTime(mf(n), t, 0.04);
    const prevRest = i === 0 || melody[i - 1] < 0;
    if (prevRest) venv.gain.setTargetAtTime(0.5, t, 0.02);
    else { venv.gain.setTargetAtTime(0.3, t - 0.03, 0.01); venv.gain.setTargetAtTime(0.5, t, 0.02); }
  });
  venv.gain.setTargetAtTime(0, t0 + melody.length * beat, 0.05);
  vo.start(0); lfo.start(0); vo.stop(dur); lfo.stop(dur);

  // --- 伴奏 ---
  const chords = [[60, 64, 67], [57, 60, 64], [53, 57, 60], [55, 59, 62]];
  const roots = [36, 33, 41, 43];
  const hi = [[84, 88, 91], [81, 84, 88], [77, 81, 84], [79, 83, 86]];
  const noiseBuf = oc.createBuffer(1, sr, sr);
  const nd = noiseBuf.getChannelData(0);
  for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
  function noise(t, len, type, freq, vol) {
    const s = oc.createBufferSource(), f = oc.createBiquadFilter(), gn = oc.createGain();
    s.buffer = noiseBuf; f.type = type; f.frequency.value = freq;
    gn.gain.setValueAtTime(vol, t);
    gn.gain.exponentialRampToValueAtTime(0.001, t + len);
    s.connect(f).connect(gn).connect(out);
    s.start(t); s.stop(t + len);
  }
  function kick(t) {
    const o = oc.createOscillator(), gn = oc.createGain();
    o.frequency.setValueAtTime(150, t);
    o.frequency.exponentialRampToValueAtTime(45, t + 0.15);
    gn.gain.setValueAtTime(0.9, t);
    gn.gain.exponentialRampToValueAtTime(0.001, t + 0.3);
    o.connect(gn).connect(out);
    o.start(t); o.stop(t + 0.3);
  }
  for (let bar = 0; bar < bars; bar++) {
    const tb = t0 + bar * 4 * beat, ci = bar % 4;
    for (const n of chords[ci]) tone(n, tb, 4 * beat - 0.05, 'triangle', 0.09, 2500, left);
    for (let e = 0; e < 8; e++) {
      tone(roots[ci], tb + e * beat / 2, beat / 2 - 0.04, 'sawtooth', 0.16, 600, out);
      tone(hi[ci][e % 3], tb + e * beat / 2, beat / 2 - 0.06, 'square', 0.07, 5000, right);
    }
    for (let q = 0; q < 4; q++) {
      if (q % 2 === 0) kick(tb + q * beat);
      else noise(tb + q * beat, 0.18, 'bandpass', 2500, 0.5);
    }
  }
  return oc.startRendering();
}
