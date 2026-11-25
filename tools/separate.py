# 開発用: Demucs で曲を 4 つの音 (歌・ベース・ドラム・その他) に分ける
#   .tools/venv/Scripts/python.exe tools/separate.py <音声ファイル> <出力フォルダ> [開始秒] [長さ秒] [モデル名]
# 出力: <出力フォルダ>/vocals.wav, bass.wav, drums.wav, other.wav (48kHz ステレオ)
# 音声の読み書きは ffmpeg / soundfile で行う (torchaudio の版に依存しないため)
import os, subprocess, sys, time

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

src, out = sys.argv[1], sys.argv[2]
start = sys.argv[3] if len(sys.argv) > 3 else '0'
dur = sys.argv[4] if len(sys.argv) > 4 else '99999'
name = sys.argv[5] if len(sys.argv) > 5 else 'htdemucs'
os.makedirs(out, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4)

model = get_model(name)
model.eval()
sr = model.samplerate  # 44100
raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', start, '-t', dur, '-i', src, '-f', 'f32le', '-ac', '2', '-ar', str(sr), '-'],
                     capture_output=True, check=True).stdout
wav = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy())
ref = wav.mean(0)
mean, std = ref.mean(), ref.std() + 1e-8
t0 = time.time()
with torch.no_grad():
    stems = apply_model(model, ((wav - mean) / std)[None], device='cpu', shifts=1, split=True, overlap=0.25, progress=False)[0]
stems = stems * std + mean
el = time.time() - t0
print(f'{name}: {wav.shape[1] / sr:.1f} 秒の音を {el:.1f} 秒で分離 (実時間の {el / (wav.shape[1] / sr):.2f} 倍)')
for i, s in enumerate(model.sources):
    tmp = os.path.join(out, s + '_44k.wav')
    sf.write(tmp, stems[i].T.numpy(), sr, subtype='FLOAT')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-ar', '48000', os.path.join(out, s + '.wav')], check=True)
    os.remove(tmp)
print('出力:', out)
