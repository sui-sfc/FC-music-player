# 高精度音源分離: GPU (CUDA / MPS) を優先使用し、曲を 歌・ベース・ドラム・その他 に分離する
#   python tools/separate.py <音声ファイル> <出力フォルダ> [開始秒] [長さ秒] [モデル名]
#   追加オプション:
#     --device auto|cuda|mps|cpu   使用デバイス (既定: auto = GPU 優先)
#     --shifts N                   シフト平均回数 (既定: GPU=4, CPU=1。大きいほど高精度)
#     --overlap F                  窓の重なり率 (既定: GPU=0.5, CPU=0.25)
#     --roformer                   audio-separator 導入時に BS-RoFormer でボーカルを最高精度分離
#   モデル例: htdemucs_ft (4 音源・最高精度), htdemucs_6s (6 音源: guitar/piano 追加), htdemucs
import argparse, os, shutil, subprocess, sys, tempfile, time

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


def parse_args():
    p = argparse.ArgumentParser(description='GPU 対応・高精度音源分離 (Demucs / BS-RoFormer)')
    p.add_argument('src', help='入力音声ファイル')
    p.add_argument('out', help='出力フォルダ')
    p.add_argument('start', nargs='?', default='0', help='開始秒')
    p.add_argument('dur', nargs='?', default='99999', help='長さ秒')
    p.add_argument('model', nargs='?', default='htdemucs_ft', help='Demucs モデル名 (htdemucs_ft / htdemucs_6s / htdemucs)')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    p.add_argument('--shifts', type=int, default=0, help='シフト平均回数 (0 = GPU:4 / CPU:1)')
    p.add_argument('--overlap', type=float, default=0.0, help='窓の重なり率 (0 = GPU:0.5 / CPU:0.25)')
    p.add_argument('--roformer', action='store_true', help='BS-RoFormer でボーカルをさらに高精度分離 (要 audio-separator)')
    return p.parse_args()


def pick_device(want):
    if want != 'auto':
        return want
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def write_48k(path_out, audio_ch_first, sr):
    tmp = path_out + '.44k.wav'
    sf.write(tmp, audio_ch_first.T, sr, subtype='FLOAT')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-ar', '48000', path_out], check=True)
    os.remove(tmp)


def run_roformer_vocals(src_wav_path, out_dir):
    """audio-separator が入っていれば BS-RoFormer でボーカルを分離して vocals.wav を上書きする"""
    try:
        from audio_separator.separator import Separator
    except ImportError:
        print('[注意] --roformer を使うには `pip install audio-separator[gpu]` が必要です。Demucs のボーカルをそのまま使います。')
        return False
    tmp_dir = tempfile.mkdtemp(prefix='roformer_')
    try:
        sep = Separator(output_dir=tmp_dir, output_format='WAV', output_single_stem='Vocals')
        sep.load_model('model_bs_roformer_ep_317_sdr_12.9755.ckpt')
        files = sep.separate(src_wav_path)
        if not files:
            return False
        voc_file = os.path.join(tmp_dir, files[0]) if not os.path.isabs(files[0]) else files[0]
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', voc_file, '-ar', '48000', '-ac', '2', os.path.join(out_dir, 'vocals.wav')], check=True)
        print('BS-RoFormer でボーカル (vocals.wav) を高精度分離しました')
        return True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    if device == 'cpu':
        torch.set_num_threads(os.cpu_count() or 4)
    shifts = args.shifts if args.shifts > 0 else (4 if device != 'cpu' else 1)
    overlap = args.overlap if args.overlap > 0 else (0.5 if device != 'cpu' else 0.25)
    print(f'デバイス: {device} | モデル: {args.model} | shifts={shifts}, overlap={overlap}')

    model = get_model(args.model)
    model.eval()
    model.to(device)
    sr = model.samplerate  # 44100

    raw = subprocess.run(
        ['ffmpeg', '-v', 'error', '-ss', str(args.start), '-t', str(args.dur), '-i', args.src, '-f', 'f32le', '-ac', '2', '-ar', str(sr), '-'],
        capture_output=True, check=True,
    ).stdout
    wav = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy())
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8

    t0 = time.time()
    with torch.no_grad():
        stems = apply_model(
            model, ((wav - mean) / std)[None].to(device),
            device=device, shifts=shifts, split=True, overlap=overlap, progress=True,
        )[0].cpu()
    stems = stems * std + mean
    el = time.time() - t0
    dur_s = wav.shape[1] / sr
    print(f'{args.model} ({device}): {dur_s:.1f} 秒の音を {el:.1f} 秒で分離 (実時間の {el / max(dur_s, 1e-6):.2f} 倍)')

    stem_map = {s: stems[i].numpy() for i, s in enumerate(model.sources)}
    for s, arr in stem_map.items():
        write_48k(os.path.join(args.out, s + '.wav'), arr, sr)

    # 6 音源モデル (htdemucs_6s) のときは guitar + piano + other を合成した other_all.wav も出力
    if 'guitar' in stem_map or 'piano' in stem_map:
        merged = stem_map['other'].copy()
        if 'guitar' in stem_map:
            merged += stem_map['guitar']
        if 'piano' in stem_map:
            merged += stem_map['piano']
        write_48k(os.path.join(args.out, 'other.wav'), merged, sr)

    if args.roformer:
        tmp_mix = os.path.join(args.out, '_mix_44k.wav')
        sf.write(tmp_mix, wav.T.numpy(), sr, subtype='FLOAT')
        try:
            run_roformer_vocals(tmp_mix, args.out)
        finally:
            if os.path.exists(tmp_mix):
                os.remove(tmp_mix)

    print('出力:', args.out)


if __name__ == '__main__':
    main()
