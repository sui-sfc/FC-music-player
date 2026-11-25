# 開発用: Demucs (htdemucs) を ONNX に書き出す (ブラウザ / ラズパイの onnxruntime で動かすため)
#   .tools/venv/Scripts/python.exe tools/export-demucs.py [出力パス] [モデル名] [番号]
#   htdemucs_ft は音ごとに専用のモデルが 4 つある (番号 0=drums, 1=bass, 2=other, 3=vocals。それぞれ自分の音だけを使う)
# STFT / 逆 STFT は ONNX にしにくいので外に出し、ネットワーク本体だけを書き出す:
#   入力  mix [1, 2, L] (44.1kHz ステレオ, L = 343980 = 7.8 秒), spec [1, 4, 2048, 336] (STFT の実部・虚部)
#   出力  zs [1, 4, 4, 2048, 336] (音ごとの STFT, 正規化を戻した後), xt [1, 4, 2, L] (時間領域の出力)
#   最終出力 = iSTFT(zs) + xt  (前後処理は worklet/demucs.js)
import sys

import numpy as np
import torch
from demucs.pretrained import get_model

out = sys.argv[1] if len(sys.argv) > 1 else 'models/htdemucs.onnx'
name = sys.argv[2] if len(sys.argv) > 2 else 'htdemucs'
idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
bag = get_model(name)
print('モデル', name, len(bag.models), '個, 重み', bag.weights)
m = bag.models[idx].eval()
L = int(m.segment * m.samplerate)


class Core(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, mix, mag):
        m = self.m
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)
        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = m.channel_upsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = m.channel_downsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = m.channel_downsampler_t(xt)
        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)
        S = len(m.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        xt = xt.view(B, S, -1, mix.shape[-1])
        xt = xt * stdt[:, None] + meant[:, None]
        return x, xt


core = Core(m).eval()
torch.manual_seed(0)
mix = torch.randn(1, 2, L) * 0.1
with torch.no_grad():
    z = m._spec(mix)
    mag = m._magnitude(z)
    ref = m(mix)
    x, xt = core(mix, mag)
    # 検算: 本体 + 外の後処理 = 元のモデル
    full = m._ispec(m._mask(z, x), L) + xt
    print('PyTorch 分割の誤差:', float((full - ref).abs().max()))
torch.onnx.export(core, (mix, mag), out, input_names=['mix', 'spec'], output_names=['zs', 'xt'], opset_version=17, dynamo=False)
print('書き出し:', out, 'L =', L, 'spec', tuple(mag.shape), 'sources', m.sources)

import onnxruntime as ort
s = ort.InferenceSession(out)
import time
t0 = time.time()
ox, oxt = s.run(None, {'mix': mix.numpy(), 'spec': mag.numpy()})
print(f'onnxruntime: {time.time() - t0:.2f} 秒 / 7.8 秒の窓, 誤差 zs {np.abs(ox - x.numpy()).max():.2e} xt {np.abs(oxt - xt.numpy()).max():.2e}')
np.save(out + '.test-mix.npy', mix.numpy())
np.save(out + '.test-out.npy', ref.numpy())
