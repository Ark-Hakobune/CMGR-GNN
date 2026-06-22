import numpy as np
import torch
from train import TrainConfig, build_batches_from_corr
from model import DualModalityMaskedEncoder

cfg = TrainConfig(device='cpu', use_signed=True)
B, N = 2, 16
fc = np.random.randn(B, N, N).astype(np.float32)
msn = np.random.randn(B, N, N).astype(np.float32)
fb, mb = build_batches_from_corr(fc, msn, cfg)
enc = DualModalityMaskedEncoder(in_channels_func=fb.x.size(1), in_channels_morph=mb.x.size(1), hidden_channels=32, num_layers=2, out_channels=64, dropout=0.0, share_backbone=False, use_signed=True)
enc.eval()
with torch.no_grad():
    zf, zm = enc(fb, mb)
    print('zf', zf.shape, 'zm', zm.shape)
print('OK')
