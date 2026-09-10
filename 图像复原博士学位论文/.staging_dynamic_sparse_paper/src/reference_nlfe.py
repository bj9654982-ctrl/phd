"""Reference-inspired dense-Q / max-pooled-KV operator, NOT official NLCNet.

Explicit prototype choices: scaled dot product, no positional bias or norm,
GELU FFN expansion 4, two residual additions. No IBT. Require exact windows
instead of silently padding features (padding can change max pooling).
"""
import torch
from torch import nn
from torch.nn import functional as F


class ReferenceNLFE(nn.Module):
    def __init__(self, channels, heads=1, window=64, pool=8):
        super().__init__()
        if min(channels, heads, window, pool) < 1:
            raise ValueError('dimensions must be positive')
        if channels % heads or window % pool:
            raise ValueError('channels/heads and window/pool must divide exactly')
        self.channels, self.heads = channels, heads
        self.window, self.pool = window, pool
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.ffn = nn.Sequential(nn.Conv2d(channels, channels * 4, 1),
                                 nn.GELU(), nn.Conv2d(channels * 4, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        s, p = self.window, self.pool
        if c != self.channels or h % s or w % s:
            raise ValueError('channel mismatch or non-divisible window; no implicit padding')
        tiles = x.reshape(b, c, h // s, s, w // s, s)
        tiles = tiles.permute(0, 2, 4, 1, 3, 5).reshape(-1, c, s, s)
        q, k, v = self.qkv(tiles).chunk(3, dim=1)
        k, v = F.max_pool2d(k, p, p), F.max_pool2d(v, p, p)
        def tokens(t):
            return t.reshape(-1, self.heads, c // self.heads,
                             t.shape[-2] * t.shape[-1]).transpose(-1, -2)
        q, k, v = map(tokens, (q, k, v))
        weights = (q @ k.transpose(-1, -2) / (c // self.heads) ** .5).softmax(-1)
        a = (weights @ v).transpose(-1, -2).reshape(-1, c, s, s)
        y = tiles + self.proj(a)
        y = y + self.ffn(y)
        return y.reshape(b, h // s, w // s, c, s, s).permute(
            0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
