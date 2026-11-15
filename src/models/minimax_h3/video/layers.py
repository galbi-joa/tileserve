"""Attention and checkpoint-compatible feed-forward layers shared by H3 video models."""

import torch
from torch import nn
from torch.nn import functional as F


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def rotate(x, cos, sin):
    if x.is_cuda and x.dtype == torch.bfloat16:
        from tileserve_kernels.minimax_h3.rotary import partial_rotary

        return partial_rotary(x, cos, sin)
    part, rest = x.split((cos.shape[-1], x.shape[-1] - cos.shape[-1]), dim=-1)
    a, b = part.chunk(2, dim=-1)
    return torch.cat((part * cos.to(x.dtype)[:, None] + torch.cat((-b, a), dim=-1) * sin.to(x.dtype)[:, None],
                      rest), dim=-1)


class _SwiGLU(nn.Module):
    def __init__(self, width, inner, bias):
        super().__init__()
        self.proj = nn.Linear(width, 2 * inner, bias=bias)
        self.inner = inner

    def forward(self, x):
        packed = self.proj(x)
        if packed.is_cuda and packed.dtype == torch.bfloat16:
            from tileserve_kernels.minimax_h3.pointwise import packed_swiglu

            return packed_swiglu(self.inner)(packed.flatten(0, 1)).unflatten(0, x.shape[:2])
        value, gate = packed.chunk(2, dim=-1)
        return value * F.silu(gate)


class VideoFeedForward(nn.Module):
    def __init__(self, width, inner, *, bias):
        super().__init__()
        self.net = nn.Sequential(_SwiGLU(width, inner, bias), nn.Identity(), nn.Linear(inner, width, bias=bias))

    def forward(self, x):
        return self.net(x)
