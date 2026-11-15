"""Native FL2VA Omni Transformer over text, audio and video rows.

Architecture reference: MiniMaxH3Transformer3DModel in Apache-2.0 Diffusers
43e30028af5120d8b0868cb6975b9fad39cc4fab by MiniMax and Hugging Face.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import VideoFeedForward, rotate, sdpa


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        width = config.hidden_size
        inner = self.heads * config.attention_head_dim
        self.to_q = nn.Linear(width, inner, bias=False)
        self.to_k = nn.Linear(width, inner, bias=False)
        self.to_v = nn.Linear(width, inner, bias=False)
        self.norm_q = nn.RMSNorm(config.attention_head_dim, eps=config.qk_norm_eps)
        self.norm_k = nn.RMSNorm(config.attention_head_dim, eps=config.qk_norm_eps)
        self.to_out = nn.Sequential(nn.Linear(inner, width, bias=False))

    def forward(self, x, rotary=None):
        q, k, v = (linear(x).unflatten(-1, (self.heads, -1)) for linear in (self.to_q, self.to_k, self.to_v))
        q, k = self.norm_q(q), self.norm_k(k)
        if rotary is not None:
            q, k = rotate(q, *rotary), rotate(k, *rotary)
        return self.to_out(sdpa(q, k, v).flatten(2).type_as(q))


class _RefinerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = _Attention(config)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ff = VideoFeedForward(config.hidden_size, config.ffn_dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class _Refiner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.refiner_blocks = nn.Sequential(*(_RefinerBlock(config) for _ in range(config.num_refiner_layers)))
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)

    def forward(self, x):
        return self.final_norm(self.refiner_blocks(x))


class _Modulation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = config.hidden_size
        self.linear = nn.Linear(config.time_embed_dim, 18 * self.width)

    def forward(self, time_embedding):
        return self.linear(time_embedding.to(self.linear.weight.dtype)).view(-1, 6, self.width)


def _indexed_affine(normed, table, rows, offset):
    if normed.is_cuda and normed.dtype == torch.bfloat16:
        from tileserve_kernels.minimax_h3.dit_modulation import indexed_affine

        return indexed_affine(normed.shape[-1], table.shape[1], offset)(
            normed.flatten(0, 1), table, rows
        ).unflatten(0, normed.shape[:2])
    shift = table[:, offset].index_select(0, rows)
    scale = table[:, offset + 1].index_select(0, rows)
    return normed * (1 + scale) + shift


def _gated_residual(x, update, table, rows, offset):
    if x.is_cuda and x.dtype == torch.bfloat16:
        from tileserve_kernels.minimax_h3.pointwise import gated_residual

        return gated_residual(x.shape[-1], offset)(
            x.flatten(0, 1), update.flatten(0, 1), table, rows
        ).unflatten(0, x.shape[:2])
    return x + table[:, offset + 2].index_select(0, rows) * update


class _Block(_RefinerBlock):
    def __init__(self, config):
        super().__init__(config)
        self.adaln_proj = _Modulation(config)

    def forward(self, x, time_embedding, rows, rotary):
        table = self.adaln_proj(time_embedding)
        normed = _indexed_affine(self.norm1(x), table, rows, 0)
        x = _gated_residual(x, self.attn(normed, rotary), table, rows, 0)
        normed = _indexed_affine(self.norm2(x), table, rows, 3)
        return _gated_residual(x, self.ff(normed), table, rows, 3)


class _OutputNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)
        self.linear = nn.Linear(config.time_embed_dim, 2 * config.hidden_size)

    def forward(self, x, time_embedding, rows):
        table = self.linear(time_embedding.to(self.linear.weight.dtype)).view(-1, 2, x.shape[-1])
        return _indexed_affine(self.norm(x), table, rows, 0)


class _TimeEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.half = config.freq_dim // 2
        self.linear_1 = nn.Linear(config.freq_dim, config.time_embed_hidden_dim)
        self.linear_2 = nn.Linear(config.time_embed_hidden_dim, config.time_embed_dim)

    def forward(self, times):
        phases = times.float()[:, None] * (-math.log(10000) * torch.arange(
            self.half, device=times.device, dtype=torch.float32
        ) / self.half).exp()[None]
        return self.linear_2(F.silu(self.linear_1(torch.cat((phases.cos(), phases.sin()), dim=-1))))


class VideoDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        video_dim = config.in_channels * math.prod(config.patch_size)
        self.proj_in = nn.Linear(video_dim, width)
        self.audio_proj_in = nn.Linear(config.audio_in_channels, width)
        self.context_embedder = nn.Linear(config.text_dim, width)
        self.time_embedder = _TimeEmbedding(config)
        count = config.rope_freq_dim
        frequencies = 1 / (config.rope_theta ** (
            torch.arange(0, 2 * count, 2, dtype=torch.float32, device="cpu") / (2 * count)
        ))
        self.register_buffer("inv_freq", frequencies, persistent=False)
        self.token_refiner = _Refiner(config)
        self.transformer_blocks = nn.ModuleList(_Block(config) for _ in range(config.num_layers))
        self.norm_out = _OutputNorm(config)
        self.proj_out = nn.Linear(width, video_dim)
        self.audio_proj_out = nn.Linear(width, config.audio_in_channels)

    def prepare(self, prompt_embeds, positions):
        text = self.token_refiner(self.context_embedder(prompt_embeds.to(self.context_embedder.weight.dtype)))
        angles = (positions.float().unsqueeze(-1) * self.inv_freq).flatten(1)
        angles = torch.cat((angles, angles), dim=-1)
        return text, (angles.cos(), angles.sin())

    def forward(self, video, audio, times, time_rows, layout, conditioning):
        text, rotary = conditioning
        video_embeds = self.proj_in(video[None].to(self.proj_in.weight.dtype)).to(text.dtype)
        audio_embeds = self.audio_proj_in(audio[None].to(self.audio_proj_in.weight.dtype)).to(text.dtype)
        x = text.new_empty((1, layout.tags.numel(), text.shape[-1]))
        x.index_copy_(1, layout.text, text)
        x.index_copy_(1, layout.video, video_embeds)
        x.index_copy_(1, layout.audio, audio_embeds)

        embedding = F.silu(self.time_embedder(times))
        time_rows = time_rows.to(torch.int32)
        rows = time_rows * 3 + layout.tags.to(torch.int32)
        for block in self.transformer_blocks:
            x = block(x, embedding, rows, rotary)
        x = self.norm_out(x, embedding, time_rows).to(self.proj_out.weight.dtype)
        return self.proj_out(x)[0, layout.video], self.audio_proj_out(x)[0, layout.audio]
