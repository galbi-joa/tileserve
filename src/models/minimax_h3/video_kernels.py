"""Use the shared TileLang kernels in the MiniMax-H3 video DiT."""

from collections import defaultdict
from functools import partial
from types import MethodType

import torch

class TileLangAttentionProcessor:
    def __init__(self, reference, rotary, attention):
        self.reference = reference
        self.rotary = rotary
        self.attention = attention

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if (attention_mask is not None or hidden_states.device.type != "cuda"
                or hidden_states.dtype != torch.bfloat16):
            return self.reference(attn, hidden_states, rotary_emb, attention_mask)
        if attn.fused_projections:
            q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            q, k, v = (proj(hidden_states) for proj in (attn.to_q, attn.to_k, attn.to_v))
        q, k, v = (tensor.unflatten(-1, (-1, attn.head_dim)) for tensor in (q, k, v))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if rotary_emb is not None:
            q, k = (self.rotary(tensor, *rotary_emb) for tensor in (q, k))
        output = self.attention(q, k, v).flatten(2, 3).type_as(q)
        return attn.to_out[1](attn.to_out[0](output))


def _install_swiglu(module, kernel_factory, counts):
    original = module.forward
    width = module.proj.out_features // 2

    def forward(self, hidden_states):
        if hidden_states.device.type != "cuda" or hidden_states.dtype != torch.bfloat16:
            return original(hidden_states)
        projected = self.proj(hidden_states)
        rows = projected.reshape(-1, 2 * width).contiguous()
        result = kernel_factory(width)(rows)
        counts["swiglu_calls"] += 1
        return result.reshape(*projected.shape[:-1], width)

    module.forward = MethodType(forward, module)


def install_video_kernels(transformer, *, attention_backend="sdpa", fused_swiglu=True, attention_config=None):
    """Install DiT kernels and return mutable call counters for profiling."""
    if attention_backend not in {"sdpa", "tilelang"}:
        raise ValueError(f"Unknown MiniMax H3 video attention backend: {attention_backend}")
    if attention_config is not None and attention_backend != "tilelang":
        raise ValueError("An attention configuration requires the TileLang attention backend")

    from diffusers.models.activations import SwiGLU
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor, dispatch_attention_fn
    from tileserve_kernels.minimax_h3.rotary import partial_rotary

    attention_modules = [module for module in transformer.modules()
                         if isinstance(getattr(module, "processor", None), MiniMaxH3AttnProcessor)]
    swiglu_modules = [module for module in transformer.modules() if isinstance(module, SwiGLU)] if fused_swiglu else []
    if not attention_modules or (fused_swiglu and not swiglu_modules):
        raise ValueError("Expected MiniMax H3 video DiT attention and SwiGLU modules")
    counts = {"attention_modules": len(attention_modules), "swiglu_modules": len(swiglu_modules),
              "rotary_calls": 0, "tilelang_attention_calls": 0, "swiglu_calls": 0}

    def rotary(x, cos, sin):
        result = partial_rotary(x, cos, sin)
        counts["rotary_calls"] += 1
        return result

    def sdpa(q, k, v):
        return dispatch_attention_fn(q, k, v, is_causal=False)

    attention = sdpa
    if attention_backend == "tilelang":
        from tileserve_kernels.minimax_h3.attention import noncausal_attention

        run_attention = partial(noncausal_attention, **(attention_config or {}))

        def attention(q, k, v):
            result = run_attention(q, k, v)
            counts["tilelang_attention_calls"] += 1
            return result

    for module in attention_modules:
        module.set_processor(TileLangAttentionProcessor(module.processor, rotary, attention))

    if swiglu_modules:
        from tileserve_kernels.minimax_h3.pointwise import packed_swiglu

        for module in swiglu_modules:
            _install_swiglu(module, packed_swiglu, counts)
    return counts


def install_video_vae_batching(vae, tile_batch=8):
    """Batch equal-sized spatial tiles within each video VAE temporal clip."""
    if tile_batch < 1:
        raise ValueError("Video VAE tile batch must be positive")
    counts = {"vae_tile_batch": tile_batch, "vae_clip_calls": 0, "vae_tiles": 0, "vae_decoder_calls": 0}
    if tile_batch == 1:
        return counts
    original_decode = vae._decode_clip

    def decode_clip(self, z):
        counts["vae_clip_calls"] += 1
        if not self.use_tiling:
            counts["vae_tiles"] += 1
            counts["vae_decoder_calls"] += 1
            return original_decode(z)

        factor = self.spatial_compression_ratio
        y_starts, y_sizes, y_overlaps = self._split_tiles(
            z.shape[-2] * factor, self.tile_sample_min_height, self.tile_sample_min_overlap_height,
        )
        x_starts, x_sizes, x_overlaps = self._split_tiles(
            z.shape[-1] * factor, self.tile_sample_min_width, self.tile_sample_min_overlap_width,
        )
        groups = defaultdict(list)
        for y, (y_start, height) in enumerate(zip(y_starts, y_sizes, strict=True)):
            for x, (x_start, width) in enumerate(zip(x_starts, x_sizes, strict=True)):
                top, left = y_start // factor, x_start // factor
                bottom, right = top + height // factor, left + width // factor
                groups[bottom - top, right - left].append((y, x, z[..., top:bottom, left:right]))

        grid = [[None for _ in x_starts] for _ in y_starts]
        for tiles in groups.values():
            for start in range(0, len(tiles), tile_batch):
                chunk = tiles[start:start + tile_batch]
                crops = [tile for _, _, tile in chunk]
                packed = crops[0] if len(crops) == 1 else torch.cat(crops)
                decoded = self.decoder(self.post_quant_conv(packed))
                for (y, x, _), pixels in zip(chunk, decoded.split(z.shape[0]), strict=True):
                    grid[y][x] = pixels
                counts["vae_tiles"] += len(chunk)
                counts["vae_decoder_calls"] += 1
        return self._stitch_tiles(grid, y_overlaps, x_overlaps)

    vae._decode_clip = MethodType(decode_clip, vae)
    return counts
