"""Fused pointwise and reduction kernels for the H3 video VAE decoder."""

from functools import lru_cache

import tilelang
import tilelang.language as T


@lru_cache(maxsize=16)
@tilelang.jit(out_idx=[-2, -1], compile_flags=["--fmad=false"])
def normalize_rotate_qk(heads, head_dim, rotary_dim, eps):
    batch = T.dynamic("batch")
    tokens = T.dynamic("tokens")
    heads_per_block = 4
    half = rotary_dim // 2
    dtype = T.bfloat16

    @T.macro
    def norm_rotate(x, cos, sin, output, b, pos, first_head):
        squares = T.alloc_fragment((heads_per_block, head_dim), T.float32)
        values = T.alloc_fragment((heads_per_block, head_dim), T.float32)
        sums = T.alloc_fragment((heads_per_block,), T.float32)
        for h, d in T.Parallel(heads_per_block, head_dim):
            values[h, d] = T.cast(x[b, pos, first_head + h, d], T.float32)
            squares[h, d] = values[h, d] * values[h, d]
        T.reduce_sum(squares, sums, dim=1)
        for h, d in T.Parallel(heads_per_block, head_dim):
            inv_rms = T.rsqrt(sums[h] * (1.0 / head_dim) + eps)
            normalized = T.cast(values[h, d] * inv_rms, dtype)
            if d < rotary_dim:
                partner = T.if_then_else(d < half, d + half, d - half)
                paired = T.cast(x[b, pos, first_head + h, partner], T.float32)
                signed = T.if_then_else(d < half, -paired, paired)
                rotated = T.cast(signed * inv_rms, dtype)
                # Keep the BF16 rounding points of the separate RMS and RoPE operations.
                c = T.cast(T.cast(cos[pos, d], dtype), T.float32)
                s = T.cast(T.cast(sin[pos, d], dtype), T.float32)
                a = T.cast(T.cast(normalized, T.float32) * c, dtype)
                z = T.cast(T.cast(rotated, T.float32) * s, dtype)
                output[b, pos, first_head + h, d] = T.cast(T.cast(a, T.float32) + T.cast(z, T.float32), dtype)
            else:
                output[b, pos, first_head + h, d] = normalized

    @T.prim_func
    def main(
        q: T.Tensor((batch, tokens, heads, head_dim), dtype),
        k: T.Tensor((batch, tokens, heads, head_dim), dtype),
        cos: T.Tensor((tokens, rotary_dim), T.float32),
        sin: T.Tensor((tokens, rotary_dim), T.float32),
        q_out: T.Tensor((batch, tokens, heads, head_dim), dtype),
        k_out: T.Tensor((batch, tokens, heads, head_dim), dtype),
    ):
        with T.Kernel(batch * tokens, heads // heads_per_block, 2, threads=128) as (row, head, projection):
            b = row // tokens
            pos = row % tokens
            first_head = head * heads_per_block
            if projection == 0:
                norm_rotate(q, cos, sin, q_out, b, pos, first_head)
            else:
                norm_rotate(k, cos, sin, k_out, b, pos, first_head)

    return main
