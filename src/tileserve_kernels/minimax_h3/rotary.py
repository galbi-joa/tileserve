"""Fused partial three-axis RoPE for MiniMax-H3 video attention."""

from functools import lru_cache

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def _rotate(heads, head_dim, rotary_dim):
    batch = T.dynamic("batch")
    tokens = T.dynamic("tokens")
    half = rotary_dim // 2
    dtype = T.bfloat16

    @T.prim_func
    def main(
        x: T.Tensor((batch, tokens, heads, head_dim), dtype),
        cos: T.Tensor((tokens, rotary_dim), T.float32),
        sin: T.Tensor((tokens, rotary_dim), T.float32),
        out: T.Tensor((batch, tokens, heads, head_dim), dtype),
    ):
        with T.Kernel(tokens, heads, batch, threads=32) as (pos, head, b):
            for d in T.Parallel(head_dim):
                value = x[b, pos, head, d]
                if d < rotary_dim:
                    rotated = T.if_then_else(d < half, -T.cast(x[b, pos, head, d + half], T.float32),
                                             T.cast(x[b, pos, head, d - half], T.float32))
                    c = T.cast(T.cast(cos[pos, d], dtype), T.float32)
                    s = T.cast(T.cast(sin[pos, d], dtype), T.float32)
                    lhs = T.cast(T.cast(value, T.float32) * c, dtype)
                    rhs = T.cast(rotated * s, dtype)
                    out[b, pos, head, d] = T.cast(T.cast(lhs, T.float32) + T.cast(rhs, T.float32), dtype)
                else:
                    out[b, pos, head, d] = value

    return main


_get_rotate = lru_cache(maxsize=16)(_rotate)


def partial_rotary(x, cos, sin):
    return _get_rotate(x.shape[-2], x.shape[-1], cos.shape[-1])(
        x.contiguous(), cos.float().contiguous(), sin.float().contiguous(),
    )
