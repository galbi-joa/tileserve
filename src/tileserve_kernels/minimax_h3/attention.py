"""Dense BF16 FlashAttention for MiniMax H3's unmasked DiT sequence."""

from functools import lru_cache
from math import e, lcm, log2

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F


@lru_cache(maxsize=32)
def _attention_kernel(heads, dim, block_m, block_n, threads, stages, fixed_tokens, async_copy):
    return tilelang.compile(
        _attention_ir(heads, dim, block_m, block_n, threads, stages, fixed_tokens),
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TIR_USE_ASYNC_COPY: async_copy,
        },
    )


def _attention_ir(heads, dim, block_m, block_n, threads, stages, fixed_tokens):
    batch = T.dynamic("batch")
    tokens = fixed_tokens if fixed_tokens else T.dynamic("tokens")
    shape = (batch, tokens, heads, dim)
    dtype = T.bfloat16
    scale = dim ** -0.5 * log2(e)

    @T.macro
    def attend(query, key, value, score, probability, accum,
               maximum, previous, rescale, row_sum, normalizer, valid_keys):
        T.clear(score)
        T.gemm(query, key, score, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
        for i, j in T.Parallel(block_m, block_n):
            score[i, j] = T.if_then_else(
                j < valid_keys, score[i, j] * scale, -T.infinity(T.float32)
            )
        T.copy(maximum, previous)
        T.reduce_max(score, maximum, dim=1)
        for i in T.Parallel(block_m):
            maximum[i] = T.max(maximum[i], previous[i])
            rescale[i] = T.exp2(previous[i] - maximum[i])
        for i, j in T.Parallel(block_m, block_n):
            score[i, j] = T.exp2(score[i, j] - maximum[i])
        T.reduce_sum(score, row_sum, dim=1)
        for i in T.Parallel(block_m):
            normalizer[i] = normalizer[i] * rescale[i] + row_sum[i]
        for i, d in T.Parallel(block_m, dim):
            accum[i, d] *= rescale[i]
        T.copy(score, probability)
        T.gemm(probability, value, accum, policy=T.GemmWarpPolicy.FullRow)

    @T.prim_func
    def main(
        q: T.Tensor(shape, dtype),
        k: T.Tensor(shape, dtype),
        v: T.Tensor(shape, dtype),
        valid_tokens: T.int32,
        out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(tokens, block_m), heads, batch, threads=threads) as (m, h, b):
            query = T.alloc_shared((block_m, dim), dtype)
            key = T.alloc_shared((block_n, dim), dtype)
            value = T.alloc_shared((block_n, dim), dtype)
            score = T.alloc_fragment((block_m, block_n), T.float32)
            probability = T.alloc_shared((block_m, block_n), dtype)
            accum = T.alloc_fragment((block_m, dim), T.float32)
            maximum = T.alloc_fragment((block_m,), T.float32)
            previous = T.alloc_fragment((block_m,), T.float32)
            rescale = T.alloc_fragment((block_m,), T.float32)
            row_sum = T.alloc_fragment((block_m,), T.float32)
            normalizer = T.alloc_fragment((block_m,), T.float32)

            T.copy(q[b, m * block_m:(m + 1) * block_m, h, :], query)
            T.clear(accum)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity(T.float32))

            for n in T.Pipelined(T.ceildiv(tokens, block_n), num_stages=stages):
                T.copy(k[b, n * block_n:(n + 1) * block_n, h, :], key)
                T.copy(v[b, n * block_n:(n + 1) * block_n, h, :], value)
                attend(query, key, value, score, probability, accum, maximum, previous,
                       rescale, row_sum, normalizer, valid_tokens - n * block_n)

            for i, d in T.Parallel(block_m, dim):
                accum[i, d] /= normalizer[i]
            T.copy(accum, query)
            T.copy(query, out[b, m * block_m:(m + 1) * block_m, h, :])

    return main


def noncausal_attention(query, key, value, *, block_m=128, block_n=64, threads=256,
                    stages=None, async_copy=True):
    _, tokens, heads, dim = query.shape
    # On the measured SM120, staging pays off on the long DiT sequence only.
    if stages is None:
        pipelined = (tokens >= 16384 and dim == 128
                     and torch.cuda.get_device_capability(query.device) == (12, 0))
        stages = 1 if pipelined else 0
    padded = tokens
    if stages:
        multiple = lcm(block_m, block_n)
        padded = (tokens + multiple - 1) // multiple * multiple
        if padded != tokens:
            query, key, value = (F.pad(tensor, (0, 0, 0, 0, 0, padded - tokens))
                                 for tensor in (query, key, value))
    fixed_tokens = padded if stages else None
    kernel = _attention_kernel(heads, dim, block_m, block_n, threads, stages, fixed_tokens, async_copy)
    return kernel(query.contiguous(), key.contiguous(), value.contiguous(), tokens)[:, :tokens].contiguous()
