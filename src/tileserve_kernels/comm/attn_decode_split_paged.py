"""Single-token grouped-query attention over paged KV with split context."""

from math import e, log2

import tilelang as tl
import tilelang.language as T


@tl.jit(
    out_idx=[-2, -1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tl.PassConfigKey.TIR_USE_ASYNC_COPY: False,
    },
)
def split_paged_partials(
    batch: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    max_pages_per_seq: int,
    num_splits: int,
    dtype: T.dtype,
    softmax_scale: float,
):
    group_size = q_heads // kv_heads
    block_h = 16
    num_pages = T.dynamic("num_pages")
    log2_scale = softmax_scale * log2(e)

    @T.prim_func
    def main(
        Q: T.Tensor((batch, q_heads, head_dim), dtype),
        K: T.Tensor((num_pages, kv_heads, page_size, head_dim), dtype),
        V: T.Tensor((num_pages, kv_heads, page_size, head_dim), dtype),
        block_table: T.Tensor((batch, max_pages_per_seq), T.int32),
        lengths: T.Tensor((batch,), T.int32),
        Partial: T.Tensor((batch, num_splits, q_heads, head_dim), T.float32),
        LogSumExp: T.Tensor((batch, q_heads, num_splits), T.float32),
    ):
        with T.Kernel(num_splits, kv_heads, batch, threads=32) as (split, kv_head, bid):
            q_shared = T.alloc_shared((block_h, head_dim), dtype)
            k_shared = T.alloc_shared((page_size, head_dim), dtype)
            v_shared = T.alloc_shared((page_size, head_dim), dtype)
            partial_shared = T.alloc_shared((group_size, head_dim), T.float32)
            lse_shared = T.alloc_shared((group_size,), T.float32)

            scores = T.alloc_fragment((block_h, page_size), T.float32)
            probabilities = T.alloc_fragment((block_h, page_size), dtype)
            output = T.alloc_fragment((block_h, head_dim), T.float32)
            old_max = T.alloc_fragment((block_h,), T.float32)
            row_max = T.alloc_fragment((block_h,), T.float32)
            row_scale = T.alloc_fragment((block_h,), T.float32)
            row_sum = T.alloc_fragment((block_h,), T.float32)
            row_lse = T.alloc_fragment((block_h,), T.float32)

            length = lengths[bid]
            pages = T.ceildiv(length, page_size)
            even, extra = pages // num_splits, pages % num_splits
            first_page = split * even + T.min(split, extra)
            split_pages = even + T.if_then_else(split < extra, 1, 0)
            first_head = kv_head * group_size
            last_head = first_head + group_size

            T.clear(q_shared)
            T.copy(Q[bid, first_head:last_head, :], q_shared[:group_size, :])
            T.clear(output)
            T.clear(row_lse)
            T.fill(row_max, -T.infinity(T.float32))

            for local_page in T.serial(split_pages):
                logical_page = first_page + local_page
                physical_page = block_table[bid, logical_page]
                T.copy(K[physical_page, kv_head, :, :], k_shared)
                T.clear(scores)
                T.gemm(q_shared, k_shared, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for h, pos in T.Parallel(block_h, page_size):
                    scores[h, pos] = T.if_then_else(
                        logical_page * page_size + pos < length,
                        scores[h, pos] * log2_scale,
                        -T.infinity(T.float32),
                    )

                T.copy(row_max, old_max)
                T.reduce_max(scores, row_max, dim=1)
                for h in T.Parallel(block_h):
                    row_max[h] = T.max(row_max[h], old_max[h])
                    row_scale[h] = T.exp2(old_max[h] - row_max[h])
                for h, pos in T.Parallel(block_h, page_size):
                    scores[h, pos] = T.exp2(scores[h, pos] - row_max[h])
                T.reduce_sum(scores, row_sum, dim=1)
                for h in T.Parallel(block_h):
                    row_lse[h] = row_lse[h] * row_scale[h] + row_sum[h]
                for h, d in T.Parallel(block_h, head_dim):
                    output[h, d] *= row_scale[h]

                T.copy(scores, probabilities)
                T.copy(V[physical_page, kv_head, :, :], v_shared)
                T.gemm(probabilities, v_shared, output, policy=T.GemmWarpPolicy.FullRow)

            # Empty chunks contribute zero mass and a finite zero value vector.
            for h, d in T.Parallel(block_h, head_dim):
                output[h, d] = T.if_then_else(row_lse[h] > 0, output[h, d] / row_lse[h], 0.0)
            for h in T.Parallel(block_h):
                row_lse[h] = T.if_then_else(
                    row_lse[h] > 0, T.log2(row_lse[h]) + row_max[h], -T.infinity(T.float32)
                )
            T.copy(output[:group_size, :], partial_shared)
            T.copy(partial_shared, Partial[bid, split, first_head:last_head, :])
            T.copy(row_lse[:group_size], lse_shared)
            T.copy(lse_shared, LogSumExp[bid, first_head:last_head, split])

    return main


@tl.jit(out_idx=[-1])
def merge_paged_partials(batch: int, q_heads: int, head_dim: int, num_splits: int, dtype: T.dtype):
    @T.prim_func
    def main(
        Partial: T.Tensor((batch, num_splits, q_heads, head_dim), T.float32),
        LogSumExp: T.Tensor((batch, q_heads, num_splits), T.float32),
        Output: T.Tensor((batch, q_heads, head_dim), dtype),
    ):
        with T.Kernel(q_heads, batch, threads=32) as (head, bid):
            lse = T.alloc_fragment((1, num_splits), T.float32)
            weight = T.alloc_shared((1, num_splits), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            accum = T.alloc_fragment((1, head_dim), T.float32)
            T.copy(LogSumExp[bid, head:head + 1, :], lse)
            T.reduce_max(lse, maximum, dim=1)
            for i, split in T.Parallel(1, num_splits):
                lse[i, split] = T.if_then_else(
                    maximum[i] > -T.infinity(T.float32), T.exp2(lse[i, split] - maximum[i]), 0.0
                )
            T.reduce_sum(lse, total, dim=1)
            for i, split in T.Parallel(1, num_splits):
                lse[i, split] = T.if_then_else(total[i] > 0, lse[i, split] / total[i], 0.0)
            T.copy(lse, weight)
            T.clear(accum)
            for split in T.serial(num_splits):
                for i, d in T.Parallel(1, head_dim):
                    accum[i, d] += Partial[bid, split, head, d] * weight[i, split]
            T.copy(accum, Output[bid, head:head + 1, :])

    return main


def split_gqa_decode(
    batch: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    max_pages_per_seq: int,
    num_splits: int,
    dtype: T.dtype = T.bfloat16,
    softmax_scale: float | None = None,
):
    if not (batch > 0 and kv_heads > 0 and q_heads > 0 and q_heads % kv_heads == 0):
        raise ValueError("the query head count must be a positive multiple of the KV head count")
    if not 1 <= q_heads // kv_heads <= 16:
        raise ValueError("the group size must fit in the 16-head MMA tile")
    if head_dim <= 0 or page_size <= 0 or head_dim % 16 or page_size % 16:
        raise ValueError("head_dim and page_size must be positive and divisible by 16")
    if num_splits not in (2, 4, 8, 16, 32, 64) or max_pages_per_seq <= 0:
        raise ValueError("expected a power of two split count from 2 to 64 and a positive page limit")
    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    partial = split_paged_partials(batch, q_heads, kv_heads, head_dim, page_size,
                                   max_pages_per_seq, num_splits, dtype, scale)
    merge = merge_paged_partials(batch, q_heads, head_dim, num_splits, dtype)

    def run(q, k_pages, v_pages, block_table, lengths):
        return merge(*partial(q, k_pages, v_pages, block_table, lengths))

    return run
