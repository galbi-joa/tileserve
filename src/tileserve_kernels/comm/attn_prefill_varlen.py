from typing import Callable

import tilelang as tl
import tilelang.language as T


_tilelang_015 = tl.__version__.split("+", 1)[0] == "0.1.15"


@tl.jit(
    out_idx=[6],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tl.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
        # v0.1.15 can emit a nonconstant cp.async wait for this dynamic loop.
        **({
            tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tl.PassConfigKey.TIR_USE_ASYNC_COPY: False,
        } if _tilelang_015 else {}),
    },
)
def gqa_prefill_varlen(
    batch_size: int,
    groups: int,
    UQ: int | None,
    UKV: int | None,
    heads: int,
    dim: int,
    is_causal: bool,
    *,
    sm_scale: float | None = None,
    block_M: int = 128,
    block_N: int = 128,
    num_stages: int = 2,
    threads: int = 256,
    dtype: T.dtype = T.bfloat16,
) -> Callable:
    """
    变长 GQA prefill 注意力（unpad 格式）。

    输入输出形状说明：
      - Q_unpad: [UQ, heads, dim]
      - K_unpad/V_unpad: [UKV, head_kv, dim]，其中 head_kv = heads // groups
      - cu_seqlens_q/cu_seqlens_k: [batch_size+1] int32
      - max_seqlen_q: int32（当前 batch 的最大长度）
      - Output_unpad: [UQ, heads, dim]
    """
    if heads % groups != 0:
        raise ValueError("heads must be divisible by groups")
    if block_M <= 0 or block_N <= 0:
        raise ValueError("block_M and block_N must be > 0")
    if block_M % 16 != 0 or block_N % 16 != 0:
        raise ValueError("block_M and block_N must be divisible by 16")
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if threads % 32 != 0:
        raise ValueError("threads must be divisible by 32")
    if num_stages <= 0:
        raise ValueError("num_stages must be > 0")

    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    scale = float(sm_scale * 1.44269504)  # log2(e)

    head_kv = heads // groups
    query_tokens = T.dynamic("query_tokens") if UQ is None else UQ
    kv_tokens = T.dynamic("kv_tokens") if UKV is None else UKV
    q_shape = [query_tokens, heads, dim]
    kv_shape = [kv_tokens, head_kv, dim]
    o_shape = [query_tokens, heads, dim]

    accum_dtype = T.float32
    # Multi-warp QK and PV can assign score columns to different warps.
    allocate_scores = T.alloc_shared if _tilelang_015 and threads > 32 else T.alloc_fragment

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),  # type: ignore
        K_unpad: T.Tensor(kv_shape, dtype),  # type: ignore
        V_unpad: T.Tensor(kv_shape, dtype),  # type: ignore
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),  # type: ignore
        cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),  # type: ignore
        max_seqlen_q: T.int32,  # type: ignore
        Output_unpad: T.Tensor(o_shape, dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size, threads=threads) as (bx, by, bz):
            T.use_swizzle(10)
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = allocate_scores([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            batch_idx = bz
            head_idx = by
            kv_head_idx = head_idx // groups

            q_start_idx = cu_seqlens_q[batch_idx]
            kv_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]

            q_len = q_end_idx - q_start_idx
            kv_len = k_end_idx - kv_start_idx

            # Skip query tiles outside shorter sequences in a ragged batch.
            if batch_size == 1 or bx * block_M < q_len:
                T.copy(
                    Q_unpad[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, head_idx, :],
                    Q_shared,
                )

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                offset = kv_len - q_len  # 右对齐，用于 causal mask
                max_visible_k_idx = offset + (bx + 1) * block_M

                if is_causal:
                    start = 0
                    end = T.min(T.ceildiv(max_visible_k_idx, block_N), T.ceildiv(kv_len, block_N))
                else:
                    start = 0
                    end = T.ceildiv(kv_len, block_N)

                loop_range = T.max(0, end - start)

                for k in T.Pipelined(loop_range, num_stages=0 if _tilelang_015 else num_stages):
                    actual_k = k + start
                    T.copy(
                        K_unpad[kv_start_idx + actual_k * block_N : kv_start_idx + (actual_k + 1) * block_N, kv_head_idx, :],
                        KV_shared,
                    )

                    # 先清 accumulating fragment
                    T.clear(acc_s)
                    T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            q_idx = bx * block_M + i + offset
                            k_idx = actual_k * block_N + j
                            acc_s[i, j] = T.if_then_else(
                                (q_idx < k_idx)
                                or (bx * block_M + i >= q_len)
                                or (actual_k * block_N + j >= kv_len),
                                -T.infinity(accum_dtype),
                                acc_s[i, j],
                            )
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                (bx * block_M + i >= q_len) or (actual_k * block_N + j >= kv_len),
                                -T.infinity(accum_dtype),
                                acc_s[i, j],
                            )

                    # softmax 前的 max / logsum 调整
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                    # 学习 softmax scale / sum
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.if_then_else(
                            scores_max[i] == -T.infinity(accum_dtype),
                            1.0,
                            T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                        )
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            scores_max[i] == -T.infinity(accum_dtype),
                            0.0,
                            T.exp2(acc_s[i, j] * scale - scores_max[i] * scale),
                        )
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]

                    T.copy(
                        V_unpad[kv_start_idx + actual_k * block_N : kv_start_idx + (actual_k + 1) * block_N, kv_head_idx, :],
                        KV_shared,
                    )
                    # A globally in-bounds copy can still pick up rows from the next
                    # sequence. Clear them before PV, where 0 * NaN is still NaN.
                    if (actual_k + 1) * block_N > kv_len:
                        for j, d in T.Parallel(block_N, dim):
                            if actual_k * block_N + j >= kv_len:
                                KV_shared[j, d] = 0
                    T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, d in T.Parallel(block_M, dim):
                    denom = logsum[i]
                    acc_o[i, d] = T.if_then_else(denom == 0, 0, acc_o[i, d] / denom)
                T.copy(acc_o, O_shared)
                for i, d in T.Parallel(block_M, dim):
                    if bx * block_M + i < q_len:
                        Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = O_shared[i, d]

    return main
