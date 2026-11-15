from typing import Callable

import tilelang as tl
import tilelang.language as T


_tilelang_015 = tl.__version__.split("+", 1)[0] == "0.1.15"


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tl.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
        tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # v0.1.15 requires a constant cp.async wait for paged decode.
        **({tl.PassConfigKey.TIR_USE_ASYNC_COPY: False} if _tilelang_015 else {}),
    },
)
def gqa_decode_paged_attention(
    batch: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    max_blocks_per_seq: int,
    block_H: int = 64,
    num_stages: int = 2,
    threads: int = 256,
    dtype: T.dtype = T.bfloat16,
    softmax_scale: float | None = None,
) -> Callable:
    """
    Decode 阶段的 paged GQA attention，每条序列只处理一个 query token。

    参考 TileLang flash-decoding 示例结构，从 paged cache 读取 KV。

    输入：
      - Q: [batch, q_heads, head_dim]
      - k_cache/v_cache: [num_blocks, kv_heads, block_size, head_dim]
      - block_table: [batch, max_blocks_per_seq] int32（每行代表一个 seq 的 block 映射）
      - cache_seqlens: [batch] int32（每条序列当前已缓存 token 数）

    输出：
      - O: [batch, q_heads, head_dim]
    """
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    # 软max 里的 scale = softmax_scale * log2(e)（TileLang 要求 log2(e)）
    scale = float(softmax_scale * 1.44269504)  # log2(e)

    if q_heads <= 0 or kv_heads <= 0:
        raise ValueError("q_heads and kv_heads must be > 0")
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    kv_group = q_heads // kv_heads
    if kv_group <= 0:
        raise ValueError("q_heads must be >= kv_heads")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if block_size % 16 != 0:
        raise ValueError("block_size must be divisible by 16 for this GEMM policy")

    threads = int(threads)
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if threads % 32 != 0:
        raise ValueError("threads must be divisible by 32")

    # 对于 `Q @ K^T` GEMM，N == block_size。
    # FullRow + 小 M（如 16）时， TileLang 会把大部分 warp 分配给 N 方向，此时需要 `warp_col_tiles >= 8`。
    # 因此我们把 threads 限制在 (block_size/8)*32 的范围内，确保 layout 约束成立。
    #
    # 经验上（也符合 TileLang 的 policy）只要：
    #   num_warps <= block_size / 8
    # 就可以让 MMA emitter 正确推导出布局。
    '''
    mma emitter ? ->
    TileLang 在生成 CUDA/tensor-core kernel 时，会依据你传进去的 block 大
    小、warp 数、GEMM 方向等信息推导 layout（像 warp_col_tiles、thread 分布、shared memory usage），
    然后再 emit 出对应的 MMA（Matrix Multiply-Accumulate）指令。
    '''


    max_threads = int(block_size) * 4  # (block_size/8) * 32
    if threads > max_threads:
        threads = max_threads
    requested_block_H = int(block_H)
    if requested_block_H <= 0:
        raise ValueError("block_H must be > 0")

    # 但只对 `valid_block_H` 个 head 写回结果，剩下用于 layout 约束。
    valid_block_H = min(requested_block_H, int(kv_group))
    if q_heads % valid_block_H != 0:
        raise ValueError("q_heads must be divisible by valid_block_H")
    if kv_group % valid_block_H != 0:
        raise ValueError("kv_group must be divisible by valid_block_H")

    # 将 block_H 提升到最接近的 16 的倍数，满足 GEMM/Layout 限制。
    block_H = max(16, requested_block_H)
    if block_H % 16 != 0:
        block_H = ((block_H + 15) // 16) * 16

    num_blocks = T.dynamic("num_blocks")
    shape_q = [batch, q_heads, head_dim]
    shape_cache = [num_blocks, kv_heads, block_size, head_dim]
    shape_table = [batch, max_blocks_per_seq]
    shape_o = [batch, q_heads, head_dim]

    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, dtype),
        k_cache: T.Tensor(shape_cache, dtype),
        v_cache: T.Tensor(shape_cache, dtype),
        block_table: T.Tensor(shape_table, T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        O: T.Tensor(shape_o, dtype),
    ):
        # kernel grid：每个 (bx, by) 负责一个 batch 的 valid_block_H 个 heads
        with T.Kernel(batch, q_heads // valid_block_H, threads=threads) as (bx, by):
            T.use_swizzle(10)
            Q_shared = T.alloc_shared([block_H, head_dim], dtype)
            K_shared = T.alloc_shared([block_size, head_dim], dtype)
            V_shared = T.alloc_shared([block_size, head_dim], dtype)
            O_shared = T.alloc_shared([valid_block_H, head_dim], dtype)

            acc_s = T.alloc_fragment([block_H, block_size], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_size], dtype)
            acc_o = T.alloc_fragment([block_H, head_dim], accum_dtype)

            # softmax 辅助变量
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            bid = bx
            hid = by
            cur_kv_head = hid // (kv_group // valid_block_H)
            cur_len = cache_seqlens[bid]

            q_start = hid * valid_block_H
            T.clear(Q_shared)
            T.copy(Q[bid, q_start : q_start + valid_block_H, :], Q_shared[:valid_block_H, :])

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(cur_len, block_size)

            for k in T.Pipelined(loop_range, num_stages=0 if _tilelang_015 else num_stages):
                phys = block_table[bid, k]

                T.copy(k_cache[phys, cur_kv_head, :, :], K_shared)

                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_H, block_size):
                    acc_s[i, j] = T.if_then_else(
                        k * block_size + j < cur_len,
                        acc_s[i, j],
                        -T.infinity(accum_dtype),
                    )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

                for i, j in T.Parallel(block_H, block_size):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, acc_s_cast)

                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i, j in T.Parallel(block_H, head_dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(v_cache[phys, cur_kv_head, :, :], V_shared)

                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_H, head_dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o[:valid_block_H, :], O_shared)
            T.copy(O_shared, O[bid, q_start : q_start + valid_block_H, :])

    return main
