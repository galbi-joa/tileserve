from typing import Callable

import tilelang as tl
import tilelang.language as T


@tl.jit(out_idx=[])
def kvcache_store_paged(
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_seqs: int,
    max_blocks_per_seq: int,
    dtype: T.dtype = T.bfloat16,
    threads: int = 256,
) -> Callable:
    """
    使用 block table 将每个 token 的 K/V 保存到 paged KV cache。

    Layout:
      - `block_table`: [max_seqs, max_blocks_per_seq] int32
      - `k_cache/v_cache`: [num_blocks, num_kv_heads, block_size, head_dim] dtype

    输入是 token-major（flattened）格式：
      - seq_ids: [N] int32
      - positions: [N] int32（每个 token 在序列中的绝对位置）
      - k_in/v_in: [N, num_kv_heads, head_dim] dtype
    """
    N = T.dynamic("N")
    num_blocks = T.dynamic("num_blocks")

    @T.prim_func
    def main(
        seq_ids: T.Tensor([N], T.int32),  # type: ignore
        positions: T.Tensor([N], T.int32),  # type: ignore
        k_in: T.Tensor([N, num_kv_heads, head_dim], dtype),  # type: ignore
        v_in: T.Tensor([N, num_kv_heads, head_dim], dtype),  # type: ignore
        block_table: T.Tensor([max_seqs, max_blocks_per_seq], T.int32),  # type: ignore
        k_cache: T.Tensor([num_blocks, num_kv_heads, block_size, head_dim], dtype),  # type: ignore
        v_cache: T.Tensor([num_blocks, num_kv_heads, block_size, head_dim], dtype),  # type: ignore
    ):
        # 每个线程处理一个 token，并把它的 K/V 写入对应 block
        with T.Kernel(N, threads=threads) as (i,):
            seq_id = seq_ids[i]
            pos = positions[i]
            block_idx = T.floordiv(pos, block_size)
            in_block = T.floormod(pos, block_size)
            phys = block_table[seq_id, block_idx]

            # KV 维度并行写入
            for h, d in T.Parallel(num_kv_heads, head_dim):
                k_cache[phys, h, in_block, d] = k_in[i, h, d]
                v_cache[phys, h, in_block, d] = v_in[i, h, d]

    return main
