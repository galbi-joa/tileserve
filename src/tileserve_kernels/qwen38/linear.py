"""BF16/FP16 linear projections for text prefill and decode."""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def gemv(out_features, in_features, dtype, *, rows_per_block=4, vector=8):
    if in_features % (32 * vector):
        vector = 1

    @T.prim_func
    def main(
        x: T.Tensor((1, in_features), dtype),
        weight: T.Tensor((out_features, in_features), dtype),
        out: T.Tensor((1, out_features), dtype),
    ):
        with T.Kernel(T.ceildiv(out_features, rows_per_block), threads=rows_per_block * 32) as block:
            thread = T.get_thread_binding()
            lane = thread % 32
            row = block * rows_per_block + thread // 32
            acc = T.alloc_local((1,), T.float32)
            data = T.alloc_local((vector,), dtype)
            weights = T.alloc_local((vector,), dtype)
            acc[0] = 0.0
            if row < out_features:
                for i in T.serial(T.ceildiv(in_features, 32 * vector)):
                    for j in T.vectorized(vector):
                        col = (i * 32 + lane) * vector + j
                        data[j] = T.if_then_else(col < in_features, x[0, col], T.cast(0, dtype))
                        weights[j] = T.if_then_else(col < in_features, weight[row, col], T.cast(0, dtype))
                    for j in T.unroll(vector):
                        acc[0] += T.cast(data[j], T.float32) * T.cast(weights[j], T.float32)
            total = T.warp_reduce_sum(acc[0])
            if row < out_features and lane == 0:
                out[0, row] = T.cast(total, dtype)

    return main


@tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def gemm(out_features, in_features, dtype, *, block_m=32, block_n=64, block_k=64, threads=128, stages=2):
    tokens = T.dynamic("tokens")

    @T.prim_func
    def main(
        x: T.Tensor((tokens, in_features), dtype),
        weight: T.Tensor((out_features, in_features), dtype),
        out: T.Tensor((tokens, out_features), dtype),
    ):
        with T.Kernel(T.ceildiv(tokens, block_m), T.ceildiv(out_features, block_n), threads=threads) as (m, n):
            a = T.alloc_shared((block_m, block_k), dtype)
            b = T.alloc_shared((block_n, block_k), dtype)
            c = T.alloc_fragment((block_m, block_n), T.float32)
            T.clear(c)
            for k in T.Pipelined(T.ceildiv(in_features, block_k), num_stages=stages):
                T.copy(x[m * block_m, k * block_k], a)
                T.copy(weight[n * block_n, k * block_k], b)
                T.gemm(a, b, c, transpose_B=True)
            T.copy(c, out[m * block_m, n * block_n])

    return main
