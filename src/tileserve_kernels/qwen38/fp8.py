"""TileLang linear operators for E4M3 weights with 128×128 FP32 scales."""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], execution_backend="cython")
def fp8_gemv(n, k, dtype, *, rows_per_block=4, vector=4):
    """Stream compressed weights and scale each 128-column dot product."""
    @T.prim_func
    def main(
        x: T.Tensor((1, k), dtype),
        w: T.Tensor((n, k), T.float8_e4m3fn),
        scales: T.Tensor((T.ceildiv(n, 128), k // 128), T.float32),
        out: T.Tensor((1, n), dtype),
    ):
        with T.Kernel(T.ceildiv(n, rows_per_block), threads=rows_per_block * 32) as block:
            thread = T.get_thread_binding()
            lane = thread % 32
            row = block * rows_per_block + thread // 32
            acc = T.alloc_local((1,), T.float32)
            partial = T.alloc_local((1,), T.float32)
            values = T.alloc_local((vector,), T.float8_e4m3fn)
            acc[0] = 0.0
            if row < n:
                for group in T.serial(k // 128):
                    partial[0] = 0.0
                    for part in T.unroll(4 // vector):
                        for j in T.vectorized(vector):
                            col = group * 128 + (part * 32 + lane) * vector + j
                            values[j] = w[row, col]
                        for j in T.unroll(vector):
                            col = group * 128 + (part * 32 + lane) * vector + j
                            partial[0] += T.cast(x[0, col], T.float32) * T.cast(values[j], T.float32)
                    acc[0] += partial[0] * scales[row // 128, group]
            total = T.warp_reduce_sum(acc[0])
            if row < n and lane == 0:
                out[0, row] = T.cast(total, dtype)

    return main


@tilelang.jit(out_idx=[-2, -1], execution_backend="cython")
def quantize_fp8(k, dtype):
    rows = T.dynamic("rows")

    @T.prim_func
    def main(
        x: T.Tensor((rows, k), dtype),
        quant: T.Tensor((rows, k), T.float8_e4m3fn),
        scale: T.Tensor((rows, k // 128), T.float32),
    ):
        with T.Kernel(rows, k // 128, threads=32) as (row, group):
            values = T.alloc_fragment((1, 128), T.float32)
            amax = T.alloc_fragment((1,), T.float32)
            for i in T.Parallel(128):
                values[0, i] = T.abs(T.cast(x[row, group * 128 + i], T.float32))
            T.reduce_max(values, amax, dim=1)
            scale[row, group] = T.max(amax[0], 1e-4) / 448.0
            for i in T.Parallel(128):
                val = T.cast(x[row, group * 128 + i], T.float32)
                quant[row, group * 128 + i] = T.cast(
                    val * (448.0 / T.max(amax[0], 1e-4)), T.float8_e4m3fn
                )

    return main


@tilelang.jit(out_idx=[-1], execution_backend="cython", pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TIR_USE_ASYNC_COPY: False,
})
def fp8_gemm(n, k, dtype, *, block_m=16, block_n=64, threads=128):
    rows = T.dynamic("rows")
    f8 = T.float8_e4m3fn

    @T.prim_func
    def main(
        x: T.Tensor((rows, k), f8),
        w: T.Tensor((n, k), f8),
        a_scale: T.Tensor((rows, k // 128), T.float32),
        w_scale: T.Tensor((T.ceildiv(n, 128), k // 128), T.float32),
        out: T.Tensor((rows, n), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(rows, block_m), threads=threads) as (bn, bm):
            a = T.alloc_shared((block_m, 128), f8)
            b = T.alloc_shared((block_n, 128), f8)
            scale = T.alloc_shared((block_m,), T.float32)
            partial = T.alloc_fragment((block_m, block_n), T.float32)
            total = T.alloc_fragment((block_m, block_n), T.float32)
            T.clear(total)
            for group in T.serial(k // 128):
                T.copy(x[bm * block_m, group * 128], a)
                T.copy(w[bn * block_n, group * 128], b)
                for i in T.Parallel(block_m):
                    row = bm * block_m + i
                    scale[i] = T.if_then_else(
                        row < rows, a_scale[row, group] * w_scale[bn * block_n // 128, group], 0.0
                    )
                T.clear(partial)
                T.gemm(a, b, partial, transpose_B=True)
                for i, j in T.Parallel(block_m, block_n):
                    total[i, j] += partial[i, j] * scale[i]
            T.copy(total, out[bm * block_m, bn * block_n])

    return main
