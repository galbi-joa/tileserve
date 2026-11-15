"""Pointwise kernels for the MiniMax H3 video transformer."""

from functools import lru_cache

import tilelang
import tilelang.language as T


@lru_cache(maxsize=16)
@tilelang.jit(out_idx=[-1])
def packed_swiglu(width, tile_size=512, threads=128):
    rows = T.dynamic("rows")
    dtype = T.bfloat16
    tiles = T.ceildiv(width, tile_size)

    @T.prim_func
    def main(
        packed: T.Tensor((rows, 2 * width), dtype),
        output: T.Tensor((rows, width), dtype),
    ):
        with T.Kernel(rows * tiles, threads=threads) as block:
            row = block // tiles
            tile = block % tiles
            for i in T.Parallel(tile_size):
                col = tile * tile_size + i
                if col < width:
                    gate = T.cast(packed[row, width + col], T.float32)
                    up = T.cast(packed[row, col], T.float32)
                    # Preserve the checkpoint reference's BF16 SiLU intermediate.
                    activated = T.cast(gate / (1.0 + T.exp(-gate)), dtype)
                    output[row, col] = T.cast(up * T.cast(activated, T.float32), dtype)

    return main


@lru_cache(maxsize=16)
@tilelang.jit(out_idx=[-1])
def gated_residual(width, offset, tile_size=512, threads=128):
    rows = T.dynamic("rows")
    tokens = T.dynamic("tokens")
    choices = T.dynamic("choices")
    dtype = T.bfloat16
    tiles = T.ceildiv(width, tile_size)

    @T.prim_func
    def main(
        x: T.Tensor((rows, width), dtype),
        update: T.Tensor((rows, width), dtype),
        table: T.Tensor((choices, 6, width), dtype),
        indices: T.Tensor((tokens,), T.int32),
        output: T.Tensor((rows, width), dtype),
    ):
        with T.Kernel(rows * tiles, threads=threads) as block:
            row = block // tiles
            tile = block % tiles
            choice = indices[row % tokens]
            for i in T.Parallel(tile_size):
                col = tile * tile_size + i
                if col < width:
                    gate = T.cast(table[choice, offset + 2, col], T.float32)
                    gated = T.cast(gate * T.cast(update[row, col], T.float32), dtype)
                    output[row, col] = T.cast(T.cast(x[row, col], T.float32) + T.cast(gated, T.float32), dtype)

    return main
