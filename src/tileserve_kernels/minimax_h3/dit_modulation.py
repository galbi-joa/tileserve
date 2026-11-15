"""Indexed AdaLN affine transform with the eager BF16 rounding points."""

from functools import lru_cache

import tilelang
import tilelang.language as T


@lru_cache(maxsize=16)
@tilelang.jit(out_idx=[-1], compile_flags=["--fmad=false"])
def indexed_affine(width, fields, offset):
    rows = T.dynamic("rows")
    choices = T.dynamic("choices")
    dtype = T.bfloat16
    tile_width = 512
    tiles = T.ceildiv(width, tile_width)

    @T.prim_func
    def main(
        normed: T.Tensor((rows, width), dtype),
        table: T.Tensor((choices, fields, width), dtype),
        indices: T.Tensor((rows,), T.int32),
        out: T.Tensor((rows, width), dtype),
    ):
        with T.Kernel(rows * tiles, threads=128) as block:
            row = block // tiles
            tile = block % tiles
            choice = indices[row]
            for i in T.Parallel(tile_width):
                col = tile * tile_width + i
                if col < width:
                    scale = T.cast(1.0 + T.cast(table[choice, offset + 1, col], T.float32), dtype)
                    shifted = T.cast(
                        T.cast(normed[row, col], T.float32) * T.cast(scale, T.float32), dtype
                    )
                    out[row, col] = T.cast(shifted, T.float32) + T.cast(table[choice, offset, col], T.float32)

    return main
