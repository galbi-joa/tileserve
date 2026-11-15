import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def swiglu(intermediate_size: int, dtype: T.dtype, tile_size: int = 256, threads: int = 128):
    rows = T.dynamic("rows")
    tiles_per_row = (intermediate_size + tile_size - 1) // tile_size

    @T.prim_func
    def main(
        gate: T.Tensor([rows, intermediate_size], dtype),
        up: T.Tensor([rows, intermediate_size], dtype),
        out: T.Tensor([rows, intermediate_size], dtype),
    ):
        with T.Kernel(rows * tiles_per_row, threads=threads) as block:
            row = block // tiles_per_row
            tile = block % tiles_per_row
            for i in T.Parallel(tile_size):
                col = tile * tile_size + i
                if col < intermediate_size:
                    value = T.cast(gate[row, col], T.float32)
                    # Keep the activation rounding of F.silu(gate) * up.
                    activated = T.cast(value / (1.0 + T.exp(-value)), dtype)
                    out[row, col] = T.cast(T.cast(activated, T.float32) * T.cast(up[row, col], T.float32), dtype)

    return main
