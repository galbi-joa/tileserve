"""Text-model normalization and gated projections."""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def embedding(vocab, width, dtype, tile=128):
    tokens = T.dynamic("tokens")

    @T.prim_func
    def main(
        ids: T.Tensor((tokens,), T.int32),
        weight: T.Tensor((vocab, width), dtype),
        out: T.Tensor((tokens, width), dtype),
    ):
        with T.Kernel(tokens, T.ceildiv(width, tile), threads=tile) as (row, block):
            token = ids[row]
            for i in T.Parallel(tile):
                col = block * tile + i
                if col < width:
                    out[row, col] = weight[token, col]

    return main


@tilelang.jit(out_idx=[-1])
def residual_add(width, dtype, tile=128):
    tokens = T.dynamic("tokens")

    @T.prim_func
    def main(
        residual: T.Tensor((tokens, width), dtype),
        update: T.Tensor((tokens, width), dtype),
        out: T.Tensor((tokens, width), dtype),
    ):
        with T.Kernel(tokens, T.ceildiv(width, tile), threads=tile) as (row, block):
            for i in T.Parallel(tile):
                col = block * tile + i
                if col < width:
                    out[row, col] = T.cast(
                        T.cast(residual[row, col], T.float32) + T.cast(update[row, col], T.float32), dtype
                    )

    return main


@tilelang.jit(out_idx=[-1])
def normalize(width, eps, dtype, *, gated=False, threads=128):
    rows = T.dynamic("rows")

    @T.prim_func
    def main(
        x: T.Tensor((rows, width), dtype),
        weight: T.Tensor((width,), dtype),
        gate: T.Tensor((rows, width), dtype),
        out: T.Tensor((rows, width), dtype),
    ):
        with T.Kernel(rows, threads=threads) as row:
            values = T.alloc_fragment((width,), T.float32)
            squares = T.alloc_fragment((width,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for i in T.Parallel(width):
                values[i] = T.cast(x[row, i], T.float32)
                squares[i] = values[i] * values[i]
            T.reduce_sum(squares, total, dim=0)
            for i in T.Parallel(width):
                normalized = values[i] * T.rsqrt(total[0] / width + eps)
                w = T.cast(weight[i], T.float32)
                if gated:
                    z = T.cast(gate[row, i], T.float32)
                    affine = T.cast(T.cast(T.cast(normalized, dtype), T.float32) * w, dtype)
                    out[row, i] = T.cast(T.cast(affine, T.float32) * z / (1.0 + T.exp(-z)), dtype)
                else:
                    out[row, i] = T.cast(normalized * (1.0 + w), dtype)

    return main


@tilelang.jit(out_idx=[])
def rotate_partial(heads, head_dim, rotary_dim, dtype):
    tokens = T.dynamic("tokens")
    capacity = T.dynamic("capacity")
    half = rotary_dim // 2

    @T.prim_func
    def main(
        x: T.Tensor((tokens, heads, head_dim), dtype),
        cos: T.Tensor((capacity, half), dtype),
        sin: T.Tensor((capacity, half), dtype),
        positions: T.Tensor((tokens,), T.int32),
    ):
        with T.Kernel(tokens, heads, threads=32) as (token, head):
            for d in T.Parallel(half):
                lo = T.cast(x[token, head, d], T.float32)
                hi = T.cast(x[token, head, d + half], T.float32)
                c = T.cast(cos[positions[token], d], T.float32)
                s = T.cast(sin[positions[token], d], T.float32)
                lo_c = T.cast(lo * c, dtype)
                hi_s = T.cast(hi * s, dtype)
                hi_c = T.cast(hi * c, dtype)
                lo_s = T.cast(lo * s, dtype)
                x[token, head, d] = T.cast(T.cast(lo_c, T.float32) - T.cast(hi_s, T.float32), dtype)
                x[token, head, d + half] = T.cast(T.cast(hi_c, T.float32) + T.cast(lo_s, T.float32), dtype)

    return main


@tilelang.jit(out_idx=[-1])
def attention_gate(width, dtype, tile=128):
    tokens = T.dynamic("tokens")

    @T.prim_func
    def main(
        x: T.Tensor((tokens, width), dtype),
        gate: T.Tensor((tokens, width), dtype),
        out: T.Tensor((tokens, width), dtype),
    ):
        with T.Kernel(tokens, T.ceildiv(width, tile), threads=tile) as (token, block):
            for i in T.Parallel(tile):
                d = block * tile + i
                if d < width:
                    z = T.cast(gate[token, d], T.float32)
                    scale = T.cast(1.0 / (1.0 + T.exp(-z)), dtype)
                    out[token, d] = T.cast(T.cast(x[token, d], T.float32) * T.cast(scale, T.float32), dtype)

    return main
