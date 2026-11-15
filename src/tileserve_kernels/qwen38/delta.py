"""Causal convolution and FP32 gated delta recurrence for Qwen3.8."""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def causal_conv(channels, width, dtype, *, decode=False, packed=False, tile=128):
    tokens = T.dynamic("tokens")
    batch = T.dynamic("batch")
    history = width - 1

    @T.prim_func
    def main(
        x: T.Tensor((tokens, channels), dtype),
        weight: T.Tensor((channels, width), dtype),
        tail: T.Tensor((batch, history, channels), dtype),
        seq_ids: T.Tensor((tokens,), T.int32),
        boundaries: T.Tensor((batch + 1,), T.int32),
        output: T.Tensor((tokens, channels), dtype),
    ):
        with T.Kernel(T.ceildiv(channels, tile), tokens, threads=tile) as (block, token):
            acc = T.alloc_fragment((tile,), T.float32)
            seq = seq_ids[token] if packed or decode else 0
            start = token if decode else (boundaries[seq] if packed else 0)
            end = boundaries[seq + 1] if packed else tokens
            for i in T.Parallel(tile):
                channel = block * tile + i
                acc[i] = 0.0
                if channel < channels:
                    for j in T.unroll(width):
                        pos = token + j - history
                        if pos >= start:
                            acc[i] += T.cast(x[pos, channel], T.float32) * T.cast(weight[channel, j], T.float32)
                        elif decode:
                            acc[i] += T.cast(tail[seq, pos - start + history, channel], T.float32) * T.cast(weight[channel, j], T.float32)
                    value = T.cast(T.cast(acc[i], dtype), T.float32)
                    output[token, channel] = T.cast(value / (1.0 + T.exp(-value)), dtype)
                    if decode:
                        for j in T.unroll(history - 1):
                            tail[seq, j, channel] = tail[seq, j + 1, channel]
                        tail[seq, history - 1, channel] = x[token, channel]
                    elif token == end - 1:
                        for j in T.unroll(history):
                            pos = end - history + j
                            tail[seq, j, channel] = T.if_then_else(pos >= start, x[pos, channel], T.cast(0.0, dtype))

    return main


@tilelang.jit(out_idx=[-4, -3, -2, -1])
def delta_inputs(key_heads, value_heads, key_dim, value_dim, dtype):
    tokens = T.dynamic("tokens")
    key_width = key_heads * key_dim
    channels = 2 * key_width + value_heads * value_dim

    @T.prim_func
    def main(
        qkv: T.Tensor((tokens, channels), dtype),
        ab: T.Tensor((tokens, 2 * value_heads), dtype),
        a_log: T.Tensor((value_heads,), T.float32),
        dt_bias: T.Tensor((value_heads,), T.float32),
        q: T.Tensor((tokens, key_heads, key_dim), T.float32),
        k: T.Tensor((tokens, key_heads, key_dim), T.float32),
        decay: T.Tensor((tokens, value_heads), T.float32),
        beta: T.Tensor((tokens, value_heads), dtype),
    ):
        with T.Kernel(tokens, 2 * key_heads + 1, threads=128) as (token, head):
            x = T.alloc_fragment((key_dim,), T.float32)
            square = T.alloc_fragment((key_dim,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            if head < 2 * key_heads:
                for d in T.Parallel(key_dim):
                    x[d] = T.cast(qkv[token, head * key_dim + d], T.float32)
                    square[d] = x[d] * x[d]
                T.reduce_sum(square, total, dim=0)
                for d in T.Parallel(key_dim):
                    normalized = x[d] * T.rsqrt(total[0] + 1e-6)
                    if head < key_heads:
                        q[token, head, d] = normalized * (key_dim ** -0.5)
                    else:
                        k[token, head - key_heads, d] = normalized
            else:
                for h in T.Parallel(value_heads):
                    step = T.cast(ab[token, h], T.float32) + dt_bias[h]
                    softplus = T.max(step, 0.0) + T.log(1.0 + T.exp(-T.abs(step)))
                    decay[token, h] = -T.exp(a_log[h]) * softplus
                    b = T.cast(ab[token, h + value_heads], T.float32)
                    beta[token, h] = T.cast(1.0 / (1.0 + T.exp(-b)), dtype)

    return main


@tilelang.jit(out_idx=[-1])
def recurrent_delta(key_heads, value_heads, key_dim, value_dim, dtype, *,
                    initialize=False, packed=False, decode=False, warps=4):
    """Keep one value column of the recurrent state in each warp's registers."""
    tokens = T.dynamic("tokens")
    batch = T.dynamic("batch")
    groups = value_heads // key_heads
    key_width = key_heads * key_dim
    channels = key_width * 2 + value_heads * value_dim

    @T.prim_func
    def main(
        q: T.Tensor((tokens, key_heads, key_dim), T.float32),
        k: T.Tensor((tokens, key_heads, key_dim), T.float32),
        qkv: T.Tensor((tokens, channels), dtype),
        decay: T.Tensor((tokens, value_heads), T.float32),
        beta: T.Tensor((tokens, value_heads), dtype),
        state: T.Tensor((batch, value_heads, value_dim, key_dim), T.float32),
        boundaries: T.Tensor((batch + 1,), T.int32),
        seq_ids: T.Tensor((tokens,), T.int32),
        output: T.Tensor((tokens, value_heads, value_dim), dtype),
    ):
        with T.Kernel(T.ceildiv(value_dim, warps), value_heads, tokens if decode else batch, threads=32 * warps) as (block, head, row):
            seq = seq_ids[row] if decode else row
            thread = T.get_thread_binding()
            lane = thread % 32
            value = block * warps + thread // 32
            key_head = head // groups
            column = T.alloc_local((key_dim // 32,), T.float32)
            q_local = T.alloc_local((key_dim // 32,), T.float32)
            k_local = T.alloc_local((key_dim // 32,), T.float32)
            dot = T.alloc_local((1,), T.float32)
            result = T.alloc_local((1,), T.float32)
            start = row if decode else (boundaries[seq] if packed else 0)
            end = row + 1 if decode else (boundaries[seq + 1] if packed else tokens)
            if value < value_dim:
                for i in T.unroll(key_dim // 32):
                    if initialize:
                        column[i] = 0.0
                    else:
                        column[i] = state[seq, head, value, i * 32 + lane]

                for t in T.serial(start, end):
                    dot[0] = 0.0
                    d = T.exp(decay[t, head])
                    for i in T.unroll(key_dim // 32):
                        q_local[i] = q[t, key_head, i * 32 + lane]
                        k_local[i] = k[t, key_head, i * 32 + lane]
                        column[i] *= d
                        dot[0] += k_local[i] * column[i]
                    prediction = T.warp_reduce_sum(dot[0])
                    update = (T.cast(qkv[t, 2 * key_width + head * value_dim + value], T.float32)
                              - prediction) * T.cast(beta[t, head], T.float32)
                    dot[0] = 0.0
                    for i in T.unroll(key_dim // 32):
                        column[i] += k_local[i] * update
                        dot[0] += q_local[i] * column[i]
                    result[0] = T.warp_reduce_sum(dot[0])
                    if lane == 0:
                        output[t, head, value] = T.cast(result[0], dtype)

                for i in T.unroll(key_dim // 32):
                    state[seq, head, value, i * 32 + lane] = column[i]

    return main
