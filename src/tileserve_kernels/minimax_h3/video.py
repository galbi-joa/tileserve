"""Pointwise operations used by the native FL2VA sampler and waveform decoder."""

import tilelang
import tilelang.language as T


@tilelang.jit(compile_flags=["--fmad=false"])
def euler_blend():
    size = T.dynamic("size")
    block = 256

    @T.prim_func
    def update(
        sample: T.Tensor((size,), T.float32),
        velocity: T.Tensor((size,), T.float32),
        coefficients: T.Tensor((3,), T.float32),
    ):
        with T.Kernel(T.ceildiv(size, block), threads=128) as block_id:
            for local in T.Parallel(block):
                index = block_id * block + local
                if index < size:
                    x = sample[index]
                    estimate = x + coefficients[0] * velocity[index]
                    sample[index] = coefficients[1] * x + coefficients[2] * estimate

    return update


@tilelang.jit(out_idx=[-1], compile_flags=["--fmad=false"])
def snake_beta(channels):
    length = T.dynamic("length")
    batches = T.dynamic("batches")
    block = 256

    @T.prim_func
    def apply(
        x: T.Tensor((batches, channels, length), T.float32),
        alpha: T.Tensor((channels,), T.float32),
        inverse_beta: T.Tensor((channels,), T.float32),
        y: T.Tensor((batches, channels, length), T.float32),
    ):
        with T.Kernel(T.ceildiv(length, block), channels, batches, threads=128) as (bx, channel, batch):
            for local in T.Parallel(block):
                index = bx * block + local
                if index < length:
                    value = x[batch, channel, index]
                    sine = T.sin(alpha[channel] * value)
                    y[batch, channel, index] = value + inverse_beta[channel] * (sine * sine)

    return apply
