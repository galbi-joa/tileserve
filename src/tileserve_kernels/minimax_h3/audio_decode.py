"""Anti-aliased SnakeBeta for H3's FP32 BigVGAN decoder."""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], compile_flags=["--fmad=false"])
def filtered_snake_beta():
    """Fuse the 12-tap interpolation and decimation around SnakeBeta."""
    batches = T.dynamic("batches")
    channels = T.dynamic("channels")
    length = T.dynamic("length")
    block = 256
    halo = 5

    @T.prim_func
    def apply(
        x: T.Tensor((batches, channels, length), T.float32),
        alpha: T.Tensor((channels,), T.float32),
        inverse_beta: T.Tensor((channels,), T.float32),
        up_filter: T.Tensor((12,), T.float32),
        down_filter: T.Tensor((12,), T.float32),
        y: T.Tensor((batches, channels, length), T.float32),
    ):
        with T.Kernel(T.ceildiv(length, block), channels, batches, threads=128) as (bx, channel, batch):
            source = T.alloc_shared((block + 2 * halo,), T.float32)
            activated = T.alloc_shared((2, block + halo), T.float32)
            start = bx * block

            for i in T.Parallel(block + 2 * halo):
                sample = T.min(T.max(start + i - halo, 0), length - 1)
                source[i] = x[batch, channel, sample]

            # The downsampler reads 12 samples from each length-2 window.
            # Keep the even and odd positions in separate shared-memory rows.
            for phase, i in T.Parallel(2, block + halo):
                oversample = T.min(T.max(2 * (start + i) + phase - halo, 0), 2 * length - 1)
                # Fold the transposed convolution's input pad and output crop into its polyphase indices.
                base = (oversample + halo) // 2 - start + halo
                parity = (oversample + 1) % 2
                interpolated = T.alloc_var(T.float32, init=0.0)
                for tap in T.unroll(6):
                    interpolated += source[base - tap] * up_filter[parity + 2 * tap]
                value = interpolated * 2.0
                sine = T.sin(alpha[channel] * value)
                activated[phase, i] = value + inverse_beta[channel] * (sine * sine)

            for i in T.Parallel(block):
                if start + i < length:
                    filtered = T.alloc_var(T.float32, init=0.0)
                    for tap in T.unroll(12):
                        filtered += activated[tap % 2, i + tap // 2] * down_filter[tap]
                    y[batch, channel, start + i] = filtered

    return apply
