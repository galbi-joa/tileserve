# Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MiniMax H3's mono BigVGAN waveform decoder.

Adapted from Hugging Face Diffusers commit 43e30028af5120d8b0868cb6975b9fad39cc4fab.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    half_size = kernel_size // 2

    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    # Normalize to sum 1 so a constant input does not leak through the resampler.
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class MiniMaxH3AudioSnakeBeta(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.exp()
        inverse_beta = (self.beta.exp() + 1e-9).reciprocal()
        if hidden_states.is_cuda:
            from tileserve_kernels.minimax_h3.video import snake_beta

            return snake_beta(hidden_states.shape[1])(hidden_states.contiguous(), alpha, inverse_beta)
        return hidden_states + inverse_beta[None, :, None] * torch.sin(alpha[None, :, None] * hidden_states).square()


class MiniMaxH3AudioLowPassFilter1d(nn.Module):

    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int):
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(
            hidden_states, self.filter.expand(num_channels, -1, -1), stride=self.stride, groups=num_channels
        )


class MiniMaxH3AudioUpSample1d(nn.Module):

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states, self.filter.expand(num_channels, -1, -1), stride=self.stride, groups=num_channels
        )
        return hidden_states[..., self.pad_left : -self.pad_right]


class MiniMaxH3AudioDownSample1d(nn.Module):

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.lowpass = MiniMaxH3AudioLowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=kernel_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lowpass(hidden_states)


class MiniMaxH3AudioActivation1d(nn.Module):

    def __init__(self, activation: nn.Module, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = MiniMaxH3AudioUpSample1d(ratio, kernel_size)
        self.downsample = MiniMaxH3AudioDownSample1d(ratio, kernel_size)
        self._filtered_snake_beta = ratio == 2 and kernel_size == 12 and isinstance(activation, MiniMaxH3AudioSnakeBeta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._filtered_snake_beta and hidden_states.is_cuda and hidden_states.dtype == torch.float32:
            from tileserve_kernels.minimax_h3.audio_decode import filtered_snake_beta

            return filtered_snake_beta()(
                hidden_states.contiguous(),
                self.act.alpha.exp(),
                (self.act.beta.exp() + 1e-9).reciprocal(),
                self.upsample.filter.flatten(),
                self.downsample.lowpass.filter.flatten(),
            )
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.downsample(hidden_states)


class MiniMaxH3AudioAMPBlock(nn.Module):

    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, ...]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2))
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2))
                for _ in dilation
            ]
        )
        self.activations = nn.ModuleList(
            [
                MiniMaxH3AudioActivation1d(activation=MiniMaxH3AudioSnakeBeta(channels))
                for _ in range(2 * len(dilation))
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            residual = conv1(act1(hidden_states))
            residual = conv2(act2(residual))
            hidden_states = residual + hidden_states
        return hidden_states


class MiniMaxH3AudioBigVGANDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        upsample_initial_channel: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))

        # Each upsampler is wrapped in a one-element `ModuleList` in the original checkpoint
        # (`ups.<i>.0`); the extra nesting is kept so the state dict stays a passthrough.
        self.ups = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            nn.ConvTranspose1d(
                                upsample_initial_channel // (2**i),
                                upsample_initial_channel // (2 ** (i + 1)),
                                kernel,
                                rate,
                                padding=(kernel - rate) // 2,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(MiniMaxH3AudioAMPBlock(channels, kernel, tuple(dilation)))

        self.activation_post = MiniMaxH3AudioActivation1d(activation=MiniMaxH3AudioSnakeBeta(channels))
        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_pre(hidden_states)

        for i in range(self.num_upsamples):
            hidden_states = self.ups[i][0](hidden_states)
            residual = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j](hidden_states)
                residual = block if residual is None else residual + block
            hidden_states = residual / self.num_kernels

        hidden_states = self.activation_post(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        return torch.clamp(hidden_states, min=-1.0, max=1.0)


class AudioVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config["latent_dim"]
        self.dec_in_proj = nn.Conv1d(config["latent_channels"], channels, 1)
        self.decoder = MiniMaxH3AudioBigVGANDecoder(
            in_channels=channels,
            upsample_initial_channel=config["decoder_dim"],
            upsample_rates=config["decoder_rates"],
            upsample_kernel_sizes=config["decoder_kernel_sizes"],
            resblock_kernel_sizes=config["resblock_kernel_sizes"],
            resblock_dilation_sizes=config["resblock_dilation_sizes"],
        )

    def decode(self, latents):
        return self.decoder(self.dec_in_proj(latents.float()))
