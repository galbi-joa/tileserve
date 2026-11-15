# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

"""MiniMax H3's keyframe encoder and temporally chunked video decoder.

Adapted from Hugging Face Diffusers commit 43e30028af5120d8b0868cb6975b9fad39cc4fab.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import VideoFeedForward, rotate, sdpa


class MiniMaxH3VideoCausalConv3d(nn.Conv3d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_padding > 0:
            padding = self.spatial_padding
            hidden_states = F.pad(
                hidden_states, (padding, padding, padding, padding, 0, 0), mode=self.spatial_padding_mode
            )
        if self.temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant")
        return F.conv3d(hidden_states, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


class MiniMaxH3VideoGroupNorm(nn.GroupNorm):

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(batch_size * num_frames, num_channels, 1, height, width)
        hidden_states = super().forward(hidden_states)
        hidden_states = hidden_states.view(batch_size, num_frames, num_channels, height, width)
        return hidden_states.permute(0, 2, 1, 3, 4).contiguous()


class MiniMaxH3VideoResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = MiniMaxH3VideoGroupNorm(norm_num_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = MiniMaxH3VideoGroupNorm(norm_num_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = MiniMaxH3VideoCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = MiniMaxH3VideoCausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.silu(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = F.silu(self.norm2(hidden_states))
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class MiniMaxH3VideoDownsample3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=0,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_stride == 2:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode)
        return self.conv(hidden_states)


class MiniMaxH3VideoDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                MiniMaxH3VideoResnetBlock3d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(num_layers)
            ]
        )
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = nn.ModuleList(
                [
                    MiniMaxH3VideoDownsample3d(
                        out_channels,
                        out_channels,
                        temporal_stride=temporal_downsample_factor,
                        spatial_stride=spatial_downsample_factor,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                ]
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class MiniMaxH3VideoEncoder3d(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 48,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        self.conv_in = MiniMaxH3VideoCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoDownBlock3d(
                    in_channels=block_in_channels[i],
                    out_channels=block_out_channels[i],
                    num_layers=layers_per_block,
                    temporal_downsample_factor=temporal_downsample_factors[i],
                    spatial_downsample_factor=spatial_downsample_factors[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(len(block_out_channels))
            ]
        )

        self.norm_out = MiniMaxH3VideoGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps, affine=True)
        self.conv_out = MiniMaxH3VideoCausalConv3d(
            block_out_channels[-1],
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        hidden_states = F.silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states)


class MiniMaxH3VideoAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, eps=1e-5, bias=True):
        super().__init__()
        self.heads = heads
        self.fused_qk = True
        inner = heads * dim_head
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, inner, bias=bias)
        self.to_k = nn.Linear(dim, inner, bias=bias)
        self.to_v = nn.Linear(dim, inner, bias=bias)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=bias))

    def forward(self, x, rotary):
        q, k, v = [projection(x).unflatten(-1, (self.heads, -1))
                   for projection in (self.to_q, self.to_k, self.to_v)]
        if rotary is not None and self.fused_qk and q.is_cuda and q.dtype == torch.bfloat16 and self.heads % 4 == 0:
            from tileserve_kernels.minimax_h3.video_decode import normalize_rotate_qk

            cos, sin = (item[0, :, 0] for item in rotary)
            q, k = normalize_rotate_qk(self.heads, q.shape[-1], cos.shape[-1], self.norm_q.eps)(
                q, k, cos.contiguous(), sin.contiguous()
            )
        else:
            q = self.norm_q(q.float()).to(q.dtype)
            k = self.norm_k(k.float()).to(k.dtype)
            if rotary is not None:
                cos, sin = (item[0, :, 0] for item in rotary)
                q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        return self.to_out(sdpa(q, k, v).flatten(2))


class MiniMaxH3VideoRotaryPosEmbed(nn.Module):

    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3) -> None:
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(f"`dim` {dim} must be divisible by `2 * num_axes` {2 * num_axes}.")
        inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * num_axes / dim, dtype=torch.float32, device="cpu")
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


class MiniMaxH3VideoTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = MiniMaxH3VideoAttention(dim=dim, heads=heads, dim_head=dim_head, eps=eps, bias=bias)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = VideoFeedForward(dim, dim * ffn_mult, bias=bias)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        # The reference normalizes in float32 regardless of the compute dtype. The residual stream is float32 too
        # because the anchors are, so the normed activation follows the projections instead.
        compute_dtype = self.attn.to_q.weight.dtype
        norm_hidden_states = self.norm1(hidden_states.float()).to(compute_dtype)
        hidden_states = hidden_states + self.attn(norm_hidden_states, rotary_emb) * self.scale1
        norm_hidden_states = self.norm2(hidden_states.float()).to(compute_dtype)
        hidden_states = hidden_states + self.ff(norm_hidden_states) * self.scale2
        return hidden_states


class MiniMaxH3VideoViTDecoder3d(nn.Module):

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        patch_size_t: int = 4,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.75,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.rope = MiniMaxH3VideoRotaryPosEmbed(int(attention_head_dim * rope_dim_ratio), theta=rope_theta)
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoTransformerBlock(
                    dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch_size, num_frames * height * width, num_channels
        )
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]

        register_tokens = self.register_tokens.expand(batch_size, -1, -1).to(hidden_states.dtype)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
            for size in (num_frames, height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        rotary_emb = self.rope(position_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)

        # `norm_out` is pinned to float32 and `proj_out` is not.
        hidden_states = self.norm_out(hidden_states).to(self.proj_out.weight.dtype)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :num_patches, :]

        patch_size, patch_size_t = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            patch_size_t,
            patch_size,
            patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )


class VideoVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 24,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
        decoder_num_layers: int = 36,
        decoder_num_attention_heads: int = 32,
        decoder_attention_head_dim: int = 64,
        decoder_num_register_tokens: int = 4,
        decoder_ffn_mult: int = 4,
        decoder_rope_theta: float = 100.0,
        decoder_rope_dim_ratio: float = 0.75,
        decoder_norm_eps: float = 1e-5,
        clip_length: int = 17,
        token_drop: int = 3,
        *,
        with_encoder: bool,
    ) -> None:
        super().__init__()
        self.clip_length = clip_length
        self.token_drop = token_drop

        self.spatial_compression_ratio = math.prod(spatial_downsample_factors)
        self.temporal_compression_ratio = math.prod(temporal_downsample_factors)

        if with_encoder:
            self.encoder = MiniMaxH3VideoEncoder3d(
                in_channels=in_channels,
                out_channels=2 * latent_channels,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                spatial_downsample_factors=spatial_downsample_factors,
                temporal_downsample_factors=temporal_downsample_factors,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                spatial_padding_mode=spatial_padding_mode,
            )
            self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            patch_size=self.spatial_compression_ratio,
            patch_size_t=self.temporal_compression_ratio,
            num_layers=decoder_num_layers,
            num_attention_heads=decoder_num_attention_heads,
            attention_head_dim=decoder_attention_head_dim,
            num_register_tokens=decoder_num_register_tokens,
            ffn_mult=decoder_ffn_mult,
            rope_theta=decoder_rope_theta,
            rope_dim_ratio=decoder_rope_dim_ratio,
            norm_eps=decoder_norm_eps,
        )

        # Derived temporal-chunking geometry. `clip_length` pixel frames are encoded at a time; because
        # `clip_length` is not a multiple of `temporal_compression_ratio`, the decoder has to re-derive the
        # implicit leading pad (`frame_pre_padding`) and the overlap that `token_drop` leaves behind.
        self.frame_pre_padding = (-clip_length) % self.temporal_compression_ratio
        self.tokens_chunk_size = math.ceil(clip_length / self.temporal_compression_ratio)
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.temporal_compression_ratio - self.frame_pre_padding, 0)

        # The tile size in pixel space, and the minimum overlap between two neighbouring tiles. The actual overlaps are
        # widened (in multiples of `spatial_compression_ratio`) so that the tiles cover the frame exactly.
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64
        self.decode_tile_batch = 4


    def _split_tiles(self, length: int, tile_size: int, min_overlap: int) -> tuple[list[int], list[int], list[int]]:
        if tile_size >= length:
            return [0], [length], []

        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1

        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // self.spatial_compression_ratio):
            overlaps[i % (num_tiles - 1)] += self.spatial_compression_ratio

        tile_start_indices = [0]
        for i in range(num_tiles - 1):
            tile_start_indices.append(tile_start_indices[-1] + tile_size - overlaps[i])
        return tile_start_indices, [tile_size] * num_tiles, overlaps


    def _blend(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)


    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)


    def _encode_clip(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2], x.shape[-1]
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        rows = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            row = []
            for j_pos, j_len in zip(x_indices, x_lengths):
                tile = x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
                row.append(self.quant_conv(self.encoder(tile)))
            rows.append(row)

        latent_y_overlaps = [overlap // self.spatial_compression_ratio for overlap in y_overlaps]
        latent_x_overlaps = [overlap // self.spatial_compression_ratio for overlap in x_overlaps]
        return self._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)


    def _decode_clip(self, z: torch.Tensor) -> torch.Tensor:
        # Tiles are laid out in pixel space and then mapped back onto the latent grid.
        height = z.shape[-2] * self.spatial_compression_ratio
        width = z.shape[-1] * self.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        ratio = self.spatial_compression_ratio
        crops = [
            z[..., y // ratio : (y + h) // ratio, x // ratio : (x + w) // ratio]
            for y, h in zip(y_indices, y_lengths, strict=True)
            for x, w in zip(x_indices, x_lengths, strict=True)
        ]
        decoded = []
        # Within a clip, every tile has the same shape. Keep the original order for blending.
        for start in range(0, len(crops), self.decode_tile_batch):
            group = crops[start : start + self.decode_tile_batch]
            packed = group[0] if len(group) == 1 else torch.cat(group, dim=0)
            decoded.extend(self.decoder(self.post_quant_conv(packed)).split(z.shape[0], dim=0))
        cols = len(x_indices)
        rows = [decoded[start : start + cols] for start in range(0, len(decoded), cols)]

        return self._stitch_tiles(rows, y_overlaps, x_overlaps)


    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.token_drop
        temporal_ratio = self.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio

        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        decoded_chunks = []
        overlap = None
        for i in range(num_chunks):
            start = i * tokens_chunk_size
            clip = self._decode_clip(z[:, :, start : start + tokens_chunk_size + self.token_overlap])
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                    decoded_chunks.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded_chunks.append(overlap)

        dec = torch.cat(decoded_chunks, dim=2)

        # `pad_tokens` repeated latent frames produced trailing pixel frames that were never requested. A chunk's
        # last latent frame only covers `clip_length % temporal_ratio` pixel frames, the others cover `temporal_ratio`.
        if pad_tokens > 0:
            intra_tail = self.clip_length % temporal_ratio
            num_tokens_before_pad = z.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
                for k in range(pad_tokens)
            )
            dec = dec[:, :, :-pad_frames]
        return dec


    def encode_keyframe(self, pixels, generator):
        moments = self._encode_clip(pixels.to(self.quant_conv.weight.dtype))
        mean, logvar = moments.chunk(2, dim=1)
        noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
        return mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * noise.to(mean.device)

    def decode(self, latents):
        return self._decode(latents.to(self.post_quant_conv.weight.dtype))
