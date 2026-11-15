"""Patch layout and independent video/audio noise schedules for FL2VA."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PackedLayout:
    positions: torch.Tensor
    tags: torch.Tensor
    text: torch.Tensor
    video: torch.Tensor
    audio: torch.Tensor
    condition_rows: int

    def to(self, device):
        return PackedLayout(self.positions.to(device), self.tags.to(device), self.text.to(device),
                            self.video.to(device), self.audio.to(device), self.condition_rows)


def pack_video(latent, patch):
    pt, ph, pw = patch
    _, channels, time, height, width = latent.shape
    return (latent.reshape(channels, time // pt, pt, height // ph, ph, width // pw, pw)
            .permute(1, 3, 5, 0, 2, 4, 6).reshape(-1, channels * pt * ph * pw).contiguous())


def unpack_video(rows, shape, patch):
    _, channels, time, height, width = shape
    pt, ph, pw = patch
    return (rows.reshape(1, time // pt, height // ph, width // pw, channels, pt, ph, pw)
            .permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(shape).contiguous())


def _spatial_axis(length, patch, area_root):
    ratio = length / area_root
    start = (1 - ratio) / 2
    return torch.from_numpy(np.linspace(start, start + ratio, length // patch, endpoint=False) * 32)


def build_layout(text_tags, time, height, width, audio_length, patch, anchors):
    text_count = text_tags.numel()
    area_root = np.sqrt(height * width)
    h, w = (_spatial_axis(size, tile, area_root) for size, tile in zip((height, width), patch[1:], strict=True))
    frame = torch.stack(torch.meshgrid(h, w, indexing="ij"), dim=-1).reshape(-1, 2)
    rows_per_frame = len(frame)
    condition_rows = len(anchors) * rows_per_frame
    audio_start = text_count + condition_rows
    video_start = audio_start + 2 * audio_length
    total = video_start + time * rows_per_frame
    positions = torch.zeros(total, 3, dtype=torch.float64)
    positions[:text_count, 0] = torch.arange(text_count, dtype=torch.float64)

    spans = np.resize(np.array([1, 4, 4, 4, 4], dtype=np.float64), time) * (5 / 3)
    for i, anchor in enumerate(anchors):
        start = text_count + i * rows_per_frame
        rows = positions[start:start + rows_per_frame]
        rows[:, 0] = text_count if anchor == "first" else text_count + float(spans.sum()) - 5 / 3
        rows[:, 1:] = frame

    positions[audio_start:video_start, 0] = text_count + torch.arange(audio_length, dtype=torch.float64).repeat(2)
    positions[audio_start:audio_start + audio_length, 2] = w[0]
    positions[audio_start + audio_length:video_start, 2] = w[-1]
    video_xyz = positions[video_start:].view(time, rows_per_frame, 3)
    t = torch.from_numpy(spans)
    video_xyz[:, :, 0] = (text_count + torch.cat((t.new_zeros(1), t[:-1].cumsum(0))))[:, None]
    video_xyz[:, :, 1:] = frame

    text = torch.arange(text_count)
    video = torch.cat((torch.arange(text_count, audio_start), torch.arange(video_start, total)))
    audio = torch.arange(audio_start, video_start)
    tags = torch.zeros(total, dtype=torch.long)
    tags[text] = text_tags
    tags[audio] = 2
    return PackedLayout(positions, tags, text, video, audio, condition_rows)


def euler_schedule(steps, shift):
    base = torch.linspace(1, 0, steps + 1, dtype=torch.float32)
    sigmas = torch.unique_consecutive(shift * base / (1 + (shift - 1) * base))
    if len(sigmas) != steps + 1:
        raise ValueError("Requested denoiser steps collide on the FP32 sigma grid")
    times = 1 - sigmas[:-1]
    ratio = sigmas[1:] / sigmas[:-1]
    return times, torch.stack((1 - times, ratio, 1 - ratio), dim=1)


def row_times(layout, video_time, audio_time, keyframe_time):
    times = torch.full(layout.tags.shape, float(video_time), dtype=torch.float32)
    times[layout.video[:layout.condition_rows]] = max(float(video_time), keyframe_time)
    times[layout.audio] = audio_time
    return times.unique(sorted=True, return_inverse=True)
