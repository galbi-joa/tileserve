"""Read indexed safetensors for the Qwen3.8 FP8 checkpoint."""

import json
from pathlib import Path

from safetensors import safe_open
import torch


class SafetensorsReader:
    def __init__(self, model_dir: Path, device: torch.device):
        self.model_dir = model_dir
        self.device = str(device)
        index = model_dir / "model.safetensors.index.json"
        self.weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]

    def get(self, key: str) -> torch.Tensor:
        with safe_open(self.model_dir / self.weight_map[key], framework="pt", device=self.device) as weights:
            return weights.get_tensor(key)


def load_tensor(
    reader: SafetensorsReader,
    key: str,
    *,
    shape: tuple[int, ...] | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    t = reader.get(key)
    if shape is not None and tuple(t.shape) != tuple(shape):
        raise RuntimeError(f"{key} has shape {tuple(t.shape)}, expected {shape}")
    if t.device != device:
        t = t.to(device)
    if t.dtype != dtype:
        t = t.to(dtype)
    return t
