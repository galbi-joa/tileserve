"""Validate the supported Qwen3.8 FP8 checkpoint format."""

import json
from pathlib import Path


def read_fp8_config(path: Path) -> dict:
    config = json.loads((path / "config.json").read_text())
    quantization = config.get("quantization_config") if isinstance(config, dict) else None
    if (
        not isinstance(config, dict)
        or config.get("model_type") != "qwen3_5"
        or not isinstance(config.get("text_config"), dict)
        or not isinstance(quantization, dict)
        or quantization.get("quant_method") != "fp8"
        or quantization.get("fmt") != "e4m3"
        or quantization.get("weight_block_size") != [128, 128]
        or quantization.get("activation_scheme") != "dynamic"
    ):
        raise ValueError("Expected a Qwen3.8 FP8 checkpoint with dynamic E4M3 and 128×128 weight blocks")
    return config
