"""Load the released mixed-precision FL2VA components directly from safetensors."""

from itertools import groupby
import json

from safetensors import safe_open
import torch


def validate_video_checkpoint(path):
    index = json.loads((path / "modular_model_index.json").read_text())
    if index.get("_class_name") != "MiniMaxH3ModularPipeline":
        raise ValueError("Expected the MiniMax H3 FL2VA video and audio checkpoint")


def read_config(path):
    return {key: value for key, value in json.loads((path / "config.json").read_text()).items()
            if not key.startswith("_")}


def load_weights(model, path, *, name="diffusion_pytorch_model", prefix="", dtype=torch.bfloat16, fp32=()):
    index = path / f"{name}.safetensors.index.json"
    state = model.state_dict()
    if index.is_file():
        files = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(path / f"{name}.safetensors", framework="pt", device="cpu") as weights:
            files = dict.fromkeys(weights.keys(), f"{name}.safetensors")

    expected = {prefix + key: key for key in state}
    missing = expected.keys() - files.keys()
    if missing:
        raise ValueError(f"Incomplete {path.name} checkpoint: {', '.join(sorted(missing)[:5])}")
    loaded = {}
    for shard, keys in groupby(sorted(expected, key=files.__getitem__), key=files.__getitem__):
        with safe_open(path / shard, framework="pt", device="cpu") as weights:
            for key in keys:
                target = expected[key]
                tensor = weights.get_tensor(key)
                if state[target].shape != tensor.shape:
                    raise ValueError(f"{path.name}/{key} has unexpected shape {tuple(tensor.shape)}")
                if tensor.is_floating_point():
                    target_dtype = torch.float32 if any(part in fp32 for part in target.split(".")) else dtype
                    tensor = tensor.to(target_dtype)
                loaded[target] = tensor
    model.load_state_dict(loaded, strict=True, assign=True)
    return model.requires_grad_(False).eval()
