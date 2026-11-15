"""Qwen3-VL prompt and keyframe conditioning for the released FL2VA model."""

import numpy as np
from PIL import Image
import torch
from torch import nn

from .checkpoint import load_weights


_TEXT_LAYERS = 50


def load_conditioner(path, with_keyframes):
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel, Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding,
    )

    config = Qwen3VLConfig.from_pretrained(path, local_files_only=True)
    if config.text_config.num_hidden_layers <= _TEXT_LAYERS:
        raise ValueError("FL2VA requires Qwen3-VL's layer 50 conditioning")
    config.text_config.num_hidden_layers = _TEXT_LAYERS
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3VLModel(config) if with_keyframes else Qwen3VLTextModel(config.text_config)
    decoder = model.language_model if with_keyframes else model
    # The conditioning is the hidden state after layer 50, before the final RMSNorm.
    decoder.norm = nn.Identity()
    decoder.rotary_emb = Qwen3VLTextRotaryEmbedding(config.text_config)
    if not with_keyframes:
        return load_weights(model, path, name="model", prefix="model.language_model.")
    # The vision RoPE frequencies are nonpersistent and absent from safetensors.
    rotary = model.visual.rotary_pos_emb
    model.visual.rotary_pos_emb = type(rotary)(rotary.config)
    return load_weights(model, path, name="model", prefix="model.")


def resize_keyframes(first, last, width, height):
    output = {}
    for anchor, image in (("first", first), ("last", last)):
        if image is None:
            continue
        image = image.convert("RGB")
        if output and image.size != (width, height):
            scale = max(width / image.width, height / image.height)
            size = max(width, round(image.width * scale)), max(height, round(image.height * scale))
            image = image.resize(size, Image.Resampling.LANCZOS)
            left, top = (size[0] - width) // 2, (size[1] - height) // 2
            image = image.crop((left, top, left + width, top + height))
        elif image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        output[anchor] = image
    return output


def encode_prompt(model, tokenizer, processor, prompt, keyframes, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not prompt_ids:
        raise ValueError("The prompt must tokenize to at least one token")
    ids, tags = [], []
    vision = {}
    if keyframes:
        prepared = processor.image_processor(images=list(keyframes.values()), return_tensors="pt")
        vision = {"pixel_values": prepared.pixel_values.to(device, torch.bfloat16),
                  "image_grid_thw": prepared.image_grid_thw.to(device)}
        for index, grid in enumerate(prepared.image_grid_thw, start=1):
            label = tokenizer(f"<Picture {index}>: ", add_special_tokens=False)["input_ids"]
            size = int(grid.prod()) // processor.image_processor.merge_size**2
            begin, pad, end = tokenizer.convert_tokens_to_ids(["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"])
            ids += label + [begin] + [pad] * size + [end]
            tags += [1] * len(label) + [0] * (size + 2)
        vision["mm_token_type_ids"] = torch.tensor(processor.create_mm_token_type_ids([ids + prompt_ids]), device=device)
    ids = torch.tensor([ids + prompt_ids], dtype=torch.long, device=device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, **vision)
    return output.last_hidden_state.to(torch.bfloat16), torch.tensor(tags + [1] * len(prompt_ids))


def pixels_for_keyframe(image, device):
    return torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, None].to(device, torch.float32)
