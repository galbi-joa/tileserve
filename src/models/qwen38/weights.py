"""Load the text tensors from a Qwen3.8 FP8 checkpoint."""

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm

from models.comm.weights import SafetensorsReader, load_tensor


@dataclass(frozen=True)
class FP8Projection:
    values: torch.Tensor
    scales: torch.Tensor

    @property
    def shape(self):
        return self.values.shape


Projection = torch.Tensor | FP8Projection


def merge_projections(*projections: Projection) -> Projection:
    if isinstance(projections[0], FP8Projection):
        if any(not isinstance(weight, FP8Projection) for weight in projections):
            raise ValueError("Packed projections must use the same quantization")
        if any(weight.shape[0] % 128 for weight in projections[:-1]):
            raise ValueError("Packed FP8 projection boundaries must align to 128 weight rows")
        return FP8Projection(
            torch.cat([weight.values for weight in projections]),
            torch.cat([weight.scales for weight in projections]),
        )
    if any(isinstance(weight, FP8Projection) for weight in projections):
        raise ValueError("Packed projections must use the same quantization")
    return torch.cat(projections)


@dataclass(frozen=True)
class FullAttentionWeights:
    qkv: Projection
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    output: Projection


@dataclass(frozen=True)
class DeltaWeights:
    qkvz: Projection
    ab: Projection
    conv: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    norm: torch.Tensor
    output: Projection


@dataclass(frozen=True)
class TextLayerWeights:
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    gate_up: Projection
    down: Projection
    attention: FullAttentionWeights | DeltaWeights


@dataclass(frozen=True)
class TextWeights:
    embedding: torch.Tensor
    norm: torch.Tensor
    lm_head: Projection
    layers: list[TextLayerWeights]


def load_text_weights(path: Path, cfg, device, dtype, *, progress=True) -> TextWeights:
    reader = SafetensorsReader(path, device)
    fp8_scales = {name for name in reader.weight_map if name.endswith(".weight_scale_inv")}
    if not any(name.startswith("model.language_model.") for name in fp8_scales):
        raise ValueError("The Qwen3.8 FP8 checkpoint has no FP8 text weight scales")

    def projection(name, shape):
        scale_name = f"{name}_scale_inv"
        values = reader.get(name)
        if tuple(values.shape) != shape:
            raise ValueError(f"{name} has shape {tuple(values.shape)}, expected {shape}")
        if scale_name not in fp8_scales:
            # Some projections in the official FP8 checkpoint remain unquantized.
            if values.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
                raise ValueError(f"Missing FP8 weight scale for {name}")
            return values.to(device=device, dtype=dtype)
        if shape[1] % 128:
            raise ValueError(f"{name} input dimension is not divisible by 128")
        if values.dtype != torch.float8_e4m3fn:
            raise ValueError(f"Invalid E4M3 weight tensor for {name}")
        scales = read(scale_name, ((shape[0] + 127) // 128, shape[1] // 128), torch.float32)
        return FP8Projection(values.to(device), scales)

    base = "model.language_model"
    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
    query_size = cfg.num_attention_heads * cfg.head_dim
    kv_size = cfg.num_key_value_heads * cfg.head_dim
    delta_value = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    delta_key = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    conv_channels = 2 * delta_key + delta_value

    def read(name, shape, weight_dtype=dtype):
        return load_tensor(reader, name, shape=shape, dtype=weight_dtype, device=device)

    with tqdm(total=len(cfg.layer_types) + 2, desc="Loading Qwen3.8 weights",
              unit="group", disable=not progress) as bar:
        embedding = read(f"{base}.embed_tokens.weight", (cfg.vocab_size, hidden))
        bar.update(1)
        layers = []
        for i, layer_type in enumerate(cfg.layer_types):
            prefix = f"{base}.layers.{i}"

            def layer(name, shape, weight_dtype=dtype):
                return read(f"{prefix}.{name}", shape, weight_dtype)

            def project(name, shape):
                return projection(f"{prefix}.{name}", shape)

            if layer_type == "full_attention":
                attention = FullAttentionWeights(
                    qkv=merge_projections(
                        project("self_attn.q_proj.weight", (2 * query_size, hidden)),
                        project("self_attn.k_proj.weight", (kv_size, hidden)),
                        project("self_attn.v_proj.weight", (kv_size, hidden)),
                    ),
                    query_norm=layer("self_attn.q_norm.weight", (cfg.head_dim,)),
                    key_norm=layer("self_attn.k_norm.weight", (cfg.head_dim,)),
                    output=project("self_attn.o_proj.weight", (hidden, query_size)),
                )
            else:
                heads = cfg.linear_num_value_heads
                attention = DeltaWeights(
                    qkvz=merge_projections(
                        project("linear_attn.in_proj_qkv.weight", (conv_channels, hidden)),
                        project("linear_attn.in_proj_z.weight", (delta_value, hidden)),
                    ),
                    ab=merge_projections(
                        project("linear_attn.in_proj_a.weight", (heads, hidden)),
                        project("linear_attn.in_proj_b.weight", (heads, hidden)),
                    ),
                    conv=layer("linear_attn.conv1d.weight", (conv_channels, 1, cfg.linear_conv_kernel_dim)).squeeze(1),
                    a_log=layer("linear_attn.A_log", (heads,), torch.float32),
                    dt_bias=layer("linear_attn.dt_bias", (heads,), torch.float32),
                    norm=layer("linear_attn.norm.weight", (cfg.linear_value_head_dim,)),
                    output=project("linear_attn.out_proj.weight", (hidden, delta_value)),
                )
            layers.append(TextLayerWeights(
                input_norm=layer("input_layernorm.weight", (hidden,)),
                post_attention_norm=layer("post_attention_layernorm.weight", (hidden,)),
                gate_up=merge_projections(
                    project("mlp.gate_proj.weight", (inter, hidden)),
                    project("mlp.up_proj.weight", (inter, hidden)),
                ),
                down=project("mlp.down_proj.weight", (hidden, inter)),
                attention=attention,
            ))
            bar.update(1)
        norm = read(f"{base}.norm.weight", (hidden,))
        lm_head = embedding if cfg.tie_word_embeddings else projection("lm_head.weight", (cfg.vocab_size, hidden))
        bar.update(1)

    return TextWeights(embedding, norm, lm_head, layers)
