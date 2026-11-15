"""TileLang backend for Qwen3.8 FP8 text generation."""

from itertools import accumulate
import json
import math
from pathlib import Path
from types import SimpleNamespace

import tilelang.language as T
import torch

from tileserve_kernels.comm.attn_decode_paged import gqa_decode_paged_attention
from tileserve_kernels.comm.attn_decode_split_paged import split_gqa_decode
from tileserve_kernels.comm.attn_prefill_varlen import gqa_prefill_varlen
from tileserve_kernels.comm.swiglu import swiglu
from tileserve_kernels.qwen38.delta import causal_conv, delta_inputs, recurrent_delta
from tileserve_kernels.qwen38.kvcache_store_paged import kvcache_store_paged
from tileserve_kernels.qwen38.fp8 import fp8_gemm, fp8_gemv, quantize_fp8
from tileserve_kernels.qwen38.linear import gemm, gemv
from tileserve_kernels.qwen38.pointwise import attention_gate, embedding, normalize, residual_add, rotate_partial
from paging.comm.kv_cache import KVLayout, PagedKVCache
from paging.comm.page_table import PageTable, PagingConfig
from paging.qwen38.hybrid import DeltaState, HybridCache
from .config import read_fp8_config
from .weights import DeltaWeights, FP8Projection, load_text_weights


class Qwen38Model:
    skips_finished_decode = True

    def __init__(self, path: Path, device, dtype, max_seq_len, *, page_size=32):
        if device.type != "cuda":
            raise ValueError("The Qwen3.8 TileLang backend requires an NVIDIA CUDA device")
        if page_size <= 0 or page_size % 16:
            raise ValueError("Qwen3.8 page size must be a positive multiple of 16")
        config = read_fp8_config(path)
        self.cfg = cfg = SimpleNamespace(**config["text_config"])
        rope = cfg.rope_parameters
        rotary_dim = int(cfg.head_dim * rope["partial_rotary_factor"])
        if (cfg.hidden_act != "silu" or cfg.attention_bias or not cfg.attn_output_gate
                or rope["rope_type"] != "default" or rotary_dim <= 0 or rotary_dim % 2
                or cfg.linear_key_head_dim % 32 or cfg.linear_value_head_dim % 4
                or cfg.linear_num_value_heads % cfg.linear_num_key_heads
                or cfg.linear_conv_kernel_dim < 2
                or len(cfg.layer_types) != cfg.num_hidden_layers
                or set(cfg.layer_types) != {"full_attention", "linear_attention"}):
            raise ValueError("Unsupported Qwen3.8 text architecture")
        self.device = device
        self.dtype = dtype or torch.bfloat16
        self.tl_dtype = tl_dtype = T.bfloat16 if self.dtype == torch.bfloat16 else T.float16
        self.runtime = SimpleNamespace(max_seq_len=min(max_seq_len, cfg.max_position_embeddings))
        self.page_size = page_size
        self.num_pages = math.ceil(self.runtime.max_seq_len / page_size)
        self.full_layers = sum(kind == "full_attention" for kind in cfg.layer_types)
        self.query_size = cfg.num_attention_heads * cfg.head_dim
        self.kv_size = cfg.num_key_value_heads * cfg.head_dim
        self.value_size = cfg.linear_num_value_heads * cfg.linear_value_head_dim
        self.conv_channels = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + self.value_size
        eos_path = path / "generation_config.json"
        eos_config = json.loads(eos_path.read_text()) if eos_path.is_file() else {}
        self.eos_token_ids = eos_config.get("eos_token_id", cfg.eos_token_id)

        self.weights = load_text_weights(path, cfg, device, self.dtype)
        inv_freq = rope["rope_theta"] ** (
            -torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim
        )
        angles = torch.outer(torch.arange(self.runtime.max_seq_len, device=device, dtype=torch.float32), inv_freq)
        self.cos, self.sin = angles.cos().to(self.dtype), angles.sin().to(self.dtype)

        self.linear_kernels = {}
        self.fp8_quantizers = {}
        self.embed = embedding(cfg.vocab_size, cfg.hidden_size, tl_dtype)
        self.add_residual = residual_add(cfg.hidden_size, tl_dtype)
        self.norm_hidden = normalize(cfg.hidden_size, cfg.rms_norm_eps, tl_dtype)
        self.norm_head = normalize(cfg.head_dim, cfg.rms_norm_eps, tl_dtype)
        self.norm_delta = normalize(cfg.linear_value_head_dim, cfg.rms_norm_eps, tl_dtype, gated=True)
        self.mlp_activation = swiglu(cfg.intermediate_size, tl_dtype)
        self.rotate_query = rotate_partial(cfg.num_attention_heads, cfg.head_dim, rotary_dim, tl_dtype)
        self.rotate_key = rotate_partial(cfg.num_key_value_heads, cfg.head_dim, rotary_dim, tl_dtype)
        self.gate = attention_gate(self.query_size, tl_dtype)
        self.convolve = {
            decode: causal_conv(self.conv_channels, cfg.linear_conv_kernel_dim, tl_dtype, decode=decode)
            for decode in (False, True)
        }
        delta_dims = (cfg.linear_num_key_heads, cfg.linear_num_value_heads,
                      cfg.linear_key_head_dim, cfg.linear_value_head_dim, tl_dtype)
        self.prepare_delta = delta_inputs(*delta_dims)
        self.run_delta = {
            False: recurrent_delta(*delta_dims, initialize=True, warps=4),
            True: recurrent_delta(*delta_dims, decode=True, warps=8),
        }
        self.store = kvcache_store_paged(
            cfg.num_key_value_heads, cfg.head_dim, page_size, 1, self.num_pages, tl_dtype,
        )
        self.prefill_attention = {}
        self.prefill_store = {1: self.store}
        self.packed_convolution = None
        self.packed_delta = None
        self.delta_dims = delta_dims
        self.decode_attention = {}
        self.caches: list[HybridCache] = []

    def _get_prefill_kernels(self, batch_size):
        cfg = self.cfg
        if batch_size not in self.prefill_attention:
            self.prefill_attention[batch_size] = gqa_prefill_varlen(
                batch_size, cfg.num_attention_heads // cfg.num_key_value_heads, None, None,
                cfg.num_attention_heads, cfg.head_dim, True,
                block_M=32, block_N=64, threads=32, dtype=self.tl_dtype,
            )
        if batch_size not in self.prefill_store:
            self.prefill_store[batch_size] = kvcache_store_paged(
                cfg.num_key_value_heads, cfg.head_dim, self.page_size, batch_size,
                self.num_pages, self.tl_dtype,
            )
        if batch_size > 1 and self.packed_convolution is None:
            self.packed_convolution = causal_conv(
                self.conv_channels, cfg.linear_conv_kernel_dim, self.tl_dtype, packed=True,
            )
            self.packed_delta = recurrent_delta(*self.delta_dims, initialize=True, packed=True, warps=4)

    def _new_cache(self, lengths):
        cfg = self.cfg
        batch_size = len(lengths)
        table = PageTable(
            batch_size * self.num_pages, PagingConfig(self.page_size, batch_size, self.num_pages), self.device,
        )
        kv = PagedKVCache(table, KVLayout(self.full_layers, cfg.num_key_value_heads, cfg.head_dim, self.dtype))
        delta = {
            i: DeltaState(
                torch.empty((batch_size, cfg.linear_conv_kernel_dim - 1, self.conv_channels), device=self.device, dtype=self.dtype),
                torch.empty((batch_size, cfg.linear_num_value_heads, cfg.linear_value_head_dim, cfg.linear_key_head_dim),
                            device=self.device, dtype=torch.float32),
            )
            for i, kind in enumerate(cfg.layer_types) if kind == "linear_attention"
        }
        return HybridCache(table, kv, delta, lengths, [])

    def _get_linear_kernel(self, rows, weight):
        out_features, in_features = weight.shape
        fp8 = isinstance(weight, FP8Projection)
        tile = 0 if rows == 1 else (16 if rows <= 16 else 64) if fp8 else (64 if rows <= 128 else 128)
        key = (fp8, tile, out_features, in_features)
        if key not in self.linear_kernels:
            dims = (out_features, in_features, self.tl_dtype)
            if fp8 and tile:
                if in_features not in self.fp8_quantizers:
                    self.fp8_quantizers[in_features] = quantize_fp8(in_features, self.tl_dtype)
                self.linear_kernels[key] = fp8_gemm(*dims, block_m=tile)
            elif fp8:
                self.linear_kernels[key] = fp8_gemv(*dims)
            elif tile:
                self.linear_kernels[key] = gemm(*dims, block_m=tile, block_n=tile)
            else:
                self.linear_kernels[key] = gemv(*dims)
        return self.linear_kernels[key]

    def _linear(self, x, weight):
        rows, width = x.shape
        kernel = self._get_linear_kernel(rows, weight)
        x = x.contiguous()
        if not isinstance(weight, FP8Projection):
            return kernel(x, weight)
        if rows == 1:
            return kernel(x, weight.values, weight.scales)
        quantized, scales = self.fp8_quantizers[width](x)
        return kernel(quantized, weight.values, scales, weight.scales)

    def prepare_kernels(self, *, batch_size: int) -> None:
        self._get_prefill_kernels(batch_size)
        self._get_linear_kernel(batch_size, self.weights.lm_head)
        self._get_linear_kernel(1, self.weights.lm_head)
        row_sizes = (1, 2, 17, 129)
        for rows in row_sizes:
            if rows > batch_size * self.runtime.max_seq_len:
                break
            for layer in self.weights.layers:
                attention = layer.attention
                projections = (attention.qkvz, attention.ab) if isinstance(
                    attention, DeltaWeights
                ) else (attention.qkv,)
                for weight in (*projections, attention.output, layer.gate_up, layer.down):
                    self._get_linear_kernel(rows, weight)
        for length in (2, 512, 2048):
            if length <= self.runtime.max_seq_len:
                for active_size in range(1, batch_size + 1):
                    self._get_decode_attention(length, active_size)

    def _delta(self, x, weights, state, seq_ids, boundaries, *, packed, decode):
        rows = x.shape[0]
        qkv, z = self._linear(x, weights.qkvz).split((self.conv_channels, self.value_size), dim=-1)
        convolve = self.packed_convolution if packed else self.convolve[decode]
        qkv = convolve(qkv.contiguous(), weights.conv, state.convolution, seq_ids, boundaries)
        ab = self._linear(x, weights.ab)
        q, k, decay, beta = self.prepare_delta(qkv, ab, weights.a_log, weights.dt_bias)
        delta = self.packed_delta if packed else self.run_delta[decode]
        mixed = delta(q, k, qkv, decay, beta, state.recurrent, boundaries, seq_ids)
        dim = self.cfg.linear_value_head_dim
        mixed = self.norm_delta(mixed.view(-1, dim), weights.norm, z.reshape(-1, dim).contiguous())
        return self._linear(mixed.view(rows, self.value_size), weights.output)

    def _get_decode_attention(self, length, batch_size=1):
        cfg = self.cfg
        splits = (16 if length < 2048 else 32) if length >= 512 else 0
        key = (batch_size, splits)
        if key not in self.decode_attention:
            args = (batch_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                    cfg.head_dim, self.page_size, self.num_pages)
            if splits:
                kernel = split_gqa_decode(*args, splits, self.tl_dtype)
            else:
                kernel = gqa_decode_paged_attention(*args, block_H=16, threads=32, dtype=self.tl_dtype)
            self.decode_attention[key] = kernel
        return self.decode_attention[key]

    def _attention(self, x, weights, cache, layer, positions, seq_ids, boundaries,
                   max_length, decode_table, cache_length):
        cfg = self.cfg
        rows = x.shape[0]
        qg, k, v = self._linear(x, weights.qkv).split((2 * self.query_size, self.kv_size, self.kv_size), dim=-1)
        q, gate = qg.view(rows, cfg.num_attention_heads, 2, cfg.head_dim).unbind(dim=2)
        q = q.reshape(-1, cfg.head_dim).contiguous()
        k = k.reshape(-1, cfg.head_dim).contiguous()
        v = v.reshape(rows, cfg.num_key_value_heads, cfg.head_dim).contiguous()
        q = self.norm_head(q, weights.query_norm, q).view(rows, cfg.num_attention_heads, cfg.head_dim)
        k = self.norm_head(k, weights.key_norm, k).view(rows, cfg.num_key_value_heads, cfg.head_dim)
        self.rotate_query(q, self.cos, self.sin, positions)
        self.rotate_key(k, self.cos, self.sin, positions)
        table = cache.pages.table
        kc, vc = cache.kv.k_cache[layer], cache.kv.v_cache[layer]
        if decode_table is None:
            batch_size = len(cache.lengths)
            self.prefill_store[batch_size](seq_ids, positions, k, v, table, kc, vc)
            output = self.prefill_attention[batch_size](q, k, v, boundaries, boundaries, max_length)
        else:
            self.prefill_store[len(cache.lengths)](seq_ids, positions, k, v, table, kc, vc)
            output = self._get_decode_attention(max_length, rows)(q, kc, vc, decode_table, cache_length)
        mixed = self.gate(output.view(rows, self.query_size), gate.reshape(rows, self.query_size).contiguous())
        return self._linear(mixed, weights.output)

    def _forward(self, ids, cache, positions, seq_ids, boundaries, max_length, *, decode_table=None):
        decode = decode_table is not None
        packed = not decode and len(cache.lengths) > 1
        if any(token < 0 or token >= self.cfg.vocab_size for token in ids):
            raise ValueError("A token ID is outside the Qwen3.8 vocabulary")
        tokens = torch.tensor(ids, dtype=torch.int32, device=self.device)
        x = self.embed(tokens, self.weights.embedding)
        cache_length = positions + 1 if decode else None
        full_layer = 0
        for i, layer in enumerate(self.weights.layers):
            norm = self.norm_hidden(x, layer.input_norm, x)
            if isinstance(layer.attention, DeltaWeights):
                state = cache.delta[i]
                mix = self._delta(norm, layer.attention, state, seq_ids, boundaries,
                                  packed=packed, decode=decode)
            else:
                mix = self._attention(norm, layer.attention, cache, full_layer, positions, seq_ids,
                                      boundaries, max_length, decode_table, cache_length)
                full_layer += 1
            x = self.add_residual(x, mix)
            norm = self.norm_hidden(x, layer.post_attention_norm, x)
            gate, up = self._linear(norm, layer.gate_up).chunk(2, dim=-1)
            update = self._linear(self.mlp_activation(gate.contiguous(), up.contiguous()), layer.down)
            x = self.add_residual(x, update)

        if packed:
            last = x.index_select(0, (boundaries[1:] - 1).long())
        else:
            last = x if decode else x[-1:].contiguous()
        last = self.norm_hidden(last, self.weights.norm, last)
        return self._linear(last, self.weights.lm_head).float()

    def _prefill_batch(self, prompts):
        lengths = [len(ids) for ids in prompts]
        if min(lengths) <= 0:
            raise ValueError("Empty prompts are not supported")
        max_length = max(lengths)
        if max_length > self.runtime.max_seq_len:
            raise ValueError("Qwen3.8 sequence length exceeds the configured context")
        self._get_prefill_kernels(len(lengths))
        cache = self._new_cache(lengths)
        cache.pages.allocate_batch_upto(lengths)
        device = self.device
        ids = [token for prompt in prompts for token in prompt]
        positions = torch.tensor([pos for length in lengths for pos in range(length)], dtype=torch.int32, device=device)
        seq_ids = torch.tensor([i for i, length in enumerate(lengths) for _ in range(length)],
                               dtype=torch.int32, device=device)
        boundaries = torch.tensor(list(accumulate(lengths, initial=0)), dtype=torch.int32, device=device)
        logits = self._forward(ids, cache, positions, seq_ids, boundaries, max_length)
        cache.logits = list(logits.unbind(dim=0))
        return logits, cache

    def prefill_logits(self, prompt_token_ids):
        self.caches.clear()
        if not prompt_token_ids:
            return torch.empty((0, self.cfg.vocab_size), device=self.device, dtype=torch.float32)
        logits, cache = self._prefill_batch(prompt_token_ids)
        self.caches.append(cache)
        return logits

    def prefill_logits_sequential(self, prompt_token_ids):
        """Run one prefill per request for comparison with packed prefill."""
        self.caches.clear()
        logits = []
        for prompt in prompt_token_ids:
            output, cache = self._prefill_batch([prompt])
            self.caches.append(cache)
            logits.append(output[0])
        if not logits:
            return torch.empty((0, self.cfg.vocab_size), device=self.device, dtype=torch.float32)
        return torch.stack(logits)

    def decode_logits(self, last_token_ids, total_lengths):
        return self._decode_batch(last_token_ids, total_lengths, sequential=False)

    def decode_logits_sequential(self, last_token_ids, total_lengths):
        """Use the same prefill cache, decoding one active sequence at a time."""
        return self._decode_batch(last_token_ids, total_lengths, sequential=True)

    def _decode_batch(self, last_token_ids, total_lengths, *, sequential):
        previous_lengths = [length for cache in self.caches for length in cache.lengths]
        if len(last_token_ids) != len(previous_lengths) or len(total_lengths) != len(previous_lengths):
            raise ValueError("Qwen3.8 decode batch does not match prefill")
        for token, length, previous in zip(last_token_ids, total_lengths, previous_lengths, strict=True):
            if length != previous and length != previous + 1:
                raise ValueError("Qwen3.8 decode requires consecutive token positions")
            if length > self.runtime.max_seq_len:
                raise ValueError("Qwen3.8 sequence length exceeds the configured context")
            if length > previous and not 0 <= token < self.cfg.vocab_size:
                raise ValueError("A token ID is outside the Qwen3.8 vocabulary")
        if not previous_lengths:
            return torch.empty((0, self.cfg.vocab_size), device=self.device, dtype=torch.float32)
        offset = 0
        for cache in self.caches:
            size = len(cache.lengths)
            lengths = total_lengths[offset:offset + size]
            ids = last_token_ids[offset:offset + size]
            active = [seq for seq, (length, previous) in enumerate(zip(lengths, cache.lengths, strict=True))
                      if length > previous]
            groups = ([seq] for seq in active) if sequential else (active,)
            for group in groups:
                if group:
                    self._decode_active(cache, ids, lengths, group)
            offset += size
        return torch.stack([logit for cache in self.caches for logit in cache.logits])

    def _decode_active(self, cache, ids, lengths, active):
        device = self.device
        positions = torch.tensor([cache.lengths[seq] for seq in active], dtype=torch.int32, device=device)
        seq_ids = torch.tensor(active, dtype=torch.int32, device=device)
        boundaries = torch.arange(len(cache.lengths) + 1, dtype=torch.int32, device=device)
        for seq in active:
            cache.pages.allocate_sequence_upto(seq, lengths[seq])
        table = cache.pages.table
        if len(active) != len(cache.lengths):
            table = table.index_select(0, seq_ids)
        output = self._forward(
            [ids[seq] for seq in active], cache, positions, seq_ids, boundaries,
            max(lengths[seq] for seq in active), decode_table=table,
        )
        for seq, logits in zip(active, output.unbind(dim=0), strict=True):
            cache.logits[seq] = logits
            cache.lengths[seq] = lengths[seq]
