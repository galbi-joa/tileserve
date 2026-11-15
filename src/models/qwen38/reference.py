"""Text generation adapter for the hybrid Qwen3.8-27B-FP8 checkpoint."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import read_fp8_config


@dataclass(frozen=True)
class Qwen38RuntimeConfig:
    max_seq_len: int


class Qwen38ReferenceModel:
    """Use the reference model for attention and its convolution/recurrent caches."""

    def __init__(
        self, path: Path, device: torch.device, dtype: torch.dtype | None, max_seq_len: int
    ) -> None:
        from transformers import Qwen3_5ForConditionalGeneration

        read_fp8_config(path)
        self.device = device
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(path),
            local_files_only=True,
            dtype=dtype or torch.bfloat16,
            device_map={"": str(device)},
            attn_implementation="sdpa",
        ).eval()
        self.dtype = self._model.dtype
        self.cfg = self._model.config.text_config
        self.eos_token_ids = self._model.generation_config.eos_token_id
        self.runtime = Qwen38RuntimeConfig(
            max_seq_len=min(max_seq_len, self.cfg.max_position_embeddings)
        )
        self._caches = []
        self._lengths: list[int] = []
        self._logits: list[torch.Tensor] = []

    def _forward(self, ids: Sequence[int], cache=None):
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        output = self._model(
            input_ids=inputs, past_key_values=cache, use_cache=True, logits_to_keep=1
        )
        return output.logits[0, -1].float(), output.past_key_values

    def prefill_logits(self, prompt_token_ids: list[list[int]]) -> torch.Tensor:
        self._caches.clear()
        self._lengths = [len(ids) for ids in prompt_token_ids]
        self._logits.clear()
        for ids in prompt_token_ids:
            logits, cache = self._forward(ids)
            self._caches.append(cache)
            self._logits.append(logits)
        return torch.stack(self._logits)

    def decode_logits(self, last_token_ids: list[int], total_lengths: list[int]) -> torch.Tensor:
        for index, (token, length) in enumerate(zip(last_token_ids, total_lengths, strict=True)):
            if length == self._lengths[index]:
                continue
            if length != self._lengths[index] + 1:
                raise ValueError("Qwen3.8 decode requires consecutive token positions")
            self._logits[index], self._caches[index] = self._forward(
                [token], self._caches[index]
            )
            self._lengths[index] = length
        return torch.stack(self._logits)
