"""Transformers batch reference for Qwen3.8-27B-FP8."""

import torch

from models.qwen38.config import read_fp8_config


class TransformersReference:
    skips_finished_decode = True

    def __init__(self, model_class, path, device, dtype):
        self.device = device
        self.model = model_class.from_pretrained(
            str(path), local_files_only=True, device_map={"": str(device)},
            dtype=dtype, attn_implementation="sdpa",
        ).eval()
        self.cache = None

    def _forward(self, input_ids, attention_mask=None, position_ids=None):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=self.cache, use_cache=True, logits_to_keep=1,
        )
        self.cache = output.past_key_values
        return output.logits[:, -1].float()

    def prefill_logits(self, prompt_token_ids):
        self.cache = None
        self.lengths = [len(ids) for ids in prompt_token_ids]
        self.active = list(range(len(self.lengths)))
        self.mask = None
        if len(self.lengths) == 1:
            ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device)
            self.logits = self._forward(ids)
            return self.logits

        width = max(self.lengths)
        pad = self.model.generation_config.pad_token_id
        pad = 0 if pad is None else pad
        ids = torch.tensor([[pad] * (width - len(row)) + row for row in prompt_token_ids],
                           dtype=torch.long, device=self.device)
        lengths = torch.tensor(self.lengths, device=self.device)
        self.mask = torch.arange(width, device=self.device)[None, :] >= (width - lengths[:, None])
        positions = (self.mask.long().cumsum(dim=-1) - 1).clamp_min(0)
        self.logits = self._forward(ids, self.mask, positions)
        return self.logits

    def decode_logits(self, last_token_ids, total_lengths):
        active = [idx for idx, (length, previous) in enumerate(zip(total_lengths, self.lengths, strict=True))
                  if length > previous]
        if not active:
            return self.logits
        if active != self.active:
            keep = torch.tensor([self.active.index(idx) for idx in active], device=self.device)
            self.cache.reorder_cache(keep)
            self.mask = self.mask.index_select(0, keep)
            self.active = active

        ids = torch.tensor([[last_token_ids[idx]] for idx in active], dtype=torch.long, device=self.device)
        positions = None
        if self.mask is not None:
            self.mask = torch.cat((self.mask, self.mask.new_ones(len(active), 1)), dim=1)
            positions = torch.tensor([[total_lengths[idx] - 1] for idx in active],
                                     dtype=torch.long, device=self.device)
        result = self._forward(ids, self.mask, positions)
        self.lengths = list(total_lengths)
        if len(active) == len(total_lengths):
            self.logits = result
        else:
            output_rows = torch.tensor(active, device=self.device)
            self.logits = self.logits.index_copy(0, output_rows, result)
        return self.logits


def load_model(path, device, dtype):
    from transformers import Qwen3_5ForConditionalGeneration

    read_fp8_config(path)
    return TransformersReference(Qwen3_5ForConditionalGeneration, path, device, dtype)
