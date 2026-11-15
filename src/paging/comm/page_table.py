from __future__ import annotations

import heapq
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PagingConfig:
    block_size: int = 16
    max_seqs: int = 128
    max_blocks_per_seq: int = 8192


class PageTable:

    def __init__(self, num_blocks: int, cfg: PagingConfig, device: torch.device):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if cfg.block_size <= 0:
            raise ValueError("block_size must be > 0")
        self.cfg = cfg
        self.device = device
        self.num_blocks = int(num_blocks)

        self.table = torch.full(
            (cfg.max_seqs, cfg.max_blocks_per_seq),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.num_seq_blocks = torch.zeros((cfg.max_seqs,), dtype=torch.int32, device=device)
        self.is_free = torch.ones((self.num_blocks,), dtype=torch.bool, device=device)
        self._blocks_by_sequence: list[list[int]] = [[] for _ in range(cfg.max_seqs)]
        self._free_blocks = list(range(self.num_blocks))

    def allocate_upto(self, seq_ids: torch.Tensor, seq_lens: torch.Tensor) -> None:

        if seq_ids.numel() == 0:
            return
        if seq_ids.dtype != torch.int32 or seq_lens.dtype != torch.int32:
            raise ValueError("seq_ids/seq_lens must be int32")

        self._allocate(zip(seq_ids.tolist(), seq_lens.tolist()))

    def allocate_batch_upto(self, seq_lens: Sequence[int]) -> None:
        self._allocate(enumerate(seq_lens))

    def allocate_sequence_upto(self, seq_id: int, seq_len: int) -> None:
        self._allocate(((seq_id, seq_len),))

    def _allocate(self, requests: Iterable[tuple[int, int]]) -> None:
        for seq_id, seq_len in requests:
            blocks = self._blocks_by_sequence[seq_id]
            have = len(blocks)
            need = (seq_len + self.cfg.block_size - 1) // self.cfg.block_size
            if need > self.cfg.max_blocks_per_seq:
                raise RuntimeError("sequence exceeds max_blocks_per_seq")
            extra = need - have
            if extra <= 0:
                continue
            if extra > len(self._free_blocks):
                raise RuntimeError("out of KV cache blocks (GPU-only; no CPU offload swap)")
            added = [heapq.heappop(self._free_blocks) for _ in range(extra)]
            indices = torch.tensor(added, dtype=torch.int32, device=self.device)
            self.table[seq_id, have:need] = indices
            self.is_free[indices] = False
            self.num_seq_blocks[seq_id] = need
            blocks.extend(added)

    def free(self, seq_ids: torch.Tensor) -> None:
        if seq_ids.numel() == 0:
            return
        for seq_id in seq_ids.tolist():
            blocks = self._blocks_by_sequence[seq_id]
            if not blocks:
                continue
            indices = torch.tensor(blocks, dtype=torch.int32, device=self.device)
            self.is_free[indices] = True
            self.table[seq_id, :len(blocks)] = -1
            self.num_seq_blocks[seq_id] = 0
            for block in blocks:
                heapq.heappush(self._free_blocks, block)
            blocks.clear()
