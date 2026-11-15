"""Batch state for the hybrid text decoder."""

from dataclasses import dataclass

import torch

from paging.comm.kv_cache import PagedKVCache
from paging.comm.page_table import PageTable


@dataclass
class DeltaState:
    convolution: torch.Tensor
    # [value heads, value dimension, key dimension] keeps each warp's column contiguous.
    recurrent: torch.Tensor


@dataclass
class HybridCache:
    pages: PageTable
    kv: PagedKVCache
    delta: dict[int, DeltaState]
    lengths: list[int]
    logits: list[torch.Tensor]
