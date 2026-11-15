from __future__ import annotations

from dataclasses import dataclass

import torch

from .page_table import PageTable


@dataclass(frozen=True)
class KVLayout:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16


class PagedKVCache:
    """
    GPU-only 的 KV cache.

    形状:
      k_cache/v_cache: [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
    """

    def __init__(self, page_table: PageTable, layout: KVLayout):
        self.page_table = page_table
        self.layout = layout

        cfg = page_table.cfg
        device = page_table.device
        # 使用 zeros 而不是 empty。避免未写入slot中出现 NaN。
        self.k_cache = torch.zeros(
            (layout.num_layers, page_table.num_blocks, layout.num_kv_heads, cfg.block_size, layout.head_dim),
            device=device,
            dtype=layout.dtype,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
