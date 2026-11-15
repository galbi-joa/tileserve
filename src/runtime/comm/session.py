from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BatchSession:
    """
    离线生成时每个 batch 的状态容器。

    设计非常精简：没有请求队列、没有多租户调度，仅记录 prompt/token 长度等。
    """

    # prompt_token_ids 存储每条序列的 token ids
    prompt_token_ids: list[list[int]]
    # max_new_tokens 是计划生成的额外 token 数（不含 prompt）
    max_new_tokens: int

    generated_token_ids: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if not isinstance(self.prompt_token_ids, list) or any(
            not isinstance(x, list) for x in self.prompt_token_ids
        ):
            raise TypeError("prompt_token_ids must be list[list[int]]")
        if self.generated_token_ids:
            return
        self.generated_token_ids = [[] for _ in self.prompt_token_ids]

    @property
    def batch_size(self) -> int:
        return len(self.prompt_token_ids)

    def total_lengths(self) -> list[int]:
        return [
            len(prompt) + len(gen)
            for prompt, gen in zip(self.prompt_token_ids, self.generated_token_ids)
        ]
