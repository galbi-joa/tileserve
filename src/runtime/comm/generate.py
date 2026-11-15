from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Protocol

import torch

from .session import BatchSession


class NextTokenLogitsModel(Protocol):
    """
    Offline generator 需要的 简单的 definition，不提供实现。

    具体实现由 Qwen3Model 等提供，所以这部分看起来“空着”也是正常的。
    """

    def prefill_logits(self, prompt_token_ids: list[list[int]]) -> torch.Tensor:
        """
        使用整段 prompt 计算 logits。返回 [B, vocab] 的张量，是第一步要生成的 token 的分布。
        """

    def decode_logits(self, last_token_ids: list[int], total_lengths: list[int]) -> torch.Tensor:
        """
        每次只处理 1 token 的 decode，使用上一步 token + 当前长度，返回下一个 token 的 logits。
        """


@dataclass(frozen=True)
class GenerateResult:
    output_token_ids: list[list[int]]


@dataclass(frozen=True)
class SamplingConfig:
    """
    控制采样行为：

    - temperature <= 0：greedy（直接 argmax）
    - top_k == 0、top_p == 1.0：对应策略禁用
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    eos_token_id: int | tuple[int, ...] | None = None
    seed: int | None = None


def _validate_sampling(cfg: SamplingConfig) -> None:
    if not math.isfinite(cfg.temperature):
        raise ValueError("temperature must be finite")
    if not math.isfinite(cfg.top_p):
        raise ValueError("top_p must be finite")
    if cfg.temperature < 0:
        raise ValueError("temperature must be >= 0")
    if cfg.top_k < 0:
        raise ValueError("top_k must be >= 0")
    if cfg.top_p <= 0 or cfg.top_p > 1.0:
        raise ValueError("top_p must be in (0, 1]")


def sample_token_ids(
    logits: torch.Tensor, cfg: SamplingConfig, *, generator: torch.Generator | None
) -> torch.Tensor:
    """
    logits: [B, V]
    returns: token_ids [B] int64
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D [B, V], got shape={tuple(logits.shape)}")

    if cfg.temperature <= 0:
        # temperature <= 0 则直接取 argmax（greedy）
        return torch.argmax(logits, dim=-1).to(torch.int64)

    x = logits / float(cfg.temperature)

    if cfg.top_k > 0:
        k = min(int(cfg.top_k), x.shape[-1])
        values, _ = torch.topk(x, k=k, dim=-1)
        kth = values[..., -1, None]
        neg_inf = float("-inf")
        x = torch.where(x < kth, neg_inf, x)

    if cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        remove = cum > float(cfg.top_p)
        remove[..., 0] = False  # keep at least 1 token
        neg_inf = float("-inf")
        sorted_logits = torch.where(remove, neg_inf, sorted_logits)
        x = torch.full_like(x, neg_inf)
        x.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    probs = torch.softmax(x, dim=-1)
    if not torch.isfinite(probs).all():
        # 避免 NaN/Inf 传播，退化成 greedy
        return torch.argmax(logits, dim=-1).to(torch.int64)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1).to(torch.int64)


def generate(
    model: NextTokenLogitsModel,
    session: BatchSession,
    *,
    sampling: SamplingConfig,
    assert_finite: bool = False,
    on_token: Callable[[int, int], None] | None = None,
) -> GenerateResult:
    """
    离线生成主循环：
    1) prefill 一次，拿到每条序列第一 token 的 logits
    2) decode 循环：每步调用 decode_logits，采样新的 token，直到 max_new_tokens

    on_token receives the batch index and newly sampled token ID before the
    next model step. Each sequence's EOS token is included.
    """
    _validate_sampling(sampling)
    if session.max_new_tokens == 0:
        return GenerateResult(output_token_ids=[[] for _ in range(session.batch_size)])

    finished = [False for _ in range(session.batch_size)]
    generator: torch.Generator | None = None
    eos = sampling.eos_token_id
    eos_tokens = (eos,) if isinstance(eos, int) else eos or ()

    def append_step(token_ids: list[int]) -> None:
        for index, token_id in enumerate(token_ids):
            if finished[index]:
                continue
            session.generated_token_ids[index].append(token_id)
            finished[index] = token_id in eos_tokens
            if on_token is not None:
                on_token(index, token_id)

    first_logits = model.prefill_logits(session.prompt_token_ids)
    if assert_finite and not torch.isfinite(first_logits).all():
        raise RuntimeError("model.prefill_logits produced non-finite logits (NaN/Inf)")
    if first_logits.shape[0] != session.batch_size:
        raise RuntimeError("model.prefill_logits returned wrong batch size")

    if sampling.seed is not None:
        generator = torch.Generator(device=first_logits.device)
        generator.manual_seed(int(sampling.seed))

    append_step(sample_token_ids(first_logits, sampling, generator=generator).tolist())

    for _ in range(1, session.max_new_tokens):
        if all(finished):
            break

        last = [seq[-1] for seq in session.generated_token_ids]
        total_lengths = session.total_lengths()
        if getattr(model, "skips_finished_decode", False):
            # EOS is sampled and emitted, but never fed back into the model.
            total_lengths = [length - done for length, done in zip(total_lengths, finished, strict=True)]
        step_logits = model.decode_logits(last, total_lengths)
        if assert_finite and not torch.isfinite(step_logits).all():
            raise RuntimeError("model.decode_logits produced non-finite logits (NaN/Inf)")
        if step_logits.shape[0] != session.batch_size:
            raise RuntimeError("model.decode_logits returned wrong batch size")

        append_step(sample_token_ids(step_logits, sampling, generator=generator).tolist())

    return GenerateResult(output_token_ids=session.generated_token_ids)


def generate_greedy(model: NextTokenLogitsModel, session: BatchSession) -> GenerateResult:
    return generate(model, session, sampling=SamplingConfig(temperature=0.0))
