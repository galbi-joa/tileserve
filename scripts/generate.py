from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from typing import TYPE_CHECKING


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


from models.qwen38.config import read_fp8_config
from runtime.comm.generate import SamplingConfig, generate
from runtime.comm.session import BatchSession


FP8_CHECKPOINT = "Qwen3.8-27B-FP8"
INVALID_UNICODE = re.compile(r"[\ud800-\udfff]+")

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class TextOutputStream:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, batch_size: int) -> None:
        from tokenizers.decoders import DecodeStream

        self.backend = getattr(tokenizer, "backend_tokenizer", None)
        if self.backend is None:
            raise ValueError("Streaming requires a fast tokenizer")
        self.decoders = [DecodeStream(skip_special_tokens=True) for _ in range(batch_size)]
        self.emitted_chars = [0] * batch_size
        self.json_lines = batch_size > 1

    def write_delta(self, index: int, text: str) -> None:
        if not text:
            return
        if self.json_lines:
            print(json.dumps({"index": index, "text": text}, ensure_ascii=False), flush=True)
        else:
            print(text, end="", flush=True)
        self.emitted_chars[index] += len(text)

    def on_token(self, index: int, token_id: int) -> None:
        text = self.decoders[index].step(self.backend, token_id)
        if text is not None:
            self.write_delta(index, text)

    def finish(self, token_ids: list[list[int]], *, print_token_ids: bool) -> None:
        for index, ids in enumerate(token_ids):
            # DecodeStream can buffer incomplete UTF-8 at the generation limit.
            text = self.backend.decode(ids, skip_special_tokens=True)
            self.write_delta(index, text[self.emitted_chars[index] :])
        if not self.json_lines:
            print(flush=True)
        if print_token_ids:
            for index, ids in enumerate(token_ids):
                print(json.dumps({"index": index, "token_ids": ids}), file=sys.stderr)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", type=str.casefold, choices=("qwen3.8-fp8", FP8_CHECKPOINT.casefold()),
                       help="Use the Qwen3.8 FP8 checkpoint under model/.")
    model.add_argument("--model-path", type=Path, help="Use a model directory at an explicit path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16"), default="auto",
                        help="Activation dtype. The checkpoint projections remain FP8 where quantized.")
    parser.add_argument("--block-size", type=int, help="KV page size. Defaults to 32.")
    parser.add_argument(
        "--qwen38-backend", choices=("transformers", "tilelang"),
        help="Qwen3.8 text backend. Defaults to TileLang on CUDA and Transformers on other devices.",
    )


def resolve_model(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    model_path = args.model_path if args.model is None else _ROOT / "model" / FP8_CHECKPOINT
    try:
        read_fp8_config(model_path)
    except OSError as exc:
        parser.error(f"Cannot read model configuration at {model_path / 'config.json'}: {exc.strerror}")
    except ValueError as exc:
        parser.error(str(exc))
    return model_path


def load_model(model_path, args, max_seq_len, *, streaming=False):
    import torch

    device = torch.device(args.device)
    dtype = {"auto": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    backend = args.qwen38_backend or ("tilelang" if device.type == "cuda" else "transformers")
    if backend == "tilelang":
        from models.qwen38.model import Qwen38Model

        if streaming:
            configure_tilelang_logging()
        page_size = 32 if args.block_size is None else args.block_size
        return Qwen38Model(model_path, device, dtype, max_seq_len, page_size=page_size)

    from models.qwen38.reference import Qwen38ReferenceModel

    return Qwen38ReferenceModel(model_path, device, dtype, max_seq_len)


def configure_tilelang_logging() -> None:
    # stdout and stderr share the terminal during interactive streaming.
    logger = logging.getLogger("tilelang")
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    logger.handlers[:] = [handler]


@contextmanager
def utf8_terminal():
    """Enable UTF-8-aware Backspace in the Linux terminal line discipline."""
    if sys.platform != "linux" or not sys.stdin.isatty():
        yield
        return

    import termios

    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    # Python does not always expose termios.IUTF8, including on the RunPod image.
    try:
        changed = subprocess.run(
            ["stty", "iutf8"], stdin=sys.stdin, check=False, stderr=subprocess.DEVNULL
        ).returncode == 0
    except FileNotFoundError:
        changed = False
    try:
        yield
    finally:
        if changed:
            termios.tcsetattr(fd, termios.TCSANOW, original)


def read_chat_line():
    if sys.stdin.isatty():
        raw = sys.stdin.buffer.readline()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            line = raw.decode("utf-8", errors="replace")
    else:
        line, replacements = INVALID_UNICODE.subn("\ufffd", sys.stdin.readline())
        if not replacements:
            return line
    print("Some unreadable input bytes were replaced with �.", file=sys.stderr)
    return line


def chat_tokens(tokenizer, messages, *, think):
    try:
        for message in messages:
            for field in ("content", "reasoning_content"):
                if field in message:
                    message[field].encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Chat contains invalid UTF-8. Re-enter the message using a UTF-8 terminal.") from exc

    options = {} if think is None else {"enable_thinking": think}
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False, **options
    )


def run_turn(model, tokenizer, args, prompts, prompt_ids, sampling, *, show_prompts=False):
    import torch

    budget = model.runtime.max_seq_len - max(map(len, prompt_ids))
    if args.max_new_tokens is not None:
        budget = min(budget, args.max_new_tokens)
    if budget <= 0 and args.max_new_tokens != 0:
        raise ValueError(f"Prompt fills or exceeds the {model.runtime.max_seq_len}-token context")
    if budget < 0:
        raise ValueError(f"Prompt exceeds the {model.runtime.max_seq_len}-token context")

    stream = TextOutputStream(tokenizer, len(prompts)) if args.stream else None
    with torch.inference_mode():
        result = generate(
            model,
            BatchSession(prompt_token_ids=prompt_ids, max_new_tokens=budget),
            sampling=sampling,
            on_token=stream.on_token if stream is not None else None,
        )

    texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in result.output_token_ids]
    if stream is not None:
        stream.finish(result.output_token_ids, print_token_ids=args.print_token_ids)
        return texts

    for prompt, ids, response in zip(prompts, result.output_token_ids, texts):
        if show_prompts:
            print("-----")
            print(prompt)
        if args.print_token_ids:
            print(ids)
        print(response)
    return texts


def interactive_chat(model, tokenizer, args, sampling):
    history = []
    print("Enter a message. Press Ctrl-D to exit.", file=sys.stderr)
    while True:
        print("You: ", end="", file=sys.stderr, flush=True)
        line = read_chat_line()
        if not line:
            print(file=sys.stderr)
            return
        prompt = line.strip()
        if not prompt:
            continue

        messages = history + [{"role": "user", "content": prompt}]
        try:
            prompt_ids = chat_tokens(tokenizer, messages, think=args.think)
            print("Assistant: ", end="", file=sys.stderr, flush=True)
            response, = run_turn(model, tokenizer, args, [prompt], [prompt_ids], sampling)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            continue
        assistant_message = {"role": "assistant", "content": response}
        if "</think>" in response:
            reasoning, answer = response.split("</think>", 1)
            assistant_message["content"] = answer.strip()
            assistant_message["reasoning_content"] = reasoning.removeprefix("<think>").strip()
        history = messages + [assistant_message]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--prompt", action="append", default=[], help="Supply text directly. Repeat for a batch.")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Open an interactive streaming chat, or read one prompt from piped stdin.",
    )
    parser.add_argument("--max-new-tokens", type=int, help="Limit generated tokens per response.")
    parser.add_argument("--warmup", action="store_true", help="Prepare TileLang kernels before the first request.")
    parser.add_argument(
        "--think", action=argparse.BooleanOptionalAction,
        help="Open or close the reasoning section in chat prompts. Defaults to the model template.",
    )
    parser.add_argument("--max-seq-len", type=int, default=0, help="Override the context capacity.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy, >0 = sampling.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--stream", action=argparse.BooleanOptionalAction,
        help="Emit text incrementally. Enabled by default in chat mode.",
    )
    parser.add_argument("--print-token-ids", action="store_true", help="Print generated token IDs.")
    args = parser.parse_args()
    if args.max_new_tokens is not None and args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be nonnegative")
    if args.max_seq_len < 0:
        parser.error("--max-seq-len must be nonnegative")
    if args.think is not None and not args.chat:
        parser.error("--think and --no-think require --chat")
    if args.stream is None:
        args.stream = args.chat

    model_path = resolve_model(args, parser)
    interactive = args.chat and not args.prompt and sys.stdin.isatty()
    if args.chat and not args.prompt and not interactive:
        pipe_input = sys.stdin.read().strip()
        if pipe_input:
            args.prompt = [pipe_input]
    if not interactive and not args.prompt:
        parser.error("Supply --prompt or use --chat with a terminal or piped input")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    prompt_ids = []
    if not interactive:
        if args.chat:
            try:
                prompt_ids = [
                    chat_tokens(tokenizer, [{"role": "user", "content": prompt}],
                                think=args.think)
                    for prompt in args.prompt
                ]
            except ValueError as exc:
                parser.error(str(exc))
        else:
            prompt_ids = tokenizer(args.prompt, return_attention_mask=False)["input_ids"]
        if any(not ids for ids in prompt_ids):
            parser.error("Each prompt must contain at least one token")

    max_prompt_len = max(map(len, prompt_ids), default=0)
    max_seq_len = args.max_seq_len or max(4096, max_prompt_len)
    if args.max_new_tokens is not None and not interactive:
        needed = max_prompt_len + args.max_new_tokens
        if args.max_seq_len and needed > args.max_seq_len:
            parser.error(f"--max-seq-len must be at least {needed} for this request")
        max_seq_len = args.max_seq_len or needed

    model = load_model(model_path, args, max_seq_len, streaming=args.stream)
    eos = getattr(model, "eos_token_ids", None)
    if eos is None:
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else model.cfg.eos_token_id
    sampling = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_token_id=tuple(eos) if isinstance(eos, list) else eos,
        seed=args.seed if args.seed is not None and args.seed >= 0 else None,
    )
    try:
        if args.warmup:
            import torch

            prepare = getattr(model, "prepare_kernels", None)
            if prepare is None or model.device.type != "cuda":
                parser.error("--warmup requires a CUDA TileLang model")
            print("Preparing TileLang kernels…", file=sys.stderr, flush=True)
            start = perf_counter()
            with torch.cuda.device(model.device):
                prepare(batch_size=1 if interactive else len(prompt_ids))
            print(f"TileLang kernels ready in {perf_counter() - start:.1f}s.", file=sys.stderr, flush=True)
        if interactive:
            with utf8_terminal():
                interactive_chat(model, tokenizer, args, sampling)
        else:
            run_turn(model, tokenizer, args, args.prompt, prompt_ids, sampling, show_prompts=not args.chat)
    except KeyboardInterrupt:
        print(file=sys.stderr)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
