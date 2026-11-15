"""Measure batched prefill and cached decode for a local Qwen3.8 FP8 checkpoint."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from generate import add_model_arguments, load_model, resolve_model


def workload(batch_size, length, steps, vocab_size):
    def token(seq, position):
        return (17 + 67 * seq + 3571 * position) % vocab_size

    prompts = [[token(seq, i) for i in range(length)] for seq in range(batch_size)]
    decode = [([token(seq, i) for seq in range(batch_size)], [i + 1] * batch_size)
              for i in range(length, length + steps)]
    return prompts, decode


def measure(model, prompts, continuation):
    device = model.device
    torch.cuda.synchronize(device)
    start = perf_counter()
    logits = model.prefill_logits(prompts)
    torch.cuda.synchronize(device)
    prefill_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    for tokens, lengths in continuation:
        logits = model.decode_logits(tokens, lengths)
    torch.cuda.synchronize(device)
    decode_ms = (perf_counter() - start) * 1000
    if logits.shape[0] != len(prompts) or not torch.isfinite(logits).all():
        raise RuntimeError("Invalid decode logits")
    return {"prefill_ms": prefill_ms, "decode_ms": decode_ms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--backend", required=True, choices=("transformers", "tileserve"))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--prompt-tokens", nargs="+", type=int, default=[128, 1024, 2048])
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.batch_sizes, *args.prompt_tokens, args.decode_steps, args.warmups, args.repeats) < 1:
        parser.error("Batch sizes, lengths, and repetition counts must be positive")
    if len(set(args.batch_sizes)) != len(args.batch_sizes) or len(set(args.prompt_tokens)) != len(args.prompt_tokens):
        parser.error("Batch sizes and prompt lengths must be unique")
    device = torch.device(args.device)
    if device.type != "cuda" or args.dtype not in ("auto", "bf16"):
        parser.error("This benchmark measures FP8 weights with CUDA BF16 activations")
    if args.block_size is not None or args.qwen38_backend is not None:
        parser.error("This benchmark uses the default TileServe kernel configuration")

    path = resolve_model(args, parser)
    config = json.loads((path / "config.json").read_text())
    text_config = config["text_config"]
    max_len = max(args.prompt_tokens) + args.decode_steps
    if max_len > text_config["max_position_embeddings"]:
        parser.error("Prompt and decode lengths exceed the checkpoint context")
    args.dtype = "bf16"
    with torch.cuda.device(device), torch.inference_mode():
        if args.backend == "transformers":
            from transformers_reference import load_model as load_reference

            model = load_reference(path, device, torch.bfloat16)
        else:
            model = load_model(path, args, max_len, streaming=True)

        if args.backend == "transformers" or 1 not in args.batch_sizes:
            activation_precision = "dynamic e4m3"
        elif args.batch_sizes == [1]:
            activation_precision = "bf16"
        else:
            activation_precision = "bf16 at batch 1, dynamic e4m3 for multi-row FP8 projections"
        results = []
        for batch_size in args.batch_sizes:
            for length in args.prompt_tokens:
                prompts, continuation = workload(batch_size, length, args.decode_steps, text_config["vocab_size"])
                for _ in range(args.warmups):
                    measure(model, prompts, continuation)
                samples = [measure(model, prompts, continuation) for _ in range(args.repeats)]
                prefill_ms = median(s["prefill_ms"] for s in samples)
                decode_ms = median(s["decode_ms"] for s in samples)
                row = {
                    "batch_size": batch_size,
                    "prompt_tokens": length,
                    "prefill_ms": prefill_ms,
                    "prefill_tokens_per_second": 1000 * batch_size * length / prefill_ms,
                    "decode_ms_per_token": decode_ms / args.decode_steps,
                    "decode_tokens_per_second": 1000 * batch_size * args.decode_steps / decode_ms,
                    "samples": samples,
                }
                results.append(row)
                print(f"{args.backend} {path.name}, batch {batch_size}, {length} prompt tokens: "
                      f"prefill {prefill_ms:.2f} ms, decode {row['decode_tokens_per_second']:.2f} tok/s "
                      f"({row['decode_ms_per_token']:.2f} ms/token per request)", flush=True)

    report = {
        "model": path.name, "backend": args.backend, "dtype": args.dtype, "batch_sizes": args.batch_sizes,
        "weight_precision": "fp8",
        "decode_activation_precision": activation_precision,
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "transformers": version("transformers"), "tilelang": version("tilelang"),
        "attention": "sdpa" if args.backend == "transformers" else "tilelang",
        "decode_steps": args.decode_steps, "warmups": args.warmups, "repeats": args.repeats,
        "results": results,
    }
    if len(args.batch_sizes) == 1:
        report["batch_size"] = args.batch_sizes[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
