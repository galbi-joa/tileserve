"""Profile the official MiniMax-H3 text-to-video-and-audio checkpoint on one CUDA GPU."""

import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack, nullcontext
from functools import wraps
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
from statistics import median
import sys
from time import perf_counter
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.minimax_h3.video.checkpoint import validate_video_checkpoint
from models.minimax_h3.video.media import encode_mp4, prepare_media


PROMPT = "A red fox trots through a snowy pine forest, with soft footsteps and wind in the trees."
CHECKPOINT_FILES = (
    "modular_model_index.json",
    "text_encoder/model.safetensors.index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors.index.json",
    "audio_vae/config.json",
    "scheduler/scheduler_config.json",
    "audio_scheduler/scheduler_config.json",
)


class PhaseTiming:
    def __init__(self):
        self.records = defaultdict(list)

    @contextmanager
    def phase(self, name):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wall = perf_counter()
        with torch.profiler.record_function(f"h3::{name}"):
            yield
        end.record()
        self.records[name].append(((perf_counter() - wall) * 1000, start, end))

    def report(self):
        # The caller synchronizes CUDA after the request. Overlapping or nested scopes are not additive.
        return {
            name: {"calls": len(records), "host_wall_ms": sum(row[0] for row in records),
                   "cuda_stream_ms": sum(start.elapsed_time(end) for _, start, end in records)}
            for name, records in self.records.items()
        }


def compare(actual, expected):
    metrics = {}
    if actual.keys() != expected.keys():
        raise ValueError(f"Trace keys differ: {sorted(actual.keys() ^ expected.keys())}")
    for name, observed in actual.items():
        reference = expected[name]
        if observed.shape != reference.shape:
            raise ValueError(f"{name} shape: actual {tuple(observed.shape)}, reference {tuple(reference.shape)}")
        error = squared = dot = ref_squared = obs_squared = greatest = 0.0
        equal = count = 0
        for x, y in zip(observed.flatten().split(1_048_576), reference.flatten().split(1_048_576), strict=True):
            x, y = x.double(), y.double()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"{name} contains non-finite values")
            delta = (x - y).abs()
            error += delta.sum().item()
            squared += delta.square().sum().item()
            dot += torch.dot(x, y).item()
            ref_squared += y.square().sum().item()
            obs_squared += x.square().sum().item()
            greatest = max(greatest, delta.max().item())
            equal += (x == y).sum().item()
            count += x.numel()
        rmse = math.sqrt(squared / count)
        metrics[name] = {
            "shape": list(observed.shape), "dtype": str(observed.dtype), "reference_dtype": str(reference.dtype),
            "exact_fraction": equal / count, "mae": error / count, "max_abs": greatest, "rmse": rmse,
            "relative_rmse": math.sqrt(squared / ref_squared) if ref_squared else None,
            "cosine": dot / math.sqrt(ref_squared * obs_squared) if ref_squared and obs_squared else None,
        }
        if name == "pixels":
            metrics[name]["psnr_db"] = 20 * math.log10(255 / rmse) if rmse else None
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "model" / "MiniMax-H3")
    parser.add_argument("--backend", choices=("native", "diffusers", "tilelang-rope", "tilelang", "tilelang-attention"),
                        required=True, help="Use the standalone FL2VA pipeline or a Diffusers comparison backend.")
    parser.add_argument("--first-frame", type=Path, help="Condition native FL2VA on this opening image.")
    parser.add_argument("--last-frame", type=Path, help="Condition native FL2VA on this closing image.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=49,
                        help="Number of DiT forwards. The scheduler uses this many points plus its terminal zero.")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, help="Use a shorter schedule for untimed warmup requests.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-reserve", default="12GB", help="Free-memory target for Diffusers component offload.")
    parser.add_argument("--checkpoint-offload", action="store_true", help="Restore immutable CPU weights when offloading.")
    parser.add_argument("--resident-dit", action="store_true", help="Keep the native DiT in VRAM across requests when it fits.")
    parser.add_argument("--vae-tile-batch", type=int, default=1, help="Decode this many spatial VAE tiles in one batch.")
    parser.add_argument("--profile", type=Path, help="Save a Chrome trace from a separate request.")
    parser.add_argument("--trace", action="store_true", help="Save pre/post-denoise video and audio latent tensors.")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--reference", type=Path, help="Compare with a baseline .pt trace from the same settings.")
    reference.add_argument("--media-reference", type=Path,
                           help="Check RGB and audio against a baseline .pt trace without tracing denoising.")
    parser.add_argument("--no-export", action="store_true", help="Skip H.264/AAC encoding, which needs ffmpeg.")
    parser.add_argument("--output", type=Path, required=True, help="Write the benchmark JSON here and use its stem for MP4.")
    args = parser.parse_args()
    if min(args.width, args.height) < 32 or args.width % 32 or args.height % 32:
        parser.error("Canvas bounds must be positive multiples of 32")
    aligned = (args.frames - 5 + 16) // 17 * 17 + 5
    if args.frames < 1 or not 120 <= aligned <= 360:
        parser.error("Frame count, rounded up to 17*n+5, must correspond to 5–15 seconds at 24 fps")
    if args.steps < 1 or args.warmups < 0 or args.repeats < 1 or args.vae_tile_batch < 1 or not args.prompt.strip():
        parser.error("Use a nonempty prompt and positive steps, repeats, and VAE tile batch")
    if args.warmup_steps is not None and (args.warmups == 0 or args.warmup_steps < 1):
        parser.error("--warmup-steps requires at least one warmup and a positive step count")
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")
    if (args.first_frame or args.last_frame) and args.backend != "native":
        parser.error("Keyframes require --backend native")
    if args.resident_dit and args.backend != "native":
        parser.error("--resident-dit requires --backend native")
    if args.backend == "native" and (args.checkpoint_offload or args.vae_tile_batch != 1):
        parser.error("Native FL2VA selects its own checkpoint placement and VAE tile batch")
    return args, aligned


def fingerprint_checkpoint(root):
    validate_video_checkpoint(root)
    hashes = {}
    for name in CHECKPOINT_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing MiniMax-H3 video checkpoint file {path}")
        hashes[name] = sha256(path.read_bytes()).hexdigest()
        if name.endswith(".index.json"):
            shards = set(json.loads(path.read_text())["weight_map"].values())
            missing = sorted(shard for shard in shards if not (path.parent / shard).is_file())
            if missing:
                raise FileNotFoundError(f"Incomplete checkpoint directory {path.parent}: {', '.join(missing)}")
    if not (root / "audio_vae/diffusion_pytorch_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing audio VAE weights in {root / 'audio_vae'}")
    return hashes


@contextmanager
def instrument_video(phase, trace, steps):
    from diffusers.modular_pipelines.components_manager import CustomOffloadHook
    from diffusers.modular_pipelines.minimax_h3 import before_denoise, decoders, denoise, encoders

    def wrap(original, scope, capture=None):
        @wraps(original)
        def run(self, *args, **kwargs):
            with phase(scope):
                result = original(self, *args, **kwargs)
            if trace is not None and capture is not None:
                capture(result[1], kwargs)
            return result
        return run

    def capture_state(state, prefix):
        for name in ("latents", "audio_latents"):
            trace[f"{prefix}_{name}"] = state.get(name).detach().clone()

    def capture_step(state, kwargs):
        if kwargs["i"] in (0, steps - 1):
            for name in ("noise_pred", "audio_noise_pred"):
                trace[f"{name}_{kwargs['i']}"] = getattr(state, name).detach().clone()

    hooks = (
        (encoders.MiniMaxH3TextEncoderStep, "text_encoder", None),
        (before_denoise.MiniMaxH3PrepareLayoutStep, "layout", None),
        (before_denoise.MiniMaxH3PrepareLatentsStep, "prepare_latents", lambda state, _: capture_state(state, "initial")),
        (before_denoise.MiniMaxH3SetTimestepsStep, "prepare_schedule", None),
        (denoise.MiniMaxH3DenoiseLoopWrapper, "denoise", None),
        (denoise.MiniMaxH3LoopDenoiser, "dit", capture_step),
        (denoise.MiniMaxH3LoopSchedulerStep, "scheduler", None),
        (decoders.MiniMaxH3AfterDenoiseStep, "unpack", lambda state, _: capture_state(state, "final")),
        (decoders.MiniMaxH3VideoDecodeStep, "video_vae_and_postprocess", None),
        (decoders.MiniMaxH3AudioDecodeStep, "audio_vae", None),
    )
    original_placement = CustomOffloadHook.pre_forward

    @wraps(original_placement)
    def place(hook, module, *args, **kwargs):
        context = phase("placement") if module.device != hook.execution_device else nullcontext()
        with context:
            return original_placement(hook, module, *args, **kwargs)

    with ExitStack() as stack:
        for cls, name, capture in hooks:
            stack.enter_context(patch.object(cls, "__call__", wrap(cls.__call__, name, capture)))
        stack.enter_context(patch.object(CustomOffloadHook, "pre_forward", place))
        yield


def hash_tensor(tensor):
    return sha256(memoryview(tensor.numpy())).hexdigest()


def load_reference(path, settings, fingerprint):
    reference = torch.load(path, map_location="cpu", weights_only=True)
    if reference["settings"] != settings or reference["checkpoint"] != fingerprint:
        raise ValueError("Baseline trace requires the same settings and video checkpoint")
    report = json.loads(path.with_suffix(".json").read_text())
    return reference["tensors"], report


def compare_media(video, audio, report, reference, baseline):
    rgb_match = report["video"]["sha256"] == baseline["video"]["sha256"]
    audio_match = report["audio"]["sha256"] == baseline["audio"]["sha256"]
    report["complete_rgb_sha256_match"] = rgb_match
    report["complete_waveform_sha256_match"] = audio_match
    metrics = {}
    if not rgb_match:
        frames = baseline["sampled_pixel_frames"]
        expected = reference["pixels"]
        actual = video[frames]
        if actual.shape != expected.shape:
            raise ValueError("Sampled RGB frames do not match the baseline shape")
        mse = (actual.double() - expected.double()).square().mean().item()
        metrics["sampled_rgb_psnr_db"] = 10 * math.log10(255**2 / mse) if mse else None
    if not audio_match:
        expected = reference["waveform"].double()
        if audio.shape != expected.shape:
            raise ValueError("Audio waveform does not match the baseline shape")
        mse = (audio.double() - expected).square().mean().item()
        power = expected.square().mean().item()
        metrics["waveform_snr_db"] = 10 * math.log10(power / mse) if power and mse else None
    return metrics


def save_report(path, report):
    stage = path.with_suffix(".json.tmp")
    stage.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    stage.replace(path)


def profile_ops(profiler):
    ranked = sorted(profiler.key_averages(), key=lambda event: event.self_device_time_total, reverse=True)
    return [{"operator": e.key, "calls": e.count, "self_device_ms": e.self_device_time_total / 1000}
            for e in ranked[:20] if e.self_device_time_total > 0]


def storage_read_bytes():
    try:
        lines = Path("/proc/self/io").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        name, _, value = line.partition(":")
        if name == "read_bytes":
            return int(value)
    return None


def load_native(path, device, args):
    from PIL import Image

    from models.minimax_h3.video.pipeline import NativeFL2VA

    images = {}
    for name, filename in (("first", args.first_frame), ("last", args.last_frame)):
        if filename is not None:
            with Image.open(filename) as image:
                images[name] = image.convert("RGB")
    pipeline = NativeFL2VA(path, device, with_keyframes=bool(images), resident_dit=args.resident_dit)

    def generate(phase, trace, steps=args.steps):
        def capture(name, value):
            if trace is not None and name != "prompt_embeds":
                trace[name] = value.detach().clone()

        return pipeline.generate(args.prompt, args.width, args.height, args.frames, steps, args.seed,
                                 phase=phase, trace=capture, **images)

    return generate, getattr(pipeline.vae.model, "decode_tile_batch", 1), pipeline.resident_placement


def load_diffusers(path, device, args, contexts):
    from diffusers import ComponentsManager, ModularPipeline

    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=device, memory_reserve_margin=args.memory_reserve)
    pipe = ModularPipeline.from_pretrained(str(path), components_manager=manager, workflow="t2va", local_files_only=True)
    pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=str(path), local_files_only=True)
    kernels = {}
    if args.backend != "diffusers":
        from models.minimax_h3.video_kernels import install_video_kernels

        kernels = install_video_kernels(pipe.transformer,
                                        attention_backend="tilelang" if args.backend == "tilelang-attention" else "sdpa",
                                        fused_swiglu=args.backend != "tilelang-rope")
    vae_kernels = {}
    if args.vae_tile_batch > 1:
        from models.minimax_h3.video_kernels import install_video_vae_batching

        vae_kernels = install_video_vae_batching(pipe.vae, tile_batch=args.vae_tile_batch)
    if args.checkpoint_offload:
        from models.minimax_h3.video_placement import checkpoint_offload

        contexts.enter_context(checkpoint_offload(pipe))

    def generate(phase, trace, steps=args.steps):
        with instrument_video(phase, trace, steps):
            return pipe(prompt=args.prompt, width=args.width, height=args.height, num_frames=args.frames,
                        num_inference_steps=steps + 1, output_type="pt",
                        generator=torch.Generator(device="cpu").manual_seed(args.seed),
                        output=["videos", "audio", "sampling_rate"])

    return generate, kernels, vae_kernels


def main():
    args, aligned = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    if not args.no_export and not shutil.which("ffmpeg"):
        raise RuntimeError("MP4 export requires ffmpeg. Use --no-export for profiling without a video file.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = args.model_path.resolve()
    fingerprint = fingerprint_checkpoint(path)
    settings = dict(prompt=args.prompt, seed=args.seed, width=args.width, height=args.height,
                    requested_frames=args.frames, frames=aligned, dit_steps=args.steps, sigma_points=args.steps + 1)

    for name, filename in (("first", args.first_frame), ("last", args.last_frame)):
        if filename is not None:
            settings[name + "_frame_sha256"] = sha256(filename.read_bytes()).hexdigest()

    reference_report = None
    reference = None
    if reference_path := args.reference or args.media_reference:
        reference, reference_report = load_reference(reference_path, settings, fingerprint)
        if args.media_reference and (reference_report["settings"] != settings
                                     or reference_report["checkpoint"] != fingerprint):
            raise ValueError("Baseline report requires the same settings and video checkpoint")

    native = args.backend == "native"
    vae_tile_batch = args.vae_tile_batch
    placement = None
    with torch.cuda.device(device), torch.inference_mode(), ExitStack() as contexts:
        start = perf_counter()
        if native:
            generate, vae_tile_batch, placement = load_native(path, device, args)
            kernels, vae_kernels = {}, {}
        else:
            generate, kernels, vae_kernels = load_diffusers(path, device, args, contexts)
        torch.cuda.synchronize(device)
        load_seconds = perf_counter() - start
        print(f"{args.backend} weights loaded in {load_seconds:.3f} s", flush=True)

        samples = []
        warmups = []
        results = None
        trace = {} if args.trace or args.reference else None
        for i in range(args.warmups + args.repeats):
            del results
            timing = PhaseTiming()
            resident_before = placement.resident if placement is not None else False
            read_before = storage_read_bytes() if placement is not None else None
            last_sample = i == args.warmups + args.repeats - 1
            torch.cuda.reset_peak_memory_stats(device)
            start = perf_counter()
            steps = (args.warmup_steps or args.steps) if i < args.warmups else args.steps
            results = generate(timing.phase, trace if last_sample else None, steps)
            torch.cuda.synchronize(device)
            elapsed = perf_counter() - start
            phases = timing.report()
            if phases.get("dit", {}).get("calls") != steps:
                raise RuntimeError(f"Expected {steps} DiT forwards, recorded {phases.get('dit')}")
            sample = dict(seconds=elapsed, dit_steps=steps, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device), phases=phases)
            if placement is not None:
                sample.update(dit_resident_before=resident_before, dit_resident_after=placement.resident,
                              resident_text_layers=placement.resident_text_layers)
                read_after = storage_read_bytes() if read_before is not None else None
                if read_after is not None:
                    sample["storage_read_bytes"] = read_after - read_before
            collection = warmups if i < args.warmups else samples
            collection.append(sample)
            print(f"{args.backend} {'warmup' if collection is warmups else 'timed'}: {elapsed:.3f} s", flush=True)

        start = perf_counter()
        video, audio = prepare_media(results, frames=aligned, height=args.height, width=args.width)
        media_prepare_seconds = perf_counter() - start
        rate = int(results["sampling_rate"])
        del results
        counters = dict(kernels)
        if kernels and not counters["rotary_calls"]:
            raise RuntimeError("The installed TileLang RoPE processor did not run")
        if args.backend in ("tilelang", "tilelang-attention") and not counters["swiglu_calls"]:
            raise RuntimeError("The installed TileLang SwiGLU did not run")
        if args.backend == "tilelang-attention" and not counters["tilelang_attention_calls"]:
            raise RuntimeError("The installed TileLang attention did not run")
        seconds = median(sample["seconds"] for sample in samples)
        report = dict(backend=args.backend, settings=settings, checkpoint=fingerprint, device=torch.cuda.get_device_name(device),
                      torch=torch.__version__, diffusers=None if native else version("diffusers"),
                      memory_reserve=None if native else args.memory_reserve,
                      checkpoint_offload=native or args.checkpoint_offload, vae_tile_batch=vae_tile_batch,
                      load_seconds=load_seconds, median_seconds=seconds, warmups=warmups, samples=samples, kernel_calls=counters,
                      vae_kernel_calls=dict(vae_kernels),
                      video=dict(shape=list(video.shape), fps=24, sha256=hash_tensor(video),
                                 first_and_last_frames_differ=not torch.equal(video[0], video[-1])),
                      audio=dict(shape=list(audio.shape), sampling_rate=rate, sha256=hash_tensor(audio),
                                 rms=audio.square().mean().sqrt().item()),
                      media_prepare_seconds=media_prepare_seconds)
        if native:
            report["resident_dit_requested"] = args.resident_dit
        if trace is not None:
            trace = {name: value.cpu() for name, value in trace.items()}
            expected = {f"{prefix}_{name}" for prefix in ("initial", "final") for name in ("latents", "audio_latents")}
            expected |= {f"{name}_{n}" for name in ("noise_pred", "audio_noise_pred") for n in (0, args.steps - 1)}
            if trace.keys() != expected:
                raise RuntimeError(f"Unexpected video trace stages: {sorted(trace.keys() ^ expected)}")
            trace["pixels"] = video[[0, aligned // 2, -1]]
            report["sampled_prediction_steps"] = sorted({0, args.steps - 1})
            report["sampled_pixel_frames"] = [0, aligned // 2, aligned - 1]
            trace["waveform"] = audio
            if args.trace:
                torch.save(dict(settings=settings, checkpoint=fingerprint, tensors=trace), args.output.with_suffix(".pt"))
        if args.reference:
            report["comparison"] = compare(trace, reference)
            report["speedup"] = reference_report["median_seconds"] / seconds
            report["complete_rgb_sha256_match"] = report["video"]["sha256"] == reference_report["video"]["sha256"]
            report["complete_waveform_sha256_match"] = report["audio"]["sha256"] == reference_report["audio"]["sha256"]
            print(f"Reference: {report['speedup']:.3f}x speedup, RGB match "
                  f"{report['complete_rgb_sha256_match']}, audio match {report['complete_waveform_sha256_match']}",
                  flush=True)
        if args.media_reference:
            report["media_comparison"] = compare_media(video, audio, report, reference, reference_report)
            print(f"RGB match {report['complete_rgb_sha256_match']}, "
                  f"audio match {report['complete_waveform_sha256_match']}", flush=True)
        save_report(args.output, report)

        if not args.no_export:
            start = perf_counter()
            mp4 = args.output.with_suffix(".mp4")
            encode_mp4(video, audio, rate, mp4)
            report.update(mp4=mp4.name, mp4_bytes=mp4.stat().st_size, encode_seconds=perf_counter() - start)
            save_report(args.output, report)

        if args.profile:
            timing = PhaseTiming()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                generate(timing.phase, None)
                torch.cuda.synchronize(device)
            args.profile.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(args.profile))
            report.update(profile=args.profile.name, profile_phases=timing.report(), profile_ops=profile_ops(profiler))
            save_report(args.output, report)


if __name__ == "__main__":
    main()
