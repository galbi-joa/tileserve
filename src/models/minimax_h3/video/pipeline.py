"""Run the FL2VA BF16 checkpoint without importing Diffusers."""

from contextlib import contextmanager, nullcontext
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from ..native_placement import CheckpointPlacement
from .audio_vae import AudioVAE
from .checkpoint import load_weights, read_config, validate_video_checkpoint
from .conditioning import encode_prompt, load_conditioner, pixels_for_keyframe, resize_keyframes
from .dit import VideoDiT
from .layout import build_layout, euler_schedule, pack_video, row_times, unpack_video
from .video_vae import VideoVAE


class NativeFL2VA:
    def __init__(self, path, device="cuda", *, with_keyframes=False, resident_dit=False):
        from transformers import AutoProcessor, AutoTokenizer

        path = Path(path)
        validate_video_checkpoint(path)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Native FL2VA requires a CUDA device")
        config = SimpleNamespace(**read_config(path / "transformer"))
        self.patch = tuple(config.patch_size)
        self.video_channels = config.in_channels
        self.audio_channels = config.audio_in_channels
        self.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
        self.processor = (AutoProcessor.from_pretrained(path / "processor", local_files_only=True)
                          if with_keyframes else None)
        self.text = CheckpointPlacement(load_conditioner(path / "text_encoder", with_keyframes), self.device)
        visuals = read_config(path / "vae")
        audio = read_config(path / "audio_vae")
        if (self.patch[0] != 1 or self.video_channels != visuals["latent_channels"]
                or self.audio_channels != audio["latent_channels"]):
            raise ValueError("Incompatible FL2VA transformer and VAE checkpoint configurations")
        with torch.device("meta"):
            dit = VideoDiT(config)
            vae = VideoVAE(with_encoder=with_keyframes,
                           **{k: v for k, v in visuals.items() if not k.startswith("latents_")})
        self.dit = CheckpointPlacement(load_weights(
            dit, path / "transformer",
            fp32=("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"),
        ), self.device)
        self.vae = CheckpointPlacement(load_weights(
            vae, path / "vae", fp32=("encoder", "quant_conv", "norm1", "norm2", "norm_out", "scale1", "scale2"),
        ), self.device)
        self.audio_vae = CheckpointPlacement(load_weights(
            AudioVAE(audio), path / "audio_vae", dtype=torch.float32,
        ), self.device)
        self.resident_placement = None
        if resident_dit:
            from .placement import ResidentVideoPlacement

            self.resident_placement = ResidentVideoPlacement(
                self.text, self.dit, self.vae, self.audio_vae, hidden_size=config.hidden_size,
            )
            self.text = self.resident_placement.text
            self.dit = self.resident_placement.dit

        self.sample_rate = audio["sampling_rate"]
        self.audio_hop = math.prod(audio["decoder_rates"])
        self.pixel_mean = self._channels((0.485, 0.456, 0.406), 5)
        self.pixel_std = self._channels((0.229, 0.224, 0.225), 5)
        self.video_mean = self._channels(visuals["latents_mean"], 5)
        self.video_std = self._channels(visuals["latents_std"], 5)
        self.audio_mean = self._channels(audio["latents_mean"], 3)
        self.audio_std = self._channels(audio["latents_std"], 3)
        self.video_shift, self.audio_shift = (
            json.loads((path / name / "scheduler_config.json").read_text())["shift"]
            for name in ("scheduler", "audio_scheduler")
        )

    def _channels(self, values, dims):
        return torch.tensor(values, dtype=torch.float32, device=self.device).view(1, -1, *([1] * (dims - 2)))

    @contextmanager
    def _placed(self, component, phase):
        with phase("placement"):
            model = component.load()
        try:
            yield model
        finally:
            with phase("placement"):
                component.release()

    def _keyframe_latents(self, keyframes, phase):
        if not keyframes:
            return []
        with self._placed(self.vae, phase) as vae, phase("keyframe_encode"):
            result = []
            for image in keyframes.values():
                pixels = pixels_for_keyframe(image, self.device)
                normalized = (pixels / 255 - self.pixel_mean) / self.pixel_std
                latent = vae.encode_keyframe(normalized, torch.Generator().manual_seed(42))
                result.append((latent.half().float() - self.video_mean) / self.video_std)
            return result

    @torch.inference_mode()
    def generate(self, prompt, width, height, frames, steps, seed, *, first=None, last=None, phase=None, trace=None):
        try:
            return self._generate(prompt, width, height, frames, steps, seed, first, last, phase, trace)
        except torch.cuda.OutOfMemoryError:
            if self.resident_placement is not None:
                self.resident_placement.evict(disable=True)
            raise

    def _generate(self, prompt, width, height, frames, steps, seed, first, last, phase, trace):
        from tileserve_kernels.minimax_h3.video import euler_blend

        if not prompt.strip() or steps < 1:
            raise ValueError("A nonempty prompt and at least one denoiser step are required")
        vae = self.vae.model
        spatial = vae.spatial_compression_ratio
        align_h, align_w = (spatial * p for p in self.patch[1:])
        if min(width, height, frames) < 1 or width % align_w or height % align_h:
            raise ValueError(f"Canvas must be positive and divisible by {align_w}×{align_h}")
        chunks = math.ceil((frames - vae.tokens_chunk_size) / vae.clip_length)
        aligned = chunks * vae.clip_length + vae.tokens_chunk_size
        if not 120 <= aligned <= 360:
            raise ValueError("The aligned video must be 5–15 seconds at 24 fps")
        time = chunks * vae.tokens_chunk_size + 2
        audio_time = round(aligned / 24 * self.sample_rate / self.audio_hop)
        shape = (1, self.video_channels, time, height // spatial, width // spatial)
        phase = phase or (lambda _: nullcontext())
        emit = trace or (lambda *_: None)
        keyframes = resize_keyframes(first, last, width, height)
        if keyframes and self.processor is None:
            raise ValueError("Load FL2VA with keyframe conditioning enabled")
        if self.resident_placement is not None:
            text_tokens = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            text_tokens += len(keyframes) * (math.ceil(width / 28) * math.ceil(height / 28) + 256)
            sequence_tokens = (time + len(keyframes)) * (height // align_h) * (width // align_w) + 2 * audio_time
            with phase("placement"):
                self.resident_placement.prepare(sequence_tokens=sequence_tokens + text_tokens,
                                                output_pixels=width * height * aligned, phase=phase)

        with self._placed(self.text, phase) as text_encoder, phase("text_encoder"):
            prompt_embeds, text_tags = encode_prompt(
                text_encoder, self.tokenizer, self.processor, prompt, keyframes, self.device
            )
            emit("prompt_embeds", prompt_embeds)
        condition_latents = self._keyframe_latents(keyframes, phase)

        with phase("prepare"):
            layout = build_layout(text_tags, *shape[2:], audio_time, self.patch, tuple(keyframes))
            generator = torch.Generator().manual_seed(seed)

            def noise(size):
                return torch.randn(size, generator=generator, dtype=torch.float32).to(self.device)

            keyframe_time = 0.999
            anchor_time = torch.tensor(keyframe_time, device=self.device)
            anchors = [pack_video(latent * anchor_time + noise(latent.shape) * (1 - anchor_time), self.patch)
                       for latent in condition_latents]
            video = torch.cat((*anchors, pack_video(noise(shape), self.patch)))
            audio = noise((2 * audio_time, self.audio_channels))
            emit("initial_latents", video)
            emit("initial_audio_latents", audio)
            video_times, video_steps = euler_schedule(steps, self.video_shift)
            audio_times, audio_steps = euler_schedule(steps, self.audio_shift)
            step_layouts = [tuple(t.to(self.device) for t in row_times(layout, v, a, keyframe_time))
                            for v, a in zip(video_times, audio_times, strict=True)]
            layout = layout.to(self.device)
            video_steps, audio_steps = video_steps.to(self.device), audio_steps.to(self.device)

        with self._placed(self.dit, phase) as dit, phase("denoise"):
            conditioning = dit.prepare(prompt_embeds, layout.positions)
            update = euler_blend()
            target = video[layout.condition_rows:].view(-1)
            for i, (times, rows) in enumerate(step_layouts):
                with phase("dit"):
                    v, a = dit(video, audio, times, rows, layout, conditioning)
                if i in (0, steps - 1):
                    emit(f"noise_pred_{i}", v[None])
                    emit(f"audio_noise_pred_{i}", a[None])
                with phase("scheduler"):
                    update(target, v[layout.condition_rows:].contiguous().view(-1), video_steps[i])
                    update(audio.view(-1), a.contiguous().view(-1), audio_steps[i])
        video_latents = unpack_video(video[layout.condition_rows:], shape, self.patch)
        audio_latents = audio.reshape(2, audio_time, self.audio_channels).transpose(1, 2).contiguous()
        emit("final_latents", video_latents)
        emit("final_audio_latents", audio_latents)

        with self._placed(self.vae, phase) as vae, phase("video_vae_and_postprocess"):
            rgb = vae.decode(video_latents * self.video_std + self.video_mean)
            videos = (rgb.float() * self.pixel_std + self.pixel_mean).clamp(0, 1).permute(0, 2, 1, 3, 4)
        with self._placed(self.audio_vae, phase) as vae, phase("audio_vae"):
            waveform = vae.decode(audio_latents * self.audio_std + self.audio_mean).float().transpose(0, 1)
        return {"videos": videos, "audio": waveform, "sampling_rate": self.sample_rate}
