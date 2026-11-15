"""Validate and write a generated FL2VA video and its stereo audio."""

import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import wave

import torch


def prepare_media(results, *, frames, height, width):
    video = results["videos"]
    audio = results["audio"]
    if tuple(video.shape) != (1, frames, 3, height, width):
        raise RuntimeError(f"Unexpected video tensor shape {tuple(video.shape)}")
    if audio.ndim != 3 or audio.shape[0:2] != (1, 2) or not audio.shape[-1]:
        raise RuntimeError(f"Unexpected, non-stereo, or empty audio tensor {tuple(audio.shape)}")
    if not torch.isfinite(video).all():
        raise RuntimeError("Video contains nonfinite pixels")
    frames = (video[0] * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu()
    waveform = audio[0].detach().float().cpu().contiguous()
    if not torch.isfinite(waveform).all():
        raise RuntimeError("Audio waveform is nonfinite")
    return frames, waveform


def encode_mp4(video, audio, rate, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    _, height, width, _ = video.shape
    with TemporaryDirectory(prefix="h3-mp4-", dir=path.parent) as tmp:
        waveform = Path(tmp) / "audio.wav"
        staged = Path(tmp) / path.name
        samples = (audio.clamp(-1, 1) * 32767).round().short().T.contiguous()
        with wave.open(str(waveform), "wb") as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(samples.numpy().astype("<i2", copy=False).tobytes())
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", "24", "-i", "pipe:0", "-i", str(waveform),
             "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-movflags", "+faststart", str(staged)],
            input=memoryview(video.numpy()).cast("B"), check=True,
        )
        if not staged.stat().st_size:
            raise RuntimeError("Video export wrote an empty MP4")
        if shutil.which("ffprobe"):
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(staged)],
                check=True, capture_output=True, text=True,
            )
            streams = {s["codec_type"]: s for s in json.loads(probe.stdout)["streams"]}
            rendered = streams.get("video", {})
            sound = streams.get("audio", {})
            if (rendered.get("width") != width or rendered.get("height") != height
                    or int(rendered.get("nb_frames", 0)) != video.shape[0]
                    or rendered.get("avg_frame_rate") != "24/1" or sound.get("channels") != 2):
                raise RuntimeError("Export did not preserve the expected video size, frame count, or stereo audio")
        staged.replace(path)
