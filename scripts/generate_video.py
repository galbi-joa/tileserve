"""Generate a MiniMax H3 FL2VA video and stereo soundtrack without Diffusers."""

import argparse
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "model" / "MiniMax-H3")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--first-frame", type=Path)
    parser.add_argument("--last-frame", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=49, help="Number of denoiser evaluations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must have the .mp4 extension")
    if not shutil.which("ffmpeg"):
        parser.error("FFmpeg is required to write the MP4")

    from PIL import Image
    import torch

    from models.minimax_h3.video.media import encode_mp4, prepare_media
    from models.minimax_h3.video.pipeline import NativeFL2VA

    keyframes = {}
    for name, filename in (("first", args.first_frame), ("last", args.last_frame)):
        if filename is not None:
            with Image.open(filename) as image:
                keyframes[name] = image.convert("RGB")
    device = torch.device(args.device)
    with torch.cuda.device(device), torch.inference_mode():
        pipeline = NativeFL2VA(args.model_path, device, with_keyframes=bool(keyframes))
        print("FL2VA weights loaded.", flush=True)
        start = perf_counter()
        result = pipeline.generate(args.prompt, args.width, args.height, args.frames, args.steps, args.seed,
                                   **keyframes)
        frames = result["videos"].shape[1]
        video, audio = prepare_media(result, frames=frames, height=args.height, width=args.width)
        encode_mp4(video, audio, result["sampling_rate"], args.output)
    print(f"{args.output} ({frames} frames, {perf_counter() - start:.2f} s including MP4 export)")


if __name__ == "__main__":
    main()
