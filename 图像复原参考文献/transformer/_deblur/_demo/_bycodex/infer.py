from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from deblur_transformer.io_utils import list_images, load_image, save_image
from deblur_transformer.model import DeblurTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore blurred images with a trained Transformer.")
    parser.add_argument("--input", required=True, help="Blurred image path or a directory of images.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Trained checkpoint path.")
    parser.add_argument("--output", default="outputs", help="Output file or directory.")
    return parser.parse_args()


def output_path(input_path: Path, input_root: Path, requested_output: Path, multiple: bool) -> Path:
    if not multiple and requested_output.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return requested_output
    if input_root.is_dir():
        relative_parent = input_path.relative_to(input_root).parent
    else:
        relative_parent = Path()
    return requested_output / relative_parent / f"{input_path.stem}_restored.png"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This project is CUDA-only, but PyTorch cannot access an NVIDIA GPU.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the model first with train.py."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model = DeblurTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.cuda().eval()

    input_root = Path(args.input)
    images = list_images(input_root)
    requested_output = Path(args.output)
    print(
        f"device=cuda, checkpoint_epoch={checkpoint.get('epoch', '?')}, "
        f"images={len(images)}"
    )

    for image_path in images:
        image = load_image(image_path).unsqueeze(0).cuda()
        started = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            restored = model(image)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        destination = output_path(
            image_path,
            input_root,
            requested_output,
            multiple=len(images) > 1,
        )
        save_image(restored, destination)
        print(f"{image_path} -> {destination} ({elapsed:.3f}s)")


if __name__ == "__main__":
    main()
