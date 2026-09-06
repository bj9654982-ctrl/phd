"""Overfit one deterministic synthetic batch to validate the CUDA training loop."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_debug_batch
from src.losses import restoration_loss
from src.metrics import psnr
from src.models import CompactDeblurUNet
from src.models.compact_unet import count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "runs" / "P1_overfit_smoke")
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_comparison(path: Path, blurry: torch.Tensor, restored: torch.Tensor, sharp: torch.Tensor) -> None:
    images = [blurry, restored.clamp(0, 1), sharp]
    titles = ["Blurry input", "Baseline overfit output", "Sharp target"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title in zip(axes, images, titles):
        array = image[0].detach().float().cpu().permute(1, 2, 0).numpy()
        axis.imshow(array)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script has no CPU fallback")
    if args.steps < 1:
        raise ValueError("steps must be positive")

    set_determinism(args.seed)
    device = torch.device("cuda:0")
    blurry, sharp = make_debug_batch(batch_size=2, size=args.size, seed=args.seed)
    model = CompactDeblurUNet(width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")

    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        initial_prediction = model(blurry)
    initial_psnr = psnr(initial_prediction, sharp)

    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(blurry)
            loss = restoration_loss(prediction, sharp)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 25 == 0 or step == args.steps:
            current_psnr = psnr(prediction.detach(), sharp)
            history.append({"step": step, "loss": float(loss.detach()), "psnr": current_psnr})
            print(f"step={step:04d} loss={float(loss.detach()):.6f} psnr={current_psnr:.3f} dB")

    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        final_prediction = model(blurry)
    final_psnr = psnr(final_prediction, sharp)
    passed = final_psnr >= 28.0 and final_psnr - initial_psnr >= 8.0

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "curve.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "loss", "psnr"])
        writer.writeheader()
        writer.writerows(history)

    report = {
        "experiment": "P1_overfit_smoke",
        "seed": args.seed,
        "steps": args.steps,
        "patch_size": args.size,
        "width": args.width,
        "parameters": count_parameters(model),
        "initial_psnr_db": initial_psnr,
        "final_psnr_db": final_psnr,
        "gain_db": final_psnr - initial_psnr,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "passed": passed,
        "note": "Synthetic data validates the pipeline only and is not a paper result."
    }
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_comparison(args.output / "comparison.png", blurry, final_prediction, sharp)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

