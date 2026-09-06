"""CUDA-only Transformer deblurring demonstration.

Edit IMAGE_PATH and run this file directly. The script creates a blurred/clean
pair, trains a tiny Transformer on the GPU, and saves a comparison image. This
is an educational overfitting demo, not a general pretrained deblurring model.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from torch.nn import functional as F

from deblur_transformer.data import apply_kernel, motion_kernel
from deblur_transformer.io_utils import pil_to_tensor, save_image, tensor_to_pil
from deblur_transformer.model import DeblurTransformer


# ====================== 在这里修改 Demo 参数 ======================
# 填写一张清晰图片路径；相对路径以本工程目录为基准。
# 注意：这个教学 Demo 会自动把该图像模糊，并把原图作为训练真值。
IMAGE_PATH = r"building.jpg"

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "simple_demo_outputs"

IMAGE_SIZE = 160
MOTION_LENGTH = 15
MOTION_ANGLE = 28.0
TRAIN_STEPS = 500
LEARNING_RATE = 5e-4
SEED = 7
# ================================================================


def load_sharp_image() -> tuple[torch.Tensor, Path]:
    path = Path(IMAGE_PATH)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"IMAGE_PATH does not exist: {path}")
    with Image.open(path) as source:
        image = ImageOps.fit(source.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return pil_to_tensor(image), path


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(prediction.float(), target.float()).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def make_comparison(sharp: torch.Tensor, blurred: torch.Tensor, restored: torch.Tensor, path: Path) -> None:
    images = [tensor_to_pil(sharp), tensor_to_pil(blurred), tensor_to_pil(restored)]
    labels = ["Sharp target", "Motion blurred", "Transformer restored"]
    label_height = 28
    canvas = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * IMAGE_SIZE
        canvas.paste(image, (x, label_height))
        draw.text((x + 8, 7), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This project is CUDA-only, but PyTorch cannot access an NVIDIA GPU.")

    torch.manual_seed(SEED)

    sharp, source_path = load_sharp_image()
    sharp = sharp.cuda()
    kernel = motion_kernel(MOTION_LENGTH, MOTION_ANGLE).cuda()
    blurred = apply_kernel(sharp, kernel).clamp(0.0, 1.0)

    model = DeblurTransformer(
        embed_dim=16,
        depths=(1, 1, 2),
        heads=(1, 2, 4),
        expansion=2.0,
    ).cuda()
    # Start from the identity mapping: output = input + zero residual.
    torch.nn.init.zeros_(model.output_projection.weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)
    scaler = torch.amp.GradScaler("cuda")

    before_psnr = psnr(blurred, sharp)
    print(f"source: {source_path}")
    print(f"device: cuda; model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"blurred PSNR: {before_psnr:.2f} dB")
    started = time.perf_counter()

    blur_batch = blurred.unsqueeze(0)
    sharp_batch = sharp.unsqueeze(0)
    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            prediction = model(blur_batch)
            pixel = F.mse_loss(prediction, sharp_batch)
            edges = gradient_loss(prediction, sharp_batch)
            loss = pixel + 0.1 * edges
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == TRAIN_STEPS:
            print(f"step {step:04d}/{TRAIN_STEPS}: loss={loss.item():.6f}")

    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        restored = model(blur_batch)[0].float()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    after_psnr = psnr(restored, sharp)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_image(sharp, OUTPUT_DIR / "01_sharp.png")
    save_image(blurred, OUTPUT_DIR / "02_blurred.png")
    save_image(restored, OUTPUT_DIR / "03_restored.png")
    make_comparison(sharp, blurred, restored, OUTPUT_DIR / "04_comparison.png")

    print(f"restored PSNR: {after_psnr:.2f} dB (improvement {after_psnr - before_psnr:+.2f} dB)")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
