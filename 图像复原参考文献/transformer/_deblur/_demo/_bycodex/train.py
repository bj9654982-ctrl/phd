from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from deblur_transformer.data import SyntheticBlurDataset
from deblur_transformer.model import DeblurTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact Transformer deblurring model.")
    parser.add_argument("--train-dir", required=True, help="Directory containing sharp training images.")
    parser.add_argument("--output-dir", default="checkpoints", help="Checkpoint output directory.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--frequency-weight", type=float, default=0.05)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--depths", type=int, nargs=3, default=(2, 2, 4))
    parser.add_argument("--heads", type=int, nargs=3, default=(1, 2, 4))
    parser.add_argument("--expansion", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 by default on Windows.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def frequency_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft2(prediction, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    return torch.mean(torch.abs(pred_fft - target_fft))


def save_checkpoint(
    path: Path,
    model: DeblurTransformer,
    epoch: int,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_config": model.model_config,
            "model_state": model.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This project is CUDA-only, but PyTorch cannot access an NVIDIA GPU.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = SyntheticBlurDataset(args.train_dir, args.patch_size, args.noise_std)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    model = DeblurTransformer(args.embed_dim, args.depths, args.heads, args.expansion).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda")

    print(f"device=cuda, images={len(dataset)}, parameters={sum(p.numel() for p in model.parameters()):,}")
    output_dir = Path(args.output_dir)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        started = time.perf_counter()
        for step, (blurred, sharp) in enumerate(loader, start=1):
            blurred = blurred.cuda(non_blocking=True)
            sharp = sharp.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                restored = model(blurred)
                pixel = F.l1_loss(restored, sharp)
                freq = frequency_loss(restored, sharp)
                loss = pixel + args.frequency_weight * freq
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(
                    f"epoch {epoch:03d}/{args.epochs:03d} step {step:04d}/{len(loader):04d} "
                    f"loss={loss.item():.6f}"
                )

        scheduler.step()
        epoch_loss = running_loss / max(1, len(loader))
        elapsed = time.perf_counter() - started
        save_checkpoint(output_dir / "last.pt", model, epoch, epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(output_dir / "best.pt", model, epoch, epoch_loss)
        print(f"epoch {epoch:03d} mean_loss={epoch_loss:.6f} time={elapsed:.1f}s best={best_loss:.6f}")


if __name__ == "__main__":
    main()
