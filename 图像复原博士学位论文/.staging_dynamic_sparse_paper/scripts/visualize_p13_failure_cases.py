"""Render representative two-seed failure/success cases for local A versus G."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset  # noqa: E402
from src.models import CompactDeblurUNet, MultiScaleNonlocalBlock  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize representative P13 A/G cases.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recording-group", default="GOPR0372_07")
    parser.add_argument("--worst-count", type=int, default=4)
    parser.add_argument("--best-count", type=int, default=2)
    return parser.parse_args()


def tensor_image(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def gate_image(gate: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    resized = F.interpolate(gate.float(), size=size, mode="bilinear", align_corners=False)[0, 0]
    red = resized
    blue = 1.0 - resized
    green = 1.0 - (2.0 * resized - 1.0).abs()
    return tensor_image(torch.stack((red, green, blue), dim=0))


def load_selection(path: Path, group: str, worst: int, best: int) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["recording_group"] == group]
    if len(rows) < worst + best:
        raise RuntimeError(f"not enough rows for {group}")
    rows.sort(key=lambda row: float(row["two_seed_mean_candidate_minus_local_psnr_db"]))
    selected = rows[:worst] + rows[-best:]
    for index, row in enumerate(selected):
        row["selection"] = "worst" if index < worst else "best"
    return selected


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this project intentionally has no CPU fallback")
    if args.worst_count <= 0 or args.best_count <= 0:
        raise ValueError("worst-count and best-count must be positive")

    baseline_checkpoint = torch.load(
        args.baseline_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    baseline_args = dict(baseline_checkpoint["args"])
    local_checkpoint = torch.load(
        args.local_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    candidate_checkpoint = torch.load(
        args.candidate_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    candidate_args = dict(candidate_checkpoint["args"])
    if candidate_args.get("variant") != "gate_adaptive":
        raise ValueError("candidate checkpoint must be gate_adaptive")

    width = int(baseline_args["width"])
    local_model = CompactDeblurUNet(width=width)
    local_model.load_state_dict(local_checkpoint["model"], strict=True)
    block = MultiScaleNonlocalBlock(
        channels=width * 4,
        scales=tuple(int(value) for value in candidate_args.get("scales", (1, 3, 7))),
        top_k=int(candidate_args.get("top_k", 6)),
        statistics_window=int(candidate_args.get("statistics_window", 5)),
        mode="gate_adaptive",
    )
    candidate_model = CompactDeblurUNet(width=width, bottleneck_module=block)
    candidate_model.load_state_dict(candidate_checkpoint["model"], strict=True)
    device = torch.device("cuda:0")
    local_model = local_model.to(device).eval()
    candidate_model = candidate_model.to(device).eval()

    selected = load_selection(
        args.comparison_csv.expanduser().resolve(),
        args.recording_group,
        args.worst_count,
        args.best_count,
    )
    dataset = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=int(baseline_args["val_patch_size"]),
        augment=False,
        include_recording_groups=(args.recording_group,),
    )
    item_index = {
        f"{pair.scene}/{pair.name}": index for index, pair in enumerate(dataset.pairs)
    }

    columns = ("Blur input", "Sharp target", "Local A", "Candidate G", "|G-A| x10", "Gate map")
    tile = int(baseline_args["val_patch_size"])
    label_width = 285
    header_height = 34
    row_height = tile + 34
    canvas = Image.new(
        "RGB",
        (label_width + len(columns) * tile, header_height + len(selected) * row_height),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, title in enumerate(columns):
        draw.text((label_width + column * tile + 6, 10), title, fill="black", font=font)

    saved_selection = []
    for row_index, selection in enumerate(selected):
        name = selection["name"]
        if name not in item_index:
            raise RuntimeError(f"selected image not found in dataset: {name}")
        item = dataset[item_index[name]]
        blurry = item["blur"].unsqueeze(0).to(device)
        sharp = item["sharp"].unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            local_output = local_model(blurry)
            candidate_output, diagnostics = candidate_model(blurry, return_geometry=True)
        difference = (candidate_output - local_output).abs().mul(10.0).clamp(0, 1)
        gate = diagnostics["nonlocal_gate"]
        images = (
            tensor_image(blurry[0]),
            tensor_image(sharp[0]),
            tensor_image(local_output[0]),
            tensor_image(candidate_output[0]),
            tensor_image(difference[0]),
            gate_image(gate, (tile, tile)),
        )
        top = header_height + row_index * row_height
        for column, image in enumerate(images):
            canvas.paste(image, (label_width + column * tile, top))
        mean_delta = float(selection["two_seed_mean_candidate_minus_local_psnr_db"])
        seed2_delta = float(selection["seed7813_candidate_minus_local_psnr_db"])
        gate_mean = float(gate.float().mean())
        label_lines = (
            f"{selection['selection'].upper()}  {name}",
            f"two-seed delta: {mean_delta:+.4f} dB",
            f"seed7813 delta: {seed2_delta:+.4f} dB",
            f"input PSNR: {float(selection['input_psnr_db']):.3f} dB",
            f"gate mean: {gate_mean:.4f}",
        )
        for line_index, line in enumerate(label_lines):
            draw.text((8, top + 12 + 18 * line_index), line, fill="black", font=font)
        saved_selection.append(
            {
                **selection,
                "visualized_seed": int(candidate_args["seed"]),
                "gate_mean": gate_mean,
            }
        )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    canvas.save(output / "failure_success_grid.png")
    (output / "selection.json").write_text(
        json.dumps(saved_selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved {output / 'failure_success_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
