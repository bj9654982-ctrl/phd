"""Show real frozen native-image restorations; no enhancement or new training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_p21_full_image import load_model, clean_gpu, save_json
from scripts.analyze_p20_stability import RUNS, ROLES, check, close, FROZEN
from scripts.analyze_p16_factorial import read_csv, sha256
from scripts.visualize_p13_failure_cases import tensor_image
from scripts import train_p12_matched_nonlocal as training
from src.data.gopro_pairs import _load_rgb


def select_cases(rows, fixed):
    indexed = {r["name"]: r for r in rows}
    check(len(indexed) == 385 and len(rows) == 385, "Expected all 385 P21 rows")
    ordered = sorted(rows, key=lambda r: (float(r["C01_minus_C00_psnr_db"]), r["name"]))
    chosen = [(name, "预先固定图像") for name in fixed]
    chosen += [(ordered[0]["name"], "按结果选出的PSNR最差案例"),
               (ordered[-1]["name"], "按结果选出的PSNR最好案例")]
    return [{"name": name, "selection": label, "metrics": indexed[name]} for name, label in chosen]


def detail_boxes(blurry, size=192, stride=96):
    gray = np.asarray(blurry, dtype=np.float32).mean(axis=2) / 255.
    height, width = gray.shape
    check(min(height, width) >= size, "Input too small for detail boxes")
    gy, gx = np.gradient(gray)
    gradient = np.sqrt(gx * gx + gy * gy)
    candidates = [(float(gradient[y:y+size, x:x+size].mean()), x, y)
                  for y in range(0, height-size+1, stride)
                  for x in range(0, width-size+1, stride)]
    # First maximum in fixed row-major order resolves ties.
    _, x, y = max(candidates, key=lambda item: item[0])
    return [(x, y, size, size), ((width-size)//2, (height-size)//2, size, size)]


def render(images, boxes, case, output, native_scores):
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    titles = ["模糊输入", "Local：局部基线", "C00：无监督非局部", "C01：有监督非局部", "清晰真值"]
    keys = ["blur", "local", "C00", "C01", "sharp"]
    fig, axes = plt.subplots(3, 5, figsize=(16, 9), gridspec_kw={"height_ratios": [1, 1.15, 1.15]})
    for col, (key, title) in enumerate(zip(keys, titles)):
        array = np.asarray(images[key])
        axes[0, col].imshow(array, interpolation="nearest")
        score = native_scores.get(key)
        caption = "" if score is None else f"\n全图 {score['psnr_db']:.4f} dB / SSIM {score['ssim']:.6f}"
        axes[0, col].set_title(title + caption, fontsize=10)
        for i, (x, y, w, h) in enumerate(boxes):
            axes[0, col].add_patch(Rectangle((x, y), w, h, fill=False,
                                           edgecolor=("red", "cyan")[i], linewidth=1.2))
            axes[i+1, col].imshow(array[y:y+h, x:x+w], interpolation="nearest")
            if col == 0:
                axes[i+1, col].set_title(("红框：输入边缘最强区域", "青框：固定中心区域")[i], fontsize=10)
        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    metric = case["metrics"]
    fig.suptitle(f"{case['selection']} | {case['name']}\n"
        f"C01−C00 全图：PSNR {float(metric['C01_minus_C00_psnr_db']):+.5f} dB，"
        f"SSIM {float(metric['C01_minus_C00_ssim']):+.7f}", fontsize=13)
    fig.text(.5, .014, "相同位置、相同显示范围；细节192×192，最近邻放大。未锐化/去噪/调色。"
             "标题指标基于原始浮点全图，非放大块，也非8位PNG重算。", ha="center", fontsize=10)
    fig.subplots_adjust(left=.025, right=.995, top=.865, bottom=.05, wspace=.06, hspace=.15)
    fig.savefig(output, dpi=160)
    plt.close(fig)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.runs_root.resolve(), args.output.resolve()
    check(not output.exists(), "Refusing overwrite")
    check(torch.cuda.is_available(), "CUDA required")
    training.set_determinism(3407)
    p21 = root / "P21_native_seed3407_v0"
    provenance = json.loads((p21 / "provenance.json").read_text(encoding="utf-8"))
    for role, files in provenance["run_sha256"].items():
        for name, digest in files.items():
            check(sha256(root / RUNS[3407][role] / name) == digest, "Frozen training artifact changed")
    for relative, digest in provenance["source_sha256"].items():
        if relative.startswith("src"):
            check(sha256(ROOT / relative) == digest, "Model/data source changed")
    check(sha256(p21 / "manifest.csv") == provenance["manifest_sha256"], "Manifest changed")
    manifest = {r["name"]: r for r in read_csv(p21 / "manifest.csv")}
    fixed = json.loads((p21 / "resource_preflight.json").read_text(encoding="utf-8"))["benchmark_names"]
    cases = select_cases(read_csv(p21 / "paired_full_image.csv"), fixed)
    reference = {role: {r["name"]: r for r in read_csv(p21 / f"{role}_full_image.csv")} for role in ROLES}
    first = torch.load(root / RUNS[3407]["local"] / "best.pt", map_location="cpu", weights_only=False)
    check(sha256(Path(first["args"]["baseline_checkpoint"])) == FROZEN["baseline"], "B0 changed")
    baseline = torch.load(first["args"]["baseline_checkpoint"], map_location="cpu", weights_only=False)
    output.mkdir(parents=True)
    save_json(output / "selection_before_render.json", {
        "cases": cases, "selection_rule": "three previously fixed + minimum/maximum C01-C00 full PSNR",
        "roi_rule": "192 square, strongest mean input gradient on stride96 grid + fixed center",
        "not_representative_sample": True, "new_training": False, "official_test_accessed": False,
        "script_sha256": sha256(Path(__file__)),
        "paired_csv_sha256": sha256(p21 / "paired_full_image.csv")})
    completed = []
    for index, case in enumerate(cases):
        print(f"Rendering {index+1}/{len(cases)}: {case['name']}", flush=True)
        directory = output / f"case{index+1}"
        directory.mkdir()
        entry = manifest[case["name"]]
        for key in ("blurry", "sharp"):
            check(sha256(Path(entry[f"{key}_path"])) == entry[f"{key}_sha256"], "Image file changed")
        blurry = _load_rgb(Path(entry["blurry_path"]))
        sharp = _load_rgb(Path(entry["sharp_path"]))
        images = {"blur": tensor_image(blurry), "sharp": tensor_image(sharp)}
        boxes = detail_boxes(images["blur"])
        scores = {"blur": {"psnr_db": float(reference["local"][case["name"]]["input_psnr_db"]),
                           "ssim": float(reference["local"][case["name"]]["input_ssim"])}}
        for role in ROLES:
            clean_gpu()
            checkpoint = torch.load(root / RUNS[3407][role] / "best.pt",
                                    map_location="cpu", weights_only=False)
            model = load_model(checkpoint, baseline)
            gpu_input, gpu_sharp = blurry[None].cuda(), sharp[None].cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model(gpu_input)
            check(prediction.shape == gpu_input.shape and bool(torch.isfinite(prediction).all()),
                  "Invalid restoration")
            score = {"psnr_db": training._psnr_value(prediction, gpu_sharp),
                     "ssim": training._ssim_value(prediction[0], gpu_sharp[0])}
            for metric, value in score.items():
                check(close(value, reference[role][case["name"]][f"candidate_{metric}"]),
                      f"Native replay changed: {role}/{case['name']}/{metric}")
            scores[role] = score
            images[role] = tensor_image(prediction[0])
            del model, prediction, gpu_input, gpu_sharp, checkpoint
        for key, image in images.items():
            image.save(directory / f"{key}.png")
        render(images, boxes, case, directory / "comparison.png", scores)
        completed.append({**case, "detail_boxes_xywh": boxes, "native_replay_scores": scores,
                          "figure": f"case{index+1}/comparison.png"})
        save_json(output / "summary.json", {"cases": completed, "all_cases_complete": len(completed)==len(cases),
            "native_replay_tolerance": 1e-6, "display_png_quantization": "clamp0..1, round to RGB8",
            "metrics_before_png_quantization": True, "no_visual_enhancement": True,
            "official_test_accessed": False, "paper_result": False})
    print("All five native replays match P21; comparison images saved.", flush=True)


if __name__ == "__main__":
    main()
