"""Five-image official Restormer reference; TRAIN-SPLIT OVERLAP, NOT FAIR TEST."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_p16_factorial import read_csv, write_csv, sha256
from scripts.analyze_p20_stability import check, close
from scripts.evaluate_p21_full_image import save_json
from scripts.visualize_p13_failure_cases import tensor_image
from scripts import train_p12_matched_nonlocal as training

WEIGHT_HASH = "194e38fb5b607c9dc5a5b3e08e65b2e79ee2bf0ef5048e0612f6b2ff2f79da31"


def pad_native(image):
    h, w = image.shape[-2:]
    return F.pad(image, (0, (-w) % 8, 0, (-h) % 8), mode="reflect"), h, w


def load_rgb(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    check(image is not None, "OpenCV decode failed")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with Image.open(path) as pil:
        check(np.array_equal(rgb, np.asarray(pil.convert("RGB"))), "PIL/OpenCV pixels differ")
    return torch.from_numpy(np.float32(rgb) / 255.).permute(2, 0, 1).contiguous()[None]


def draw(images, case, public_score, target):
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    keys = ["blur", "local", "C00", "C01", "Restormer", "sharp"]
    titles = ["模糊输入", "Local", "C00", "C01", "Restormer公开预训练", "清晰真值"]
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), gridspec_kw={"height_ratios": [1, 1.15, 1.15]})
    for col, (key, title) in enumerate(zip(keys, titles)):
        array = np.asarray(images[key])
        score = public_score if key == "Restormer" else case["native_replay_scores"].get(key)
        if score is not None:
            title += f"\n{score['psnr_db']:.4f} dB / {score['ssim']:.6f}"
        axes[0, col].imshow(array, interpolation="nearest")
        axes[0, col].set_title(title, fontsize=10)
        for i, (x, y, w, h) in enumerate(case["detail_boxes_xywh"]):
            axes[0, col].add_patch(Rectangle((x, y), w, h, fill=False,
                                           edgecolor=("red", "cyan")[i], linewidth=1))
            axes[i+1, col].imshow(array[y:y+h, x:x+w], interpolation="nearest")
        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle("仅作能力展示：公开权重与这些图像的训练划分重叠，模型/预算/精度不同，非公平测试\n"
                 f"{case['selection']} | {case['name']}", fontsize=13)
    fig.text(.5, .012, "原生整图推理；五组相同放大位置，无后期锐化。分数为项目浮点指标，"
             "不是官方MATLAB论文复分。", ha="center", fontsize=10)
    fig.subplots_adjust(left=.015, right=.995, top=.86, bottom=.045, wspace=.07, hspace=.15)
    fig.savefig(target, dpi=160)
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
    vendor = ROOT / "third_party/restormer_official_68dc6ac_v1"
    provenance = json.loads((vendor / "download_provenance.json").read_text(encoding="utf-8"))
    check(provenance["complete"], "Incomplete official download")
    for file in provenance["files"]:
        check(sha256(vendor / file["path"]) == file["sha256"], "Vendor file changed")
    check(sha256(vendor / "motion_deblurring.pth") == WEIGHT_HASH, "Unexpected weight")
    settings = yaml.safe_load((vendor / "Motion_Deblurring/Options/Deblurring_Restormer.yml").read_text())["network_g"]
    check(settings.pop("type") == "Restormer", "Wrong official network")
    architecture = vendor / "basicsr/models/archs/restormer_arch.py"
    spec = importlib.util.spec_from_file_location("official_restormer_frozen", architecture)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    training.set_determinism(3407)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = module.Restormer(**settings)
    weights = torch.load(vendor / "motion_deblurring.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(weights["params"], strict=True)
    model = model.cuda().eval()
    visuals = root / "P21_visual_review_v0"
    selected = json.loads((visuals / "summary.json").read_text(encoding="utf-8"))["cases"]
    check(len(selected) == 5, "Expected five existing cases")
    manifest = {r["name"]: r for r in read_csv(root / "P21_native_seed3407_v0/manifest.csv")}
    output.mkdir(parents=True)
    rows = []
    try:
        save_json(output / "plan_before_evaluation.json", {
            "cases": [c["name"] for c in selected], "network_settings": settings,
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_split_overlap": True, "fair_comparison": False, "official_test_accessed": False,
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
            "AMP": False, "TF32": False, "weight_sha256": WEIGHT_HASH,
            "code_commit": provenance["commit"], "adapter_sha256": sha256(Path(__file__)),
            "protocol_sha256": sha256(ROOT / "docs/22_P23公开预训练能力参照协议.md")})
        for i, case in enumerate(selected):
            entry = manifest[case["name"]]
            for key in ("blurry", "sharp"):
                check(sha256(Path(entry[f"{key}_path"])) == entry[f"{key}_sha256"], "Data changed")
            save_json(output / "progress.json", {"case": i+1, "name": case["name"]})
            blurry = load_rgb(Path(entry["blurry_path"])).cuda()
            padded, h, w = pad_native(blurry)
            check((h, w) == (720, 1280), "Unexpected native shape")
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", enabled=False):
                prediction = model(padded)[..., :h, :w]
            torch.cuda.synchronize()
            check(prediction.shape == blurry.shape and bool(torch.isfinite(prediction).all()), "Invalid output")
            if i == 0:
                save_json(output / "resource_preflight.json", {
                    "passed": True, "quality_not_computed_yet": True, "shape": list(blurry.shape),
                    "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20})
                print("Native FP32 probe passed; starting five-image quality reference.", flush=True)
            sharp = load_rgb(Path(entry["sharp_path"])).cuda()
            score = {"psnr_db": training._psnr_value(prediction, sharp),
                     "ssim": training._ssim_value(prediction[0], sharp[0])}
            row = {"name": case["name"], "selection": case["selection"],
                   "Restormer_psnr_db": score["psnr_db"], "Restormer_ssim": score["ssim"],
                   **{f"{r}_{m}": case["native_replay_scores"][r][m]
                      for r in ("local", "C00", "C01") for m in ("psnr_db", "ssim")},
                   "training_split_overlap": True, "fair_comparison": False}
            directory = output / f"case{i+1}"
            directory.mkdir()
            public_image = tensor_image(prediction[0])
            public_image.save(directory / "Restormer.png")
            images = {"Restormer": public_image}
            for key in ("blur", "local", "C00", "C01", "sharp"):
                with Image.open(visuals / f"case{i+1}/{key}.png") as image:
                    images[key] = image.convert("RGB").copy()
            check(np.array_equal(np.asarray(images["blur"]), np.asarray(tensor_image(blurry[0]))),
                  "Visual input differs")
            draw(images, case, score, directory / "comparison.png")
            rows.append(row)
            write_csv(output / "per_image_reference.csv", rows)
            print(f"{i+1}/5 {case['name']} Restormer={score['psnr_db']:.4f}dB/{score['ssim']:.6f}", flush=True)
            del prediction, blurry, padded, sharp
        save_json(output / "summary.json", {"complete": True, "rows": rows,
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_split_overlap": True, "fair_comparison": False,
            "paper_result": False, "official_test_accessed": False})
        save_json(output / "progress.json", {"complete": True})
    except Exception as error:
        save_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                                           "traceback": traceback.format_exc(), "no_automatic_fallback": True})
        raise


if __name__ == "__main__":
    main()
