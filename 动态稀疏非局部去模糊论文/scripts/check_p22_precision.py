"""Bounded three-image precision sensitivity check, not a new trained method."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_p21_full_image import load_model, clean_gpu, save_json
from scripts.analyze_p20_stability import RUNS, ROLES, close, check, FROZEN
from scripts.analyze_p16_factorial import read_csv, write_csv, sha256
from scripts import train_p12_matched_nonlocal as training
from src.data.gopro_pairs import _load_rgb


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
    for relative, digest in provenance["source_sha256"].items():
        if relative.startswith("src"):
            check(sha256(ROOT / relative) == digest, "Model/data source changed")
    check(sha256(p21 / "manifest.csv") == provenance["manifest_sha256"], "Manifest changed")
    names = json.loads((p21 / "resource_preflight.json").read_text(encoding="utf-8"))["benchmark_names"]
    check(len(names) == 3, "Expected three previously fixed images")
    manifest = {r["name"]: r for r in read_csv(p21 / "manifest.csv")}
    reference = {role: {r["name"]: r for r in read_csv(p21 / f"{role}_full_image.csv")} for role in ROLES}
    checkpoints = {}
    for role, run in RUNS[3407].items():
        path = root / run / "best.pt"
        check(sha256(path) == provenance["run_sha256"][role]["best.pt"], "Checkpoint changed")
        checkpoints[role] = torch.load(path, map_location="cpu", weights_only=False)
    baseline_path = Path(checkpoints["local"]["args"]["baseline_checkpoint"])
    check(sha256(baseline_path) == FROZEN["baseline"], "B0 changed")
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    output.mkdir(parents=True)
    save_json(output / "plan_before_evaluation.json", {
        "names": names, "seed": 3407, "roles": list(ROLES),
        "selection": "same three midpoint images frozen before P21 full quality evaluation",
        "modes": ["P21 AMP float16 with cudnn TF32 allowed", "AMP disabled, both TF32 flags disabled"],
        "purpose": "precision sensitivity only; not isolate sampling-grid error or prove full-set quality",
        "stop_rule": "all nine paired forwards; stop on technical errors, no resize/tile/retraining/search",
        "new_training": False, "official_test_accessed": False, "script_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": {r: provenance["run_sha256"][r]["best.pt"] for r in ROLES}})
    rows = []
    for name in names:
        entry = manifest[name]
        for key in ("blurry", "sharp"):
            check(sha256(Path(entry[f"{key}_path"])) == entry[f"{key}_sha256"], "Image changed")
        for role in ROLES:
            print(f"Precision check {role}: {name}", flush=True)
            clean_gpu()
            model = load_model(checkpoints[role], baseline)
            blurry = _load_rgb(Path(entry["blurry_path"]))[None].cuda()
            sharp = _load_rgb(Path(entry["sharp_path"]))[None].cuda()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
            with torch.autocast("cuda", dtype=torch.float16):
                amp = model(blurry)
            amp_psnr = training._psnr_value(amp, sharp)
            amp_ssim = training._ssim_value(amp[0], sharp[0])
            check(close(amp_psnr, reference[role][name]["candidate_psnr_db"]) and
                  close(amp_ssim, reference[role][name]["candidate_ssim"]), "P21 AMP replay changed")
            torch.backends.cudnn.allow_tf32 = False
            with torch.autocast("cuda", enabled=False):
                fp32 = model(blurry)
            check(bool(torch.isfinite(fp32).all()), "Nonfinite FP32 output")
            fp32_psnr = training._psnr_value(fp32, sharp)
            fp32_ssim = training._ssim_value(fp32[0], sharp[0])
            diff = (fp32.float().clamp(0, 1)-amp.float().clamp(0, 1)).abs()
            rows.append({"name": name, "role": role, "amp_psnr_db": amp_psnr,
                         "amp_ssim": amp_ssim, "fp32_psnr_db": fp32_psnr, "fp32_ssim": fp32_ssim,
                         "fp32_minus_amp_psnr_db": fp32_psnr-amp_psnr,
                         "fp32_minus_amp_ssim": fp32_ssim-amp_ssim,
                         "output_mae_0to1": float(diff.mean()), "output_max_abs_0to1": float(diff.max())})
            write_csv(output / "per_image_precision.csv", rows)
            del model, blurry, sharp, amp, fp32, diff
    indexed = {(r["name"], r["role"]): r for r in rows}
    comparisons = []
    for name in names:
        a, b = indexed[name, "C01"], indexed[name, "C00"]
        comparisons.append({"name": name, **{
            f"{mode}_C01_minus_C00_{metric}": a[f"{mode}_{metric}"]-b[f"{mode}_{metric}"]
            for mode in ("amp", "fp32") for metric in ("psnr_db", "ssim")}})
    save_json(output / "summary.json", {"complete": True, "comparisons": comparisons,
        "full_dataset_conclusion": False, "grid_error_isolated": False,
        "original_weights_changed": False, "official_test_accessed": False})
    print(json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()
