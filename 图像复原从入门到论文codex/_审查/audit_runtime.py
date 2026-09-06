from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


SRC = Path(r"J:\phd\图像复原从入门到论文")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR):
    """Audit-only reader that bypasses cv2.imread's Windows Unicode-path failure."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags)


def r2_score(x: list[float], y: list[float]) -> tuple[float, float, float]:
    coeff = np.polyfit(x, y, 1)
    predicted = np.polyval(coeff, x)
    y_arr = np.asarray(y)
    denom = np.sum((y_arr - y_arr.mean()) ** 2)
    r2 = float("nan") if denom == 0 else float(1 - np.sum((y_arr - predicted) ** 2) / denom)
    return float(coeff[0]), float(coeff[1]), r2


def main() -> None:
    degradation = load_module(
        "degradation_template",
        SRC / "01_退化模型与评价指标" / "degradation_template.py",
    )
    metrics = load_module(
        "metrics_template",
        SRC / "01_退化模型与评价指标" / "metrics_template.py",
    )
    deconv = load_module(
        "deconvolution_template",
        SRC / "02_非盲反卷积" / "deconvolution_template.py",
    )
    transition = load_module(
        "transition_zone_template",
        SRC / "03_边缘与过渡区提取" / "transition_zone_template.py",
    )

    report: dict[str, object] = {}

    # Validate the line-length semantics of the motion PSF.
    psf_checks = []
    for length in [3, 5, 10, 15, 20, 25]:
        psf = degradation.motion_blur_psf(kernel_size=61, angle=0, length=length)
        cols = np.flatnonzero(psf.sum(axis=0) > 0)
        psf_checks.append(
            {
                "requested_length": length,
                "nonzero_samples": int(np.count_nonzero(psf)),
                "support_span_pixels": int(cols[-1] - cols[0] + 1),
                "sum": float(psf.sum()),
            }
        )
    report["motion_psf_length_checks"] = psf_checks

    # Show what the documented root-directory invocation resolves to.
    report["path_resolution"] = {
        "cwd_project_root": str((SRC / ".." / "data" / "clean").resolve()),
        "exists_when_run_from_project_root": (SRC / ".." / "data" / "clean").resolve().exists(),
        "cwd_stage_01": str((SRC / "01_退化模型与评价指标" / ".." / "data" / "clean").resolve()),
        "exists_when_run_from_stage_dir": (SRC / "01_退化模型与评价指标" / ".." / "data" / "clean").resolve().exists(),
    }

    # Compare the hand-written SSIM against the standard implementation.
    ssim_checks = []
    for clean_path in sorted((SRC / "data" / "clean").glob("*.jpg")):
        blur_path = SRC / "data" / "synthetic_blur" / f"{clean_path.stem}_motion.jpg"
        clean = imread_unicode(clean_path, cv2.IMREAD_GRAYSCALE)
        blurred = imread_unicode(blur_path, cv2.IMREAD_GRAYSCALE)
        ours = float(metrics.compute_ssim(clean, blurred))
        reference = float(structural_similarity(clean, blurred, data_range=255))
        ssim_checks.append(
            {
                "image": clean_path.stem,
                "ours": ours,
                "skimage": reference,
                "difference": ours - reference,
            }
        )
    report["ssim_checks"] = ssim_checks

    # Measure actual restoration performance on all supplied synthetic pairs.
    restoration = []
    configs = {
        "motion": degradation.motion_blur_psf(kernel_size=21, angle=25, length=10),
        "defocus": degradation.defocus_blur_psf(kernel_size=21, radius=5),
        "gaussian": degradation.gaussian_blur_psf(kernel_size=21, sigma=3.0),
    }
    for clean_path in sorted((SRC / "data" / "clean").glob("*.jpg")):
        clean = imread_unicode(clean_path)
        for blur_type, psf in configs.items():
            blurred = imread_unicode(
                SRC / "data" / "synthetic_blur" / f"{clean_path.stem}_{blur_type}.jpg"
            )
            baseline = metrics.evaluate_restoration(blurred, clean)
            wiener = metrics.evaluate_restoration(deconv.wiener_filter(blurred, psf, K=0.01), clean)
            rl = metrics.evaluate_restoration(deconv.richardson_lucy(blurred, psf, iterations=30), clean)
            restoration.append(
                {
                    "image": clean_path.stem,
                    "blur": blur_type,
                    "baseline": baseline,
                    "wiener": wiener,
                    "rl": rl,
                }
            )
    report["restoration_checks"] = restoration

    # Reproduce the claimed transition-width trend on every supplied clean image.
    transition_checks = []
    lengths = [0, 5, 10, 15, 20, 25]
    for clean_path in sorted((SRC / "data" / "clean").glob("*.jpg")):
        clean = imread_unicode(clean_path)
        values = transition.analyze_blur_effect(clean, blur_lengths=lengths[1:])
        widths = [float(item["width"]) for item in values]
        mean_grads = [float(item["mean_grad"]) for item in values]
        slope, intercept, r2 = r2_score(lengths[1:], widths[1:])
        transition_checks.append(
            {
                "image": clean_path.stem,
                "lengths": lengths,
                "widths": widths,
                "mean_gradients": mean_grads,
                "fit_excluding_clean": {"slope": slope, "intercept": intercept, "r2": r2},
                "strictly_increasing": all(b > a for a, b in zip(widths, widths[1:])),
            }
        )
    report["transition_width_checks"] = transition_checks

    # Inventory images and exact duplicate content.
    images = []
    hashes: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        rel = str(path.relative_to(SRC))
        record = {"path": rel, "readable": image is not None}
        if image is not None:
            record.update({"shape": list(image.shape), "dtype": str(image.dtype)})
        images.append(record)
        hashes.setdefault(sha256(path), []).append(rel)
    report["image_inventory"] = images
    report["exact_duplicate_groups"] = [paths for paths in hashes.values() if len(paths) > 1]

    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
