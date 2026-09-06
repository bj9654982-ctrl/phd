"""Validate the CUDA-only research environment without modifying it."""

from __future__ import annotations

import json
import platform
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


REQUIRED_IMPORTS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "tqdm": "tqdm",
    "matplotlib": "matplotlib",
    "opencv-python": "cv2",
    "scikit-image": "skimage",
    "einops": "einops",
}


def package_version(distribution: str, module: object) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
        "errors": [],
    }

    modules: dict[str, object] = {}
    for distribution, import_name in REQUIRED_IMPORTS.items():
        try:
            module = import_module(import_name)
            modules[import_name] = module
            report["packages"][distribution] = package_version(distribution, module)  # type: ignore[index]
        except Exception as exc:  # diagnostic script must report every missing dependency
            report["errors"].append(f"{distribution}: {exc}")  # type: ignore[union-attr]

    torch = modules.get("torch")
    if torch is None:
        report["errors"].append("PyTorch is unavailable; CUDA checks were skipped.")  # type: ignore[union-attr]
    elif not torch.cuda.is_available():
        report["errors"].append("CUDA is unavailable. This project intentionally has no CPU training path.")  # type: ignore[union-attr]
    else:
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        report["cuda"] = {
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(device),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_gib": round(props.total_memory / 1024**3, 2),
        }

        try:
            torch.manual_seed(3407)
            torch.cuda.manual_seed_all(3407)
            x = torch.randn(2, 16, 64, 64, device=device)
            conv = torch.nn.Conv2d(16, 16, 3, padding=1).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y = conv(x).square().mean()
            y.backward()
            torch.cuda.synchronize()
            report["cuda_smoke_test"] = {
                "passed": bool(torch.isfinite(y).item()),
                "loss": float(y.detach().cpu()),
                "dtype": str(y.dtype),
                "allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
            }
        except Exception as exc:
            report["errors"].append(f"CUDA smoke test failed: {exc}")  # type: ignore[union-attr]

    report["ok"] = not report["errors"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

