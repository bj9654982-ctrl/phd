"""Frozen seed-3407 native development evaluation and isolated CUDA latency."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
import skimage
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_p12_matched_nonlocal as training
from scripts.analyze_p20_stability import (
    RUNS, ROLES, GROUPS, STRENGTHS, FROZEN, check, close, audit_run,
)
from scripts.analyze_p16_factorial import read_csv, write_csv, sha256
from src.data.gopro_pairs import _load_rgb, recording_group

P20_HASH = "9724c2ee37d1dd48d43f13432b3e2df9ccb182a9cdc4e48cbb205aee04772a98"
SEED = 3407
WARMUP = 10
REPEATS = 30
ORDER = (("local", "C00", "C01"), ("C00", "C01", "local"), ("C01", "local", "C00"))
CONTRASTS = {"C01_minus_local": ("C01", "local"), "C01_minus_C00": ("C01", "C00")}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def telemetry():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,"
            "utilization.gpu,temperature.gpu,pstate,clocks.sm", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return {"returncode": result.returncode, "output": result.stdout.strip(),
                "error": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


def benchmark_names(manifest):
    check(Counter(row["recording_group"] for row in manifest) == GROUPS, "Wrong benchmark groups")
    return [sorted(row["name"] for row in manifest if row["recording_group"] == group)[n // 2]
            for group, n in sorted(GROUPS.items())]


def audit_manifest(manifest, reference):
    check(len(manifest) == 385 and len({r["name"] for r in manifest}) == 385, "Wrong manifest size")
    check({r["name"] for r in manifest} == set(reference), "Manifest differs from P20")
    check(Counter(r["recording_group"] for r in manifest) == GROUPS, "Manifest groups differ")
    check(Counter(r["crop_proxy_class"] for r in manifest) == STRENGTHS, "Manifest proxy groups differ")
    check(all((r["height"], r["width"]) == (720, 1280) for r in manifest),
          "Unexpected native dimensions; stop and preregister expanded shape probes")
    for row in manifest:
        check(row["crop_proxy_class"] == reference[row["name"]]["strength_class"],
              "Crop-defined proxy labels changed")


def aggregate_quality(records):
    names = sorted(records["local"])
    check(len(names) == 385 and all(set(records[r]) == set(names) for r in ROLES),
          "Missing or unmatched full-image results")
    fields = ("candidate_psnr_db", "candidate_ssim", "input_psnr_db", "input_ssim")
    for role in ROLES:
        for name, row in records[role].items():
            check(all(np.isfinite(float(row[f])) for f in fields), "Nonfinite full-image metric")
            for field in ("recording_group", "crop_proxy_class", "height", "width"):
                check(row[field] == records["local"][name][field], "Full-image metadata mismatch")
            check(close(row["input_psnr_db"], records["local"][name]["input_psnr_db"]),
                  "Different full-image input")

    def subset(selected):
        scores = {role: {m: float(np.mean([records[role][n][f"candidate_{m}"] for n in selected]))
                         for m in ("psnr_db", "ssim")} for role in ROLES}
        return {"pairs": len(selected), "scores": scores,
                "input": {m: float(np.mean([records["local"][n][f"input_{m}"] for n in selected]))
                          for m in ("psnr_db", "ssim")},
                "contrasts": {key: {m: scores[a][m] - scores[b][m] for m in ("psnr_db", "ssim")}
                              for key, (a, b) in CONTRASTS.items()}}
    return {"overall": subset(names),
            **{key: {label: subset([n for n in names if records["local"][n][field] == label])
                     for label in sorted({records["local"][n][field] for n in names})}
               for key, field in (("per_group", "recording_group"),
                                  ("per_crop_proxy", "crop_proxy_class"))}}


def direction_decision(quality):
    signs = {key: value["psnr_db"] > 0 and value["ssim"] >= 0
             for key, value in quality["overall"]["contrasts"].items()}
    return {"overall_direction_checks": signs, "overall_direction_continued": all(signs.values()),
            "statistical_significance_established": False, "three_seed_full_replication": False}


def latency_statistics(values):
    values = np.asarray(values, dtype=float)
    check(values.size > 0 and np.isfinite(values).all() and (values > 0).all(), "Invalid latency")
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95))}


def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def load_model(checkpoint, baseline):
    model = training._make_candidate(baseline["model"], int(baseline["args"]["width"]),
                                    checkpoint["args"]["variant"], (1, 3, 7), 6, 5)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.cuda().eval()


def snapshot(output, protocol):
    directory = output / "source_snapshot"
    directory.mkdir()
    files = list((ROOT / "src").rglob("*.py")) + [
        Path(__file__), ROOT / "scripts/train_p12_matched_nonlocal.py",
        ROOT / "scripts/analyze_p20_stability.py", ROOT / "scripts/analyze_p16_factorial.py"]
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = sha256(source)
    shutil.copy2(protocol, directory / "protocol.md")
    hashes["protocol.md"] = sha256(protocol)
    return hashes


@torch.inference_mode()
def probe_and_benchmark(checkpoints, baseline, manifest, output):
    by_name = {r["name"]: r for r in manifest}
    names = benchmark_names(manifest)
    probes, timings, blocks = [], [], []
    save_json(output / "resource_preflight.json", {"passed": False, "quality_computed": False,
                                                  "benchmark_names": names, "probes": probes})
    # Probe all models/shapes first, before any quality result or formal timing.
    for name in names:
        for role in ROLES:
            save_json(output / "run_state.json", {"phase": "resource_probe", "role": role, "name": name})
            clean_gpu()
            model = load_model(checkpoints[role], baseline)
            blurry = _load_rgb(Path(by_name[name]["blurry_path"]))[None].cuda()
            dtype_record = {}
            handle = None
            if model.bottleneck_module is not None:
                def record_dtype(module, inputs, outputs):
                    dtype_record["key_value_output_dtype"] = str(outputs.dtype)
                    dtype_record["grid_dtype_by_unchanged_implementation"] = str(outputs.dtype)
                handle = model.bottleneck_module.key_value.register_forward_hook(record_dtype)
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model(blurry)
            torch.cuda.synchronize()
            check(prediction.shape == blurry.shape and bool(torch.isfinite(prediction).all()),
                  "Probe nonfinite output or shape mismatch")
            if handle is not None:
                handle.remove()
            probes.append({"name": name, "role": role, "shape": list(blurry.shape),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, **dtype_record})
            save_json(output / "resource_preflight.json", {"passed": False, "quality_computed": False,
                                                          "benchmark_names": names, "probes": probes})
            del prediction, blurry, model
    save_json(output / "resource_preflight.json", {"passed": True, "quality_computed": False,
                                                  "benchmark_names": names, "probes": probes})
    print("Native 720x1280 resource probes passed for all three models; benchmarking", flush=True)
    for round_index, name in enumerate(names):
        for role in ORDER[round_index]:
            save_json(output / "run_state.json", {"phase": "benchmark", "role": role, "name": name})
            clean_gpu()
            model = load_model(checkpoints[role], baseline)
            blurry = _load_rgb(Path(by_name[name]["blurry_path"]))[None].cuda()
            before = telemetry()
            with torch.autocast("cuda", dtype=torch.float16):
                for _ in range(WARMUP):
                    prediction = model(blurry)
                    del prediction
            torch.cuda.synchronize()
            base_alloc = torch.cuda.memory_allocated()
            base_reserved = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            for repetition in range(REPEATS):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                wall_start = time.perf_counter()
                begin.record()
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = model(blurry)
                end.record()
                end.synchronize()
                wall_ms = (time.perf_counter() - wall_start) * 1000
                timings.append({"round": round_index, "name": name, "role": role,
                    "repetition": repetition, "cuda_event_ms": begin.elapsed_time(end),
                    "synchronized_host_ms": wall_ms})
                del prediction, begin, end
            peak = torch.cuda.max_memory_allocated()
            blocks.append({"round": round_index, "name": name, "role": role,
                "warmup": WARMUP, "repeats": REPEATS, "parameters": sum(p.numel() for p in model.parameters()),
                "baseline_allocated_mib": base_alloc / 2**20, "baseline_reserved_mib": base_reserved / 2**20,
                "peak_allocated_mib": peak / 2**20,
                "incremental_peak_allocated_mib": (peak - base_alloc) / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "telemetry_before": before, "telemetry_after": telemetry()})
            del blurry, model
            write_csv(output / "latency_raw.csv", timings)
            save_json(output / "latency_blocks.json", blocks)
    clean_gpu()
    summary = {role: {
        "cuda_event_ms": latency_statistics([r["cuda_event_ms"] for r in timings if r["role"] == role]),
        "synchronized_host_ms": latency_statistics([r["synchronized_host_ms"] for r in timings if r["role"] == role]),
        "max_peak_allocated_mib": max(b["peak_allocated_mib"] for b in blocks if b["role"] == role),
        "max_incremental_peak_allocated_mib": max(b["incremental_peak_allocated_mib"] for b in blocks if b["role"] == role),
        "max_peak_reserved_mib": max(b["peak_reserved_mib"] for b in blocks if b["role"] == role),
        "per_image": {name: {metric: latency_statistics([
            r[metric] for r in timings if r["role"] == role and r["name"] == name])
            for metric in ("cuda_event_ms", "synchronized_host_ms")} for name in names},
        } for role in ROLES}
    return summary


@torch.inference_mode()
def evaluate(checkpoints, baseline, data, manifest, calibration, original_records, output):
    records, evaluation_wall = {}, {}
    by_name = {r["name"]: r for r in manifest}
    for role in ROLES:
        save_json(output / "run_state.json", {"phase": "quality", "role": role, "name": None})
        clean_gpu()
        print(f"Full-image quality: {role}, 385 images", flush=True)
        model = load_model(checkpoints[role], baseline)
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=2,
                            pin_memory=True, persistent_workers=True,
                            generator=torch.Generator().manual_seed(SEED + 2))
        records[role] = {}
        start = time.perf_counter()
        with (output / f"{role}_full_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = None
            for index, batch in enumerate(loader):
                name = batch["name"][0]
                save_json(output / "run_state.json", {"phase": "quality", "role": role, "name": name})
                check(name in by_name and name not in records[role], "Unexpected/duplicate quality image")
                blurry, sharp = batch["blur"].cuda(), batch["sharp"].cuda()
                check(tuple(blurry.shape) == (1, 3, 720, 1280) and sharp.shape == blurry.shape,
                      "Quality image shape mismatch")
                if role == "local":
                    # Same 256 center crop and contiguous layout as the historical loader.
                    crop_input_psnr = training._psnr_value(
                        blurry[..., 232:488, 512:768].contiguous(),
                        sharp[..., 232:488, 512:768].contiguous())
                    check(close(crop_input_psnr, original_records["local"][name]["input_psnr_db"]),
                          f"Historical center-crop input changed: {name}")
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = model(blurry)
                check(prediction.shape == blurry.shape and bool(torch.isfinite(prediction).all()),
                      "Invalid quality prediction")
                input_psnr = training._psnr_value(blurry, sharp)
                input_ssim = (training._ssim_value(blurry[0], sharp[0]) if role == "local" else
                              records["local"][name]["input_ssim"])
                row = {"name": name, "recording_group": by_name[name]["recording_group"],
                    "height": 720, "width": 1280, "crop_proxy_class": by_name[name]["crop_proxy_class"],
                    "full_input_proxy_class_descriptive": training._strength_class(input_psnr,
                        calibration["heavy_medium_threshold_input_psnr_db"],
                        calibration["medium_light_threshold_input_psnr_db"]),
                    "input_psnr_db": input_psnr, "input_ssim": input_ssim,
                    "candidate_psnr_db": training._psnr_value(prediction, sharp),
                    "candidate_ssim": training._ssim_value(prediction[0], sharp[0])}
                records[role][name] = row
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                del prediction, blurry, sharp
                if (index + 1) % 50 == 0 or index == 0:
                    print(f"{role}: {index+1}/385; elapsed={time.perf_counter()-start:.1f}s", flush=True)
        evaluation_wall[role] = time.perf_counter() - start
        del model, loader
    return records, evaluation_wall


def paired_rows(records):
    return [{"name": name, "recording_group": records["local"][name]["recording_group"],
             "crop_proxy_class": records["local"][name]["crop_proxy_class"],
             **{f"{role}_{metric}": records[role][name][f"candidate_{metric}"]
                for role in ROLES for metric in ("psnr_db", "ssim")},
             **{f"{label}_{metric}": records[a][name][f"candidate_{metric}"] -
                                    records[b][name][f"candidate_{metric}"]
                for label, (a, b) in CONTRASTS.items() for metric in ("psnr_db", "ssim")}}
            for name in sorted(records["local"])]


def report(result):
    quality, speed = result["quality"], result["latency"]
    lines = ["# P21 原生整图开发评价与效率", "",
        "种子3407；385张官方train内部开发图；720×1280原生整图；batch1 CUDA AMP。不是官方测试。", "",
        "## 质量", "", "| 模型 | PSNR/dB | SSIM |", "|---|---:|---:|"]
    for role, score in quality["overall"]["scores"].items():
        lines.append(f"| {role} | {score['psnr_db']:.6f} | {score['ssim']:.7f} |")
    lines += ["", "## 全部匹配差值", "",
        "| 分组 | 图数 | C01−Local dB | ΔSSIM | C01−C00 dB | ΔSSIM |",
        "|---|---:|---:|---:|---:|---:|"]
    for label, group in [("overall", quality["overall"]), *quality["per_group"].items(),
                         *quality["per_crop_proxy"].items()]:
        a, b = (group["contrasts"][key] for key in CONTRASTS)
        lines.append(f"| {label} | {group['pairs']} | {a['psnr_db']:+.6f} | {a['ssim']:+.7f} | "
                     f"{b['psnr_db']:+.6f} | {b['ssim']:+.7f} |")
    lines += ["", "代理档沿用P20中心裁剪标签；不是整图阈值重校准。", "",
              f"预定总体方向延续：{result['decision']['overall_direction_continued']}。", "",
              "## 热态延迟与PyTorch显存", "",
              "| 模型 | CUDA中位/ms | CUDA p95/ms | 同步host中位/ms | 峰值allocated/MiB | 增量allocated/MiB |",
              "|---|---:|---:|---:|---:|---:|"]
    for role, item in speed.items():
        lines.append(f"| {role} | {item['cuda_event_ms']['median']:.3f} | "
                     f"{item['cuda_event_ms']['p95']:.3f} | {item['synchronized_host_ms']['median']:.3f} | "
                     f"{item['max_peak_allocated_mib']:.2f} | {item['max_incremental_peak_allocated_mib']:.2f} |")
    lines += ["", "3张固定图×30次/模型，分别预热10次，三轮顺序轮转；原始记录与各图统计另存。",
        "不含读图、传输、指标、权重加载；不是端到端帧率。峰值不等于整个桌面显存。",
        "C00和C01的推理图相同，轻微耗时差不说明监督使模型加速。",
        "", "## 限制与状态", "",
        "- 单个整图种子；开发集已反复用于设计，不能替代独立测试或宣称显著性。",
        "- 不改FP16采样网格；较大尺寸的坐标量化及GroupNorm空间范围变化均可能影响结果。",
        "- 没有缩放、分块或TTA；没有按整图分数重新选择权重。",
        "- 完整分组和负值保留；阶段符号检查不等于+0.15 dB目标。",
        "- 版本、协议、源码、文件指纹、资源预检、计时遥测见同目录记录。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.runs_root.resolve(), args.output.resolve()
    check(not output.exists(), "Refusing to overwrite P21 output")
    check(torch.cuda.is_available(), "CUDA required; no CPU fallback")
    output.mkdir(parents=True)
    try:
        training.set_determinism(SEED)
        p20_path = root / "P20_three_seed_stability_v1/summary.json"
        check(sha256(p20_path) == P20_HASH, "P20 provenance changed")
        p20 = json.loads(p20_path.read_text(encoding="utf-8"))
        protocol_candidates = list((ROOT / "docs").glob("19_P21*.md"))
        check(len(protocol_candidates) == 1, "Expected fixed P21 protocol")
        sources = snapshot(output, protocol_candidates[0])
        check(sha256(ROOT / "scripts/train_p12_matched_nonlocal.py") == FROZEN["trainer"], "Trainer changed")
        for relative, digest in p20["evaluation_source_sha256"].items():
            check(sha256(ROOT / relative) == digest, "Model/data implementation changed since P20")
        checkpoints, original_records, run_hashes = {}, {}, {}
        common = torch.load(root / RUNS[SEED]["local"] / "best.pt",
                            map_location="cpu", weights_only=False)["args"]
        check(sha256(Path(common["baseline_checkpoint"])) == FROZEN["baseline"], "B0 changed")
        check(sha256(Path(common["calibration_summary"])) == FROZEN["calibration"], "Calibration changed")
        calibration = json.loads(Path(common["calibration_summary"]).read_text(encoding="utf-8"))
        for role, run in RUNS[SEED].items():
            historical = p20["by_seed"][str(SEED)]["audit"][role]["artifact_sha256"]
            for name, digest in historical.items():
                check(sha256(root / run / name) == digest, f"Historical file changed: {run}/{name}")
            _, rows, checkpoint, _, audit = audit_run(root / run, SEED, role, common, calibration)
            check(checkpoint["step"] == 3000, "P21 weight selection differs")
            checkpoints[role], original_records[role], run_hashes[role] = checkpoint, rows, audit["artifact_sha256"]
        baseline = torch.load(common["baseline_checkpoint"], map_location="cpu", weights_only=False)
        data = training.GoProPairDataset(common["data_root"], split="train", patch_size=None, augment=False,
                                        include_recording_groups=baseline["args"]["validation_groups"])
        manifest = []
        for pair in data.pairs:
            name = f"{pair.scene}/{pair.name}"
            with Image.open(pair.blurry) as image:
                width, height = image.size
            with Image.open(pair.sharp) as image:
                check(image.size == (width, height), "Blur/sharp header shape mismatch")
            manifest.append({"name": name, "recording_group": recording_group(pair.scene),
                "width": width, "height": height,
                "crop_proxy_class": original_records["local"][name]["strength_class"],
                "blurry_path": str(pair.blurry), "sharp_path": str(pair.sharp),
                "blurry_sha256": sha256(pair.blurry), "sharp_sha256": sha256(pair.sharp)})
        write_csv(output / "manifest.csv", manifest)
        audit_manifest(manifest, original_records["local"])
        environment = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "skimage": skimage.__version__,
            "gpu": torch.cuda.get_device_name(0), "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "telemetry": telemetry()}
        save_json(output / "environment.json", environment)
        save_json(output / "provenance.json", {"source_sha256": sources, "run_sha256": run_hashes,
            "p20_sha256": P20_HASH, "manifest_sha256": sha256(output / "manifest.csv")})
        print("Manifest audited: 385 native images; beginning no-quality resource probes", flush=True)
        latency = probe_and_benchmark(checkpoints, baseline, manifest, output)
        save_json(output / "latency_summary.json", latency)
        records, wall = evaluate(checkpoints, baseline, data, manifest, calibration, original_records, output)
        quality = aggregate_quality(records)
        write_csv(output / "paired_full_image.csv", paired_rows(records))
        for role, run in RUNS[SEED].items():
            for name, digest in run_hashes[role].items():
                check(sha256(root / run / name) == digest, "Training artifact changed during P21")
        for row in manifest:
            for key in ("blurry", "sharp"):
                check(sha256(Path(row[f"{key}_path"])) == row[f"{key}_sha256"], "Input file changed during P21")
        result = {"experiment": "P21 native development and latency", "seed": SEED,
            "official_test_accessed": False, "paper_result": False,
            "quality": quality, "latency": latency, "decision": direction_decision(quality),
            "quality_phase_wall_seconds_including_IO_and_metrics": wall,
            "environment": environment, "source_sha256": sources,
            "raw_data_and_training_artifacts_unchanged": True}
        save_json(output / "summary.json", result)
        (output / "report.md").write_text(report(result), encoding="utf-8")
        save_json(output / "run_state.json", {"phase": "complete"})
        print(json.dumps({"completed": True, "decision": result["decision"],
                          "quality": quality["overall"]}), flush=True)
    except Exception as error:
        save_json(output / "failure.json", {"error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "original_runs_modified": False,
            "no_automatic_tile_resize_or_checkpoint_fallback": True})
        raise


if __name__ == "__main__":
    main()
