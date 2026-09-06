"""Audit nine frozen runs and replay best/fixed-step development crops on CUDA."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_p16_factorial import read_csv, write_csv, sha256
from scripts import train_p12_matched_nonlocal as training

RUNS = {
    3407: {"local": "P12A_local_control_v0", "C00": "P12G_uniform_scale_adaptive_gate_v0",
           "C01": "P16C01_teacher_only_seed3407_v0"},
    7813: {"local": "P13A_local_control_seed7813_v0", "C00": "P13G_gate_adaptive_seed7813_v0",
           "C01": "P17C01_teacher_only_seed7813_v0"},
    12011: {"local": "P20A_local_seed12011_v0", "C00": "P20G_unsupervised_seed12011_v0",
            "C01": "P20C01_paired_teacher_seed12011_v0"},
}
ROLES = ("local", "C00", "C01")
CONTRASTS = {"C01_minus_local": ("C01", "local"), "C01_minus_C00": ("C01", "C00")}
METRICS = ("psnr_db", "ssim")
GROUPS = {"GOPR0372_07": 175, "GOPR0378_13": 110, "GOPR0385_11": 100}
STRENGTHS = {"heavy": 96, "medium": 106, "light": 183}
FROZEN = {
    "trainer": "37fe90523cd9e50a8f3a3d07aecae1e175680bdd2bc9846f41277cf2fa378b24",
    "baseline": "8538e91d8327dc6e17e08ab008d114fe33be1c7a31e619352f62a5cedf3a78c5",
    "calibration": "dd704bbf6707cebfb418078da7c2d1f649090dcbc8c25193a45d3cb092cffc99",
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, tolerance=1e-6):
    a, b = float(a), float(b)
    return np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tolerance


def audit_setting(setting, seed, role):
    check(seed in RUNS and role in ROLES, "Unsupported seed/role")
    expected = dict(seed=seed, steps=3000, batch_size=6, train_patch_size=256,
                    learning_rate=.0002, frequency_weight=.1, val_interval=375,
                    num_workers=2, train_pairs=0, val_pairs=0,
                    variant="local" if role == "local" else "gate_adaptive")
    for key, value in expected.items():
        check(setting.get(key) == value, f"Unexpected setting: {key}")
    check(setting.get("sampling", "image_uniform") == "image_uniform", "Sampling mismatch")
    check(setting.get("gate_supervision_weight", 0.) == (.01 if role == "C01" else 0.),
          "Teacher weight mismatch")
    # Historical runs predate opt-in shuffled mode; absent mode was paired.
    check(setting.get("teacher_mode", "paired") == "paired" and
          setting.get("teacher_seed") is None, "Teacher assignment mismatch")
    if role != "local":
        check(tuple(setting["scales"]) == (1, 3, 7) and setting["top_k"] == 6 and
              setting["statistics_window"] == 5, "Nonlocal configuration mismatch")


def index_rows(rows):
    result = {row["name"]: row for row in rows}
    check(len(result) == len(rows) and len(rows) > 0, "Duplicate/empty image list")
    for row in rows:
        for field in ("input_psnr_db", "baseline_psnr_db", "baseline_ssim",
                      "candidate_psnr_db", "candidate_ssim"):
            check(np.isfinite(float(row[field])), f"Nonfinite {field}")
    return result


def compare_rows(rows, reference, *, predictions=False):
    check(set(rows) == set(reference), "Image list mismatch")
    fields = ["input_psnr_db", "baseline_psnr_db", "baseline_ssim"]
    if predictions:
        fields += ["candidate_psnr_db", "candidate_ssim"]
    for name, row in rows.items():
        for field in ("name", "scene", "recording_group", "strength_class"):
            check(row[field] == reference[name][field], f"Metadata mismatch: {name}/{field}")
        for field in fields:
            check(close(row[field], reference[name][field]), f"Replay mismatch: {name}/{field}")


def audit_history(history, summary, best, last):
    check([int(row["step"]) for row in history] == [1, *range(375, 3001, 375)],
          "Validation schedule mismatch")
    check(all(np.isfinite(float(r["candidate_psnr_db"])) for r in history), "Nonfinite history")
    winner = max(history, key=lambda r: float(r["candidate_psnr_db"]))
    check(int(winner["step"]) == best["step"] == summary["best_checkpoint_step"],
          "Best-checkpoint selection mismatch")
    check(close(winner["candidate_psnr_db"], summary["candidate_psnr_db"]),
          "Best summary/history mismatch")
    check(last["step"] == 3000 and last["args"] == best["args"], "Last-checkpoint mismatch")


def same_endpoint(best, last):
    if best["step"] != last["step"]:
        return False
    check(set(best["model"]) == set(last["model"]) and
          all(torch.equal(v, last["model"][k]) for k, v in best["model"].items()),
          "Same-step weights differ")
    return True


def aggregate(records):
    names = sorted(records["local"])
    check(all(set(records[r]) == set(names) for r in ROLES), "Unmatched aggregate rows")

    def subset(selected):
        scores = {role: {m: float(np.mean([float(records[role][n][f"candidate_{m}"])
                                         for n in selected])) for m in METRICS} for role in ROLES}
        return {"pairs": len(selected), "scores": scores,
                "contrasts": {label: {m: scores[a][m] - scores[b][m] for m in METRICS}
                              for label, (a, b) in CONTRASTS.items()}}

    output = {"overall": subset(names)}
    for key, field in (("per_group", "recording_group"), ("per_strength", "strength_class")):
        output[key] = {label: subset([n for n in names if records["local"][n][field] == label])
                       for label in sorted({records["local"][n][field] for n in names})}
    return output


def decision(best, last):
    gain = best["overall"]["contrasts"]["C01_minus_local"]
    gates = {"overall_psnr": gain["psnr_db"] >= .02, "overall_ssim": gain["ssim"] >= 0}
    for dimension in ("per_group", "per_strength"):
        gates[dimension] = all(g["contrasts"]["C01_minus_local"]["psnr_db"] >= -.01
                               for g in best[dimension].values())
    marginal = best["overall"]["contrasts"]["C01_minus_C00"]
    sign = marginal["psnr_db"] > 0 and marginal["ssim"] >= 0
    terminal = all(last["overall"]["contrasts"][key][metric] >= 0
                   for key in CONTRASTS for metric in METRICS)
    return {"local_screening_gates": gates, "local_screening_passed": all(gates.values()),
            "teacher_best_sign_passed": sign, "last_no_direction_reversal": terminal,
            "checkpoint_dependence_detected": not terminal,
            "all_preregistered_rules_passed": all(gates.values()) and sign and terminal}


def seed_statistics(values):
    values = np.asarray(values, dtype=float)
    check(values.shape == (3,) and np.isfinite(values).all(), "Need three finite seed values")
    return {"values_in_seed_order_3407_7813_12011": values.tolist(),
            "mean": float(values.mean()), "sample_sd_ddof1": float(values.std(ddof=1))}


def audit_config(directory, seed, setting):
    path = directory / "config.json"
    if not path.exists():
        check(seed in (3407, 7813), "New P20 run missing config.json")
        return {"config_available": False,
                "configuration_source": "checkpoint args; legacy run has no separate config.json"}
    config = json.loads(path.read_text(encoding="utf-8"))
    for key, value in setting.items():
        check(config.get(key) == (str(value) if isinstance(value, Path) else value),
              f"Config/checkpoint mismatch: {key}")
    return {"config_available": True, "configuration_source": "checkpoint args and config.json"}


def audit_run(directory, seed, role, common, calibration):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    setting = best["args"]
    audit_setting(setting, seed, role)
    for field in ("baseline_checkpoint", "calibration_summary", "data_root"):
        check(Path(setting[field]).resolve() == Path(common[field]).resolve(), f"Different {field}")
    audit_history(read_csv(directory / "metrics.csv"), summary, best, last)
    for key in ("seed", "steps", "variant"):
        check(summary[key] == setting[key], f"Summary {key} mismatch")
    check(summary.get("gate_supervision_weight", 0.) == (.01 if role == "C01" else 0.),
          "Summary teacher weight mismatch")
    check(summary.get("teacher_mode", "paired") == "paired" and
          summary.get("teacher_seed") is None, "Summary teacher mode mismatch")
    check(summary["candidate_parameters"] == (1064619 if role == "local" else 1095511),
          "Parameter count mismatch")
    check(summary["official_test_accessed"] is False and summary["pairs"] == 385
          and summary["val_pairs"] == 385 and summary["train_pairs"] == 1718
          and not summary["recording_group_overlap"]
          and not set(summary["train_recording_groups"]) & set(summary["validation_recording_groups"])
          and summary["validation_patch_size"] == 256
          and summary["exact_initialization_max_abs_error"] == 0
          and summary["baseline_checkpoint_step"] == 22500, "Split/initialization mismatch")
    records = index_rows(read_csv(directory / "per_image_metrics.csv"))
    check(Counter(r["recording_group"] for r in records.values()) == GROUPS, "Group counts mismatch")
    check(Counter(r["strength_class"] for r in records.values()) == STRENGTHS, "Strength counts mismatch")
    for row in records.values():
        label = training._strength_class(float(row["input_psnr_db"]),
                  calibration["heavy_medium_threshold_input_psnr_db"],
                  calibration["medium_light_threshold_input_psnr_db"])
        check(row["strength_class"] == label, "Calibration label mismatch")
    for metric in METRICS:
        check(close(np.mean([float(r[f"candidate_{metric}"]) for r in records.values()]),
                    summary[f"candidate_{metric}"], 1e-8), "Summary mean mismatch")
    audit = audit_config(directory, seed, setting)
    files = ["summary.json", "best.pt", "last.pt", "metrics.csv", "per_image_metrics.csv"]
    if audit["config_available"]:
        files.append("config.json")
    audit["artifact_sha256"] = {name: sha256(directory / name) for name in files}
    if seed == 12011:
        snapshot = directory / "source_snapshot"
        hashes = {str(p.relative_to(snapshot)): sha256(p) for p in snapshot.rglob("*")
                  if p.suffix in (".py", ".ps1")}
        check(len(hashes) == 16, "Incomplete P20 source snapshot")
        for relative, digest in hashes.items():
            check(digest == sha256(directory.parents[2] / relative), f"Source changed: {relative}")
        protocol = list((directory.parents[2] / "docs").glob("17_P20*.md"))
        check(len(protocol) == 1 and sha256(protocol[0]) == sha256(snapshot / "protocol.md"),
              "Protocol changed since training")
        audit.update(source_snapshot_sha256=hashes, protocol_sha256=sha256(snapshot / "protocol.md"))
    else:
        audit["historical_source_limitation"] = (
            "Historical configuration/results audited; not a complete historical training replay.")
    return summary, records, best, last, audit


def evaluate_checkpoint(checkpoint, baseline_checkpoint, baseline, loader, calibration):
    setting = checkpoint["args"]
    candidate = training._make_candidate(baseline_checkpoint["model"],
        int(baseline_checkpoint["args"]["width"]), setting["variant"], (1, 3, 7), 6, 5)
    candidate.load_state_dict(checkpoint["model"], strict=True)
    candidate = candidate.cuda().eval()
    _, rows, _, _ = training.evaluate(baseline, candidate, loader,
        calibration["heavy_medium_threshold_input_psnr_db"],
        calibration["medium_light_threshold_input_psnr_db"], detailed=True)
    del candidate
    return index_rows(rows)


def make_report(result):
    lines = ["# P20 三种子及固定末点复核", "",
        "固定 B0 的三次微调；385 张开发图，256×256 中心裁剪。不是独立测试。", "",
        "| 种子 | 权重规则 | Local PSNR | C00 PSNR | C01 PSNR | C01−Local | C01−C00 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for seed, item in result["by_seed"].items():
        for endpoint in ("best", "last"):
            overall = item[endpoint]["overall"]
            score, delta = overall["scores"], overall["contrasts"]
            lines.append(f"| {seed} | {endpoint} | {score['local']['psnr_db']:.6f} | "
                f"{score['C00']['psnr_db']:.6f} | {score['C01']['psnr_db']:.6f} | "
                f"{delta['C01_minus_local']['psnr_db']:+.6f} | {delta['C01_minus_C00']['psnr_db']:+.6f} |")
    lines += ["", "## 三种子均值与样本标准差", "",
              "| 权重规则 | 对比 | PSNR 均值 ± SD/dB | SSIM 均值 ± SD |",
              "|---|---|---:|---:|"]
    for endpoint, contrasts in result["seed_statistics"].items():
        for label, metrics in contrasts.items():
            p, s = metrics["psnr_db"], metrics["ssim"]
            lines.append(f"| {endpoint} | {label} | {p['mean']:+.6f} ± {p['sample_sd_ddof1']:.6f} | "
                         f"{s['mean']:+.7f} ± {s['sample_sd_ddof1']:.7f} |")
    lines += ["", "样本标准差采用 ddof=1，衡量三次微调波动，不是置信区间或显著性证明。", "",
              "## 预登记判定", "", "| 种子 | 最佳点局部筛选线 | 监督边际符号 | 末点无反转 |",
              "|---|---|---|---|"]
    for seed, item in result["by_seed"].items():
        d = item["decision"]
        lines.append(f"| {seed} | {d['local_screening_passed']} | {d['teacher_best_sign_passed']} | "
                     f"{d['last_no_direction_reversal']} |")
    lines += ["", f"三种子全部预登记规则通过：{result['all_preregistered_rules_passed']}。",
              "", "## 完整分组差值", "",
              "下表 best/last、PSNR/SSIM 和全部分组均保留，负值没有筛除。", "",
              "| 种子 | 权重 | 分组 | 帧数 | C01−Local dB | ΔSSIM | C01−C00 dB | ΔSSIM |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for seed, item in result["by_seed"].items():
        for endpoint in ("best", "last"):
            for dimension in ("per_group", "per_strength"):
                for label, group in item[endpoint][dimension].items():
                    a, b = (group["contrasts"][k] for k in CONTRASTS)
                    lines.append(f"| {seed} | {endpoint} | {label} | {group['pairs']} | "
                                 f"{a['psnr_db']:+.6f} | {a['ssim']:+.7f} | "
                                 f"{b['psnr_db']:+.6f} | {b['ssim']:+.7f} |")
    lines += ["", "## 边界", "",
              "- 三个种子共享 B0，不是三个独立从头训练主干。",
              "- 同一开发集反复用于方法设计、检查点选择；重评末点不能消除开发集适配。",
              "- 同步末点只排查训练时刻选择依赖，不证明原创性、SOTA 或正式测试有效。",
              "- 早期 +0.15 dB 目标不可由 +0.02 dB 筛选线替代；不自动调参补救负结果。",
              "- 后续应预登记整图开发评价和效率测量，再补可比公开基线与独立测试。",
              "- 原训练文件只读；末点评价另存。审计细节、全部原值见 summary.json 和 CSV。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(torch.cuda.is_available(), "CUDA required; no CPU inference fallback")
    root, output = args.runs_root.resolve(), args.output.resolve()
    check(not output.exists(), "Refusing to overwrite analysis output")
    common = torch.load(root / RUNS[3407]["local"] / "best.pt",
                        map_location="cpu", weights_only=False)["args"]
    check(sha256(Path(common["baseline_checkpoint"])) == FROZEN["baseline"], "B0 changed")
    check(sha256(Path(common["calibration_summary"])) == FROZEN["calibration"], "Calibration changed")
    check(sha256(ROOT / "scripts/train_p12_matched_nonlocal.py") == FROZEN["trainer"], "Trainer changed")
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "src").rglob("*.py")}
    for relative, digest in source_hashes.items():
        check(digest == sha256(root.parents[1] / relative), "Evaluation source differs from project")
    baseline_checkpoint = torch.load(common["baseline_checkpoint"], map_location="cpu", weights_only=False)
    check(baseline_checkpoint["step"] == 22500 and
          baseline_checkpoint["args"]["validation_source"] == "train_holdout", "B0 protocol mismatch")
    calibration = json.loads(Path(common["calibration_summary"]).read_text(encoding="utf-8"))
    check(calibration["official_test_accessed"] is False, "Calibration test access")
    data = training.GoProPairDataset(common["data_root"], split="train", patch_size=256, augment=False,
                                    include_recording_groups=baseline_checkpoint["args"]["validation_groups"])
    check(len(data) == 385, "Unexpected development size")
    baseline = training.CompactDeblurUNet(width=int(baseline_checkpoint["args"]["width"])).cuda()
    baseline.load_state_dict(baseline_checkpoint["model"], strict=True)
    output.mkdir(parents=True)
    result = {"experiment": "P20 three fine-tuning seeds", "official_test_accessed": False,
              "paper_result": False, "statistical_significance_established": False,
              "frozen_input_sha256": FROZEN, "evaluation_source_sha256": source_hashes,
              "analyzer_sha256": sha256(Path(__file__)), "by_seed": {}}
    canonical_rows, canonical_split = None, None
    for seed, mapping in RUNS.items():
        training.set_determinism(seed)
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True,
                            persistent_workers=True, generator=torch.Generator().manual_seed(seed + 2))
        records, terminals, audits, best_steps = {}, {}, {}, {}
        for role, run in mapping.items():
            print(f"Audit and replay seed={seed} role={role}", flush=True)
            directory = root / run
            summary, original, best, last, audit = audit_run(directory, seed, role, common, calibration)
            split = {k: summary[k] for k in ("train_recording_groups", "validation_recording_groups")}
            if canonical_rows is None:
                canonical_rows, canonical_split = original, split
            compare_rows(original, canonical_rows)
            check(split == canonical_split, "Cross-run split mismatch")
            replay = evaluate_checkpoint(best, baseline_checkpoint, baseline, loader, calibration)
            compare_rows(replay, original, predictions=True)
            reused = same_endpoint(best, last)
            terminal = replay if reused else evaluate_checkpoint(
                last, baseline_checkpoint, baseline, loader, calibration)
            compare_rows(terminal, original)
            history = read_csv(directory / "metrics.csv")
            check(close(np.mean([float(r["candidate_psnr_db"]) for r in terminal.values()]),
                        history[-1]["candidate_psnr_db"]), "Terminal history/replay mismatch")
            audit.update(best_replay_per_image_tolerance=1e-6, same_step_reused=reused,
                         training_wall_seconds=summary["wall_seconds"])
            for endpoint, rows in (("best", replay), ("last", terminal)):
                write_csv(output / f"seed{seed}_{role}_{endpoint}.csv", [rows[n] for n in sorted(rows)])
            records[role], terminals[role], audits[role] = original, terminal, audit
            best_steps[role] = best["step"]
        best_stats, last_stats = aggregate(records), aggregate(terminals)
        result["by_seed"][str(seed)] = {"run_names": mapping, "best_steps": best_steps,
            "audit": audits, "best": best_stats, "last": last_stats,
            "decision": decision(best_stats, last_stats)}
        del loader
    result["seed_statistics"] = {
        endpoint: {contrast: {metric: seed_statistics([
            result["by_seed"][str(seed)][endpoint]["overall"]["contrasts"][contrast][metric]
            for seed in RUNS]) for metric in METRICS} for contrast in CONTRASTS}
        for endpoint in ("best", "last")}
    result["all_preregistered_rules_passed"] = all(
        item["decision"]["all_preregistered_rules_passed"] for item in result["by_seed"].values())
    # Ensure evaluation did not mutate any historical input artifact.
    for item in result["by_seed"].values():
        for role, audit in item["audit"].items():
            for name, digest in audit["artifact_sha256"].items():
                check(sha256(root / item["run_names"][role] / name) == digest, "Run artifact changed")
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                   allow_nan=False), encoding="utf-8")
    (output / "report.md").write_text(make_report(result), encoding="utf-8")
    print(json.dumps({"output": str(output),
                      "all_preregistered_rules_passed": result["all_preregistered_rules_passed"],
                      "seed_statistics": result["seed_statistics"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
