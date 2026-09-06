"""Replicate the outer-gate/teacher factorial across two fixed fine-tuning seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_p16_factorial import (
    CELLS, EFFECT_LABELS, RUNS, aggregate, factorial_effects, load_and_audit, sha256, write_csv,
)
import matplotlib.pyplot as plt


SEEDS = (3407, 7813)
RUNS_7813 = {
    "local": "P13A_local_control_seed7813_v0",
    "C00": "P13G_gate_adaptive_seed7813_v0",
    "C10": "P17C10_outer_gate_only_seed7813_v0",
    "C01": "P17C01_teacher_only_seed7813_v0",
    "C11": "P15_protected_nonlocal_seed7813_v0",
}
REQUIRED_SNAPSHOT_FILES = (
    "scripts/train_p12_matched_nonlocal.py", "scripts/run_p17_factorial_variant.ps1",
    "src/__init__.py", "src/data/__init__.py", "src/data/gopro_pairs.py",
    "src/models/__init__.py", "src/models/compact_unet.py", "src/models/multiscale_nonlocal.py",
    "src/models/blur_strength.py", "src/losses.py",
)


def screening_gates(result, cell):
    gates = {
        "overall_psnr": result["overall"]["minus_local"][cell]["psnr_db"] >= .02,
        "overall_ssim": result["overall"]["minus_local"][cell]["ssim"] >= 0,
        "groups": all(group["minus_local"][cell]["psnr_db"] >= -.01
                      for group in result["per_group"].values()),
        "strengths": all(group["minus_local"][cell]["psnr_db"] >= -.01
                         for group in result["per_strength"].values()),
    }
    return {**gates, "passed": all(gates.values())}


def build_seed_result(root, seed, runs):
    summaries, records, audit = load_and_audit(root, seed=seed, runs=runs)
    names = sorted(records["local"])
    result = {"seed": seed, "run_names": runs, "runs": summaries, "audit": audit,
              "overall": aggregate(names, records)}
    for key, field in (("per_group", "recording_group"), ("per_strength", "strength_class")):
        labels = sorted({records["local"][name][field] for name in names})
        if key == "per_strength":
            labels = ["heavy", "medium", "light"]
        result[key] = {label: aggregate([name for name in names
                                        if records["local"][name][field] == label], records)
                       for label in labels}
    result["original_screening_gates"] = {cell: screening_gates(result, cell) for cell in ("C01", "C11")}
    return result, records


def audit_between_seeds(first, second, first_records, second_records):
    for key in ("baseline_sha256", "calibration_sha256"):
        if first["audit"][key] != second["audit"][key]:
            raise ValueError(f"Cross-seed mismatch: {key}")
    for key, value in first["audit"]["training_settings"].items():
        if key != "seed" and value != second["audit"]["training_settings"][key]:
            raise ValueError(f"Cross-seed training setting differs: {key}")
    for key in ("data_root", "train_recording_groups", "validation_recording_groups", "validation_patch_size"):
        if first["runs"]["local"][key] != second["runs"]["local"][key]:
            raise ValueError(f"Cross-seed data protocol differs: {key}")
    a, b = first_records["local"], second_records["local"]
    if set(a) != set(b):
        raise ValueError("Cross-seed image names differ")
    for name, reference in a.items():
        for key in ("recording_group", "scene", "strength_class"):
            if reference[key] != b[name][key]:
                raise ValueError(f"Cross-seed metadata differs: {name}/{key}")
        for key in ("input_psnr_db", "baseline_psnr_db", "baseline_ssim"):
            values = (float(reference[key]), float(b[name][key]))
            if not all(np.isfinite(value) for value in values) or abs(values[0] - values[1]) > 1e-6:
                raise ValueError(f"Cross-seed input/B0 differs: {name}/{key}")
    return {"passed": True, "same_trained_b0": True, "same_development_images": True,
            "independent_full_backbone_retrainings": False}


def audit_new_source_snapshots(root):
    snapshots = {}
    for cell in ("C10", "C01"):
        folder = root / RUNS_7813[cell] / "source_snapshot"
        files = sorted([*folder.rglob("*.py"), *folder.rglob("*.ps1")])
        if any(not (folder / name).is_file() for name in REQUIRED_SNAPSHOT_FILES):
            raise ValueError(f"Missing new-run source snapshot: {cell}")
        snapshots[cell] = {path.relative_to(folder).as_posix(): sha256(path) for path in files}
    if snapshots["C10"] != snapshots["C01"]:
        raise ValueError("New-run source snapshots differ between C10 and C01")
    return {"passed": True, "source_sha256": snapshots,
            "scope": "Only the two new P17 runs have these pre-training snapshots."}


def mean_bucket(buckets):
    result = {"pairs_per_seed": buckets[0]["pairs"], "seed_count": len(buckets)}
    for category in ("metrics", "minus_local", "effects"):
        result[category] = {
            item: {metric: float(np.mean([bucket[category][item][metric] for bucket in buckets]))
                   for metric in values}
            for item, values in buckets[0][category].items()
        }
    return result


def summarize_effect_stability(results):
    return {
        effect: {
            "per_seed_psnr_db": {str(seed): results[seed]["overall"]["effects"]["psnr_db"][effect]
                                 for seed in SEEDS},
            "positive_seed_count": sum(results[seed]["overall"]["effects"]["psnr_db"][effect] > 0
                                       for seed in SEEDS),
            "mean_psnr_db": float(np.mean([results[seed]["overall"]["effects"]["psnr_db"][effect]
                                           for seed in SEEDS])),
            "sample_sd_psnr_db": float(np.std([results[seed]["overall"]["effects"]["psnr_db"][effect]
                                               for seed in SEEDS], ddof=1)),
        }
        for effect in EFFECT_LABELS
    }


def write_report(result, output):
    results, mean = result["by_seed"], result["two_seed_mean"]
    lines = ["# P17：外层门与监督的双种子消融复核", "",
             "日期：2026-09-05。种子 3407 与 7813，共享同一训练好的 B0 和 385 张开发裁剪。",
             "这是微调随机性的复核，不是两个独立主干重训，不是独立测试。", "",
             "## 各种子结果", "",
             "| 种子 | 组合 | PSNR/dB | SSIM | 相对局部/dB | 最佳步数 | C01/C11 原筛选线 |",
             "|---:|---|---:|---:|---:|---:|---|"]
    for seed in SEEDS:
        current = results[seed]
        for cell in RUNS:
            metric = current["overall"]["metrics"][cell]
            delta = 0 if cell == "local" else current["overall"]["minus_local"][cell]["psnr_db"]
            gate = current["original_screening_gates"].get(cell)
            status = "—" if gate is None else ("通过" if gate["passed"] else "未通过")
            lines.append(f"| {seed} | {cell} | {metric['psnr_db']:.5f} | {metric['ssim']:.7f} | "
                         f"{delta:+.5f} | {current['runs'][cell]['best_checkpoint_step']} | {status} |")
    lines += ["", "C00=都不新增；C10=仅外层门；C01=仅监督；C11=两者都有。各组仍保留内部融合门。",
              "原筛选线：相对同种子局部总体 ≥ +0.02 dB、总体 SSIM 不降，各录像组/代理层次 ≥ −0.01 dB。", "",
              "## 因素作用与种子波动", "",
              "| 效应 | 3407/dB | 7813/dB | 两种子均值/dB | 样本标准差/dB |",
              "|---|---:|---:|---:|---:|"]
    for effect, label in EFFECT_LABELS.items():
        values = result["effect_stability"][effect]
        lines.append(f"| {label} | {values['per_seed_psnr_db']['3407']:+.5f} | "
                     f"{values['per_seed_psnr_db']['7813']:+.5f} | {values['mean_psnr_db']:+.5f} | "
                     f"{values['sample_sd_psnr_db']:.5f} |")
    lines += ["", "标准差仅描述两个种子的变化；两个样本不足以证明显著性或非劣性。",
              "两种子平均是分数的平均，不是两个模型输出的集成。完整 SSIM 条件效应见 effects.csv。", ""]
    for key, label in (("per_group", "录像组"), ("per_strength", "输入 PSNR 退化代理层次")):
        lines += [f"## {label}：各自相对局部", "",
                  "| 种子 | 分组 | 帧数 | C00/dB | C10/dB | C01/dB | C11/dB |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for seed in SEEDS:
            for name, bucket in results[seed][key].items():
                values = " | ".join(f"{bucket['minus_local'][cell]['psnr_db']:+.5f}" for cell in CELLS)
                lines.append(f"| {seed} | {name} | {bucket['pairs']} | {values} |")
        for name, bucket in mean[key].items():
            values = " | ".join(f"{bucket['minus_local'][cell]['psnr_db']:+.5f}" for cell in CELLS)
            lines.append(f"| 两种子均值 | {name} | {bucket['pairs_per_seed']} | {values} |")
        lines += ["", "各分组的条件效应与 SSIM 均保存在 CSV/JSON 中，没有删除负结果。", ""]
    teacher_keys = ("teacher_without_outer", "teacher_with_outer")
    outer_keys = ("outer_without_teacher", "outer_with_teacher")
    teacher_positive = all(result["effect_stability"][key]["positive_seed_count"] == 2 for key in teacher_keys)
    outer_positive = all(result["effect_stability"][key]["positive_seed_count"] == 2 for key in outer_keys)
    lines += ["## 证据支持到哪里", "",
              f"监督在两个条件、两个种子中平均 PSNR 均为正：{'是' if teacher_positive else '否'}。",
              f"外层门在两个条件、两个种子中平均 PSNR 均为正：{'是' if outer_positive else '否'}。", "",
              "这些是符号一致性描述，不是显著性阈值，也不自动决定最终架构。",
              "若外层门的符号反转，应报告其种子依赖性；不能只保留有利种子或继续换种子追求预期。", "",
              "- 主效应平均了两个不同条件；应同时看条件效应、SSIM 和分组权衡。",
              "- 教师只在训练使用配对清晰图，推理仅输入模糊图。代理强度不是物理模糊真值。",
              "- 仍为 256×256 中心裁剪、多次参与开发和检查点选择的数据；未访问官方测试或 TMT。",
              "- 录像相邻帧相关，不能将 385 帧或两个种子重复的 770 条记录当作独立样本。",
              "- C01/C11 参数量相同，删除一个外层乘法不代表显著降低模型参数或获得加速。",
              "- 新训练保存源码快照；旧对照的全部历史源码仍无法重建。",
              "- 后续仍需确定方法主张、第三种子、完整分辨率和独立外部评价。", "",
              "![P17 replication](replication.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_figures(result, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), layout="constrained")
    x = np.arange(len(CELLS))
    labels = ["Outer | no teacher", "Outer | teacher", "Teacher | no outer", "Teacher | outer"]
    keys = list(EFFECT_LABELS)[:4]
    for index, seed in enumerate(SEEDS):
        current = result["by_seed"][seed]["overall"]
        position = (index - .5) * .34
        axes[0].bar(x + position, [current["minus_local"][cell]["psnr_db"] for cell in CELLS],
                    width=.32, label=str(seed), color=("#458b98", "#db9a4b")[index])
        axes[1].scatter([current["effects"]["psnr_db"][key] for key in keys],
                        np.arange(4) + position, label=str(seed), s=50,
                        marker=("o", "s")[index], color=("#458b98", "#db9a4b")[index])
    axes[0].set(xticks=x, xticklabels=CELLS, ylabel="PSNR gain over same-seed local (dB)",
                title="Matched fine-tuning controls")
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].legend(title="Seed")
    axes[1].set(yticks=np.arange(4), yticklabels=labels, xlabel="Conditional effect (dB)",
                title="Each seed shown separately")
    axes[1].axvline(0, color="black", linewidth=.7)
    axes[1].invert_yaxis()
    fig.suptitle("P17 | two fine-tuning seeds | same B0 + 385 development crops | not independent testing")
    fig.savefig(output / "replication.png", dpi=180)
    fig.savefig(output / "replication.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.runs_root.resolve()
    results, records = {}, {}
    for seed, runs in ((3407, RUNS), (7813, RUNS_7813)):
        results[seed], records[seed] = build_seed_result(root, seed, runs)
    cross_audit = audit_between_seeds(results[3407], results[7813], records[3407], records[7813])
    snapshot_audit = audit_new_source_snapshots(root)
    means = {"overall": mean_bucket([results[seed]["overall"] for seed in SEEDS])}
    for key in ("per_group", "per_strength"):
        means[key] = {label: mean_bucket([results[seed][key][label] for seed in SEEDS])
                      for label in results[3407][key]}
    result = {"experiment": "P17 two-seed independent-factor training replication", "seeds": SEEDS,
              "by_seed": results, "two_seed_mean": means, "cross_seed_audit": cross_audit,
              "source_snapshot_audit": snapshot_audit, "effect_stability": summarize_effect_stability(results),
              "official_test_accessed": False, "paper_result": False}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    scores, effects, groups, images = [], [], [], []
    for seed in SEEDS:
        current = results[seed]
        for cell in RUNS:
            scores.append({"seed": seed, "cell": cell, **current["overall"]["metrics"][cell],
                "minus_local_psnr_db": 0 if cell == "local" else current["overall"]["minus_local"][cell]["psnr_db"],
                "minus_local_ssim": 0 if cell == "local" else current["overall"]["minus_local"][cell]["ssim"],
                "best_step": current["runs"][cell]["best_checkpoint_step"],
                "wall_seconds": current["runs"][cell]["wall_seconds"],
                "peak_vram_mib": current["runs"][cell]["peak_vram_mib"]})
        for effect in EFFECT_LABELS:
            effects.append({"seed": seed, "effect": effect, **{
                metric: current["overall"]["effects"][metric][effect] for metric in ("psnr_db", "ssim")}})
        for dimension in ("per_group", "per_strength"):
            for label, bucket in current[dimension].items():
                row = {"seed": seed, "dimension": dimension, "group": label, "pairs": bucket["pairs"]}
                for metric in ("psnr_db", "ssim"):
                    row.update({f"{cell}_minus_local_{metric}": bucket["minus_local"][cell][metric] for cell in CELLS})
                    row.update({f"{effect}_{metric}": value for effect, value in bucket["effects"][metric].items()})
                groups.append(row)
        for name in sorted(records[seed]["local"]):
            row = {"seed": seed, "name": name, **{key: records[seed]["local"][name][key]
                   for key in ("recording_group", "strength_class")}}
            for metric in ("psnr_db", "ssim"):
                values = {cell: float(records[seed][cell][name][f"candidate_{metric}"]) for cell in RUNS}
                row.update({f"{cell}_{metric}": value for cell, value in values.items()})
                row.update({f"{effect}_{metric}": float(value) for effect, value in factorial_effects(values).items()})
            images.append(row)
    for effect in EFFECT_LABELS:
        effects.append({"seed": "two_seed_mean", "effect": effect, **{
            metric: means["overall"]["effects"][metric][effect] for metric in ("psnr_db", "ssim")}})
    for filename, rows in (("scores.csv", scores), ("effects.csv", effects),
                            ("group_effects.csv", groups), ("per_image_two_seed.csv", images)):
        write_csv(output / filename, rows)
    write_report(result, output)
    write_figures(result, output)
    print(json.dumps({"audit_passed": cross_audit["passed"], "means": means["overall"],
                      "effects": result["effect_stability"], "gates": {
                          seed: results[seed]["original_screening_gates"] for seed in SEEDS}}, indent=2))


if __name__ == "__main__":
    main()
