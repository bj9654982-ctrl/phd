"""Audited, paired 2x2 development-set ablation; no image inference or test access."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


RUNS = {
    "local": "P12A_local_control_v0",
    "C00": "P12G_uniform_scale_adaptive_gate_v0",
    "C10": "P16C10_outer_gate_only_seed3407_v0",
    "C01": "P16C01_teacher_only_seed3407_v0",
    "C11": "P15_protected_nonlocal_seed3407_v0",
}
CELLS = ("C00", "C10", "C01", "C11")
FACTORS = {"local": (False, 0.0), "C00": (False, 0.0),
           "C10": (True, 0.0), "C01": (False, 0.01), "C11": (True, 0.01)}
EFFECT_LABELS = {
    "outer_without_teacher": "外层门：无监督时 C10-C00",
    "outer_with_teacher": "外层门：有监督时 C11-C01",
    "teacher_without_outer": "监督：无外层门时 C01-C00",
    "teacher_with_outer": "监督：有外层门时 C11-C10",
    "outer_main": "外层门主效应（两种监督条件平均）",
    "teacher_main": "监督主效应（两种外层门条件平均）",
    "interaction": "交互效应 C11-C10-C01+C00",
}


def factorial_effects(values):
    c00, c10, c01, c11 = (np.asarray(values[cell]) for cell in CELLS)
    return {
        "outer_without_teacher": c10 - c00,
        "outer_with_teacher": c11 - c01,
        "teacher_without_outer": c01 - c00,
        "teacher_with_outer": c11 - c10,
        "outer_main": 0.5 * ((c10 - c00) + (c11 - c01)),
        "teacher_main": 0.5 * ((c01 - c00) + (c11 - c10)),
        "interaction": c11 - c10 - c01 + c00,
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_and_audit(root, *, seed=3407, runs=None):
    runs = RUNS if runs is None else runs
    if seed not in (3407, 7813):
        raise ValueError("Unsupported factorial replication seed")
    if set(runs) != set(RUNS) or len(set(runs.values())) != len(RUNS):
        raise ValueError("Expected one distinct run for each factorial cell and local control")
    summaries, records, settings, provenance = {}, {}, {}, {}
    expected = {"seed": seed, "steps": 3000, "batch_size": 6,
                "train_patch_size": 256, "learning_rate": 0.0002,
                "frequency_weight": 0.1, "val_interval": 375, "num_workers": 2,
                "train_pairs": 0, "val_pairs": 0}
    for cell, run in runs.items():
        directory = root / run
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        rows = read_csv(directory / "per_image_metrics.csv")
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        setting = dict(checkpoint["args"])
        outer, weight = FACTORS[cell]
        variant = "local" if cell == "local" else (
            "protected_gate_adaptive" if outer else "gate_adaptive")
        for field, value in {**expected, "variant": variant}.items():
            if setting[field] != value:
                raise ValueError(f"{cell}: unexpected {field}={setting[field]}")
        if setting.get("sampling", "image_uniform") != "image_uniform":
            raise ValueError(f"{cell}: sampling differs")
        if setting.get("gate_supervision_weight", 0.0) != weight:
            raise ValueError(f"{cell}: supervision differs")
        if summary.get("gate_supervision_weight", 0.0) != weight:
            raise ValueError(f"{cell}: summary supervision differs")
        for field in ("seed", "steps", "variant"):
            if summary[field] != setting[field]:
                raise ValueError(f"{cell}: summary/checkpoint {field} differs")
        if cell != "local":
            for field, value in (("scales", (1, 3, 7)), ("top_k", 6), ("statistics_window", 5)):
                actual = tuple(setting[field]) if field == "scales" else setting[field]
                if actual != value:
                    raise ValueError(f"{cell}: nonlocal setting {field} differs")
            if summary["candidate_parameters"] != 1095511:
                raise ValueError(f"{cell}: parameter count differs")
        if (summary.get("official_test_accessed") is not False
                or summary["pairs"] != 385 or summary["val_pairs"] != 385
                or summary["train_pairs"] != 1718 or summary["recording_group_overlap"]
                or summary["exact_initialization_max_abs_error"] != 0
                or summary["validation_patch_size"] != 256):
            raise ValueError(f"{cell}: split/initialization audit failed")
        if summary["best_checkpoint_step"] != checkpoint["step"]:
            raise ValueError(f"{cell}: checkpoint step differs")
        history = read_csv(directory / "metrics.csv")
        if [int(row["step"]) for row in history] != [1, *range(375, 3001, 375)]:
            raise ValueError(f"{cell}: validation schedule differs")
        best = max(history, key=lambda row: float(row["candidate_psnr_db"]))
        if int(best["step"]) != checkpoint["step"]:
            raise ValueError(f"{cell}: checkpoint selection differs")
        if abs(float(best["candidate_psnr_db"]) - summary["candidate_psnr_db"]) > 1e-6:
            raise ValueError(f"{cell}: final evaluation differs from best history")
        by_name = {row["name"]: row for row in rows}
        if len(by_name) != len(rows) or len(rows) != 385:
            raise ValueError(f"{cell}: duplicate/missing image")
        for field in ("candidate_psnr_db", "candidate_ssim"):
            array = np.array([float(row[field]) for row in rows])
            if not np.isfinite(array).all() or abs(array.mean() - summary[field]) > 1e-8:
                raise ValueError(f"{cell}: nonfinite/incorrect mean {field}")
        summaries[cell], records[cell], settings[cell] = summary, by_name, setting
        provenance[cell] = {file: sha256(directory / file) for file in
                            ("best.pt", "metrics.csv", "summary.json", "per_image_metrics.csv")}
    for cell in CELLS:
        for field in ("baseline_checkpoint", "data_root", "calibration_summary"):
            if Path(settings[cell][field]).resolve() != Path(settings["local"][field]).resolve():
                raise ValueError(f"{cell}: {field} differs")
        for field in ("train_recording_groups", "validation_recording_groups"):
            if summaries[cell][field] != summaries["local"][field]:
                raise ValueError(f"{cell}: {field} differs")
        if set(records[cell]) != set(records["local"]):
            raise ValueError(f"{cell}: evaluation image names differ")
        for name, reference in records["local"].items():
            row = records[cell][name]
            for field in ("recording_group", "scene", "strength_class"):
                if row[field] != reference[field]:
                    raise ValueError(f"{cell}: {name}/{field} differs")
            for field in ("input_psnr_db", "baseline_psnr_db", "baseline_ssim"):
                values = (float(row[field]), float(reference[field]))
                if not all(np.isfinite(value) for value in values) or abs(values[0] - values[1]) > 1e-6:
                    raise ValueError(f"{cell}: {name}/{field} differs")
    calibration_path = Path(settings["local"]["calibration_summary"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("official_test_accessed") is not False:
        raise ValueError("Calibration must not access the official test set")
    group_counts = Counter(row["recording_group"] for row in records["local"].values())
    if group_counts != {"GOPR0372_07": 175, "GOPR0378_13": 110, "GOPR0385_11": 100}:
        raise ValueError("Unexpected development recording groups/counts")
    for row in records["local"].values():
        psnr = float(row["input_psnr_db"])
        expected_class = ("heavy" if psnr <= calibration["heavy_medium_threshold_input_psnr_db"]
                          else "light" if psnr >= calibration["medium_light_threshold_input_psnr_db"]
                          else "medium")
        if row["strength_class"] != expected_class:
            raise ValueError("Strength label differs from frozen calibration")
    return summaries, records, {
        "passed": True, "training_settings": expected, "run_artifact_sha256": provenance,
        "baseline_sha256": sha256(Path(settings["local"]["baseline_checkpoint"])),
        "calibration_sha256": sha256(calibration_path),
        "historical_source_hashes_available": False,
        "note": "Recorded arguments and evaluation rows audited; historical source versions cannot be fully reconstructed.",
    }


def aggregate(names, records):
    values = {metric: {cell: np.array([float(records[cell][name][f"candidate_{metric}"])
                                      for name in names]) for cell in RUNS}
              for metric in ("psnr_db", "ssim")}
    return {
        "pairs": len(names),
        "metrics": {cell: {metric: float(values[metric][cell].mean()) for metric in values}
                    for cell in RUNS},
        "minus_local": {cell: {metric: float((values[metric][cell] - values[metric]["local"]).mean())
                               for metric in values} for cell in CELLS},
        "effects": {metric: {effect: float(array.mean()) for effect, array in
                             factorial_effects(values[metric]).items()} for metric in values},
        "positive_frames_vs_local": {cell: int((values["psnr_db"][cell] >
                                               values["psnr_db"]["local"]).sum()) for cell in CELLS},
    }


def make_report(result, output):
    overall, summaries = result["overall"], result["runs"]
    lines = ["# P16 外层门 × 强度监督：独立训练消融", "",
             "2026-09-05；种子 3407；固定开发集 385 帧、256×256 中心裁剪。官方测试集未使用。", "",
             "## 四组结果", "",
             "| 组合 | 外层门 | 监督权重 | PSNR/dB | SSIM | 相对局部/dB | 最佳步数 |",
             "|---|---|---|---:|---:|---:|---:|"]
    for cell in RUNS:
        metric = overall["metrics"][cell]
        delta = 0 if cell == "local" else overall["minus_local"][cell]["psnr_db"]
        outer, weight = FACTORS[cell]
        lines.append(f"| {cell} | {'有' if outer else '无'} | {weight} | {metric['psnr_db']:.5f} | "
                     f"{metric['ssim']:.7f} | {delta:+.5f} | {summaries[cell]['best_checkpoint_step']} |")
    lines += ["", "C00/C11 复用已有训练；C10/C01 本次分别从同一 B0 新训练 3000 步。", "",
              "## 因素作用", "", "| 对比 | PSNR 差/dB | SSIM 差 |", "|---|---:|---:|"]
    for effect, label in EFFECT_LABELS.items():
        lines.append(f"| {label} | {overall['effects']['psnr_db'][effect]:+.5f} | "
                     f"{overall['effects']['ssim'][effect]:+.7f} |")
    lines += ["", "主效应是两种条件下效应的平均，不表示该因素在每种条件下都有效。",
              "交互效应衡量组合收益是否偏离两个单因素收益之和；其为正也不等于证明某种物理机制。", ""]
    effects = overall["effects"]["psnr_db"]
    if (effects["teacher_without_outer"] > 0 and effects["teacher_with_outer"] > 0
            and effects["outer_without_teacher"] <= 0 and effects["outer_with_teacher"] <= 0):
        lines += ["本种子中，监督在两种外层门条件下均提高平均 PSNR，而外层门在两种监督条件下均未提高平均 PSNR。",
                  "因此当前证据支持监督带来的平均收益，不支持将外层门列为已经证实的独立有效部件。",
                  "这不是严格显著性结论，不能由单种子的小差值宣称外层门普遍有害。", ""]
    for dimension, title in (("per_group", "录像组"), ("per_strength", "输入 PSNR 退化代理层次")):
        lines += [f"## {title}：相对同种子局部对照", "",
                  "| 分组 | 帧数 | C00/dB | C10/dB | C01/dB | C11/dB |", "|---|---:|---:|---:|---:|---:|"]
        for name, group in result[dimension].items():
            columns = " | ".join(f"{group['minus_local'][cell]['psnr_db']:+.5f}" for cell in CELLS)
            lines.append(f"| {name} | {group['pairs']} | {columns} |")
        lines += ["", "分组条件效应及 SSIM 详见 summary.json 和对应 CSV，未只保留有利分组。", ""]
    lines += ["## 筛选线与解读边界", "",
              "沿用 P15 组合对局部的开发筛选线：总体 ≥ +0.02 dB、总体 SSIM 不降，"
              "各录像组与各退化代理层次 ≥ −0.01 dB。没有为因素贡献临时新增通过线。", "",
              f"C11 保留原筛选线：{'通过' if result['c11_original_gates']['passed'] else '未通过'}。", "",
              "- 四组来自同一训练好的 B0；P16 仅一个微调种子，不能声称稳定的独立因果贡献。",
              "- 同一开发集反复用于设计与最佳检查点选择，指标是开发证据，不是独立测试结论。",
              "- 按预定的总体开发 PSNR 选各自最佳检查点，所以比较的是相同训练/选择预算，非同一训练时刻。",
              "- 相邻帧相关，385 帧不能当作 385 个独立实验；未据此宣称统计显著性。",
              "- 输入 PSNR 教师受图像内容影响，是退化代理，不是物理模糊核/强度真值。",
              "- 教师只训练时使用清晰配对图；推理仅输入模糊图。关闭外层门不等于恢复原 B0。",
              "- 固定多尺度候选中的 Top-K 不等于全图非局部检索，也尚未证明实际计算加速。",
              "- 参数与逐帧结果已对齐审计；早期源代码未全部快照，不能声称历史源码完全一致。", "",
              "下一步应先据四组结果收缩或保留方法贡献，再预先固定复核方案；"
              "不要看到结果后直接在同一开发集反复调参。", "",
              "![P16 factorial](factorial.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def make_figure(result, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    gains = [result["overall"]["minus_local"][cell]["psnr_db"] for cell in CELLS]
    bars = axes[0].bar(CELLS, gains, color=["#8896a5", "#508bbd", "#c59c56", "#378779"])
    axes[0].bar_label(bars, fmt="%+.4f", padding=3)
    axes[0].margins(y=.18)
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].set(ylabel="PSNR gain over matched local (dB)", title="2 x 2 independent training")
    effects = result["overall"]["effects"]["psnr_db"]
    keys = list(EFFECT_LABELS)[:4]
    labels = ["Outer | no teacher", "Outer | teacher", "Teacher | no outer", "Teacher | outer"]
    points = [effects[key] for key in keys]
    axes[1].barh(labels, points, color=["#378779" if x >= 0 else "#bf625b" for x in points])
    axes[1].axvline(0, color="black", linewidth=.7)
    axes[1].set(xlabel="Conditional PSNR effect (dB)", title=f"Interaction = {effects['interaction']:+.4f} dB")
    axes[1].invert_yaxis()
    fig.suptitle("P16 | seed 3407 | 385 development crops | not an independent test", fontsize=12)
    fig.savefig(output / "factorial.png", dpi=180)
    fig.savefig(output / "factorial.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries, records, audit = load_and_audit(args.runs_root.resolve())
    names = sorted(records["local"])
    result = {"experiment": "P16 independent factorial ablation", "seed": 3407,
              "run_names": RUNS, "runs": summaries, "audit": audit,
              "official_test_accessed": False, "paper_result": False,
              "overall": aggregate(names, records)}
    for key, field in (("per_group", "recording_group"), ("per_strength", "strength_class")):
        labels = sorted({records["local"][name][field] for name in names})
        if key == "per_strength":
            labels = ["heavy", "medium", "light"]
        result[key] = {label: aggregate([name for name in names if records["local"][name][field] == label], records)
                       for label in labels}
    overall = result["overall"]
    gates = {"overall_psnr": overall["minus_local"]["C11"]["psnr_db"] >= .02,
             "overall_ssim": overall["minus_local"]["C11"]["ssim"] >= 0,
             "groups": all(group["minus_local"]["C11"]["psnr_db"] >= -.01 for group in result["per_group"].values()),
             "strengths": all(group["minus_local"]["C11"]["psnr_db"] >= -.01 for group in result["per_strength"].values())}
    result["c11_original_gates"] = {**gates, "passed": all(gates.values())}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    per_image = []
    for name in names:
        row = {field: records["local"][name][field] for field in ("name", "recording_group", "strength_class")}
        for metric in ("psnr_db", "ssim"):
            values = {cell: float(records[cell][name][f"candidate_{metric}"]) for cell in RUNS}
            row.update({f"{cell}_{metric}": value for cell, value in values.items()})
            row.update({f"{effect}_{metric}": float(value) for effect, value in factorial_effects(values).items()})
        per_image.append(row)
    write_csv(output / "per_image_factorial.csv", per_image)
    for dimension in ("per_group", "per_strength"):
        rows = []
        for name, group in result[dimension].items():
            row = {"group": name, "pairs": group["pairs"]}
            for metric in ("psnr_db", "ssim"):
                row.update({f"{cell}_minus_local_{metric}": group["minus_local"][cell][metric] for cell in CELLS})
                row.update({f"{effect}_{metric}": value for effect, value in group["effects"][metric].items()})
            rows.append(row)
        write_csv(output / f"{dimension}_factorial.csv", rows)
    make_report(result, output)
    make_figure(result, output)
    print(json.dumps({"audit_passed": audit["passed"], "overall": overall,
                      "c11_original_gates": result["c11_original_gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
