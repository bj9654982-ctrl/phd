"""Save an auditable two-seed P15 report and a standalone scientific figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_analysis(directory: Path) -> tuple[dict, dict]:
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "per_image_comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {row["name"]: row for row in rows}
    if len(lookup) != 385 or len(rows) != 385:
        raise ValueError("Expected 385 unique development images")
    if report.get("official_test_accessed") is not False:
        raise ValueError("Unexpected official-test state")
    return report, lookup


def make_figure(reports: list[dict], directory: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    colors = ("#2878B5", "#E58638")
    categories = (
        ("per_group", ("GOPR0372_07", "GOPR0378_13", "GOPR0385_11"), ("0372_07", "0378_13", "0385_11")),
        ("per_strength", ("heavy", "medium", "light"), ("Heavy", "Medium", "Light")),
    )
    for axis, (section, keys, labels) in zip(axes[:2], categories):
        for index, (seed, report) in enumerate(zip((3407, 7813), reports)):
            values = [report[section][key]["mean_candidate_minus_local_psnr_db"] for key in keys]
            positions = np.arange(3) + (index - 0.5) * 0.35
            bars = axis.bar(positions, values, width=0.34, label=f"Seed {seed}", color=colors[index])
            for bar, value in zip(bars, values):
                axis.annotate(
                    f"{value:+.3f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, -18 if value < 0 else 3), textcoords="offset points",
                    ha="center", va="top" if value < 0 else "bottom", fontsize=8,
                )
        axis.axhline(0, color="#444444", linewidth=0.7)
        axis.axhline(-0.01, color="#BB3333", linestyle="--", linewidth=1, label="-0.01 dB floor")
        axis.set_xticks(np.arange(3), labels)
        axis.set_ylabel("P15 minus matched local (dB)")
        axis.margins(y=0.25)
        axis.grid(axis="y", alpha=0.15)
    axes[0].set_title("Recording-group mean gain")
    axes[1].set_title("Degradation-stratum mean gain")
    axes[0].legend(fontsize=8)
    for seed, report, color in zip((3407, 7813), reports, colors):
        values = [report["per_strength"][key]["mean_intervention_gate"] for key in ("heavy", "medium", "light")]
        axes[2].plot(range(3), values, "o-", label=f"Seed {seed}", color=color)
    axes[2].set_xticks(range(3), ("Heavy", "Medium", "Light"))
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Mean learned gate (0 to 1)")
    axes[2].set_title("Gate response by degradation")
    axes[2].grid(alpha=0.15)
    axes[2].legend(fontsize=8)
    fig.suptitle("P15 replication | 385 development crops | shared B0 | not official-test results")
    fig.savefig(directory / "replication.png", dpi=180)
    fig.savefig(directory / "replication.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed3407-analysis", required=True, type=Path)
    parser.add_argument("--seed7813-analysis", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    loaded = [load_analysis(path) for path in (args.seed3407_analysis, args.seed7813_analysis)]
    reports = [item[0] for item in loaded]
    lookups = [item[1] for item in loaded]
    names = sorted(lookups[0])
    if set(names) != set(lookups[1]):
        raise ValueError("Cross-seed image sets disagree")
    for seed, report in zip((3407, 7813), reports):
        if report.get("seed") != seed or not report.get("protocol_audit", {}).get("passed"):
            raise ValueError("Seed-specific matched protocol has not passed audit")
    for name in names:
        for field in ("recording_group", "scene", "strength_class", "input_psnr_db"):
            if lookups[0][name][field] != lookups[1][name][field]:
                raise ValueError(f"Cross-seed evaluation mismatch: {name}/{field}")
    deltas = np.asarray([
        [float(lookup[name]["candidate_minus_local_psnr_db"]) for name in names]
        for lookup in lookups
    ])
    mean_deltas = deltas.mean(axis=0)
    gates = {}
    for seed, report in zip((3407, 7813), reports):
        overall = report["overall"]
        flags = {
            "overall_psnr": overall["mean_candidate_minus_local_psnr_db"] >= 0.02,
            "overall_ssim": overall["mean_candidate_minus_local_ssim"] >= 0,
            "recording_groups": min(r["mean_candidate_minus_local_psnr_db"] for r in report["per_group"].values()) >= -0.01,
            "strength_strata": min(r["mean_candidate_minus_local_psnr_db"] for r in report["per_strength"].values()) >= -0.01,
        }
        flags["passed"] = all(flags.values())
        if flags["passed"] != report["predeclared_gates"]["passed"]:
            raise ValueError("Recomputed gate disagrees with input analysis")
        gates[str(seed)] = flags
    passed = all(item["passed"] for item in gates.values())
    cross_seed = {
        "two_seed_mean_gain_db": float(mean_deltas.mean()),
        "per_image_gain_pearson": float(np.corrcoef(deltas)[0, 1]),
        "positive_in_both_seeds": int(np.all(deltas > 0, axis=0).sum()),
        "negative_in_both_seeds": int(np.all(deltas < 0, axis=0).sum()),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image_two_seed.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("name", "recording_group", "strength_class", "seed3407_gain_db", "seed7813_gain_db", "mean_gain_db"))
        for index, name in enumerate(names):
            row = lookups[0][name]
            writer.writerow((name, row["recording_group"], row["strength_class"], deltas[0, index], deltas[1, index], mean_deltas[index]))
    summary = {
        "stage": "P15 two-seed development replication",
        "pairs": 385,
        "seeds": [3407, 7813],
        "per_seed": {str(seed): report for seed, report in zip((3407, 7813), reports)},
        "cross_seed": cross_seed,
        "predeclared_gates": gates,
        "replication_passed": passed,
        "official_test_accessed": False,
        "paper_result": False,
        "limitations": [
            "Both runs fine-tune one shared B0 rather than independently retraining B0.",
            "Only three held-out recording groups and 256x256 center crops were evaluated.",
            "This development set has been used repeatedly to design and select variants.",
            "Gate multiplication and teacher supervision changed together; their individual causal contributions remain unisolated.",
            "Group-average protection does not guarantee protection for every image or subgroup.",
            "Paired input PSNR is a degradation proxy affected by image content, not a physical blur-kernel measurement.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# P15 两种子复核结果", "", f"预先规定的复核判定：{'通过' if passed else '未通过'}。", "",
             "在同一 B0 上分别用 3407、7813 微调。每个种子与其同种子局部对照比较；1718 对训练、385 对开发验证，3000 步，batch 6，训练及验证裁剪 256×256。", "",
             "| 项目 | 种子 3407 | 种子 7813 |", "|---|---:|---:|"]
    for label, accessor, digits in (
        ("P15 PSNR / dB", lambda r: json.loads((args.runs_root / r["candidate_run"] / "summary.json").read_text(encoding="utf-8"))["candidate_psnr_db"], 5),
        ("相对局部 PSNR / dB", lambda r: r["overall"]["mean_candidate_minus_local_psnr_db"], 5),
        ("相对局部 SSIM", lambda r: r["overall"]["mean_candidate_minus_local_ssim"], 6),
        ("相对旧 G 的 PSNR / dB", lambda r: r["mean_candidate_minus_previous_g_psnr_db"], 5),
    ):
        values = [accessor(r) for r in reports]
        fmt = f".{digits}f" if label == "P15 PSNR / dB" else f"+.{digits}f"
        lines.append(f"| {label} | {format(values[0], fmt)} | {format(values[1], fmt)} |")
    for section, keys in (("per_group", ("GOPR0372_07", "GOPR0378_13", "GOPR0385_11")), ("per_strength", ("heavy", "medium", "light"))):
        for key in keys:
            values = [r[section][key]["mean_candidate_minus_local_psnr_db"] for r in reports]
            lines.append(f"| {key} 相对局部 / dB | {values[0]:+.5f} | {values[1]:+.5f} |")
    lines += ["", "通过条件：每个种子整体至少 +0.02 dB、整体 SSIM 不下降、各录像组及各强度档至少 -0.01 dB。", "",
              f"两种子平均增益 {cross_seed['two_seed_mean_gain_db']:+.5f} dB；逐图增益跨种子相关系数 {cross_seed['per_image_gain_pearson']:.4f}；两种子均获益 {cross_seed['positive_in_both_seeds']}/385 张，均退化 {cross_seed['negative_in_both_seeds']}/385 张。", "",
              "门均值（重／中／轻）：", ""]
    for seed, report in zip((3407, 7813), reports):
        values = [report["per_strength"][key]["mean_intervention_gate"] for key in ("heavy", "medium", "light")]
        lines.append(f"- {seed}：{values[0]:.3f} / {values[1]:.3f} / {values[2]:.3f}。")
    lines += ["", "![两种子复核](replication.png)", "", "## 解释范围", "",
              "这是开发集筛选证据。两个种子共享同一个已训练的 B0；这 385 张图曾用于方法迭代和检查点选择，相邻视频帧也不独立。数值不等同于独立测试集上的论文结论，仍需完整分辨率评价和外部数据验证。", "",
              "门关闭时新增模块对输入特征为恒等，但已微调的骨干不等于原始 B0。输入 PSNR 是受内容影响的退化代理；按整幅裁剪监督门均值，也不能保证逐像素门等于真实模糊程度。", "",
              "P15 同时改变了残差乘门方式与训练监督，因此目前证明的是这个组合的效果。只有匹配的独立训练消融，才能确定两项各自贡献。各组均值通过也不代表每张图都获得改善。", "",
              "## 下一阶段", "",
              ("进入机制消融：保留相同骨干和训练设置，分别检验仅增加外层残差门、仅增加强度监督。之后再决定是否扩展完整分辨率与 TMT 评价。" if passed else "当前候选停止晋升。记录具体未通过项，并先检查机制与训练内验证设计；不因本轮结果调整保护阈值或挑选更有利的种子。"), ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    make_figure(reports, output)
    print(json.dumps({"replication_passed": passed, "cross_seed": cross_seed, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
