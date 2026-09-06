"""P19 paired-vs-shuffled teacher audit, CUDA endpoint evaluation, and report."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_p16_factorial import RUNS, load_and_audit, read_csv, sha256, write_csv
from scripts.analyze_p17_replication import RUNS_7813
from scripts import train_p12_matched_nonlocal as training

SEEDS = (3407, 7813)
ROLES = ("local", "C00", "paired", "shuffled")
CONTRASTS = {"paired_minus_shuffled": ("paired", "shuffled"),
             "paired_minus_C00": ("paired", "C00"),
             "shuffled_minus_C00": ("shuffled", "C00"),
             "paired_minus_local": ("paired", "local"),
             "shuffled_minus_local": ("shuffled", "local")}


def audit_assignment(rows, seed):
    if len(rows) != 3000 or [int(row["step"]) for row in rows] != list(range(1, 3001)):
        raise ValueError("Teacher diagnostics must cover all 3000 steps")
    generator = torch.Generator().manual_seed(seed + 19000)
    metrics = {key: [] for key in ("target_changed_fraction", "target_mean_absolute_change",
                                   "teacher_mean", "teacher_std", "teacher_unique_values",
                                   "teacher_endpoint_fraction")}
    all_values = []
    for row in rows:
        shift = int(torch.randint(1, 6, (1,), generator=generator).item())
        if shift != int(row["shift"]):
            raise ValueError("Teacher RNG sequence differs from preregistration")
        values = np.array([float(row[f"teacher_{i}"]) for i in range(6)])
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Invalid teacher values")
        target = np.roll(values, -shift)
        expected = {
            "target_changed_fraction": float(np.mean(values != target)),
            "target_mean_absolute_change": float(np.abs(values - target).mean()),
            "teacher_mean": float(values.mean()), "teacher_std": float(values.std()),
            "teacher_unique_values": len(np.unique(values)),
            "teacher_endpoint_fraction": float(np.mean((values == 0) | (values == 1))),
        }
        for key, value in expected.items():
            if abs(float(row[key]) - value) > 1e-10:
                raise ValueError(f"Teacher diagnostic inconsistent: {key}")
            metrics[key].append(value)
        all_values.extend(values.tolist())
    result = {key: {"mean": float(np.mean(values)),
                    "min": float(np.min(values)), "max": float(np.max(values)),
                    "q05": float(np.quantile(values, .05)), "q50": float(np.median(values)),
                    "q95": float(np.quantile(values, .95))}
              for key, values in metrics.items()}
    result.update({"passed": True, "steps": 3000, "samples": len(all_values),
                   "zero_change_batches": metrics["target_changed_fraction"].count(0),
                   "all_teacher_quantiles": np.quantile(all_values, [0, .05, .25, .5, .75, .95, 1]).tolist()})
    return result


def aggregate(names, records):
    values = {role: {metric: np.array([float(records[role][name][f"candidate_{metric}"])
                                     for name in names])
                     for metric in ("psnr_db", "ssim")} for role in ROLES}
    return {
        "pairs": len(names),
        "scores": {role: {metric: float(array.mean()) for metric, array in data.items()}
                   for role, data in values.items()},
        "contrasts": {label: {metric: float((values[a][metric] - values[b][metric]).mean())
                              for metric in ("psnr_db", "ssim")}
                      for label, (a, b) in CONTRASTS.items()},
    }


def pairing_decision(by_seed):
    best = all(item["overall"]["contrasts"]["paired_minus_shuffled"]["psnr_db"] > 0
               and item["overall"]["contrasts"]["paired_minus_shuffled"]["ssim"] >= 0
               for item in by_seed.values())
    last = all(item["last_paired_minus_shuffled"]["psnr_db"] >= 0
               and item["last_paired_minus_shuffled"]["ssim"] >= 0
               for item in by_seed.values())
    changed = all(item["assignment"]["target_mean_absolute_change"]["mean"] > 0
                  for item in by_seed.values())
    return {"best_sign_rule_met": best, "last_no_direction_reversal": last,
            "targets_actually_changed": changed,
            "preliminary_pairing_support": best and last and changed,
            "statistical_significance_established": False}


def finite_correlation(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


@torch.inference_mode()
def evaluate_checkpoint(checkpoint, baseline, setting, reference, calibration):
    model = training._make_candidate(baseline["model"], int(baseline["args"]["width"]),
                                     "gate_adaptive", (1, 3, 7), 6, 5)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.cuda().eval()
    data = training.GoProPairDataset(
        setting["data_root"], split="train", patch_size=256, augment=False,
        include_recording_groups=baseline["args"]["validation_groups"])
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True,
                        generator=torch.Generator().manual_seed(int(setting["seed"]) + 2))
    if len(data) != 385:
        raise ValueError("Unexpected endpoint evaluation split")
    heavy = calibration["heavy_medium_threshold_input_psnr_db"]
    light = calibration["medium_light_threshold_input_psnr_db"]
    result = {}
    for batch in loader:
        name = batch["name"][0]
        if name not in reference or name in result:
            raise ValueError("Endpoint evaluation image mismatch")
        blurry, sharp = batch["blur"].cuda(), batch["sharp"].cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            prediction, geometry = model(blurry, return_geometry=True)
        input_psnr = training._psnr_value(blurry, sharp)
        if abs(input_psnr - float(reference[name]["input_psnr_db"])) > 1e-6:
            raise ValueError("Endpoint input differs from original evaluation")
        teacher = training.teacher_blur_strength(
            training.input_psnr_per_image(blurry, sharp), heavy, light)
        gate = geometry["nonlocal_gate"].float()
        result[name] = {
            **{field: reference[name][field] for field in
               ("name", "scene", "recording_group", "strength_class")},
            "input_psnr_db": input_psnr,
            "candidate_psnr_db": training._psnr_value(prediction, sharp),
            "candidate_ssim": training._ssim_value(prediction[0], sharp[0]),
            "gate_mean": float(gate.mean()), "gate_spatial_std": float(gate.std(unbiased=False)),
            "true_teacher": float(teacher[0]),
        }
    if set(result) != set(reference):
        raise ValueError("Missing endpoint images")
    del model, loader
    return result


def endpoint_audit(root, run, baseline, reference, calibration, output):
    best = torch.load(root / run / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(root / run / "last.pt", map_location="cpu", weights_only=False)
    setting = dict(best["args"])
    if last["step"] != 3000 or dict(last["args"]) != setting:
        raise ValueError("Terminal checkpoint protocol mismatch")
    evaluated = evaluate_checkpoint(best, baseline, setting, reference, calibration)
    for name in evaluated:
        for key in ("candidate_psnr_db", "candidate_ssim"):
            if abs(evaluated[name][key] - float(reference[name][key])) > 1e-6:
                raise ValueError(f"Best-checkpoint replay differs: {run}/{name}/{key}")
    reused = best["step"] == last["step"]
    if reused:
        if not all(torch.equal(value, last["model"][key]) for key, value in best["model"].items()):
            raise ValueError("Same-step best/last parameters differ")
        terminal = copy.deepcopy(evaluated)
    else:
        terminal = evaluate_checkpoint(last, baseline, setting, reference, calibration)
    write_csv(output / f"{run}_best_replay.csv", list(evaluated.values()))
    write_csv(output / f"{run}_last_replay.csv", list(terminal.values()))
    gates = np.array([row["gate_mean"] for row in evaluated.values()])
    teachers = np.array([row["true_teacher"] for row in evaluated.values()])
    psnrs = np.array([row["input_psnr_db"] for row in evaluated.values()])
    report = {"best_replay_max_error_below": 1e-6, "same_step_reused": reused,
              "last_checkpoint_sha256": sha256(root / run / "last.pt"),
              "gate_vs_teacher_pearson": finite_correlation(gates, teachers),
              "gate_vs_input_psnr_pearson": finite_correlation(gates, psnrs),
              "gate_mean": float(gates.mean()), "gate_across_image_std": float(gates.std()),
              "gate_teacher_mae": float(np.abs(gates - teachers).mean()),
              "gate_mean_by_strength": {
                  label: float(np.mean([row["gate_mean"] for row in evaluated.values()
                                        if row["strength_class"] == label]))
                  for label in ("heavy", "medium", "light")},
              "last_scores": {metric: float(np.mean([row[f"candidate_{metric}"]
                                                    for row in terminal.values()]))
                              for metric in ("psnr_db", "ssim")}}
    return report, terminal


def build_seed(root, seed, output):
    mapping = RUNS if seed == 3407 else RUNS_7813
    summaries, records, old_audit = load_and_audit(root, seed=seed, runs=mapping)
    shuffled_run = f"P19C01_shuffled_teacher_seed{seed}_v0"
    new_map = {**mapping, "C01": shuffled_run}
    new_summaries, new_records, new_audit = load_and_audit(root, seed=seed, runs=new_map)
    directory = root / shuffled_run
    new_checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    setting = dict(new_checkpoint["args"])
    for source in (setting, new_summaries["C01"]):
        if source.get("teacher_mode") != "shuffled" or source.get("teacher_seed") != seed + 19000:
            raise ValueError("Shuffled-teacher configuration differs")
    paired_checkpoint = torch.load(root / mapping["C01"] / "best.pt",
                                   map_location="cpu", weights_only=False)
    if paired_checkpoint["args"].get("teacher_mode", "paired") != "paired":
        raise ValueError("Historical paired teacher is not paired")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    for key, value in setting.items():
        if config[key] != (str(value) if isinstance(value, Path) else value):
            raise ValueError(f"Saved config differs: {key}")
    assignment = audit_assignment(read_csv(directory / "teacher_diagnostics.csv"), seed)
    recorded = new_summaries["C01"]["teacher_assignment_diagnostics"]
    for target, source in (("mean_target_changed_fraction", "target_changed_fraction"),
                           ("mean_target_absolute_change", "target_mean_absolute_change"),
                           ("mean_within_batch_teacher_std", "teacher_std"),
                           ("mean_teacher_endpoint_fraction", "teacher_endpoint_fraction")):
        if abs(recorded[target] - assignment[source]["mean"]) > 1e-10:
            raise ValueError("Teacher summary differs from complete per-step diagnostics")
    snapshot = directory / "source_snapshot"
    source_hashes = {str(p.relative_to(snapshot)): sha256(p)
                     for p in snapshot.rglob("*") if p.suffix in (".py", ".ps1")}
    if len(source_hashes) != 16:
        raise ValueError("Incomplete training source snapshot")
    for relative, digest in source_hashes.items():
        if digest != sha256(root.parent.parent / relative):
            raise ValueError(f"Training source changed: {relative}")
    paired_rows = {"local": records["local"], "C00": records["C00"],
                   "paired": records["C01"], "shuffled": new_records["C01"]}
    names = sorted(records["local"])
    result = {
        "seed": seed, "run_names": {"local": mapping["local"], "C00": mapping["C00"],
                                   "paired": mapping["C01"], "shuffled": shuffled_run},
        "overall": aggregate(names, paired_rows), "assignment": assignment,
        "historical_audit": old_audit, "shuffled_audit": new_audit,
        "source_snapshot_sha256": source_hashes,
        "diagnostics_sha256": sha256(directory / "teacher_diagnostics.csv"),
        "shuffled_wall_seconds": new_summaries["C01"]["wall_seconds"],
        "shuffled_peak_vram_mib": new_summaries["C01"]["peak_vram_mib"],
        "best_steps": {role: source["best_checkpoint_step"] for role, source in
                       (("local", summaries["local"]), ("C00", summaries["C00"]),
                        ("paired", summaries["C01"]), ("shuffled", new_summaries["C01"]))},
    }
    for dimension, field in (("per_group", "recording_group"), ("per_strength", "strength_class")):
        labels = sorted({records["local"][name][field] for name in names})
        result[dimension] = {label: aggregate([name for name in names
                                             if records["local"][name][field] == label], paired_rows)
                             for label in labels}
    result["screening_gates"] = {}
    for role in ("paired", "shuffled"):
        key = f"{role}_minus_local"
        gates = {"overall_psnr": result["overall"]["contrasts"][key]["psnr_db"] >= .02,
                 "overall_ssim": result["overall"]["contrasts"][key]["ssim"] >= 0,
                 "groups": all(g["contrasts"][key]["psnr_db"] >= -.01 for g in result["per_group"].values()),
                 "strengths": all(g["contrasts"][key]["psnr_db"] >= -.01 for g in result["per_strength"].values())}
        result["screening_gates"][role] = {**gates, "passed": all(gates.values())}
    baseline = torch.load(setting["baseline_checkpoint"], map_location="cpu", weights_only=False)
    calibration = json.loads(Path(setting["calibration_summary"]).read_text(encoding="utf-8"))
    endpoints, terminals = {}, {}
    for role in ("paired", "shuffled"):
        print(f"Evaluating seed={seed} role={role} best/last", flush=True)
        endpoints[role], terminals[role] = endpoint_audit(
            root, result["run_names"][role], baseline, paired_rows[role], calibration, output)
    result["endpoints"] = endpoints
    result["last_paired_minus_shuffled"] = {
        metric: endpoints["paired"]["last_scores"][metric] - endpoints["shuffled"]["last_scores"][metric]
        for metric in ("psnr_db", "ssim")}
    write_csv(output / f"seed{seed}_per_image.csv", [
        {"name": name, "recording_group": records["local"][name]["recording_group"],
         "strength_class": records["local"][name]["strength_class"],
         **{f"{role}_{metric}": float(paired_rows[role][name][f"candidate_{metric}"])
            for role in ROLES for metric in ("psnr_db", "ssim")},
         **{f"last_{role}_{metric}": terminals[role][name][f"candidate_{metric}"]
            for role in ("paired", "shuffled") for metric in ("psnr_db", "ssim")}}
        for name in names])
    return result


def make_report(result, output):
    lines = ["# P19：配对监督与批内错配监督", "",
             "两个微调种子共享同一 B0；385 张冻结开发图，256 中心裁剪。不是独立测试。", "",
             "## 总体结果", "",
             "| 种子 | 模型 | PSNR/dB | SSIM | 相对局部/dB | 最佳步数 |",
             "|---|---|---:|---:|---:|---:|"]
    for seed, item in result["by_seed"].items():
        for role in ROLES:
            score = item["overall"]["scores"][role]
            gain = score["psnr_db"] - item["overall"]["scores"]["local"]["psnr_db"]
            lines.append(f"| {seed} | {role} | {score['psnr_db']:.5f} | {score['ssim']:.7f} | "
                         f"{gain:+.5f} | {item['best_steps'][role]} |")
    lines += ["", "## 正确配对的边际作用（paired − shuffled）", "",
              "| 种子 | 最佳点 PSNR/dB | 最佳点 SSIM | 第3000步 PSNR/dB | 第3000步 SSIM |",
              "|---|---:|---:|---:|---:|"]
    for seed, item in result["by_seed"].items():
        best, last = item["overall"]["contrasts"]["paired_minus_shuffled"], item["last_paired_minus_shuffled"]
        lines.append(f"| {seed} | {best['psnr_db']:+.5f} | {best['ssim']:+.7f} | "
                     f"{last['psnr_db']:+.5f} | {last['ssim']:+.7f} |")
    supported = result["decision"]["preliminary_pairing_support"]
    lines += ["", "预登记符号规则：" + ("满足，初步支持正确配对的作用。" if supported else
             "未全部满足，不能据此将收益主要归因于正确配对的退化信息。"),
             "该判定不等于统计显著性或实际意义；差值很小时必须保留这一限制。", "",
             "## 错配是否实际改变监督", "",
             "| 种子 | 目标改变比例 | 平均绝对改变量 | 批内教师标准差 | 零改动批次/3000 |",
             "|---|---:|---:|---:|---:|"]
    for seed, item in result["by_seed"].items():
        d = item["assignment"]
        lines.append(f"| {seed} | {d['target_changed_fraction']['mean']:.2%} | "
                     f"{d['target_mean_absolute_change']['mean']:.5f} | "
                     f"{d['teacher_std']['mean']:.5f} | {d['zero_change_batches']} |")
    lines += ["", "保存全部 3000 步教师值、循环偏移和统计量，并重放独立 RNG 核验；"
              "目标值的批内集合保持不变。数值重复的目标仍可能未改变，批内错配不等价于全数据独立随机标签。", ""]
    for dimension, title in (("per_group", "录像组"), ("per_strength", "输入 PSNR 代理档")):
        lines += [f"## {title}：配对与错配的差异", "",
                  "| 种子 | 分组 | 张数 | PSNR差/dB | SSIM差 |", "|---|---|---:|---:|---:|"]
        for seed, item in result["by_seed"].items():
            for label, group in item[dimension].items():
                delta = group["contrasts"]["paired_minus_shuffled"]
                lines.append(f"| {seed} | {label} | {group['pairs']} | "
                             f"{delta['psnr_db']:+.5f} | {delta['ssim']:+.7f} |")
    lines += ["", "## 门诊断", "",
              "| 种子 | 教师 | 门与真实代理相关系数 | 平均门值 | 重/中/轻平均门值 |",
              "|---|---|---:|---:|---|"]
    for seed, item in result["by_seed"].items():
        for role, values in item["endpoints"].items():
            groups = "/".join(f"{values['gate_mean_by_strength'][key]:.3f}"
                              for key in ("heavy", "medium", "light"))
            correlation = values["gate_vs_teacher_pearson"]
            formatted = "未定义" if correlation is None else f"{correlation:.5f}"
            lines.append(f"| {seed} | {role} | {formatted} | {values['gate_mean']:.5f} | {groups} |")
    lines += ["", "相关性只是描述：门均值被直接监督，不能用相关性证明空间定位或物理模糊核估计。", "",
              "## 审计与限制", "",
              "- 保留有利、不利的录像组和代理档；总体提升不等于逐图安全。",
              "- 历史预算、初始 B0、数据划分、逐图 B0 指标、检查点选择和新教师配置经过核验。",
              "- 新训练源文件快照及原始结果指纹保留；历史配对实验并非全部拥有完整源码快照。",
              "- 最佳权重重评与原表逐图 PSNR/SSIM 误差须不超过 1e-6；末点另存，未覆盖历史结果。",
              "- 两个种子共享 B0，反复开发同一划分；没有据相邻帧计算强显著性结论。",
              "- 即使正确配对优于错配，仍不能排除一般辅助表征学习等解释。",
              "- 官方测试未使用；第三种子、整图、效率与跨退化证据尚待完成。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.runs_root.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("Refusing to overwrite analysis")
    output.mkdir(parents=True)
    training.set_determinism(19019)
    by_seed = {str(seed): build_seed(root, seed, output) for seed in SEEDS}
    first, second = by_seed.values()
    if (first["historical_audit"]["baseline_sha256"] != second["historical_audit"]["baseline_sha256"]
            or first["historical_audit"]["calibration_sha256"] != second["historical_audit"]["calibration_sha256"]):
        raise ValueError("Cross-seed baseline/calibration differ")
    if first["source_snapshot_sha256"] != second["source_snapshot_sha256"]:
        raise ValueError("Cross-seed source snapshots differ")
    result = {"experiment": "P19 paired vs shuffled teacher", "by_seed": by_seed,
              "decision": pairing_decision(by_seed),
              "mean_contrasts": {label: {metric: float(np.mean([
                  item["overall"]["contrasts"][label][metric] for item in by_seed.values()]))
                  for metric in ("psnr_db", "ssim")} for label in CONTRASTS},
              "total_training_seconds": sum(item["shuffled_wall_seconds"] for item in by_seed.values()),
              "analysis_source_sha256": sha256(Path(__file__)),
              "official_test_accessed": False, "paper_result": False}
    (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                                        encoding="utf-8")
    make_report(result, output)
    print(json.dumps({"decision": result["decision"], "mean_contrasts": result["mean_contrasts"],
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
