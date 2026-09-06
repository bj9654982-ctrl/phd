"""Read-only historical compatibility checks; writes only its requested audit JSON."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_p12_matched_nonlocal as current


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_signature(seed, root, groups, shuffled):
    current.set_determinism(seed)
    data = current.GoProPairDataset(root, split="train", patch_size=256, augment=True,
                                    exclude_recording_groups=groups)
    loader = DataLoader(data, batch_size=6, shuffle=True, drop_last=True, num_workers=2,
                        pin_memory=True, persistent_workers=True,
                        generator=torch.Generator().manual_seed(seed + 1))
    generator = torch.Generator().manual_seed(seed + 19000) if shuffled else None
    signatures = []
    for index, batch in enumerate(loader):
        if index == 4:
            break
        signatures.append({
            "names": batch["name"],
            "blur_sha256": hashlib.sha256(batch["blur"].numpy().tobytes()).hexdigest(),
            "sharp_sha256": hashlib.sha256(batch["sharp"].numpy().tobytes()).hexdigest(),
        })
        if shuffled:
            current.assign_teacher_targets(torch.arange(6), "shuffled", generator)
    return signatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite audit")
    project = args.project_root.resolve()
    snapshot = project / "results/runs/P17C01_teacher_only_seed7813_v0/source_snapshot"
    legacy_file = snapshot / "scripts/train_p12_matched_nonlocal.py"
    if sha(legacy_file) != "7f2c81611011557a3b3a37aa2cfc9c80965e762f99149ba56b688cf9f7bfb5a3":
        raise ValueError("Historical trainer fingerprint differs from P18 freeze")
    checked_sources = {}
    for old_path in (snapshot / "src").rglob("*.py"):
        relative = old_path.relative_to(snapshot)
        if sha(old_path) != sha(ROOT / relative) or sha(old_path) != sha(project / relative):
            raise ValueError(f"Model/data source changed: {relative}")
        checked_sources[str(relative)] = sha(old_path)
    spec = importlib.util.spec_from_file_location("p19_legacy_trainer", legacy_file)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    def definitions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef)}
    old_defs, new_defs = definitions(legacy_file), definitions(Path(current.__file__))
    for name in ("_make_candidate", "set_determinism", "evaluate", "_audit_exact_initialization"):
        if old_defs[name] != new_defs[name]:
            raise ValueError(f"Historical helper changed: {name}")
    base_file = project / "results/runs/P3_baseline_group_split_v0/best.pt"
    baseline = torch.load(base_file, map_location="cpu", weights_only=False)
    results = {}
    for seed in (3407, 7813):
        legacy.set_determinism(seed)
        old_model = legacy._make_candidate(baseline["model"], 24, "gate_adaptive", (1, 3, 7), 6, 5)
        current.set_determinism(seed)
        new_model = current._make_candidate(baseline["model"], 24, "gate_adaptive", (1, 3, 7), 6, 5)
        if not all(torch.equal(value, new_model.state_dict()[key])
                   for key, value in old_model.state_dict().items()):
            raise ValueError("Initial parameters differ")
        if sum(p.numel() for p in new_model.parameters()) != 1095511:
            raise ValueError("Capacity differs")
        paired = data_signature(seed, project / "data/GOPRO_Large",
                                baseline["args"]["validation_groups"], False)
        shuffled = data_signature(seed, project / "data/GOPRO_Large",
                                  baseline["args"]["validation_groups"], True)
        if paired != shuffled:
            raise ValueError("Data/augmentation stream changed")
        model = new_model.cuda()
        rng = torch.Generator(device="cuda").manual_seed(seed + 77)
        blurry = torch.rand(6, 3, 48, 64, device="cuda", generator=rng)
        sharp = (blurry + .06).clamp(0, 1)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction, geometry = model(blurry, return_geometry=True)
            old_aux = legacy.compute_gate_supervision_loss(geometry, blurry, sharp, 24.02, 28.30)
            new_aux = current.compute_gate_supervision_loss(geometry, blurry, sharp, 24.02, 28.30)
            old_loss = legacy.restoration_loss(prediction, sharp, .1) + .01 * old_aux
            new_loss, _, _ = current.compute_training_objective(model, blurry, sharp, 24.02, 28.30, .1, .01)
        if not torch.equal(old_aux, new_aux) or not torch.equal(old_loss, new_loss):
            raise ValueError("Default objective changed")
        results[str(seed)] = {"initial_parameters_identical": True,
                              "parameter_count": 1095511, "default_amp_loss_identical": True,
                              "first_four_training_batches": paired}
        del model, old_model, new_model
    result = {"passed": True, "seeds": results, "source_sha256": checked_sources,
              "legacy_trainer_sha256": sha(legacy_file),
              "current_trainer_sha256": sha(Path(current.__file__)),
              "baseline_sha256": sha(base_file), "official_test_accessed": False,
              "note": "Four-batch stream sample plus unchanged loader code; not a full historical replay."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": True, "seeds": list(results), "source_files": len(checked_sources),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
