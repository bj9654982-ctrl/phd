import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from scripts.analyze_p16_factorial import (
    FACTORS, RUNS, factorial_effects, load_and_audit, read_csv, write_csv,
)


class P16FactorialTests(unittest.TestCase):
    def test_additive_factors_have_zero_interaction(self):
        effects = factorial_effects({"C00": [1., 2.], "C10": [3., 4.],
                                     "C01": [4., 5.], "C11": [6., 7.]})
        for key in ("outer_without_teacher", "outer_with_teacher", "outer_main"):
            np.testing.assert_array_equal(effects[key], [2., 2.])
        for key in ("teacher_without_outer", "teacher_with_outer", "teacher_main"):
            np.testing.assert_array_equal(effects[key], [3., 3.])
        np.testing.assert_array_equal(effects["interaction"], [0., 0.])

    def test_interaction_can_reverse_conditional_effect(self):
        effects = factorial_effects({"C00": 10., "C10": 9., "C01": 12., "C11": 14.})
        self.assertEqual(float(effects["outer_without_teacher"]), -1.)
        self.assertEqual(float(effects["outer_with_teacher"]), 2.)
        self.assertEqual(float(effects["interaction"]), 3.)
        self.assertEqual(float(effects["outer_main"]), .5)


class P16AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="p16_audit_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        baseline = self.root / "baseline.pt"
        torch.save({"test_fixture": True}, baseline)
        calibration = self.root / "calibration.json"
        calibration.write_text(json.dumps({"official_test_accessed": False,
            "heavy_medium_threshold_input_psnr_db": 24.,
            "medium_light_threshold_input_psnr_db": 28.}), encoding="utf-8")
        group_names = ["GOPR0372_07"] * 175 + ["GOPR0378_13"] * 110 + ["GOPR0385_11"] * 100
        for cell, run in RUNS.items():
            directory = self.root / run
            directory.mkdir()
            outer, weight = FACTORS[cell]
            variant = "local" if cell == "local" else (
                "protected_gate_adaptive" if outer else "gate_adaptive")
            setting = {"seed": 3407, "steps": 3000, "batch_size": 6,
                "train_patch_size": 256, "learning_rate": .0002, "frequency_weight": .1,
                "val_interval": 375, "num_workers": 2, "train_pairs": 0, "val_pairs": 0,
                "variant": variant, "gate_supervision_weight": weight, "sampling": "image_uniform",
                "scales": [1, 3, 7], "top_k": 6, "statistics_window": 5,
                "baseline_checkpoint": baseline, "data_root": self.root / "data",
                "calibration_summary": calibration}
            torch.save({"args": setting, "step": 3000}, directory / "best.pt")
            summary = {"seed": 3407, "steps": 3000, "variant": variant,
                "gate_supervision_weight": weight, "candidate_parameters": 1095511,
                "official_test_accessed": False, "pairs": 385, "val_pairs": 385,
                "train_pairs": 1718, "recording_group_overlap": [],
                "exact_initialization_max_abs_error": 0, "validation_patch_size": 256,
                "best_checkpoint_step": 3000, "candidate_psnr_db": 29., "candidate_ssim": .8,
                "train_recording_groups": ["train_fixture"],
                "validation_recording_groups": list(dict.fromkeys(group_names))}
            (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            write_csv(directory / "metrics.csv", [{"step": step, "candidate_psnr_db": 29. if step == 3000 else 28.}
                                                  for step in [1, *range(375, 3001, 375)]])
            write_csv(directory / "per_image_metrics.csv", [
                {"name": str(i), "scene": group, "recording_group": group,
                 "strength_class": "light", "input_psnr_db": 30., "baseline_psnr_db": 28.,
                 "baseline_ssim": .7, "candidate_psnr_db": 29., "candidate_ssim": .8}
                for i, group in enumerate(group_names)])

    def test_matching_fixture_passes(self):
        self.assertTrue(load_and_audit(self.root)[2]["passed"])

    def test_second_seed_requires_explicit_matching_seed(self):
        for run in RUNS.values():
            path = self.root / run / "best.pt"
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["args"]["seed"] = 7813
            torch.save(checkpoint, path)
            path = self.root / run / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["seed"] = 7813
            path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unexpected seed"):
            load_and_audit(self.root)
        self.assertTrue(load_and_audit(self.root, seed=7813, runs=RUNS)[2]["passed"])

    def test_reusing_one_run_for_two_cells_is_rejected(self):
        invalid = {**RUNS, "C01": RUNS["C00"]}
        with self.assertRaisesRegex(ValueError, "distinct run"):
            load_and_audit(self.root, runs=invalid)

    def test_wrong_teacher_cell_is_rejected(self):
        path = self.root / RUNS["C01"] / "best.pt"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["args"]["gate_supervision_weight"] = 0.
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "supervision differs"):
            load_and_audit(self.root)

    def test_wrong_checkpoint_selection_is_rejected(self):
        path = self.root / RUNS["C10"] / "metrics.csv"
        write_csv(path, [{"step": step, "candidate_psnr_db": 30. if step == 375 else 29.}
                        for step in [1, *range(375, 3001, 375)]])
        with self.assertRaisesRegex(ValueError, "selection differs"):
            load_and_audit(self.root)

    def test_wrong_seed_is_rejected(self):
        path = self.root / RUNS["C10"] / "best.pt"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["args"]["seed"] = 7813
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "unexpected seed"):
            load_and_audit(self.root)

    def test_baseline_metric_misalignment_is_rejected(self):
        path = self.root / RUNS["C11"] / "per_image_metrics.csv"
        rows = read_csv(path)
        rows[0]["baseline_psnr_db"] = 27.
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, "baseline_psnr_db differs"):
            load_and_audit(self.root)

    def test_nonfinite_input_metric_is_rejected(self):
        path = self.root / RUNS["C00"] / "per_image_metrics.csv"
        rows = read_csv(path)
        rows[0]["input_psnr_db"] = "nan"
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, "input_psnr_db differs"):
            load_and_audit(self.root)


if __name__ == "__main__":
    unittest.main()
