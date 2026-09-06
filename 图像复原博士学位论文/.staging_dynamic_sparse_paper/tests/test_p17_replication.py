import copy
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_p17_replication import (
    EFFECT_LABELS, REQUIRED_SNAPSHOT_FILES, RUNS_7813, audit_between_seeds, audit_new_source_snapshots,
    mean_bucket, screening_gates, summarize_effect_stability,
)


class P17ReplicationTests(unittest.TestCase):
    def data_protocol(self):
        return {"local": {"data_root": "data_fixture", "train_recording_groups": ["train"],
                           "validation_recording_groups": ["val"], "validation_patch_size": 256}}

    def test_cross_seed_audit_accepts_only_seed_change(self):
        first = {"audit": {"baseline_sha256": "same_b0", "calibration_sha256": "same_teacher",
                           "training_settings": {"seed": 3407, "steps": 3000}}, "runs": self.data_protocol()}
        second = copy.deepcopy(first)
        second["audit"]["training_settings"]["seed"] = 7813
        rows = {"local": {"image1": {"recording_group": "group", "scene": "scene", "strength_class": "heavy",
                     "input_psnr_db": 22., "baseline_psnr_db": 24., "baseline_ssim": .8}}}
        self.assertTrue(audit_between_seeds(first, second, rows, rows)["passed"])
        second["audit"]["baseline_sha256"] = "different_b0"
        with self.assertRaisesRegex(ValueError, "baseline_sha256"):
            audit_between_seeds(first, second, rows, rows)

    def test_cross_seed_misaligned_rows_are_rejected(self):
        result = {"audit": {"baseline_sha256": "b0", "calibration_sha256": "teacher",
                            "training_settings": {"seed": 3407}}, "runs": self.data_protocol()}
        with self.assertRaisesRegex(ValueError, "image names differ"):
            audit_between_seeds(result, result, {"local": {"a": {}}}, {"local": {"b": {}}})

    def test_positive_mean_does_not_hide_sign_reversal(self):
        results = {seed: {"overall": {"effects": {"psnr_db": {
                    effect: value for effect in EFFECT_LABELS}}}}
                   for seed, value in ((3407, -.01), (7813, .03))}
        effects = summarize_effect_stability(results)
        self.assertEqual(effects["outer_main"]["positive_seed_count"], 1)
        self.assertAlmostEqual(effects["outer_main"]["mean_psnr_db"], .01)
        self.assertAlmostEqual(effects["outer_main"]["sample_sd_psnr_db"], .028284271247)

    def test_one_failing_group_cannot_be_hidden_by_overall_mean(self):
        bucket = {"minus_local": {"C01": {"psnr_db": .1, "ssim": .001}}}
        result = {"overall": bucket, "per_group": {"a": copy.deepcopy(bucket)},
                  "per_strength": {"medium": copy.deepcopy(bucket)}}
        self.assertTrue(screening_gates(result, "C01")["passed"])
        result["per_group"]["a"]["minus_local"]["C01"]["psnr_db"] = -.010001
        self.assertFalse(screening_gates(result, "C01")["passed"])

    def test_mean_scores_are_not_an_ensemble(self):
        a = {"pairs": 385, "metrics": {"C01": {"psnr_db": 28., "ssim": .8}},
             "minus_local": {"C01": {"psnr_db": .1, "ssim": .01}},
             "effects": {"psnr_db": {"teacher_main": .03}}}
        b = copy.deepcopy(a)
        b["metrics"]["C01"]["psnr_db"] = 30.
        mean = mean_bucket([a, b])
        self.assertEqual(mean["metrics"]["C01"]["psnr_db"], 29.)
        self.assertEqual(mean["pairs_per_seed"], 385)
        self.assertEqual(mean["seed_count"], 2)

    def test_source_snapshot_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="p17_snapshot_test_") as temporary:
            root = Path(temporary)
            for cell in ("C10", "C01"):
                folder = root / RUNS_7813[cell] / "source_snapshot"
                for name in REQUIRED_SNAPSHOT_FILES:
                    path = folder / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("# fixture\n", encoding="utf-8")
            self.assertTrue(audit_new_source_snapshots(root)["passed"])
            path = root / RUNS_7813["C01"] / "source_snapshot" / "scripts" / "train_p12_matched_nonlocal.py"
            path.write_text("# changed fixture\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "snapshots differ"):
                audit_new_source_snapshots(root)


if __name__ == "__main__":
    unittest.main()
