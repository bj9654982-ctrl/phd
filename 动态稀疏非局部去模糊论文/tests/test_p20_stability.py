import copy
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.analyze_p20_stability import (
    aggregate, audit_config, audit_history, audit_setting, compare_rows, decision, index_rows,
    same_endpoint, seed_statistics,
)


def setting():
    return dict(seed=12011, steps=3000, batch_size=6, train_patch_size=256,
                learning_rate=.0002, frequency_weight=.1, val_interval=375,
                num_workers=2, train_pairs=0, val_pairs=0, variant="gate_adaptive",
                gate_supervision_weight=.01, scales=[1, 3, 7], top_k=6, statistics_window=5)


def rows():
    return {role: {"x": dict(name="x", scene="scene", recording_group="group",
                            strength_class="medium", input_psnr_db=25, baseline_psnr_db=28,
                            baseline_ssim=.8, candidate_psnr_db=score, candidate_ssim=ssim)}
            for role, score, ssim in (("local", 29, .85), ("C00", 29.04, .8502),
                                      ("C01", 29.07, .8505))}


class P20Tests(unittest.TestCase):
    def test_missing_legacy_config_is_explicit_but_new_missing_config_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(audit_config(Path(directory), 3407, {})["config_available"])
            with self.assertRaises(ValueError):
                audit_config(Path(directory), 12011, {})

    def test_matched_gains(self):
        result = aggregate(rows())
        self.assertAlmostEqual(result["overall"]["contrasts"]["C01_minus_local"]["psnr_db"], .07)
        self.assertAlmostEqual(result["overall"]["contrasts"]["C01_minus_C00"]["psnr_db"], .03)
        self.assertTrue(decision(result, result)["all_preregistered_rules_passed"])

    def test_wrong_setting_rejected(self):
        for key, value in (("seed", 3407), ("steps", 2999), ("variant", "protected_gate_adaptive"),
                           ("top_k", 8), ("gate_supervision_weight", 0), ("teacher_mode", "shuffled"),
                           ("teacher_seed", 123), ("sampling", "group_balanced")):
            config = {**setting(), key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit_setting(config, 12011, "C01")

    def test_historical_paired_default(self):
        audit_setting(setting(), 12011, "C01")

    def test_last_negative_psnr_or_ssim_blocks(self):
        best = aggregate(rows())
        for contrast in ("C01_minus_local", "C01_minus_C00"):
            for metric in ("psnr_db", "ssim"):
                last = copy.deepcopy(best)
                last["overall"]["contrasts"][contrast][metric] = -.00001
                result = decision(best, last)
                self.assertFalse(result["all_preregistered_rules_passed"])
                self.assertTrue(result["checkpoint_dependence_detected"])

    def test_negative_group_cannot_hide_in_positive_mean(self):
        best = aggregate(rows())
        best["per_group"]["group"]["contrasts"]["C01_minus_local"]["psnr_db"] = -.011
        self.assertFalse(decision(best, best)["local_screening_passed"])

    def test_negative_strength_cannot_hide(self):
        best = aggregate(rows())
        best["per_strength"]["medium"]["contrasts"]["C01_minus_local"]["psnr_db"] = -.011
        self.assertFalse(decision(best, best)["all_preregistered_rules_passed"])

    def test_teacher_sign_not_substituted_by_local_gain(self):
        best = aggregate(rows())
        best["overall"]["contrasts"]["C01_minus_C00"]["psnr_db"] = 0
        self.assertFalse(decision(best, best)["teacher_best_sign_passed"])

    def test_sample_sd_not_population_sd(self):
        result = seed_statistics([1, 2, 3])
        self.assertEqual(result["mean"], 2)
        self.assertEqual(result["sample_sd_ddof1"], 1)
        for values in ([1, 2], [1, 2, float("nan")]):
            with self.assertRaises(ValueError):
                seed_statistics(values)

    def test_nonfinite_duplicate_misaligned_rejected(self):
        row = rows()["local"]["x"]
        for values in ([row, row], [{**row, "candidate_ssim": float("nan")}], []):
            with self.assertRaises(ValueError):
                index_rows(values)
        for modified in ({"y": row}, {"x": {**row, "input_psnr_db": 26}},
                         {"x": {**row, "recording_group": "changed"}}):
            with self.assertRaises(ValueError):
                compare_rows(modified, {"x": row})

    def test_replay_prediction_comparison(self):
        row = rows()["local"]["x"]
        modified = {"x": {**row, "candidate_ssim": .9}}
        compare_rows(modified, {"x": row})
        with self.assertRaises(ValueError):
            compare_rows(modified, {"x": row}, predictions=True)

    def test_same_step_requires_equal_tensor_keys_and_values(self):
        best = dict(step=3000, model={"weight": torch.ones(2)})
        self.assertTrue(same_endpoint(best, copy.deepcopy(best)))
        self.assertFalse(same_endpoint({**best, "step": 2625}, best))
        for model in ({"weight": torch.zeros(2)}, {"other": torch.ones(2)}):
            with self.assertRaises(ValueError):
                same_endpoint(best, dict(step=3000, model=model))

    def test_history_strict_best_tie_and_fixed_last(self):
        history = [{"step": step, "candidate_psnr_db": 29 if step >= 2625 else 28}
                   for step in [1, *range(375, 3001, 375)]]
        best, last = dict(step=2625, args={}), dict(step=3000, args={})
        summary = dict(best_checkpoint_step=2625, candidate_psnr_db=29)
        audit_history(history, summary, best, last)
        with self.assertRaises(ValueError):
            audit_history(history, {**summary, "best_checkpoint_step": 3000},
                          {**best, "step": 3000}, last)
        with self.assertRaises(ValueError):
            audit_history(history, summary, best, {**last, "step": 2625})
        with self.assertRaises(ValueError):
            audit_history(history[:-1], summary, best, last)


if __name__ == "__main__":
    unittest.main()
