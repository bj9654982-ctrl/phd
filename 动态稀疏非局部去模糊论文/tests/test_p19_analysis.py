import copy
import unittest
import numpy as np
import torch

from scripts.analyze_p19_teacher import audit_assignment, aggregate, pairing_decision


def diagnostic_rows(seed=3407):
    generator = torch.Generator().manual_seed(seed + 19000)
    rows = []
    values = np.array([0, .2, .4, .6, .8, 1.])
    for step in range(1, 3001):
        shift = int(torch.randint(1, 6, (1,), generator=generator))
        target = np.roll(values, -shift)
        rows.append({"step": step, "shift": shift,
                     "target_changed_fraction": float(np.mean(values != target)),
                     "target_mean_absolute_change": float(np.abs(values - target).mean()),
                     "teacher_mean": float(values.mean()), "teacher_std": float(values.std()),
                     "teacher_unique_values": 6, "teacher_endpoint_fraction": 2/6,
                     **{f"teacher_{i}": float(value) for i, value in enumerate(values)}})
    return rows


def decision_fixture():
    seed = {"overall": {"contrasts": {"paired_minus_shuffled": {"psnr_db": .01, "ssim": .001}}},
            "last_paired_minus_shuffled": {"psnr_db": .01, "ssim": .001},
            "assignment": {"target_mean_absolute_change": {"mean": .4}}}
    return {"3407": copy.deepcopy(seed), "7813": copy.deepcopy(seed)}


class P19AnalysisTests(unittest.TestCase):
    def test_assignment_replay_passes(self):
        result = audit_assignment(diagnostic_rows(), 3407)
        self.assertTrue(result["passed"])
        self.assertEqual(result["samples"], 18000)
        self.assertEqual(result["zero_change_batches"], 0)

    def test_missing_step_rejected(self):
        with self.assertRaises(ValueError):
            audit_assignment(diagnostic_rows()[:-1], 3407)

    def test_wrong_rng_or_reported_effect_rejected(self):
        rows = diagnostic_rows()
        with self.assertRaises(ValueError):
            audit_assignment(rows, 7813)
        rows[50]["target_mean_absolute_change"] = 9
        with self.assertRaises(ValueError):
            audit_assignment(rows, 3407)

    def test_nonfinite_teacher_rejected(self):
        rows = diagnostic_rows()
        rows[100]["teacher_0"] = float("nan")
        with self.assertRaises(ValueError):
            audit_assignment(rows, 3407)

    def test_positive_mean_cannot_hide_seed_reversal(self):
        fixture = decision_fixture()
        fixture["7813"]["overall"]["contrasts"]["paired_minus_shuffled"]["psnr_db"] = -.001
        self.assertFalse(pairing_decision(fixture)["preliminary_pairing_support"])

    def test_last_checkpoint_reversal_blocks_support(self):
        fixture = decision_fixture()
        fixture["7813"]["last_paired_minus_shuffled"]["psnr_db"] = -.001
        self.assertFalse(pairing_decision(fixture)["preliminary_pairing_support"])

    def test_ssim_and_ineffective_assignment_are_not_ignored(self):
        fixture = decision_fixture()
        self.assertTrue(pairing_decision(fixture)["preliminary_pairing_support"])
        fixture["3407"]["overall"]["contrasts"]["paired_minus_shuffled"]["ssim"] = -.0001
        self.assertFalse(pairing_decision(fixture)["preliminary_pairing_support"])
        fixture = decision_fixture()
        fixture["3407"]["assignment"]["target_mean_absolute_change"]["mean"] = 0
        self.assertFalse(pairing_decision(fixture)["preliminary_pairing_support"])

    def test_contrasts_use_matched_not_raw_baseline(self):
        records = {role: {"x": {"candidate_psnr_db": score, "candidate_ssim": score / 100}}
                   for role, score in (("local", 29), ("C00", 29.05), ("paired", 29.08),
                                       ("shuffled", 29.06))}
        result = aggregate(["x"], records)
        self.assertAlmostEqual(result["contrasts"]["paired_minus_shuffled"]["psnr_db"], .02)
        self.assertAlmostEqual(result["contrasts"]["paired_minus_C00"]["psnr_db"], .03)
        self.assertAlmostEqual(result["contrasts"]["shuffled_minus_local"]["psnr_db"], .06)


if __name__ == "__main__":
    unittest.main()
