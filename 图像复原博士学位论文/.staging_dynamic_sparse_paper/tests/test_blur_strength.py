from __future__ import annotations

import unittest

import torch

from src.models import (
    blend_with_baseline,
    deployable_strength_gate,
    input_psnr_per_image,
    teacher_blur_strength,
)


class BlurStrengthTests(unittest.TestCase):
    def test_teacher_strength_is_monotonic_and_bounded(self) -> None:
        psnr = torch.tensor([20.0, 24.0, 26.0, 28.5, 35.0])
        strength = teacher_blur_strength(psnr, 24.0, 28.5)
        self.assertTrue(bool(((strength >= 0.0) & (strength <= 1.0)).all()))
        self.assertTrue(bool((strength[:-1] >= strength[1:]).all()))
        self.assertEqual(float(strength[0]), 1.0)
        self.assertEqual(float(strength[-1]), 0.0)

    def test_deployable_gate_uses_only_baseline_correction(self) -> None:
        blurry = torch.zeros(3, 3, 4, 4)
        correction_l1 = torch.tensor([0.005, 0.010, 0.020])
        baseline = blurry + correction_l1[:, None, None, None]
        gate, proxy = deployable_strength_gate(
            blurry,
            baseline,
            light_cut_log_l1=float(torch.tensor(0.008).log()),
            heavy_cut_log_l1=float(torch.tensor(0.012).log()),
        )
        self.assertEqual(tuple(gate.shape), (3, 1, 1, 1))
        self.assertEqual(float(gate[0]), 0.0)
        self.assertEqual(float(gate[-1]), 1.0)
        self.assertGreater(float(gate[2]), float(gate[1]))
        self.assertTrue(torch.allclose(proxy.exp(), correction_l1, atol=1e-6))

    def test_blend_gate_endpoints_are_exact(self) -> None:
        baseline = torch.rand(2, 3, 8, 8)
        dynamic = torch.rand(2, 3, 8, 8)
        gate = torch.tensor([0.0, 1.0])[:, None, None, None]
        blended = blend_with_baseline(baseline, dynamic, gate)
        self.assertTrue(torch.equal(blended[0], baseline[0]))
        self.assertTrue(torch.equal(blended[1], dynamic[1]))

    def test_input_psnr_teacher_matches_known_mse(self) -> None:
        blurry = torch.zeros(2, 3, 4, 4)
        sharp = torch.stack((torch.full((3, 4, 4), 0.1), torch.full((3, 4, 4), 0.01)))
        psnr = input_psnr_per_image(blurry, sharp)
        self.assertTrue(torch.allclose(psnr, torch.tensor([20.0, 40.0]), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
