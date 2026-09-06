from __future__ import annotations

import unittest

import torch

from src.models import (
    ImageResidualGate,
    SpatialResidualCorrection,
    SpatialResidualGate,
    apply_anchored_residual_correction,
    apply_residual_gate,
    compose_protected_expert,
    mse_optimal_residual_alpha,
    mse_optimal_residual_alpha_map,
)


class ImageResidualGateTests(unittest.TestCase):

    def test_complementary_composition_keeps_branch_responsibilities_separate(self) -> None:
        blurry = torch.zeros(1, 3, 4, 4)
        baseline = torch.full_like(blurry, 0.4)
        expert = torch.full_like(blurry, 0.7)
        alpha = torch.full((1, 1, 1, 1), 0.5)
        expert_gate = torch.full((1, 1, 1, 1), 0.25)
        actual = compose_protected_expert(blurry, baseline, expert, alpha, expert_gate)
        expected = blurry + 0.5 * (baseline - blurry) + 0.25 * (expert - baseline)
        torch.testing.assert_close(actual, expected)

    def test_complementary_composition_reduces_to_protected_b0_for_zero_expert_gate(self) -> None:
        blurry = torch.rand(2, 3, 5, 7)
        baseline = torch.rand_like(blurry)
        expert = torch.rand_like(blurry)
        alpha = torch.rand(2, 1, 5, 7)
        expert_gate = torch.zeros(2, 1, 1, 1)
        actual = compose_protected_expert(blurry, baseline, expert, alpha, expert_gate)
        expected = apply_residual_gate(blurry, baseline, alpha)
        torch.testing.assert_close(actual, expected)
    def test_initial_gate_is_near_baseline_and_has_valid_shape(self) -> None:
        gate = ImageResidualGate(width=8, initial_alpha=0.95)
        blurry = torch.rand(2, 3, 32, 40)
        residual = torch.rand_like(blurry) - 0.5
        alpha = gate(blurry, residual)
        self.assertEqual(tuple(alpha.shape), (2, 1, 1, 1))
        self.assertTrue(torch.all((alpha > 0) & (alpha < 1)))
        self.assertTrue(torch.allclose(alpha, torch.full_like(alpha, 0.95), atol=0.01))

    def test_gate_interpolation_endpoints(self) -> None:
        blurry = torch.rand(2, 3, 16, 16)
        baseline = torch.rand_like(blurry)
        zeros = torch.zeros(2, 1, 1, 1)
        ones = torch.ones(2, 1, 1, 1)
        self.assertTrue(torch.equal(apply_residual_gate(blurry, baseline, zeros), blurry))
        self.assertTrue(torch.allclose(apply_residual_gate(blurry, baseline, ones), baseline))

    def test_gate_receives_finite_gradients(self) -> None:
        gate = ImageResidualGate(width=8)
        blurry = torch.rand(2, 3, 24, 24)
        baseline = torch.rand_like(blurry)
        target = torch.rand_like(blurry)
        alpha = gate(blurry, baseline - blurry)
        prediction = apply_residual_gate(blurry, baseline, alpha)
        prediction.sub(target).square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in gate.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in gate.parameters()))

    def test_spatial_gate_is_near_initial_alpha_and_preserves_resolution(self) -> None:
        gate = SpatialResidualGate(width=8, initial_alpha=0.9)
        blurry = torch.rand(2, 3, 31, 37)
        residual = torch.rand_like(blurry) - 0.5
        alpha = gate(blurry, residual)
        self.assertEqual(tuple(alpha.shape), (2, 1, 31, 37))
        self.assertTrue(torch.all((alpha > 0) & (alpha < 1)))
        self.assertTrue(torch.allclose(alpha, torch.full_like(alpha, 0.9), atol=0.01))

    def test_spatial_gate_interpolation_accepts_dense_map(self) -> None:
        blurry = torch.rand(1, 3, 12, 14)
        baseline = torch.rand_like(blurry)
        alpha = torch.zeros(1, 1, 12, 14)
        alpha[:, :, :, 7:] = 1.0
        prediction = apply_residual_gate(blurry, baseline, alpha)
        self.assertTrue(torch.equal(prediction[:, :, :, :7], blurry[:, :, :, :7]))
        self.assertTrue(torch.allclose(prediction[:, :, :, 7:], baseline[:, :, :, 7:]))

    def test_anchored_correction_starts_as_exact_scalar_gate(self) -> None:
        correction_gate = SpatialResidualCorrection(width=8, max_correction=0.15)
        blurry = torch.rand(2, 3, 23, 29)
        residual = torch.rand_like(blurry) - 0.5
        scalar_alpha = torch.tensor([0.7, 0.9])[:, None, None, None]
        correction = correction_gate(blurry, residual)
        anchored_alpha = apply_anchored_residual_correction(scalar_alpha, correction)
        self.assertTrue(torch.equal(correction, torch.zeros_like(correction)))
        self.assertTrue(torch.equal(anchored_alpha, scalar_alpha.expand_as(correction)))

    def test_anchored_correction_is_zero_mean_and_bounded(self) -> None:
        correction_gate = SpatialResidualCorrection(width=8, max_correction=0.15)
        torch.nn.init.normal_(correction_gate.head.weight, std=1.0)
        blurry = torch.rand(2, 3, 21, 27)
        residual = torch.rand_like(blurry) - 0.5
        correction = correction_gate(blurry, residual)
        self.assertTrue(torch.allclose(correction.mean(dim=(-2, -1)), torch.zeros(2, 1), atol=1e-7))
        self.assertLessEqual(float(correction.abs().max()), 0.15 + 1e-7)

    def test_anchored_correction_receives_finite_gradients(self) -> None:
        correction_gate = SpatialResidualCorrection(width=8)
        blurry = torch.rand(2, 3, 20, 24)
        baseline = torch.rand_like(blurry)
        sharp = torch.rand_like(blurry)
        scalar_alpha = torch.full((2, 1, 1, 1), 0.8)
        correction = correction_gate(blurry, baseline - blurry)
        alpha = apply_anchored_residual_correction(scalar_alpha, correction)
        apply_residual_gate(blurry, baseline, alpha).sub(sharp).square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in correction_gate.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in correction_gate.parameters()))

    def test_mse_oracle_has_expected_endpoints_and_midpoint(self) -> None:
        blurry = torch.zeros(3, 3, 8, 8)
        baseline = torch.ones_like(blurry)
        sharp = torch.stack(
            (
                torch.zeros(3, 8, 8),
                torch.full((3, 8, 8), 0.5),
                torch.ones(3, 8, 8),
            )
        )
        alpha = mse_optimal_residual_alpha(blurry, baseline, sharp).flatten()
        self.assertTrue(torch.allclose(alpha, torch.tensor([0.0, 0.5, 1.0])))

    def test_local_mse_oracle_recovers_two_spatial_strengths(self) -> None:
        blurry = torch.zeros(1, 3, 9, 10)
        baseline = torch.ones_like(blurry)
        sharp = torch.full_like(blurry, 0.25)
        sharp[:, :, :, 5:] = 0.75
        alpha = mse_optimal_residual_alpha_map(
            blurry,
            baseline,
            sharp,
            window_size=1,
        )
        self.assertEqual(tuple(alpha.shape), (1, 1, 9, 10))
        self.assertTrue(torch.allclose(alpha[:, :, :, :5], torch.full_like(alpha[:, :, :, :5], 0.25)))
        self.assertTrue(torch.allclose(alpha[:, :, :, 5:], torch.full_like(alpha[:, :, :, 5:], 0.75)))

    def test_local_mse_oracle_rejects_even_window(self) -> None:
        image = torch.zeros(1, 3, 8, 8)
        with self.assertRaises(ValueError):
            mse_optimal_residual_alpha_map(image, image, image, window_size=4)


if __name__ == "__main__":
    unittest.main()
