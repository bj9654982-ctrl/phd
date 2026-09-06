from __future__ import annotations

import unittest

import torch

from scripts.evaluate_p11_uncertainty_mixture import uncertainty_weights


class UncertaintyMixtureTests(unittest.TestCase):
    def test_extreme_gates_select_the_corresponding_specialist(self) -> None:
        gate = torch.tensor([0.0, 1.0])[:, None, None, None]
        light, baseline, heavy = uncertainty_weights(gate)
        torch.testing.assert_close(light.flatten(), torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(baseline.flatten(), torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(heavy.flatten(), torch.tensor([0.0, 1.0]))

    def test_ambiguous_gate_falls_back_to_baseline(self) -> None:
        light, baseline, heavy = uncertainty_weights(torch.full((1, 1, 1, 1), 0.5))
        self.assertEqual(float(light), 0.0)
        self.assertEqual(float(heavy), 0.0)
        self.assertEqual(float(baseline), 1.0)

    def test_weights_are_nonnegative_and_sum_to_one(self) -> None:
        gate = torch.linspace(0.0, 1.0, 101)[:, None, None, None]
        weights = uncertainty_weights(gate)
        self.assertTrue(all(torch.all(weight >= 0.0) for weight in weights))
        torch.testing.assert_close(sum(weights), torch.ones_like(gate))


if __name__ == "__main__":
    unittest.main()
