from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_p9_benefit_router import _design_apply, _design_fit, _gains


class BenefitRouterTests(unittest.TestCase):
    @staticmethod
    def _records() -> list[dict[str, object]]:
        return [
            {
                "features": np.asarray([1.0, 2.0]),
                "optimal_gate": 0.5,
                "baseline_mse": 0.04,
                "cross": -0.02,
                "delta_square": 0.04,
                "candidate_mse": 0.04,
                "recording_group": "g0",
                "strength_class": "medium",
            },
            {
                "features": np.asarray([2.0, 4.0]),
                "optimal_gate": 0.0,
                "baseline_mse": 0.01,
                "cross": 0.01,
                "delta_square": 0.02,
                "candidate_mse": 0.05,
                "recording_group": "g1",
                "strength_class": "light",
            },
        ]

    def test_zero_gate_is_exact_baseline(self) -> None:
        baseline, final, gain = _gains(self._records(), np.zeros(2))
        np.testing.assert_allclose(final, baseline, atol=1e-12)
        np.testing.assert_allclose(gain, np.zeros(2), atol=1e-12)

    def test_known_continuous_optimum_improves_first_record(self) -> None:
        baseline, final, gain = _gains(self._records(), np.asarray([0.5, 0.0]))
        self.assertGreater(final[0], baseline[0])
        self.assertGreater(gain[0], 0.0)
        self.assertEqual(gain[1], 0.0)

    def test_quadratic_design_can_be_reapplied(self) -> None:
        features = np.asarray([[1.0, 2.0], [2.0, 4.0], [4.0, 3.0]])
        fitted, transform = _design_fit(features, degree=2)
        reapplied = _design_apply(features, transform)
        np.testing.assert_allclose(reapplied, fitted, atol=1e-12)
        self.assertEqual(fitted.shape, (3, 5))


if __name__ == "__main__":
    unittest.main()
