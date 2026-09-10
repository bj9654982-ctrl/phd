import unittest
from scripts.pilot_p24_restormer import timing_summary


class PilotTests(unittest.TestCase):
    def test_timing_units(self):
        result = timing_summary([1., 3.])
        self.assertEqual(result['mean_step_seconds'], 2.)
        self.assertAlmostEqual(result['same_settings_30k_hours'], 60000/3600)

    def test_invalid_timing(self):
        for values in ([], [0.], [-1.]):
            with self.assertRaises(ValueError):
                timing_summary(values)
