import unittest
import numpy as np
from scripts.visualize_p21_restoration import detail_boxes, select_cases


class VisualTests(unittest.TestCase):
    def test_fixed_and_both_extremes_retained(self):
        rows = [{"name": str(i), "C01_minus_C00_psnr_db": i-192} for i in range(385)]
        selected = select_cases(rows, ["10", "20", "30"])
        self.assertEqual([r["name"] for r in selected], ["10", "20", "30", "0", "384"])

    def test_boxes_use_only_input_and_match_fixed_center(self):
        array = np.zeros((720, 1280, 3), dtype=np.uint8)
        boxes = detail_boxes(array)
        self.assertEqual(boxes, [(0, 0, 192, 192), (544, 264, 192, 192)])
        for x, y, w, h in boxes:
            self.assertLessEqual(x+w, 1280)
            self.assertLessEqual(y+h, 720)

    def test_incomplete_table_rejected(self):
        with self.assertRaises(ValueError):
            select_cases([], [])
