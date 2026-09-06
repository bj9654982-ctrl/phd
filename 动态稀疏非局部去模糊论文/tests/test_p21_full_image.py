import copy
import unittest

from scripts.evaluate_p21_full_image import (
    GROUPS, ORDER, ROLES, aggregate_quality, audit_manifest, benchmark_names,
    direction_decision, latency_statistics, paired_rows,
)


def manifest_fixture():
    rows = []
    for group, count in sorted(GROUPS.items()):
        for i in range(count):
            n = len(rows)
            rows.append({"name": f"{group}/{i:04d}", "recording_group": group,
                         "height": 720, "width": 1280,
                         "crop_proxy_class": "heavy" if n < 96 else "medium" if n < 202 else "light"})
    reference = {r["name"]: {"strength_class": r["crop_proxy_class"]} for r in rows}
    return rows, reference


def quality_fixture():
    manifest, _ = manifest_fixture()
    return {role: {r["name"]: {**r, "input_psnr_db": 25., "input_ssim": .8,
                               "candidate_psnr_db": score, "candidate_ssim": ssim}
                   for r in manifest}
            for role, score, ssim in (("local", 29., .85), ("C00", 29.04, .8502),
                                      ("C01", 29.07, .8505))}


class P21Tests(unittest.TestCase):
    def test_manifest_and_deterministic_midpoints(self):
        manifest, reference = manifest_fixture()
        audit_manifest(manifest, reference)
        self.assertEqual(benchmark_names(manifest),
                         ["GOPR0372_07/0087", "GOPR0378_13/0055", "GOPR0385_11/0050"])
        self.assertEqual(benchmark_names(list(reversed(manifest))), benchmark_names(manifest))

    def test_missing_duplicate_wrong_shapes_rejected(self):
        rows, reference = manifest_fixture()
        for bad in (rows[:-1], rows[:-1] + [rows[0]],
                    [{**rows[0], "height": 1080}] + rows[1:]):
            with self.assertRaises(ValueError):
                audit_manifest(bad, reference)

    def test_wrong_label_or_names_rejected(self):
        rows, reference = manifest_fixture()
        bad = copy.deepcopy(rows)
        bad[0]["crop_proxy_class"], bad[-1]["crop_proxy_class"] = (
            bad[-1]["crop_proxy_class"], bad[0]["crop_proxy_class"])
        with self.assertRaises(ValueError):
            audit_manifest(bad, reference)
        with self.assertRaises(ValueError):
            audit_manifest([{**rows[0], "name": "not_in_P20"}] + rows[1:], reference)

    def test_order_balanced_across_positions(self):
        for position in range(3):
            self.assertEqual(set(order[position] for order in ORDER), set(ROLES))
        self.assertTrue(all(set(order) == set(ROLES) for order in ORDER))

    def test_matched_quality_differences(self):
        result = aggregate_quality(quality_fixture())
        self.assertEqual(result["overall"]["pairs"], 385)
        self.assertAlmostEqual(result["overall"]["contrasts"]["C01_minus_local"]["psnr_db"], .07)
        self.assertAlmostEqual(result["overall"]["contrasts"]["C01_minus_C00"]["psnr_db"], .03)
        self.assertEqual({k: v["pairs"] for k, v in result["per_crop_proxy"].items()},
                         {"heavy": 96, "medium": 106, "light": 183})
        self.assertTrue(direction_decision(result)["overall_direction_continued"])

    def test_arithmetic_mean_of_per_image_db(self):
        rows = quality_fixture()
        name = next(iter(rows["local"]))
        rows["local"][name]["candidate_psnr_db"] = 39.
        result = aggregate_quality(rows)
        self.assertAlmostEqual(result["overall"]["scores"]["local"]["psnr_db"], 29 + 10 / 385)

    def test_all_per_image_contrasts_saved(self):
        result = paired_rows(quality_fixture())
        self.assertEqual(len(result), 385)
        self.assertAlmostEqual(result[0]["C01_minus_C00_psnr_db"], .03)
        self.assertAlmostEqual(result[0]["C01_minus_local_ssim"], .0005)

    def test_negative_subgroup_retained_without_inventing_gate(self):
        records = quality_fixture()
        for row in records["C01"].values():
            if row["crop_proxy_class"] == "medium":
                row["candidate_psnr_db"] = 29.02
        quality = aggregate_quality(records)
        self.assertLess(quality["per_crop_proxy"]["medium"]["contrasts"]["C01_minus_C00"]["psnr_db"], 0)
        self.assertTrue(direction_decision(quality)["overall_direction_continued"])

    def test_each_overall_negative_or_zero_psnr_blocks(self):
        quality = aggregate_quality(quality_fixture())
        for contrast in ("C01_minus_local", "C01_minus_C00"):
            for metric, value in (("psnr_db", -.01), ("psnr_db", 0), ("ssim", -.0001)):
                modified = copy.deepcopy(quality)
                modified["overall"]["contrasts"][contrast][metric] = value
                self.assertFalse(direction_decision(modified)["overall_direction_continued"])

    def test_mismatched_and_nonfinite_quality_rejected(self):
        for alteration in ("missing", "input_psnr_db", "candidate_ssim", "height"):
            records = quality_fixture()
            name = next(iter(records["C01"]))
            if alteration == "missing":
                del records["C01"][name]
            elif alteration == "input_psnr_db":
                records["C01"][name][alteration] = 26.
            elif alteration == "height":
                records["C01"][name][alteration] = 256
            else:
                records["C01"][name][alteration] = float("nan")
            with self.assertRaises(ValueError):
                aggregate_quality(records)

    def test_latency_includes_all_samples_and_rejects_invalid(self):
        result = latency_statistics([1, 2, 3, 100])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["mean"], 26.5)
        self.assertEqual(result["median"], 2.5)
        self.assertGreater(result["p95"], 80)
        for values in ([], [0], [-1], [float("nan")]):
            with self.assertRaises(ValueError):
                latency_statistics(values)


if __name__ == "__main__":
    unittest.main()
