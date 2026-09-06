from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import discover_gopro_pairs  # noqa: E402


EXPECTED_COUNTS = {"train": 2103, "test": 1111}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a GOPRO_Large extraction and write a pair manifest.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "GoPro",
        help="Directory containing train/ and test/ (default: project data/raw/GoPro).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "datasets" / "gopro",
        help="Directory for audit.json and pairs.csv.",
    )
    parser.add_argument(
        "--check-dimensions",
        action="store_true",
        help="Open every image to verify paired dimensions; slower but recommended once.",
    )
    parser.add_argument(
        "--strict-counts",
        action="store_true",
        help="Exit with an error unless official counts (2103/1111) are present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pairs, issues = discover_gopro_pairs(root)

    dimension_mismatches: list[dict[str, object]] = []
    if args.check_dimensions:
        for pair in pairs:
            with Image.open(pair.blurry) as blur_image, Image.open(pair.sharp) as sharp_image:
                if blur_image.size != sharp_image.size:
                    dimension_mismatches.append(
                        {
                            "split": pair.split,
                            "scene": pair.scene,
                            "name": pair.name,
                            "blur_size": list(blur_image.size),
                            "sharp_size": list(sharp_image.size),
                        }
                    )

    counts = Counter(pair.split for pair in pairs)
    scenes = {split: len({pair.scene for pair in pairs if pair.split == split}) for split in EXPECTED_COUNTS}
    count_matches = {split: counts[split] == expected for split, expected in EXPECTED_COUNTS.items()}
    passed = not issues and not dimension_mismatches and all(count_matches.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "scene", "name", "blurry", "sharp"])
        for pair in pairs:
            writer.writerow([pair.split, pair.scene, pair.name, pair.blurry, pair.sharp])

    report = {
        "root": str(root),
        "expected_pairs": EXPECTED_COUNTS,
        "found_pairs": dict(counts),
        "scene_counts": scenes,
        "count_matches": count_matches,
        "pairing_issues": [issue.__dict__ for issue in issues],
        "dimension_mismatches": dimension_mismatches,
        "dimension_check_performed": args.check_dimensions,
        "passed": passed,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Manifest: {output_dir / 'pairs.csv'}")
    print(f"Audit:    {output_dir / 'audit.json'}")

    if issues or dimension_mismatches:
        raise SystemExit(1)
    if args.strict_counts and not all(count_matches.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
