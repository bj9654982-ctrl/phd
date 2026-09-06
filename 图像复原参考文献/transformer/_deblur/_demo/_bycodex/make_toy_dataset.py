from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clean toy images for an end-to-end smoke demo.")
    parser.add_argument("--output", default="data/toy_sharp")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def create_image(size: int, rng: random.Random) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.zeros((size, size, 3), dtype=np.uint8)
    colors = np.array([rng.randrange(30, 180) for _ in range(3)], dtype=np.float32)
    gradient = (0.35 + 0.65 * (xx + yy) / max(1, 2 * size - 2))[..., None]
    base[:] = np.clip(colors * gradient, 0, 255).astype(np.uint8)
    image = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(8, 18)):
        x0, y0 = rng.randrange(size), rng.randrange(size)
        x1, y1 = rng.randrange(x0, size), rng.randrange(y0, size)
        color = tuple(rng.randrange(30, 256) for _ in range(3))
        width = rng.randint(1, 4)
        if rng.random() < 0.5:
            draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
        else:
            draw.ellipse((x0, y0, x1, y1), outline=color, width=width)
    for _ in range(rng.randint(5, 12)):
        points = [(rng.randrange(size), rng.randrange(size)) for _ in range(rng.randint(2, 5))]
        draw.line(points, fill=tuple(rng.randrange(256) for _ in range(3)), width=rng.randint(1, 4))
    return image


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for index in range(args.count):
        create_image(args.size, rng).save(output / f"toy_{index:04d}.png")
    print(f"Generated {args.count} clean toy images in {output}")


if __name__ == "__main__":
    main()
