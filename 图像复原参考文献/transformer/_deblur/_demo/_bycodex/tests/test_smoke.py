import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from deblur_transformer.data import gaussian_kernel, motion_kernel, random_blur
from deblur_transformer.io_utils import load_image, save_image
from deblur_transformer.model import DeblurTransformer


class DeblurDemoSmokeTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_model_preserves_arbitrary_image_shape(self) -> None:
        model = DeblurTransformer(embed_dim=8, depths=(1, 1, 1), heads=(1, 2, 4)).cuda()
        image = torch.rand(1, 3, 37, 45, device="cuda")
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            output = model(image)
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_blur_kernels_are_normalized(self) -> None:
        self.assertAlmostEqual(float(gaussian_kernel(7, 1.5).sum()), 1.0, places=5)
        self.assertAlmostEqual(float(motion_kernel(9, 31.0).sum()), 1.0, places=5)
        blurred = random_blur(torch.rand(3, 32, 32), noise_std=0.0)
        self.assertEqual(tuple(blurred.shape), (3, 32, 32))

    def test_image_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.png"
            output = Path(temp_dir) / "output.png"
            Image.new("RGB", (19, 23), (10, 20, 30)).save(source)
            tensor = load_image(source)
            save_image(tensor, output)
            self.assertTrue(output.is_file())
            with Image.open(output) as restored:
                self.assertEqual(restored.size, (19, 23))


if __name__ == "__main__":
    unittest.main()
