from __future__ import annotations

import unittest

import torch

from src.data import make_debug_batch
from src.losses import restoration_loss
from src.models import CompactDeblurUNet
from src.models.compact_unet import count_parameters


class CompactBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this research project")
        cls.device = torch.device("cuda:0")

    def test_forward_preserves_odd_image_size(self) -> None:
        model = CompactDeblurUNet(width=8).to(self.device).eval()
        image = torch.rand(1, 3, 65, 67, device=self.device)
        with torch.no_grad():
            restored = model(image)
        self.assertEqual(restored.shape, image.shape)

    def test_model_is_compact(self) -> None:
        model = CompactDeblurUNet(width=16)
        parameters = count_parameters(model)
        self.assertGreater(parameters, 100_000)
        self.assertLess(parameters, 5_000_000)

    def test_amp_backward_is_finite(self) -> None:
        blurry, sharp = make_debug_batch(batch_size=1, size=32)
        model = CompactDeblurUNet(width=8).to(self.device).train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(blurry)
            loss = restoration_loss(prediction, sharp)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()

