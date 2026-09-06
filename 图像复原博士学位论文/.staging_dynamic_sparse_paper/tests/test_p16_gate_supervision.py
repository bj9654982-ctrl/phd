import unittest

import torch
from torch.nn import functional as F

from scripts.train_p12_matched_nonlocal import (
    compute_gate_supervision_loss,
    validate_gate_supervision,
)
from src.models import MultiScaleNonlocalBlock, input_psnr_per_image, teacher_blur_strength


class P16GateSupervisionTests(unittest.TestCase):
    def test_four_cells_are_valid_but_unlearned_gates_cannot_be_supervised(self):
        for variant in ("gate_adaptive", "protected_gate_adaptive"):
            for weight in (0.0, 0.01):
                validate_gate_supervision(variant, weight)
        for variant in ("local", "fixed", "topk", "scale_adaptive"):
            with self.assertRaises(ValueError):
                validate_gate_supervision(variant, 0.01)
        with self.assertRaises(ValueError):
            validate_gate_supervision("gate_adaptive", -0.01)

    def test_teacher_only_has_gradient_and_combined_preserves_p15_loss(self):
        if not torch.cuda.is_available():
            self.fail("CUDA is required")
        torch.manual_seed(3407)
        blurry = torch.rand(2, 3, 44, 52, device="cuda")
        sharp = blurry.clone()  # Light endpoint, teacher=0, avoids zero loss at g=.5.
        features = torch.randn(2, 32, 11, 13, device="cuda")
        for mode in ("gate_adaptive", "protected_gate_adaptive"):
            block = MultiScaleNonlocalBlock(32, mode=mode).cuda()
            _, geometry = block(features, geometry_source=blurry, return_geometry=True)
            loss = compute_gate_supervision_loss(geometry, blurry, sharp, 24.02, 28.30)
            if mode == "gate_adaptive":
                self.assertTrue(torch.equal(geometry["intervention_gate"],
                                            torch.ones_like(geometry["intervention_gate"])))
                self.assertFalse(geometry["intervention_gate"].requires_grad)
            else:
                old_loss = F.mse_loss(
                    geometry["intervention_gate"].float().mean((1, 2, 3)),
                    teacher_blur_strength(input_psnr_per_image(blurry, sharp), 24.02, 28.30),
                )
                self.assertTrue(torch.equal(loss, old_loss))
            loss.backward()
            gradient = block.condition_head.bias.grad[-1]
            self.assertTrue(torch.isfinite(gradient))
            self.assertGreater(float(gradient.abs()), 0.0)


if __name__ == "__main__":
    unittest.main()
