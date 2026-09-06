from __future__ import annotations

import unittest

import torch

from src.models import CompactDeblurUNet, TrajectoryNonlocalBlock
from src.models.compact_unet import count_parameters


class TrajectoryNonlocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this research project")
        cls.device = torch.device("cuda:0")

    def test_geometry_ranges_and_backward(self) -> None:
        block = TrajectoryNonlocalBlock(32, samples=5, max_feature_length=7.0).to(self.device)
        features = torch.randn(2, 32, 15, 17, device=self.device, requires_grad=True)
        source = torch.rand(2, 3, 60, 68, device=self.device)
        restored, geometry = block(features, geometry_source=source, return_geometry=True)
        restored.square().mean().backward()
        self.assertEqual(restored.shape, features.shape)
        self.assertTrue(torch.allclose(geometry["double_angle"].norm(dim=1), torch.ones_like(geometry["length"][:, 0]), atol=1e-4))
        self.assertTrue(bool(((geometry["length"] >= 1.0) & (geometry["length"] <= 7.0)).all()))
        self.assertTrue(torch.allclose(geometry["attention"].sum(dim=1), torch.ones_like(geometry["confidence"][:, 0]), atol=1e-4))
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_parameter_overhead_is_below_five_percent(self) -> None:
        baseline = CompactDeblurUNet(width=24).to(self.device)
        augmented = CompactDeblurUNet(
            width=24,
            bottleneck_module=TrajectoryNonlocalBlock(96, samples=5),
        ).to(self.device)
        overhead = count_parameters(augmented) / count_parameters(baseline) - 1.0
        self.assertLess(overhead, 0.05)

    def test_amp_forward_and_backward(self) -> None:
        model = CompactDeblurUNet(
            width=16,
            bottleneck_module=TrajectoryNonlocalBlock(64, samples=5),
        ).to(self.device)
        blurry = torch.rand(1, 3, 65, 67, device=self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            restored = model(blurry)
            loss = restored.square().mean()
        loss.backward()
        self.assertEqual(restored.shape, blurry.shape)
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_block_starts_as_exact_identity(self) -> None:
        block = TrajectoryNonlocalBlock(32, samples=5).to(self.device)
        features = torch.randn(1, 32, 12, 13, device=self.device)
        with torch.no_grad():
            restored = block(features)
        self.assertTrue(torch.equal(restored, features))

    def test_oracle_geometry_override_is_respected(self) -> None:
        block = TrajectoryNonlocalBlock(32, samples=5).to(self.device)
        features = torch.randn(2, 32, 12, 13, device=self.device)
        direction = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device)
        length = torch.tensor([2.0, 6.0], device=self.device)
        restored, geometry = block(
            features,
            geometry_override={"direction": direction, "length": length},
            return_geometry=True,
        )
        self.assertTrue(torch.equal(restored, features))
        self.assertTrue(torch.allclose(geometry["direction"][:, :, 0, 0], direction))
        self.assertTrue(torch.allclose(geometry["length"][:, 0, 0, 0], length))
        self.assertTrue(torch.equal(geometry["confidence"], torch.ones_like(geometry["confidence"])))

    def test_model_accepts_oracle_geometry_override(self) -> None:
        model = CompactDeblurUNet(
            width=16,
            bottleneck_module=TrajectoryNonlocalBlock(64, samples=5, fusion_mode="detail"),
        ).to(self.device)
        blurry = torch.rand(2, 3, 65, 67, device=self.device)
        direction = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device)
        length = torch.tensor([2.0, 4.0], device=self.device)
        restored, geometry = model(
            blurry,
            return_geometry=True,
            geometry_override={"direction": direction, "length": length},
        )
        self.assertEqual(restored.shape, blurry.shape)
        self.assertTrue(torch.allclose(geometry["direction"][:, :, 0, 0], direction))

    def test_multiple_angle_hypotheses_expand_attention_support(self) -> None:
        block = TrajectoryNonlocalBlock(
            32,
            samples=5,
            angle_hypotheses=3,
            max_angle_offset_degrees=30.0,
        ).to(self.device)
        features = torch.randn(2, 32, 12, 13, device=self.device)
        _, geometry = block(features, return_geometry=True)
        self.assertEqual(geometry["attention"].shape, (2, 15, 12, 13))
        self.assertTrue(
            torch.allclose(
                geometry["attention"].sum(dim=1),
                torch.ones_like(geometry["confidence"][:, 0]),
                atol=1e-4,
            )
        )

    def test_wide_direction_residual_can_reverse_double_angle_prior(self) -> None:
        block = TrajectoryNonlocalBlock(
            32,
            samples=5,
            direction_residual_scale=2.0,
        ).to(self.device)
        with torch.no_grad():
            block.direction_residual.weight.zero_()
            block.direction_residual.bias.copy_(
                torch.tensor([-10.0, 0.0], device=self.device)
            )
        features = torch.randn(1, 32, 8, 8, device=self.device)
        _, geometry = block(features, geometry_source=None, return_geometry=True)
        self.assertLess(float(geometry["double_angle"][:, 0].mean()), -0.99)
        self.assertGreater(float(geometry["direction"][:, 1].abs().mean()), 0.99)


if __name__ == "__main__":
    unittest.main()
