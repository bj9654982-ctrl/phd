from __future__ import annotations

import unittest

import torch

from src.data.synthetic_motion import line_motion_blur
from src.models.motion_geometry import (
    GeometryProbe,
    dense_paired_geometry_supervision_loss,
    geometry_supervision_loss,
    paired_direction_supervision_loss,
    paired_motion_geometry_pseudo_label,
    paired_motion_direction_pseudo_label,
)


class MotionGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this research project")
        cls.device = torch.device("cuda:0")

    def test_line_blur_preserves_constant_image(self) -> None:
        image = torch.full((2, 3, 32, 32), 0.4, device=self.device)
        angles = torch.tensor([0.0, 1.2], device=self.device)
        lengths = torch.tensor([5.0, 21.0], device=self.device)
        blurred = line_motion_blur(image, angles, lengths)
        self.assertTrue(torch.allclose(blurred, image, atol=2e-5))

    def test_geometry_probe_constraints_and_backward(self) -> None:
        model = GeometryProbe(width=8).to(self.device)
        blurry = torch.rand(2, 3, 32, 32, device=self.device)
        target_direction = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device)
        target_length = torch.tensor([5.0, 17.0], device=self.device)
        direction, length, confidence = model(blurry)
        loss, _ = geometry_supervision_loss(
            direction,
            length,
            confidence,
            target_direction,
            target_length,
            max_length=25.0,
        )
        loss.backward()
        self.assertEqual(tuple(direction.shape[-2:]), (1, 1))
        self.assertTrue(torch.allclose(direction.norm(dim=1), torch.ones_like(length[:, 0]), atol=1e-4))
        self.assertTrue(bool(((length >= 1.0) & (length <= 25.0)).all()))
        self.assertTrue(bool(((confidence >= 0.0) & (confidence <= 1.0)).all()))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_paired_direction_pseudo_label_shape_and_range(self) -> None:
        sharp = torch.rand(2, 3, 64, 64, device=self.device)
        angles = torch.tensor([0.0, 1.2], device=self.device)
        lengths = torch.tensor([9.0, 17.0], device=self.device)
        blurry = line_motion_blur(sharp, angles, lengths)
        direction, confidence = paired_motion_direction_pseudo_label(blurry, sharp, (16, 16), window=31)
        self.assertEqual(direction.shape, (2, 2, 16, 16))
        self.assertEqual(confidence.shape, (2, 1, 16, 16))
        self.assertTrue(torch.allclose(direction.norm(dim=1), torch.ones_like(confidence[:, 0]), atol=1e-4))
        self.assertTrue(bool(((confidence >= 0.0) & (confidence <= 1.0)).all()))
        loss = paired_direction_supervision_loss(direction, direction, confidence)
        self.assertLess(float(loss), 1e-6)

    def test_paired_geometry_pseudo_label_shape_range_and_blur_order(self) -> None:
        sharp = torch.rand(2, 3, 96, 96, device=self.device)
        angles = torch.tensor([0.2, 1.0], device=self.device)
        input_lengths = torch.tensor([5.0, 21.0], device=self.device)
        blurry = line_motion_blur(sharp, angles, input_lengths)
        direction, length, confidence = paired_motion_geometry_pseudo_label(
            blurry,
            sharp,
            output_size=(24, 24),
            window=31,
            max_feature_length=7.0,
        )
        self.assertEqual(direction.shape, (2, 2, 24, 24))
        self.assertEqual(length.shape, (2, 1, 24, 24))
        self.assertEqual(confidence.shape, (2, 1, 24, 24))
        self.assertTrue(torch.allclose(direction.norm(dim=1), torch.ones_like(length[:, 0]), atol=1e-4))
        self.assertTrue(bool(((length >= 1.0) & (length <= 7.0)).all()))
        self.assertTrue(bool(((confidence >= 0.0) & (confidence <= 1.0)).all()))
        self.assertGreater(float(length[1].mean()), float(length[0].mean()))

    def test_dense_paired_geometry_loss_zero_target_and_backward(self) -> None:
        target_direction = torch.randn(2, 2, 8, 8, device=self.device)
        target_direction = torch.nn.functional.normalize(target_direction, dim=1)
        target_double_angle = torch.cat(
            (
                target_direction[:, :1].square() - target_direction[:, 1:2].square(),
                2.0 * target_direction[:, :1] * target_direction[:, 1:2],
            ),
            dim=1,
        )
        target_length = 1.0 + 6.0 * torch.rand(2, 1, 8, 8, device=self.device)
        target_confidence = torch.rand(2, 1, 8, 8, device=self.device)
        exact_loss, _ = dense_paired_geometry_supervision_loss(
            target_double_angle,
            target_length,
            target_confidence,
            target_direction,
            target_length,
            target_confidence,
            max_feature_length=7.0,
        )
        self.assertLess(float(exact_loss), 1e-6)

        predicted_direction = torch.randn(
            2,
            2,
            8,
            8,
            device=self.device,
            requires_grad=True,
        )
        predicted_length = torch.rand(
            2,
            1,
            8,
            8,
            device=self.device,
            requires_grad=True,
        )
        predicted_confidence = torch.rand(
            2,
            1,
            8,
            8,
            device=self.device,
            requires_grad=True,
        )
        loss, components = dense_paired_geometry_supervision_loss(
            predicted_direction,
            predicted_length,
            predicted_confidence,
            target_direction,
            target_length,
            target_confidence,
            max_feature_length=7.0,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in components.values()))
        self.assertTrue(torch.isfinite(predicted_direction.grad).all())
        self.assertTrue(torch.isfinite(predicted_length.grad).all())
        self.assertTrue(torch.isfinite(predicted_confidence.grad).all())


if __name__ == "__main__":
    unittest.main()
