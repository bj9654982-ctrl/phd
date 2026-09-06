from __future__ import annotations

import unittest

import torch

from src.models import CompactDeblurUNet, MultiScaleNonlocalBlock
from src.models.compact_unet import count_parameters


class MultiScaleNonlocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this research project")
        cls.device = torch.device("cuda:0")

    def test_all_modes_start_as_exact_identity(self) -> None:
        features = torch.randn(1, 32, 13, 15, device=self.device)
        source = torch.rand(1, 3, 52, 60, device=self.device)
        for mode in (
            "fixed",
            "topk",
            "scale_adaptive",
            "gate_adaptive",
            "protected_gate_adaptive",
            "adaptive",
        ):
            block = MultiScaleNonlocalBlock(32, mode=mode).to(self.device)
            with torch.no_grad():
                restored = block(features, geometry_source=source)
            self.assertTrue(torch.equal(restored, features), msg=mode)

    def test_controls_have_identical_parameter_counts(self) -> None:
        counts = {
            mode: count_parameters(MultiScaleNonlocalBlock(96, mode=mode))
            for mode in (
                "fixed",
                "topk",
                "scale_adaptive",
                "gate_adaptive",
                "protected_gate_adaptive",
                "adaptive",
            )
        }
        self.assertEqual(len(set(counts.values())), 1, msg=str(counts))

    def test_attention_sparsity_and_diagnostic_ranges(self) -> None:
        features = torch.randn(2, 32, 11, 13, device=self.device)
        source = torch.rand(2, 3, 44, 52, device=self.device)
        for mode in (
            "fixed",
            "topk",
            "scale_adaptive",
            "gate_adaptive",
            "protected_gate_adaptive",
            "adaptive",
        ):
            block = MultiScaleNonlocalBlock(32, top_k=5, mode=mode).to(self.device)
            _, diagnostics = block(features, geometry_source=source, return_geometry=True)
            attention = diagnostics["attention"]
            selected = diagnostics["selected_mask"].sum(dim=1)
            expected = block.candidate_count if mode == "fixed" else 5
            self.assertTrue(torch.equal(selected, torch.full_like(selected, expected)), msg=mode)
            self.assertTrue(
                torch.allclose(
                    attention.sum(dim=1),
                    torch.ones_like(attention.sum(dim=1)),
                    atol=1e-4,
                ),
                msg=mode,
            )
            self.assertTrue(
                torch.allclose(
                    diagnostics["scale_weights"].sum(dim=1),
                    torch.ones_like(diagnostics["nonlocal_gate"][:, 0]),
                    atol=1e-4,
                ),
                msg=mode,
            )
            self.assertTrue(bool(((diagnostics["nonlocal_gate"] >= 0) & (diagnostics["nonlocal_gate"] <= 1)).all()))

    def test_protected_gate_can_close_the_entire_module_residual(self) -> None:
        features = torch.randn(1, 32, 11, 13, device=self.device)
        source = torch.rand(1, 3, 44, 52, device=self.device)
        block = MultiScaleNonlocalBlock(
            32,
            top_k=5,
            mode="protected_gate_adaptive",
        ).to(self.device)
        with torch.no_grad():
            block.output.weight.normal_(mean=0.0, std=1e-2)
            block.condition_head.bias[-1] = -20.0
            closed = block(features, geometry_source=source)
            block.condition_head.bias[-1] = 20.0
            opened = block(features, geometry_source=source)
        closed_change = float((closed - features).abs().max())
        opened_change = float((opened - features).abs().max())
        self.assertLess(closed_change, 1e-7)
        self.assertGreater(opened_change, closed_change + 1e-6)

    def test_factorial_modes_disable_only_the_intended_condition(self) -> None:
        features = torch.randn(1, 32, 11, 13, device=self.device)
        source = torch.rand(1, 3, 44, 52, device=self.device)
        diagnostics = {}
        for mode in ("scale_adaptive", "gate_adaptive"):
            block = MultiScaleNonlocalBlock(32, top_k=5, mode=mode).to(self.device)
            with torch.no_grad():
                block.condition_head.weight.normal_(mean=0.0, std=0.1)
                block.condition_head.bias.copy_(
                    torch.tensor((1.0, 0.0, -1.0, 1.0), device=self.device)
                )
                _, diagnostics[mode] = block(
                    features,
                    geometry_source=source,
                    return_geometry=True,
                )

        uniform = torch.full_like(diagnostics["gate_adaptive"]["scale_weights"], 1.0 / 3.0)
        fixed_gate = torch.full_like(diagnostics["scale_adaptive"]["nonlocal_gate"], 0.5)
        self.assertTrue(torch.equal(diagnostics["gate_adaptive"]["scale_weights"], uniform))
        self.assertTrue(torch.equal(diagnostics["scale_adaptive"]["nonlocal_gate"], fixed_gate))
        self.assertFalse(
            torch.allclose(diagnostics["scale_adaptive"]["scale_weights"], uniform)
        )
        self.assertFalse(
            torch.allclose(diagnostics["gate_adaptive"]["nonlocal_gate"], fixed_gate)
        )

    def test_parameter_overhead_is_below_five_percent(self) -> None:
        baseline = CompactDeblurUNet(width=24).to(self.device)
        augmented = CompactDeblurUNet(
            width=24,
            bottleneck_module=MultiScaleNonlocalBlock(96, mode="adaptive"),
        ).to(self.device)
        overhead = count_parameters(augmented) / count_parameters(baseline) - 1.0
        self.assertLess(overhead, 0.05)

    def test_amp_model_forward_and_backward_are_finite(self) -> None:
        model = CompactDeblurUNet(
            width=16,
            bottleneck_module=MultiScaleNonlocalBlock(64, top_k=5, mode="adaptive"),
        ).to(self.device)
        blurry = torch.rand(1, 3, 65, 67, device=self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            restored, diagnostics = model(blurry, return_geometry=True)
            loss = restored.square().mean()
        loss.backward()
        self.assertEqual(restored.shape, blurry.shape)
        self.assertEqual(diagnostics["attention"].shape[1], 24)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_adaptive_condition_head_receives_gradient_after_identity_is_opened(self) -> None:
        block = MultiScaleNonlocalBlock(32, top_k=5, mode="adaptive").to(self.device)
        with torch.no_grad():
            block.output.weight.normal_(mean=0.0, std=1e-3)
        features = torch.randn(2, 32, 11, 13, device=self.device, requires_grad=True)
        source = torch.rand(2, 3, 44, 52, device=self.device)
        restored = block(features, geometry_source=source)
        restored.square().mean().backward()
        gradient = block.condition_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
