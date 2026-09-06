import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts.train_p12_matched_nonlocal import (
    assign_teacher_targets, compute_gate_supervision_loss, compute_training_objective,
)
from src.losses import restoration_loss
from src.models import input_psnr_per_image, teacher_blur_strength


class P19TeacherTests(unittest.TestCase):
    def test_paired_is_identity_and_does_not_draw_rng(self):
        values = torch.arange(6, device="cuda", dtype=torch.float32)
        rng = torch.Generator().manual_seed(22407)
        state = rng.get_state().clone()
        assigned, shift = assign_teacher_targets(values, "paired", rng)
        self.assertIs(assigned, values)
        self.assertEqual(shift, 0)
        self.assertTrue(torch.equal(state, rng.get_state()))

    def test_cyclic_shuffle_preserves_values_and_has_no_fixed_positions(self):
        values = torch.arange(6, device="cuda", dtype=torch.float32)
        rng = torch.Generator().manual_seed(22407)
        shifts = set()
        for _ in range(50):
            assigned, shift = assign_teacher_targets(values, "shuffled", rng)
            shifts.add(shift)
            self.assertTrue(torch.equal(assigned.sort().values, values))
            self.assertTrue(bool((assigned != values).all()))
            self.assertTrue(torch.equal(assigned, values.roll(-shift)))
        self.assertEqual(shifts, {1, 2, 3, 4, 5})

    def test_private_generator_reproducible_and_global_rng_untouched(self):
        values = torch.arange(6, device="cuda", dtype=torch.float32)
        torch.manual_seed(17)
        cpu_state, cuda_state = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        def sequence():
            rng = torch.Generator().manual_seed(26813)
            return [assign_teacher_targets(values, "shuffled", rng)[1] for _ in range(30)]
        self.assertEqual(sequence(), sequence())
        self.assertTrue(torch.equal(cpu_state, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_state, torch.cuda.get_rng_state()))

    def test_invalid_shuffle_rejected(self):
        values = torch.zeros(6, device="cuda")
        with self.assertRaises(ValueError):
            assign_teacher_targets(values, "unknown")
        with self.assertRaises(ValueError):
            assign_teacher_targets(values, "shuffled")
        with self.assertRaises(ValueError):
            assign_teacher_targets(values[:1], "shuffled", torch.Generator())

    def test_paired_loss_and_gradient_are_exactly_historical(self):
        gate = torch.linspace(.1, .9, 6, device="cuda").reshape(6, 1, 1, 1).requires_grad_()
        blurry = torch.zeros(6, 3, 12, 16, device="cuda")
        sharp = torch.linspace(0, .12, 6, device="cuda").reshape(6, 1, 1, 1).expand_as(blurry)
        old = F.mse_loss(gate.float().mean((1, 2, 3)),
                        teacher_blur_strength(input_psnr_per_image(blurry, sharp), 24.02, 28.30))
        new = compute_gate_supervision_loss({"nonlocal_gate": gate}, blurry, sharp, 24.02, 28.30)
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(torch.autograd.grad(old, gate)[0],
                                    torch.autograd.grad(new, gate)[0]))

    def test_diagnostics_record_collapsed_targets_honestly(self):
        blurry = torch.zeros(6, 3, 12, 16, device="cuda")
        gate = torch.full((6, 1, 3, 4), .5, device="cuda", requires_grad=True)
        diagnostics = {}
        loss = compute_gate_supervision_loss(
            {"nonlocal_gate": gate}, blurry, blurry, 24.02, 28.30,
            teacher_mode="shuffled", teacher_generator=torch.Generator().manual_seed(22407),
            diagnostics=diagnostics,
        )
        self.assertEqual(diagnostics["target_changed_fraction"], 0)
        self.assertEqual(diagnostics["teacher_unique_values"], 1)
        self.assertEqual(diagnostics["teacher_endpoint_fraction"], 1)
        self.assertEqual(float(loss), .25)
        loss.backward()
        self.assertGreater(float(gate.grad.abs().sum()), 0)

    def test_zero_weight_does_not_compute_teacher_or_request_geometry(self):
        class Identity(torch.nn.Module):
            def forward(self, blurry):
                return blurry
        blurry = torch.zeros(6, 3, 12, 16, device="cuda")
        sharp = torch.ones_like(blurry)
        rng = torch.Generator().manual_seed(22407)
        state = rng.get_state().clone()
        with patch("scripts.train_p12_matched_nonlocal.compute_gate_supervision_loss",
                   side_effect=AssertionError("must not be called")):
            loss, restoration, auxiliary = compute_training_objective(
                Identity(), blurry, sharp, 24.02, 28.30, .1, 0,
                teacher_mode="shuffled", teacher_generator=rng,
            )
        self.assertTrue(torch.equal(loss, restoration_loss(blurry, sharp, .1)))
        self.assertTrue(torch.equal(restoration, loss))
        self.assertEqual(float(auxiliary), 0)
        self.assertTrue(torch.equal(state, rng.get_state()))

    def test_shuffling_changes_only_auxiliary_target_not_restoration(self):
        class ToyModel(torch.nn.Module):
            def forward(self, blurry, return_geometry=False):
                output = blurry + .01
                gate = torch.linspace(.1, .9, 6, device="cuda").reshape(6, 1, 1, 1)
                return (output, {"nonlocal_gate": gate}) if return_geometry else output
        blurry = torch.zeros(6, 3, 12, 16, device="cuda")
        sharp = torch.linspace(0, .12, 6, device="cuda").reshape(6, 1, 1, 1).expand_as(blurry).clone()
        before = sharp.clone()
        paired = compute_training_objective(ToyModel(), blurry, sharp, 24.02, 28.30, .1, .01)
        shuffled = compute_training_objective(
            ToyModel(), blurry, sharp, 24.02, 28.30, .1, .01, teacher_mode="shuffled",
            teacher_generator=torch.Generator().manual_seed(22407))
        self.assertTrue(torch.equal(paired[1], shuffled[1]))
        self.assertFalse(torch.equal(paired[2], shuffled[2]))
        self.assertTrue(torch.equal(sharp, before))


if __name__ == "__main__":
    unittest.main()
