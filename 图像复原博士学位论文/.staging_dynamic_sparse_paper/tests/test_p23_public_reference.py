import unittest
import torch
from scripts.evaluate_p23_public_reference import pad_native


class PublicReferenceTests(unittest.TestCase):
    def test_native_multiple_of_eight_unchanged(self):
        image = torch.arange(3*16*24).reshape(1, 3, 16, 24).float()
        padded, h, w = pad_native(image)
        self.assertEqual((h, w), (16, 24))
        self.assertTrue(torch.equal(image, padded))

    def test_reflect_right_bottom_and_unpad(self):
        image = torch.arange(3*17*25).reshape(1, 3, 17, 25).float()
        padded, h, w = pad_native(image)
        self.assertEqual(tuple(padded.shape), (1, 3, 24, 32))
        self.assertTrue(torch.equal(padded[..., :h, :w], image))
        self.assertTrue(torch.equal(padded[..., 0, 25], image[..., 0, 23]))
