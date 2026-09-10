"""Synthetic CPU correctness checks; no dataset, checkpoint or optimizer."""
import unittest
import torch
from torch.nn import functional as F
from src.reference_nlfe import ReferenceNLFE


class OperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7400)
        torch.set_num_threads(2)

    def test_independent_window_and_head_loop(self):
        m = ReferenceNLFE(4, heads=2, window=4, pool=2).double()
        x = torch.randn(2, 4, 8, 12, dtype=torch.float64, requires_grad=True)
        expected = torch.zeros_like(x)
        for b in range(2):
            for i in range(0, 8, 4):
                for j in range(0, 12, 4):
                    tile = x[b:b+1, :, i:i+4, j:j+4]
                    q, k, v = m.qkv(tile).chunk(3, 1)
                    # Explicit per-cell maximum, independent of pooling API.
                    def pool(t):
                        return torch.stack([t[:, :, r:r+2, c:c+2].flatten(2).max(-1).values
                                            for r in (0, 2) for c in (0, 2)], -1)
                    k, v = pool(k), pool(v)
                    out = []
                    for head in range(2):
                        sl = slice(head * 2, head * 2 + 2)
                        qh = q[:, sl].flatten(2).transpose(1, 2)
                        kh, vh = k[:, sl], v[:, sl].transpose(1, 2)
                        out.append(((qh @ kh / 2**.5).softmax(-1) @ vh).transpose(1, 2))
                    y = tile + m.proj(torch.cat(out, 1).reshape(1, 4, 4, 4))
                    expected[b:b+1, :, i:i+4, j:j+4] = y + m.ffn(y)
        actual = m(x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in m.parameters()))

    def test_identity_and_invalid_sizes(self):
        m = ReferenceNLFE(4, window=4, pool=2)
        with torch.no_grad():
            m.proj.weight.zero_(); m.proj.bias.zero_()
            m.ffn[-1].weight.zero_(); m.ffn[-1].bias.zero_()
        x = torch.randn(1, 4, 8, 12)
        torch.testing.assert_close(m(x), x, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            m(torch.randn(1, 4, 5, 8))
        with self.assertRaises(ValueError):
            ReferenceNLFE(3, heads=2)

    def test_full_pool_is_broadcast_per_head(self):
        # One key per window => softmax is exactly one, independent of Q.
        m = ReferenceNLFE(2, window=4, pool=4)
        x = torch.randn(1, 2, 4, 4)
        _, _, v = m.qkv(x).chunk(3, 1)
        y = x + m.proj(F.max_pool2d(v, 4).expand(-1, -1, 4, 4))
        torch.testing.assert_close(m(x), y + m.ffn(y))


if __name__ == '__main__':
    unittest.main()
