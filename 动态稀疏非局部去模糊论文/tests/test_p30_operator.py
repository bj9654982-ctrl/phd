import unittest
import torch
from src.models.reciprocal_nonlocal import ReciprocalNonlocal

class OperatorTests(unittest.TestCase):
    def test_zero_init(self):
        x=torch.randn(2,8,5,6)
        for mode in ('fixed','content','reciprocal'):
            m=ReciprocalNonlocal(8,4,mode)
            self.assertTrue(torch.equal(m(x),x))

    def test_no_neighbor(self):
        m=ReciprocalNonlocal(8,4)
        torch.nn.init.normal_(m.output.weight)
        x=torch.randn(2,8,1,1,requires_grad=True)
        y,stats=m(x,True)
        self.assertTrue(torch.equal(x,y));self.assertEqual(stats['accepted_mass'].sum(),0)
        y.sum().backward();self.assertTrue(torch.isfinite(x.grad).all())

    def test_boundary(self):
        m=ReciprocalNonlocal(8,4)
        idx,valid=m.candidates(3,3,'cpu')
        self.assertEqual(valid[0].sum(),3)
        self.assertEqual(set(idx[0][valid[0]].tolist()),{1,3,4})

    def test_gradients_and_batches(self):
        for mode in ('fixed','content','reciprocal'):
            torch.manual_seed(7)
            m=ReciprocalNonlocal(8,4,mode)
            torch.nn.init.normal_(m.output.weight,std=.01)
            x=torch.randn(2,8,5,6,requires_grad=True)
            y=m(x)
            self.assertTrue(torch.allclose(y[:1],m(x[:1]),atol=1e-6))
            y.square().mean().backward()
            for p in m.parameters():
                self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())

    def test_constant_features(self):
        m=ReciprocalNonlocal(8,4)
        torch.nn.init.normal_(m.output.weight)
        x=torch.ones(1,8,4,4)
        self.assertTrue(torch.equal(m(x),x))

if __name__=='__main__':unittest.main()
