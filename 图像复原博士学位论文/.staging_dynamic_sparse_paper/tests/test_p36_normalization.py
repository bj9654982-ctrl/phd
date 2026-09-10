import unittest
import torch
from tests.test_p33_semantics import SemanticsTests
from src.models.normalized_reciprocal_nonlocal import NormalizedReciprocalNonlocal

class NormalizationTests(unittest.TestCase):
    def setUp(self):
        fixture=SemanticsTests();fixture.setUp()
        self.x=fixture.x;self.old=fixture.m
        self.m=NormalizedReciprocalNonlocal(4,4).cuda()
        self.m.load_state_dict(self.old.state_dict())

    def test_normalized_mass_and_empty_identity(self):
        y,s=self.m(self.x,True)
        torch.testing.assert_close(s['accepted_mass'],torch.tensor([[1.,1.,0.,1.]],device='cuda'))
        torch.testing.assert_close(y[0,:,1,0],self.x[0,:,1,0],rtol=0,atol=0)

    def test_single_edge_amplifies_even_if_semantically_wrong(self):
        # No semantic labels enter either rule: declare edge 0->2 wrong externally.
        y,_=self.m(self.x,True);old,_=self.old(self.x,True)
        torch.testing.assert_close(y[0,:,0,0],self.x[0,:,1,0])
        self.assertGreater((y-self.x)[0,:,0,0].norm().item(),(old-self.x)[0,:,0,0].norm().item())

    def test_batch_and_finite_gradients(self):
        torch.manual_seed(4)
        x=torch.randn(2,4,5,6,device='cuda',requires_grad=True)
        y=self.m(x)
        torch.testing.assert_close(y[:1],self.m(x[:1]))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for p in self.m.parameters():
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())

    def test_zero_initialization_bf16(self):
        torch.nn.init.zeros_(self.m.output.weight)
        with torch.autocast('cuda',dtype=torch.bfloat16):y=self.m(self.x)
        torch.testing.assert_close(y,self.x,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
