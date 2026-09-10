import unittest
import torch
from src.models.pooled_context import PooledContext

class ContextTests(unittest.TestCase):
    def test_zero_init_shapes_bf16(self):
        m=PooledContext(4,4).cuda()
        for h,w in ((1,1),(5,7),(16,16)):
            x=torch.randn(2,4,h,w,device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):y=m(x)
            torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_remote_influence_and_batch_isolation(self):
        m=PooledContext(1,1,2).cuda()
        with torch.no_grad():
            m.query.weight.zero_();m.key.weight.zero_()
            m.value.weight.fill_(1);m.output.weight.fill_(1)
        x=torch.zeros(2,1,4,4,device='cuda');x[0,0,3,3]=16
        y=m(x)
        self.assertAlmostEqual(y[0,0,0,0].item(),1.)
        self.assertEqual(y[1].abs().sum().item(),0.)

    def test_nonzero_projection_gradients(self):
        torch.manual_seed(7);m=PooledContext(4,4).cuda()
        torch.nn.init.normal_(m.output.weight,std=.01)
        x=torch.randn(2,4,5,7,device='cuda',requires_grad=True)
        m(x).square().mean().backward()
        for p in m.parameters():
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())

if __name__=='__main__':unittest.main()
