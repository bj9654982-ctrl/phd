"""Hand-computable CUDA checks against the unchanged production operator."""
import unittest
import torch
from src.models.reciprocal_nonlocal import ReciprocalNonlocal


class SemanticsTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.fail('CUDA is required for this experiment check')
        self.x = torch.eye(4, device='cuda').reshape(1, 4, 2, 2)
        self.m = ReciprocalNonlocal(4, 4, 'reciprocal', topk=6).cuda()
        # Rows are query locations, columns are key locations.
        # Best targets: 0->1, 1->3, 2->0, 3->1.
        scores = torch.tensor([[0.,3.,2.,1.], [1.,0.,2.,3.],
                               [3.,2.,0.,1.], [1.,3.,2.,0.]], device='cuda')
        with torch.no_grad():
            self.m.query.weight.copy_(scores.T.reshape(4,4,1,1))
            for layer in (self.m.key, self.m.value, self.m.output):
                layer.weight.copy_(torch.eye(4,device='cuda').reshape(4,4,1,1))
        # Cosine logits: every query row has norm sqrt(14).
        self.second_weight = torch.softmax(torch.tensor([3.,2.,1.],device='cuda') / 14**.5, 0)[1]

    def test_forward_second_reverse_first_is_accepted(self):
        y, stats = self.m(self.x, True)
        expected = self.x[0,:,0,0] + self.second_weight * (self.x[0,:,1,0]-self.x[0,:,0,0])
        torch.testing.assert_close(y[0,:,0,0], expected)
        torch.testing.assert_close(stats['accepted_mass'][0,0], self.second_weight)

    def test_rejected_mass_is_not_renormalized(self):
        _, stats = self.m(self.x, True)
        self.assertGreater(stats['accepted_mass'][0,0].item(), 0)
        self.assertLess(stats['accepted_mass'][0,0].item(), 1)
        self.m.mode = 'content'
        _, content = self.m(self.x, True)
        torch.testing.assert_close(content['accepted_mass'], torch.ones_like(content['accepted_mass']))

    def test_valid_neighbors_but_no_return_is_identity(self):
        y, stats = self.m(self.x, True)
        self.assertEqual(stats['valid'][2].sum().item(), 3)
        self.assertEqual(stats['accepted_mass'][0,2].item(), 0)
        torch.testing.assert_close(y[0,:,1,0], self.x[0,:,1,0], rtol=0, atol=0)

if __name__ == '__main__':
    unittest.main()
