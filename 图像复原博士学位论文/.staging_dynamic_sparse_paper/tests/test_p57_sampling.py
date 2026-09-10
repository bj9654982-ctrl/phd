import unittest
import numpy as np
import torch
from src.dpdd_sampling import choose_crop
from src.dpdd import paired_crop

class SamplingTests(unittest.TestCase):
    def test_identity_when_all_kept(self):
        r={'coordinates':[[0,0],[204,204],[512,512]],'kept':[0,1,2]}
        for seed in range(100):
            self.assertEqual(choose_crop(r,seed,False),choose_crop(r,seed,True))
    def test_membership_reproducible(self):
        r={'coordinates':[[0,0],[204,204],[512,512]],'kept':[1]}
        for seed in range(100):
            t,l,s=choose_crop(r,seed,True)
            self.assertEqual((t,l),(204,204))
            self.assertEqual((t,l,s),choose_crop(r,seed,True))
    def test_shared_inner_seed(self):
        r={'coordinates':[[0,0],[204,204]],'kept':[1]}
        # Both consume one candidate draw, including the singleton case;
        # do not require NumPy to advance identically for different bounds.
        for filtered in (False,True):
            t,l,s=choose_crop(r,123,filtered)
            a=np.arange(512*512*3,dtype=np.uint32).reshape(512,512,3).astype(np.uint16)
            x,y=paired_crop(a,a.copy(),128,s)
            self.assertTrue(torch.equal(x,y));self.assertEqual(tuple(x.shape),(3,128,128))

if __name__=='__main__':unittest.main()
