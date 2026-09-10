import unittest
import torch
from src.optimizer_transfer import transfer_adamw

class TransferTests(unittest.TestCase):
    def fixture(self):
        model=torch.nn.Sequential(torch.nn.Linear(2,2,bias=False),torch.nn.Linear(2,2,bias=False))
        extra=torch.nn.Linear(2,2,bias=False)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        saved=opt.state_dict();ids=saved['param_groups'][0]['params']
        # Synthetic states, no training or optimizer step performed.
        for i,pid in enumerate(ids):
            saved['state'][pid]={'step':torch.tensor(10000.),'exp_avg':torch.full((2,2),float(i+1)),
                                  'exp_avg_sq':torch.full((2,2),float(i+3))}
        return model,extra,saved,['0.weight','1.weight']

    def test_name_mapping_not_target_order(self):
        model,extra,saved,names=self.fixture()
        # Checkpoint ids now intentionally describe reverse name order.
        opt=transfer_adamw(model,extra,saved,names[::-1])
        self.assertEqual(opt.state[model[0].weight]['exp_avg'][0,0].item(),2.)
        self.assertEqual(opt.state[model[1].weight]['exp_avg'][0,0].item(),1.)
        self.assertNotIn(extra.weight,opt.state)
        self.assertEqual(opt.param_groups[1]['lr'],3e-4)
        opt.state[model[0].weight]['exp_avg'].zero_()
        self.assertEqual(saved['state'][1]['exp_avg'][0,0].item(),2.)

    def test_duplicate_names_rejected(self):
        model,extra,saved,names=self.fixture()
        with self.assertRaises(ValueError):transfer_adamw(model,extra,saved,[names[0]]*2)

    def test_bad_shape_rejected(self):
        model,extra,saved,names=self.fixture();saved['state'][0]['exp_avg']=torch.zeros(3)
        with self.assertRaises(ValueError):transfer_adamw(model,extra,saved,names)

    def test_no_module(self):
        model,_,saved,names=self.fixture();opt=transfer_adamw(model,None,saved,names)
        self.assertEqual(len(opt.param_groups),1)
        self.assertTrue(all(s['step'].item()==10000 for s in opt.state.values()))

if __name__=='__main__':unittest.main()
