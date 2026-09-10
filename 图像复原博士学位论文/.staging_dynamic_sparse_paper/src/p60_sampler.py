"""Second preregistered sampling stream; same absolute positions."""
import torch
from torch.utils.data import Sampler

class StepBatchesRepeat(Sampler):
    def __init__(self,count,batch,start,stop):
        self.count,self.batch,self.start,self.stop=count,batch,start,stop
    def __len__(self):return self.stop-self.start
    def __iter__(self):
        epoch,order=-1,None
        for step in range(self.start,self.stop):
            keys=[]
            for slot in range(self.batch):
                position=step*self.batch+slot
                current,offset=divmod(position,self.count)
                if current!=epoch:
                    epoch=current
                    order=torch.randperm(self.count,generator=torch.Generator().manual_seed(4408+epoch)).tolist()
                keys.append((order[offset],200000+position))
            yield keys
