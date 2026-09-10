"""Bounded global context probe, not a claimed novel architecture."""
import torch
from torch import nn
from torch.nn import functional as F

class PooledContext(nn.Module):
    def __init__(self, channels=384, dim=32, grid=8):
        super().__init__()
        if grid<1:raise ValueError('grid must be positive')
        self.grid=grid
        self.query=nn.Conv2d(channels,dim,1,bias=False)
        self.key=nn.Conv2d(channels,dim,1,bias=False)
        self.value=nn.Conv2d(channels,dim,1,bias=False)
        self.output=nn.Conv2d(dim,channels,1,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,x):
        b,c,h,w=x.shape
        # At most 64 representative tokens by default, including odd-size maps.
        context=F.adaptive_avg_pool2d(x,(min(h,self.grid),min(w,self.grid)))
        q=F.normalize(self.query(x).float().flatten(2).transpose(1,2),dim=-1)
        k=F.normalize(self.key(context).float().flatten(2),dim=1)
        v=self.value(context).flatten(2).transpose(1,2)
        weights=torch.softmax(q@k,dim=-1)
        aggregate=(weights@v.float()).transpose(1,2).reshape(b,-1,h,w)
        delta=aggregate-self.value(x).float()
        return x+self.output(delta.to(v.dtype))
