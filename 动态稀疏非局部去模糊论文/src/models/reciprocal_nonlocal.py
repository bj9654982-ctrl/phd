"""Experimental bounded spatial aggregation; no novelty/quality claim."""
import torch
from torch import nn
from torch.nn import functional as F


class ReciprocalNonlocal(nn.Module):
    def __init__(self, channels=384, dim=32, mode='reciprocal', topk=6):
        super().__init__()
        if mode not in ('fixed','content','reciprocal') or not 1<=topk<=24:
            raise ValueError('Invalid mode or topk')
        self.mode,self.topk,self.dim=mode,topk,dim
        self.offsets=tuple((dy*s,dx*s) for s in (1,3,7)
            for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)))
        self.value=nn.Conv2d(channels,dim,1,bias=False)
        self.query=None if mode=='fixed' else nn.Conv2d(channels,dim,1,bias=False)
        self.key=None if mode=='fixed' else nn.Conv2d(channels,dim,1,bias=False)
        self.output=nn.Conv2d(dim,channels,1,bias=False)
        nn.init.zeros_(self.output.weight)

    def candidates(self,h,w,device):
        yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
        off=torch.tensor(self.offsets,device=device)
        y=yy.flatten()[:,None]+off[:,0];x=xx.flatten()[:,None]+off[:,1]
        valid=(y>=0)&(y<h)&(x>=0)&(x<w)
        index=y.clamp(0,h-1)*w+x.clamp(0,w-1)
        return index,valid

    def forward(self,x,return_stats=False):
        b,_,h,w=x.shape;n=h*w
        index,valid=self.candidates(h,w,x.device)
        v=self.value(x).flatten(2).transpose(1,2)
        neighbors=v[:,index.reshape(-1)].reshape(b,n,24,self.dim)
        if self.mode=='fixed':
            weights=valid.float()[None].expand(b,-1,-1)
        else:
            q=F.normalize(self.query(x).float().flatten(2).transpose(1,2),dim=-1)
            k=F.normalize(self.key(x).float().flatten(2).transpose(1,2),dim=-1)
            keys=k[:,index.reshape(-1)].reshape(b,n,24,self.dim)
            logits=(q[:,:,None,:]*keys).sum(-1).masked_fill(~valid[None],-1e9)
            # Stable ties follow the documented offset order; discrete selection has no gradient.
            order=torch.argsort(logits,dim=-1,descending=True,stable=True)
            selected=torch.zeros_like(logits,dtype=torch.bool).scatter_(-1,order[...,:self.topk],True)
            weights=torch.softmax(logits.masked_fill(~selected,-1e9),dim=-1)*valid[None]*selected
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        if self.mode=='reciprocal':
            best=index[None].expand(b,-1,-1).gather(-1,order[...,:1]).squeeze(-1)
            has=valid.any(-1)[None]
            best=torch.where(has,best,torch.full_like(best,-1))
            returns=best[:,index.reshape(-1)].reshape(b,n,24)
            mutual=(returns==torch.arange(n,device=x.device)[None,:,None])&valid[None]
            weights=weights*mutual # no renormalization: rejected mass retains local features
        delta=(weights[...,None]*(neighbors.float()-v.float()[:,:,None,:])).sum(2)
        correction=self.output(delta.transpose(1,2).reshape(b,self.dim,h,w).to(v.dtype))
        result=x+correction
        if return_stats:
            return result,{'accepted_mass':weights.sum(-1).detach(),'valid':valid.detach()}
        return result
