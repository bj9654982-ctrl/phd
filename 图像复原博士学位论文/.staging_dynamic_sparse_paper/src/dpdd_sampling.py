"""Matched all-candidate/filtered-candidate DPDD sampling."""
import json,hashlib
import numpy as np
from pathlib import Path
from src.dpdd import DPDDPairs,read_rgb16,paired_crop

def choose_crop(record,seed,filtered):
    rng=np.random.default_rng(seed)
    candidates=record['kept'] if filtered else list(range(len(record['coordinates'])))
    t,l=record['coordinates'][candidates[int(rng.integers(len(candidates)))]]
    return t,l,int(rng.integers(0,2**31))

class DPDDCandidatePairs(DPDDPairs):
    def __init__(self,root,lists,manifest,filtered):
        super().__init__(root,lists,'train',128)
        m=json.loads(Path(manifest).read_text(encoding='utf-8'))
        assert m['complete'] and m['split']=='train'
        for name,digest in m['list_hashes'].items():assert hashlib.sha256((Path(lists)/name).read_bytes()).hexdigest()==digest
        self.records=m['rows'];self.filtered=filtered
        assert len(self.records)==len(self.pairs)
        for r,(s,t) in zip(self.records,self.pairs):
            assert (r['source'],r['target'])==(s.name,t.name)
            assert r['kept'] and len(set(r['kept']))==len(r['kept'])
            assert all(0<=k<len(r['coordinates']) for k in r['kept'])
            assert all(0<=a<=r['shape'][0]-512 and 0<=b<=r['shape'][1]-512 for a,b in r['coordinates'])
    def __getitem__(self,key):
        index,seed=key
        s,t=self.pairs[index];r=self.records[index]
        top,left,inner_seed=choose_crop(r,seed,self.filtered)
        x,y=read_rgb16(s),read_rgb16(t)
        assert list(x.shape)==r['shape'] and x.shape==y.shape
        x,y=paired_crop(x[top:top+512,left:left+512],y[top:top+512,left:left+512],128,inner_seed)
        return {'blur':x,'sharp':y,'source':s.name,'target':t.name}
