"""Synthetic reciprocal-matching counterexamples; not restoration experiments."""
import json
from pathlib import Path
import torch

def reciprocal_mask(best):
    if best.ndim!=2 or best.dtype!=torch.long:
        raise ValueError('Expected batch x positions integer indices')
    n=best.shape[1]
    if n==0 or bool(((best<0)|(best>=n)).any()):
        raise ValueError('Invalid correspondence index')
    source=torch.arange(n,device=best.device)[None].expand_as(best)
    return (best.gather(1,best)==source)&(best!=source)

def main():
    # Reciprocal does not mean semantically correct: an erroneous pair can pass.
    cases={'reciprocal_pair':[[1,0]],'three_cycle':[[1,2,0]],
           'self_only':[[0,1]],'wrong_but_reciprocal':[[2,3,0,1]],
           'independent_batches':[[1,0,2],[0,2,1]]}
    expected={'reciprocal_pair':[[True,True]],'three_cycle':[[False]*3],
              'self_only':[[False,False]],'wrong_but_reciprocal':[[True]*4],
              'independent_batches':[[True,True,False],[False,True,True]]}
    results={}
    for name,mapping in cases.items():
        got=reciprocal_mask(torch.tensor(mapping,dtype=torch.long)).tolist()
        assert got==expected[name]
        results[name]=got
    out=Path(__file__).resolve().parents[1]/'results/runs/P29_consistency_synthetic_v0'
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps({'passed':True,'cases':results,
         'conclusion':'reciprocity alone cannot certify correct matches',
         'new_training':False,'real_image_quality_claim':False},indent=2),encoding='utf-8')
    print('5 synthetic checks passed; wrong reciprocal pairs also pass the criterion.')

if __name__=='__main__':main()
