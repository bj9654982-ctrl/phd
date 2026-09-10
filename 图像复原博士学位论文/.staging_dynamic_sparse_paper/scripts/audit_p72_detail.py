"""Offline necessary-condition diagnostic on fixed correspondences only."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd8 import DPDD8Pairs

def grad(x):
    b=x.numpy().transpose(1,2,0).reshape(16,16,16,16,3).transpose(0,2,1,3,4).reshape(256,16,16,3).astype(np.float64)
    b-=b.mean((1,2),keepdims=True)
    return np.concatenate([np.diff(b,axis=2).reshape(256,-1),np.diff(b,axis=1).reshape(256,-1)],axis=1)

def main():
    out=ROOT/'results/runs/P72_detail_v0';out.mkdir(parents=True,exist_ok=False)
    path=ROOT/'results/runs/P70_structure_v0/matches.json';old=json.loads(path.read_text())
    ds=DPDD8Pairs('H:/phddata/dd_dp_dataset_png','H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names','train',256)
    index={s.name:i for i,(s,t) in enumerate(ds.pairs)};rows=[];excluded=0
    for name in dict.fromkeys(r['source'] for r in old):
        s=ds[index[name]];x,y=grad(s['blur']),grad(s['sharp']);energy=(y*y).sum(1)
        error=((x-y)**2).sum(1)/np.maximum(energy,1e-12)
        for r in old:
            if r['source']!=name:continue
            i,j=r['query'],r['match']
            if energy[i]<=1e-12 or energy[j]<=1e-12:excluded+=1;continue
            rows.append({'source':name,'query':i,'match':j,'query_error':float(error[i]),'reference_error':float(error[j]),
                'reference_lower_error':bool(error[j]<error[i]),'gt_structure_cosine':1-r['gradient_error']})
    ordered=sorted(rows,key=lambda r:r['gt_structure_cosine']);n=len(rows)//4
    def stats(rs):return {'count':len(rs),'reference_lower_error_fraction':float(np.mean([r['reference_lower_error'] for r in rs])),
        'median_query_error':float(np.median([r['query_error'] for r in rs])),
        'median_reference_error':float(np.median([r['reference_error'] for r in rs])),
        'median_gt_structure_cosine':float(np.median([r['gt_structure_cosine'] for r in rs]))}
    result={'complete':True,'excluded_flat_gt':excluded,'overall':stats(rows),
        'low_structure_quartile':stats(ordered[:n]),'high_structure_quartile':stats(ordered[-n:]),
        'per_scene':{name:stats([r for r in rows if r['source']==name]) for name in dict.fromkeys(r['source'] for r in rows)},
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'training':False,'test_accessed':False,'val_accessed':False,
        'warning':'GT diagnostic only; relative gradient error includes noise/misalignment and is not calibrated blur severity'}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(out/'pairs.json').write_text(json.dumps(rows))
    print(json.dumps({k:v for k,v in result.items() if k!='per_scene'}))
if __name__=='__main__':main()
