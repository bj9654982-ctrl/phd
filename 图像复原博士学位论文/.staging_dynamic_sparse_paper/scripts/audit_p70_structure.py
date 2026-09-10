"""Relabel fixed P69 correspondences with GT interior spatial gradients."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd8 import DPDD8Pairs

def summarize(rows):
    result={}
    for key in ('top1','margin'):
        ordered=sorted(rows,key=lambda r:r[key]);n=len(rows)//4
        result[key]={'low_error':float(np.mean([r['gradient_error'] for r in ordered[:n]])),
                     'high_error':float(np.mean([r['gradient_error'] for r in ordered[-n:]]))}
    return result

def main():
    out=ROOT/'results/runs/P70_structure_v0';out.mkdir(parents=True,exist_ok=False)
    path=ROOT/'results/runs/P69_matching_v0/matches.json';old=json.loads(path.read_text())
    ds=DPDD8Pairs('H:/phddata/dd_dp_dataset_png','H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names','train',256)
    indices={s.name:i for i,(s,t) in enumerate(ds.pairs)};rows=[];per_scene={};excluded=0
    for name in dict.fromkeys(r['source'] for r in old):
        a=ds[indices[name]]['sharp'].numpy().transpose(1,2,0)
        blocks=a.reshape(16,16,16,16,3).transpose(0,2,1,3,4).reshape(256,16,16,3).astype(np.float64)
        blocks-=blocks.mean((1,2),keepdims=True)
        # Interior finite differences exclude inter-block and padding boundaries.
        gx=np.diff(blocks,axis=2).reshape(256,-1);gy=np.diff(blocks,axis=1).reshape(256,-1)
        g=np.concatenate([gx,gy],axis=1);norm=np.linalg.norm(g,axis=1);g/=np.maximum(norm[:,None],1e-12)
        current=[]
        for r in old:
            if r['source']!=name:continue
            i,j=r['query'],r['match']
            if norm[i]<=1e-8 or norm[j]<=1e-8:excluded+=1;continue
            current.append({**r,'gradient_error':float(1-np.clip(g[i]@g[j],-1,1))})
        rows.extend(current);per_scene[name]=summarize(current)
    result={'complete':True,'source_matches_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'same_correspondences_no_reranking':True,'valid':len(rows),'excluded_zero_gradient':excluded,
        'pooled':summarize(rows),'per_scene':per_scene,
        'test_accessed':False,'val_accessed':False,'training':False}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'matches.json').write_text(json.dumps(rows),encoding='utf-8')
    print(json.dumps({'pooled':result['pooled'],'valid':len(rows),'excluded':excluded,
        'high_better_scenes':{k:sum(v[k]['high_error']<v[k]['low_error'] for v in per_scene.values()) for k in ('top1','margin')}}))
if __name__=='__main__':main()
