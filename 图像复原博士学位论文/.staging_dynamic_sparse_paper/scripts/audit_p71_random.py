"""One fixed random-match control for P69 queries; GT never selects matches."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd8 import DPDD8Pairs
from scripts.audit_p69_matching import features
def main():
    out=ROOT/'results/runs/P71_random_v0';out.mkdir(parents=True,exist_ok=False)
    path=ROOT/'results/runs/P70_structure_v0/matches.json';prior=json.loads(path.read_text())
    ds=DPDD8Pairs('H:/phddata/dd_dp_dataset_png','H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names','train',256)
    indices={s.name:i for i,(s,t) in enumerate(ds.pairs)};rng=np.random.default_rng(7100);rows=[];scenes=[]
    yy,xx=np.divmod(np.arange(256),16)
    for name in dict.fromkeys(r['source'] for r in prior):
        sample=ds[indices[name]];_,norm_input=features(sample['blur'])
        a=sample['sharp'].numpy().transpose(1,2,0)
        b=a.reshape(16,16,16,16,3).transpose(0,2,1,3,4).reshape(256,16,16,3).astype(np.float64)
        b-=b.mean((1,2),keepdims=True)
        g=np.concatenate([np.diff(b,axis=2).reshape(256,-1),np.diff(b,axis=1).reshape(256,-1)],axis=1)
        norm=np.linalg.norm(g,axis=1);g/=np.maximum(norm[:,None],1e-12)
        current=[]
        for r in prior:
            if r['source']!=name:continue
            i=r['query'];mask=((abs(yy-yy[i])>1)|(abs(xx-xx[i])>1))&(norm_input>1e-8)
            candidates=np.flatnonzero(mask);j=int(rng.choice(candidates))
            if norm[j]<=1e-8:continue
            error=float(1-np.clip(g[i]@g[j],-1,1))
            current.append({'source':name,'query':i,'retrieved':r['match'],'random':j,
                'retrieved_error':r['gradient_error'],'random_error':error})
        rows.extend(current)
        scenes.append({'source':name,'count':len(current),'retrieved_error':float(np.mean([r['retrieved_error'] for r in current])),
            'random_error':float(np.mean([r['random_error'] for r in current]))})
    result={'complete':True,'seed':7100,'count':len(rows),'excluded':len(prior)-len(rows),'scenes':scenes,
        'retrieved_error':float(np.mean([r['retrieved_error'] for r in rows])),
        'random_error':float(np.mean([r['random_error'] for r in rows])),
        'retrieval_better_scenes':sum(s['retrieved_error']<s['random_error'] for s in scenes),
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'test_accessed':False,'training':False,
        'limitation':'Single random control; structure correspondence is not deblurring gain or novelty'}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(out/'pairs.json').write_text(json.dumps(rows))
    print(json.dumps(result))
if __name__=='__main__':main()
