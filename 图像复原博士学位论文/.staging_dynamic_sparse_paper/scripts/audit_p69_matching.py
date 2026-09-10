"""Training-only patch correspondence diagnostic, not a restoration method."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd8 import DPDD8Pairs

def features(x):
    a=x.numpy().transpose(1,2,0).reshape(16,16,16,16,3).transpose(0,2,1,3,4).reshape(256,-1).astype(np.float64)
    a-=a.mean(1,keepdims=True)
    norm=np.linalg.norm(a,axis=1)
    return a/np.maximum(norm[:,None],1e-12),norm

def main():
    out=ROOT/'results/runs/P69_matching_v0';out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    ds=DPDD8Pairs('H:/phddata/dd_dp_dataset_png','H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names','train',256)
    indices=np.linspace(0,len(ds)-1,12,dtype=int).tolist();rows=[];excluded=0
    y,x=np.divmod(np.arange(256),16)
    far=(np.abs(y[:,None]-y[None,:])>1)|(np.abs(x[:,None]-x[None,:])>1)
    for index in indices:
        s=ds[index];f,n=features(s['blur']);g,gn=features(s['sharp'])
        valid=n>1e-8
        sim=f@f.T;sim[~far]=-np.inf;sim[:,~valid]=-np.inf
        for i in range(256):
            if not valid[i] or (np.isfinite(sim[i]).sum()<2):excluded+=1;continue
            order=np.argsort(-sim[i],kind='stable')[:2];j,k=order
            # GT is read exclusively for these diagnostic labels, never ranking.
            if gn[i]<=1e-8 or gn[j]<=1e-8:excluded+=1;continue
            rows.append({'source':s['source'],'query':i,'match':int(j),'top1':float(sim[i,j]),
                'margin':float(sim[i,j]-sim[i,k]),'gt_cosine_error':float(1-g[i]@g[j])})
    groups={}
    for field in ('top1','margin'):
        ordered=sorted(rows,key=lambda r:r[field]);q=len(rows)//4
        groups[field]={'low_quartile_gt_error':float(np.mean([r['gt_cosine_error'] for r in ordered[:q]])),
            'high_quartile_gt_error':float(np.mean([r['gt_cosine_error'] for r in ordered[-q:]]))}
    result={'complete':True,'indices':indices,'pairs':len(rows),'excluded':excluded,'groups':groups,
        'seconds':time.monotonic()-start,'test_accessed':False,'val_accessed':False,'training':False,
        'limitations':'Pooled descriptive quartiles; scene confounding and correlated patches; cosine correspondence not restoration benefit'}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'matches.json').write_text(json.dumps(rows),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
