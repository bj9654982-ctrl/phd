"""Cache training-only 512-patch selection metadata, never image copies."""
import hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd import DPDDPairs,read_rgb16

def main():
    out=ROOT/'results/runs/P57_sampling_cache_v0';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();cv2.setNumThreads(1)
    try:
        lists=Path('H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names')
        ds=DPDDPairs('H:/phddata/dd_dp_dataset_png',lists,'train',128)
        audit=json.loads((ROOT/'results/runs/P50_dpdd_audit_v0/images.json').read_text())
        hashes={r['path']:r['file_sha256'] for r in audit if r['split']=='train'}
        rows=[]
        for i,(sp,tp) in enumerate(ds.pairs):
            for p in (sp,tp):assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)]
            y=read_rgb16(tp);h,w=y.shape[:2]
            coords=list(dict.fromkeys((min(t,h-512),min(l,w-512)) for t in range(0,h,204) for l in range(0,w,204)))
            scores=[]
            for t,l in coords:
                gray=cv2.cvtColor(y[t:t+512,l:l+512],cv2.COLOR_RGB2GRAY)
                scores.append([float(np.hypot(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=k),cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=k)).sum()) for k in (3,7,11,15)])
            a=np.asarray(scores);span=np.ptp(a,axis=0)
            norm=np.divide(a-a.min(0),span,out=np.ones_like(a),where=span>0)
            order=np.argsort(norm.prod(1),kind='stable')
            kept=sorted(order[int(.3*len(coords)):].tolist())
            rows.append({'source':sp.name,'target':tp.name,'shape':list(y.shape),'coordinates':coords,'kept':kept,
                'zero_ranges':int((span==0).sum()),'source_hash':hashes[str(sp)],'target_hash':hashes[str(tp)]})
            if (i+1)%25==0:
                progress={'pairs':i+1,'seconds':time.monotonic()-started}
                (out/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
        result={'complete':True,'split':'train','rows':rows,'test_accessed':False,'val_accessed':False,
            'seconds':time.monotonic()-started,'list_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in lists.glob('train_*.npy')},
            'script_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('COMPLETE',flush=True)
    except Exception as e:
        (out/'failure.json').write_text(json.dumps({'error':repr(e)}));raise
if __name__=='__main__':main()
