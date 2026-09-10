"""Training-only descriptive sampling audit; no optimizer/model/test access."""
import json,sys,time,hashlib
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd import DPDDPairs,read_rgb16

def texture(a):
    gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float64)/65535
    dx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    dy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    return float(np.hypot(dx,dy).mean())

def main():
    out=ROOT/'results/runs/P56_sampling_v0';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();cv2.setNumThreads(1)
    ds=DPDDPairs('H:/phddata/dd_dp_dataset_png','H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names','train',128)
    indices=np.linspace(0,len(ds)-1,12,dtype=int).tolist()
    plan={'indices':indices,'selection':'12 equally spaced official train indices',
        'seed':5600,'samples_per_image_per_strategy':64,'strategies':['uniform_full','all_512','filtered_512'],
        'test_accessed':False,'val_accessed':False,'updates':0,
        'note':'author-like 16bit score; zero-range channel contributes ones, not NaN; crop boundary anchors deduplicated'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    records=[]
    for index in indices:
        sp,tp=ds.pairs[index];x,y=read_rgb16(sp),read_rgb16(tp);h,w=y.shape[:2]
        coords=list(dict.fromkeys((min(i,h-512),min(j,w-512)) for i in range(0,h,204) for j in range(0,w,204)))
        scores=[]
        for top,left in coords:
            gray=cv2.cvtColor(y[top:top+512,left:left+512],cv2.COLOR_RGB2GRAY)
            scores.append([float(np.hypot(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=k),cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=k)).sum()) for k in (3,7,11,15)])
        a=np.array(scores);span=np.ptp(a,axis=0)
        normalized=np.divide(a-a.min(0),span,out=np.ones_like(a),where=span>0)
        combined=normalized.prod(1);order=np.argsort(combined,kind='stable')
        kept=sorted(order[int(.3*len(coords)):].tolist())
        values={}
        for strategy in plan['strategies']:
            rng=np.random.default_rng(5600+index);measure=[]
            for _ in range(64):
                if strategy=='uniform_full':top,left=int(rng.integers(h-127)),int(rng.integers(w-127))
                else:
                    choices=list(range(len(coords))) if strategy=='all_512' else kept
                    t,l=coords[choices[int(rng.integers(len(choices)))]]
                    top,left=t+int(rng.integers(385)),l+int(rng.integers(385))
                xx=x[top:top+128,left:left+128];yy=y[top:top+128,left:left+128]
                mse=np.mean(((xx.astype(np.float64)-yy)/65535)**2)
                measure.append({'texture':texture(yy),'input_psnr':float(-10*np.log10(max(mse,1e-12)))})
            values[strategy]={'mean_texture':float(np.mean([m['texture'] for m in measure])),
                'mean_input_psnr':float(np.mean([m['input_psnr'] for m in measure]))}
        records.append({'index':index,'source':sp.name,'target':tp.name,'candidates':len(coords),'kept':len(kept),
            'zero_score_ranges':int((span==0).sum()),'candidate_coordinates':coords,'kept_indices':kept,'measurements':values})
        print('audited '+sp.name,flush=True)
    averages={strategy:{key:float(np.mean([r['measurements'][strategy][key] for r in records])) for key in ('mean_texture','mean_input_psnr')} for strategy in plan['strategies']}
    result={**plan,'complete':True,'seconds':time.monotonic()-started,'averages':averages,'images':records,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':'12-image descriptive sample, no full-dataset or restoration-quality conclusion; strategies not paired same coordinates'}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(averages))
if __name__=='__main__':main()
