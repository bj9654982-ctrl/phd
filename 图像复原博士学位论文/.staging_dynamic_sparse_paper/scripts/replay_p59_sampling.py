"""Fixed P52 samples after both P58 branches, descriptive paired distribution."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd import DPDDPairs
from scripts.train_p25_restormer import network
from scripts.train_p12_matched_nonlocal import _psnr_value
from scripts.audit_p53_precision import phase

def main():
    out=ROOT/'results/runs/P59_fixed_sampling_v0';out.mkdir(parents=True,exist_ok=False)
    run=ROOT/'results/runs/P58_sampling_v0'
    audit=json.loads((run/'analysis.json').read_text())
    chosen=json.loads((ROOT/'results/runs/P52_dpdd_samples_v0/summary.json').read_text())['samples']
    ds=DPDDPairs('H:/phddata/dd_dp_dataset_png','H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names','val',256)
    indices={s.name:i for i,(s,t) in enumerate(ds.pairs)}
    sheet=Image.new('RGB',(1024,1200),'white');draw=ImageDraw.Draw(sheet)
    model=network().eval();rows=[];allvals={}
    for role in ('A','B'):
        path=run/role/'step_5000.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['checkpoint_hashes'][role]
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['step']==5000 and all(int(v['step'])==5000 for v in state['optimizer']['state'].values())
        model.load_state_dict(state['model']);del state
        allvals[role]=json.loads((run/role/'val_5000.json').read_text())['images']
        expected={r['source']:r for r in allvals[role]}
        with torch.no_grad():
            for j,r in enumerate(chosen):
                s=ds[indices[r['source']]];x=s['blur'][None].cuda();y=s['sharp'][None].cuda();p=model(x).clamp(0,1)
                score=_psnr_value(p,y);assert abs(score-expected[s['source']]['psnr'])<1e-4
                rows.append({'role':role,'source':s['source'],'psnr':score,'gain':score-_psnr_value(x,y),
                    'phase2':phase(p-x,2),'phase4':phase(p-x,4)})
                for col,label,t in [(0,'input',x[0]),(1 if role=='A' else 2,role+' output',p[0]),(3,'GT',y[0])]:
                    im=Image.fromarray((t.cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8))
                    sheet.paste(im,(col*256,j*300+44));draw.text((col*256+4,j*300+24),label,fill='black')
                draw.text((4,j*300+4),s['source'],fill='black')
    diffs=np.array([b['psnr']-a['psnr'] for a,b in zip(allvals['A'],allvals['B'])])
    assert all(a['source']==b['source'] for a,b in zip(allvals['A'],allvals['B']))
    result={'complete':True,'test_accessed':False,'samples':rows,'B_minus_A_quantiles':np.quantile(diffs,[0,.25,.5,.75,1]).tolist(),
        'B_minus_A_mean':float(diffs.mean()),'B_better':int((diffs>0).sum()),
        'limitation':'Post-hoc development diagnostics, no significance or independent-seed claim'}
    sheet.save(out/'comparison.png');(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
