"""Replay exactly the P52 examples after P54; no re-selection or training."""
import json,sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd import DPDDPairs
from scripts.train_p25_restormer import network
from scripts.train_p12_matched_nonlocal import _psnr_value
from scripts.audit_p53_precision import phase

def main():
    run=ROOT/'results/runs/P54_dpdd_continue_v0'
    out=ROOT/'results/runs/P54_fixed_samples_v0';out.mkdir(parents=True,exist_ok=False)
    old=json.loads((ROOT/'results/runs/P52_dpdd_samples_v0/summary.json').read_text())
    expected={r['source']:r for r in json.loads((run/'val_4000.json').read_text())['images']}
    model=network().eval()
    ckpt=torch.load(run/'step_4000.pt',map_location='cpu',weights_only=False)
    assert ckpt['step']==4000 and all(int(v['step'])==4000 for v in ckpt['optimizer']['state'].values())
    model.load_state_dict(ckpt['model']);del ckpt
    ds=DPDDPairs('H:/phddata/dd_dp_dataset_png','H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names','val',256)
    indices={s.name:i for i,(s,t) in enumerate(ds.pairs)}
    sheet=Image.new('RGB',(1024,1200),'white');draw=ImageDraw.Draw(sheet);rows=[]
    with torch.no_grad():
        for j,r in enumerate(old['samples']):
            s=ds[indices[r['source']]];x=s['blur'][None].cuda();y=s['sharp'][None].cuda();p=model(x).clamp(0,1)
            score=_psnr_value(p,y);assert abs(score-expected[r['source']]['psnr'])<1e-4
            rows.append({'source':r['source'],'psnr':score,'gain':score-_psnr_value(x,y),
                'phase2':phase(p-x,2),'phase4':phase(p-x,4),'residual_mae':float((p-x).abs().mean())})
            draw.text((4,j*300+4),f'{r["role"]}: {r["source"]} gain={rows[-1]["gain"]:+.3f}dB',fill='black')
            for col,(label,a) in enumerate(zip(['input','output step4000','GT','abs residual x8'],[x[0],p[0],y[0],(p-x)[0].abs()*8])):
                im=Image.fromarray((a.clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8))
                sheet.paste(im,(col*256,j*300+44));draw.text((col*256+4,j*300+24),label,fill='black')
    sheet.save(out/'comparison.png')
    result={'complete':True,'step':4000,'optimizer_steps_verified':True,'fixed_P52_selection':True,'test_accessed':False,'samples':rows}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
