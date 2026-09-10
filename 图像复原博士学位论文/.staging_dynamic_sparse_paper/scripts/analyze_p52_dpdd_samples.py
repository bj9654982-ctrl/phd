"""Replay fixed development examples; no fitting, no test access."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.dpdd import DPDDPairs
from scripts.train_p25_restormer import network
from scripts.train_p12_matched_nonlocal import _psnr_value,_ssim_value,set_determinism

def main():
    out=ROOT/'results/runs/P52_dpdd_samples_v0';out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    try:
        old=ROOT/'results/runs/P51_dpdd_baseline_v1'
        rows=json.loads((old/'val_1000.json').read_text())['images']
        ranked=sorted(rows,key=lambda r:r['psnr']-r['input_psnr'])
        chosen=[('worst_gain',ranked[0]),('best_gain',ranked[-1]),('median_gain',ranked[len(rows)//2]),('fixed_first',rows[0])]
        plan={'selection':'worst/best/upper-median by P51 dev gain plus official first dev image',
            'sources':[(role,r['source']) for role,r in chosen],'test_accessed':False,'training':False}
        (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
        ds=DPDDPairs('H:/phddata/dd_dp_dataset_png','H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names','val',256)
        index={s.name:i for i,(s,t) in enumerate(ds.pairs)}
        set_determinism(3407)
        ckpt=old/'final.pt'
        checkpoint_hash=hashlib.sha256(ckpt.read_bytes()).hexdigest()
        state=torch.load(ckpt,map_location='cpu',weights_only=False)
        assert state['step']==1000
        model=network().eval();model.load_state_dict(state['model'],strict=True)
        sheet=Image.new('RGB',(1024,4*300),'white');draw=ImageDraw.Draw(sheet)
        results=[]
        with torch.no_grad():
            for j,(role,prior) in enumerate(chosen):
                s=ds[index[prior['source']]]
                x,y=s['blur'].unsqueeze(0).cuda(),s['sharp'].unsqueeze(0).cuda()
                raw=model(x);p=raw.clamp(0,1)
                psnr=_psnr_value(p,y);ssim=_ssim_value(p[0],y[0])
                assert abs(psnr-prior['psnr'])<1e-4 and abs(ssim-prior['ssim'])<1e-5
                result={'role':role,'source':s['source'],'target':s['target'],'psnr':psnr,'ssim':ssim,
                    'gain_psnr':psnr-prior['input_psnr'],
                    'input_mae':float((x-y).abs().mean()),'output_mae':float((p-y).abs().mean()),
                    'residual_mae':float((p-x).abs().mean()),
                    'residual_channel_mean':(p-x).mean((0,2,3)).cpu().tolist(),
                    'raw_out_of_range_fraction':float(((raw<0)|(raw>1)).float().mean())}
                results.append(result)
                tensors=[x[0],p[0],y[0],(p[0]-x[0]).abs()*8]
                for col,(label,a) in enumerate(zip(['input','output','GT','abs(output-input) x8'],tensors)):
                    im=Image.fromarray((a.clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8))
                    im.save(out/f'{role}_{col}.png')
                    sheet.paste(im,(col*256,j*300+44))
                    draw.text((col*256+4,j*300+24),label,fill='black')
                draw.text((4,j*300+4),f'{role}: {s["source"]} gain={result["gain_psnr"]:+.3f}dB',fill='black')
        sheet.save(out/'comparison.png')
        summary={**plan,'complete':True,'checkpoint_sha256':checkpoint_hash,'replay_verified':True,
            'seconds':time.monotonic()-start,'samples':results,
            'display':'8-bit preview only; metrics computed from float normalized 16-bit data',
            'limitation':'post-hoc dev diagnostics, not representative test performance'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary),flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'error':repr(exc)}),encoding='utf-8');raise

if __name__=='__main__':
    main()
