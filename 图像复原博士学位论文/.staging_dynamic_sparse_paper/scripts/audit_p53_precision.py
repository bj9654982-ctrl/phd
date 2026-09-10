"""Fixed four-image precision/phase diagnostics without parameter updates."""
import json,sys,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd import DPDDPairs
from scripts.train_p25_restormer import network
from scripts.train_p12_matched_nonlocal import _psnr_value

def phase(a,n):
    a=a[...,16:-16,16:-16]
    means=torch.stack([a[...,i::n,j::n].mean((-2,-1)) for i in range(n) for j in range(n)])
    return float((means.max(0).values-means.min(0).values).max())

def main():
    out=ROOT/'results/runs/P53_precision_v0';out.mkdir(parents=True,exist_ok=False)
    try:
        prior=json.loads((ROOT/'results/runs/P52_dpdd_samples_v0/summary.json').read_text())
        ckpt=ROOT/'results/runs/P51_dpdd_baseline_v1/final.pt'
        assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==prior['checkpoint_sha256']
        m=network().eval();m.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=False)['model'])
        ds=DPDDPairs('H:/phddata/dd_dp_dataset_png','H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names','val',256)
        indices={s.name:i for i,(s,t) in enumerate(ds.pairs)};rows=[]
        with torch.no_grad():
            for old in prior['samples']:
                s=ds[indices[old['source']]];x=s['blur'][None].cuda();y=s['sharp'][None].cuda()
                p=m(x).clamp(0,1)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    b=m(x).float().clamp(0,1)
                assert abs(_psnr_value(p,y)-old['psnr'])<1e-4
                rows.append({'source':old['source'],'fp32_psnr':_psnr_value(p,y),
                    'bf16_psnr':_psnr_value(b,y),'precision_output_mae':float((p-b).abs().mean()),
                    'phase_range':{str(n):{key:phase(a,n) for key,a in [('input_error',x-y),('fp32_residual',p-x),('bf16_residual',b-x)]} for n in (2,4)}})
        result={'complete':True,'rows':rows,'test_accessed':False,'updates':0,
            'source_verified_by_network':True,'checkpoint_sha256':prior['checkpoint_sha256'],
            'limitation':'Phase means are descriptive, not proof of aliasing cause; BF16 inference comparison does not isolate BF16 training effects'}
        (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
    except Exception as e:
        (out/'failure.json').write_text(json.dumps({'error':repr(e)}),encoding='utf-8');raise
if __name__=='__main__':main()
