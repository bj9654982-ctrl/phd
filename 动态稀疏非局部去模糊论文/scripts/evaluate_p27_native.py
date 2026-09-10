"""Frozen native development evaluation; no training, resizing or test access."""
import json
from pathlib import Path
import sys
import time
import statistics
import traceback
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_p25_restormer import network,autocast
from scripts.train_p12_matched_nonlocal import set_determinism,_psnr_value,_ssim_value
from scripts.analyze_p16_factorial import read_csv,write_csv,sha256
from scripts.evaluate_p21_full_image import save_json
from src.data.gopro_pairs import _load_rgb,recording_group

@torch.inference_mode()
def main():
    project=Path('J:/phd/动态稀疏非局部去模糊论文')
    runs=project/'results/runs'
    out=ROOT/'results/runs/P27_native_full_v0'
    out.mkdir(parents=True,exist_ok=False)
    try:
        probe=json.loads((ROOT/'results/runs/P27_native_probe_v0/summary.json').read_text(encoding='utf-8'))
        assert probe['complete'] and probe['eligible_for_full_eval']
        manifest=runs/'P21_native_seed3407_v0/manifest.csv'
        rows=read_csv(manifest)
        assert len(rows)==385 and len({r['name'] for r in rows})==385
        checkpoint=runs/'P26_restormer_continue10000_v0/step_10000.pt'
        save_json(out/'plan.json',{'names':[r['name'] for r in rows],'manifest_hash':sha256(manifest),
            'checkpoint_hash':sha256(checkpoint),'precision':'bf16','shape':[720,1280],
            'script_hash':sha256(Path(__file__)),'new_training':False,'official_test_accessed':False})
        set_determinism(3407)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        model=network().eval()
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ckpt['step']==10000
        model.load_state_dict(ckpt['model'],strict=True)
        del ckpt
        records=[];started=time.perf_counter()
        reference={r['name']:r for r in probe['rows'] if r['step']==10000}
        for i,row in enumerate(rows):
            for key in ('blurry','sharp'):
                path=Path(row[key+'_path']).resolve()
                assert path.is_relative_to((project/'data/GOPRO_Large/train').resolve())
                assert sha256(path)==row[key+'_sha256']
            x=_load_rgb(Path(row['blurry_path']))[None].cuda()
            y=_load_rgb(Path(row['sharp_path']))[None].cuda()
            assert tuple(x.shape)==tuple(y.shape)==(1,3,720,1280)
            with autocast('bf16'):pred=model(x)
            assert pred.shape==x.shape and bool(torch.isfinite(pred).all())
            p,s=_psnr_value(pred,y),_ssim_value(pred[0],y[0])
            if row['name'] in reference:
                assert abs(p-reference[row['name']]['psnr_db'])<1e-5
                assert abs(s-reference[row['name']]['ssim'])<1e-5
            inp,ins=_psnr_value(x,y),_ssim_value(x[0],y[0])
            records.append({'name':row['name'],'group':recording_group(row['name'].split('/')[0]),
                'psnr_db':p,'ssim':s,'input_psnr_db':inp,'input_ssim':ins,'gain_db':p-inp,'gain_ssim':s-ins})
            del x,y,pred
            if (i+1)%10==0 or i==0 or i==384:
                write_csv(out/'per_image.csv',records)
                progress={'completed':i+1,'total':385,'elapsed_seconds':time.perf_counter()-started}
                save_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        def aggregate(items):
            return {'count':len(items),**{k:statistics.fmean(r[k] for r in items) for k in
                ('psnr_db','ssim','input_psnr_db','input_ssim','gain_db','gain_ssim')},
                'positive_psnr_images':sum(r['gain_db']>0 for r in items)}
        save_json(out/'summary.json',{'complete':True,'overall':aggregate(records),
            'groups':{g:aggregate([r for r in records if r['group']==g]) for g in sorted({r['group'] for r in records})},
            'new_training':False,'official_test_accessed':False,'paper_result':False,'elapsed_seconds':time.perf_counter()-started})
        save_json(out/'progress.json',{'complete':True,'completed':385,'total':385})
    except Exception as e:
        save_json(out/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'automatic_fallback':False})
        raise

if __name__=='__main__':main()
