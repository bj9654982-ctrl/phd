"""P27 fixed three-image native checkpoint comparison, no training."""
import json
from pathlib import Path
import sys
import traceback
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_p25_restormer import network,autocast
from scripts.train_p12_matched_nonlocal import set_determinism,_psnr_value,_ssim_value
from scripts.analyze_p16_factorial import read_csv,write_csv,sha256
from scripts.evaluate_p21_full_image import save_json
from scripts.visualize_p13_failure_cases import tensor_image
from src.data.gopro_pairs import _load_rgb

@torch.inference_mode()
def main():
    project=Path('J:/phd/动态稀疏非局部去模糊论文')
    runs=project/'results/runs'
    out=ROOT/'results/runs/P27_native_probe_v0'
    out.mkdir(parents=True,exist_ok=False)
    try:
        cases=json.loads((runs/'P21_visual_review_v0/summary.json').read_text(encoding='utf-8'))['cases'][:3]
        manifest={r['name']:r for r in read_csv(runs/'P21_native_seed3407_v0/manifest.csv')}
        paths={3000:runs/'P25_restormer_scratch3000_v0/step_3000.pt',10000:runs/'P26_restormer_continue10000_v0/step_10000.pt'}
        save_json(out/'plan.json',{'names':[c['name'] for c in cases],'precision':'bf16','native_shape':[720,1280],
                                  'checkpoint_hashes':{str(s):sha256(p) for s,p in paths.items()},
                                  'protocol_hash':sha256(ROOT/'docs/30_P27原生评价协议.md'),
                                  'script_hash':sha256(Path(__file__)),'new_training':False,'official_test_accessed':False})
        set_determinism(3407)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        model=network().eval()
        outputs={i:{} for i in range(3)}
        scores=[]
        for step,path in paths.items():
            ckpt=torch.load(path,map_location='cpu',weights_only=False)
            assert ckpt['step']==step
            model.load_state_dict(ckpt['model'],strict=True)
            del ckpt
            for i,case in enumerate(cases):
                row=manifest[case['name']]
                for key in ('blurry','sharp'):
                    assert sha256(Path(row[key+'_path']))==row[key+'_sha256']
                x=_load_rgb(Path(row['blurry_path']))[None].cuda()
                assert tuple(x.shape)==(1,3,720,1280)
                torch.cuda.reset_peak_memory_stats()
                save_json(out/'progress.json',{'step':step,'case':i+1,'phase':'forward'})
                with autocast('bf16'):
                    pred=model(x)
                torch.cuda.synchronize()
                assert pred.shape==x.shape and bool(torch.isfinite(pred).all())
                peak=torch.cuda.max_memory_allocated()/2**20
                if i==0:
                    save_json(out/f'resource_{step}.json',{'passed':True,'peak_allocated_mib':peak,'quality_not_computed_yet':True})
                y=_load_rgb(Path(row['sharp_path']))[None].cuda()
                p,s=_psnr_value(pred,y),_ssim_value(pred[0],y[0])
                scores.append({'name':case['name'],'step':step,'psnr_db':p,'ssim':s,
                               'input_psnr_db':_psnr_value(x,y),'peak_allocated_mib':peak})
                outputs[i][str(step)]=tensor_image(pred[0])
                outputs[i]['blur']=tensor_image(x[0]);outputs[i]['sharp']=tensor_image(y[0])
                write_csv(out/'per_image.csv',scores)
                print(f'{step} {case["name"]}: {p:.5f}dB/{s:.6f} peak={peak:.1f}MiB',flush=True)
                del x,y,pred
        for i,case in enumerate(cases):
            fig,axes=plt.subplots(3,4,figsize=(16,9))
            for col,key in enumerate(('blur','3000','10000','sharp')):
                im=outputs[i][key]
                im.save(out/f'case{i+1}_{key}.png')
                axes[0,col].imshow(im);axes[0,col].set_title(key)
                for j,(x,y,w,h) in enumerate(case['detail_boxes_xywh']):
                    axes[j+1,col].imshow(im.crop((x,y,x+w,y+h)),interpolation='nearest')
                for ax in axes[:,col]:ax.axis('off')
            fig.suptitle(case['name']+' | Native development image, NOT independent test\nFixed P21 ROIs, no enhancement')
            fig.tight_layout();fig.savefig(out/f'case{i+1}.png',dpi=140);plt.close(fig)
        save_json(out/'summary.json',{'complete':True,'rows':scores,'new_training':False,'official_test_accessed':False,
                                    'full_385_evaluation_complete':False,'paper_result':False,
                                    'eligible_for_full_eval':max(r['peak_allocated_mib'] for r in scores)<=9000})
        save_json(out/'progress.json',{'complete':True})
    except Exception as e:
        save_json(out/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'automatic_fallback':False})
        raise

if __name__=='__main__':main()
