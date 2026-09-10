"""Fixed-input module activity/gradient diagnosis, never optimizer updates."""
import json,gc
from pathlib import Path
import sys
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.reciprocal_nonlocal import ReciprocalNonlocal
from src.data import GoProPairDataset
from scripts.pilot_p24_restormer import GROUPS
from scripts.train_p25_restormer import network,autocast
from scripts.train_p12_matched_nonlocal import set_determinism,_psnr_value,_ssim_value
from scripts.analyze_p16_factorial import read_csv,sha256
from scripts.evaluate_p21_full_image import save_json
from scripts.visualize_p13_failure_cases import tensor_image

def main():
    project=Path('J:/phd/动态稀疏非局部去模糊论文');runs=ROOT/'results/runs'
    out=runs/'P32_mechanism_v0';out.mkdir(parents=True,exist_ok=False)
    train=GoProPairDataset(project/'data/GOPRO_Large','train',128,False,exclude_recording_groups=GROUPS)
    dev=GoProPairDataset(project/'data/GOPRO_Large','train',256,False,include_recording_groups=GROUPS)
    fixed=[train[i] for i in range(4)]
    name='GOPR0378_13_00/000096'
    item=dev[next(i for i,p in enumerate(dev.pairs) if f'{p.scene}/{p.name}'==name)]
    save_json(out/'plan.json',{'train_names':[r['name'] for r in fixed],'dev_name':name,
        'train_crop':128,'dev_crop':256,'roles':['B','C','D'],'updates':0,
        'diagnosis_only':True,'official_test_accessed':False})
    results=[];images=[tensor_image(item['blur'])]
    for role,mode in [('B','fixed'),('C','content'),('D','reciprocal')]:
        set_determinism(3407);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        path=runs/'P31_screen_v0'/role/'last.pt'
        model=network().eval();module=ReciprocalNonlocal(mode=mode).cuda()
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['model'],strict=True);module.load_state_dict(ckpt['module'],strict=True);del ckpt
        captured={}
        def inject(_m,_i,z):
            result,stats=module(z,return_stats=True)
            mass=stats['accepted_mass']
            captured.update({'mass_mean':mass.mean().item(),'zero_mass_fraction':(mass==0).float().mean().item(),
                'relative_correction_rms':((result.float()-z.float()).square().mean().sqrt()/z.float().square().mean().sqrt().clamp_min(1e-12)).item()})
            return result
        hook=model.latent.register_forward_hook(inject)
        x=torch.stack([r['blur'] for r in fixed]).cuda();y=torch.stack([r['sharp'] for r in fixed]).cuda()
        with autocast('bf16'):prediction=model(x);loss=torch.nn.functional.l1_loss(prediction.float(),y)
        loss.backward()
        gradient={k:None if p.grad is None else p.grad.float().norm().item() for k,p in module.named_parameters()}
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in module.parameters())
        train_stats=dict(captured);model.zero_grad(set_to_none=True);module.zero_grad(set_to_none=True)
        del x,y,prediction,loss
        x=item['blur'][None].cuda();y=item['sharp'][None].cuda()
        with torch.no_grad(),autocast('bf16'):prediction=model(x)
        p,s=_psnr_value(prediction,y),_ssim_value(prediction[0],y[0]);dev_stats=dict(captured)
        reference=next(r for r in read_csv(runs/'P31_screen_v0'/role/'validation_1000.csv') if r['name']==name)
        assert abs(p-float(reference['psnr_db']))<1e-5 and abs(s-float(reference['ssim']))<1e-5
        images.append(tensor_image(prediction[0]));hook.remove()
        with torch.no_grad(),autocast('bf16'):disabled=model(x)
        results.append({'role':role,'checkpoint_hash':sha256(path),'train_activity':train_stats,
            'module_gradient_norms':gradient,'dev_activity':dev_stats,'dev_psnr':p,'dev_ssim':s,
            'dev_module_disabled_psnr':_psnr_value(disabled,y)})
        save_json(out/'progress.json',results);print(json.dumps(results[-1]),flush=True)
        del model,module,prediction,disabled,x,y;gc.collect();torch.cuda.empty_cache()
    images.append(tensor_image(item['sharp']))
    fig,axes=plt.subplots(2,5,figsize=(15,6))
    for col,(im,label) in enumerate(zip(images,['Blur','B fixed','C content','D reciprocal','Sharp GT'])):
        im.save(out/f'column{col}.png');axes[0,col].imshow(im);axes[0,col].set_title(label)
        axes[1,col].imshow(im.crop((96,96,160,160)),interpolation='nearest')
        for ax in axes[:,col]:ax.axis('off')
    fig.suptitle(name+' | Fixed development center256, NOT native/test; center64 detail')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    save_json(out/'summary.json',{'complete':True,'results':results,'updates':0,'paper_result':False,
        'disabled_probe_not_retrained_ablation':True,'official_test_accessed':False})

if __name__=='__main__':main()
