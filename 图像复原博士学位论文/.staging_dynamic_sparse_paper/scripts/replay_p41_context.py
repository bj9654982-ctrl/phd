"""Fixed development image replay and descriptive training-log audit; no updates."""
import sys,json,gc,statistics
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_p25_restormer import network,autocast
from scripts.train_p12_matched_nonlocal import set_determinism,_psnr_value,_ssim_value
from scripts.pilot_p24_restormer import GROUPS
from scripts.analyze_p16_factorial import read_csv,sha256
from scripts.evaluate_p21_full_image import save_json
from scripts.visualize_p13_failure_cases import tensor_image
from src.data import GoProPairDataset
from src.models.reciprocal_nonlocal import ReciprocalNonlocal
from src.models.pooled_context import PooledContext

def main():
    run=ROOT/'results/runs/P40_context_screen_v0';out=ROOT/'results/runs/P41_replay_v0'
    out.mkdir(parents=True,exist_ok=False)
    ds=GoProPairDataset(Path('J:/phd/动态稀疏非局部去模糊论文/data/GOPRO_Large'),'train',256,False,include_recording_groups=GROUPS)
    name='GOPR0378_13_00/000096'
    item=ds[next(i for i,p in enumerate(ds.pairs) if f'{p.scene}/{p.name}'==name)]
    save_json(out/'plan.json',{'name':name,'crop':256,'updates':0,'official_test_accessed':False})
    images=[tensor_image(item['blur'])];rows=[]
    for role in 'ACP':
        set_determinism(3407);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        ckpt=torch.load(run/role/'last.pt',map_location='cpu',weights_only=False)
        model=network().eval();model.load_state_dict(ckpt['model'],strict=True)
        module=None if role=='A' else (PooledContext() if role=='P' else ReciprocalNonlocal(mode='content')).cuda()
        hook=None
        if module is not None:
            module.load_state_dict(ckpt['module'],strict=True)
            hook=model.latent.register_forward_hook(lambda _m,_i,z:module(z))
        x=item['blur'][None].cuda();y=item['sharp'][None].cuda()
        with torch.no_grad(),autocast('bf16'):pred=model(x)
        p,s=_psnr_value(pred,y),_ssim_value(pred[0],y[0])
        ref=next(r for r in read_csv(run/role/'validation_1000.csv') if r['name']==name)
        assert abs(p-float(ref['psnr_db']))<1e-5 and abs(s-float(ref['ssim']))<1e-5
        logs=read_csv(run/role/'training.csv')
        row={'role':role,'psnr':p,'ssim':s,'checkpoint_hash':sha256(run/role/'last.pt'),
             'loss_first100':statistics.fmean(float(r['loss']) for r in logs[:100]),
             'loss_last100':statistics.fmean(float(r['loss']) for r in logs[-100:]),
             'optimizer_lr':[g['lr'] for g in ckpt['optimizer']['param_groups']]}
        rows.append(row);images.append(tensor_image(pred[0]));print(json.dumps(row),flush=True)
        if hook:hook.remove()
        del model,module,ckpt,x,y,pred;gc.collect();torch.cuda.empty_cache()
    images.append(tensor_image(item['sharp']))
    fig,axes=plt.subplots(2,5,figsize=(15,6))
    for i,(im,label) in enumerate(zip(images,['Blur','A baseline','C content','P pooled','Sharp GT'])):
        im.save(out/f'column{i}.png');axes[0,i].imshow(im);axes[0,i].set_title(label)
        axes[1,i].imshow(im.crop((96,96,160,160)),interpolation='nearest')
        for ax in axes[:,i]:ax.axis('off')
    fig.suptitle(name+' | fixed development center256, center64 detail; NOT test')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    save_json(out/'summary.json',{'complete':True,'rows':rows,'updates':0,'official_test_accessed':False})
if __name__=='__main__':main()
