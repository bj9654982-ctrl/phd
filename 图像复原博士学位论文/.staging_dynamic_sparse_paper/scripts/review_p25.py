"""Fixed development center-crop review, no training or test access."""
import json
from pathlib import Path
import sys
import statistics
import torch
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
from src.data import GoProPairDataset
from scripts.pilot_p24_restormer import GROUPS

@torch.inference_mode()
def main():
    project=Path('J:/phd/动态稀疏非局部去模糊论文')
    run=project/'results/runs/P25_restormer_scratch3000_v0'
    out=ROOT/'results/runs/P25_review_v0'
    out.mkdir(parents=True,exist_ok=False)
    names=[c['name'] for c in json.loads((project/'results/runs/P21_visual_review_v0/summary.json').read_text(encoding='utf-8'))['cases'][:3]]
    save_json(out/'plan.json',{'names':names,'crop':256,'detail_box':[96,96,64,64],
                             'steps':[1000,2000,3000],'official_test_accessed':False,'new_training':False})
    ds=GoProPairDataset(project/'data/GOPRO_Large','train',256,False,include_recording_groups=GROUPS)
    indexed={f'{p.scene}/{p.name}':i for i,p in enumerate(ds.pairs)}
    items=[ds[indexed[n]] for n in names]
    images=[[tensor_image(d['blur'])] for d in items]
    scores=[]
    set_determinism(3407)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=network().eval()
    hashes={}
    for step in (1000,2000,3000):
        path=run/f'step_{step:04d}.pt'
        hashes[str(step)]=sha256(path)
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['step']==step and checkpoint['precision']=='bf16'
        model.load_state_dict(checkpoint['model'],strict=True)
        del checkpoint
        ref={r['name']:r for r in read_csv(run/f'validation_{step:04d}.csv')}
        for i,d in enumerate(items):
            x,y=d['blur'][None].cuda(),d['sharp'][None].cuda()
            with autocast('bf16'):
                pred=model(x)
            assert bool(torch.isfinite(pred).all())
            p,s=_psnr_value(pred,y),_ssim_value(pred[0],y[0])
            assert abs(p-float(ref[d['name']]['psnr_db']))<1e-5
            assert abs(s-float(ref[d['name']]['ssim']))<1e-5
            scores.append({'name':d['name'],'step':step,'psnr':p,'ssim':s})
            images[i].append(tensor_image(pred[0]))
    for i,d in enumerate(items):
        images[i].append(tensor_image(d['sharp']))
        fig,axes=plt.subplots(2,5,figsize=(15,6))
        titles=['Blur','1000 steps','2000 steps','3000 steps','Sharp GT']
        for c,(im,title) in enumerate(zip(images[i],titles)):
            im.save(out/f'case{i+1}_{c}.png')
            axes[0,c].imshow(im); axes[0,c].set_title(title)
            axes[1,c].imshow(im.crop((96,96,160,160)),interpolation='nearest')
            axes[0,c].axis('off'); axes[1,c].axis('off')
        fig.suptitle(names[i]+' | development center crop 256, NOT native/test\nFixed center detail 64; no enhancement')
        fig.tight_layout(); fig.savefig(out/f'case{i+1}.png',dpi=140); plt.close(fig)
    rows=read_csv(run/'training.csv')
    blocks=[{'step_end':int(rows[i+99]['step']),'mean_loss':statistics.fmean(float(r['loss']) for r in rows[i:i+100])} for i in range(0,3000,100)]
    vals=json.loads((run/'summary.json').read_text(encoding='utf-8'))['validations']
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].plot([r['step_end'] for r in blocks],[r['mean_loss'] for r in blocks]);axes[0].set_title('Training L1: 100-step means')
    axes[1].plot([r['step'] for r in vals[1:]],[r['psnr_db'] for r in vals[1:]],marker='o');axes[1].axhline(vals[0]['input_psnr_db'],color='gray',linestyle='--');axes[1].set_title('Development PSNR (dB), dashed=input')
    axes[2].plot([r['step'] for r in vals[1:]],[r['ssim'] for r in vals[1:]],marker='o');axes[2].set_title('Development SSIM')
    for ax in axes: ax.set_xlabel('step');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'curves.png',dpi=140);plt.close(fig)
    write_csv(out/'replay.csv',scores);write_csv(out/'loss_blocks.csv',blocks)
    comparisons=[]
    a={r['name']:r for r in read_csv(run/'validation_2000.csv')}
    b={r['name']:r for r in read_csv(run/'validation_3000.csv')}
    assert set(a)==set(b) and len(a)==385
    for group in GROUPS:
        selected=[n for n in a if n.startswith(group+'_')]
        comparisons.append({'group':group,'count':len(selected),
                            'delta_psnr':statistics.fmean(float(b[n]['psnr_db'])-float(a[n]['psnr_db']) for n in selected),
                            'delta_ssim':statistics.fmean(float(b[n]['ssim'])-float(a[n]['ssim']) for n in selected)})
    save_json(out/'summary.json',{'complete':True,'checkpoint_hashes':hashes,'replays':scores,'groups_3000_minus_2000':comparisons,
                                 'new_training':False,'official_test_accessed':False,'paper_result':False})
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
