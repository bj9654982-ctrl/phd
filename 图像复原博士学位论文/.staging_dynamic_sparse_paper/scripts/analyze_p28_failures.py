"""Post-hoc failure diagnosis; no training or independent test."""
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
from src.data.gopro_pairs import _load_rgb

@torch.inference_mode()
def main():
    project=Path('J:/phd/动态稀疏非局部去模糊论文');runs=project/'results/runs'
    out=ROOT/'results/runs/P28_failure_review_v0';out.mkdir(parents=True,exist_ok=False)
    rows=read_csv(ROOT/'results/runs/P27_native_full_v0/per_image.csv')
    bad=sorted([r for r in rows if float(r['gain_db'])<0],key=lambda r:float(r['gain_db']))
    assert len(rows)==385 and len(bad)==15
    write_csv(out/'negative_cases.csv',bad)
    worst=bad[0]
    save_json(out/'plan.json',{'name':worst['name'],'selection':'post-hoc worst PSNR gain',
        'roi_rule':'post-hoc maximum added squared error on disjoint 96 tiles; plus fixed center192',
        'not_representative':True,'new_training':False,'official_test_accessed':False})
    entry=next(r for r in read_csv(runs/'P21_native_seed3407_v0/manifest.csv') if r['name']==worst['name'])
    for key in ('blurry','sharp'):assert sha256(Path(entry[key+'_path']))==entry[key+'_sha256']
    set_determinism(3407);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=network().eval()
    ckpt=torch.load(runs/'P26_restormer_continue10000_v0/step_10000.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['model'],strict=True);del ckpt
    x=_load_rgb(Path(entry['blurry_path']))[None].cuda();y=_load_rgb(Path(entry['sharp_path']))[None].cuda()
    with autocast('bf16'):pred=model(x)
    assert bool(torch.isfinite(pred).all())
    p=_psnr_value(pred,y);s=_ssim_value(pred[0],y[0])
    assert abs(p-float(worst['psnr_db']))<1e-5 and abs(s-float(worst['ssim']))<1e-5
    added=((pred.float().clamp(0,1)-y).square()-(x-y).square()).mean(1)[0].cpu().numpy()
    tiles=[(float(added[t:t+96,l:l+96].mean()),l,t) for t in range(0,720-95,96) for l in range(0,1280-95,96)]
    _,left,top=max(tiles)
    boxes=[(left,top,96,96),(544,264,192,192)]
    ims=[tensor_image(t[0]) for t in (x,pred,y)]
    fig,axes=plt.subplots(3,3,figsize=(12,10))
    for col,(im,label) in enumerate(zip(ims,['Blur','Restormer 10000','Sharp GT'])):
        im.save(out/(label.split()[0]+'.png'))
        axes[0,col].imshow(im);axes[0,col].set_title(label)
        for j,(l,t,w,h) in enumerate(boxes):axes[j+1,col].imshow(im.crop((l,t,l+w,t+h)),interpolation='nearest')
        for ax in axes[:,col]:ax.axis('off')
    fig.suptitle(worst['name']+' | POST-HOC worst development case\nRow2: GT-error-selected ROI; row3: fixed center. No enhancement.')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4));bound=float(abs(added).max())
    im=ax.imshow(added,cmap='coolwarm',vmin=-bound,vmax=bound);fig.colorbar(im,ax=ax)
    ax.set_title('Added squared error: output minus input (red=worse, blue=better)')
    fig.tight_layout();fig.savefig(out/'added_error.png',dpi=140);plt.close(fig)
    summary={'complete':True,'negative_count':len(bad),'worst':worst,'boxes':boxes,
        'negative_groups':{g:sum(r['group']==g for r in bad) for g in sorted({r['group'] for r in rows})},
        'negative_ssim_among_negative_psnr':sum(float(r['gain_ssim'])<0 for r in bad),
        'negative_input_psnr_mean':statistics.fmean(float(r['input_psnr_db']) for r in bad),
        'positive_input_psnr_mean':statistics.fmean(float(r['input_psnr_db']) for r in rows if float(r['gain_db'])>0),
        'new_training':False,'official_test_accessed':False,'paper_result':False}
    save_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
