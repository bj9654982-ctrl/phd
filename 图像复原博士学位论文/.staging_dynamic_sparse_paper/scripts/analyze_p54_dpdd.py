"""Post-completion integrity checks; no evaluation on test data."""
import json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'results/runs/P54_dpdd_continue_v0'
    summary=json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['step']==4000
    logs=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs]==list(range(1001,4001))
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad']) for r in logs)
    old=json.loads((ROOT/'results/runs/P51_dpdd_baseline_v1/val_1000.json').read_text())['images']
    names=[r['source'] for r in old]
    assert len(set(names))==74
    evaluated=[]
    for step in (1000,2000,3000,4000):
        v=json.loads((out/f'val_{step}.json').read_text())
        assert [r['source'] for r in v['images']]==names
        assert all(abs(r['input_psnr']-o['input_psnr'])<1e-6 for r,o in zip(v['images'],old))
        gains=[r['psnr']-r['input_psnr'] for r in v['images']]
        evaluated.append({'step':step,'psnr':v['psnr'],'ssim':v['ssim'],
            'gain_psnr':v['psnr']-v['input_psnr'],'gain_ssim':v['ssim']-v['input_ssim'],
            'positive_images':sum(x>0 for x in gains),'negative_images':sum(x<0 for x in gains)})
    final=evaluated[-1]
    result={'complete':True,'verified_updates':len(logs),'curve':evaluated,
        'mean_positive_gate':final['gain_psnr']>0 and final['gain_ssim']>0,
        'sample_artifact_gate':'pending fixed P52 examples and phase replay; cannot yet authorize module comparison',
        'test_accessed':False,'checkpoint_sha256':{str(step):hashlib.sha256((out/f'step_{step}.pt').read_bytes()).hexdigest() for step in (2000,3000,4000)}}
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
