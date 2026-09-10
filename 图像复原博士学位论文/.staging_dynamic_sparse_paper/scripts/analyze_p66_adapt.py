"""Verify fixed endpoint, checkpoint and per-image adaptation deltas."""
import json,hashlib,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'results/runs/P66_public_adapt_v0'
    summary=json.loads((out/'summary.json').read_text());assert summary['complete']
    logs=[json.loads(s) for s in (out/'train.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs]==list(range(1,501))
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad']) for r in logs)
    a=json.loads((out/'val_0.json').read_text())['images'];b=json.loads((out/'val_500.json').read_text())['images']
    assert len(a)==len(b)==74 and [r['source'] for r in a]==[r['source'] for r in b]
    diffs=[y['psnr']-x['psnr'] for x,y in zip(a,b)]
    path=out/'final.pt';ckpt=torch.load(path,map_location='cpu',weights_only=True)
    assert ckpt['step']==500 and all(int(s['step'])==500 for s in ckpt['optimizer']['state'].values())
    assert all(g['lr']==1e-5 for g in ckpt['optimizer']['param_groups'])
    result={'complete':True,'verified_steps':500,'psnr_improved_images':sum(d>0 for d in diffs),
        'psnr_degraded_images':sum(d<0 for d in diffs),'min_delta':min(diffs),'max_delta':max(diffs),
        'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'optimizer_steps_verified':True,
        'passed':summary['passed'],'test_accessed':False}
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
