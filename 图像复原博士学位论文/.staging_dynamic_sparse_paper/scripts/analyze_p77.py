"""Read-only artifact audit; write a new analysis file, no inference."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]


def main():
    run=ROOT/'results/runs/P77_three_arm_v0'
    out=run/'audit.json'
    if out.exists():raise FileExistsError(out)
    summary=json.loads((run/'summary.json').read_text())
    vals={};seqs=[];checks=[]
    for name in ('A','L','N'):
        arm=run/name
        rows=[json.loads(s) for s in (arm/'train.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,501))
        seq=[(r['source'],r['target']) for r in rows];seqs.append(seq)
        expected=summary['arms'][name]['checkpoint_sha256']
        assert hashlib.sha256((arm/'final.pt').read_bytes()).hexdigest()==expected
        ck=torch.load(arm/'final.pt',map_location='cpu',weights_only=True)
        assert ck['step']==500 and ck['variant']==name
        assert all(int(v['step'])==500 for v in ck['optimizer']['state'].values())
        assert all(g['lr']==1e-5 for g in ck['optimizer']['param_groups'])
        assert all(torch.isfinite(v).all() for v in ck['model'].values())
        checks.append(dict(variant=name,checkpoint_sha256=expected,optimizer_states=len(ck['optimizer']['state'])))
        del ck
        vals[name]=json.loads((arm/'val_500.json').read_text())['images']
        assert len(vals[name])==74
    assert seqs[0]==seqs[1]==seqs[2]
    paired={}
    for name in ('A','L'):
        ds=[]
        for a,n in zip(vals[name],vals['N']):
            assert a['source']==n['source']
            ds.append(n['psnr']-a['psnr'])
        paired[name]=dict(mean=sum(ds)/74,positive=sum(d>0 for d in ds),negative=sum(d<0 for d in ds),min=min(ds),max=max(ds))
        assert abs(paired[name]['mean']-summary['deltas'][name]['psnr'])<1e-10
    result=dict(complete=True,checks=checks,same_500_sample_sequences=True,same_74_source_ids=True,paired=paired,passed=summary['passed'])
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
