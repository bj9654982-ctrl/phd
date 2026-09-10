"""Preregistered optimizer-state comparison; run after both branches complete."""
import json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_p16_factorial import read_csv,write_csv,sha256
from scripts.evaluate_p21_full_image import save_json
from src.data.gopro_pairs import recording_group

def main():
    run=ROOT/'results/runs/P43_optimizer_control_v0';out=ROOT/'results/runs/P43_analysis_v0'
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'));assert summary['complete']
    data={}
    for role in 'RS':
        assert [int(r['step']) for r in read_csv(run/role/'training.csv')]==list(range(1,1001))
        for step in (0,1000):
            rows=read_csv(run/role/f'validation_{step:04d}.csv')
            data[role,step]={r['name']:r for r in rows};assert len(rows)==len(data[role,step])==385
    names=set(data['R',0]);assert all(set(d)==names for d in data.values())
    for result in summary['results']:
        for step,key in ((0,'before'),(1000,'after')):
            for m in ('psnr_db','ssim'):
                assert abs(statistics.fmean(float(r[m]) for r in data[result['role'],step].values())-result[key][m])<1e-8
    replay={}
    for step in (0,1000):
        previous={r['name']:r for r in read_csv(ROOT/f'results/runs/P40_context_screen_v0/A/validation_{step:04d}.csv')}
        assert set(previous)==names
        replay[str(step)]={m:max(abs(float(previous[n][m])-float(data['R',step][n][m])) for n in names) for m in ('psnr_db','ssim')}
    pairs=[]
    for n in sorted(names):
        row={'name':n,'group':recording_group(n.split('/')[0])}
        for m in ('psnr_db','ssim'):
            base=float(data['R',0][n][m]);assert base==float(data['S',0][n][m])
            r=float(data['R',1000][n][m]);s=float(data['S',1000][n][m])
            row.update({f'S_minus_R_{m}':s-r,f'S_minus_base_{m}':s-base,f'R_minus_base_{m}':r-base})
        pairs.append(row)
    keys=[k for k in pairs[0] if k not in ('name','group')]
    def agg(rows):return {'count':len(rows),**{k:statistics.fmean(r[k] for r in rows) for k in keys}}
    overall=agg(pairs);groups={g:agg([r for r in pairs if r['group']==g]) for g in sorted({r['group'] for r in pairs})}
    checks={'overall':overall['S_minus_R_psnr_db']>=.02 and overall['S_minus_R_ssim']>=0,
            'groups':all(r['S_minus_R_psnr_db']>=0 for r in groups.values())}
    out.mkdir(parents=True,exist_ok=False);write_csv(out/'paired.csv',pairs)
    result={'complete':True,'gate_passed':all(checks.values()),'checks':checks,'overall':overall,'groups':groups,
            'R_replay_max_error':replay,'source_hash':sha256(run/'summary.json'),
            'protocol_hash':sha256(ROOT/'docs/49_P42优化器核验与P43协议.md'),'paper_result':False}
    save_json(out/'summary.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
