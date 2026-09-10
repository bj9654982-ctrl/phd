"""Paired development analysis, unchanged preregistered gate."""
import json, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_p16_factorial import read_csv,write_csv,sha256
from scripts.evaluate_p21_full_image import save_json
from src.data.gopro_pairs import recording_group

def main():
    run=ROOT/'results/runs/P37_normalization_v0';out=ROOT/'results/runs/P37_analysis_v0'
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'));assert summary['complete']
    data={}
    for role in 'DN':
        training=read_csv(run/role/'training.csv');assert [int(r['step']) for r in training]==list(range(1,1001))
        for step in (0,1000):
            rows=read_csv(run/role/f'validation_{step:04d}.csv')
            data[role,step]={r['name']:r for r in rows};assert len(rows)==len(data[role,step])==385
    names=set(data['D',0]);assert all(set(rows)==names for rows in data.values())
    for result in summary['results']:
        assert result['backbone_unchanged'] and result['initial_equivalent'] and result['module_initial_identical']
        for step,key in ((0,'before'),(1000,'after')):
            for metric in ('psnr_db','ssim'):
                actual=statistics.fmean(float(r[metric]) for r in data[result['role'],step].values())
                assert abs(actual-result[key][metric])<1e-8
    replay={}
    for step in (0,1000):
        old={r['name']:r for r in read_csv(ROOT/f'results/runs/P34_frozen_v0/D/validation_{step:04d}.csv')}
        assert set(old)==names
        replay[str(step)]={m:max(abs(float(old[n][m])-float(data['D',step][n][m])) for n in names) for m in ('psnr_db','ssim')}
    historical=read_csv(ROOT/'results/runs/P34_frozen_v0/C/validation_1000.csv')
    historical_mean={m:statistics.fmean(float(r[m]) for r in historical) for m in ('psnr_db','ssim')}
    paired=[]
    for name in sorted(names):
        row={'name':name,'group':recording_group(name.split('/')[0])}
        for metric in ('psnr_db','ssim'):
            base=float(data['D',0][name][metric]);assert base==float(data['N',0][name][metric])
            c=float(data['D',1000][name][metric]);d=float(data['N',1000][name][metric])
            row.update({f'D_minus_base_{metric}':c-base,f'N_minus_base_{metric}':d-base,f'N_minus_D_{metric}':d-c})
        paired.append(row)
    keys=[k for k in paired[0] if k not in ('name','group')]
    def agg(rows):return {'count':len(rows),**{k:statistics.fmean(r[k] for r in rows) for k in keys}}
    overall=agg(paired);groups={g:agg([r for r in paired if r['group']==g]) for g in sorted({r['group'] for r in paired})}
    checks={f'N_vs_{r}':overall[f'N_minus_{r}_psnr_db']>=.02 and overall[f'N_minus_{r}_ssim']>=0 for r in ('base','D')}
    checks['all_groups_N_vs_D_nonnegative']=all(r['N_minus_D_psnr_db']>=0 for r in groups.values())
    out.mkdir(parents=True,exist_ok=False);write_csv(out/'paired.csv',paired)
    result={'complete':True,'gate_passed':all(checks.values()),'checks':checks,'overall':overall,'groups':groups,
            'source_hash':sha256(run/'summary.json'),'protocol_hash':sha256(ROOT/'docs/42_P36归一化验证与P37协议.md'),
            'D_replay_max_error':replay,'historical_C_mean':historical_mean,'paper_result':False,'official_test_accessed':False}
    save_json(out/'summary.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
