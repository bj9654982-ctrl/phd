"""Paired development analysis, unchanged preregistered gate."""
import json, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_p16_factorial import read_csv,write_csv,sha256
from scripts.evaluate_p21_full_image import save_json
from src.data.gopro_pairs import recording_group

def main():
    run=ROOT/'results/runs/P34_frozen_v0';out=ROOT/'results/runs/P34_analysis_v0'
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'));assert summary['complete']
    data={}
    for role in 'CD':
        training=read_csv(run/role/'training.csv');assert [int(r['step']) for r in training]==list(range(1,1001))
        for step in (0,1000):
            rows=read_csv(run/role/f'validation_{step:04d}.csv')
            data[role,step]={r['name']:r for r in rows};assert len(rows)==len(data[role,step])==385
    names=set(data['C',0]);assert all(set(rows)==names for rows in data.values())
    for result in summary['results']:
        assert result['backbone_unchanged'] and result['initial_equivalent'] and result['module_initial_identical']
        for step,key in ((0,'before'),(1000,'after')):
            for metric in ('psnr_db','ssim'):
                actual=statistics.fmean(float(r[metric]) for r in data[result['role'],step].values())
                assert abs(actual-result[key][metric])<1e-8
    paired=[]
    for name in sorted(names):
        row={'name':name,'group':recording_group(name.split('/')[0])}
        for metric in ('psnr_db','ssim'):
            base=float(data['C',0][name][metric]);assert base==float(data['D',0][name][metric])
            c=float(data['C',1000][name][metric]);d=float(data['D',1000][name][metric])
            row.update({f'C_minus_base_{metric}':c-base,f'D_minus_base_{metric}':d-base,f'D_minus_C_{metric}':d-c})
        paired.append(row)
    keys=[k for k in paired[0] if k not in ('name','group')]
    def agg(rows):return {'count':len(rows),**{k:statistics.fmean(r[k] for r in rows) for k in keys}}
    overall=agg(paired);groups={g:agg([r for r in paired if r['group']==g]) for g in sorted({r['group'] for r in paired})}
    checks={f'D_vs_{r}':overall[f'D_minus_{r}_psnr_db']>=.02 and overall[f'D_minus_{r}_ssim']>=0 for r in ('base','C')}
    checks['all_groups_D_vs_C_nonnegative']=all(r['D_minus_C_psnr_db']>=0 for r in groups.values())
    out.mkdir(parents=True,exist_ok=False);write_csv(out/'paired.csv',paired)
    result={'complete':True,'gate_passed':all(checks.values()),'checks':checks,'overall':overall,'groups':groups,
            'source_hash':sha256(run/'summary.json'),'protocol_hash':sha256(ROOT/'docs/39_P33语义验证与冻结主干对照协议.md'),
            'paper_result':False,'official_test_accessed':False}
    save_json(out/'summary.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
