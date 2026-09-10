"""Evaluate the preregistered screening gate without changing thresholds."""
import json,statistics
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_p16_factorial import read_csv,write_csv,sha256
from scripts.evaluate_p21_full_image import save_json
from src.data.gopro_pairs import recording_group

def main():
    run=ROOT/'results/runs/P47_context_retained_v0';out=ROOT/'results/runs/P47_analysis_v0'
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'));assert summary['complete']
    data={}
    for role in 'ACP':
        rows=read_csv(run/role/'validation_1000.csv');assert len(rows)==385
        data[role]={r['name']:r for r in rows};assert len(data[role])==385
        assert (run/role/'last.pt').is_file()
        training=read_csv(run/role/'training.csv')
        assert [int(r['step']) for r in training]==list(range(1,1001))
        initial=read_csv(run/role/'validation_0000.csv')
        assert len(initial)==len({r['name'] for r in initial})==385
        initial={r['name']:r for r in initial}
        assert set(initial)==set(data[role])
        if role=='A': base_initial=initial
        else:
            for name in initial:
                for metric in ('psnr_db','ssim'):
                    assert float(initial[name][metric])==float(base_initial[name][metric])
    assert all(set(data[r])==set(data['A']) for r in 'CP')
    for r in summary['results']:
        for metric in ('psnr_db','ssim'):
            assert abs(statistics.fmean(float(row[metric]) for row in data[r['role']].values())-r['after'][metric])<1e-8
    out.mkdir(parents=True,exist_ok=False)
    paired=[]
    for name in sorted(data['A']):
        row={'name':name,'group':recording_group(name.split('/')[0])}
        for other in 'AC':
            for metric in ('psnr_db','ssim'):
                row[f'P_minus_{other}_{metric}']=float(data['P'][name][metric])-float(data[other][name][metric])
        paired.append(row)
    keys=[k for k in paired[0] if k not in ('name','group')]
    def agg(rows):return {'count':len(rows),**{k:statistics.fmean(r[k] for r in rows) for k in keys}}
    overall=agg(paired);groups={g:agg([r for r in paired if r['group']==g]) for g in sorted({r['group'] for r in paired})}
    checks={f'P_vs_{r}_overall':overall[f'P_minus_{r}_psnr_db']>=.02 and overall[f'P_minus_{r}_ssim']>=0 for r in 'AC'}
    checks['all_groups_P_vs_C_nonnegative']=all(r['P_minus_C_psnr_db']>=0 for r in groups.values())
    p_result=next(r for r in summary['results'] if r['role']=='P')
    gains={m:p_result['after'][m]-p_result['before'][m] for m in ('psnr_db','ssim')}
    checks['P_vs_start_nonnegative']=all(v>=0 for v in gains.values())
    write_csv(out/'paired.csv',paired)
    save_json(out/'summary.json',{'complete':True,'gate_passed':all(checks.values()),'checks':checks,
        'overall':overall,'groups':groups,'P_vs_start':gains,'source_summary_hash':sha256(run/'summary.json'),
        'protocol_hash':sha256(ROOT/'docs/53_P46真实状态核验与P47协议.md'),'paper_result':False})
    print(json.dumps({'checks':checks,'overall':overall},indent=2))

if __name__=='__main__':main()
