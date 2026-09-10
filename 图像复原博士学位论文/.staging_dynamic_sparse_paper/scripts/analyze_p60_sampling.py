"""Verify completed matched sampling experiment and fixed gate."""
import json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'results/runs/P60_sampling_repeat_v0'
    s=json.loads((out/'summary.json').read_text());assert s['complete']
    logs={r:[json.loads(x) for x in (out/r/'train.jsonl').read_text().splitlines()] for r in ('A','B')}
    for rows in logs.values():
        assert [r['step'] for r in rows]==list(range(4001,5001))
        assert all(math.isfinite(r['loss']) and math.isfinite(r['grad']) for r in rows)
    assert all(a['source']==b['source'] and a['target']==b['target'] for a,b in zip(logs['A'],logs['B']))
    initial=[json.loads((out/r/'val_4000.json').read_text()) for r in ('A','B')]
    assert initial[0]==initial[1]
    vals={r:json.loads((out/r/'val_5000.json').read_text()) for r in ('A','B')}
    aa,bb=vals['A']['images'],vals['B']['images'];assert len(aa)==len(bb)==74
    assert [x['source'] for x in aa]==[x['source'] for x in bb]==[x['source'] for x in initial[0]['images']]
    result={'complete':True,'same_image_order':True,'same_initial_validation':True,'updates_each':1000,
        'B_minus_A_psnr':vals['B']['psnr']-vals['A']['psnr'],'B_minus_A_ssim':vals['B']['ssim']-vals['A']['ssim'],
        'B_better_images':sum(b['psnr']>a['psnr'] for a,b in zip(aa,bb)),
        'versus_input':{r:{'psnr':v['psnr']-v['input_psnr'],'ssim':v['ssim']-v['input_ssim']} for r,v in vals.items()},
        'checkpoint_hashes':{r:hashlib.sha256((out/r/'step_5000.pt').read_bytes()).hexdigest() for r in ('A','B')},
        'passed':s['passed'],'test_accessed':False}
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
