"""Summarize completed development experiment without selecting test samples."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/runs/P51_dpdd_baseline_v1'
s=json.loads((out/'summary.json').read_text())
assert s['complete'] and s['step']==1000
rows=json.loads((out/'val_1000.json').read_text())['images']
logs=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()]
assert [r['step'] for r in logs]==list(range(1,1001))
assert len(rows)==74 and len({r['source'] for r in rows})==74
for r in rows:
    r['gain_psnr']=r['psnr']-r['input_psnr']
a={'positive_psnr_images':sum(r['gain_psnr']>0 for r in rows),
   'negative_psnr_images':sum(r['gain_psnr']<0 for r in rows),
   'best_gain':max(rows,key=lambda r:r['gain_psnr']),
   'worst_gain':min(rows,key=lambda r:r['gain_psnr']),
   'training_steps_verified':len(logs),'test_accessed':False}
(out/'analysis.json').write_text(json.dumps(a,indent=2),encoding='utf-8')
print(json.dumps({'summary':s,'analysis':a},indent=2))
