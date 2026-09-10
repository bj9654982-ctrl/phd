"""Read-only training-only TMT sample audit. No test images or model inference."""
import json,hashlib,time
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def main():
    base=Path('H:/phddata/TMT/turb_syn_static/syn_static/train_turb')
    scenes=sorted(p for p in base.iterdir() if p.is_dir())
    selected=[scenes[i] for i in (0,len(scenes)//2,len(scenes)-1)]
    rows=[]
    for p in selected:
        blur=sorted((p/'blur').glob('*.jpg'));turb=sorted((p/'turb').glob('*.jpg'))
        row={'scene':p.name,'blur_count':len(blur),'turb_count':len(turb),'images':[]}
        for path in (p/'gt.jpg',blur[0],turb[0]):
            started=time.perf_counter()
            with Image.open(path) as im:
                im.load();size=list(im.size);mode=im.mode
            row['images'].append({'path':str(path),'size':size,'mode':mode,'decode_seconds':time.perf_counter()-started,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        row['metadata_files']=[str(f) for f in p.iterdir() if f.is_file() and f.name!='gt.jpg']
        rows.append(row)
    result={'train_scene_count':len(scenes),'sample_selection':'sorted indices 0, midpoint, last',
            'samples':rows,'full_pair_integrity_audited':False,'test_images_accessed':False,'training_started':False}
    out=ROOT/'results/runs/P48_tmt_audit_v0';out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
