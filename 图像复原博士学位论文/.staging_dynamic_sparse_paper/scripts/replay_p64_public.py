"""Fixed four development examples of public model, 8bit protocol only."""
import json,hashlib,sys
from pathlib import Path
import torch,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_p62_capability import read8
from scripts.train_p25_restormer import network
from scripts.train_p12_matched_nonlocal import _psnr_value
from scripts.audit_p53_precision import phase
from src.dpdd import DPDDPairs
def main():
    out=ROOT/'results/runs/P64_public_samples_v0';out.mkdir(parents=True,exist_ok=False)
    path=ROOT/'third_party/restormer_dpdd_public_v0/single_image_defocus_deblurring.pth'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='7dce451f33f8f5e0faf7c4e3996e5dcc1bd425ecd1ada99b0f9750e490fd4c9e'
    model=network().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['params'])
    ds=DPDDPairs('H:/phddata/dd_dp_dataset_png','H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names','val',256)
    pairs={s.name:(s,t) for s,t in ds.pairs}
    old=json.loads((ROOT/'results/runs/P52_dpdd_samples_v0/summary.json').read_text())['samples']
    reference=json.loads((ROOT/'results/runs/P62_pretrained_capability_v0/summary.json').read_text())['images']
    expected={r['source']:r for r in reference}
    sheet=Image.new('RGB',(1024,1200),'white');draw=ImageDraw.Draw(sheet);rows=[]
    with torch.no_grad():
        for j,r in enumerate(old):
            s,t=pairs[r['source']];x,y=read8(s)[None].cuda(),read8(t)[None].cuda();p=model(x).clamp(0,1)
            score=_psnr_value(p,y);assert abs(score-expected[s.name]['psnr'])<1e-4
            row={'source':s.name,'psnr':score,'input_psnr':_psnr_value(x,y),'gain':score-_psnr_value(x,y),
                'phase2':phase(p-x,2),'phase4':phase(p-x,4)};rows.append(row)
            draw.text((4,j*300+4),f'{s.name}: gain={row["gain"]:+.3f}dB',fill='black')
            for col,(label,a) in enumerate(zip(['input8','public output','GT8','abs residual x8'],[x[0],p[0],y[0],(p-x)[0].abs()*8])):
                im=Image.fromarray((a.clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8))
                sheet.paste(im,(col*256,j*300+44));draw.text((col*256+4,j*300+24),label,fill='black')
    sheet.save(out/'comparison.png')
    result={'complete':True,'samples':rows,'test_accessed':False,'own_method':False,'protocol':'8bit val center256',
        'all74_positive':sum(r['psnr']>r['input_psnr'] for r in reference)}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
