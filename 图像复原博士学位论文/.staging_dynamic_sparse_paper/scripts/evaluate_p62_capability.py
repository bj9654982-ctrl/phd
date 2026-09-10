"""Isolated pretrained capability check on val only; explicit checkpoint hash."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd import DPDDPairs
from scripts.train_p25_restormer import network
from scripts.train_p51_dpdd import validate

def read8(path):
    a=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if a is None or a.dtype!=np.uint8:raise ValueError('Expected 8bit color decoding')
    a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
    h,w=a.shape[:2]
    if min(h,w)<256:raise ValueError('Image too small')
    a=a[(h-256)//2:(h-256)//2+256,(w-256)//2:(w-256)//2+256]
    return torch.from_numpy(a.copy()).permute(2,0,1).float()/255

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--sha256',required=True)
    args=ap.parse_args()
    assert hashlib.sha256(args.weights.read_bytes()).hexdigest()==args.sha256
    state=torch.load(args.weights,map_location='cpu',weights_only=True)
    if 'params' not in state:raise ValueError('Expected author params checkpoint, not local model checkpoint')
    model=network().eval();model.load_state_dict(state['params'],strict=True)
    ds=DPDDPairs('H:/phddata/dd_dp_dataset_png','H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names','val',256)
    samples=[{'source':s.name,'target':t.name,'blur':read8(s),'sharp':read8(t)} for s,t in ds.pairs]
    result=validate(model,samples)
    result.update({'weights_sha256':args.sha256,'protocol':'OpenCV default uint8 RGB /255, val center256',
        'test_accessed':False,'training':False,'own_method_result':False,
        'warning':'Public pretrained capability diagnostic, not independent test or strict full-image benchmark reproduction'})
    out=ROOT/'results/runs/P62_pretrained_capability_v0';out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='images'}))
if __name__=='__main__':main()
