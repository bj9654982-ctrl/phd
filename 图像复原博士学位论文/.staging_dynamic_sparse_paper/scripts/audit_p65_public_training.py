"""No optimizer step: uint8 protocol tests and shared-init resource check."""
import hashlib,json,sys,time,gc
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd8 import DPDD8Pairs,crop8
from scripts.evaluate_p62_capability import read8
from scripts.train_p25_restormer import network

def main():
    out=ROOT/'results/runs/P65_public_resource_v0';out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    try:
        a=np.arange(256,dtype=np.uint8).reshape(16,16,1).repeat(3,axis=2)
        x,y=crop8(a,a,16)
        assert torch.equal(x,y) and torch.equal(x,torch.from_numpy(a.copy()).permute(2,0,1).float()/255)
        data='H:/phddata/dd_dp_dataset_png';lists='H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names'
        train=DPDD8Pairs(data,lists,'train',128);val=DPDD8Pairs(data,lists,'val',256)
        assert torch.equal(val[0]['blur'],read8(val.pairs[0][0]))
        assert torch.equal(train[(0,65)]['blur'],train[(0,65)]['blur'])
        try:DPDD8Pairs(data,lists,'test',128)
        except ValueError:pass
        else:raise AssertionError('Test split not sealed')
        samples=[train[(i,65000+i)] for i in (0,116,232,349)]
        x=torch.stack([s['blur'] for s in samples]).cuda();y=torch.stack([s['sharp'] for s in samples]).cuda()
        path=ROOT/'third_party/restormer_dpdd_public_v0/single_image_defocus_deblurring.pth'
        assert hashlib.sha256(path.read_bytes()).hexdigest()=='7dce451f33f8f5e0faf7c4e3996e5dcc1bd425ecd1ada99b0f9750e490fd4c9e'
        state=torch.load(path,map_location='cpu',weights_only=True)['params'];outputs=[];rows=[]
        for branch in ('A','B'):
            model=network().train();model.load_state_dict(state,strict=True)
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p=model(x);loss=(p.float()-y).abs().mean()
            assert torch.isfinite(loss)
            outputs.append(p.detach().cpu());loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items())
            rows.append({'branch':branch,'l1':float(loss),'grad_norm':float(norm),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
            del p,loss,model;gc.collect();torch.cuda.empty_cache()
        assert torch.equal(outputs[0],outputs[1])
        result={'complete':True,'rows':rows,'seconds':time.monotonic()-start,'updates':0,'test_accessed':False,
            'same_initial_outputs':True,'weights_unchanged':True,'checks':'256 levels /255, val reader parity, repeatable crop, sealed test',
            'no_optimizer_created':True,'gradients_discarded':True}
        (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
    except Exception as e:
        (out/'failure.json').write_text(json.dumps({'error':repr(e)}));raise
if __name__=='__main__':main()
