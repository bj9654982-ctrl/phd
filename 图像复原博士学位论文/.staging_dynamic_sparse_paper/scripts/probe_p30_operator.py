"""Three disposable training updates per variant; resource check only."""
import json
from pathlib import Path
import sys
import time
import gc
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models.reciprocal_nonlocal import ReciprocalNonlocal
from src.data import GoProPairDataset
from scripts.train_p25_restormer import network,autocast
from scripts.pilot_p24_restormer import GROUPS
from scripts.train_p12_matched_nonlocal import set_determinism
from scripts.evaluate_p21_full_image import save_json

def main():
    project=Path('J:/phd/动态稀疏非局部去模糊论文')
    out=ROOT/'results/runs/P30_operator_probe_v0';out.mkdir(parents=True,exist_ok=False)
    ds=GoProPairDataset(project/'data/GOPRO_Large','train',128,False,exclude_recording_groups=GROUPS)
    items=[ds[i] for i in range(4)]
    x=torch.stack([d['blur'] for d in items]).cuda();y=torch.stack([d['sharp'] for d in items]).cuda()
    save_json(out/'plan.json',{'names':[d['name'] for d in items],'batch':4,'patch':128,'precision':'bf16',
        'updates_per_variant':3,'variants':['fixed','content','reciprocal'],'probe_weights_discarded':True,
        'official_test_accessed':False,'quality_comparison':False})
    rows=[]
    for mode in ('fixed','content','reciprocal'):
        set_determinism(3407);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=network().train()
        ckpt=torch.load(project/'results/runs/P26_restormer_continue10000_v0/step_10000.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['model'],strict=True);del ckpt
        module=ReciprocalNonlocal(mode=mode).cuda()
        with torch.no_grad(),autocast('bf16'):baseline=model(x)
        hook=model.latent.register_forward_hook(lambda _m,_i,z:module(z))
        with torch.no_grad(),autocast('bf16'):initial=model(x)
        assert torch.equal(initial,baseline)
        del initial,baseline
        parameters=list(model.parameters())+list(module.parameters())
        opt=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats();times=[];losses=[]
        for i in range(3):
            start=time.perf_counter();opt.zero_grad(set_to_none=True)
            with autocast('bf16'):pred=model(x);loss=torch.nn.functional.l1_loss(pred.float(),y)
            assert bool(torch.isfinite(loss));loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            opt.step();torch.cuda.synchronize()
            times.append(time.perf_counter()-start);losses.append(loss.item())
        assert module.output.weight.abs().max()>0
        rows.append({'mode':mode,'module_parameters':sum(p.numel() for p in module.parameters()),
            'initial_output_exact':True,'finite_updates':3,'seconds':times,'losses_not_validation':losses,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
        print(json.dumps(rows[-1]),flush=True);save_json(out/'progress.json',rows)
        hook.remove();del model,module,opt,parameters,pred,loss;gc.collect();torch.cuda.empty_cache()
    save_json(out/'summary.json',{'complete':True,'rows':rows,'quality_claim':False,'training_checkpoint_saved':False})

if __name__=='__main__':main()
