"""Synthetic inference/reload resource audit; zero optimization steps."""
import gc, hashlib, io, json, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_p25_restormer import network
from src.p76_model import attach


def main():
    out=ROOT/'results/runs/P76_reload_v0';out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    try:
        path=ROOT/'third_party/restormer_dpdd_public_v0/single_image_defocus_deblurring.pth'
        assert hashlib.sha256(path.read_bytes()).hexdigest()=='7dce451f33f8f5e0faf7c4e3996e5dcc1bd425ecd1ada99b0f9750e490fd4c9e'
        state=torch.load(path,map_location='cpu',weights_only=True)['params']
        torch.manual_seed(7600);x=torch.rand(1,3,256,256,device='cuda');rows=[]
        for variant in ('L','N'):
            model=network().eval();model.load_state_dict(state,strict=True)
            with torch.no_grad(): base=model(x).cpu()
            model=attach(model,variant).eval()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(): pred=model(x).cpu()
            assert torch.equal(base,pred)
            peak=torch.cuda.max_memory_allocated()/2**20
            count=sum(p.numel() for p in model.parameters())
            # Exercise nonzero branch weights so a missing branch cannot pass.
            with torch.no_grad():
                model.encoder_level3.reference_branch.proj.bias.fill_(.001)
                altered=model(x).cpu()
            assert not torch.equal(altered,base)
            buffer=io.BytesIO();torch.save({'variant':variant,'model':model.state_dict()},buffer)
            del model;gc.collect();torch.cuda.empty_cache();buffer.seek(0)
            saved=torch.load(buffer,map_location='cpu',weights_only=True)
            restored=attach(network(),saved['variant']).eval()
            restored.load_state_dict(saved['model'],strict=True)
            with torch.no_grad(): replay=restored(x).cpu()
            assert torch.equal(altered,replay)
            rows.append(dict(variant=variant,exact_initial_output=True,nonzero_branch_reload_exact=True,
                             fp32_256_peak_mib=peak,parameters=count,
                             adamw_two_fp32_moments_mib=count*8/2**20))
            del restored,saved,buffer;gc.collect();torch.cuda.empty_cache()
            assert time.monotonic()-start<600
        result=dict(complete=True,rows=rows,updates=0,synthetic_only=True,seconds=time.monotonic()-start)
        (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
    except Exception as e:
        (out/'failure.json').write_text(json.dumps({'error':repr(e)}),encoding='utf-8');raise


if __name__=='__main__':main()
