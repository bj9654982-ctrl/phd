"""Preregistered A/L/N 500-step screen; sequential, no test access."""
import gc, hashlib, json, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd8 import DPDD8Pairs
from src.p76_model import attach
from scripts.train_p25_restormer import network, update, StepBatches
from scripts.train_p51_dpdd import validate, save
from scripts.train_p12_matched_nonlocal import set_determinism


def main():
    out=ROOT/'results/runs/P77_three_arm_v0';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();variant=None;step=0;model=opt=None
    try:
        path=ROOT/'third_party/restormer_dpdd_public_v0/single_image_defocus_deblurring.pth'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest=='7dce451f33f8f5e0faf7c4e3996e5dcc1bd425ecd1ada99b0f9750e490fd4c9e'
        state=torch.load(path,map_location='cpu',weights_only=True)['params']
        lists=Path('H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names')
        files=[Path(__file__),ROOT/'src/p76_model.py',ROOT/'src/reference_nlfe.py',ROOT/'src/dpdd8.py',ROOT/'src/dpdd.py',ROOT/'scripts/train_p25_restormer.py',ROOT/'scripts/train_p51_dpdd.py']
        plan=dict(source_hash=digest,steps=500,batch=4,patch=128,lr=1e-5,weight_decay=1e-4,
                  precision='bf16',val_precision='fp32',seed=3407,branch_seed=7501,
                  variants=['A','L','N'],seconds_per_arm=900,total_seconds=2700,
                  gate='N-A and N-L PSNR >= .02; both SSIM deltas >= 0',test_accessed=False,
                  code_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  list_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in lists.glob('*.npy') if not p.name.startswith('test')})
        save(out/'plan.json',plan)
        train=DPDD8Pairs('H:/phddata/dd_dp_dataset_png',lists,'train',128)
        dev=DPDD8Pairs('H:/phddata/dd_dp_dataset_png',lists,'val',256)
        samples=[dev[i] for i in range(len(dev))]
        ref=json.loads((ROOT/'results/runs/P62_pretrained_capability_v0/summary.json').read_text())
        results={};sequences=[]
        for variant in ('A','L','N'):
            arm=out/variant;arm.mkdir();tick=time.monotonic();step=0
            set_determinism(3407);model=network();model.load_state_dict(state,strict=True)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(7501);model=attach(model,variant)
            model.train();opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
            curve=[];sequence=hashlib.sha256();torch.cuda.reset_peak_memory_stats()
            def budget():
                if time.monotonic()-tick>900 or time.monotonic()-started>2700:
                    raise TimeoutError('preregistered budget exceeded')
            def evaluate():
                budget();v=validate(model,samples);budget();save(arm/f'val_{step}.json',v)
                curve.append({'step':step,**{k:x for k,x in v.items() if k!='images'}})
                save(arm/'curve.json',curve);print(variant+' VAL '+json.dumps(curve[-1]),flush=True)
            evaluate()
            assert abs(curve[0]['psnr']-ref['psnr'])<1e-4 and abs(curve[0]['ssim']-ref['ssim'])<1e-5
            loader=DataLoader(train,batch_sampler=StepBatches(len(train),4,0,500),num_workers=4,pin_memory=True)
            with (arm/'train.jsonl').open('w',encoding='utf-8') as log:
                for batch in loader:
                    budget()
                    loss,grad=update(model,opt,batch['blur'].cuda(non_blocking=True),batch['sharp'].cuda(non_blocking=True),'bf16');step+=1
                    names={'step':step,'source':batch['source'],'target':batch['target']}
                    sequence.update(json.dumps(names,sort_keys=True).encode())
                    row={**names,'loss':loss,'grad':grad,'seconds':time.monotonic()-tick}
                    log.write(json.dumps(row)+'\n');log.flush()
                    if step%50==0:
                        save(out/'progress.json',{'variant':variant,**row});print(variant+' '+str(step),flush=True)
                    if step%250==0:evaluate()
            assert step==500 and len(opt.state)==len(list(model.parameters()))
            assert all(int(v['step'])==500 for v in opt.state.values())
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'variant':variant,'plan':plan,
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},arm/'final.pt')
            sequences.append(sequence.hexdigest())
            results[variant]=dict(complete=True,curve=curve,sequence_sha256=sequence.hexdigest(),
                                  checkpoint_sha256=hashlib.sha256((arm/'final.pt').read_bytes()).hexdigest(),
                                  peak_mib=torch.cuda.max_memory_allocated()/2**20,seconds=time.monotonic()-tick)
            save(arm/'summary.json',results[variant]);budget()
            del model,opt,loader;model=opt=None;gc.collect();torch.cuda.empty_cache()
        assert len(set(sequences))==1
        deltas={v:{k:results['N']['curve'][-1][k]-results[v]['curve'][-1][k] for k in ('psnr','ssim')} for v in ('A','L')}
        save(out/'summary.json',dict(complete=True,arms=results,deltas=deltas,
             passed=all(d['psnr']>=.02 and d['ssim']>=0 for d in deltas.values()),
             test_accessed=False,new_method=False,seconds=time.monotonic()-started))
        print('COMPLETE',flush=True)
    except Exception as e:
        save(out/'failure.json',dict(error=repr(e),variant=variant,step=step));raise


if __name__=='__main__':main()
