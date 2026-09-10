"""Matched all/filtered candidate sampling; two bounded continuation branches."""
import gc,hashlib,json,sys,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd import DPDDPairs
from src.dpdd_sampling import DPDDCandidatePairs
from scripts.train_p25_restormer import network,update
from src.p60_sampler import StepBatchesRepeat as StepBatches
from scripts.train_p51_dpdd import validate,save
from scripts.train_p12_matched_nonlocal import set_determinism

def main():
    out=ROOT/'results/runs/P60_sampling_repeat_v0';out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();step=4000;role='setup';model=opt=None
    try:
        source=ROOT/'results/runs/P54_dpdd_continue_v0/step_4000.pt'
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest=='5cd6cca38116299ece7ed387e1517291cd939fb2691796e39208880d03d80f54'
        manifest=ROOT/'results/runs/P57_sampling_cache_v0/manifest.json'
        plan={'sampler_order_seed':4408,'sampler_crop_seed':200000,'A':'all512','B':'filtered512','start':4000,'stop':5000,'source_hash':digest,
            'manifest_hash':hashlib.sha256(manifest.read_bytes()).hexdigest(),'budget_seconds':3600,
            'batch':4,'patch':128,'lr':3e-4,'precision':'bf16','test_accessed':False,
            'code_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__),ROOT/'src/p60_sampler.py',ROOT/'src/dpdd_sampling.py',ROOT/'src/dpdd.py',ROOT/'scripts/train_p25_restormer.py',ROOT/'scripts/train_p51_dpdd.py']}}
        save(out/'plan.json',plan)
        lists=Path('H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names')
        data='H:/phddata/dd_dp_dataset_png'
        dev=DPDDPairs(data,lists,'val',256);samples=[dev[i] for i in range(len(dev))]
        baseline=json.loads((source.parent/'val_4000.json').read_text())
        results={}
        for role,filtered in [('A',False),('B',True)]:
            branch=out/role;branch.mkdir();step=4000
            train=DPDDCandidatePairs(data,lists,manifest,filtered)
            state=torch.load(source,map_location='cpu',weights_only=False)
            set_determinism(3407);model=network().train();model.load_state_dict(state['model'])
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);opt.load_state_dict(state['optimizer'])
            assert all(int(s['step'])==4000 for s in opt.state.values())
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
            curve=[]
            def evaluate():
                v=validate(model,samples);save(branch/f'val_{step}.json',v)
                row={'step':step,**{k:v for k,v in v.items() if k!='images'}}
                curve.append(row);save(branch/'curve.json',curve);print(role+' VAL '+json.dumps(row),flush=True)
            evaluate();assert abs(curve[0]['psnr']-baseline['psnr'])<1e-4 and abs(curve[0]['ssim']-baseline['ssim'])<1e-5
            loader=DataLoader(train,batch_sampler=StepBatches(len(train),4,4000,5000),num_workers=4,pin_memory=True,persistent_workers=False)
            with (branch/'train.jsonl').open('w',encoding='utf-8') as log:
                for batch in loader:
                    if time.monotonic()-start>3600:raise TimeoutError('Two-branch one-hour budget')
                    loss,grad=update(model,opt,batch['blur'].cuda(non_blocking=True),batch['sharp'].cuda(non_blocking=True),'bf16');step+=1
                    row={'role':role,'step':step,'loss':loss,'grad':grad,'source':batch['source'],'target':batch['target'],'seconds':time.monotonic()-start}
                    log.write(json.dumps(row)+'\n');log.flush()
                    if step%100==0:save(out/'progress.json',row);print(json.dumps(row),flush=True)
                    if step%500==0:
                        evaluate()
                        assert all(int(s['step'])==step for s in opt.state.values())
                        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'plan':plan,
                            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},branch/f'step_{step}.pt')
            results[role]=curve;save(out/'partial.json',results)
            del loader,model,opt;model=opt=None;gc.collect();torch.cuda.empty_cache()
        a,b=results['A'][-1],results['B'][-1]
        gain=b['psnr']-a['psnr'];ssim=b['ssim']-a['ssim']
        save(out/'summary.json',{'complete':True,'results':results,'seconds':time.monotonic()-start,
            'B_minus_A_psnr':gain,'B_minus_A_ssim':ssim,'passed':gain>=.02 and ssim>=0 and b['psnr']>b['input_psnr'] and b['ssim']>b['input_ssim'],
            'test_accessed':False,'paper_result':False})
        print('COMPLETE',flush=True)
    except Exception as e:
        save(out/'failure.json',{'error':repr(e),'role':role,'step':step})
        if model is not None and opt is not None:torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'role':role},out/'interrupted.pt')
        raise
if __name__=='__main__':main()
