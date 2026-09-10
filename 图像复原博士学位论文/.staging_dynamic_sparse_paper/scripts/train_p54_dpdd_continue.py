"""P54: frozen-budget continuation from absolute step 1000 to 4000."""
import hashlib,json,sys,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd import DPDDPairs
from scripts.train_p25_restormer import network,update,StepBatches
from scripts.train_p51_dpdd import validate,save
from scripts.train_p12_matched_nonlocal import set_determinism

def main():
    out=ROOT/'results/runs/P54_dpdd_continue_v0';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();step=1000;model=optimizer=None
    def checkpoint(path):
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'plan':plan},path)
    try:
        source=ROOT/'results/runs/P51_dpdd_baseline_v1/final.pt'
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest=='aeb0a339a60c72ae159fec4d9e44fc45247d1be5941847b545255494ce9f1467'
        state=torch.load(source,map_location='cpu',weights_only=False);assert state['step']==1000
        plan={'start':1000,'stop':4000,'budget_seconds':3600,'source_sha256':digest,
            'batch':4,'patch':128,'val_center':256,'val_count':74,'precision':'bf16',
            'lr':3e-4,'weight_decay':1e-4,'clip':1,'test_accessed':False,
            'optimizer_retained':True,'absolute_sampling':True,'paper_result':False,
            'code_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__),ROOT/'src/dpdd.py',ROOT/'scripts/train_p25_restormer.py',ROOT/'scripts/train_p51_dpdd.py']},
            'list_hashes':state['plan']['list_hashes']}
        save(out/'plan.json',plan)
        lists=Path('H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names')
        for name,h in plan['list_hashes'].items():assert hashlib.sha256((lists/name).read_bytes()).hexdigest()==h
        train=DPDDPairs('H:/phddata/dd_dp_dataset_png',lists,'train',128)
        dev=DPDDPairs('H:/phddata/dd_dp_dataset_png',lists,'val',256)
        samples=[dev[i] for i in range(len(dev))]
        set_determinism(3407);model=network().train();model.load_state_dict(state['model'],strict=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        optimizer.load_state_dict(state['optimizer'])
        assert len(optimizer.state)==len(list(model.parameters()))
        assert all(int(s['step'])==1000 for s in optimizer.state.values())
        assert all(g['lr']==3e-4 and g['weight_decay']==1e-4 for g in optimizer.param_groups)
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        curve=[]
        def evaluate():
            result=validate(model,samples);save(out/f'val_{step}.json',result)
            row={'step':step,**{k:v for k,v in result.items() if k!='images'}}
            curve.append(row);save(out/'curve.json',curve);print('VAL '+json.dumps(row),flush=True)
        evaluate()
        prior=json.loads((source.parent/'summary.json').read_text())['curve'][-1]
        assert abs(curve[0]['psnr']-prior['psnr'])<1e-4 and abs(curve[0]['ssim']-prior['ssim'])<1e-5
        torch.cuda.reset_peak_memory_stats()
        loader=DataLoader(train,batch_sampler=StepBatches(len(train),4,1000,4000),num_workers=4,pin_memory=True,persistent_workers=True)
        with (out/'train.jsonl').open('w',encoding='utf-8') as log:
            for batch in loader:
                if time.monotonic()-started>3600:raise TimeoutError('One-hour budget reached')
                loss,grad=update(model,optimizer,batch['blur'].cuda(non_blocking=True),batch['sharp'].cuda(non_blocking=True),'bf16');step+=1
                row={'step':step,'loss':loss,'grad':grad,'seconds':time.monotonic()-started,'source':batch['source'],'target':batch['target']}
                log.write(json.dumps(row)+'\n');log.flush()
                if step%100==0:save(out/'progress.json',row);print('TRAIN '+json.dumps(row),flush=True)
                if step%1000==0:
                    assert all(int(s['step'])==step for s in optimizer.state.values())
                    evaluate();checkpoint(out/f'step_{step}.pt')
        save(out/'summary.json',{'complete':True,'step':step,'curve':curve,'seconds':time.monotonic()-started,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'test_accessed':False,
            'gain_psnr':curve[-1]['psnr']-curve[-1]['input_psnr'],'gain_ssim':curve[-1]['ssim']-curve[-1]['input_ssim']})
        print('COMPLETE',flush=True)
    except Exception as e:
        save(out/'failure.json',{'error':repr(e),'step':step})
        if model is not None and optimizer is not None:checkpoint(out/'interrupted.pt')
        raise
if __name__=='__main__':main()
