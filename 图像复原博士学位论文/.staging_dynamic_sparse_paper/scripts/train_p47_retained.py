"""Three bounded matched development-screening runs; no independent-test claim."""
import gc,json,time,traceback
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_p25_restormer import network,autocast,SeededPairs,StepBatches,validate
from scripts.train_p12_matched_nonlocal import set_determinism
from scripts.pilot_p24_restormer import GROUPS
from scripts.analyze_p16_factorial import sha256,write_csv
from scripts.evaluate_p21_full_image import save_json
from src.models.reciprocal_nonlocal import ReciprocalNonlocal
from src.data import GoProPairDataset
from src.models.pooled_context import PooledContext
from src.optimizer_transfer import transfer_adamw

def main():
    project=Path('J:/phd/动态稀疏非局部去模糊论文')
    out=ROOT/'results/runs/P47_context_retained_v0';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();checkpoint=project/'results/runs/P26_restormer_continue10000_v0/step_10000.pt'
    try:
        train=GoProPairDataset(project/'data/GOPRO_Large','train',128,True,exclude_recording_groups=GROUPS)
        dev=GoProPairDataset(project/'data/GOPRO_Large','train',256,False,include_recording_groups=GROUPS)
        assert len(train)==1718 and len(dev)==385 and not set(train.recording_groups)&set(dev.recording_groups)
        plan={'roles':{'A':'baseline','C':'content','P':'pooled'},'steps_each':1000,'batch':4,
          'precision':'bf16','trainable_backbone':True,'checkpoint_hash':sha256(checkpoint),
          'script_hash':sha256(Path(__file__)),'module_hash':sha256(ROOT/'src/models/reciprocal_nonlocal.py'),
          'pooled_hash':sha256(ROOT/'src/models/pooled_context.py'),
          'protocol_hash':sha256(ROOT/'docs/53_P46真实状态核验与P47协议.md'),'official_test_accessed':False,
          'max_wall_seconds':7200,'paper_result':False,'absolute_sample_start':10000,
          'backbone_optimizer_retained':True,'transfer_hash':sha256(ROOT/'src/optimizer_transfer.py')}
        assert plan['checkpoint_hash'].lower()=='47052b4b47d437d736ee1cb637ba744a22d86744d93abfd226382e7e8ef2d3c2'
        save_json(out/'plan.json',plan);results=[];matched_initial=None
        for role,mode in plan['roles'].items():
            if time.monotonic()-started>=7200:raise RuntimeError('Total wall budget reached')
            directory=out/role;directory.mkdir()
            set_determinism(3407);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            model=network().train()
            ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(ckpt['model'],strict=True)
            torch.manual_seed(9127)
            module=None if role=='A' else (PooledContext() if role=='P' else ReciprocalNonlocal(mode=mode)).cuda()
            if role=='C':
                matched_initial={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
            if role=='P':
                module.load_state_dict(matched_initial,strict=True)
                assert all(torch.equal(v.detach().cpu(),matched_initial[k]) for k,v in module.state_dict().items())
            def check_budget(_m,_i):
                if time.monotonic()-started>=7200:raise RuntimeError('Wall budget reached during forward')
            budget_hook=model.register_forward_pre_hook(check_budget)
            parameters=list(model.parameters())+([] if module is None else list(module.parameters()))
            opt=transfer_adamw(model,module,ckpt['optimizer'],[n for n,p in model.named_parameters()])
            assert all(int(opt.state[p]['step'].item())==10000 for p in model.parameters())
            assert module is None or all(p not in opt.state for p in module.parameters())
            del ckpt
            loader=DataLoader(SeededPairs(train),batch_sampler=StepBatches(1718,4,10000,11000),num_workers=4,
               pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(7001))
            val_loader=DataLoader(dev,batch_size=1,num_workers=2,pin_memory=True,persistent_workers=True,
               generator=torch.Generator().manual_seed(7002))
            iterator=iter(loader);first=next(iterator);x=first['blur'].cuda()
            with torch.no_grad(),autocast('bf16'):base=model(x)
            hook=None if module is None else model.latent.register_forward_hook(lambda _m,_i,z:module(z))
            with torch.no_grad(),autocast('bf16'):initial=model(x)
            assert torch.equal(base,initial);del x,base,initial
            save_json(out/'progress.json',{'role':role,'phase':'initial_validation'})
            before=validate(model,val_loader,directory,0,'bf16');rows=[]
            for step in range(1,1001):
                if time.monotonic()-started>=7200:
                    torch.save({'model':model.state_dict(),'module':None if module is None else module.state_dict(),
                      'optimizer':opt.state_dict(),'completed_steps':step-1,'role':role},directory/'budget_stop.pt')
                    raise RuntimeError('Total wall budget reached; partial snapshot saved')
                t=time.monotonic();item=first if step==1 else next(iterator)
                x,y=item['blur'].cuda(non_blocking=True),item['sharp'].cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with autocast('bf16'):pred=model(x);loss=torch.nn.functional.l1_loss(pred.float(),y)
                assert bool(torch.isfinite(loss));loss.backward()
                if module is not None:
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in module.parameters())
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                opt.step();torch.cuda.synchronize()
                rows.append({'step':step,'loss':loss.item(),'grad_norm':norm.item(),'seconds':time.monotonic()-t})
                if step==1 or step%100==0:
                    write_csv(directory/'training.csv',rows)
                    state={'role':role,'phase':'training','step':step,'total':1000,'wall_seconds':time.monotonic()-started}
                    save_json(out/'progress.json',state);print(json.dumps(state),flush=True)
            assert all(int(opt.state[p]['step'].item())==11000 for p in model.parameters())
            assert module is None or all(int(opt.state[p]['step'].item())==1000 for p in module.parameters())
            torch.save({'model':model.state_dict(),'module':None if module is None else module.state_dict(),
               'optimizer':opt.state_dict(),'completed_steps':1000,'role':role,'plan':plan},directory/'last.pt')
            save_json(out/'progress.json',{'role':role,'phase':'final_validation','step':1000})
            after=validate(model,val_loader,directory,1000,'bf16')
            result={'role':role,'mode':mode,'before':before,'after':after,'initial_equivalent':True,
               'module_parameters':0 if module is None else sum(p.numel() for p in module.parameters())}
            save_json(directory/'summary.json',result);results.append(result);save_json(out/'partial_results.json',results)
            budget_hook.remove()
            if hook is not None:hook.remove()
            del model,module,opt,parameters,loader,val_loader,iterator,first,item,x,y,pred,loss
            gc.collect();torch.cuda.empty_cache()
        save_json(out/'summary.json',{'complete':True,'results':results,'wall_seconds':time.monotonic()-started,
                                     'official_test_accessed':False,'paper_result':False})
        save_json(out/'progress.json',{'complete':True})
    except Exception as error:
        save_json(out/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
