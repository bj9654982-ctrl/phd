"""Preregistered frozen-backbone mechanism screen; never reads official test."""
import gc, json, time, traceback, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_p25_restormer import network,autocast,SeededPairs,StepBatches,validate
from scripts.train_p12_matched_nonlocal import set_determinism
from scripts.pilot_p24_restormer import GROUPS
from scripts.analyze_p16_factorial import sha256,write_csv
from scripts.evaluate_p21_full_image import save_json
from src.models.reciprocal_nonlocal import ReciprocalNonlocal
from src.models.normalized_reciprocal_nonlocal import NormalizedReciprocalNonlocal
from src.data import GoProPairDataset

def main():
    project=Path('J:/phd/动态稀疏非局部去模糊论文')
    out=ROOT/'results/runs/P37_normalization_v0';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def budget():
        if time.monotonic()-started>=7200:raise RuntimeError('7200 second wall budget reached')
    try:
        checkpoint=project/'results/runs/P26_restormer_continue10000_v0/step_10000.pt'
        digest=sha256(checkpoint)
        assert digest.lower()=='47052b4b47d437d736ee1cb637ba744a22d86744d93abfd226382e7e8ef2d3c2'
        plan={'roles':{'D':'reciprocal','N':'reciprocal'},'steps_each':1000,'batch':4,'precision':'bf16',
              'checkpoint_hash':digest,'trainable_backbone':False,'max_wall_seconds':7200,
              'script_hash':sha256(Path(__file__)),'protocol_hash':sha256(ROOT/'docs/42_P36归一化验证与P37协议.md'),
              'normalized_module_hash':sha256(ROOT/'src/models/normalized_reciprocal_nonlocal.py'),'module_hash':sha256(ROOT/'src/models/reciprocal_nonlocal.py'),'official_test_accessed':False}
        save_json(out/'plan.json',plan)
        train=GoProPairDataset(project/'data/GOPRO_Large','train',128,True,exclude_recording_groups=GROUPS)
        fixed_ds=GoProPairDataset(project/'data/GOPRO_Large','train',128,False,exclude_recording_groups=GROUPS)
        dev=GoProPairDataset(project/'data/GOPRO_Large','train',256,False,include_recording_groups=GROUPS)
        assert len(train)==1718 and len(dev)==385 and not set(train.recording_groups)&set(dev.recording_groups)
        fixed=torch.stack([fixed_ds[i]['blur'] for i in range(4)]).cuda()
        save_json(out/'fixed_names.json',[fixed_ds[i]['name'] for i in range(4)])
        results=[];initial_state=None
        for role,mode in plan['roles'].items():
            budget();directory=out/role;directory.mkdir()
            set_determinism(3407);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            model=network().eval()
            ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(ckpt['model'],strict=True)
            reference=ckpt['model'];del ckpt
            model.requires_grad_(False)
            torch.manual_seed(9127)
            cls=NormalizedReciprocalNonlocal if role=='N' else ReciprocalNonlocal
            module=cls(mode=mode).cuda()
            if initial_state is None:initial_state={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
            assert all(torch.equal(v.detach().cpu(),initial_state[k]) for k,v in module.state_dict().items())
            opt=torch.optim.AdamW(module.parameters(),lr=3e-4,weight_decay=1e-4)
            loader=DataLoader(SeededPairs(train),batch_sampler=StepBatches(1718,4,0,1000),num_workers=4,
                              pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(7001))
            val_loader=DataLoader(dev,batch_size=1,num_workers=2,pin_memory=True,persistent_workers=True,
                                  generator=torch.Generator().manual_seed(7002))
            with torch.no_grad(),autocast('bf16'):base=model(fixed)
            activity={}
            def inject(_m,_i,z):
                budget()
                value,stats=module(z,True);mass=stats['accepted_mass']
                with torch.no_grad():
                    activity.update(mass_mean=mass.mean().item(),zero_mass_fraction=(mass==0).float().mean().item(),
                        relative_correction_rms=((value.float()-z.float()).square().mean().sqrt()/z.float().square().mean().sqrt().clamp_min(1e-12)).item())
                return value
            hook=model.latent.register_forward_hook(inject)
            with torch.no_grad(),autocast('bf16'):initial=model(fixed)
            assert torch.equal(base,initial);fixed_before=dict(activity);del base,initial
            save_json(out/'progress.json',{'role':role,'phase':'initial_validation'})
            before=validate(model,val_loader,directory,0,'bf16');model.eval();rows=[]
            for step,item in enumerate(loader,1):
                budget();t=time.monotonic()
                x,y=item['blur'].cuda(non_blocking=True),item['sharp'].cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with autocast('bf16'):pred=model(x);loss=torch.nn.functional.l1_loss(pred.float(),y)
                assert bool(torch.isfinite(loss));loss.backward()
                assert all(p.grad is None for p in model.parameters())
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in module.parameters())
                norm=torch.nn.utils.clip_grad_norm_(module.parameters(),1.,error_if_nonfinite=True)
                opt.step();torch.cuda.synchronize()
                rows.append({'step':step,'loss':loss.item(),'grad_norm':norm.item(),'seconds':time.monotonic()-t,**activity})
                if step==1 or step%100==0:
                    write_csv(directory/'training.csv',rows)
                    state={'role':role,'phase':'training','step':step,'total':1000,'wall_seconds':time.monotonic()-started}
                    save_json(out/'progress.json',state);print(json.dumps(state),flush=True)
            assert step==1000
            assert all(torch.equal(v.detach().cpu(),reference[k]) for k,v in model.state_dict().items())
            assert any(not torch.equal(v.detach().cpu(),initial_state[k]) for k,v in module.state_dict().items())
            with torch.no_grad(),autocast('bf16'):model(fixed)
            fixed_after=dict(activity)
            torch.save({'module':module.state_dict(),'optimizer':opt.state_dict(),'completed_steps':1000,
                        'role':role,'plan':plan,'backbone_unchanged':True},directory/'last.pt')
            save_json(out/'progress.json',{'role':role,'phase':'final_validation','step':1000})
            after=validate(model,val_loader,directory,1000,'bf16');model.eval()
            assert all(torch.equal(v.detach().cpu(),reference[k]) for k,v in model.state_dict().items())
            result={'role':role,'before':before,'after':after,'backbone_unchanged':True,
                    'initial_equivalent':True,'module_initial_identical':True,'fixed_before':fixed_before,'fixed_after':fixed_after}
            save_json(directory/'summary.json',result);results.append(result);save_json(out/'partial_results.json',results)
            hook.remove();del model,module,opt,loader,val_loader,reference,item,x,y,pred,loss
            gc.collect();torch.cuda.empty_cache()
        budget()
        save_json(out/'summary.json',{'complete':True,'results':results,'wall_seconds':time.monotonic()-started,
                                      'official_test_accessed':False,'paper_result':False})
        save_json(out/'progress.json',{'complete':True})
    except Exception as error:
        save_json(out/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
