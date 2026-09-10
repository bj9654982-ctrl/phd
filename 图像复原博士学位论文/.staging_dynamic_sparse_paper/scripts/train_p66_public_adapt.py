"""Single preregistered 500-update public baseline stability check."""
import hashlib,json,sys,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dpdd8 import DPDD8Pairs
from scripts.train_p25_restormer import network,update,StepBatches
from scripts.train_p51_dpdd import validate,save
from scripts.train_p12_matched_nonlocal import set_determinism

def main():
    out=ROOT/'results/runs/P66_public_adapt_v0';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();step=0;model=opt=None
    try:
        path=ROOT/'third_party/restormer_dpdd_public_v0/single_image_defocus_deblurring.pth'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest=='7dce451f33f8f5e0faf7c4e3996e5dcc1bd425ecd1ada99b0f9750e490fd4c9e'
        lists=Path('H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names')
        plan={'source_hash':digest,'steps':500,'batch':4,'patch':128,'lr':1e-5,'weight_decay':1e-4,
            'precision':'bf16','val_precision':'fp32','seed':3407,'budget_seconds':900,
            'optimizer':'fresh AdamW; no author moments','test_accessed':False,'input':'OpenCV uint8 RGB /255',
            'code_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/dpdd8.py',ROOT/'src/dpdd.py',ROOT/'scripts/train_p25_restormer.py',ROOT/'scripts/train_p51_dpdd.py']},
            'list_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in lists.glob('*.npy') if not p.name.startswith('test')}}
        save(out/'plan.json',plan)
        train=DPDD8Pairs('H:/phddata/dd_dp_dataset_png',lists,'train',128)
        dev=DPDD8Pairs('H:/phddata/dd_dp_dataset_png',lists,'val',256)
        samples=[dev[i] for i in range(len(dev))]
        set_determinism(3407);model=network().train()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['params'],strict=True)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4);assert not opt.state
        curve=[]
        def evaluate():
            v=validate(model,samples);save(out/f'val_{step}.json',v)
            curve.append({'step':step,**{k:x for k,x in v.items() if k!='images'}})
            save(out/'curve.json',curve);print('VAL '+json.dumps(curve[-1]),flush=True)
        evaluate()
        ref=json.loads((ROOT/'results/runs/P62_pretrained_capability_v0/summary.json').read_text())
        assert abs(curve[0]['psnr']-ref['psnr'])<1e-4 and abs(curve[0]['ssim']-ref['ssim'])<1e-5
        loader=DataLoader(train,batch_sampler=StepBatches(len(train),4,0,500),num_workers=4,pin_memory=True)
        with (out/'train.jsonl').open('w',encoding='utf-8') as log:
            for batch in loader:
                if time.monotonic()-started>900:raise TimeoutError('15-minute budget')
                loss,grad=update(model,opt,batch['blur'].cuda(non_blocking=True),batch['sharp'].cuda(non_blocking=True),'bf16');step+=1
                row={'step':step,'loss':loss,'grad':grad,'source':batch['source'],'target':batch['target'],'seconds':time.monotonic()-started}
                log.write(json.dumps(row)+'\n');log.flush()
                if step%50==0:save(out/'progress.json',row);print(json.dumps(row),flush=True)
                if step%250==0:evaluate()
        assert all(int(v['step'])==500 for v in opt.state.values())
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'plan':plan,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/'final.pt')
        delta=curve[-1]['psnr']-curve[0]['psnr'];ds=curve[-1]['ssim']-curve[0]['ssim']
        save(out/'summary.json',{'complete':True,'curve':curve,'delta_psnr':delta,'delta_ssim':ds,
            'passed':delta>=0 and ds>=0,'seconds':time.monotonic()-started,'test_accessed':False,'new_method':False})
        print('COMPLETE',flush=True)
    except Exception as e:
        save(out/'failure.json',{'error':repr(e),'step':step})
        if model is not None and opt is not None:torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step},out/'interrupted.pt')
        raise
if __name__=='__main__':main()
