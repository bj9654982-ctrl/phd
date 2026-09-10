"""Fixed-budget single-RGB DPDD baseline; development crops only."""
import hashlib, json, time, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.dpdd import DPDDPairs
from scripts.train_p25_restormer import network, update, StepBatches
from scripts.train_p12_matched_nonlocal import set_determinism, _psnr_value, _ssim_value

def save(path, obj):
    path.write_text(json.dumps(obj,indent=2),encoding='utf-8')

@torch.no_grad()
def validate(model, samples):
    model.eval()
    rows=[]
    for s in samples:
        x,y=s['blur'].unsqueeze(0).cuda(),s['sharp'].unsqueeze(0).cuda()
        p=model(x).clamp(0,1)
        rows.append({'source':s['source'],'psnr':_psnr_value(p,y),
            'ssim':_ssim_value(p[0],y[0]),'input_psnr':_psnr_value(x,y),'input_ssim':_ssim_value(x[0],y[0])})
    model.train()
    return {'psnr':sum(r['psnr'] for r in rows)/len(rows),
        'ssim':sum(r['ssim'] for r in rows)/len(rows),
        'input_psnr':sum(r['input_psnr'] for r in rows)/len(rows),
        'input_ssim':sum(r['input_ssim'] for r in rows)/len(rows),'images':rows}

def main():
    out=ROOT/'results/runs/P51_dpdd_baseline_v1'
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); step=0; model=optimizer=None
    try:
        audit=json.loads((ROOT/'results/runs/P50_dpdd_audit_v0/summary.json').read_text())
        assert audit['complete'] and audit['passed']
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        data=Path('H:/phddata/dd_dp_dataset_png')
        lists=Path('H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names')
        for name,digest in audit['list_sha256'].items():
            assert hashlib.sha256((lists/name).read_bytes()).hexdigest()==digest
        train=DPDDPairs(data,lists,'train',128)
        dev=DPDDPairs(data,lists,'val',256)
        set_determinism(3407)
        plan={'steps':1000,'batch':4,'train_patch':128,'val_center_crop':256,'val_count':74,
            'seed':3407,'initialization':'scratch','precision_train':'bf16','precision_val':'fp32',
            'lr':3e-4,'weight_decay':1e-4,'clip':1,'workers':4,'test_accessed':False,
            'paper_result':False,'seconds_budget':3600,'list_hashes':audit['list_sha256'],
            'code_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__),ROOT/'src/dpdd.py',ROOT/'scripts/train_p25_restormer.py']}}
        save(out/'plan.json',plan)
        samples=[dev[i] for i in range(len(dev))]
        loader=DataLoader(train,batch_sampler=StepBatches(len(train),4,0,1000),
            num_workers=4,pin_memory=True,persistent_workers=True)
        model=network().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats()
        curve=[]
        def evaluate():
            result=validate(model,samples)
            save(out/f'val_{step:04d}.json',result)
            curve.append({'step':step,**{k:v for k,v in result.items() if k!='images'}})
            save(out/'curve.json',curve)
            print('VAL '+json.dumps(curve[-1]),flush=True)
        evaluate()
        with (out/'train.jsonl').open('w',encoding='utf-8') as log:
            for batch in loader:
                loss,grad=update(model,optimizer,batch['blur'].cuda(non_blocking=True),batch['sharp'].cuda(non_blocking=True),'bf16')
                step+=1
                row={'step':step,'loss':loss,'grad':grad,'seconds':time.monotonic()-started,
                    'source':batch['source'],'target':batch['target']}
                log.write(json.dumps(row)+'\n');log.flush()
                if step%50==0:
                    save(out/'progress.json',row);print('TRAIN '+json.dumps(row),flush=True)
                if step%250==0:
                    evaluate()
                if time.monotonic()-started>3600:
                    raise TimeoutError('Fixed one-hour budget reached')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'step':step,'plan':plan,'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all()},out/'final.pt')
        save(out/'summary.json',{'complete':True,'step':step,'curve':curve,
            'seconds':time.monotonic()-started,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'gain_psnr':curve[-1]['psnr']-curve[-1]['input_psnr'],
            'gain_ssim':curve[-1]['ssim']-curve[-1]['input_ssim'],
            'test_accessed':False,'paper_result':False})
        print('COMPLETE',flush=True)
    except Exception as exc:
        if model is not None and optimizer is not None:
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step},out/'interrupted.pt')
        save(out/'failure.json',{'error':repr(exc),'step':step})
        raise

if __name__=='__main__':
    main()
