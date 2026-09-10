"""P26: bounded continuation of trusted P25 checkpoints; unchanged optimization."""
import argparse
import gc
import importlib.util
import json
import random
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import GoProPairDataset
from scripts.pilot_p24_restormer import GROUPS
from scripts.train_p12_matched_nonlocal import set_determinism, _psnr_value, _ssim_value
from scripts.analyze_p16_factorial import sha256, write_csv
from scripts.evaluate_p21_full_image import save_json


def network():
    vendor = ROOT / 'third_party/restormer_official_68dc6ac_v1'
    source = vendor / 'basicsr/models/archs/restormer_arch.py'
    config = vendor / 'Motion_Deblurring/Options/Deblurring_Restormer.yml'
    provenance = json.loads((vendor/'download_provenance.json').read_text(encoding='utf-8'))
    for path in (source, config):
        expected = next(r['sha256'] for r in provenance['files'] if r['path']==path.relative_to(vendor).as_posix())
        if sha256(path) != expected:
            raise RuntimeError('Official source changed')
    spec = importlib.util.spec_from_file_location('p25_official', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = yaml.safe_load(config.read_text(encoding='utf-8'))['network_g']
    settings.pop('type')
    return module.Restormer(**settings).cuda()


def autocast(precision):
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision=='bf16')


def update(model, optimizer, x, y, precision):
    optimizer.zero_grad(set_to_none=True)
    with autocast(precision):
        pred = model(x)
        loss = torch.nn.functional.l1_loss(pred.float(), y)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return loss.item(), norm.item()


class SeededPairs(Dataset):
    """Per-sample augmentation seed makes worker prefetch independent of resume."""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, key):
        index, seed = key
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return self.base[index]


class StepBatches(Sampler):
    def __init__(self, count, batch, start=0, stop=3000):
        self.count, self.batch, self.start, self.stop = count, batch, start, stop

    def __len__(self):
        return self.stop-self.start

    def __iter__(self):
        epoch, order = -1, None
        for step in range(self.start, self.stop):
            keys = []
            for slot in range(self.batch):
                position = step*self.batch+slot
                current, offset = divmod(position, self.count)
                if current != epoch:
                    epoch = current
                    order = torch.randperm(self.count, generator=torch.Generator().manual_seed(3408+epoch)).tolist()
                keys.append((order[offset], 100000+position))
            yield keys


def choose(rows):
    safe = [r for r in rows if r.get('passed') and r['allocated_mib']<=8500 and r['reserved_mib']<=9500]
    if not safe:
        raise RuntimeError('No safe throughput candidate')
    best = max(r['images_per_second'] for r in safe)
    # Prefer less memory when throughput differs by under 5%.
    return min((r for r in safe if r['images_per_second']>=.95*best), key=lambda r:r['allocated_mib'])


def benchmark(args, out):
    ds = GoProPairDataset(args.project/'data/GOPRO_Large', 'train',128,False,exclude_recording_groups=GROUPS)
    samples = [ds[i] for i in range(6)]
    x0 = torch.stack([r['blur'] for r in samples])
    y0 = torch.stack([r['sharp'] for r in samples])
    candidates = [('fp32',1),('fp32',2),('bf16',2),('bf16',4),('bf16',6)]
    save_json(out/'plan.json', {'candidates':candidates,'warmup':3,'timed_updates':12,
                              'quality_selection':False,'official_test_accessed':False})
    rows=[]
    for precision,batch in candidates:
        if precision=='bf16' and not torch.cuda.is_bf16_supported():
            rows.append({'precision':precision,'batch':batch,'passed':False,'reason':'bf16 unsupported'})
            continue
        previous=[r for r in rows if r.get('passed') and r['precision']==precision]
        if previous and previous[-1]['allocated_mib']*batch/previous[-1]['batch']>8500:
            rows.append({'precision':precision,'batch':batch,'passed':False,'reason':'conservative memory projection'})
            continue
        model=optimizer=x=y=None
        try:
            set_determinism(3407)
            model=network().train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
            x,y=x0[:batch].cuda(),y0[:batch].cuda()
            torch.cuda.reset_peak_memory_stats()
            times=[]
            for i in range(15):
                torch.cuda.synchronize()
                start=time.perf_counter()
                update(model,optimizer,x,y,precision)
                torch.cuda.synchronize()
                if i>=3:
                    times.append(time.perf_counter()-start)
            mean=statistics.fmean(times)
            row={'precision':precision,'batch':batch,'passed':True,'seconds_per_step':mean,
                 'images_per_second':batch/mean,'allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                 'reserved_mib':torch.cuda.max_memory_reserved()/2**20}
            rows.append(row)
            print(json.dumps(row),flush=True)
        except torch.cuda.OutOfMemoryError:
            rows.append({'precision':precision,'batch':batch,'passed':False,'reason':'CUDA OOM'})
        finally:
            del model,optimizer,x,y
            gc.collect()
            torch.cuda.empty_cache()
        save_json(out/'candidates.json',rows)
    selected=choose(rows)
    save_json(out/'summary.json', {'complete':True,'candidates':rows,'selected':selected,
              'gpu_only_cached_input':True,'paper_result':False,'benchmark_weights_discarded':True})
    print('SELECTED '+json.dumps(selected),flush=True)


@torch.inference_mode()
def validate(model, loader, out, step, precision):
    model.eval()
    rows=[]
    for batch in loader:
        x,y=batch['blur'].cuda(non_blocking=True),batch['sharp'].cuda(non_blocking=True)
        with autocast(precision):
            pred=model(x)
        if not bool(torch.isfinite(pred).all()):
            raise RuntimeError('Nonfinite validation')
        rows.append({'name':batch['name'][0], 'psnr_db':_psnr_value(pred,y),
                     'input_psnr_db':_psnr_value(x,y),'ssim':_ssim_value(pred[0],y[0])})
    write_csv(out/f'validation_{step:04d}.csv',rows)
    summary={'step':step,'count':len(rows),'psnr_db':statistics.fmean(r['psnr_db'] for r in rows),
             'ssim':statistics.fmean(r['ssim'] for r in rows),
             'input_psnr_db':statistics.fmean(r['input_psnr_db'] for r in rows)}
    summary['gain_db']=summary['psnr_db']-summary['input_psnr_db']
    print('VALIDATION '+json.dumps(summary),flush=True)
    model.train()
    return summary


def train(args,out):
    choice=json.loads(args.selection.read_text(encoding='utf-8'))['selected']
    batch,precision=choice['batch'],choice['precision']
    ds=GoProPairDataset(args.project/'data/GOPRO_Large','train',128,True,exclude_recording_groups=GROUPS)
    dev=GoProPairDataset(args.project/'data/GOPRO_Large','train',256,False,include_recording_groups=GROUPS)
    if (len(ds),len(dev))!=(1718,385) or set(ds.recording_groups)&set(dev.recording_groups):
        raise RuntimeError('Frozen split mismatch')
    set_determinism(3407)
    model=network().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    start_step=0
    if args.resume:
        ckpt=torch.load(args.resume,map_location='cpu',weights_only=False) # own trusted checkpoint only
        if ckpt['batch']!=batch or ckpt['precision']!=precision or ckpt['script_sha256'] not in (sha256(Path(__file__)),sha256(ROOT/'scripts/train_p25_restormer.py')):
            raise RuntimeError('Resume configuration mismatch')
        model.load_state_dict(ckpt['model'],strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step=ckpt['step']
        random.setstate(ckpt['python_rng']); np.random.set_state(ckpt['numpy_rng'])
        torch.set_rng_state(ckpt['torch_rng']); torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    if not args.resume or not 3000 <= start_step < args.stop_step <= 10000:
        raise RuntimeError('P26 requires trusted resume and bounded endpoint')
    del ckpt
    plan={'steps':args.stop_step,'start_step':start_step,'batch':batch,'precision':precision,'patch':128,
          'seed':3407,'lr':3e-4,'loss':'L1','gradient_clip':1.,'workers':4,
          'initialization':'resume_own' if args.resume else 'random','public_weights_loaded':False,
          'train_pairs':1718,'dev_pairs':385,'groups':GROUPS,'official_test_accessed':False,
          'validation_steps':[start_step]+[s for s in range(start_step+1,args.stop_step+1) if s%1000==0 or s==args.stop_step],'validation_crop':256,'paper_result':False,
          'script_sha256':sha256(Path(__file__)),'selection_sha256':sha256(args.selection),
          'protocol_sha256':sha256(ROOT/'docs/29_P26续训协议.md'), 'resume_sha256':sha256(args.resume)}
    save_json(out/'plan_before_training.json',plan)
    loader=DataLoader(SeededPairs(ds),batch_sampler=StepBatches(len(ds),batch,start_step,args.stop_step),num_workers=4,
                      pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(7001))
    val_loader=DataLoader(dev,batch_size=1,num_workers=2,pin_memory=True,persistent_workers=True,
                          generator=torch.Generator().manual_seed(7002))
    validations=[]
    save_json(out/'progress.json',{'phase':'initial_validation','step':start_step})
    validations.append(validate(model,val_loader,out,start_step,precision))
    if start_step==3000:
        reference=json.loads((args.project/'results/runs/P25_restormer_scratch3000_v0/summary.json').read_text(encoding='utf-8'))['validations'][-1]
        if any(abs(validations[0][k]-reference[k])>1e-5 for k in ('psnr_db','ssim')):
            raise RuntimeError('Restored validation differs from P25')
    save_json(out/'restore_check.json',{'passed':True,'step':start_step,'validation':validations[0]})
    iterator=iter(loader)
    rows=[]
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step+1,args.stop_step+1):
        started=time.perf_counter()
        item=next(iterator)
        x,y=item['blur'].cuda(non_blocking=True),item['sharp'].cuda(non_blocking=True)
        loss,norm=update(model,optimizer,x,y,precision)
        torch.cuda.synchronize()
        rows.append({'step':step,'loss':loss,'grad_norm':norm,'seconds':time.perf_counter()-started})
        if step==start_step+1 or step%50==0:
            write_csv(out/'training.csv',rows)
            recent=statistics.fmean(r['seconds'] for r in rows[-50:])
            status={'phase':'training','step':step,'total':args.stop_step,'loss':loss,'seconds_per_step':recent,
                    'images_per_second':batch/recent,'remaining_training_minutes':(args.stop_step-step)*recent/60,
                    'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
            save_json(out/'progress.json',status)
            print(json.dumps(status),flush=True)
        if step%1000==0 or step==args.stop_step:
            # Immutable milestones: original model, optimizer and RNG states preserved.
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                        'batch':batch,'precision':precision,'script_sha256':sha256(Path(__file__)),
                        'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/f'step_{step:04d}.pt')
            save_json(out/'progress.json',{'phase':'validation','step':step})
            validations.append(validate(model,val_loader,out,step,precision))
            save_json(out/'validation_summary.json',validations)
    save_json(out/'summary.json',{'complete':True,'completed_steps':args.stop_step,'validations':validations,
                                'paper_result':False,'official_test_accessed':False,
                                'sample_exposures':args.stop_step*batch,'new_sample_exposures':(args.stop_step-start_step)*batch,'precision':precision,'batch':batch})
    save_json(out/'progress.json',{'complete':True,'step':args.stop_step})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['benchmark','train'],required=True)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stop-step',type=int,default=10000)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    try:
        (benchmark if args.mode=='benchmark' else train)(args,args.output)
    except Exception as error:
        save_json(args.output/'failure.json',{'error':str(error),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    main()


