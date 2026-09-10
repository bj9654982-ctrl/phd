"""Bounded 100-update random-init Restormer pilot, not a benchmark or convergence run."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import GoProPairDataset
from scripts.train_p12_matched_nonlocal import set_determinism, _psnr_value
from scripts.analyze_p16_factorial import sha256, write_csv
from scripts.evaluate_p21_full_image import save_json

GROUPS = ('GOPR0372_07', 'GOPR0378_13', 'GOPR0385_11')


def timing_summary(seconds):
    if not seconds or any(t <= 0 for t in seconds):
        raise ValueError('Positive timings required')
    mean = statistics.fmean(seconds)
    return {'mean_step_seconds': mean, 'median_step_seconds': statistics.median(seconds),
            'same_settings_30k_hours': mean*30000/3600,
            'same_settings_300k_hours': mean*300000/3600,
            'excludes_validation_checkpoint_and_larger_patches': True}


@torch.inference_mode()
def evaluate(model, dataset, indices):
    model.eval()
    rows = []
    for idx in indices:
        item = dataset[idx]
        x, y = item['blur'][None].cuda(), item['sharp'][None].cuda()
        pred = model(x)
        if not bool(torch.isfinite(pred).all()):
            raise RuntimeError('Nonfinite probe output')
        rows.append({'name': item['name'], 'l1': torch.nn.functional.l1_loss(pred, y).item(),
                     'psnr_db': _psnr_value(pred, y), 'input_psnr_db': _psnr_value(x, y)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    out.mkdir(parents=True)
    try:
        project = args.project_root.resolve()
        data = project / 'data/GOPRO_Large'
        train = GoProPairDataset(data, 'train', 128, True, exclude_recording_groups=GROUPS)
        fixed_train = GoProPairDataset(data, 'train', 128, False, exclude_recording_groups=GROUPS)
        dev = GoProPairDataset(data, 'train', 256, False, include_recording_groups=GROUPS)
        if len(train) != 1718 or len(dev) != 385 or set(train.recording_groups) & set(dev.recording_groups):
            raise RuntimeError('Frozen split mismatch')
        p3 = json.loads((project / 'results/runs/P3_baseline_group_split_v0/summary.json').read_text(encoding='utf-8'))
        if sorted(p3['validation_recording_groups']) != sorted(GROUPS):
            raise RuntimeError('P3 group mismatch')
        cases = json.loads((project / 'results/runs/P21_visual_review_v0/summary.json').read_text(encoding='utf-8'))['cases'][:3]
        name_index = {f'{p.scene}/{p.name}': i for i,p in enumerate(dev.pairs)}
        dev_indices = [name_index[c['name']] for c in cases]
        train_indices = [0, len(train)//2, len(train)-1]
        vendor = ROOT / 'third_party/restormer_official_68dc6ac_v1'
        architecture = vendor / 'basicsr/models/archs/restormer_arch.py'
        config = vendor / 'Motion_Deblurring/Options/Deblurring_Restormer.yml'
        provenance = json.loads((vendor / 'download_provenance.json').read_text(encoding='utf-8'))
        for path in (architecture, config):
            expected = next(f['sha256'] for f in provenance['files'] if f['path'] == path.relative_to(vendor).as_posix())
            if sha256(path) != expected:
                raise RuntimeError('Official source mismatch')
        settings = yaml.safe_load(config.read_text(encoding='utf-8'))['network_g']
        if settings.pop('type') != 'Restormer':
            raise RuntimeError('Unexpected network')
        spec = importlib.util.spec_from_file_location('p24_official_network', architecture)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        set_determinism(3407)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = module.Restormer(**settings).cuda()
        initial_output_weight = model.output.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, betas=(.9,.999))
        plan = {'seed':3407, 'steps':100, 'batch_size':1, 'patch_size':128,
                'AMP':False, 'TF32':False, 'initialization':'random', 'pretrained_loaded':False,
                'network_settings':settings, 'parameters':sum(p.numel() for p in model.parameters()),
                'train_pairs':len(train), 'dev_pairs':len(dev), 'validation_groups':GROUPS,
                'train_probe_names':[f'{train.pairs[i].scene}/{train.pairs[i].name}' for i in train_indices],
                'dev_probe_names':[c['name'] for c in cases], 'official_test_accessed':False,
                'script_sha256':sha256(Path(__file__)), 'architecture_sha256':sha256(architecture),
                'protocol_sha256':sha256(ROOT / 'docs/24_P24从零训练资源试跑协议.md'),
                'torch':str(torch.__version__), 'gpu':torch.cuda.get_device_name(0)}
        save_json(out / 'plan_before_training.json', plan)
        write_csv(out / 'split_manifest.csv', [
            {'role':role, 'name':f'{p.scene}/{p.name}', 'blur':str(p.blurry), 'sharp':str(p.sharp)}
            for role, ds in [('train',train),('development',dev)] for p in ds.pairs])
        before = {'train':evaluate(model, fixed_train, train_indices), 'dev':evaluate(model, dev, dev_indices)}
        save_json(out / 'before.json', before)
        loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=2,
                            pin_memory=True, generator=torch.Generator().manual_seed(3408))
        iterator = iter(loader)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        total_start = time.perf_counter()
        for step in range(1,101):
            started = time.perf_counter()
            batch = next(iterator)
            x, y = batch['blur'].cuda(non_blocking=True), batch['sharp'].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = torch.nn.functional.l1_loss(pred,y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            rows.append({'step':step, 'name':batch['name'][0], 'loss':loss.item(),
                         'grad_norm_before_clip':norm.item(), 'seconds':elapsed})
            if step == 1 or step % 10 == 0:
                write_csv(out / 'training.csv', rows)
                save_json(out / 'progress.json', rows[-1])
                print(f"step={step}/100 loss={loss.item():.6f} seconds={elapsed:.3f}", flush=True)
        train_wall = time.perf_counter()-total_start
        memory = {'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                  'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
        changed = not torch.equal(initial_output_weight, model.output.weight)
        if not changed:
            raise RuntimeError('Output weights unchanged')
        del pred, loss, x, y, optimizer, iterator, loader
        after = {'train':evaluate(model, fixed_train, train_indices), 'dev':evaluate(model, dev, dev_indices)}
        save_json(out / 'after.json', after)
        torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'plan':plan, 'completed_steps':100, 'resumable':False}, out / 'pilot_model.pt')
        summary = {'complete':True, 'resource_passed':True, 'weights_changed':changed,
                   'fixed_train_l1_decreased':statistics.fmean(r['l1'] for r in after['train']) < statistics.fmean(r['l1'] for r in before['train']),
                   'train_wall_seconds':train_wall, **memory,
                   'steady_step_timing':timing_summary([r['seconds'] for r in rows[10:]]),
                   'before':before, 'after':after, 'paper_result':False, 'convergence_claim':False,
                   'official_test_accessed':False, 'checkpoint_sha256':sha256(out / 'pilot_model.pt')}
        save_json(out / 'summary.json', summary)
        save_json(out / 'progress.json', {'complete':True,'completed_steps':100})
        print(json.dumps({k:v for k,v in summary.items() if k not in ('before','after')},indent=2),flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type':type(error).__name__, 'message':str(error),
                                       'traceback':traceback.format_exc(), 'automatic_fallback':False})
        raise


if __name__ == '__main__':
    main()
