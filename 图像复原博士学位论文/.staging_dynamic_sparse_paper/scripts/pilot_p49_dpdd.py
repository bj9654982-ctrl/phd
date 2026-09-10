"""Bounded DPDD plumbing test, NOT a validation or paper experiment."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dpdd import DPDDPairs, paired_crop
from scripts.train_p25_restormer import network, update
from scripts.train_p12_matched_nonlocal import set_determinism


def main():
    out = ROOT/'results/runs/P49_dpdd_smoke_v0'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA with BF16 required')
        data = Path('H:/phddata/dd_dp_dataset_png')
        lists = Path('H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names')
        train, val = [DPDDPairs(data, lists, split, 128) for split in ('train', 'val')]
        train_names = {p.stem for pair in train.pairs for p in pair}
        val_names = {p.stem for pair in val.pairs for p in pair}
        assert not train_names & val_names
        # Coordinate/precision unit checks: a single transform for both images.
        grid = np.arange(16*16*3, dtype=np.uint16).reshape(16, 16, 3)
        for seed in range(10):
            x, y = paired_crop(grid, grid.copy(), 8, seed)
            x2, _ = paired_crop(grid, grid, 8, seed)
            assert torch.equal(x, y) and torch.equal(x, x2)
        x, _ = paired_crop(np.full((8, 8, 3), 65535, np.uint16), np.zeros((8, 8, 3), np.uint16), 8)
        assert x.max().item() == 1.0
        plan = {'purpose': '30 fixed-batch training updates; plumbing only',
            'data': str(data), 'train_pairs': len(train), 'val_pairs': len(val),
            'split_capture_id_overlap': 0, 'test_accessed': False,
            'initialization': 'scratch, seed 3407; no GoPro weights',
            'steps': 30, 'batch': 4, 'patch': 128, 'precision': 'bf16',
            'optimizer': 'AdamW lr=0.0003 weight_decay=0.0001 clip=1',
            'list_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in lists.glob('*.npy') if not p.name.startswith('test')},
            'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), ROOT/'src/dpdd.py')}}
        (out/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
        samples = [train[(i, 49000+i)] for i in (0, 116, 232, 349)]
        validation_sample = val[0]
        assert validation_sample['blur'].shape == (3, 128, 128)
        x = torch.stack([s['blur'] for s in samples]).cuda()
        y = torch.stack([s['sharp'] for s in samples]).cuda()
        set_determinism(3407)
        model = network().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for step in range(1, 31):
            tick = time.monotonic()
            loss, grad = update(model, optimizer, x, y, 'bf16')
            torch.cuda.synchronize()
            rows.append({'step': step, 'l1': loss, 'gradient_norm': grad, 'seconds': time.monotonic()-tick})
            if step == 1 or step % 5 == 0:
                print(json.dumps(rows[-1]), flush=True)
            if time.monotonic()-started > 300:
                raise RuntimeError('Five-minute smoke budget exceeded')
        result = {**plan, 'complete': True, 'unit_checks_passed': True,
            'decoded_pairs': {'train': 4, 'val': 1},
            'samples': [{'source': s['source'], 'target': s['target']} for s in samples],
            'gpu': torch.cuda.get_device_name(), 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
            'rows': rows, 'seconds': time.monotonic()-started,
            'weights_discarded': True, 'paper_result': False,
            'warning': 'Fixed training patches only; no generalization or convergence claim'}
        (out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('COMPLETE '+str(out), flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'error': repr(exc)}), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
