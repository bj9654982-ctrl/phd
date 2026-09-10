"""Read-only full decode and exact-pixel duplicate audit of DPDD train/val."""
import hashlib
import json
import sys
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dpdd import DPDDPairs, read_rgb16


def inspect(job):
    split, index, source, target = job
    records = []
    for role, path in [('source', source), ('target', target)]:
        image = read_rgb16(path)
        records.append({'split': split, 'pair': index, 'role': role,
            'path': str(path), 'shape': list(image.shape), 'dtype': str(image.dtype),
            'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
            'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if records[0]['shape'] != records[1]['shape']:
        raise ValueError(f'Pair dimensions differ: {source}')
    return records


def main():
    out = ROOT/'results/runs/P50_dpdd_audit_v0'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        cv2.setNumThreads(1)
        root = Path('H:/phddata/dd_dp_dataset_png')
        lists = Path('H:/phddata/defocus-deblurring-dual-pixel-master/DPDNet/file_names')
        jobs = []
        for split in ('train', 'val'):
            ds = DPDDPairs(root, lists, split)
            jobs.extend((split, i, s, t) for i, (s, t) in enumerate(ds.pairs))
        rows = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, records in enumerate(pool.map(inspect, jobs), 1):
                rows.extend(records)
                if i % 50 == 0:
                    print(f'decoded {i}/{len(jobs)} pairs', flush=True)
        groups = defaultdict(list)
        for r in rows:
            groups[(tuple(r['shape']), r['pixel_sha256'])].append(r)
        duplicates = [v for v in groups.values() if len(v) > 1]
        cross = [v for v in duplicates if len({r['split'] for r in v}) > 1]
        result = {'complete': True, 'decoded_images': len(rows),
            'counts': dict(Counter(r['split'] for r in rows)),
            'shapes': dict(Counter(str(r['shape']) for r in rows)),
            'exact_duplicate_groups': duplicates, 'cross_split_duplicate_groups': cross,
            'passed': not cross, 'test_accessed': False, 'seconds': time.monotonic()-started,
            'limitations': 'Exact duplicates only; no near-duplicate, registration or semantic correctness guarantee',
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'loader_sha256': hashlib.sha256((ROOT/'src/dpdd.py').read_bytes()).hexdigest(),
            'list_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in lists.glob('*.npy') if not p.name.startswith('test')}}
        (out/'images.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        (out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'error': repr(exc)}), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
