"""Single-RGB DPDD: official ordered pairs, lossless uint16 decoding."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_rgb16(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint16 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Expected uint16 RGB PNG: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def paired_crop(x, y, size, seed=None):
    if x.shape != y.shape:
        raise ValueError('Pair shape mismatch')
    h, w = x.shape[:2]
    if size > min(h, w) or size <= 0:
        raise ValueError('Invalid crop size')
    rng = np.random.default_rng(seed)
    top, left = ((h-size)//2, (w-size)//2) if seed is None else (
        int(rng.integers(h-size+1)), int(rng.integers(w-size+1)))
    x, y = x[top:top+size, left:left+size], y[top:top+size, left:left+size]
    if seed is not None:
        if rng.integers(2):
            x, y = x[:, ::-1], y[:, ::-1]
        if rng.integers(2):
            x, y = x[::-1], y[::-1]
    return tuple(torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float()/65535.0 for a in (x, y))


class DPDDPairs(Dataset):
    def __init__(self, root, lists, split='train', patch=128):
        if split not in ('train', 'val'):
            raise ValueError('Test split sealed in development loader')
        self.split, self.patch = split, patch
        root, lists = Path(root), Path(lists)
        sources = np.load(lists/f'{split}_src.npy', allow_pickle=False).tolist()
        targets = np.load(lists/f'{split}_trg.npy', allow_pickle=False).tolist()
        if len(sources) != len(targets) or len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
            raise ValueError('Invalid official pair lists')
        self.pairs = [(root/f'{split}_c/source/{s}.png', root/f'{split}_c/target/{t}.png') for s, t in zip(sources, targets)]
        for names, folder in ((sources, 'source'), (targets, 'target')):
            actual = {p.stem for p in (root/f'{split}_c'/folder).glob('*.png')}
            if actual != set(names):
                raise ValueError(f'Official list mismatch: {split}/{folder}')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, key):
        index, seed = key if isinstance(key, tuple) else (key, None)
        source, target = self.pairs[index]
        x, y = paired_crop(read_rgb16(source), read_rgb16(target), self.patch, seed)
        return {'blur': x, 'sharp': y, 'source': source.name, 'target': target.name}
