"""Paired TIFF indexing, reproducible patient-wise splitting, and PyTorch datasets."""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def find_pairs(data_root: str | Path) -> list[dict]:
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")

    records = []
    for image in sorted(root.rglob('*')):
        if not image.is_file() or image.suffix.lower() not in ('.tif', '.tiff'):
            continue
        if image.stem.endswith('_mask'):
            continue
        mask = image.with_name(f'{image.stem}_mask{image.suffix}')
        if not mask.is_file():
            raise FileNotFoundError(f"Missing mask for {image}; expected {mask}")
        relative_parent = image.parent.relative_to(root)
        if relative_parent == Path('.'):
            # Fallback for a flattened copy of the dataset.
            patient = image.stem.rsplit('_', 1)[0]
        else:
            patient = str(relative_parent)
        records.append({
            'image': str(image.relative_to(root)),
            'mask': str(mask.relative_to(root)),
            'patient': patient,
        })
    if not records:
        raise ValueError(f"No image/mask TIFF pairs found under {root}")
    return records


def make_splits(records: list[dict], seed: int = 42,
                train_fraction: float = 0.7, val_fraction: float = 0.15) -> dict:
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Require train_fraction > 0, val_fraction > 0, and sum < 1")
    patients = sorted({r['patient'] for r in records})
    if len(patients) < 3:
        raise ValueError("At least 3 patients are required for train/val/test split")
    random.Random(seed).shuffle(patients)
    n_train = max(1, min(len(patients) - 2, round(len(patients) * train_fraction)))
    n_val = max(1, min(len(patients) - n_train - 1, round(len(patients) * val_fraction)))
    groups = {
        'train': set(patients[:n_train]),
        'val': set(patients[n_train:n_train + n_val]),
        'test': set(patients[n_train + n_val:]),
    }
    assert len(groups['train'] & groups['val']) == 0
    assert len(groups['train'] & groups['test']) == 0
    assert len(groups['val'] & groups['test']) == 0
    splits = {name: [r for r in records if r['patient'] in pts]
              for name, pts in groups.items()}
    for name, recs in splits.items():
        if not recs:
            raise ValueError(f"Empty split: {name}")
    return {'seed': seed, 'splits': splits, 'patients':
            {key: sorted(val) for key, val in groups.items()}}


def save_manifest(manifest: dict, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare_image(image: Image.Image, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    """Return CHW float tensor and uint8 RGB preview, using same steps for train/infer.

    LGG TIFFs contain three co-registered MRI sequences packed into RGB channels.
    This demo assumes 8-bit-style intensities and scales each image to [0,1].
    """
    resized = image.convert('RGB').resize((image_size, image_size), Image.Resampling.BILINEAR)
    preview = np.asarray(resized, dtype=np.uint8).copy()
    array = preview.astype(np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    return tensor, preview


def prepare_mask(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.convert('L').resize((image_size, image_size), Image.Resampling.NEAREST)
    binary = (np.asarray(resized) > 0).astype(np.float32)
    return torch.from_numpy(binary[None, ...].copy())


class MRISliceDataset(Dataset):
    def __init__(self, data_root: str | Path, records: list[dict],
                 image_size: int = 256, augment: bool = False):
        if image_size < 32:
            raise ValueError("image_size must be >= 32")
        self.root = Path(data_root)
        self.records = records
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        with Image.open(self.root / record['image']) as im:
            image, _ = prepare_image(im, self.image_size)
        with Image.open(self.root / record['mask']) as mask_im:
            mask = prepare_mask(mask_im, self.image_size)

        # Synchronized horizontal flip: image and mask always transform together.
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        return {'image': image, 'mask': mask, 'filename': record['image'],
                'patient': record['patient']}


def split_summary(manifest: dict) -> dict:
    return {name: {'patients': len(manifest['patients'][name]), 'slices': len(records)}
            for name, records in manifest['splits'].items()}
