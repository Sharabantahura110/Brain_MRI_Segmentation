"""Fast correctness tests. Synthetic data never substitutes for MRI experiments."""
import numpy as np
import torch
from PIL import Image

from mri_seg.data import find_pairs, make_splits, MRISliceDataset, prepare_mask
from mri_seg.metrics import MetricsAccumulator, bce_dice_loss
from mri_seg.models import build_model


def make_fixture(tmp_path, patient_count=5):
    for i in range(patient_count):
        folder = tmp_path / f'PATIENT_{i}'
        folder.mkdir()
        for j in range(2):
            picture = np.zeros((32, 32, 3), dtype=np.uint8)
            picture[8:20, 9:20] = 200
            mask = np.zeros((32, 32), dtype=np.uint8)
            if j == 0:
                mask[10:16, 12:18] = 255
            Image.fromarray(picture).save(folder / f'image_{j}.tif')
            Image.fromarray(mask).save(folder / f'image_{j}_mask.tif')
    return tmp_path


def test_patient_level_splits(tmp_path):
    recs = find_pairs(make_fixture(tmp_path))
    m = make_splits(recs, seed=42)
    assert len(recs) == 10
    a, b, c = [set(m['patients'][s]) for s in ('train', 'val', 'test')]
    assert not (a & b or b & c or c & a)
    assert len(a | b | c) == 5
    ds = MRISliceDataset(tmp_path, m['splits']['train'], 32, augment=False)
    sample = ds[0]
    assert sample['image'].shape == (3, 32, 32)
    assert sample['mask'].shape == (1, 32, 32)
    assert set(torch.unique(sample['mask']).tolist()).issubset({0., 1.})


def test_models_backward():
    for name in ('unet', 'attention_unet'):
        model = build_model(name, base_channels=4)
        model.train()
        image = torch.rand(2, 3, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        logits = model(image)
        assert logits.shape == (2, 1, 32, 32)
        loss = bce_dice_loss(logits, target)
        loss.backward()
        assert torch.isfinite(loss)


def test_metrics_empty_mask_and_perfect_prediction():
    m = MetricsAccumulator()
    m.update(torch.full((1, 1, 4, 4), -20.), torch.zeros(1, 1, 4, 4))
    result = m.compute()
    assert result['mean_slice_dice_all'] == 1.0
    assert result['mean_slice_dice_nonempty_gt'] is None
    m = MetricsAccumulator()
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0, 0] = 1
    logits = torch.where(target > 0, torch.tensor(20.), torch.tensor(-20.))
    m.update(logits, target)
    assert m.compute()['global_dice'] == 1


def test_no_zero_masks_due_to_threshold():
    raw = np.zeros((32, 32), dtype=np.uint8)
    raw[12, 12] = 1
    assert prepare_mask(Image.fromarray(raw), 32).sum().item() == 1
