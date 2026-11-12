"""Evaluate a trained checkpoint on a named manifest split and save figures/metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import load_manifest, MRISliceDataset
from .metrics import MetricsAccumulator
from .utils import resolve_device, load_model
from .visualize import save_sample_grid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', default='data/splits.json')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--output-dir', default='outputs/evaluation')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-examples', type=int, default=6)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = p.parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    dataset = MRISliceDataset(args.data_root, load_manifest(args.manifest)['splits'][args.split],
                              image_size=int(checkpoint['image_size']))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    metrics = MetricsAccumulator()
    positive_examples = []
    empty_examples = []
    # Prefer positive-label examples; retain a few empty slices for error analysis.
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch['image'].to(device))
            metrics.update(logits, batch['mask'].to(device))
            predictions = (logits.sigmoid() >= 0.5).cpu().numpy()
            for i in range(len(predictions)):
                if len(positive_examples) >= args.num_examples and len(empty_examples) >= args.num_examples:
                    continue
                original = (batch['image'][i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                mask = batch['mask'][i, 0].numpy()
                pred = predictions[i, 0]
                target_examples = positive_examples if bool(mask.any()) else empty_examples
                if len(target_examples) < args.num_examples:
                    target_examples.append((original, mask, pred))
    result = metrics.compute()
    result.update({'checkpoint': str(args.checkpoint), 'split': args.split,
                   'model': checkpoint['model_name'], 'image_size': checkpoint['image_size'],
                   'selected_epoch': checkpoint['epoch']})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    selected = positive_examples[:args.num_examples]
    selected += empty_examples[:max(0, args.num_examples - len(selected))]
    save_sample_grid([e[0] for e in selected], [e[1] for e in selected],
                     [e[2] for e in selected], output / 'predictions.png')
    print(json.dumps(result, indent=2))
    print('Saved evaluation metrics and preview figures in', output)


if __name__ == '__main__':
    main()
