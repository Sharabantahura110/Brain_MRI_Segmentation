"""Compare separately evaluated model metrics without inventing results."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unet', required=True, help='Path to U-Net evaluation metrics.json')
    p.add_argument('--attention', required=True, help='Path to Attention U-Net evaluation metrics.json')
    p.add_argument('--output', default='outputs/comparison.csv')
    args = p.parse_args()
    paths = {'U-Net': args.unet, 'Attention U-Net': args.attention}
    fields = ['global_dice', 'global_iou', 'mean_slice_dice_nonempty_gt',
              'pixel_precision', 'pixel_recall', 'n_slices']
    rows = []
    for label, filename in paths.items():
        metrics = json.loads(Path(filename).read_text(encoding='utf-8'))
        if metrics.get('split') != 'test':
            raise ValueError(f"Expected test-set metrics for {label}; found split={metrics.get('split')}")
        rows.append({'model': label, **{key: metrics.get(key) for key in fields}})
    if rows[0]['n_slices'] != rows[1]['n_slices']:
        raise ValueError('Test sample counts differ. Evaluate both models with the SAME manifest.')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['model', *fields])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['model']}: global Dice={row['global_dice']}, IoU={row['global_iou']}, precision={row['pixel_precision']}, recall={row['pixel_recall']}")
    print('Saved:', output)


if __name__ == '__main__':
    main()
