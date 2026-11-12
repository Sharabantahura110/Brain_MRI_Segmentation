"""Build leakage-safe manifest: python -m mri_seg.prepare --data-root ..."""
import argparse
import json
from .data import find_pairs, make_splits, save_manifest, split_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, help='Unzipped LGG dataset root (scanned recursively)')
    p.add_argument('--manifest', default='data/splits.json')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    records = find_pairs(args.data_root)
    manifest = make_splits(records, seed=args.seed)
    save_manifest(manifest, args.manifest)
    print(f"Indexed {len(records)} image/mask pairs, {len(set(x['patient'] for x in records))} patients")
    print(json.dumps(split_summary(manifest), indent=2))
    print(f"Manifest saved at: {args.manifest}")


if __name__ == '__main__':
    main()
