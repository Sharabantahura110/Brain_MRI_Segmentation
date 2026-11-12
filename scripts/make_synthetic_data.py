"""Generate fake TIFF image/mask pairs for pipeline testing ONLY (not scientific data)."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate(root, patients=12, slices=3, size=96, seed=42):
    root = Path(root)
    rng = np.random.default_rng(seed)
    for pid in range(patients):
        folder = root / f'SYNTHETIC_PATIENT_{pid:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        for sl in range(slices):
            cx, cy = size // 2, size // 2
            yy, xx = np.mgrid[:size, :size]
            radial = ((xx - cx) / (size * 0.36)) ** 2 + ((yy - cy) / (size * 0.42)) ** 2
            brain = np.clip((1 - radial) * 170, 0, 170)
            lesion = (pid + sl) % 3 != 0
            mask = np.zeros((size, size), dtype=np.uint8)
            if lesion:
                x0 = int(size * (0.35 + 0.02 * (pid % 4)))
                y0 = int(size * (0.30 + 0.02 * sl))
                rr = int(size * 0.12)
                mask[((xx - x0) ** 2 + (yy - y0) ** 2) <= rr ** 2] = 255
            noise = rng.normal(0, 10, (size, size))
            gray = np.clip(brain + noise + (mask > 0) * 85, 0, 255).astype(np.uint8)
            image = np.stack((gray, np.clip(gray.astype(int) + 8, 0, 255).astype(np.uint8), gray), axis=-1)
            filename = f'SYNTHETIC_PATIENT_{pid:03d}_{sl:03d}'
            Image.fromarray(image).save(folder / f'{filename}.tif')
            Image.fromarray(mask).save(folder / f'{filename}_mask.tif')
    print(f'Created synthetic TEST-ONLY data in {root}; not for CV results or medical use.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data/raw/synthetic')
    p.add_argument('--patients', type=int, default=12)
    p.add_argument('--slices', type=int, default=3)
    p.add_argument('--size', type=int, default=96)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.patients < 3 or args.slices < 1 or args.size < 32:
        p.error('Need >=3 patients, >=1 slice, >=32 pixels')
    generate(args.output, args.patients, args.slices, args.size, args.seed)
