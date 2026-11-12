"""Predict binary segmentation and overlay for a single MRI slice (research only)."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import prepare_image
from .utils import resolve_device, load_model
from .visualize import make_overlay


def predict_image(model, checkpoint, pil_image, device, threshold=0.5):
    tensor, rgb = prepare_image(pil_image, int(checkpoint['image_size']))
    with torch.inference_mode():
        probability = model(tensor[None].to(device)).sigmoid()[0, 0].cpu().numpy()
    binary = (probability >= threshold).astype(np.uint8)
    return rgb, binary, probability


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True, help='TIFF/PNG/JPG MRI slice (matching training channel convention)')
    p.add_argument('--output-dir', default='outputs/inference')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = p.parse_args()
    if not 0 < args.threshold < 1:
        p.error('threshold must be strictly between 0 and 1')
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    with Image.open(args.image) as im:
        rgb, mask, proba = predict_image(model, checkpoint, im, device, args.threshold)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask * 255).save(out / 'predicted_mask.png')
    Image.fromarray(make_overlay(rgb, mask)).save(out / 'overlay.png')
    Image.fromarray(np.uint8(np.clip(proba, 0, 1) * 255)).save(out / 'probability.png')
    print(f'Saved {out / "predicted_mask.png"}, overlay.png and probability.png')


if __name__ == '__main__':
    main()
