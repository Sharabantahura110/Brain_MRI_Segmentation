"""Train 2D segmentation; best checkpoint selected ONLY with validation data."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import load_manifest, MRISliceDataset, split_summary
from .metrics import MetricsAccumulator, bce_dice_loss
from .models import build_model
from .utils import resolve_device, seed_everything
from .visualize import save_training_curve


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    metrics = MetricsAccumulator()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            image = batch['image'].to(device)
            target = batch['mask'].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = bce_dice_loss(logits, target)
            if training:
                loss.backward()
                optimizer.step()
            total_samples += image.size(0)
            total_loss += loss.item() * image.size(0)
            metrics.update(logits, target)
    return total_loss / max(1, total_samples), metrics.compute()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', default='data/splits.json')
    p.add_argument('--model', choices=['unet', 'attention_unet'], default='unet')
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--base-channels', type=int, default=16)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--output-dir', default=None)
    args = p.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.base_channels < 2:
        p.error('epochs, patience must be positive and base-channels must be >= 2')

    seed_everything(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output_dir or f'outputs/{args.model}')
    output.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    train_ds = MRISliceDataset(args.data_root, manifest['splits']['train'], args.image_size, augment=True)
    val_ds = MRISliceDataset(args.data_root, manifest['splits']['val'], args.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    model = build_model(args.model, args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = vars(args).copy()
    config['resolved_device'] = str(device)
    config['split_summary'] = split_summary(manifest)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    history = []
    best_score = float('-inf')
    stale_epochs = 0
    print(f'Device: {device}; model: {args.model}; train slices: {len(train_ds)}; val slices: {len(val_ds)}', flush=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, device)
        score = val_metrics['global_dice']
        if score is None:
            score = 0.0
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
               'train_global_dice': train_metrics['global_dice'] or 0.0,
               'val_global_dice': score,
               'val_mean_slice_dice_nonempty_gt': val_metrics['mean_slice_dice_nonempty_gt']}
        history.append(row)
        print(f"Epoch {epoch:03d} | train loss={train_loss:.4f} | val loss={val_loss:.4f} | val global Dice={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            stale_epochs = 0
            torch.save({
                'model_state': model.state_dict(), 'model_name': args.model,
                'base_channels': args.base_channels, 'image_size': args.image_size,
                'epoch': epoch, 'best_val_dice': best_score,
            }, output / 'best.pt')
        else:
            stale_epochs += 1

        with (output / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)
        save_training_curve(history, output / 'training_curve.png')
        if stale_epochs >= args.patience:
            print(f'Early stopping after {epoch} epochs; best validation global Dice={best_score:.4f}')
            break
    print(f'Best checkpoint: {output / "best.pt"}')


if __name__ == '__main__':
    main()
