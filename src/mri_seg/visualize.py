"""Save research figures (never patient-identifying metadata)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_overlay(image_uint8, binary_mask, alpha=0.40):
    out = image_uint8.astype(np.float32).copy()
    binary_mask = np.asarray(binary_mask, dtype=bool)
    red = np.array([255.0, 45.0, 45.0], dtype=np.float32)
    out[binary_mask] = out[binary_mask] * (1 - alpha) + red * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def save_sample_grid(images, ground_truths, predictions, output_file):
    """Each list holds numpy arrays (RGB image, mask, prediction)."""
    n = len(images)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(12, n * 3), squeeze=False)
    for i in range(n):
        im, gt, pred = images[i], ground_truths[i], predictions[i]
        views = [im, gt, pred, make_overlay(im, pred)]
        titles = ['MRI (3-channel)', 'Ground-truth mask', 'Predicted mask', 'Prediction overlay']
        for j, (v, title) in enumerate(zip(views, titles)):
            axes[i, j].imshow(v, cmap='gray' if j in (1, 2) else None,
                              vmin=0 if j in (1, 2) else None,
                              vmax=1 if j in (1, 2) else None)
            axes[i, j].set_title(title)
            axes[i, j].axis('off')
    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=130, bbox_inches='tight')
    plt.close(fig)


def save_training_curve(history, output_file):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row['epoch'] for row in history]
    axes[0].plot(epochs, [row['train_loss'] for row in history], label='Training')
    axes[0].plot(epochs, [row['val_loss'] for row in history], label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='BCE + Dice loss', title='Loss')
    axes[0].legend()
    axes[1].plot(epochs, [row['val_global_dice'] for row in history], label='Validation Dice')
    axes[1].set(xlabel='Epoch', ylabel='Global Dice', title='Segmentation score', ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
