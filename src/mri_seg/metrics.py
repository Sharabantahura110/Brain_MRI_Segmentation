"""Segmentation loss and evaluation, explicitly treating empty-mask cases."""
import numpy as np
import torch
from torch.nn import functional as F


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = logits.sigmoid()
    intersection = (probs * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1.0) / (
        probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    )
    return bce + (1 - dice.mean())


class MetricsAccumulator:
    """Global pixel metrics + macro slice scores (including and excluding empty GT)."""
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0
        self.dice_slices = []
        self.iou_slices = []
        self.dice_positive_slices = []
        self.iou_positive_slices = []
        self.n_empty_gt = 0

    def update(self, logits, target, threshold=0.5):
        pred = (logits.sigmoid() >= threshold).detach().cpu().numpy().astype(bool)
        truth = target.detach().cpu().numpy().astype(bool)
        for p, t in zip(pred, truth):
            tp = int(np.logical_and(p, t).sum())
            fp = int(np.logical_and(p, ~t).sum())
            fn = int(np.logical_and(~p, t).sum())
            tn = int(np.logical_and(~p, ~t).sum())
            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.tn += tn
            if not t.any():
                self.n_empty_gt += 1
            # If both masks empty, per-slice Dice and IoU are defined as 1.
            denom_dice = 2 * tp + fp + fn
            denom_iou = tp + fp + fn
            dice = 2 * tp / denom_dice if denom_dice else 1.0
            iou = tp / denom_iou if denom_iou else 1.0
            self.dice_slices.append(dice)
            self.iou_slices.append(iou)
            if t.any():
                self.dice_positive_slices.append(dice)
                self.iou_positive_slices.append(iou)

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        def safe(n, d):
            return float(n / d) if d else None
        def avg(vals):
            return float(np.mean(vals)) if vals else None
        return {
            'n_slices': len(self.dice_slices),
            'n_empty_ground_truth_slices': self.n_empty_gt,
            'global_dice': safe(2 * tp, 2 * tp + fp + fn),
            'global_iou': safe(tp, tp + fp + fn),
            'pixel_precision': safe(tp, tp + fp),
            'pixel_recall': safe(tp, tp + fn),
            'pixel_specificity': safe(tn, tn + fp),
            'mean_slice_dice_all': avg(self.dice_slices),
            'mean_slice_iou_all': avg(self.iou_slices),
            'mean_slice_dice_nonempty_gt': avg(self.dice_positive_slices),
            'mean_slice_iou_nonempty_gt': avg(self.iou_positive_slices),
            'confusion_pixels': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
        }
