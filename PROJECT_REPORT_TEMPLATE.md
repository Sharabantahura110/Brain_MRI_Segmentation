# Project report: MRI FLAIR abnormality segmentation

> Replace bracketed placeholders only after running experiments on the REAL dataset. Never report synthetic-data scores as medical image findings.

## Research question

Does incorporating attention gates into a baseline 2D U-Net improve segmentation of FLAIR abnormalities in the public TCGA-LGG dataset when both models use the same patient-level split?

## Data and protocol

- Dataset: LGG Brain MRI Segmentation (Mateusz Buda et al.)
- Participants: [number of unique patients used]
- Paired 2D image/mask slices: [number]
- Input: three MRI sequence channels, resized to [resolution]
- Patient-level split: [train patients] / [val patients] / [test patients]
- Seed: [seed]
- No patient may appear across multiple splits; validate from `data/splits.json`.

## Models

| Model | Trainable parameters | Epochs selected | Training device |
|---|---:|---:|---|
| U-Net | [TBD] | [TBD] | [TBD] |
| Attention U-Net | [TBD] | [TBD] | [TBD] |

Train with BCE+Dice loss and Adam; specify learning rate [TBD], image size [TBD], batch size [TBD].

## Test set results

| Model | Global Dice | Global IoU | Positive-GT slice Dice | Pixel recall | Pixel precision |
|---|---:|---:|---:|---:|---:|
| U-Net | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Attention U-Net | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

Include `outputs/unet_test/predictions.png`, `outputs/attention_unet_test/predictions.png` and the learning curves. Explicitly discuss empty-mask slices and patient-level generalization.

## Analysis

1. Did attention improve global Dice? By how much on the locked test set?
2. Were failure cases dominated by missing small lesions or false positive areas?
3. How did performance vary for empty-vs-positive-GT slices?
4. What are the limitations of RGB-packed multimodal MRI slices, single public dataset, and 2D modeling?

## Reproducibility

- Git commit: [TBD]
- Python/PyTorch versions: [TBD]
- Dataset source and download date: [TBD]
- CPU/GPU: [TBD]
- Random seed: [TBD]

## CV bullets (use only when factually true)

**Brain MRI FLAIR Abnormality Segmentation | Python, PyTorch, U-Net, Attention U-Net**

- Implemented a reproducible 2D MRI segmentation pipeline with patient-level dataset splitting, image/mask preprocessing, and PyTorch U-Net models.
- Compared U-Net and Attention U-Net using Dice, IoU, precision, recall, and qualitative overlay visualizations on held-out patients.
- Developed a Streamlit demonstration for research-only inference on MRI image slices.

After evaluation, add one factual measured result: "Achieved [test Dice] Dice on a [number]-patient held-out test set." Report the metric and its definition accurately.
