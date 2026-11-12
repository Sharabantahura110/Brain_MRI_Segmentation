# Brain MRI FLAIR Abnormality Segmentation: U-Net vs Attention U-Net

**Portfolio project | Python · PyTorch · medical image analysis · segmentation · Streamlit**

A reproducible 2D deep-learning pipeline comparing **U-Net** and **Attention U-Net** on the public LGG (TCGA) MRI segmentation dataset. The target is the **annotated FLAIR abnormality mask**, not a confirmed medical diagnosis or a tumor subtype.

> **Research/educational use only.** Not for patient diagnosis, treatment, or medical decisions. The repository does **not** include patient images, annotations, trained models, or claimed performance results.

## Why this makes a useful CV project

- Builds two segmentation architectures in native PyTorch, including attention-gated skip connections.
- Implements a **patient-level train/validation/test split** to avoid leakage from adjacent slices.
- Trains with binary cross-entropy + soft Dice loss; saves the best validation checkpoint.
- Evaluates with **global Dice, global IoU, precision, recall, specificity**, and mean slice metrics both with and without empty ground-truth masks.
- Generates training curves and MRI/mask/prediction/overlay comparison figures.
- Provides a command-line predictor and a local Streamlit demo.
- Includes a synthetic test dataset generator and automated tests. Synthetic scores are **not** scientific results.

## Dataset (download separately)

**Buda et al., LGG Brain MRI Segmentation**, published on Kaggle:
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

- 110 patients from the TCGA lower-grade glioma collection.
- 3,929 MRI slices with corresponding masks (3,929 pairs).
- 3-channel TIFF images: channels encode co-registered MRI sequences (not ordinary color photographs). Masks label FLAIR abnormalities.
- Organized as one directory per patient with `_mask.tif` for each input `.tif`.
- Dataset license: **CC BY-NC-SA 4.0**; respect attribution, noncommercial, and share-alike terms. Do not upload dataset files to your repository.

Citation: Buda, M., Saha, A., & Mazurowski, M. A. (2019). *Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm.* Computers in Biology and Medicine, 109, 218–225.

**Expected layout after unzip** (the outer folder name may differ):

```text
data/raw/lgg/
└── kaggle_3m/
    ├── TCGA_CS_4941_19960909/
    │   ├── TCGA_CS_4941_19960909_1.tif
    │   ├── TCGA_CS_4941_19960909_1_mask.tif
    │   └── ...
    └── ...
```

The indexing tool scans recursively. Its `--data-root` can point to `data/raw/lgg` or `data/raw/lgg/kaggle_3m`.

## Setup

Requirements: **Python 3.10+**; Windows, macOS (CPU/MPS), or Linux (CPU/CUDA). A GPU is recommended for 256×256 training. CPU can be used for testing.

```bash
# Run these commands from the project root
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The default pip installation of PyTorch depends on your platform. If you need a specific CUDA/ROCm build, follow https://pytorch.org/get-started/locally/ before installing the remaining dependencies. Check the selected device in the training output.

## Download and index the real data

Either download and extract manually from Kaggle, or (with Kaggle CLI/authentication configured):

```bash
pip install kaggle
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p data/raw/lgg --unzip
```

Then build a split manifest; same manifest must be used by **both** models:

```bash
python -m mri_seg.prepare \
  --data-root data/raw/lgg \
  --manifest data/splits.json \
  --seed 42
```

This writes relative image paths and patient IDs. The train/val/test split is approximately **70% / 15% / 15% of patients** (not exactly of image slices). `data/splits.json` should be retained when reporting results.

## Train U-Net

```bash
python -m mri_seg.train \
  --data-root data/raw/lgg \
  --manifest data/splits.json \
  --model unet \
  --epochs 20 \
  --batch-size 8 \
  --image-size 256 \
  --base-channels 16 \
  --output-dir outputs/unet
```

Outputs: `best.pt`, `config.json`, `history.csv`, `training_curve.png`. Early stopping defaults to patience 6; training runs on the detected CPU, CUDA GPU, or Apple MPS device.

## Train Attention U-Net

```bash
python -m mri_seg.train \
  --data-root data/raw/lgg \
  --manifest data/splits.json \
  --model attention_unet \
  --epochs 20 \
  --batch-size 8 \
  --image-size 256 \
  --base-channels 16 \
  --output-dir outputs/attention_unet
```

## Evaluate on the untouched test set

Do not use test-set metrics to choose epochs, thresholds, or architectures. Best checkpoint selection happens on validation data only.

```bash
python -m mri_seg.evaluate \
  --data-root data/raw/lgg \
  --manifest data/splits.json \
  --checkpoint outputs/unet/best.pt \
  --output-dir outputs/unet_test

python -m mri_seg.evaluate \
  --data-root data/raw/lgg \
  --manifest data/splits.json \
  --checkpoint outputs/attention_unet/best.pt \
  --output-dir outputs/attention_unet_test

python scripts/compare_models.py \
  --unet outputs/unet_test/metrics.json \
  --attention outputs/attention_unet_test/metrics.json
```

You get `metrics.json` and `predictions.png` per model, plus `outputs/comparison.csv` from the comparison script.

**Metric definitions:** Global Dice and IoU aggregate pixel true positives/false positives/false negatives across the test set. `mean_slice_dice_all` counts an empty target and empty prediction as Dice 1. `mean_slice_dice_nonempty_gt` excludes empty-GT slices to avoid inflated results. Precision/recall with zero denominator are recorded as `null`, not invented as 100%.

## Predict on one slice

```bash
python -m mri_seg.infer \
  --checkpoint outputs/unet/best.pt \
  --image data/raw/lgg/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1.tif \
  --output-dir outputs/inference
```

This path is an **example**; use an image file that actually exists on your computer. The command writes `predicted_mask.png`, `overlay.png` and `probability.png` at the training image resolution. Inference expects the same 3-channel sequence convention as the training images. Uploaded photos or unrelated single-channel scans are not valid substitutes.

## Launch Streamlit

```bash
streamlit run app.py
```

Enter the path to a trained `best.pt`, upload a compatible MRI slice, and see its predicted mask/overlay. The local app cannot perform medically validated analysis.

## Quick functional test *without* the dataset

A tiny, **synthetic** dataset is included as a generator to confirm that your environment and pipeline work. These toy images are not MRI scans and no results obtained from them should be added to your CV.

```bash
python scripts/make_synthetic_data.py --output data/raw/synthetic --patients 12 --slices 3 --size 96
python -m mri_seg.prepare --data-root data/raw/synthetic --manifest data/synthetic_splits.json
python -m mri_seg.train --data-root data/raw/synthetic --manifest data/synthetic_splits.json --model unet --image-size 64 --base-channels 4 --batch-size 4 --epochs 1 --device cpu --output-dir outputs/synthetic_unet
python -m mri_seg.evaluate --data-root data/raw/synthetic --manifest data/synthetic_splits.json --checkpoint outputs/synthetic_unet/best.pt --output-dir outputs/synthetic_test --device cpu
pytest -q
```

## Project tree

```text
brain-mri-segmentation/
├── README.md
├── PROJECT_REPORT_TEMPLATE.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── app.py                          # Streamlit local demo
├── src/mri_seg/
│   ├── __init__.py
│   ├── data.py                      # paired image/mask loader, patient splits
│   ├── prepare.py                   # indexed manifest CLI
│   ├── models.py                    # U-Net / Attention U-Net
│   ├── metrics.py                   # BCE+Dice loss and honest segmentation metrics
│   ├── train.py                     # training & validation
│   ├── evaluate.py                  # held-out evaluation and figures
│   ├── infer.py                     # single-image inference
│   ├── utils.py                     # seeds, checkpoints, compute device
│   └── visualize.py                 # overlays and charts
├── scripts/
│   ├── make_synthetic_data.py
│   └── compare_models.py
├── tests/test_pipeline.py
├── data/raw/                       # real data here (ignored by Git)
└── outputs/                        # weights and generated figures (ignored by Git)
```

## Fair comparison checklist

1. Same patient split, resolution, training epochs/patience, batch size, learning rate and loss for both models.
2. Choose checkpoints only using validation scores.
3. Report model size, training device, seed and overall/positive-only evaluation metrics.
4. Include a figure with original 3-channel slice, ground truth, predicted mask, and overlay.
5. Distinguish observations from clinical claims. These 2D slice models have not been validated across hospitals, scanners, patient populations, or clinical workflows.

## Possible extensions

- Try 3 random seeds and report mean ± standard deviation.
- Compare a 1-channel FLAIR-only baseline to all available 3 channels, documenting which channel is FLAIR and how missing channels are handled.
- Add a held-out-site experiment if hospital labels and adequate sample sizes permit.
- Analyze slices with and without lesions separately and investigate prediction failure cases.
- Export your model to ONNX only after verifying it produces similar predictions to the PyTorch version.

## License and privacy notes

Code in this repository is original educational implementation. Dataset licensing is independent of code licensing; honor the dataset's CC BY-NC-SA 4.0 terms. Avoid redistributing actual patient data or displaying patient-identifiable metadata. Any images or predictions from real MRI are research outputs, not clinical evidence.
