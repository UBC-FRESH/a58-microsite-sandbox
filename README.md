# A58 Microsite Sandbox

Sandbox for ML microsite detection from drone imagery, part of the
**IgniteBC × Flash Forest** precision aerial reforestation project
(FRESH, UBC Forestry).

A "microsite" is a small pocket of ground suitable for planting a seedling
(good soil contact, shelter, moisture). This repo trains and evaluates
models that segment microsites in low-altitude drone images of a
harvested/burned block ("A58", ~112 × 97 m flight coverage, Zenmuse P1).

## Current best model

| Model | Test IoU (v4, 54 imgs) | Notes |
|---|---|---|
| **U-Net (ResNet34) + Focal Loss + TTA** | **0.652** (0.725 clean subset) | current production candidate |
| U-Net + Focal (no TTA) | 0.638 | |
| SegFormer-B2 | 0.566 | capacity is not the bottleneck at 384 training images |
| SegFormer-B3 | 0.558 | trained and rejected (2026-10-08) |

TTA = test-time augmentation (8 dihedral views averaged); free +0.014 IoU
for CNN models. Architecture search is considered exhausted at this data
size — further gains come from more annotations (active learning), not new
models.

## Repo layout

```
code/
  train_unet_focal.py     # train U-Net+ResNet34 with focal+dice loss (the proven recipe)
  eval_tta_ensemble.py    # evaluate checkpoints ± TTA and ensembles on a test split
  infer.py                # run inference (with optional TTA) on images/folders
docs/
  DATASETS.md             # where the data lives and how to export it
requirements.txt
```

## Quickstart

```bash
conda create -n microsite python=3.12 -y && conda activate microsite
pip install -r requirements.txt          # needs a CUDA GPU for training

# 1. get a dataset export (see docs/DATASETS.md) and unpack to data/roboflow_v4
# 2. train
python code/train_unet_focal.py --data data/roboflow_v4 --out runs/unet_focal

# 3. evaluate with TTA
python code/eval_tta_ensemble.py --data data/roboflow_v4 --ckpt runs/unet_focal/best.pt

# 4. inference on new drone frames
python code/infer.py --ckpt runs/unet_focal/best.pt --images /path/to/frames --tta --out preds/
```

## Notes for collaborators

- Model weights and images are **not** in this repo (size/licensing) — see
  docs/DATASETS.md for access.
- The Roboflow sandbox project for annotation practice is
  `FRESH Workspace / a58-sandbox` (750 raw flight frames, unannotated).
- The annotated dataset projects are `a58-microsites` (semantic) and
  `a58-features` (instance) in the same workspace.

## License

Code: MIT (see LICENSE). The imagery and annotations are licensed
BY-NC-SA 4.0 via Roboflow — do not redistribute them outside the project.
