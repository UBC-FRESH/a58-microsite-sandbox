# Datasets

Nothing in this repo includes imagery or weights. Get them from the
**FRESH Workspace** on Roboflow (`gregory-paradis-ubc-ca`):

| Project | Type | Contents |
|---|---|---|
| `a58-sandbox` | instance segmentation | 750 raw A58 flight frames, **unannotated** — the practice sandbox |
| `a58-microsites` | semantic segmentation | 548 annotated frames (the v4 microsite dataset) |
| `a58-features` | instance segmentation | feature annotations (slash/log/rock/stump) |

## Exporting the training set

1. In Roboflow, open `a58-microsites` → Versions → pick the latest version.
2. Export as **PNG Masks for Semantic Segmentation**.
3. Unpack to `data/roboflow_v4/` so it looks like:

```
data/roboflow_v4/
  train/           # *.jpg + *_mask.png
  valid/
  test/
```

## Raw flight frames

The 750-frame A58 flight (frames 0001–0750, 1024×682 downscales) is in the
`a58-sandbox` project. Full-res originals (8192×5460 with GPS EXIF) live on
the project server — ask Gregory for the path.

## Model checkpoints

Current production candidate: `unet_resnet34_focal_best.pt` (U-Net ResNet34,
focal loss). Ask Gregory for a copy; put it anywhere and point
`--ckpt` at it.
