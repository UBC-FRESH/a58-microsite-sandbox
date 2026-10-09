#!/usr/bin/env python3
"""
Evaluate segmentation checkpoints on a test split, with optional TTA
(8 dihedral views averaged) and optional ensembling (average prob maps).

Examples:
  python eval_tta_ensemble.py --data data/roboflow_v4 --ckpt runs/unet_focal/best.pt
  python eval_tta_ensemble.py --data data/roboflow_v4 --ckpt a.pt b.pt --tta --ensemble
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(ckpt_path):
    """Guess architecture from the state dict."""
    sd = torch.load(ckpt_path, map_location="cpu")
    keys = " ".join(sd.keys())
    if "encoder.stages" in keys or "blocks.0.skip" in keys:
        # SegFormer family; infer size from patch embed dim
        dim = sd["encoder.patch_embed1.proj.weight"].shape[0]
        name = {32: "mit_b0", 64: "mit_b1", 128: "mit_b3"}.get(dim, "mit_b2")
        m = smp.Segformer(encoder_name=name, encoder_weights=None, in_channels=3, classes=2)
    else:
        m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=2)
    m.load_state_dict(sd)
    return m


def dihedral(x, k, flip):
    if flip:
        x = np.flip(x, axis=3)
    return np.rot90(x, k, axes=(2, 3))


def dihedral_inv(p, k, flip):
    p = np.rot90(p, -k, axes=(0, 1))
    if flip:
        p = np.flip(p, axis=1)
    return p


@torch.no_grad()
def prob_map(model, img, device, tta=False):
    probs = []
    x0 = img.transpose(2, 0, 1)[None]
    ops = [(0, False)] if not tta else [(k, f) for k in range(4) for f in (False, True)]
    for k, f in ops:
        x = dihedral(x0, k, f)
        t = torch.from_numpy(np.ascontiguousarray(x).astype(np.float32)).to(device)
        p = torch.softmax(model(t), 1)[0, 1].cpu().numpy()
        probs.append(dihedral_inv(p, k, f))
    return np.mean(probs, axis=0)


def metrics(pred, gt):
    tp = (pred & (gt == 1)).sum(); fp = (pred & (gt != 1)).sum(); fn = (~pred & (gt == 1)).sum()
    return tp / max(tp + fp + fn, 1), tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--ensemble", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = Path(args.data) / "test"
    items = []
    for p in sorted(test_dir.glob("*.jpg")):
        if p.name.endswith("_mask.png"):
            continue
        img = np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
        gt = np.array(Image.open(test_dir / f"{p.stem}_mask.png")).astype(np.int64)
        items.append(((img - MEAN) / STD, gt))
    print(f"{len(items)} test images")

    per_model_probs = []
    for c in args.ckpt:
        model = build_model(c).to(device).eval()
        probs = [prob_map(model, img, device, args.tta) for img, _ in items]
        per_model_probs.append(probs)
        ious = [metrics(pr > 0.5, gt)[0] for pr, (_, gt) in zip(probs, items)]
        print(f"{Path(c).name} (tta={args.tta}): IoU {np.mean(ious):.4f}")

    if args.ensemble and len(per_model_probs) > 1:
        ens = [np.mean(ps, axis=0) for ps in zip(*per_model_probs)]
        ious = [metrics(pr > 0.5, gt)[0] for pr, (_, gt) in zip(ens, items)]
        print(f"ENSEMBLE({len(args.ckpt)} models, tta={args.tta}): IoU {np.mean(ious):.4f}")


if __name__ == "__main__":
    main()
