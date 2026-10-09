#!/usr/bin/env python3
"""
Run microsite segmentation inference on a folder of drone frames.

Writes per-image probability maps (.npy), binary masks (.png), and
overlay visualizations to the output folder.

  python infer.py --ckpt runs/unet_focal/best.pt --images /path/to/frames --tta --out preds/
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import cv2
import segmentation_models_pytorch as smp

from eval_tta_ensemble import build_model, prob_map, MEAN, STD  # reuse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    (out / "probs").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.ckpt).to(device).eval()

    files = sorted([p for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png")
                    for p in Path(args.images).glob(ext)])
    print(f"{len(files)} images")
    for i, p in enumerate(files, 1):
        img_rgb = np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
        prob = prob_map(model, (img_rgb - MEAN) / STD, device, args.tta)
        mask = (prob > args.thr).astype(np.uint8)

        np.save(out / "probs" / f"{p.stem}.npy", prob.astype(np.float16))
        cv2.imwrite(str(out / "masks" / f"{p.stem}_mask.png"), mask * 255)

        vis = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        vis[mask > 0] = (0.55 * vis[mask > 0] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
        cv2.imwrite(str(out / "overlays" / f"{p.stem}_overlay.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        if i % 25 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    print(f"done. masks/overlays/probs in {out}")


if __name__ == "__main__":
    main()
