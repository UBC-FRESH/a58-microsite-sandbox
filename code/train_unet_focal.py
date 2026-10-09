#!/usr/bin/env python3
"""
Train U-Net (ResNet34) with focal+dice loss for binary microsite segmentation.

The proven recipe on the A58 v4 dataset (test IoU 0.638; 0.652 with TTA):
  - smp.Unet, resnet34 encoder, ImageNet init
  - 0.5 * focal(gamma=2, alpha=0.25) + 0.5 * dice
  - AdamW lr 3e-4, cosine schedule, batch 8, 1024x1024 images
  - dihedral augmentation (flips + 90-degree rotations)
  - early stopping (patience 12), best checkpoint by val IoU saved

Usage:
  python train_unet_focal.py --data data/roboflow_v4 --out runs/unet_focal
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class SegDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.dir = Path(root) / split
        self.imgs = sorted(p for p in self.dir.glob("*.jpg")
                           if not p.name.endswith("_mask.png"))
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
        mask = np.array(Image.open(self.dir / f"{p.stem}_mask.png")).astype(np.int64)
        img = img.transpose(2, 0, 1)
        if self.augment:
            if np.random.random() > 0.5:
                img = np.flip(img, 2).copy(); mask = np.fliplr(mask).copy()
            if np.random.random() > 0.5:
                img = np.flip(img, 1).copy(); mask = np.flipud(mask).copy()
            k = np.random.randint(0, 4)
            if k:
                img = np.rot90(img, k, (1, 2)).copy(); mask = np.rot90(mask, k).copy()
        img = (img - MEAN[:, None, None]) / STD[:, None, None]
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(mask)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, pred, target):
        ce = nn.functional.cross_entropy(pred, target, reduction="none")
        p_t = torch.softmax(pred, 1).gather(1, target.unsqueeze(1)).squeeze(1)
        return (self.alpha * (1 - p_t) ** self.gamma * ce).mean()


def binary_iou(pred, target, eps=1e-7):
    pf, tf = pred == 1, target == 1
    inter = (pf & tf).sum().item()
    return (inter + eps) / ((pf | tf).sum().item() + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    loaders = {
        s: DataLoader(SegDataset(args.data, s, augment=(s == "train")),
                      batch_size=args.batch, shuffle=(s == "train"),
                      num_workers=6, pin_memory=True)
        for s in ["train", "valid", "test"]
    }
    print({s: len(l.dataset) for s, l in loaders.items()})

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=3, classes=2).to(device)
    focal, dice = FocalLoss(), smp.losses.DiceLoss(mode="multiclass")
    loss_fn = lambda p, t: 0.5 * focal(p, t) + 0.5 * dice(p, t)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_iou, best_ep, bad = 0.0, 0, 0
    history = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tl = 0
        for x, m in loaders["train"]:
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), m)
            loss.backward()
            opt.step()
            tl += loss.item()
        sched.step()

        model.eval()
        vl, ious = 0, []
        with torch.no_grad():
            for x, m in loaders["valid"]:
                x, m = x.to(device), m.to(device)
                o = model(x)
                vl += loss_fn(o, m).item()
                for p, t in zip(o.argmax(1), m):
                    ious.append(binary_iou(p.cpu(), t.cpu()))
        viou = float(np.mean(ious))
        history.append({"epoch": ep, "train_loss": tl / len(loaders["train"]),
                        "val_loss": vl / len(loaders["valid"]), "val_iou": viou})
        improved = viou > best_iou
        if improved:
            best_iou, best_ep, bad = viou, ep, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
        print(f"{ep:>3} tl={tl/len(loaders['train']):.4f} vl={vl/len(loaders['valid']):.4f} "
              f"valIoU={viou:.4f} {time.time()-t0:.0f}s{' *' if improved else ''}", flush=True)
        if bad >= args.patience:
            print(f"early stop @ {ep} (best {best_ep}: {best_iou:.4f})")
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    model.eval()
    ious = []
    with torch.no_grad():
        for x, m in loaders["test"]:
            x = x.to(device)
            for p, t in zip(model(x).argmax(1), m):
                ious.append(binary_iou(p.cpu(), t.cpu()))
    print(f"\nTEST: IoU {np.mean(ious):.4f} ± {np.std(ious):.4f} "
          f"(best val {best_iou:.4f} @ epoch {best_ep})")
    torch.save(model.state_dict(), out / "final.pt")
    with open(out / "history.json", "w") as f:
        json.dump({"test_iou": float(np.mean(ious)), "test_iou_std": float(np.std(ious)),
                   "best_val_iou": best_iou, "best_epoch": best_ep,
                   "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
