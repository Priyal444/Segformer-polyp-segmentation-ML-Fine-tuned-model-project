"""
Fine-tune SegFormer-b0 for polyp segmentation on Kvasir-SEG.

Data setup (run once):
    wget https://datasets.simula.no/downloads/kvasir-seg.zip
    unzip kvasir-seg.zip
  -> gives Kvasir-SEG/images/*.jpg and Kvasir-SEG/masks/*.jpg with matching names.

Train:
    python train.py --data-root Kvasir-SEG --epochs 40

External validation (domain shift check) on CVC-ClinicDB:
    python train.py --eval-only --ckpt best.pt \
        --ext-images CVC-ClinicDB/Original --ext-masks "CVC-ClinicDB/Ground Truth"
"""

import argparse
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class PolypDataset(Dataset):
    """Image/mask pairs. Masks are binarised at 127 -> {0, 1}."""

    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"could not read mask: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].unsqueeze(0).float()


def build_pairs(images_dir, masks_dir):
    """Match images to masks by filename stem, so .jpg/.tif/.png mismatches are fine."""
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    masks_by_stem = {
        p.stem: p for p in masks_dir.iterdir() if p.suffix.lower() in exts
    }
    pairs = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in exts:
            continue
        mask = masks_by_stem.get(img.stem)
        if mask is not None:
            pairs.append((img, mask))

    if not pairs:
        raise RuntimeError(f"no image/mask pairs found in {images_dir} + {masks_dir}")
    return pairs


def split_pairs(pairs, seed, val_frac, test_frac):
    """Deterministic train/val/test split. stress_test.py calls this too, so it
    reproduces exactly the same held-out test images."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n_val = int(len(pairs) * val_frac)
    n_test = int(len(pairs) * test_frac)
    val_pairs = pairs[:n_val]
    test_pairs = pairs[n_val:n_val + n_test]
    train_pairs = pairs[n_val + n_test:]
    return train_pairs, val_pairs, test_pairs


def make_transforms(size):
    train_tf = A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(0.8, 1.2), rotate=(-30, 30),
                 border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        # Endoscopy footage varies a lot in illumination and white balance --
        # colour jitter is doing real work here, not just padding the pipeline.
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.7),
        # Set explicitly: the albumentations 2.x default (std ~0.2-0.44 of full range)
        # is far heavier than mild sensor noise. 0.01-0.03 is roughly std 3-8 on 0-255.
        A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ])
    eval_tf = A.Compose([
        A.Resize(size, size),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_tf, eval_tf


# --------------------------------------------------------------------------- #
# Model / loss / metrics
# --------------------------------------------------------------------------- #

def build_model(encoder="nvidia/mit-b0"):
    """mit-b0 is the ImageNet-pretrained encoder; the decode head starts fresh."""
    return SegformerForSemanticSegmentation.from_pretrained(
        encoder, num_labels=1, ignore_mismatched_sizes=True
    )


def forward_logits(model, images, out_size):
    """SegFormer emits logits at 1/4 resolution -- upsample to match the mask."""
    logits = model(pixel_values=images).logits
    return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class BCEDiceLoss(nn.Module):
    """BCE keeps per-pixel gradients alive; soft Dice handles the class imbalance
    (polyps are a small fraction of most frames)."""

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1, 2, 3)) + self.smooth
        den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth
        dice = 1 - (num / den).mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


@torch.no_grad()
def batch_counts(probs, targets, threshold):
    """Per-image TP/FP/FN so metrics are averaged over images, not over pixels.
    Pixel-pooled Dice flatters you: big polyps drown out the small ones you miss."""
    preds = (probs > threshold).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    return tp, fp, fn


def counts_to_metrics(tp, fp, fn, eps=1e-7):
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    recall = (tp + eps) / (tp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "recall": recall.mean().item(),
        "precision": precision.mean().item(),
    }


@torch.no_grad()
def evaluate(model, loader, device, thresholds=(0.5,)):
    model.eval()
    acc = {t: [[], [], []] for t in thresholds}

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = forward_logits(model, images, masks.shape[-2:])
        probs = torch.sigmoid(logits.float())
        for t in thresholds:
            tp, fp, fn = batch_counts(probs, masks, t)
            acc[t][0].append(tp)
            acc[t][1].append(fp)
            acc[t][2].append(fn)

    return {
        t: counts_to_metrics(torch.cat(a), torch.cat(b), torch.cat(c))
        for t, (a, b, c) in acc.items()
    }


@torch.no_grad()
def evaluate_stratified_by_size(model, loader, device, threshold=0.5):
    """Breakdown Dice, recall, and precision by polyp area with explicit sample counts (N).
    Small: < 1024 px (< ~0.7% of 384x384)
    Medium: 1024 - 9216 px (~0.7% - 6.25%)
    Large: > 9216 px (> 6.25%)
    """
    model.eval()
    all_tp, all_fp, all_fn, all_areas = [], [], [], []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = forward_logits(model, images, masks.shape[-2:])
        probs = torch.sigmoid(logits.float())
        tp, fp, fn = batch_counts(probs, masks, threshold)
        areas = masks.sum(dim=(1, 2, 3))
        all_tp.append(tp)
        all_fp.append(fp)
        all_fn.append(fn)
        all_areas.append(areas)

    tp = torch.cat(all_tp)
    fp = torch.cat(all_fp)
    fn = torch.cat(all_fn)
    areas = torch.cat(all_areas)

    bins = {
        "Small (< 1024 px)": areas < 1024,
        "Medium (1024-9216 px)": (areas >= 1024) & (areas <= 9216),
        "Large (> 9216 px)": areas > 9216,
    }

    results = {}
    for name, mask in bins.items():
        n = int(mask.sum().item())
        if n > 0:
            m = counts_to_metrics(tp[mask], fp[mask], fn[mask])
            m["N"] = n
            results[name] = m
        else:
            # Empty bin: report N=0 rather than fake zero scores.
            results[name] = {"N": 0}
    return results


# --------------------------------------------------------------------------- #
# Train / eval entry points
# --------------------------------------------------------------------------- #

def run_training(args, device):
    pairs = build_pairs(Path(args.data_root) / "images", Path(args.data_root) / "masks")
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, args.seed, args.val_frac, args.test_frac)
    print(f"train {len(train_pairs)} | val {len(val_pairs)} | test {len(test_pairs)}")

    train_tf, eval_tf = make_transforms(args.size)
    loader_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(PolypDataset(train_pairs, train_tf), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(PolypDataset(val_pairs, eval_tf), batch_size=args.batch_size, **loader_kw)
    test_loader = DataLoader(PolypDataset(test_pairs, eval_tf), batch_size=args.batch_size, **loader_kw)

    model = build_model(args.encoder).to(device)

    # Lower LR on the pretrained encoder than on the randomly-initialised head.
    head_params, enc_params = [], []
    for name, p in model.named_parameters():
        (head_params if "decode_head" in name else enc_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr},
         {"params": head_params, "lr": args.lr * 10}],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, args.lr * 10],
        total_steps=args.epochs * len(train_loader), pct_start=0.1,
    )
    criterion = BCEDiceLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = forward_logits(model, images, masks.shape[-2:])
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        val = evaluate(model, val_loader, device)[0.5]
        print(f"epoch {epoch:3d} | loss {running / len(train_loader):.4f} | "
              f"val dice {val['dice']:.4f} iou {val['iou']:.4f} recall {val['recall']:.4f}")

        if val["dice"] > best_dice:
            best_dice = val["dice"]
            torch.save({"model": model.state_dict(), "encoder": args.encoder,
                        "size": args.size, "val_dice": best_dice}, args.ckpt)
            print(f"           saved {args.ckpt} (dice {best_dice:.4f})")

    # Threshold sweep on val only -- picking it on test would leak.
    ckpt_data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt_data["model"])
    sweep = evaluate(model, val_loader, device, thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7))
    print("\nthreshold sweep (val):")
    best_t = 0.5
    best_val_dice_t = -1.0
    for t, m in sorted(sweep.items()):
        print(f"  t={t:.2f}  dice {m['dice']:.4f}  recall {m['recall']:.4f}  precision {m['precision']:.4f}")
        if m["dice"] > best_val_dice_t:
            best_val_dice_t = m["dice"]
            best_t = t

    print(f"\noptimal val threshold: t*={best_t:.2f} (val dice: {best_val_dice_t:.4f})")
    ckpt_data["best_threshold"] = best_t
    torch.save(ckpt_data, args.ckpt)

    report_test_metrics("held-out test", model, test_loader, device, best_t)


def report_test_metrics(label, model, loader, device, best_t):
    """Overall + size-stratified metrics at the 0.5 headline and at the val-chosen t*."""
    thresholds = (0.5,) if best_t == 0.5 else (0.5, best_t)
    overall = evaluate(model, loader, device, thresholds=thresholds)
    print(f"\n{label} @ 0.5 (headline):", overall[0.5])
    if best_t != 0.5:
        print(f"{label} @ t*={best_t:.2f} (chosen on val):", overall[best_t])

    for t in thresholds:
        print(f"\n{label} breakdown by size @ {t:.2f}:")
        for k, v in evaluate_stratified_by_size(model, loader, device, threshold=t).items():
            if v["N"] == 0:
                print(f"  {k} (N=0): no images in this bin")
                continue
            print(f"  {k} (N={v['N']}): dice {v['dice']:.4f} recall {v['recall']:.4f} "
                  f"precision {v['precision']:.4f}")


def run_external_eval(args, device):
    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model(ckpt.get("encoder", args.encoder)).to(device)
    model.load_state_dict(ckpt["model"])
    best_t = ckpt.get("best_threshold", 0.5)

    _, eval_tf = make_transforms(ckpt.get("size", args.size))
    pairs = build_pairs(args.ext_images, args.ext_masks)
    loader = DataLoader(PolypDataset(pairs, eval_tf), batch_size=args.batch_size,
                        num_workers=args.workers)

    print(f"external set: {len(pairs)} images")
    report_test_metrics("external", model, loader, device, best_t)

    print("\nCompare against the in-domain test score. The gap is the headline result.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="Kvasir-SEG")
    p.add_argument("--encoder", default="nvidia/mit-b0")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--ext-images")
    p.add_argument("--ext-masks")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.eval_only:
        if not (args.ext_images and args.ext_masks):
            raise SystemExit("--eval-only needs --ext-images and --ext-masks")
        run_external_eval(args, device)
    else:
        run_training(args, device)


if __name__ == "__main__":
    main()
