"""
Corruption stress test for SegFormer-b0 polyp segmentation.
Evaluates model degradation under 3 natural endoscopy image corruptions:
  1. Motion blur (scope movement)
  2. Low illumination (cavity lighting variations)
  3. Specular glare (fluid/mucosal light reflection)

By default it runs on the same held-out Kvasir-SEG test split that train.py uses,
so the curves measure the corruption alone, not corruption + hospital shift:
    python stress_test.py --ckpt best.pt --data-root Kvasir-SEG

Pass --images/--masks instead to stress-test an external folder (e.g. CVC-ClinicDB).
There, severity 0 is already out-of-distribution, so read each curve relative to
its own severity-0 point.

Corruptions are seeded per image, so reruns give identical numbers.

Outputs:
  - stress_test_results.csv (severity 0-5 vs Dice for each corruption)
  - stress_test_curve.png (performance degradation curves)
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train import IMAGENET_MEAN, IMAGENET_STD, build_model, build_pairs, evaluate, split_pairs

SEVERITIES = [0, 1, 2, 3, 4, 5]
CORRUPTIONS = ["motion_blur", "low_light", "specular_glare"]

# Per-severity parameters (index 0 = severity 1).
BLUR_KSIZE = [5, 9, 15, 21, 29]
# Multiplicative brightness. Training ColorJitter already covers 0.8-1.2,
# so severities 1-2 are expected to barely move the score.
LIGHT_FACTOR = [0.9, 0.75, 0.6, 0.45, 0.3]
GLARE_SPOTS = [2, 4, 6, 8, 12]
GLARE_RADIUS = [10, 15, 20, 25, 30]


def motion_blur(img, ksize, angle):
    kernel = np.zeros((ksize, ksize), np.float32)
    kernel[ksize // 2, :] = 1.0
    centre = ((ksize - 1) / 2, (ksize - 1) / 2)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D(centre, angle, 1.0), (ksize, ksize))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def low_light(img, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def specular_glare(img, n_spots, radius, rng):
    """Soft white spots blended in; only the spots are blurred, not the image."""
    h, w = img.shape[:2]
    alpha = np.zeros((h, w), np.float32)
    for _ in range(n_spots):
        cx = int(rng.integers(radius, w - radius))
        cy = int(rng.integers(radius, h - radius))
        cv2.circle(alpha, (cx, cy), radius, 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=radius / 3)[..., None]
    out = img.astype(np.float32) * (1 - alpha) + 255.0 * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_corruption(img, corruption, severity, rng):
    if severity == 0:
        return img
    i = severity - 1
    if corruption == "motion_blur":
        return motion_blur(img, BLUR_KSIZE[i], angle=float(rng.uniform(0, 180)))
    if corruption == "low_light":
        return low_light(img, LIGHT_FACTOR[i])
    if corruption == "specular_glare":
        return specular_glare(img, GLARE_SPOTS[i], GLARE_RADIUS[i], rng)
    raise ValueError(f"unknown corruption: {corruption}")


class CorruptedPolypDataset(Dataset):
    def __init__(self, pairs, corruption, severity, size=384, seed=0):
        self.pairs = pairs
        self.corruption = corruption
        self.severity = severity
        self.size = size
        self.seed = seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"could not read: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"could not read mask: {mask_path}")
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        # Same seed per image -> identical corruption on every rerun.
        rng = np.random.default_rng([self.seed, idx])
        image = apply_corruption(image, self.corruption, self.severity, rng)

        arr = (image.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN, np.float32)) \
            / np.array(IMAGENET_STD, np.float32)
        return torch.from_numpy(arr.transpose(2, 0, 1)), torch.from_numpy(mask).unsqueeze(0)


def load_pairs(args):
    if args.images or args.masks:
        if not (args.images and args.masks):
            raise SystemExit("--images and --masks must be given together")
        return build_pairs(args.images, args.masks), "external"
    pairs = build_pairs(Path(args.data_root) / "images", Path(args.data_root) / "masks")
    _, _, test_pairs = split_pairs(pairs, args.seed, args.val_frac, args.test_frac)
    return test_pairs, "Kvasir-SEG held-out test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="best.pt", help="Path to trained checkpoint")
    parser.add_argument("--data-root", default="Kvasir-SEG",
                        help="Kvasir-SEG root; the test split is rebuilt exactly as in train.py")
    # These must match the values used in train.py, or the test split will differ.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--images", help="Optional external images dir (overrides --data-root)")
    parser.add_argument("--masks", help="Optional external masks dir (overrides --data-root)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out-csv", default="stress_test_results.csv")
    parser.add_argument("--out-plot", default="stress_test_curve.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model(ckpt.get("encoder", "nvidia/mit-b0")).to(device)
    model.load_state_dict(ckpt["model"])
    size = int(ckpt.get("size", 384))
    threshold = float(ckpt.get("best_threshold", 0.5))
    print(f"Loaded {args.ckpt} (size: {size}, threshold: {threshold:.2f})")

    pairs, source = load_pairs(args)
    print(f"Stress-testing on {source}: {len(pairs)} images")

    def dice_for(corruption, severity):
        ds = CorruptedPolypDataset(pairs, corruption, severity, size=size, seed=args.seed)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        return evaluate(model, loader, device, thresholds=(threshold,))[threshold]["dice"]

    clean = dice_for(CORRUPTIONS[0], 0)  # severity 0 is identical for every corruption
    print(f"\nClean (severity 0): Dice = {clean:.4f}")

    results = {}
    for c in CORRUPTIONS:
        print(f"\n--- {c} ---")
        results[c] = [clean]
        for s in SEVERITIES[1:]:
            dice = dice_for(c, s)
            results[c].append(dice)
            print(f"  Severity {s}: Dice = {dice:.4f}")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["severity"] + CORRUPTIONS)
        for i, s in enumerate(SEVERITIES):
            writer.writerow([s] + [f"{results[c][i]:.4f}" for c in CORRUPTIONS])
    print(f"\nSaved results table to {args.out_csv}")

    plt.figure(figsize=(7, 4.5), dpi=150)
    styles = {"motion_blur": ("-o", "tab:blue", "Motion blur"),
              "low_light": ("-s", "tab:orange", "Low illumination"),
              "specular_glare": ("-^", "tab:red", "Specular glare")}
    for c in CORRUPTIONS:
        fmt, color, label = styles[c]
        plt.plot(SEVERITIES, results[c], fmt, color=color, linewidth=2, markersize=6, label=label)

    plt.xlabel("Corruption severity (0 = clean, 1-5 = increasing)", fontsize=11)
    plt.ylabel(f"Mean per-image Dice @ t={threshold:.2f}", fontsize=11)
    plt.title(f"SegFormer-b0 under camera corruptions ({source}, N={len(pairs)})", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0.0, 1.0)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(args.out_plot)
    print(f"Saved plot to {args.out_plot}")


if __name__ == "__main__":
    main()
