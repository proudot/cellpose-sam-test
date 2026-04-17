#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from cellpose import models
from scipy.ndimage import gaussian_filter
from skimage.color import label2rgb
from skimage.draw import ellipse
from skimage.measure import label as cc_label


def make_synthetic_cells(
    height: int = 512,
    width: int = 512,
    n_cells: int = 25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a simple fluorescence-like synthetic image with cell-shaped blobs
    and a ground-truth instance mask.
    """
    rng = np.random.default_rng(seed)

    image = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.int32)

    for idx in range(1, n_cells + 1):
        cy = rng.integers(40, height - 40)
        cx = rng.integers(40, width - 40)

        ry = int(rng.integers(12, 28))
        rx = int(rng.integers(12, 28))

        # random orientation by swapping shape roughness rather than explicit rotation
        if rng.random() < 0.5:
            ry = int(0.7 * ry)
        else:
            rx = int(0.7 * rx)

        rr, cc = ellipse(cy, cx, ry, rx, shape=image.shape)

        # avoid excessive overlap
        if np.any(mask[rr, cc] > 0):
            continue

        intensity = rng.uniform(0.5, 1.0)
        image[rr, cc] += intensity
        mask[rr, cc] = idx

        # add a brighter compact subregion, vaguely nucleus-like
        rr2, cc2 = ellipse(
            cy + int(rng.integers(-3, 4)),
            cx + int(rng.integers(-3, 4)),
            max(3, ry // 3),
            max(3, rx // 3),
            shape=image.shape,
        )
        image[rr2, cc2] += rng.uniform(0.2, 0.5)

    # blur to mimic optics
    image = gaussian_filter(image, sigma=2.0)

    # background + noise
    background = gaussian_filter(rng.normal(0.08, 0.02, size=image.shape).astype(np.float32), sigma=8.0)
    image = image + background

    # shot-noise-like effect
    image = np.clip(image, 0, None)
    image = rng.poisson(image * 60.0).astype(np.float32) / 60.0

    # read noise
    image += rng.normal(0, 0.03, size=image.shape).astype(np.float32)

    # normalize to 0..1
    image -= image.min()
    if image.max() > 0:
        image /= image.max()

    # relabel connected components in case some ids were skipped
    mask = cc_label(mask > 0)

    return image.astype(np.float32), mask.astype(np.int32)


def save_preview(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, out_png: Path) -> None:
    gt_rgb = label2rgb(gt_mask, image=image, bg_label=0)
    pred_rgb = label2rgb(pred_mask, image=image, bg_label=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("synthetic image")
    axes[1].imshow(gt_rgb)
    axes[1].set_title("ground truth")
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Cellpose-SAM prediction")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Cellpose-SAM on synthetic microscopy-like data.")
    parser.add_argument("--outdir", type=str, default="cellpose_sam_test_output")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--n-cells", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument("--diameter", type=float, default=None, help="Optional diameter override.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image, gt_mask = make_synthetic_cells(
        height=args.height,
        width=args.width,
        n_cells=args.n_cells,
        seed=args.seed,
    )

    tifffile.imwrite(outdir / "synthetic_cells.tif", image)
    tifffile.imwrite(outdir / "synthetic_gt_mask.tif", gt_mask.astype(np.uint16))

    gpu = bool(args.use_gpu and torch.cuda.is_available())
    print(f"Using GPU: {gpu}")

    model = models.CellposeModel(pretrained_model="cpsam", gpu=gpu)

    # single-channel grayscale input
    masks, flows, styles = model.eval(
        image,
        channels=[0, 0],
        diameter=args.diameter,
    )

    tifffile.imwrite(outdir / "cellpose_sam_mask.tif", masks.astype(np.uint16))
    save_preview(image, gt_mask, masks, outdir / "preview.png")

    gt_count = int(gt_mask.max())
    pred_count = int(masks.max())

    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"synthetic image shape: {image.shape}\n")
        f.write(f"ground-truth objects: {gt_count}\n")
        f.write(f"predicted objects: {pred_count}\n")
        f.write(f"used_gpu: {gpu}\n")
        f.write(f"diameter: {args.diameter}\n")

    print(f"Done. Results written to: {outdir}")
    print(f"Ground-truth objects: {gt_count}")
    print(f"Predicted objects:    {pred_count}")


if __name__ == "__main__":
    main()
