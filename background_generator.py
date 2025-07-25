import os
import math
import logging
from typing import Tuple

import numpy as np
from PIL import Image


from pathlib import Path
import random


def _load_yolo_labels(label_path: Path, img_w: int, img_h: int):
    """Return list of bounding boxes in pixel coords from YOLO txt file."""
    boxes = []
    if not label_path.exists():
        return boxes  # image without annotations – treat as empty
    try:
        with label_path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, xc, yc, w, h = map(float, parts)
                x1 = (xc - w / 2) * img_w
                y1 = (yc - h / 2) * img_h
                x2 = (xc + w / 2) * img_w
                y2 = (yc + h / 2) * img_h
                boxes.append((x1, y1, x2, y2))
    except Exception as exc:
        logging.warning(f"Failed reading labels {label_path}: {exc}")
    return boxes


def _boxes_intersect(rect1, rect2):
    """Axis-aligned rectangle intersection check."""
    x11, y11, x12, y12 = rect1
    x21, y21, x22, y22 = rect2
    return not (x12 <= x21 or x22 <= x11 or y12 <= y21 or y22 <= y11)


def _find_background_rect(boxes, img_w, img_h, patch_size=640, stride=64):
    """Grid-search the largest empty rectangle (patch_size×patch_size) not overlapping *boxes*.

    Returns (x1, y1, x2, y2) in pixel coords or None if not found."""
    if img_w < patch_size or img_h < patch_size:
        return None

    candidates = []
    for y in range(0, img_h - patch_size + 1, stride):
        for x in range(0, img_w - patch_size + 1, stride):
            candidate = (x, y, x + patch_size, y + patch_size)
            if all(not _boxes_intersect(candidate, b) for b in boxes):
                candidates.append(candidate)
    if not candidates:
        return None
    # Choose the candidate whose centre is farthest from any object (optional heuristic)
    def _min_dist_to_boxes(rect):
        cx = (rect[0] + rect[2]) / 2
        cy = (rect[1] + rect[3]) / 2
        dists = [abs(cx - (b[0] + b[2]) / 2) + abs(cy - (b[1] + b[3]) / 2) for b in boxes] or [0]
        return min(dists)

    best = max(candidates, key=_min_dist_to_boxes)
    return best


def extract_backgrounds_for_image(image_path: Path, labels_dir: Path, output_dir: Path, patch_size=640, stride=64):
    """Extract one background patch from *image_path* based on its YOLO labels."""
    try:
        img = Image.open(image_path)
    except Exception as exc:
        logging.warning(f"Skipping {image_path}: {exc}")
        return

    img_w, img_h = img.size
    label_path = labels_dir / (image_path.stem + ".txt")
    boxes = _load_yolo_labels(label_path, img_w, img_h)

    rect = _find_background_rect(boxes, img_w, img_h, patch_size, stride)
    if rect is None:
        logging.info(f"No background patch found in {image_path.name}")
        return

    x1, y1, x2, y2 = map(int, rect)
    patch = img.crop((x1, y1, x2, y2))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_bg.png"
    try:
        patch.save(out_path)
        logging.info(f"Saved background patch: {out_path.name}")
    except Exception as exc:
        logging.error(f"Failed saving {out_path}: {exc}")


# ------------------------
# CLI entry point
# ------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate background patches using YOLO labels.")
    parser.add_argument("--images", required=True, help="Path to directory with images")
    parser.add_argument("--labels", required=True, help="Path to directory with YOLO txt labels")
    parser.add_argument("--out", required=True, help="Output directory for background patches")
    parser.add_argument("--patch", type=int, default=640, help="Patch size (default: 640)")
    parser.add_argument("--stride", type=int, default=64, help="Grid stride in pixels (default: 64)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.out)

    image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    for img_path in image_files:
        extract_backgrounds_for_image(img_path, labels_dir, output_dir, patch_size=args.patch, stride=args.stride)
