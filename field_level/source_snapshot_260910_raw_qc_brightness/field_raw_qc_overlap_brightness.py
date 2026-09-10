"""Batch mean brightness for registered, raster-overlap-extracted image pairs.

This is a QC/illumination metric, not the canonical digital-count endpoint.
Its fixed subtraction direction is pre - post.  Do not configure raw TIFF
directories here: inputs must already be warped/cropped to valid overlap.
"""
from __future__ import annotations

import csv
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from PIL import Image

# Set only after registration exports folders such as 260827_DNA_overlap,
# containing 12-1-0.tif and 12-1-1.tif.
AUTO_SCAN_ROOTS: list[Path] = []
MANUAL_CONFIG: list[dict] = []

FOLDER_NAME_PATTERN = re.compile(r"^(\d{6})_(.+)$")
FILE_NAME_PATTERN = re.compile(r"^(\d+)-(\d+)-(0|1)\.(?:tif|tiff|png|bmp)$", re.I)
IMAGE_EXTS = {".tif", ".tiff", ".png", ".bmp"}
OUTPUT_DIR = Path("raw_qc_overlap_brightness_output")


def discover(root: Path) -> list[dict]:
    """Discover complete pre/post pairs in YYYYMM_condition folders."""
    groups: list[dict] = []
    directories = [root] if FOLDER_NAME_PATTERN.match(root.name) else root.rglob("*")
    for directory in directories:
        match = FOLDER_NAME_PATTERN.match(directory.name) if directory.is_dir() else None
        if not match:
            continue
        files: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
        for path in directory.iterdir():
            name_match = FILE_NAME_PATTERN.match(path.name)
            if path.suffix.lower() in IMAGE_EXTS and name_match:
                sample, position, phase = name_match.groups()
                files[(sample, position)][phase] = path
        pairs = []
        for (sample, position), phases in sorted(files.items(), key=lambda item: tuple(map(int, item[0]))):
            if {"0", "1"} <= phases.keys():
                pairs.append({"sample": sample, "position": position,
                              "pre": str(phases["0"]), "post": str(phases["1"])})
            else:
                warnings.warn(f"Incomplete pre/post pair: {directory}, {sample}-{position}")
        if pairs:
            groups.append({"date": match.group(1), "condition": match.group(2), "pairs": pairs})
    return groups


def read_gray(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("F"), dtype=np.float64)


def compute(group: dict, pair: dict) -> dict:
    pre, post = read_gray(pair["pre"]), read_gray(pair["post"])
    if pre.shape != post.shape:
        raise ValueError(f"non-identical overlap shapes: {pre.shape} vs {post.shape}")
    if not (np.isfinite(pre).all() and np.isfinite(post).all()):
        raise ValueError("non-finite pixel values")
    pre_mean, post_mean = float(pre.mean()), float(post.mean())
    return {"date": group["date"], "condition": group["condition"],
            "sample": pair["sample"], "position": pair["position"],
            "pre_mean": pre_mean, "post_mean": post_mean,
            "diff_pre_minus_post": pre_mean - post_mean,
            "width": pre.shape[1], "height": pre.shape[0],
            "pre_path": pair["pre"], "post_path": pair["post"]}


def main() -> None:
    groups = [group for root in AUTO_SCAN_ROOTS if root.exists() for group in discover(root)]
    groups.extend(MANUAL_CONFIG)
    if not groups:
        sys.exit("No overlap-image groups configured. Do not substitute raw TIFFs.")
    rows = []
    for group in groups:
        for pair in group["pairs"]:
            try:
                rows.append(compute(group, pair))
            except Exception as exc:
                warnings.warn(f"{group['date']}_{group['condition']} {pair['sample']}-{pair['position']}: {exc}")
    if not rows:
        sys.exit("No valid overlap pairs.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "per_pair_raw_qc_brightness_diff.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["date"], row["condition"])].append(row["diff_pre_minus_post"])
    with (OUTPUT_DIR / "per_day_raw_qc_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "condition", "n_pairs", "diff_mean_pre_minus_post", "diff_std"])
        writer.writeheader()
        for (date, condition), values in sorted(grouped.items()):
            writer.writerow({"date": date, "condition": condition, "n_pairs": len(values),
                             "diff_mean_pre_minus_post": mean(values),
                             "diff_std": stdev(values) if len(values) > 1 else 0.0})
    print(len(rows))


if __name__ == "__main__":
    main()
