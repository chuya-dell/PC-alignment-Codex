"""Per-FOV image-quality checks that are not specific to pillar- or field-level analysis:
the bright write-field-boundary band mask (docs/POSITION6_IMAGE_FORENSICS_20260916.md) and
the saturation-based acquisition QC flag it motivated.

Both operate on a single raw 16-bit image and know nothing about pillar coordinates or
registration; they are meant to be reused from acquisition-time QC, pillar_level, and
field_level code alike.
"""
from __future__ import annotations

import numpy as np
import cv2


def bright_band_mask(raw: np.ndarray, dn_threshold: int = 55000,
                      close_radius_px: int = 8, min_component_span_px: int = 200) -> np.ndarray:
    """Boolean mask, True where a pixel belongs to an elongated over-bright band.

    Thresholds the raw image at dn_threshold, closes small gaps so a line's pixels merge
    into one component, then keeps only components whose bounding box's longer side is at
    least min_component_span_px. This discriminates the write-field-boundary lines found at
    Position 6/7 (which span most of the frame) from isolated bright pillars or noise, and
    needs no per-position hand-tuned coordinates: it was verified against all of Positions
    1-8 on 260826-p50-sam and 260828-p50-SAM (see docs/POSITION6_IMAGE_FORENSICS_20260916.md
    and docs/MASKED_ALIGNMENT_FINAL_DECISION_20260916.md).
    """
    arr = np.asarray(raw)
    binary = (arr >= dn_threshold).astype(np.uint8)
    if not binary.any():
        return np.zeros(arr.shape, bool)
    k = 2 * close_radius_px + 1
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    mask = np.zeros(arr.shape, bool)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if max(w, h) >= min_component_span_px:
            mask |= (labels == label)
    if mask.any():
        dilate_k = 2 * close_radius_px + 1
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((dilate_k, dilate_k), np.uint8)) > 0
    return mask


def saturation_qc(raw: np.ndarray, bright_dn_threshold: int = 55000, bright_flag_count: int = 18000,
                   saturated_dn: int = 65000, saturated_flag_count: int = 1500) -> dict:
    """Acquisition-time QC counters and flag for the write-field-boundary defect.

    Thresholds verified against all 8 positions, pre and post, on both 260826-p50-sam and
    260828-p50-SAM Sample 1 (see data/results/v7_masked_alignment_benchmark/
    bright_band_mask_qc_all_positions.csv). Across those 32 images the highest count among
    Positions 1-5/8 (never Position 6/7) was 16252 px >=55000 DN and 1412 saturated
    (260828 Position 3 post, itself sitting near an ordinary write-field seam); the lowest
    count among Position 6/7 images was 20544 px >=55000 DN and 1445 saturated (260828
    Position 7 pre). bright_flag_count=18000 sits cleanly between 16252 and 20544 and alone
    separates every tested image correctly; saturated_flag_count=1500 is kept as a secondary
    signal (OR'd with the bright-pixel count) even though its own margin at that boundary is
    narrow. A FOV is flagged if either counter crosses its threshold.
    """
    arr = np.asarray(raw)
    bright_count = int(np.sum(arr >= bright_dn_threshold))
    saturated_count = int(np.sum(arr >= saturated_dn))
    flagged = bright_count >= bright_flag_count or saturated_count >= saturated_flag_count
    return {
        "bright_pixel_count": bright_count, "bright_dn_threshold": bright_dn_threshold,
        "bright_flag_count": bright_flag_count,
        "saturated_pixel_count": saturated_count, "saturated_dn_threshold": saturated_dn,
        "saturated_flag_count": saturated_flag_count,
        "flagged": bool(flagged),
    }
