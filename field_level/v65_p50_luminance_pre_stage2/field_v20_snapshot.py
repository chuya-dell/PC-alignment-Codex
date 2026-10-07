"""Stage 1 only: validate physical-scale masks on 260922 50x image pairs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.registration import (  # noqa: E402
    align_pair_exclusion_masks,
    load_image_unicode,
    register_image_pair_affine,
)


RECORDED_DEFECTS = {
    ("1", 5): "scratch",
    ("1", 6): "scratch",
    ("3", 2): "stain_near_cross",
    ("5", 5): "stain",
    ("5", 6): "stain",
    ("3", 3): "many_dirt",
}
CATEGORIES = ("marker", "stain", "write_boundary", "pair_change")


def load_params(path: Path) -> dict:
    params = json.loads(path.read_text(encoding="utf-8"))
    px = float(params["pixel_size_um"])
    if not np.isfinite(px) or px <= 0:
        raise ValueError("pixel_size_um must be finite and positive")
    return params


def px(um: float, pixel_size_um: float, minimum: int = 1) -> int:
    return max(minimum, int(round(float(um) / pixel_size_um)))


def robust_unit(image: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    span = max(float(hi - lo), 1.0)
    return np.clip((a - lo) / span, 0, 1), span


def line_ridge_mask(image: np.ndarray, pixel_size_um: float, params: dict,
                    polarity: str, eligible: bool = True) -> np.ndarray:
    if not eligible:
        return np.zeros(image.shape, bool)
    unit, _ = robust_unit(image)
    ridge_image = unit if polarity == "bright" else 1.0 - unit
    scales_px = [float(s) / pixel_size_um for s in params["ridge_sigmas_um"]]
    response = frangi(ridge_image, sigmas=scales_px, alpha=0.5,
                      beta=0.5, gamma=0.08, black_ridges=False)
    positive = response[response > 0]
    if positive.size == 0:
        return np.zeros(image.shape, bool)
    threshold = max(float(np.quantile(positive, params["ridge_quantile"])),
                    float(params["ridge_min_response"]))
    binary = (response >= threshold).astype(np.uint8)
    close_r = px(params["ridge_gap_close_radius_um"], pixel_size_um)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    accepted = np.zeros(image.shape, np.uint8)
    min_length = float(params["ridge_min_length_um"])
    max_width = float(params["ridge_max_width_um"])
    min_aspect = float(params["ridge_min_aspect"])
    for label in range(1, count):
        x, y, w, h, _area = stats[label]
        long_side, short_side = max(w, h), max(1, min(w, h))
        aspect = long_side / short_side
        if (long_side * pixel_size_um >= min_length and
                short_side * pixel_size_um <= max_width and aspect >= min_aspect):
            accepted[labels == label] = 1
    dilate_r = px(params["category_dilation_um"], pixel_size_um, minimum=0)
    if dilate_r:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_r + 1, 2 * dilate_r + 1))
        accepted = cv2.dilate(accepted, k)
    return accepted.astype(bool)


def profile_line_mask(image: np.ndarray, pixel_size_um: float, params: dict,
                      polarity: str) -> np.ndarray:
    """Find long, possibly interrupted fiducial lines from robust segment profiles."""
    values = image.astype(np.float32)
    detail_sigma = float(params["profile_detail_sigma_um"]) / pixel_size_um
    smooth = cv2.GaussianBlur(values, (0, 0), detail_sigma)
    _, dynamic = robust_unit(image)
    result = np.zeros(image.shape, np.uint8)
    segment_px = px(params["profile_segment_um"], pixel_size_um)
    for vertical in (True, False):
        scan_length = image.shape[0] if vertical else image.shape[1]
        profile_length = image.shape[1] if vertical else image.shape[0]
        starts = list(range(0, scan_length, segment_px))
        support = np.zeros(profile_length, np.uint16)
        for start in starts:
            stop = min(scan_length, start + segment_px)
            if stop - start < 0.4 * segment_px:
                continue
            profile = (np.mean(smooth[start:stop, :], axis=0) if vertical
                       else np.mean(smooth[:, start:stop], axis=1))
            trend = ndi.gaussian_filter1d(profile, max(1.0, float(params["profile_trend_um"]) / pixel_size_um),
                                          mode="nearest")
            residual = profile - trend
            center = float(np.median(residual))
            mad = float(np.median(np.abs(residual - center))) * 1.4826
            threshold = max(float(params["profile_mad_multiplier"]) * mad,
                            float(params["profile_dynamic_fraction"]) * dynamic)
            candidate = (residual < center - threshold) if polarity == "dark" else (residual > center + threshold)
            close_px = px(params["profile_gap_close_um"], pixel_size_um)
            candidate = cv2.morphologyEx(candidate.astype(np.uint8)[None, :], cv2.MORPH_CLOSE,
                                         np.ones((1, 2 * close_px + 1), np.uint8))[0].astype(bool)
            support += candidate.astype(np.uint16)
        required = max(int(params["profile_min_segments"]),
                       int(np.ceil(len(starts) * float(params["profile_support_fraction"]))))
        coordinates = np.flatnonzero(support >= required)
        if not len(coordinates):
            continue
        max_gap = px(params["profile_track_gap_um"], pixel_size_um)
        groups = np.split(coordinates, np.flatnonzero(np.diff(coordinates) > max_gap) + 1)
        groups = sorted(groups, key=lambda group: int(np.sum(support[group])), reverse=True)
        for group in groups[:int(params["profile_max_tracks_per_axis"])]:
            center_px = int(group[np.argmax(support[group])])
            estimated_half = int(np.ceil((group[-1] - group[0]) / 2.0))
            half_width = max(px(params["profile_min_half_width_um"], pixel_size_um),
                             estimated_half + px(params["profile_margin_um"], pixel_size_um))
            half_width = min(half_width, px(params["profile_max_half_width_um"], pixel_size_um))
            if vertical:
                result[:, max(0, center_px-half_width):min(profile_length, center_px+half_width+1)] = 1
            else:
                result[max(0, center_px-half_width):min(profile_length, center_px+half_width+1), :] = 1
    return result.astype(bool)


def blob_outlier_mask(image: np.ndarray, pixel_size_um: float, params: dict,
                      protect: np.ndarray | None = None) -> np.ndarray:
    unit, span = robust_unit(image)
    small_sigma = float(params["stain_detail_sigma_um"]) / pixel_size_um
    broad_sigma = float(params["stain_background_sigma_um"]) / pixel_size_um
    values = image.astype(np.float32)
    detail = cv2.GaussianBlur(values, (0, 0), small_sigma)
    background = cv2.GaussianBlur(values, (0, 0), broad_sigma)
    residual = detail - background
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center))) * 1.4826
    cutoff = max(float(params["stain_mad_multiplier"]) * mad,
                 float(params["stain_dynamic_fraction"]) * span, 1.0)
    binary = (np.abs(residual - center) >= cutoff).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    accepted = np.zeros(image.shape, np.uint8)
    min_area = float(params["stain_min_area_um2"]) / (pixel_size_um ** 2)
    max_extent = float(params["stain_max_extent_um"]) / pixel_size_um
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area or max(w, h) * pixel_size_um > max_extent:
            continue
        if max(w, h) / max(1, min(w, h)) > float(params["stain_max_aspect"]):
            continue
        accepted[labels == label] = 1
    dilate_r = px(params["category_dilation_um"], pixel_size_um, minimum=0)
    if dilate_r:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_r + 1, 2 * dilate_r + 1))
        accepted = cv2.dilate(accepted, k)
    result = accepted.astype(bool)
    if protect is not None:
        result &= ~protect
    return result


def native_masks(image: np.ndarray, position: int, pixel_size_um: float, params: dict) -> dict[str, np.ndarray]:
    marker = (line_ridge_mask(image, pixel_size_um, params, "dark") |
              profile_line_mask(image, pixel_size_um, params, "dark"))
    boundary = (line_ridge_mask(image, pixel_size_um, params, "bright", eligible=position in (6, 7)) |
                (profile_line_mask(image, pixel_size_um, params, "bright") if position in (6, 7)
                 else np.zeros(image.shape, bool)))
    stain = blob_outlier_mask(image, pixel_size_um, params, protect=marker | boundary)
    return {"marker": marker, "stain": stain, "write_boundary": boundary}


def differential_stain(pre: np.ndarray, post_in_pre: np.ndarray, pixel_size_um: float,
                       params: dict, protect: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pre_f, post_f = pre.astype(np.float32), post_in_pre.astype(np.float32)
    sigma = float(params["pair_change_detail_sigma_um"]) / pixel_size_um
    pre_detail = pre_f - cv2.GaussianBlur(pre_f, (0, 0), sigma)
    post_detail = post_f - cv2.GaussianBlur(post_f, (0, 0), sigma)
    delta = pre_detail - post_detail
    usable = np.asarray(valid, dtype=bool)
    center = float(np.median(delta[usable]))
    mad = float(np.median(np.abs(delta[usable] - center))) * 1.4826
    scale = max(float(np.percentile(pre_f, 99.5) - np.percentile(pre_f, 0.5)), 1.0)
    cutoff = max(float(params["pair_change_mad_multiplier"]) * mad,
                 float(params["pair_change_dynamic_fraction"]) * scale, 1.0)
    binary = ((np.abs(delta - center) >= cutoff) & usable).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    accepted = np.zeros(pre.shape, np.uint8)
    min_area = float(params["stain_min_area_um2"]) / (pixel_size_um ** 2)
    max_extent = float(params["stain_max_extent_um"]) / pixel_size_um
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area or max(w, h) * pixel_size_um > max_extent:
            continue
        if max(w, h) / max(1, min(w, h)) > float(params["stain_max_aspect"]):
            continue
        accepted[labels == label] = 1
    result = accepted.astype(bool) & ~protect
    dilate_r = px(params["category_dilation_um"], pixel_size_um, minimum=0)
    if dilate_r:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_r + 1, 2 * dilate_r + 1))
        result = cv2.dilate(result.astype(np.uint8), k) > 0
    return result


def load_pair(data_dir: Path, sample: str, pos: int) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    stem = f"50-{sample}-{pos}"
    p0, p1 = data_dir / f"{stem}-0.tif", data_dir / f"{stem}-1.tif"
    if not p0.exists() or not p1.exists():
        raise FileNotFoundError(f"Missing pair: {p0.name} or {p1.name}")
    a, b = load_image_unicode(str(p0)), load_image_unicode(str(p1))
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"Unexpected image shape for {stem}: {a.shape}, {b.shape}")
    return a, b, p0, p1


def process_pair(data_dir: Path, sample: str, pos: int, pixel_size_um: float,
                 params: dict, save_dir: Path | None = None,
                 screen_scale: float = 1.0) -> tuple[dict, dict | None]:
    pre, post, p0, p1 = load_pair(data_dir, sample, pos)
    if not 0 < screen_scale <= 1:
        raise ValueError("screen_scale must be in (0, 1]")
    if screen_scale < 1:
        pre = cv2.resize(pre, None, fx=screen_scale, fy=screen_scale, interpolation=cv2.INTER_AREA)
        post = cv2.resize(post, None, fx=screen_scale, fy=screen_scale, interpolation=cv2.INTER_AREA)
        pixel_size_um /= screen_scale
    native0 = native_masks(pre, pos, pixel_size_um, params)
    native1 = native_masks(post, pos, pixel_size_um, params)
    # Register the original, unmasked images. The dark marker lines remain visible to ORB.
    warp, qc = register_image_pair_affine(pre, post, exclude_mask=None,
                                           mask_stains=False, return_qc=True)
    h, w = pre.shape
    post_in_pre = cv2.warpAffine(post, warp, (w, h),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid_post = cv2.warpAffine(np.ones(post.shape, dtype=np.uint8), warp, (w, h),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    common: dict[str, np.ndarray] = {}
    for category in ("marker", "stain", "write_boundary"):
        common[category], _ = align_pair_exclusion_masks(native0[category], native1[category], warp)
    protected = common["marker"] | common["write_boundary"]
    edge_r = px(params["overlap_edge_exclusion_um"], pixel_size_um, minimum=0)
    if edge_r:
        edge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_r + 1, 2 * edge_r + 1))
        valid_post = cv2.erode(valid_post.astype(np.uint8), edge_k) > 0
    common["pair_change"] = differential_stain(pre, post_in_pre, pixel_size_um, params,
                                                protected, valid_post)
    pre_union = np.logical_or.reduce([common[k] for k in CATEGORIES])
    radius = px(params["final_dilation_um"], pixel_size_um, minimum=0)
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        pre_union = cv2.dilate(pre_union.astype(np.uint8), kernel) > 0
    common_post = cv2.warpAffine(pre_union.astype(np.uint8), warp, (w, h),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0) > 0
    common_post |= np.logical_or.reduce([native1[k] for k in ("marker", "stain", "write_boundary")])
    row = {
        "sample": sample, "position": pos, "known_record": RECORDED_DEFECTS.get((sample, pos), "not_recorded"),
        "pre_file": p0.name, "post_file": p1.name, "width_px": w, "height_px": h,
        "registration_dx_center_px": qc["dx_center_px"], "registration_dy_center_px": qc["dy_center_px"],
        "registration_rotation_deg": qc["rotation_deg"], "registration_scale": qc["scale"],
        "registration_accepted": qc["accepted"],
        "common_union_fraction_before_final_dilation": float(np.logical_or.reduce([common[k] for k in CATEGORIES]).mean()),
        "common_union_fraction_after_final_dilation": float(pre_union.mean()),
        "common_union_percent_after_final_dilation": float(pre_union.mean() * 100),
        "common_coordinates": "pre frame; post masks mapped with pre-to-post affine before union",
        "registration_images_masked": False,
    }
    for category in CATEGORIES:
        row[f"{category}_pre_native_percent"] = float(native0.get(category, common[category]).mean() * 100)
        row[f"{category}_post_native_percent"] = float(native1.get(category, common[category]).mean() * 100)
        row[f"{category}_common_percent"] = float(common[category].mean() * 100)
    products = {"pre": pre, "post_in_pre": post_in_pre, "pre_union": pre_union,
                "post_union": common_post, "common": common, "native0": native0,
                "native1": native1, "warp": warp, "qc": qc}
    if save_dir is not None:
        save_pair_products(save_dir, sample, pos, products, params)
    return row, products if save_dir is not None else None


def make_overlay(image: np.ndarray, layers: dict[str, np.ndarray], title: str) -> np.ndarray:
    unit, _ = robust_unit(image)
    gray = (unit * 255).astype(np.uint8)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colors = {"marker": (30, 40, 255), "stain": (0, 230, 255),
              "write_boundary": (255, 180, 0), "pair_change": (220, 40, 220)}
    for name, mask in layers.items():
        if name in colors and mask is not None:
            color = np.asarray(colors[name], dtype=np.float32)
            rgb[mask] = (0.35 * rgb[mask] + 0.65 * color).astype(np.uint8)
    banner = np.full((34, rgb.shape[1], 3), 25, np.uint8)
    cv2.putText(banner, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, rgb])


def save_pair_products(out: Path, sample: str, pos: int, products: dict, params: dict) -> None:
    stem = f"50-{sample}-{pos}"
    out.mkdir(parents=True, exist_ok=True)
    for phase, native in (("0", products["native0"]), ("1", products["native1"])):
        for category, mask in native.items():
            cv2.imwrite(str(out / f"mask_{stem}-{phase}_{category}.png"), mask.astype(np.uint8) * 255)
        marker_view = make_overlay(products["pre"] if phase == "0" else products["post_in_pre"],
                                   {"marker": native["marker"]},
                                   f"{stem}-{phase} marker ridge | red=marker mask")
        cv2.imwrite(str(out / f"marker_review_{stem}-{phase}.png"), marker_view)
    layers = {k: products["common"][k] for k in CATEGORIES}
    pre_overlay = make_overlay(products["pre"], layers, f"{stem} common-pre masks | red marker, yellow stain, blue boundary, magenta pair-change")
    post_overlay = make_overlay(products["post_in_pre"], layers, f"{stem} aligned post in pre coordinates | same common masks")
    cv2.imwrite(str(out / f"overlay_{stem}-0.png"), pre_overlay)
    cv2.imwrite(str(out / f"overlay_{stem}-1_aligned_to_0.png"), post_overlay)
    cv2.imwrite(str(out / f"mask_{stem}_common_union.png"), products["pre_union"].astype(np.uint8) * 255)


def artificial_spot_validation(image: np.ndarray, base_mask: np.ndarray, pixel_size_um: float,
                               params: dict, output: Path, seed: int = 260922) -> list[dict]:
    rng = np.random.default_rng(seed)
    unit, _ = robust_unit(image)
    base = unit.astype(np.float32)
    h, w = base.shape
    truth_rows: list[dict] = []
    spots = params["artificial_spots"]
    max_radius_px = max(px(float(s["radius_um"]), pixel_size_um) for s in spots)
    safe = (~base_mask).astype(np.uint8)
    safe[:max_radius_px * 2, :] = 0
    safe[-max_radius_px * 2:, :] = 0
    safe[:, :max_radius_px * 2] = 0
    safe[:, -max_radius_px * 2:] = 0
    ys, xs = np.where(safe > 0)
    if len(xs) < 10:
        raise RuntimeError("Not enough unmasked area for artificial spots")
    locations = []
    for spot in spots:
        for repeat in range(int(params["artificial_repeats_per_polarity"])):
            locations.append((spot, "dark", repeat))
            locations.append((spot, "bright", repeat))
    chosen: list[tuple[int, int, int]] = []
    exclusion = cv2.dilate(base_mask.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    min_sep = 2 * max_radius_px + 10
    attempts = 0
    while len(chosen) < len(locations) and attempts < len(locations) * 200:
        attempts += 1
        idx = int(rng.integers(len(xs)))
        x, y = int(xs[idx]), int(ys[idx])
        if exclusion[y, x] or any((x - xx) ** 2 + (y - yy) ** 2 < min_sep ** 2 for xx, yy, _ in chosen):
            continue
        chosen.append((x, y, idx))
    if len(chosen) != len(locations):
        raise RuntimeError("Could not place all non-overlapping artificial spots")
    yy, xx = np.mgrid[:h, :w]
    contrast = float(params["artificial_contrast_fraction"])
    baseline_mask = blob_outlier_mask((base * 65535).astype(np.uint16), pixel_size_um, params,
                                      protect=base_mask)
    injected = base.copy()
    truths = []
    for (spot, polarity, repeat), (cx, cy, _idx) in zip(locations, chosen):
        radius_um = float(spot["radius_um"])
        radius = px(radius_um, pixel_size_um)
        sigma = max(radius / 2.0, 1.0)
        amplitude = contrast if polarity == "bright" else -contrast
        injected += amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        truth = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        truths.append((spot, polarity, repeat, cx, cy, radius, radius_um, truth))
    injected_img = (np.clip(injected, 0, 1) * 65535).astype(np.uint16)
    detected = blob_outlier_mask(injected_img, pixel_size_um, params, protect=base_mask)
    incremental = detected & ~baseline_mask
    all_truth = np.logical_or.reduce([item[-1] for item in truths])
    fp_global = float((incremental & ~all_truth & ~base_mask).sum() /
                      max(1, (~all_truth & ~base_mask).sum()))
    for spot, polarity, repeat, cx, cy, radius, radius_um, truth in truths:
        hit = bool(np.any(incremental & truth))
        detected_fraction = float((incremental & truth).sum() / max(1, truth.sum()))
        local_radius = max(4 * radius, px(2.0, pixel_size_um))
        local_region = (xx - cx) ** 2 + (yy - cy) ** 2 <= local_radius ** 2
        local_negative = local_region & ~all_truth & ~base_mask
        false_positive_rate = float((incremental & local_negative).sum() / max(1, local_negative.sum()))
        truth_rows.append({"polarity": polarity, "radius_um": radius_um, "radius_px_provisional": radius,
                           "repeat": repeat + 1, "center_x_px": cx, "center_y_px": cy,
                           "detected": hit, "truth_pixel_recall_incremental": detected_fraction,
                           "local_incremental_false_positive_pixel_rate": false_positive_rate,
                           "global_incremental_false_positive_pixel_rate": fp_global,
                           "contrast_fraction_of_robust_dynamic_range": contrast})
    return truth_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-only", action="store_true",
                        help="Reuse selected_validation_views.csv and regenerate native-resolution detail outputs only")
    args = parser.parse_args()
    params = load_params(args.params)
    pixel_size_um = float(params["pixel_size_um"])
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = {"input_directory": str(args.data_dir), "parameter_file": str(args.params),
           "pixel_size_um": pixel_size_um, "pixel_size_status": params["pixel_size_status"],
           "stage": 1, "brightness_analysis_performed": False}
    (args.output / "run_metadata.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    pair_rows = []
    failures = []
    if args.selected_only:
        selection = json.loads("[]")
        with (args.output / "selected_validation_views.csv").open(encoding="utf-8-sig", newline="") as f:
            selection = [{**r, "position": int(r["position"]),
                          "screening_union_percent": float(r["screening_union_percent"])}
                         for r in csv.DictReader(f)]
        selected = [(r["sample"], r["position"]) for r in selection]
        pair_rows = []
    else:
        candidates = [(s, p) for s in ("01", "1", "2", "3", "4", "5", "6", "7", "8") for p in range(1, 9)]
        for i, (sample, pos) in enumerate(candidates, start=1):
            try:
                row, _ = process_pair(args.data_dir, sample, pos, pixel_size_um, params,
                                      screen_scale=float(params["screening_scale"]))
                pair_rows.append(row)
                print(f"screen {i}/{len(candidates)}: 50-{sample}-{pos}, mask={row['common_union_percent_after_final_dilation']:.4f}%", flush=True)
            except Exception as exc:
                failures.append({"sample": sample, "position": pos, "error": f"{type(exc).__name__}: {exc}"})
                print(f"screen {i}/{len(candidates)}: 50-{sample}-{pos}, FAILED {type(exc).__name__}: {exc}", flush=True)
        write_csv(args.output / "all_view_screening.csv", pair_rows)
        write_csv(args.output / "screening_failures.csv", failures)
        known = set(RECORDED_DEFECTS)
        controls = [r for r in pair_rows if (r["sample"], r["position"]) not in known]
        controls.sort(key=lambda r: r["common_union_fraction_after_final_dilation"])
        # Refine the ten lowest quarter-resolution control candidates at full
        # resolution before selecting the three controls to report.
        refined_controls = []
        for i, coarse in enumerate(controls[:10], start=1):
            row, _ = process_pair(args.data_dir, coarse["sample"], int(coarse["position"]),
                                  pixel_size_um, params)
            row["coarse_screening_fraction"] = coarse["common_union_fraction_after_final_dilation"]
            refined_controls.append(row)
            print(f"control refinement {i}/10: 50-{row['sample']}-{row['position']}, "
                  f"native mask={row['common_union_percent_after_final_dilation']:.4f}%", flush=True)
        refined_controls.sort(key=lambda r: r["common_union_fraction_after_final_dilation"])
        write_csv(args.output / "control_ranking_native_resolution.csv", refined_controls)
        selected_controls = [(r["sample"], int(r["position"])) for r in refined_controls[:3]]
        selected = sorted(list(RECORDED_DEFECTS) + selected_controls,
                          key=lambda item: (item[0], item[1]))
        selection = [{"sample": s, "position": p, "known_record": RECORDED_DEFECTS.get((s, p), "low_mask_control"),
                      "screening_union_percent": next((r["common_union_percent_after_final_dilation"] for r in pair_rows
                                                        if r["sample"] == s and r["position"] == p), None),
                      "native_resolution_union_percent": next((r["common_union_percent_after_final_dilation"] for r in refined_controls
                                                               if r["sample"] == s and r["position"] == p), None)}
                     for s, p in selected]
        write_csv(args.output / "selected_validation_views.csv", selection)
    selected_controls = [(r["sample"], int(r["position"])) for r in selection
                         if r.get("known_record") == "low_mask_control"]
    detailed = []
    artificial = []
    for sample, pos in selected:
        row, products = process_pair(args.data_dir, sample, pos, pixel_size_um, params, args.output / "images")
        detailed.append(row)
        if (sample, pos) in selected_controls and sample == selected_controls[0][0] and pos == selected_controls[0][1]:
            artificial.extend(artificial_spot_validation(products["pre"], products["pre_union"],
                                                          pixel_size_um, params, args.output))
    write_csv(args.output / "mask_breakdown_selected_views.csv", detailed)
    write_csv(args.output / "artificial_spot_detection.csv", artificial)
    counts = {}
    for polarity in ("dark", "bright"):
        rows = [r for r in artificial if r["polarity"] == polarity]
        counts[polarity] = {"spots": len(rows), "detected": sum(bool(r["detected"]) for r in rows),
                            "detection_rate": float(np.mean([r["detected"] for r in rows])) if rows else None,
                            "mean_local_false_positive_pixel_rate": float(np.mean([r["local_incremental_false_positive_pixel_rate"] for r in rows])) if rows else None,
                            "global_false_positive_pixel_rate": float(np.mean([r["global_incremental_false_positive_pixel_rate"] for r in rows])) if rows else None}
    summary = {"selected_views": selection,
               "screened_pairs": (len(pair_rows) if not args.selected_only else "reused previous 72-pair screen"),
               "screening_failures": failures,
               "artificial_spot_summary": counts,
               "provisional_pixel_size_um": pixel_size_um,
               "note": "Stage 1 only; no whole-field brightness changes or Stage 2 metrics computed."}
    (args.output / "stage1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
