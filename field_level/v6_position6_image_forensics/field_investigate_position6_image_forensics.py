"""Direct image-level forensics for 260826 p50 SAM Sample 1 Position 6, comparing it
against Positions 1, 2, 5, 8 (the same FOV set used in the v5 semi-synthetic alignment
benchmark and its real-range/axis-sweep follow-ups).

Registration-method comparisons already ruled out that either candidate alignment
algorithm is uniformly at fault: ECC breaks on large real-scale translation and
ORB/RANSAC breaks on real-scale rotation, both specifically at Position 6. This script
does not touch registration; it inspects the raw pre/post image content itself:
brightness/contrast, focus sharpness, lattice regularity, and automatic defect-candidate
regions (reusing shared.defect_masking, which is designed exactly for this and already
produces human-review candidates, not accepted defects).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import analyzer
from shared import lattice_indexing as li
from shared import defect_masking as dm
from shared.registration import load_image_unicode

POSITIONS = [1, 2, 5, 6, 8]
SAMPLE = 1
ROOT_DIR = Path('G:/マイドライブ/1.実験データ_gdrive/4.生データ/4.生データ D/260826-p50-sam')
PITCH_PX = 7.286


def image_for_ecc(image: np.ndarray) -> np.ndarray:
    return (np.asarray(image, np.float32) / 65535.0).astype(np.float32)


def focus_metrics(image: np.ndarray) -> dict:
    x = np.asarray(image, np.float32) / 65535.0
    laplacian_var = float(cv2.Laplacian(x, cv2.CV_32F).var())
    gx, gy = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return {'laplacian_variance': laplacian_var, 'tenengrad_mean': float(np.mean(gx * gx + gy * gy))}


def brightness_metrics(raw: np.ndarray) -> dict:
    x = np.asarray(raw, np.float64)
    p1, p50, p99 = np.percentile(x, [1, 50, 99])
    return {'mean_dn': float(x.mean()), 'std_dn': float(x.std()),
            'p1_dn': float(p1), 'median_dn': float(p50), 'p99_dn': float(p99),
            'min_dn': float(x.min()), 'max_dn': float(x.max())}


def lattice_regularity(path: str, image01: np.ndarray, pitch: float = PITCH_PX) -> tuple[dict, pd.DataFrame]:
    """Sub-pixel peak detection vs. the FFT-fit theoretical lattice: detection/missing
    rate per 8x8 tile (local dropouts/disorder) and nearest-neighbour distance spread
    among accepted peaks (irregularity of the physical pillar positions)."""
    detections = analyzer.analyze_image(path, method='peak', min_dist=3, threshold=.2)
    points = detections[['x', 'y']].to_numpy(float)
    lattice = li.lattice_from_fft(image01, pitch)
    grid_indices, grid_xy = li.grid_coordinates(lattice, image01.shape[1], image01.shape[0], margin=30)
    assigned = li.assign_lattice_indices(points, lattice, max_distance=1.5)
    accepted = assigned['within_distance']
    selected = assigned['selected']
    selected_cells = {tuple(v) for v in assigned['indices'][selected]}
    grid_cells = {tuple(v) for v in grid_indices}
    h, w = image01.shape
    rows = []
    for row in range(8):
        for col in range(8):
            in_tile = ((grid_xy[:, 0] >= col * w / 8) & (grid_xy[:, 0] < (col + 1) * w / 8) &
                       (grid_xy[:, 1] >= row * h / 8) & (grid_xy[:, 1] < (row + 1) * h / 8))
            cells = {tuple(v) for v in grid_indices[in_tile]}
            found = len(cells & selected_cells)
            rows.append({'tile_row': row, 'tile_col': col, 'theoretical_cells': len(cells),
                         'unique_detected_cells': found, 'detection_rate': found / max(len(cells), 1)})
    tile_frame = pd.DataFrame(rows)
    dist = assigned['distance'][accepted]
    summary = {
        'detected_peaks': int(len(detections)),
        'theoretical_cells': int(len(grid_cells)),
        'unique_detected_cells': len(selected_cells & grid_cells),
        'detection_rate': len(selected_cells & grid_cells) / max(len(grid_cells), 1),
        'missing_cell_rate': 1 - len(selected_cells & grid_cells) / max(len(grid_cells), 1),
        'double_peak_rate': float(assigned['duplicate'].sum() / max(int(accepted.sum()), 1)),
        'lattice_distance_mean_px': float(np.mean(dist)) if len(dist) else float('nan'),
        'lattice_distance_std_px': float(np.std(dist)) if len(dist) else float('nan'),
        'lattice_distance_p95_px': float(np.percentile(dist, 95)) if len(dist) else float('nan'),
        'worst_tile_detection_rate': float(tile_frame['detection_rate'].min()),
        'tile_detection_rate_std': float(tile_frame['detection_rate'].std()),
    }
    return summary, tile_frame


def defect_candidates(raw: np.ndarray) -> tuple[dict, list[dict]]:
    try:
        regions, zscore = dm.residual_candidate_regions(raw)
    except ValueError as exc:
        return {'candidate_count': 0, 'candidate_total_area_px': 0, 'candidate_max_z': float('nan'),
                'error': str(exc)}, []
    total_area = sum(r['area_px'] for r in regions)
    max_z = max((r['max_residual_z'] for r in regions), default=float('nan'))
    return {'candidate_count': len(regions), 'candidate_total_area_px': int(total_area),
            'candidate_max_z': float(max_z), 'error': ''}, regions


def save_side_by_side(images: dict[int, dict[str, np.ndarray]], output: Path, vmin: float, vmax: float):
    positions = sorted(images)
    fig, axes = plt.subplots(2, len(positions), figsize=(3.2 * len(positions), 6.6))
    for col, pos in enumerate(positions):
        for row, label in enumerate(['pre', 'post']):
            ax = axes[row, col]
            ax.imshow(images[pos][label], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f'Position {pos} ({label})', fontsize=9)
            ax.axis('off')
    fig.suptitle('260826 p50 SAM Sample 1 — common contrast scale '
                 f'(vmin={vmin:.0f}, vmax={vmax:.0f} DN)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_defect_overlay(pos: int, raw: np.ndarray, regions: list[dict], zscore: np.ndarray,
                          vmin: float, vmax: float, output: Path, label: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(raw, cmap='gray', vmin=vmin, vmax=vmax)
    for r in regions:
        axes[0].add_patch(plt.Rectangle((r['bbox_x'], r['bbox_y']), r['bbox_w'], r['bbox_h'],
                                          fill=False, edgecolor='red', linewidth=1.2))
    axes[0].set_title(f'Position {pos} {label}: {len(regions)} defect candidates')
    axes[0].axis('off')
    im = axes[1].imshow(zscore, cmap='inferno', vmin=0, vmax=10)
    axes[1].set_title('periodic-residual z-score')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    output = ROOT / 'data/results/v6_position6_image_forensics'
    output.mkdir(parents=True, exist_ok=True)

    raw_images: dict[int, dict[str, np.ndarray]] = {}
    summary_rows = []
    all_dn_values = []

    for pos in POSITIONS:
        raw_images[pos] = {}
        for suffix, label in [('0', 'pre'), ('1', 'post')]:
            path = ROOT_DIR / f'{SAMPLE}-{pos}-{suffix}.tif'
            raw = load_image_unicode(str(path))
            raw_images[pos][label] = raw
            all_dn_values.append(raw)
            image01 = image_for_ecc(raw)

            print(f'Position {pos} {label}: metrics', flush=True)
            row = {'position': pos, 'label': label}
            row.update(brightness_metrics(raw))
            row.update(focus_metrics(raw))
            lat_summary, tile_frame = lattice_regularity(str(path), image01)
            row.update(lat_summary)
            tile_frame.insert(0, 'position', pos)
            tile_frame.insert(1, 'label', label)
            tile_frame.to_csv(output / f'lattice_tiles_{pos}_{label}.csv', index=False)

            defect_summary, regions = defect_candidates(raw)
            row.update(defect_summary)
            if regions:
                pd.DataFrame(regions).assign(position=pos, label=label).to_csv(
                    output / f'defect_candidates_{pos}_{label}.csv', index=False)
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / 'position_image_forensics_summary.csv', index=False)
    print(summary.to_string(index=False))

    stacked = np.concatenate([np.asarray(v, np.float64).ravel() for d in raw_images.values() for v in d.values()])
    vmin, vmax = np.percentile(stacked, [1, 99])
    save_side_by_side(raw_images, output / 'side_by_side_common_contrast.png', vmin, vmax)

    for pos in POSITIONS:
        for label in ['pre', 'post']:
            raw = raw_images[pos][label]
            try:
                regions, zscore = dm.residual_candidate_regions(raw)
            except ValueError:
                continue
            save_defect_overlay(pos, raw, regions, zscore, vmin, vmax,
                                 output / f'defect_overlay_{pos}_{label}.png', label)

    print(f'Outputs written to {output}', flush=True)


if __name__ == '__main__':
    main()
