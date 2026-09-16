"""Semi-synthetic alignment benchmark variant embedding transforms drawn directly
from the 2026-09-01 record-reconstruction verification (ECC_20260901_REFERENCE_VALIDATION_20260916.md),
instead of the small synthetic sweep in field_run_semisynthetic_alignment_benchmark.py.

Ground truth per benchmark position is the (dx, dy, d_theta_deg, d_scale_pct) tuple observed for
one real FOV in that verification table, chosen so the five benchmark positions jointly span the
observed extremes (x: -26.756..+4.493 px, y: -8.061..+20.157 px, rotation: -0.19828..+0.01729 deg,
scale: -0.47135..+0.45361 pct). Position 6 (the position under scrutiny) is assigned the most
extreme case (real FOV 1-8, dx=-26.756 px) deliberately, not because that FOV corresponds to
position 6 physically.

Everything else (image loading, spike injection, detection/scoring, methods) is unchanged from
field_run_semisynthetic_alignment_benchmark.py.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('phase2', ROOT/'field_level/v3_fov16_ecc_root_cause/field_diagnose_fov16_ecc.py')
phase2 = importlib.util.module_from_spec(spec); sys.modules[spec.name] = phase2; spec.loader.exec_module(phase2)

# (dx_px, dy_px, theta_deg, delta_scale_pct) per benchmark position, sourced from
# docs/ECC_20260901_REFERENCE_VALIDATION_20260916.md table (reconstruction - record deltas).
REAL_RANGE_TABLE = {
    1: dict(source_fov='1-1', tx=-4.896, ty=-8.061, theta_deg=-0.19828, delta_scale_pct=0.45361),
    2: dict(source_fov='1-2', tx=-7.058, ty=3.464, theta_deg=-0.02826, delta_scale_pct=-0.47135),
    5: dict(source_fov='1-5', tx=3.807, ty=9.651, theta_deg=0.01729, delta_scale_pct=-0.01115),
    6: dict(source_fov='1-8', tx=-26.756, ty=1.188, theta_deg=0.00627, delta_scale_pct=-0.09860),
    8: dict(source_fov='1-3', tx=4.493, ty=20.157, theta_deg=0.00097, delta_scale_pct=0.02058),
}

def affine_metrics(m):
    a = m[:, :2]; s = math.sqrt(abs(np.linalg.det(a)))
    return dict(tx_px=float(m[0, 2]), ty_px=float(m[1, 2]),
                theta_deg=float(math.degrees(math.atan2(a[1, 0], a[0, 0]))),
                delta_scale_pct=float((s - 1) * 100))

def gt_for_position(pos):
    e = REAL_RANGE_TABLE[pos]
    theta = math.radians(e['theta_deg']); scale = 1 + e['delta_scale_pct'] / 100.0
    A = np.array([[scale * math.cos(theta), -scale * math.sin(theta)],
                  [scale * math.sin(theta), scale * math.cos(theta)]])
    t = np.array([e['tx'], e['ty']])
    return np.c_[A, t].astype(np.float32)

def make_moving(pre, render_warp):
    return cv2.warpAffine(pre, render_warp, (pre.shape[1], pre.shape[0]),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)

def inject(moving, truth_template, gt, amp, sigma):
    out = moving.copy(); h, w = out.shape
    xy = cv2.transform(truth_template[None].astype(np.float32), gt)[0]
    r = max(3, int(math.ceil(4 * sigma))); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = amp * np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    for x, y in xy:
        xi, yi = round(float(x)), round(float(y))
        x0, x1 = max(0, xi - r), min(w, xi + r + 1); y0, y1 = max(0, yi - r), min(h, yi + r + 1)
        out[y0:y1, x0:x1] += kernel[y0 - (yi - r):y1 - (yi - r), x0 - (xi - r):x1 - (xi - r)]
    return np.clip(out, 0, 1)

def sample_truth(shape, n, rng):
    h, w = shape; pts = []
    while len(pts) < n:
        p = np.array([rng.uniform(60, w - 60), rng.uniform(60, h - 60)])
        if not pts or np.min(np.linalg.norm(np.asarray(pts) - p, axis=1)) > 16:
            pts.append(p)
    return np.asarray(pts, np.float32)

def ecc(pre, mov):
    return phase2.run_ecc_path(pre, mov, 0, 0, [.25, .5, 1], iterations=50, eps=1e-6)[1]

def orb(pre, mov):
    a = np.uint8(np.clip(pre * 255, 0, 255)); b = np.uint8(np.clip(mov * 255, 0, 255))
    detector = cv2.ORB_create(nfeatures=12000, fastThreshold=3)
    ka, da = detector.detectAndCompute(a, None); kb, db = detector.detectAndCompute(b, None)
    if da is None or db is None: raise RuntimeError('ORB descriptors unavailable')
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in pairs if m.distance < .72 * n.distance]
    if len(good) < 8: raise RuntimeError(f'ORB matches insufficient: {len(good)}')
    src = np.float32([ka[m.queryIdx].pt for m in good]); dst = np.float32([kb[m.trainIdx].pt for m in good])
    m, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=4000, confidence=.995)
    if m is None: raise RuntimeError('ORB RANSAC failed')
    return m.astype(np.float32)

def dense_flow(pre, mov):
    scale = .25; size = (round(pre.shape[1] * scale), round(pre.shape[0] * scale))
    a = cv2.resize(pre, size, interpolation=cv2.INTER_AREA); b = cv2.resize(mov, size, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 25, 3, 5, 1.2, 0)
    h, w = a.shape; y, x = np.mgrid[20:h - 20:12, 20:w - 20:12]
    src = np.c_[x.ravel(), y.ravel()].astype(np.float32); dst = src + flow[y, x].reshape(-1, 2)
    m, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.0, maxIters=3000, confidence=.995)
    if m is None: raise RuntimeError('dense-flow affine fit failed')
    m[:, 2] /= scale
    return m.astype(np.float32)

METHODS = {'ecc_affine_pyramid': ecc, 'orb_ransac_affine': orb, 'dense_farneback_affine': dense_flow}

def detect_and_score(pre, mov, estimated, truth, radius):
    aligned = cv2.warpAffine(mov, estimated, (pre.shape[1], pre.shape[0]),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    d = ndimage.gaussian_filter(aligned - pre, 1)
    peak = (d == ndimage.maximum_filter(d, 5)) & (d > .018)
    peak[:10] = False; peak[-10:] = False; peak[:, :10] = False; peak[:, -10:] = False
    y, x = np.where(peak); found = np.c_[x, y].astype(float)
    tree = KDTree(found) if len(found) else None
    matched = np.zeros(len(truth), bool)
    if tree: matched = tree.query(truth)[0] <= radius
    truth_tree = KDTree(truth)
    fp = int(np.sum(truth_tree.query(found)[0] > radius)) if len(found) else 0
    return int(matched.sum()), len(found), fp

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=ROOT/'data/raw/semisynthetic_alignment_benchmark_manifest.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--positions', type=int, nargs='+', default=None)
    a = p.parse_args()
    c = json.loads(a.manifest.read_text(encoding='utf-8'))
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []; reg = []
    positions = a.positions if a.positions is not None else c['positions']
    if set(positions) - set(REAL_RANGE_TABLE): raise ValueError('No real-range entry for requested position')
    for pos in positions:
        print(f'Position {pos} (real-range src {REAL_RANGE_TABLE[pos]["source_fov"]}): loading/aligning', flush=True)
        raw = phase2.load_image_unicode(str(Path(c['root'])/f"{c['sample']}-{pos}-0.tif"))
        pre = phase2.image_for_ecc(raw)
        gt = gt_for_position(pos)
        render_warp = cv2.invertAffineTransform(gt)
        base = make_moving(pre, render_warp)
        estimates = {}
        for name, fn in METHODS.items():
            try:
                estimates[name] = fn(pre, base); ok = True; err = ''
            except Exception as e:
                estimates[name] = np.eye(2, 3, dtype=np.float32); ok = False; err = str(e)
            gm, em = affine_metrics(gt), affine_metrics(estimates[name])
            reg.append({'position': pos, 'method': name, 'converged': ok, 'error': err,
                        'source_fov': REAL_RANGE_TABLE[pos]['source_fov'],
                        **{f'ground_truth_{k}': v for k, v in gm.items()},
                        **{f'estimated_{k}': v for k, v in em.items()},
                        **{f'error_{k}': em[k] - gm[k] for k in gm}})
        for count in c['spike_counts']:
            truth = sample_truth(pre.shape, count, np.random.default_rng(c['seed'] + pos * 1000 + count))
            mov = inject(base, truth, gt, c['spike_amplitude_normalized'], c['spike_sigma_px'])
            for name, m in estimates.items():
                recovered, total, fp = detect_and_score(pre, mov, m, truth, c['match_radius_px'])
                rows.append({'position': pos, 'method': name, 'spike_count': count,
                             'recovered_count': recovered, 'recovery_rate': recovered / count,
                             'detected_count': total, 'false_positive_count': fp,
                             'false_positive_rate': fp / max(total, 1)})
        print(f'Position {pos}: done', flush=True)
    trial = pd.DataFrame(rows); registration = pd.DataFrame(reg)
    trial.to_csv(a.output/'semisynthetic_trial_results.csv', index=False)
    registration.to_csv(a.output/'registration_known_truth_errors.csv', index=False)
    summary = (trial.groupby('method', as_index=False)
               .agg(mean_recovery_rate=('recovery_rate', 'mean'), mean_false_positive_rate=('false_positive_rate', 'mean'),
                    mean_recovered=('recovered_count', 'mean')))
    f6 = (trial[trial.position == 6].groupby('method', as_index=False)
          .agg(fov6_recovery_rate=('recovery_rate', 'mean'), fov6_false_positive_rate=('false_positive_rate', 'mean')))
    summary = summary.merge(f6, on='method')
    summary.to_csv(a.output/'method_summary.csv', index=False)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
