"""v70 library: lattice model, per-pillar sub-pixel centers, physical index matching, current standard re-run.

New code only (existing versions untouched). The lattice-fit routine is adapted from v61 lattice_centers
(copied rather than imported so that importing does not create v61 output dirs).
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import numpy as np, cv2, tifffile
from scipy import ndimage, optimize
from scipy.spatial import cKDTree
from shared import registration as reg
from shared.lattice_indexing import lattice_from_fft, grid_coordinates
from shared.v2_registration_precision.refinement import register_refined
cv2.setNumThreads(1)
PITCH = 7.286
RAWPARENT = Path(r'W:\GoogleDrive\chuya2816')
STD = RAWPARENT / '5.解析結果_chu' / '20260928_digital_judgment_current_alignment'
CACHE = STD / 'tables' / 'cached_field_differences'
V57 = ROOT / 'data/results/v57_false_positive_facts_20261006'


def read(path):
    return tifffile.imread(path).astype(np.float32)


def contrast_plain(im):
    f = im.astype(np.float32) / 65535
    return f - cv2.GaussianBlur(f, (51, 51), 0)


def fit_lattice(im, guess=PITCH):
    """Frequency-refined hexagonal lattice with robust peak fit and optional quadratic distortion.
    Returns model (3x2: origin row then basis rows), quad (6x2 or None), info."""
    h, w = im.shape
    c = im - cv2.GaussianBlur(im, (51, 51), 0)
    win = np.outer(np.hanning(h), np.hanning(w))
    f = abs(np.fft.fftshift(np.fft.fft2(c * win)))
    fy = np.fft.fftshift(np.fft.fftfreq(h)); fx = np.fft.fftshift(np.fft.fftfreq(w))
    rad = np.hypot(fy[:, None], fx[None, :]); tg = 2 / (np.sqrt(3) * guess)
    cand = (ndimage.maximum_filter(f, 9) == f) & (rad > tg * .94) & (rad < tg * 1.06)
    iy, ix = np.where(cand); order = np.argsort(f[iy, ix])[::-1]; ks = []
    par = lambda a, b, c_: .5 * (a - c_) / (a - 2 * b + c_) if a - 2 * b + c_ != 0 else 0
    for n in order:
        y, x = iy[n], ix[n]
        k = np.array([fx[x] + par(f[y, x - 1], f[y, x], f[y, x + 1]) / w, fy[y] + par(f[y - 1, x], f[y, x], f[y + 1, x]) / h])
        if k[1] < 0 or any(abs(k @ q / np.linalg.norm(k) / np.linalg.norm(q)) > .8 for q in ks):
            continue
        ks.append(k)
        if len(ks) == 2:
            break
    if len(ks) < 2:
        raise ValueError('no reciprocal basis')
    yy = np.arange(0, h, 2); xx = np.arange(0, w, 2); cw = (c * win)[::2, ::2]

    def coef(k):
        return np.exp(-2j * np.pi * k[1] * yy) @ (cw @ np.exp(-2j * np.pi * k[0] * xx))
    ks = [optimize.minimize(lambda v: -abs(coef(v)) / 1e8, k, method='Nelder-Mead',
                            options={'xatol': 1e-9, 'fatol': 1e-6, 'maxiter': 100}).x for k in ks]
    phase = [np.angle(coef(k)) for k in ks]
    K = np.array(ks); B = np.linalg.inv(K)
    origin = np.linalg.solve(K, -np.array(phase) / (2 * np.pi))
    cc = cv2.GaussianBlur(c, (0, 0), .6)
    mx = (ndimage.maximum_filter(cc, 5) == cc) & (cc > np.percentile(cc, 60))
    y, x = np.where(mx); inside = (x > 30) & (x < w - 30) & (y > 30) & (y < h - 30); x = x[inside]; y = y[inside]
    dx = .5 * (cc[y, x - 1] - cc[y, x + 1]) / (cc[y, x - 1] - 2 * cc[y, x] + cc[y, x + 1] + 1e-12)
    dy = .5 * (cc[y - 1, x] - cc[y + 1, x]) / (cc[y - 1, x] - 2 * cc[y, x] + cc[y + 1, x] + 1e-12)
    pts = np.column_stack([x + np.clip(dx, -.5, .5), y + np.clip(dy, -.5, .5)])
    ids = np.rint((pts - origin) @ K.T).astype(int); design = np.column_stack([np.ones(len(ids)), ids])
    model = np.vstack([origin, B.T]); keep = np.linalg.norm(design @ model - pts, axis=1) < 1.7
    for _ in range(5):
        model = np.linalg.lstsq(design[keep], pts[keep], rcond=None)[0]
        err = np.linalg.norm(design @ model - pts, axis=1)
        keep = err < max(.4, min(1.2, 3 * np.median(err[keep])))
    nid = ids / 300.
    X = np.column_stack([design, nid[:, 0] ** 2, nid[:, 0] * nid[:, 1], nid[:, 1] ** 2])
    quad = np.linalg.lstsq(X[keep], pts[keep], rcond=None)[0]
    errq = np.linalg.norm(X @ quad - pts, axis=1)
    useq = np.median(errq[keep]) < .8 * np.median(err[keep]) and \
        np.percentile(np.linalg.norm(X[keep] @ quad - design[keep] @ model, axis=1), 95) < 1.0
    info = dict(pitch_px=float(np.mean(np.linalg.norm(model[1:], axis=1))), fitted_peaks=int(keep.sum()),
                residual_median_px=float(np.median((errq if useq else err)[keep])),
                residual95_px=float(np.percentile((errq if useq else err)[keep], 95)), quadratic=bool(useq))
    return model, (quad if useq else None), info


def lattice_pos(ids, model, quad):
    ids = np.asarray(ids); X = np.column_stack([np.ones(len(ids)), ids])
    if quad is None:
        return X @ model
    n = ids / 300.
    return np.column_stack([X, n[:, 0] ** 2, n[:, 0] * n[:, 1], n[:, 1] ** 2]) @ quad


def nodes_in_image(model, quad, w, h, margin=8):
    """All lattice nodes whose predicted position lies inside the image (margin px)."""
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    co = (corners - model[0]) @ np.linalg.inv(model[1:])
    lo = np.floor(co.min(0)).astype(int) - 3; hi = np.ceil(co.max(0)).astype(int) + 3
    i, j = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing='ij')
    ids = np.column_stack([i.ravel(), j.ravel()]); pos = lattice_pos(ids, model, quad)
    ok = (pos[:, 0] >= margin) & (pos[:, 0] < w - margin) & (pos[:, 1] >= margin) & (pos[:, 1] < h - margin)
    return ids[ok], pos[ok]


R = 3.0                      # support radius (px), provisional per request
HALF = 4
_DY, _DX = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]
_DY = _DY.ravel().astype(np.float64); _DX = _DX.ravel().astype(np.float64)


def _patch(cb, p):
    """Bilinear samples (n,81) at p + offsets; also returns offset grids."""
    xs = p[:, 0][:, None] + _DX[None]; ys = p[:, 1][:, None] + _DY[None]
    v = ndimage.map_coordinates(cb, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(len(p), -1)
    return v


def fit_centers(cb, pred, iters=5, sigma_w=1.6):
    """Iterative Gaussian-weighted centroid (support radius 3 px) from the lattice prediction.
    Returns centers, amplitude (peak minus local 20th percentile), RMS width (per-axis sigma)."""
    p = pred.astype(np.float64).copy(); n = len(p)
    for it in range(iters + 1):
        v = _patch(cb, p)
        base = np.percentile(v, 20, axis=1)[:, None]
        r2 = _DX[None] ** 2 + _DY[None] ** 2
        wgt = np.maximum(v - base, 0) * np.exp(-r2 / (2 * sigma_w ** 2)) * (r2 <= R ** 2)
        s = wgt.sum(axis=1) + 1e-12
        if it == iters:
            break
        p[:, 0] += (wgt * _DX[None]).sum(axis=1) / s
        p[:, 1] += (wgt * _DY[None]).sum(axis=1) / s
    # unweighted-by-gaussian width estimate at final center within radius 3
    w2 = np.maximum(v - base, 0) * (r2 <= R ** 2)
    s2 = w2.sum(axis=1) + 1e-12
    mx = (w2 * _DX[None]).sum(axis=1) / s2; my = (w2 * _DY[None]).sum(axis=1) / s2
    var = ((w2 * ((_DX[None] - mx[:, None]) ** 2 + (_DY[None] - my[:, None]) ** 2)).sum(axis=1) / s2) / 2.0
    # remaining refinement using the un-weighted centroid mean offset (second-moment consistent)
    amp = v[:, 40] - base[:, 0]            # center sample (offset 0,0 is index 40 of 9x9)
    return p, amp, np.sqrt(np.maximum(var, 0))


def center_fit_for_image(im):
    h, w = im.shape
    model, quad, info = fit_lattice(im)
    ids, pred = nodes_in_image(model, quad, w, h)
    cb = contrast_plain(im)
    ctr, amp, wid = fit_centers(cb, pred)
    shift = np.linalg.norm(ctr - pred, axis=1)
    pos = amp[amp > 0]
    aref = float(np.median(pos)) if len(pos) else 1.0
    return dict(ids=ids, pred=pred, ctr=ctr, amp=amp, wid=wid, shift=shift, aref=aref, info=info, model=model, quad=quad)


# substitution criteria (provisional; the 'main' set is the reported one, others give sensitivity)
CRIT = {'main': dict(shift_max=0.25 * PITCH, amp_frac=0.30, wid_lo=0.4, wid_hi=2.2),
        'strict': dict(shift_max=0.15 * PITCH, amp_frac=0.45, wid_lo=0.5, wid_hi=2.0),
        'loose': dict(shift_max=0.35 * PITCH, amp_frac=0.15, wid_lo=0.3, wid_hi=2.6)}


def substituted(cf, crit):
    c = CRIT[crit]
    bad = (cf['shift'] > c['shift_max']) | (cf['amp'] < c['amp_frac'] * cf['aref']) | \
          (cf['wid'] < c['wid_lo']) | (cf['wid'] > c['wid_hi'])
    t = cKDTree(cf['ctr'])
    for a, b in t.query_pairs(2.0):          # two nodes on the same pillar: weaker one is substituted
        bad[a if cf['amp'][a] < cf['amp'][b] else b] = True
    return bad


def match_pre_post(cf_pre, cf_post, matrix):
    """Pre lattice prediction mapped by the global affine -> nearest post lattice node."""
    exp = cf_pre['pred'] @ matrix[:, :2].T + matrix[:, 2]
    d, j = cKDTree(cf_post['pred']).query(exp)
    return j, d, exp


def standard_run(pre, post):
    """Current standard re-run from raw images (named current_standard_v70)."""
    coarse, qc = reg.register_image_pair_affine(pre, post, mask_stains=False, return_qc=True)
    matrix, ref = register_refined(pre, post, stage='subpixel', initial=coarse, mask_stains=False)
    lattice = lattice_from_fft(pre, PITCH)
    ids, xy = grid_coordinates(lattice, pre.shape[1], pre.shape[0], margin=30)
    postxy = xy @ matrix[:, :2].T + matrix[:, 2]
    aa = reg.sample_contrast(pre, xy); bb = reg.sample_contrast(post, postxy)
    valid = aa.valid_sampling.to_numpy() & bb.valid_sampling.to_numpy()
    delta = aa.contrast.to_numpy()[valid] - bb.contrast.to_numpy()[valid]
    return dict(delta=delta, xy=xy[valid], ids=ids[valid], matrix=matrix, qc=qc, ref=ref,
                lattice_basis=np.asarray(lattice.basis).tolist())
