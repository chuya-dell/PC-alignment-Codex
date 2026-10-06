"""Version 45 step 1: per-image cross-mark (two long dark lines) visibility metrics.
Metrics are computed directly from raw images (read-only). Thresholds in CONFIG were fixed before the full run.
"""
import numpy as np, cv2, tifffile
from scipy.ndimage import uniform_filter1d, gaussian_filter

CONFIG = dict(
    T_det=0.05,        # segment counts as dark line present if 1 - bottom/surround >= this (revised after 1-image pilot, before full run)
    T_full=0.10,       # median segment darkness for "sufficiently dark" line (pilot visually-black line gave 0.145)
    cov_cont=0.90,     # coverage for continuous
    gap_ok_px=128,     # max allowed gap for continuous
    cov_absent=0.30,   # coverage below this = absent
    infov_min=0.50,    # fraction of the line inside the image to be gradable
    seg=64, band=6, sur_lo=20, sur_hi=45, ctrl_offsets=(-300, -200, 200, 300),
    noise_ctrl_cov=0.50,  # if control-offset coverage above this the field is undecidable
)


def load(path):
    return tifffile.imread(path)


def _D_map(G, lo, hi):
    """horizontal local darkness 1 - centre/surround for each pixel (along axis 1)."""
    c = uniform_filter1d(G, 3, axis=1, mode='nearest')
    W = G.shape[1]
    n = hi - lo + 1
    m = uniform_filter1d(G, n, axis=1, mode='nearest')
    mid = (lo + hi) // 2
    sl = np.full_like(G, np.nan)
    sr = np.full_like(G, np.nan)
    sl[:, mid:] = m[:, :W - mid]
    sr[:, :W - mid] = m[:, mid:]
    with np.errstate(all='ignore'):
        s = np.nanmean(np.stack([sl, sr]), axis=0)
    return 1.0 - c / s


def coarse_search(A):
    """A: 2-D image (axis 1 is across the line). Returns (tan_theta, x0_full, score)."""
    G = cv2.resize(A, (A.shape[1] // 4, A.shape[0] // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    D = np.clip(np.nan_to_num(_D_map(G, 5, 11), nan=0.0), -0.5, 1.0)
    H, W = D.shape
    ys = np.arange(H) - H / 2
    best = (-9, 0.0, 0)
    for th in np.arange(-0.12, 0.1201, 0.01):
        shift = np.round(th * ys).astype(int)
        idx = np.arange(W)[None, :] + shift[:, None]
        valid = (idx >= 0) & (idx < W)
        vals = np.where(valid, D[np.arange(H)[:, None], np.clip(idx, 0, W - 1)], 0.0)
        prof = uniform_filter1d(vals.mean(axis=0), 3)
        x = int(np.argmax(prof))
        if prof[x] > best[0]:
            best = (prof[x], th, x)
    sc, th, x = best
    return th, x * 4.0 + 2, sc


def line_segments(A, th, x0, cfg=CONFIG, offset=0):
    """Segment-wise darkness along a (tilted) line. axis0 = along the line, axis1 = across.
    x(y) = x0 + offset + th*(y - H/2)."""
    H, W = A.shape
    seg, band = cfg['seg'], cfg['band']
    out = []
    dx = np.arange(-cfg['sur_hi'] - 3, cfg['sur_hi'] + 4)
    c = len(dx) // 2
    for s0 in range(0, H - seg + 1, seg):
        ys = np.arange(s0, s0 + seg)
        xc = x0 + offset + th * (ys - H / 2)
        xi = np.round(xc).astype(int)
        cols = xi[:, None] + dx[None, :]
        ok = (cols >= 0) & (cols < W)
        vals = np.where(ok, A[ys[:, None], np.clip(cols, 0, W - 1)], np.nan)
        centre_in = ((xi >= 0) & (xi < W)).mean()
        nvalid = ok.sum(axis=0)
        with np.errstate(all='ignore'):
            prof = np.nanmean(vals, axis=0)
        prof = np.where(nvalid >= seg * 0.5, prof, np.nan)
        bad = dict(y0=s0, in_fov=True, D=np.nan, width=np.nan, bottom=np.nan, surround=np.nan, dpos=np.nan)
        if centre_in < 0.5 or not np.isfinite(prof[c]):
            bad['in_fov'] = False
            out.append(bad); continue
        fill = np.where(np.isfinite(prof), prof, np.nanmean(prof))
        p = np.convolve(fill, np.ones(3) / 3, mode='same')
        win = np.arange(c - band, c + band + 1)
        k = win[np.argmin(p[win])]
        bottom = p[k]
        sw = np.r_[np.arange(c - cfg['sur_hi'], c - cfg['sur_lo'] + 1), np.arange(c + cfg['sur_lo'], c + cfg['sur_hi'] + 1)]
        sv = prof[sw]
        sv = sv[np.isfinite(sv)]
        if sv.size < 6:
            bad['bottom'] = float(bottom)
            out.append(bad); continue
        surround = float(np.mean(sv))
        half = (surround + bottom) / 2
        l = k
        while l > 0 and p[l] < half: l -= 1
        r = k
        while r < len(p) - 1 and p[r] < half: r += 1
        out.append(dict(y0=s0, in_fov=True, D=float(1 - bottom / surround), width=float(r - l - 1),
                        bottom=float(bottom), surround=surround, dpos=int(k - c)))
    return out


def summarize(segs, cfg=CONFIG):
    inf = [s for s in segs if s['in_fov'] and np.isfinite(s['D'])]
    n_tot = len(segs)
    res = dict(n_seg=n_tot, n_infov=len(inf), infov_frac=len(inf) / n_tot)
    if len(inf) == 0:
        res.update(median_D=np.nan, p10_D=np.nan, coverage=np.nan, max_gap_px=np.nan, width_median=np.nan, ratio_median=np.nan)
        return res
    D = np.array([s['D'] for s in inf])
    isd = D >= cfg['T_det']
    gap = run = 0
    for v in isd:
        run = 0 if v else run + 1
        gap = max(gap, run)
    res.update(median_D=float(np.median(D)), p10_D=float(np.percentile(D, 10)), coverage=float(isd.mean()),
               max_gap_px=int(gap * cfg['seg']), width_median=float(np.nanmedian([s['width'] for s in inf])),
               ratio_median=float(np.median(1 - D)))
    return res


def line_state(res, cfg=CONFIG):
    if res['infov_frac'] < cfg['infov_min'] or not np.isfinite(res.get('coverage', np.nan)):
        return 'out_of_view'
    if res['coverage'] < cfg['cov_absent']: return 'absent'
    if res['coverage'] >= cfg['cov_cont'] and res['max_gap_px'] <= cfg['gap_ok_px']:
        return 'full' if res['median_D'] >= cfg['T_full'] else 'faint'
    return 'broken'


def classify(sv, sh, ctrl_cov, cfg=CONFIG):
    if ctrl_cov is not None and ctrl_cov > cfg['noise_ctrl_cov']: return '判定不能'
    pres = [s in ('full', 'faint', 'broken') for s in (sv, sh)]
    if not any(pres): return '写っていない'
    if not all(pres): return '片方のみ'
    if 'broken' in (sv, sh): return '途切れている'
    if 'faint' in (sv, sh): return '薄い'
    return '完全な黒線'


def analyse_image(im, cfg=CONFIG):
    A = gaussian_filter(im.astype(np.float32), 1.0)
    out, geo = {}, {}
    for name, M in (('v', A), ('h', np.ascontiguousarray(A.T))):
        th, x0, score = coarse_search(M)
        segs = line_segments(M, th, x0, cfg)
        r = summarize(segs, cfg)
        ctrl = []
        for off in cfg['ctrl_offsets']:
            cs = summarize(line_segments(M, th, x0, cfg, offset=off), cfg)
            if np.isfinite(cs['coverage']): ctrl.append(cs['coverage'])
        r['ctrl_coverage'] = float(np.mean(ctrl)) if ctrl else np.nan
        r['state'] = line_state(r, cfg)
        r['tan_theta'] = th; r['x0'] = x0; r['coarse_score'] = score
        geo[name] = (th, x0)
        for k, v in r.items(): out[f'{name}_{k}'] = v
    cs = [out['v_ctrl_coverage'], out['h_ctrl_coverage']]
    cc = float(np.nanmax(cs)) if np.isfinite(cs).any() else None
    out['ctrl_cov_max'] = cc
    out['cls'] = classify(out['v_state'], out['h_state'], cc, cfg)
    return out, geo
