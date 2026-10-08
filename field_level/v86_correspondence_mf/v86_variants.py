"""v86 exploration (development fields only; nothing tuned on the 2026-10-08 data): which parts of a matched-filter fit raise the correspondence rate?
Variants (all keep the substitution THRESHOLDS of 'main' and the pre->post pairing of v70):
  base   : v70 estimator (stored)
  mf_cc  : MF centres refit with the lattice smoothed from the MF centres (poly cubic, as v70), MF amplitude
  mf_cb  : MF centres, but amplitude = baseline estimator (contrast sample at the centre - 20th pct of the 9x9 patch)
  cen_mf : baseline centres (stored), MF amplitude
usage: python v86_variants.py [--n 8]"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit')); sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, cv2
from scipy import ndimage
import v70_lib as L
import v70_centers2 as C
import v86_mf_centers as M
V70 = ROOT / 'data/results/v70_ledger_center_fit'
W, H = 2048, 2044


def mf_response(im, s):
    cb = L.contrast_plain(im); return cb, cv2.GaussianBlur(cb, (0, 0), s).astype(np.float32)


def peak_fit(R, pred, search=2.0, step=0.25):
    off = np.arange(-search, search + 1e-9, step); oy, ox = np.meshgrid(off, off, indexing='ij'); oy = oy.ravel(); ox = ox.ravel(); n = len(pred)
    vals = np.empty((len(ox), n), np.float32)
    for k in range(len(ox)): vals[k] = M.sample(R, pred[:, 0] + ox[k], pred[:, 1] + oy[k])
    bi = vals.argmax(axis=0); g = len(off); iy, ix = np.divmod(bi, g); ar = np.arange(n)
    def at(dy, dx): return vals[np.clip(iy + dy, 0, g - 1) * g + np.clip(ix + dx, 0, g - 1), ar]
    c0 = vals[bi, ar]; cxm, cxp, cym, cyp = at(0, -1), at(0, 1), at(-1, 0), at(1, 0)
    dxs = step * 0.5 * (cxm - cxp) / (cxm - 2 * c0 + cxp - 1e-12); dys = step * 0.5 * (cym - cyp) / (cym - 2 * c0 + cyp - 1e-12)
    return pred + np.column_stack([ox[bi] + np.clip(dxs, -step, step), oy[bi] + np.clip(dys, -step, step)])


def patch_p20(img, ctr):
    xs = ctr[:, 0][:, None] + M._PX[None]; ys = ctr[:, 1][:, None] + M._PY[None]
    p = ndimage.map_coordinates(img, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(len(ctr), -1); return np.percentile(p, 20, axis=1)


def width_at(cb, ctr):
    xs = ctr[:, 0][:, None] + M._PX[None]; ys = ctr[:, 1][:, None] + M._PY[None]
    v = ndimage.map_coordinates(cb, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(len(ctr), -1); b2 = np.percentile(v, 20, axis=1)[:, None]; r2 = M._PX[None] ** 2 + M._PY[None] ** 2
    w2 = np.maximum(v - b2, 0) * (r2 <= L.R ** 2); s2 = w2.sum(axis=1) + 1e-12; mx = (w2 * M._PX[None]).sum(axis=1) / s2; my = (w2 * M._PY[None]).sum(axis=1) / s2
    return np.sqrt(np.maximum(((w2 * ((M._PX[None] - mx[:, None]) ** 2 + (M._PY[None] - my[:, None]) ** 2)).sum(axis=1) / s2) / 2.0, 0))


def centers_mf(im, ids, pred_lin, s=1.3, amp_mode='mf'):
    cb, R = mf_response(im, s)
    ctr = peak_fit(R, pred_lin, 2.0); amp = M.sample(R, ctr[:, 0], ctr[:, 1]) - patch_p20(R, ctr); pred = pred_lin
    T = C.poly_terms(ids)
    for rnd in range(3):
        aloc, glob = C.local_aref(ctr, amp); good = (amp > 0.5 * aloc) & (np.linalg.norm(ctr - pred, axis=1) < 2.2); sel = good.copy()
        for _ in range(3):
            coef = np.linalg.lstsq(T[sel], ctr[sel], rcond=None)[0]; res = np.linalg.norm(ctr - T @ coef, axis=1); sel = good & (res < max(0.4, 3 * np.median(res[sel])))
        pred = T @ coef; ctr = peak_fit(R, pred, 1.5); amp = M.sample(R, ctr[:, 0], ctr[:, 1]) - patch_p20(R, ctr)
    if amp_mode == 'cb': amp = M.sample(cb, ctr[:, 0], ctr[:, 1]) - patch_p20(cb, ctr)
    aloc, _ = C.local_aref(ctr, amp); wid = width_at(cb, ctr)
    return dict(ctr=ctr, pred=pred, amp=amp, wid=wid, aloc=aloc)


def main():
    n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 8
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}); sel = led[led.pre_exists & led.post_exists & led.fixed29].iloc[::max(1, 29 // n)][:n]
    rows = []
    for rec in sel.to_dict('records'):
        z = np.load(V70 / 'fields2' / f"{rec['fid']}.npz"); pre = L.read(rec['pre_path']); post = L.read(rec['post_path'])
        pb = dict(ctr=z['pre_ctr'].astype(float), pred=z['pre_pred'], amp=z['pre_amp'], wid=z['pre_wid'], aloc=z['pre_aloc']); qb = dict(ctr=z['post_ctr'].astype(float), pred=z['post_pred'], amp=z['post_amp'], wid=z['post_wid'], aloc=z['post_aloc'])
        r = dict(fid=rec['fid'], base=M.rate_from(z, pb, qb)['main'][0])
        for s in (1.0, 1.3):
            for mode in ('mf', 'cb'):
                a = centers_mf(pre, z['pre_ids'], z['pre_pred_lin'].astype(np.float64), s, mode); b = centers_mf(post, z['post_ids'], z['post_pred_lin'].astype(np.float64), s, mode)
                r[f'mf_s{s}_{mode}'] = M.rate_from(z, a, b)['main'][0]
        # baseline centres + MF amplitude
        cb_, Rr = mf_response(pre, 1.3); cb2, Rq = mf_response(post, 1.3)
        ampa = M.sample(Rr, pb['ctr'][:, 0], pb['ctr'][:, 1]) - patch_p20(Rr, pb['ctr']); ampb = M.sample(Rq, qb['ctr'][:, 0], qb['ctr'][:, 1]) - patch_p20(Rq, qb['ctr'])
        a2 = dict(pb, amp=ampa, aloc=C.local_aref(pb['ctr'], ampa)[0]); b2 = dict(qb, amp=ampb, aloc=C.local_aref(qb['ctr'], ampb)[0])
        r['cen_base_amp_mf'] = M.rate_from(z, a2, b2)['main'][0]
        rows.append(r); print(r, flush=True)
    t = pd.DataFrame(rows); pd.set_option('display.width', 250); print(t.round(4).to_string()); print(t.drop(columns='fid').median().round(4))


if __name__ == '__main__':
    main()
