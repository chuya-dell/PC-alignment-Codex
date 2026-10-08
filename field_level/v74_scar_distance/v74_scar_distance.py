"""v74 step E: cross-scar problem by distance class (real blank data part), substitution rate, local focus proxy, local gradient.
Distance classes (period = 7.37 px): inside mask (depth >2 / 1-2 / 0-1 periods), outside 0-1, 1-2, 2-4, 4-8, >8 periods.
Rates are for readouts S0 (standard), S1, S3 and threshold G (date-blank pool, mean+3SD, from v72 thresholds.csv).
Also: background-estimate variant (scar pixels removed from the 51-px Gaussian background, normalized convolution) for S3 contrast.
usage: python v74_scar_distance.py"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit')); sys.path.insert(0, str(ROOT / 'field_level/v72_real_field_readout'))
import numpy as np, pandas as pd, cv2
from scipy import ndimage
import v70_lib as L
import v70_centers2 as C
import v72_readout as R
V70 = ROOT / 'data/results/v70_ledger_center_fit'
V72 = ROOT / 'data/results/v72_real_field_readout'
OUT = ROOT / 'data/results/v74_scar_distance'
PER = 7.37
EDGES = [-1e9, -2, -1, 0, 1, 2, 4, 8, 1e9]
NAMES = ['in_core(>2per)', 'in_depth1-2', 'in_depth0-1', 'out_0-1', 'out_1-2', 'out_2-4', 'out_4-8', 'out_>8']


def signed_dist_periods(mask, xy):
    out = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    inn = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    sd = np.where(mask, -inn, out) / PER
    q = np.clip(np.rint(xy).astype(int), [0, 0], [R.W - 1, R.H - 1])
    return sd[q[:, 1], q[:, 0]]


def bg_masked_contrast(im, mask):
    """Box contrast with the Gaussian background estimated WITHOUT scar pixels (normalized convolution)."""
    f = im.astype(np.float32) / 65535
    w = (~cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)).astype(np.float32)
    num = cv2.GaussianBlur(f * w, (51, 51), 0); den = cv2.GaussianBlur(w, (51, 51), 0)
    bg = np.where(den > 0.05, num / np.maximum(den, 1e-6), cv2.GaussianBlur(f, (51, 51), 0))
    return cv2.boxFilter(f - bg, -1, (3, 3), normalize=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}).set_index('fid')
    th = pd.read_csv(V72 / 'thresholds_v2.csv', dtype={'date': str}); th = th[th.thr == 'G'].set_index(['date', 'readout']).threshold
    rows = []
    for fid, r in led.iterrows():
        mask = R.load_mask(fid)
        if mask is None: continue
        z = np.load(V72 / 'fields' / f'{fid}.npz'); z2 = np.load(V70 / 'fields2' / f'{fid}.npz')
        pre = dict(ctr=z2['pre_ctr'].astype(float), pred=z2['pre_pred'], amp=z2['pre_amp'], wid=z2['pre_wid'], aloc=z2['pre_aloc'])
        Sp = C.sub_flags(pre, 'main')
        for tag, xy, S in (('S0', z['std_xy'].astype(float), z['S0']), ('S1', z['std_xy'].astype(float), z['S1']), ('S3', z['ctr_xy'].astype(float), z['S3'])):
            sd = signed_dist_periods(mask, xy); cls = np.digitize(sd, EDGES[1:-1])
            ok = np.isfinite(S); t = th[(r.date, tag)]
            for k, nm in enumerate(NAMES):
                sel = ok & (cls == k)
                n = int(sel.sum())
                row = dict(fid=fid, date=r.date, board=r.board, substrate=f'{r.date}_{r.board}', group=r.group, fixed29=r.fixed29, readout=tag, cls=nm, n=n,
                           pos=int((S[sel] > t).sum()), rate=float((S[sel] > t).mean()) if n else np.nan)
                if tag == 'S3':
                    allk = (cls == k)
                    row['n_nodes'] = int(allk.sum()); row['substituted_rate'] = float(Sp[allk].mean()) if allk.any() else np.nan
                    row['amp_rel_median'] = float(np.median(z2['pre_amp'][allk] / z2['pre_aloc'][allk])) if allk.any() else np.nan
                    row['wid_median'] = float(np.median(z2['pre_wid'][allk])) if allk.any() else np.nan
                rows.append(row)
        print('done', fid, flush=True) if len(rows) % 3000 < 24 else None
    t = pd.DataFrame(rows); t.to_csv(OUT / 'scar_distance_by_field.csv', index=False)
    print('saved', len(t))


if __name__ == '__main__':
    main()
