"""v72 step C (analysis): thresholds (date-blank pooled global / local-median), band metrics, scar inside/outside, sets."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v60_band_scar_controls'))
import numpy as np, pandas as pd
from scipy import ndimage
import field_control_common as M
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
V70 = ROOT / 'data/results/v70_ledger_center_fit'
OUT = ROOT / 'data/results/v72_real_field_readout'
READS = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
LABEL = {'S0': 'standard (ideal,round,A)', 'S1': 'ideal,round,L-map', 'S2': 'ideal,bilinear,L-map', 'S3': 'centres,round', 'S4': 'centres,bilinear', 'S5': 'centres,aperture'}
W, H = 2048, 2044
PILLARS_REF = 91000.0     # about the number of pillars per field used to convert rates to counts


def pos_of(z, s):
    return z['std_xy'].astype(float) if s in ('S0', 'S1', 'S2') else z['ctr_xy'].astype(float)


_ids_cache = {}
def ids_of(fid, s):
    if s in ('S0', 'S1', 'S2'): return np.load(V70 / 'fields' / f'{fid}.npz')['std_ids']
    return np.load(V70 / 'fields2' / f'{fid}.npz')['pre_ids']


def local_resid(xy, d, block=24):
    ok = np.isfinite(d)
    gx = np.clip((xy[:, 0] // block).astype(int), 0, W // block); gy = np.clip((xy[:, 1] // block).astype(int), 0, H // block)
    ny, nx = H // block + 1, W // block + 1
    cell = gy * nx + gx
    order = np.argsort(cell[ok]); cs = cell[ok][order]; v = d[ok][order]
    med = np.full(ny * nx, np.nan)
    b = np.flatnonzero(np.diff(cs)) + 1
    for c_, seg in zip(np.split(cs, b), np.split(v, b)):
        if len(seg) >= 15: med[c_[0]] = np.median(seg)
    med = med.reshape(ny, nx)
    glob = np.nanmedian(med) if np.isfinite(med).any() else 0
    med = np.where(np.isfinite(med), med, glob)
    sm = ndimage.median_filter(med, size=3, mode='nearest')
    loc = ndimage.map_coordinates(sm, [xy[:, 1] / block - .5, xy[:, 0] / block - .5], order=1, mode='nearest')
    return d - loc


def local_resid_ids(ids, xy, d, radius_periods=5.0, pitch=7.37):
    """Local residual: d minus the median of valid d within radius_periods (real distance) on the lattice-index grid.
    Index grid with NaN holes filled by the global median (rank filter; documented approximation)."""
    ids = np.asarray(ids); ok = np.isfinite(d)
    if ok.sum() < 1000: return d.copy()
    X = np.column_stack([np.ones(len(ids)), ids]); coef = np.linalg.lstsq(X[ok], xy[ok], rcond=None)[0]; B = coef[1:]      # rows: a1, a2
    i0, j0 = ids.min(axis=0); ni, nj = ids.max(axis=0) - ids.min(axis=0) + 1
    grid = np.full((ni, nj), np.nan, np.float32); grid[ids[:, 0] - i0, ids[:, 1] - j0] = d
    gm = np.nanmedian(grid); filled = np.where(np.isfinite(grid), grid, gm)
    r = int(np.ceil(radius_periods * pitch / min(np.linalg.norm(B[0]), np.linalg.norm(B[1])))) + 1
    di, dj = np.mgrid[-r:r + 1, -r:r + 1]
    foot = (np.linalg.norm(di[..., None] * B[0] + dj[..., None] * B[1], axis=-1) <= radius_periods * pitch)
    loc = ndimage.median_filter(filled, footprint=foot, mode='nearest')
    return d - loc[ids[:, 0] - i0, ids[:, 1] - j0]


def main():
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str})
    fids = [f for f in led.fid if (OUT / 'fields' / f'{f}.npz').exists()]
    led = led[led.fid.isin(fids)].reset_index(drop=True)
    # pass 1: blank pools per date
    pools = {}
    for r in led[led.group == 'blank'].itertuples():
        z = np.load(OUT / 'fields' / f'{r.fid}.npz')
        for s in READS:
            d = z[s].astype(float); xy = pos_of(z, s); ids = ids_of(r.fid, s)
            for thr, vv in (('G', d), ('Lc', local_resid_ids(ids, xy, d))):
                pools.setdefault((r.date, s, thr), []).append(vv[np.isfinite(vv)])
    TH = {}
    for k, v in pools.items():
        d = np.concatenate(v); TH[k] = (float(d.mean()), float(d.std()), float(d.mean() + 3 * d.std()), len(d))
    pd.DataFrame([dict(date=k[0], readout=k[1], thr=k[2], mean=v[0], sd=v[1], threshold=v[2], n=v[3]) for k, v in TH.items()]).to_csv(OUT / 'thresholds_v2.csv', index=False)
    rows = []; maps = {}
    fixed = set(led[led.fixed29].fid)
    for r in led.itertuples():
        z = np.load(OUT / 'fields' / f'{r.fid}.npz')
        for s in READS:
            d = z[s].astype(float); xy = pos_of(z, s); ids = ids_of(r.fid, s)
            inm = z['in_mask_std'] if s in ('S0', 'S1', 'S2') else z['in_mask_ctr']
            dist = z['dist_std'] if s in ('S0', 'S1', 'S2') else z['dist_ctr']
            for thr, vv in (('G', d), ('Lc', local_resid_ids(ids, xy, d))):
                t = TH[(r.date, s, thr)][2]
                ok = np.isfinite(vv)
                met, dm = M.band_metrics(xy, vv, t, True)
                pos = ok & (vv > t)
                n_in = int((ok & inm).sum()); n_out = int((ok & ~inm).sum())
                rows.append(dict(fid=r.fid, date=r.date, board=r.board, group=r.group, fixed29=r.fixed29, readout=s, thr=thr, threshold=t, n_valid=int(ok.sum()),
                                 positive=int(pos.sum()), rate=float(pos.sum() / max(ok.sum(), 1)), count_per_91k=float(pos.sum() / max(ok.sum(), 1) * PILLARS_REF),
                                 rate_in_mask=float((pos & inm).sum() / max(n_in, 1)) if n_in > 50 else np.nan, rate_out_mask=float((pos & ~inm).sum() / max(n_out, 1)),
                                 share_pos_in_mask=float((pos & inm).sum() / max(pos.sum(), 1)), mask_fraction=n_in / max(ok.sum(), 1),
                                 rate_dist_gt20=float((pos & (dist > 20)).sum() / max((ok & (dist > 20)).sum(), 1)) if np.isfinite(dist).any() else np.nan,
                                 strength=met['strength'], band_power=met['band_power'], density_sd=met['density_sd'], axis_deg=met['axis_deg'], period_px=met['period_px'], concentration=met['concentration']))
                if r.fid in fixed and thr == 'G': maps[f'{r.fid}__{s}'] = dm
    t = pd.DataFrame(rows); t.to_csv(OUT / 'metrics_by_field_v2.csv', index=False)
    np.savez_compressed(OUT / 'density_maps_fixed29.npz', **maps)
    # sets: fixed29, band_top20 (non-fixed, highest S0/G band_power), normal blanks (20, seeded, non-fixed, non-top)
    s0 = t[(t.readout == 'S0') & (t.thr == 'G')].copy()
    nonfixed = s0[~s0.fixed29]
    top20 = list(nonfixed.sort_values('band_power', ascending=False).fid.head(20))
    rng = np.random.default_rng(20261008)
    pool = nonfixed[(nonfixed.group == 'blank') & (~nonfixed.fid.isin(top20))].fid.tolist()
    normal20 = list(rng.choice(pool, size=min(20, len(pool)), replace=False))
    sets = {'fixed29': list(fixed), 'band_top20': top20, 'normal_blank20': normal20, 'all_blank': list(led[led.group == 'blank'].fid), 'all632': list(led.fid)}
    json.dump(sets, open(OUT / 'sets.json', 'w'))
    lines = []
    for sname, fl in sets.items():
        g = t[t.fid.isin(fl)]
        for (s, thr), q in g.groupby(['readout', 'thr']):
            lines.append(dict(set=sname, readout=s, thr=thr, fields=q.fid.nunique(), rate_median=q.rate.median(), rate_mean=q.rate.mean(), count91k_median=q.count_per_91k.median(), count91k_mean=q.count_per_91k.mean(),
                              count91k_p90=q.count_per_91k.quantile(.9), count91k_max=q.count_per_91k.max(), band_power_median=q.band_power.median(), strength_median=q.strength.median(), density_sd_median=q.density_sd.median(),
                              rate_in_mask_median=q.rate_in_mask.median(), rate_out_mask_median=q.rate_out_mask.median(), share_pos_in_mask_median=q.share_pos_in_mask.median(),
                              frac_fields_le2=float((q.count_per_91k <= 2).mean()), frac_fields_le10=float((q.count_per_91k <= 10).mean())))
    sm = pd.DataFrame(lines); sm.to_csv(OUT / 'set_summary_v2.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    for sname in ['fixed29', 'band_top20', 'normal_blank20', 'all_blank']:
        print('==', sname); print(sm[sm.set == sname][['readout', 'thr', 'fields', 'rate_median', 'rate_mean', 'count91k_median', 'count91k_p90', 'band_power_median', 'strength_median', 'rate_out_mask_median', 'share_pos_in_mask_median']].round(5).to_string(index=False))
    return t, sets


if __name__ == '__main__':
    main()
