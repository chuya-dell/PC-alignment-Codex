"""v85: effect of masking the camera-fixed hotspots.  Hotspots are defined from NON-blank, non-outlier development fields only (analyte/mismatch/other: 547 fields), so that the 69 blank fields and the
24 new (2026-10-08, -0/-3) fields are independent tests.  Radius R_MASK (px) around each hotspot centre with >= N_MIN_FIELDS fields.
Outputs: share of abnormal pillars and of strong (|z|>k) normal events inside the masks, FP per field before/after, calibrated k (P95<=2) before/after.   usage: python v85_hotspot_effect.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
V85 = ROOT / 'data/results/v85_abnormal_pillars'; V84 = ROOT / 'data/results/v84_1fM_strategies'; R83 = ROOT / 'data/results/v83_new_shots_261008'
R_MASK = 40.0; N_MIN_FIELDS = 10


def hotspots(points):
    X = points[['x', 'y']].values; tr = cKDTree(X); pr = tr.query_pairs(15.0, output_type='ndarray'); n = len(X)
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)); _, comp = connected_components(g, directed=False); sizes = np.bincount(comp)
    q = points.assign(c=np.where(sizes[comp] >= 8, comp, -1)); q = q[q.c >= 0]
    cl = q.groupby('c').agg(x=('x', 'mean'), y=('y', 'mean'), n=('x', 'size'), nf=('fid', 'nunique')).reset_index()
    return cl[cl.nf >= N_MIN_FIELDS]


def kcal(counts, limit=2):
    for k in np.arange(3, 30.01, 0.5):
        c = np.array([cc(k) for cc in counts])
        if np.percentile(c, 95) <= limit: return float(k)
    return 30.0


def main():
    p = pd.read_csv(V85 / 'abn_points_all632_k6_out.csv', dtype={'date': str}); t = pd.read_csv(V85 / 'E4_counts_all632.csv', dtype={'date': str, 'board': str})
    bm = t.groupby(['date', 'board']).abn6_out.mean().rename('bmean').reset_index(); t = t.merge(bm, on=['date', 'board'])
    good = t[t.bmean < 50]; nonblank = good[good.group != 'blank']; hs = hotspots(p[p.fid.isin(nonblank.fid)]); print('hotspots from non-blank fields', len(hs), 'fields', len(nonblank))
    hs.to_csv(V85 / 'E3_hotspots_from_nonblank.csv', index=False); tree = cKDTree(hs[['x', 'y']].values)
    area_frac = float(len(hs) * np.pi * R_MASK ** 2 / (2048 * 2044)); out = dict(n_hotspots=len(hs), mask_area_fraction=area_frac, R_MASK=R_MASK)
    # ---- blank fields (independent) ----
    blanks = t[(t.group == 'blank')]; rows = []; Z5 = []; ZP = []
    for r in blanks.itertuples():
        z = np.load(V84 / 'cache' / f'{r.fid}.npz'); ctr = z['ctr'].astype(np.float64); ex = z['excl']; d, _ = tree.query(ctr); inh = d < R_MASK
        z5 = z['z5'].astype(np.float64); okL = np.isfinite(z5); zp = z['z5p'].astype(np.float64); okP = np.isfinite(zp); abn = z['abn'] & okP
        Z5.append((z5[okL & ex], inh[okL & ex])); ZP.append((zp[okP & ex], inh[okP & ex], abn[okP & ex]))
        rows.append(dict(fid=r.fid, n_valid=int((okL & ex).sum()), n_in_mask=int((okL & ex & inh).sum()), abn6_in=int((abn & ex & inh & (abs(zp) > 6)).sum()), abn6_out=int((abn & ex & ~inh & (abs(zp) > 6)).sum()),
                         **{f'ev{k}_in': int((okL & ex & inh & (abs(z5) > k)).sum()) for k in (4, 6, 8, 10)}, **{f'ev{k}_out': int((okL & ex & ~inh & (abs(z5) > k)).sum()) for k in (4, 6, 8, 10)}))
    b = pd.DataFrame(rows); b.to_csv(V85 / 'E3_blank_hotspot_effect.csv', index=False)
    out['blank_area_frac_valid_in_mask'] = float(b.n_in_mask.sum() / b.n_valid.sum())
    out['blank_abn6_per_field'] = dict(inside=float(b.abn6_in.mean()), outside=float(b.abn6_out.mean()), total=float((b.abn6_in + b.abn6_out).mean()))
    for k in (4, 6, 8, 10): out[f'blank_events_|z|>{k}'] = dict(inside_per_field=float(b[f'ev{k}_in'].mean()), outside_per_field=float(b[f'ev{k}_out'].mean()), share_inside=float(b[f'ev{k}_in'].sum() / (b[f'ev{k}_in'].sum() + b[f'ev{k}_out'].sum())))
    # calibrated k (P95<=2, positive side) before / after masking the hotspots
    for side, nm in ((1, 'pos'), (-1, 'neg')):
        before = [(lambda k, z=z, m=m: int((z * side > k).sum())) for z, m in Z5]; after = [(lambda k, z=z, m=m: int((z[~m] * side > k).sum())) for z, m in Z5]
        out[f'k_P95le2_{nm}'] = dict(before=kcal(before), after=kcal(after))
    # fp at fixed k
    for k in (4, 6, 8, 9.5):
        out[f'fp_pos_k{k}'] = dict(before=float(np.mean([(z > k).sum() for z, m in Z5])), after=float(np.mean([(z[~m] > k).sum() for z, m in Z5])))
    # ---- new data: -0/-3 (24 fields) and -0/-1 ----
    rows = []
    for bd in (1, 2, 3):
        for pos in range(1, 9):
            for kind in (1, 3):
                z = np.load(R83 / 'pairs' / f'{bd}-{pos}_{kind}.npz'); S5 = z['S5'].astype(float); S5P = z['S5P'].astype(float); ctr = z['ctr_xy'].astype(float); sd = z['sd_periods']; ex = np.isfinite(sd) & (sd > 2)
                okL = np.isfinite(S5); okP = np.isfinite(S5P); v = S5[okL & ex]; med = np.median(v); sig = 1.4826 * np.median(abs(v - med)); d, _ = tree.query(ctr); inh = d < R_MASK
                zz = (S5 - med) / sig; zpp = (S5P - med) / sig; abn = okP & ~okL
                rows.append(dict(board=bd, pos=pos, kind=f'-0/-{kind}', abn6_in=int((abn & ex & inh & (abs(zpp) > 6)).sum()), abn6_out=int((abn & ex & ~inh & (abs(zpp) > 6)).sum()), n_in=int((okL & ex & inh).sum()), n_all=int((okL & ex).sum()),
                                 **{f'pos{k}_all': int((okL & ex & (zz > k)).sum()) for k in (4, 6, 8, 10.5)}, **{f'pos{k}_masked': int((okL & ex & ~inh & (zz > k)).sum()) for k in (4, 6, 8, 10.5)},
                                 **{f'two{k}_all': int((okL & ex & (abs(zz) > k)).sum()) for k in (4, 6, 8, 10.5)}, **{f'two{k}_masked': int((okL & ex & ~inh & (abs(zz) > k)).sum()) for k in (4, 6, 8, 10.5)}))
    n = pd.DataFrame(rows); n.to_csv(V85 / 'E3_new_hotspot_effect.csv', index=False)
    for kind, g in n.groupby('kind'):
        out[f'new_{kind}'] = dict(abn6_in=float(g.abn6_in.mean()), abn6_out=float(g.abn6_out.mean()), valid_in_mask_frac=float(g.n_in.sum() / g.n_all.sum()),
                                 **{f'pos{k}_all->masked': (float(g[f'pos{k}_all'].mean()), float(g[f'pos{k}_masked'].mean())) for k in (4, 6, 8, 10.5)}, **{f'two{k}_all->masked': (float(g[f'two{k}_all'].mean()), float(g[f'two{k}_masked'].mean())) for k in (4, 6, 8, 10.5)})
    json.dump(out, open(V85 / 'E3_hotspot_effect.json', 'w'), indent=1, ensure_ascii=False, default=float); print(json.dumps(out, indent=0, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
