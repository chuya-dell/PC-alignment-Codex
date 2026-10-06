import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np, pandas as pd
import field_v57_common as C
OUTD = C.OUT/'stepD'; OUTE = C.OUT/'stepE'
thr = pd.read_csv(C.OUT/'stepB'/'B2_blank_threshold_by_date.csv', dtype={'date': str}).set_index('date')
CUTS = [0, 100, 200, 300, 1e9]
def one(fid):
    date = fid.split('_')[0]; d, xy, ids = C.load_cache(fid); z = np.load(OUTD/f'{fid}_pillar_dist.npz')
    em = d > thr.loc[date, 'thr_mean']; inm = z['in_mask']; dist = z['dist_per']; edge = z['edge_px']
    rows = []
    for j in range(4):
        e = (edge >= CUTS[j]) & (edge < CUTS[j+1])
        for name, s in (('in_mask', inm & e), ('out_0_5', ~inm & (dist < 5) & e), ('out_ge20', ~inm & (dist >= 20) & e)):
            rows.append(dict(fid=fid, region=name, edge_lo=CUTS[j], n=int(s.sum()), k=int((em & s).sum())))
    return rows
if __name__ == '__main__':
    ok = pd.read_csv(OUTD/'pillar_distance_summary.csv').fid.tolist()
    with Pool(12) as p: res = p.map(one, ok, chunksize=4)
    t = pd.DataFrame([r for rs in res for r in rs])
    fl = pd.read_csv(C.OUT/'stepB'/'B3_field_flags.csv', dtype={'date': str})[['fid', 'group', 'outlier29']]
    t = t.merge(fl, on='fid'); t = t[~t.outlier29]
    g = t.groupby(['group', 'region', 'edge_lo']).agg(n=('n', 'sum'), k=('k', 'sum'), fields=('fid', 'nunique')).reset_index(); g['rate'] = g.k/g.n
    g.to_csv(OUTE/'E6_inmask_vs_outside_by_edge_class.csv', index=False)
    pd.set_option('display.width', 250)
    for grp in ('blank', 'analyte', 'mismatch'):
        x = g[g.group == grp]; print(grp); print((100*x.pivot(index='region', columns='edge_lo', values='rate')).round(3).to_string()); print(x.pivot(index='region', columns='edge_lo', values='n').to_string())
