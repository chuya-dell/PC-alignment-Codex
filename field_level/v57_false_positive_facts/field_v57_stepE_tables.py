"""v57 step E (tables): per-field counts of exceeding pillars by distance bin to the scar mask edge, with and without the edge>=300 px restriction,
and by (distance class x image-edge class). Exceedance = difference > date-blank threshold (mean+3SD main; median+3*MADs sensitivity). Read-only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np, pandas as pd
import field_v57_common as C

OUTD = C.OUT/'stepD'; OUTE = C.OUT/'stepE'
NB = 50   # 1-period bins 0..49, bin 50 = >=50
EDGE_CUTS = [0, 100, 200, 300, 500, 1e9]
DIST_CLASSES = [('0-5', 0, 5), ('5-20', 5, 20), ('20+', 20, 1e9)]
thr = pd.read_csv(C.OUT/'stepB'/'B2_blank_threshold_by_date.csv', dtype={'date': str}).set_index('date')

def one(fid):
    date = fid.split('_')[0]
    d, xy, ids = C.load_cache(fid)
    z = np.load(OUTD/f'{fid}_pillar_dist.npz'); dist = z['dist_per']; edge = z['edge_px']; inm = z['in_mask']
    em = d > thr.loc[date, 'thr_mean']; ed = d > thr.loc[date, 'thr_med']
    rows = []
    # in-mask pillars (descriptive only)
    rows.append(dict(fid=fid, b=-1, n_all=int(inm.sum()), kmean_all=int((em & inm).sum()), kmed_all=int((ed & inm).sum()),
                     n_e300=int((inm & (edge >= 300)).sum()), kmean_e300=int((em & inm & (edge >= 300)).sum()), kmed_e300=int((ed & inm & (edge >= 300)).sum())))
    keep = ~inm
    b = np.minimum(np.floor(dist).astype(int), NB)
    for bb in range(NB+1):
        s = keep & (b == bb); s3 = s & (edge >= 300)
        rows.append(dict(fid=fid, b=bb, n_all=int(s.sum()), kmean_all=int((em & s).sum()), kmed_all=int((ed & s).sum()),
                         n_e300=int(s3.sum()), kmean_e300=int((em & s3).sum()), kmed_e300=int((ed & s3).sum())))
    r2 = []
    for name, lo, hi in DIST_CLASSES:
        for j in range(len(EDGE_CUTS)-1):
            s = keep & (dist >= lo) & (dist < hi) & (edge >= EDGE_CUTS[j]) & (edge < EDGE_CUTS[j+1])
            r2.append(dict(fid=fid, dist_class=name, edge_lo=EDGE_CUTS[j], edge_hi=EDGE_CUTS[j+1], n=int(s.sum()), kmean=int((em & s).sum()), kmed=int((ed & s).sum())))
    return rows, r2

if __name__ == '__main__':
    OUTE.mkdir(exist_ok=True)
    ok = pd.read_csv(OUTD/'pillar_distance_summary.csv').fid.tolist()
    with Pool(12) as p:
        res = p.map(one, ok, chunksize=4)
    pd.DataFrame([r for rs, _ in res for r in rs]).to_csv(OUTE/'bin_counts_by_field.csv', index=False)
    pd.DataFrame([r for _, rs in res for r in rs]).to_csv(OUTE/'edge_by_dist_counts_by_field.csv', index=False)
    print('done', len(ok))
