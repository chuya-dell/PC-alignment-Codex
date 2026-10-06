"""v57 step E (signed distance): exceedance by signed distance to the scar-mask edge; negative = depth inside the mask (periods). Stored values + masks only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2
import field_v57_common as C

OUTD = C.OUT/'stepD'; OUTE = C.OUT/'stepE'; PERIOD = 7.286
thr = pd.read_csv(C.OUT/'stepB'/'B2_blank_threshold_by_date.csv', dtype={'date': str}).set_index('date')
EDGES = np.r_[-np.inf, np.arange(-30, 0), np.arange(0, 51), np.inf]   # period bins; [-inf,-30), [-30,-29)..[-1,0), [0,1)... [49,50), [50,inf)

def one(fid):
    date = fid.split('_')[0]
    z = np.load(OUTD/f'{fid}_mask.npz'); shape = tuple(z['shape'])
    m = np.unpackbits(z['mask'])[:shape[0]*shape[1]].reshape(shape).astype(bool)
    dt_out = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5); dt_in = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    d, xy, ids = C.load_cache(fid)
    xi = np.clip(np.rint(xy[:, 0]).astype(int), 0, shape[1]-1); yi = np.clip(np.rint(xy[:, 1]).astype(int), 0, shape[0]-1)
    inm = m[yi, xi]
    signed = np.where(inm, -dt_in[yi, xi], dt_out[yi, xi])/PERIOD
    edge = np.minimum.reduce([xy[:, 0], shape[1]-1-xy[:, 0], xy[:, 1], shape[0]-1-xy[:, 1]])
    em = d > thr.loc[date, 'thr_mean']
    idx = np.digitize(signed, EDGES)-1
    nb = len(EDGES)-1
    rows = []
    for e3, sel_e in (('all', np.ones(len(d), bool)), ('e300', edge >= 300)):
        n = np.bincount(idx[sel_e], minlength=nb); k = np.bincount(idx[sel_e], weights=em[sel_e].astype(float), minlength=nb)
        sd = np.array([d[sel_e & (idx == b)].std() if (sel_e & (idx == b)).sum() > 30 else np.nan for b in range(nb)])
        for b in range(nb): rows.append(dict(fid=fid, edge=e3, b=b, lo=EDGES[b], n=int(n[b]), k=int(k[b]), sd=sd[b]))
    return rows

if __name__ == '__main__':
    ok = pd.read_csv(OUTD/'pillar_distance_summary.csv').fid.tolist()
    with Pool(12) as p: res = p.map(one, ok, chunksize=4)
    pd.DataFrame([r for rs in res for r in rs]).to_csv(OUTE/'bin_counts_signed_by_field.csv', index=False)
    print('done')
