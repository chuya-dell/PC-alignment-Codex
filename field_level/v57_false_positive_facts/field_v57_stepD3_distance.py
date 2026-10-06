"""v57 step D3: per-pillar distance to the scar mask edge (periods of 7.286 px, nearest of all scars) and to the image edge (px).
Inputs: stored per-pillar differences (xy in the pre-image frame), masks from D1/D2. No registration / photometry is re-run."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2
import field_v57_common as C

PERIOD = 7.286
OUTD = C.OUT/'stepD'
H, W = 2044, 2048

def one(fid):
    z = np.load(OUTD/f'{fid}_mask.npz')
    shape = tuple(z['shape']); m = np.unpackbits(z['mask'])[:shape[0]*shape[1]].reshape(shape).astype(bool)
    dt = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
    d, xy, ids = C.load_cache(fid)
    xi = np.clip(np.rint(xy[:, 0]).astype(int), 0, shape[1]-1); yi = np.clip(np.rint(xy[:, 1]).astype(int), 0, shape[0]-1)
    dist_px = dt[yi, xi]; in_mask = m[yi, xi]
    edge_px = np.minimum.reduce([xy[:, 0], shape[1]-1-xy[:, 0], xy[:, 1], shape[0]-1-xy[:, 1]])
    np.savez_compressed(OUTD/f'{fid}_pillar_dist.npz', dist_per=(dist_px/PERIOD).astype(np.float32), edge_px=edge_px.astype(np.float32), in_mask=in_mask)
    return dict(fid=fid, n=len(d), n_in_mask=int(in_mask.sum()), mask_fraction=float(m.mean()),
                n_dist_le5=int(((dist_px/PERIOD) <= 5).sum()), n_dist_gt20=int(((dist_px/PERIOD) > 20).sum()))

if __name__ == '__main__':
    det = pd.read_csv(OUTD/'detection_results.csv')
    ok = det[det.status == 'ok'].fid.tolist()
    with Pool(12) as p:
        rs = p.map(one, ok, chunksize=4)
    o = pd.DataFrame(rs); o.to_csv(OUTD/'pillar_distance_summary.csv', index=False)
    print(len(o), 'fields; pillars in mask total', int(o.n_in_mask.sum()), ' median in-mask fraction %.3f' % (o.n_in_mask/o.n).median())
