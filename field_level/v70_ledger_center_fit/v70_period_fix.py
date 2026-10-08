"""v70 period fix (after independent critique): the integer lattice period of the pre->post translation was anchored on the feature-based affine at the
image centre; where the affine is wrong this pairs a pillar with a neighbour (71/632 fields in the critique).  The integer shift is decided by the
pillar-to-pillar similarity of high-pass pixel values: values at pre pillar centres vs values at the mapped post positions for integer index shifts
(-R..R)^2 (post basis); a shift is accepted when it raises the correlation by >0.1 and the best correlation >= 0.3 (otherwise: unresolved, zero shift).
Reads fields2_prefix/ (original), writes corrected fields2/ (jL, dL, consL, tL updated) and period_fix.csv.  usage: python v70_period_fix.py [--workers K]"""
from __future__ import annotations
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit'))
import numpy as np, pandas as pd, cv2
from scipy import ndimage
import v70_lib as L
V70 = ROOT / 'data/results/v70_ledger_center_fit'
SRC = V70 / 'fields2_prefix'; DST = V70 / 'fields2'
R = 5; RBIG = 12
W, H = 2048, 2044


def key(ids): return (ids[:, 0].astype(np.int64) + 5000) * 20000 + (ids[:, 1].astype(np.int64) + 5000)


def scan(va, pc, ca_post, Ll, tL, bas, rng_):
    res = {}
    for n1 in range(-rng_, rng_ + 1):
        for n2 in range(-rng_, rng_ + 1):
            q = pc @ Ll.T + (tL + n1 * bas[0] + n2 * bas[1])
            ok = (q[:, 0] > 10) & (q[:, 0] < W - 10) & (q[:, 1] > 10) & (q[:, 1] < H - 10)
            if ok.sum() < 5000: res[(n1, n2)] = -9; continue
            vb = ndimage.map_coordinates(ca_post, [q[ok, 1], q[ok, 0]], order=1, mode='nearest')
            res[(n1, n2)] = float(np.corrcoef(va[ok], vb)[0, 1])
    return res


def one(rec):
    fid = rec['fid']
    z = np.load(SRC / f'{fid}.npz'); A = {k: z[k] for k in z.files}
    a = L.read(rec['pre_path']); b = L.read(rec['post_path'])
    ca = a - cv2.GaussianBlur(a, (0, 0), 8); cb = b - cv2.GaussianBlur(b, (0, 0), 8)
    pc = A['pre_ctr'].astype(float); Ll = A['Llin']; tL = A['tL']; bas = A['post_model'][1:]       # rows: post lattice vectors
    va = ndimage.map_coordinates(ca, [pc[:, 1], pc[:, 0]], order=1, mode='nearest')
    res = scan(va, pc, cb, Ll, tL, bas, R)
    best = max(res, key=res.get); c0 = res[(0, 0)]; cbest = res[best]
    if cbest < 0.5 or best[0] in (-R, R) or best[1] in (-R, R):        # large errors: widen the scan (coarsely on the pre amplitude-weighted pillars)
        res2 = scan(va, pc, cb, Ll, tL, bas, RBIG)
        b2 = max(res2, key=res2.get)
        if res2[b2] > cbest: best, cbest = b2, res2[b2]
    shift = best if (cbest - c0 > 0.1 and cbest >= 0.3) else (0, 0)
    resolved = cbest >= 0.3
    # post-index-space shift: the scan used T = tL + n1*bas[0] + n2*bas[1]; bas rows are post lattice vectors, so an index shift (n1,n2) in the post basis
    tL2 = tL + shift[0] * bas[0] + shift[1] * bas[1]
    # new partner: nearest post lattice node (by predicted lattice position) to the corrected expected position
    from scipy.spatial import cKDTree
    expL2 = A['pre_pred_lin'] @ Ll.T + tL2
    d2, j2 = cKDTree(A['post_pred_lin']).query(expL2)
    pre_ids = A['pre_ids']; post_ids = A['post_ids']
    U = A['U']; di = post_ids[j2] - pre_ids @ U.T
    vals, cnt = np.unique(di[d2 < 1.8], axis=0, return_counts=True); mode = vals[np.argmax(cnt)]
    cons = np.all(di == mode, axis=1) & (d2 < 2.0)
    A['jL'] = j2.astype(np.int32); A['dL'] = d2.astype(np.float32); A['tL'] = tL2; A['consL'] = cons; A['period_shift'] = np.array(shift)
    np.savez_compressed(DST / f'{fid}.npz', **A)
    return dict(fid=fid, shift_a=shift[0], shift_b=shift[1], shift_px=float(np.linalg.norm(shift[0] * bas[0] + shift[1] * bas[1])), corr_zero=c0, corr_best=cbest, best_a=best[0], best_b=best[1],
                resolved=resolved, cons_frac=float(cons.mean()))


def main():
    DST.mkdir(exist_ok=True)
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str})
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 6
    from multiprocessing import Pool
    with Pool(workers) as p: res = p.map(one, led.to_dict('records'), chunksize=2)
    t = pd.DataFrame(res); t.to_csv(V70 / 'period_fix.csv', index=False)
    sh = (t.shift_a != 0) | (t.shift_b != 0)
    print('fields', len(t), 'shifted', int(sh.sum()), 'unresolved(corr_best<0.3)', int((~t.resolved).sum()))
    print(t[['corr_zero', 'corr_best']].describe().loc[['25%', '50%', '75%']].round(3)); print('shift px', t[sh].shift_px.describe().round(1).to_dict())


if __name__ == '__main__':
    main()
