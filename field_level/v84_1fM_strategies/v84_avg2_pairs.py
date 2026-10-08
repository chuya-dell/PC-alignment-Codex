"""v84 (part B-a): real check of frame averaging with the 2026-10-08 reshoot: pre = mean(-0,-1) (same view, nothing touched; N=2 on the pre side) vs -3, and pre = -1 vs -3.
Same C1 route (v75_pair_lib.analyse_pair), same scar masks (from -0).  Output: data/results/v83_new_shots_261008/pairs_extra/{board}-{pos}_{avg01|one}_3.npz.  usage: python v84_avg2_pairs.py"""
import sys, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v75_zero_truth_pairs', 'field_level/v72_real_field_readout', 'field_level/v70_ledger_center_fit', 'field_level/v60_band_scar_controls', 'field_level/v74_scar_distance'):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, cv2, tifffile
import v75_pair_lib as P
import v74_scar_distance as E
RAW = Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu\261008_p50_z'); R = ROOT / 'data/results/v83_new_shots_261008'; (R / 'pairs_extra').mkdir(exist_ok=True)


def one(job):
    b, p, kind = job; dst = R / 'pairs_extra' / f'{b}-{p}_{kind}_3.npz'
    if dst.exists(): return dict(tag=dst.name, ok=True)
    cv2.setNumThreads(1)
    try:
        rd = lambda k: tifffile.imread(RAW / f'{b}-{p}-{k}.tif').astype(np.float32)
        a = (rd(0) + rd(1)) / 2 if kind == 'avg01' else rd(1)
        res = P.analyse_pair(a, rd(3))
        with np.load(R / 'masks' / f'{b}-{p}_mask.npz') as z:
            shp = tuple(z['shape']); mask = np.unpackbits(z['mask'])[:np.prod(shp)].reshape(shp).astype(bool)
        sd = E.signed_dist_periods(mask, res['ctr_xy'])
        np.savez_compressed(dst, S5=res['S5'].astype(np.float32), S5P=res['S5P'].astype(np.float32), ctr=res['ctr_xy'].astype(np.float32), sd_periods=sd.astype(np.float32), A=res['A'])
        return dict(tag=dst.name, ok=True)
    except Exception as e:
        return dict(tag=str(job), ok=False, error=repr(e), trace=traceback.format_exc()[-600:])


if __name__ == '__main__':
    jobs = [(b, p, k) for b in (1, 2, 3) for p in range(1, 9) for k in ('avg01', 'one')]
    from multiprocessing import Pool
    with Pool(8) as pool:
        for r in pool.imap_unordered(one, jobs): print(r, flush=True)
