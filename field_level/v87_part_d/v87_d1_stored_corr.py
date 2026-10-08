"""v87 D1: the 8 fields whose re-run current-standard delta correlates < 0.99 with the stored delta (cached_field_differences).
Hypothesis: the stored run used a (slightly) different registration translation, and the integer-rounded 3x3 reading (S0) is very sensitive to that.  Test: re-read the post image with the re-run affine
shifted by (dx,dy) on a grid and find the shift that best reproduces the stored delta (same ids).  A 'recovered' correlation >= 0.999 at a small shift explains the difference by the registration translation.
Also report: ids identical?, delta SD, and the share of stored pillars whose rounded sampling position (post) differs between stored and re-run.   usage: python v87_d1_stored_corr.py [--workers K]"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit'))
import numpy as np, pandas as pd, cv2
from shared import registration as reg
from shared.lattice_indexing import lattice_from_fft, grid_coordinates
import v70_lib as L
V70 = ROOT / 'data/results/v70_ledger_center_fit'; OUT = ROOT / 'data/results/v87_part_d'; OUT.mkdir(parents=True, exist_ok=True)
FIDS = ['260829_5_7', '260825_4_3', '260923_7_5', '260827_5_7', '260827_8_4', '260827_8_1', '260923_6_4', '260923_9_5']


def one(fid):
    cv2.setNumThreads(1)
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}).set_index('fid'); rec = led.loc[fid]
    z = np.load(V70 / 'fields' / f'{fid}.npz'); A = z['matrix']
    with np.load(L.CACHE / f'{fid}.npz') as s: sd, sids = s['delta'], s['ids']
    pre = L.read(rec.pre_path); post = L.read(rec.post_path)
    lattice = lattice_from_fft(pre, L.PITCH); ids, xy = grid_coordinates(lattice, pre.shape[1], pre.shape[0], margin=30)
    aa = reg.sample_contrast(pre, xy); valid_a = aa.valid_sampling.to_numpy(); ca = aa.contrast.to_numpy()
    # re-run delta at the re-run affine
    def delta_at(shift):
        postxy = xy @ A[:, :2].T + A[:, 2] + np.asarray(shift)
        bb = reg.sample_contrast(post, postxy); valid = valid_a & bb.valid_sampling.to_numpy(); return ids[valid], ca[valid] - bb.contrast.to_numpy()[valid]
    ids0, d0 = delta_at((0, 0))
    key = lambda a: a[:, 0].astype(np.int64) * 100000 + a[:, 1].astype(np.int64)
    ks, k0 = key(sids), key(ids0)
    common = np.intersect1d(ks, k0); i_s = np.searchsorted(np.sort(ks), common) if False else None
    ms = pd.Series(sd, index=ks); m0 = pd.Series(d0, index=k0)
    cm = ms.index.intersection(m0.index); c0 = float(np.corrcoef(ms.loc[cm], m0.loc[cm])[0, 1])
    # grid search of the translation that best reproduces the stored delta
    best = (c0, 0.0, 0.0)
    def score(shift):
        i_, d_ = delta_at(shift); m = pd.Series(d_, index=key(i_)); cc = ms.index.intersection(m.index); return float(np.corrcoef(ms.loc[cc], m.loc[cc])[0, 1])
    for dx in np.arange(-1.5, 1.5001, 0.25):
        for dy in np.arange(-1.5, 1.5001, 0.25):
            c = score((dx, dy))
            if c > best[0]: best = (c, dx, dy)
    c1, bx, by = best
    for dx in np.arange(bx - 0.2, bx + 0.2001, 0.05):
        for dy in np.arange(by - 0.2, by + 0.2001, 0.05):
            c = score((dx, dy))
            if c > best[0]: best = (c, dx, dy)
    return dict(fid=fid, corr_rerun=c0, corr_best=best[0], best_dx=best[1], best_dy=best[2], shift_px=float(np.hypot(best[1], best[2])), n_common=int(len(cm)), n_stored=int(len(ms)), n_rerun=int(len(m0)), ids_same=bool(len(ms) == len(m0)), sd_stored=float(np.std(sd)), sd_rerun=float(np.std(d0)),
                A_translation=A[:, 2].tolist(), A_linear=A[:, :2].tolist())


if __name__ == '__main__':
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 8
    from multiprocessing import Pool
    with Pool(workers) as p: rows = p.map(one, FIDS)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'D1_stored_corr.csv', index=False); pd.set_option('display.width', 250); print(t.drop(columns=['A_translation', 'A_linear']).round(4).to_string())
