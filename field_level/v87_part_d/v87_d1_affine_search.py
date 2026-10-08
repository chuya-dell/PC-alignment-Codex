"""v87 D1 (3): the re-run standard registration is deterministic (identical matrix in 3 runs), so the stored (2026-09-28) delta of the 8 fields must come from a different registration estimate.
Search a small affine correction (rotation theta, isotropic scale s, translation dx,dy) of the re-run matrix that best reproduces the stored delta (integer-rounded 3x3 box contrast reading, same as S0).
Coarse grid on a 8000-pillar subset, then a fine grid on all pillars around the best.  If a small correction gives corr >= 0.99, the difference is explained by the registration matrix (and we report it).
usage: python v87_d1_affine_search.py [--workers K]"""
import sys, itertools
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit')); sys.path.insert(0, str(ROOT / 'field_level/v72_real_field_readout'))
import numpy as np, pandas as pd, cv2
from shared.lattice_indexing import lattice_from_fft, grid_coordinates
import v70_lib as L
import v72_readout as R
V70 = ROOT / 'data/results/v70_ledger_center_fit'; OUT = ROOT / 'data/results/v87_part_d'
FIDS = ['260829_5_7', '260825_4_3', '260923_7_5', '260827_5_7', '260827_8_4', '260827_8_1', '260923_6_4', '260923_9_5']
H, W = 2044, 2048; c0 = np.array([W / 2, H / 2])


def read_box(cb, xy):
    q = np.rint(xy).astype(int); ok = (q[:, 0] >= 1) & (q[:, 0] < W - 1) & (q[:, 1] >= 1) & (q[:, 1] < H - 1); v = np.full(len(xy), np.nan, np.float32)
    v[ok] = cb[q[ok, 1], q[ok, 0]]; return v


def one(fid):
    cv2.setNumThreads(1)
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}).set_index('fid'); rec = led.loc[fid]
    A = np.load(V70 / 'fields' / f'{fid}.npz')['matrix']
    with np.load(L.CACHE / f'{fid}.npz') as s: sd, sids = s['delta'], s['ids']
    pre = L.read(rec.pre_path); post = L.read(rec.post_path)
    lattice = lattice_from_fft(pre, L.PITCH); ids, xy = grid_coordinates(lattice, W, H, margin=30)
    ca = read_box(R.box_contrast(pre), xy); cbm = R.box_contrast(post)
    key = lambda a: a[:, 0].astype(np.int64) * 100000 + a[:, 1].astype(np.int64)
    ms = pd.Series(sd, index=key(sids)); st = ms.reindex(key(ids)).values; have = np.isfinite(st)
    rng = np.random.default_rng(0); sub = rng.choice(np.where(have)[0], 8000, replace=False)
    def corr(params, idx):
        th, sc, dx, dy = params; c, s_ = np.cos(th), np.sin(th); Rm = np.array([[c, -s_], [s_, c]]) * (1 + sc)
        # affine correction about the image centre applied to the post coordinates
        p0 = xy[idx] @ A[:, :2].T + A[:, 2]; p = (p0 - c0) @ Rm.T + c0 + np.array([dx, dy])
        d = ca[idx] - read_box(cbm, p); ok = np.isfinite(d)
        return float(np.corrcoef(st[idx][ok], d[ok])[0, 1]) if ok.sum() > 500 else -1
    best = (corr((0, 0, 0, 0), sub), (0, 0, 0, 0)); c_init = best[0]
    for th in np.arange(-0.004, 0.0041, 0.001):
        for sc in np.arange(-0.002, 0.0021, 0.001):
            for dx in np.arange(-1.5, 1.51, 0.25):
                for dy in np.arange(-1.5, 1.51, 0.25):
                    c = corr((th, sc, dx, dy), sub)
                    if c > best[0]: best = (c, (th, sc, dx, dy))
    # fine refinement on the subset then all pillars
    th0, sc0, dx0, dy0 = best[1]
    for th in th0 + np.arange(-0.0006, 0.00061, 0.0002):
        for sc in sc0 + np.arange(-0.0006, 0.00061, 0.0002):
            for dx in dx0 + np.arange(-0.3, 0.301, 0.1):
                for dy in dy0 + np.arange(-0.3, 0.301, 0.1):
                    c = corr((th, sc, dx, dy), sub)
                    if c > best[0]: best = (c, (th, sc, dx, dy))
    allidx = np.where(have)[0]
    return dict(fid=fid, corr_rerun_all=corr((0, 0, 0, 0), allidx), corr_best_subset=best[0], corr_best_all=corr(best[1], allidx), theta_rad=best[1][0], scale=best[1][1], dx=best[1][2], dy=best[1][3],
                corner_shift_px=float(np.linalg.norm(((np.array([[0, 0], [W, 0], [0, H], [W, H]]) - c0) @ (np.array([[np.cos(best[1][0]), -np.sin(best[1][0])], [np.sin(best[1][0]), np.cos(best[1][0])]]) * (1 + best[1][1])).T - (np.array([[0, 0], [W, 0], [0, H], [W, H]]) - c0)) + np.array(best[1][2:]), axis=1).max()))


if __name__ == '__main__':
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 8
    from multiprocessing import Pool
    with Pool(workers) as p: rows = p.map(one, FIDS)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'D1_affine_search.csv', index=False); pd.set_option('display.width', 250); print(t.round(5).to_string())
