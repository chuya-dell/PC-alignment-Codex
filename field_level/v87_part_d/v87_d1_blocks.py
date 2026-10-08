"""v87 D1 (2): where in the field do stored and re-run deltas disagree?  Block correlation (256 px) between the stored and the re-run S0 delta, the implied local translation (best shift per block in a 0.25-px
search of the 3x3 reading), and a global affine correction fitted from the block shifts.   usage: python v87_d1_blocks.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit'))
import numpy as np, pandas as pd, cv2
from shared import registration as reg
from shared.lattice_indexing import lattice_from_fft, grid_coordinates
import v70_lib as L
V70 = ROOT / 'data/results/v70_ledger_center_fit'; OUT = ROOT / 'data/results/v87_part_d'
FIDS = ['260829_5_7', '260825_4_3', '260923_7_5', '260827_5_7', '260827_8_4', '260827_8_1', '260923_6_4', '260923_9_5']
BL = 256


def one(fid):
    cv2.setNumThreads(1)
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}).set_index('fid'); rec = led.loc[fid]
    z = np.load(V70 / 'fields' / f'{fid}.npz'); A = z['matrix']
    with np.load(L.CACHE / f'{fid}.npz') as s: sd, sids = s['delta'], s['ids']
    pre = L.read(rec.pre_path); post = L.read(rec.post_path)
    lattice = lattice_from_fft(pre, L.PITCH); ids, xy = grid_coordinates(lattice, pre.shape[1], pre.shape[0], margin=30)
    aa = reg.sample_contrast(pre, xy); va = aa.valid_sampling.to_numpy(); ca = aa.contrast.to_numpy()
    key = lambda a: a[:, 0].astype(np.int64) * 100000 + a[:, 1].astype(np.int64); ms = pd.Series(sd, index=key(sids))
    def delta_at(shift):
        postxy = xy @ A[:, :2].T + A[:, 2] + np.asarray(shift); bb = reg.sample_contrast(post, postxy); v = va & bb.valid_sampling.to_numpy()
        return key(ids[v]), xy[v], ca[v] - bb.contrast.to_numpy()[v]
    k0, xy0, d0 = delta_at((0, 0)); s_aligned = ms.reindex(k0).values
    bx = np.clip((xy0[:, 0] // BL).astype(int), 0, 7); by = np.clip((xy0[:, 1] // BL).astype(int), 0, 7); blk = by * 8 + bx
    rows = []
    shifts = [(dx, dy) for dx in np.arange(-1.5, 1.51, 0.5) for dy in np.arange(-1.5, 1.51, 0.5)]
    cache = {s: delta_at(s) for s in shifts}
    for b in range(64):
        sel0 = (blk == b) & np.isfinite(s_aligned)
        if sel0.sum() < 300: continue
        c0 = np.corrcoef(s_aligned[sel0], d0[sel0])[0, 1]; best = (c0, 0, 0)
        for s in shifts:
            kk, xx, dd = cache[s]; m = pd.Series(dd, index=kk).reindex(k0).values; sel = sel0 & np.isfinite(m)
            if sel.sum() < 300: continue
            c = np.corrcoef(s_aligned[sel], m[sel])[0, 1]
            if c > best[0]: best = (c, s[0], s[1])
        rows.append(dict(fid=fid, bx=b % 8, by=b // 8, n=int(sel0.sum()), corr0=c0, corr_best=best[0], dx=best[1], dy=best[2], cx=(b % 8 + .5) * BL, cy=(b // 8 + .5) * BL))
    t = pd.DataFrame(rows); t.to_csv(OUT / f'D1_blocks_{fid}.csv', index=False)
    return dict(fid=fid, corr_global=float(np.corrcoef(s_aligned[np.isfinite(s_aligned)], d0[np.isfinite(s_aligned)])[0, 1]), n_blocks=len(t), corr0_block_median=float(t.corr0.median()), corr0_block_min=float(t.corr0.min()), corr0_block_max=float(t.corr0.max()),
                frac_blocks_corr_gt_0_99=float((t.corr0 > 0.99).mean()), blocks_best_gt_0_99_with_shift=float((t.corr_best > 0.99).mean()))


if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(8) as p: rows = p.map(one, FIDS)
    t = pd.DataFrame(rows); pd.set_option('display.width', 250); print(t.round(4).to_string()); t.to_csv(OUT / 'D1_blocks_summary.csv', index=False)
