"""v85 reproducibility (E3): do the SAME pillars become abnormal when the same field is imaged again?
(1) burst (-0/-1 of 2026-10-08, 24 fields; 260904_p50_repeat), (2) small stage steps (261006 stepping-motor series, 6 pairs; v75 real results), (3) re-mount (-0/-3, 24 fields),
(4) five re-mounts of the SAME field (needle-scar series 6, 6-1..6-4 with pre = 6): pillar identity is the pre-image pillar (same pre image -> same pillar list); abnormal sets are compared across the four pairs,
and in camera (post) coordinates with the hotspot centres.   usage: python v85_repro.py"""
import sys, json, itertools
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v75_zero_truth_pairs', 'field_level/v72_real_field_readout', 'field_level/v70_ledger_center_fit', 'field_level/v60_band_scar_controls', 'field_level/v74_scar_distance'):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd, tifffile, cv2
from scipy.spatial import cKDTree
V85 = ROOT / 'data/results/v85_abnormal_pillars'; R83 = ROOT / 'data/results/v83_new_shots_261008'; NEEDLE = R83 / 'needle'
RAWN = Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu\261008_p50_傷')


def abn_set(S5, S5P, ex=None, k=6.0):
    okL = np.isfinite(S5); okP = np.isfinite(S5P); e = np.ones(len(S5), bool) if ex is None else ex
    v = S5[okL & e]; med = np.median(v); sig = 1.4826 * np.median(abs(v - med)); abn = okP & ~okL & e & (abs((S5P - med) / sig) > k)
    return abn, med, sig


def main():
    hs = pd.read_csv(V85 / 'E3_hotspots_post_from_nonblank.csv'); tree = cKDTree(hs[['x', 'y']].values); out = {}
    # ---- (2) small steps and burst 260904 from v75 real ----
    rows = []
    for f in sorted((ROOT / 'data/results/v75_zero_truth_pairs/real').glob('*.npz')):
        z = np.load(f); S5 = z['S5'].astype(float); S5P = z['S5P'].astype(float); pos = z['posP'].astype(float) if 'posP' in z.files else None
        abn, med, sig = abn_set(S5, S5P)
        rows.append(dict(pair=f.stem, n_abn6=int(abn.sum()), in_hot=int((tree.query(np.nan_to_num(pos[abn]))[0] < 20).sum()) if pos is not None and abn.any() else 0, sigma=sig))
    out['v75_real_pairs'] = rows
    # ---- (1),(3): new data ----
    rows = []
    for b in (1, 2, 3):
        for p in range(1, 9):
            r = dict(board=b, pos=p)
            for kind in (1, 3):
                z = np.load(R83 / 'pairs' / f'{b}-{p}_{kind}.npz'); sd = z['sd_periods']; ex = np.isfinite(sd) & (sd > 2); abn, med, sig = abn_set(z['S5'].astype(float), z['S5P'].astype(float), ex)
                r[f'n_abn_{kind}'] = int(abn.sum()); pos = z['posP'].astype(float); r[f'in_hot_{kind}'] = int((tree.query(np.nan_to_num(pos[abn]))[0] < 20).sum()) if abn.any() else 0
                if kind == 3: r['_set3'] = set(np.where(abn)[0]); r['_pos3'] = pos[abn]
            # same post (-3), different pre (-1): the identity of the abnormal pillar (post-side) must repeat if the post image carries it
            z13 = np.load(R83 / 'pairs_extra' / f'{b}-{p}_one_3.npz'); sd = z13['sd_periods']; ex = np.isfinite(sd) & (sd > 2); abn13, _, _ = abn_set(z13['S5'].astype(float), z13['S5P'].astype(float), ex)
            ctr = np.load(R83 / 'pairs' / f'{b}-{p}_3.npz')['ctr_xy'].astype(float)
            a3 = np.array(sorted(r['_set3'])); a13 = np.where(abn13)[0]
            if len(a3) and len(a13):
                t13 = cKDTree(z13['ctr'].astype(float)[a13]); d, _ = t13.query(ctr[a3]); r['n_overlap_0_3_vs_1_3'] = int((d < 2.5).sum())
            else:
                r['n_overlap_0_3_vs_1_3'] = 0
            r['n_abn_1_3'] = len(a13); r.pop('_set3'); r.pop('_pos3'); rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(V85 / 'E3_repro_new.csv', index=False)
    out['new_burst_-0/-1'] = dict(total_abn=int(t.n_abn_1.sum()), fields=len(t)); out['new_remount_-0/-3'] = dict(total_abn=int(t.n_abn_3.sum()), in_hot=int(t.in_hot_3.sum()), fields=len(t))
    out['new_same_post_pairs'] = dict(abn_0_3=int(t.n_abn_3.sum()), abn_1_3=int(t.n_abn_1_3.sum()), overlap=int(t.n_overlap_0_3_vs_1_3.sum()))
    # ---- (4) needle series: five mounts of the same field ----
    import v75_pair_lib as P
    import v74_scar_distance as E
    names = ['6', '6-1', '6-2', '6-3', '6-4']; res = {}
    def mask_of(n):
        with np.load(NEEDLE / f'align_mask_{n}.npz') as z:
            shp = tuple(z['shape']); f = lambda k: np.unpackbits(z[k])[:np.prod(shp)].reshape(shp).astype(bool); return f('cross') | f('gouge')
    a = tifffile.imread(RAWN / '6.tif').astype(np.float32); m6 = mask_of('6')
    sets = {}; posts = {}
    for n in names[1:]:
        cache = NEEDLE / f'repro_pair_6_{n}.npz'
        if not cache.exists():
            b = tifffile.imread(RAWN / f'{n}.tif').astype(np.float32); r = P.analyse_pair(a, b)
            np.savez_compressed(cache, S5=r['S5'], S5P=r['S5P'], ctr=r['ctr_xy'], posP=r['posP'], okP=r['okP'])
        z = np.load(cache); ctr = z['ctr'].astype(float); sd = E.signed_dist_periods(m6, ctr); ex = sd > 2
        abn, med, sig = abn_set(z['S5'].astype(float), z['S5P'].astype(float), ex); sets[n] = set(np.where(abn)[0]); posts[n] = z['posP'].astype(float)[abn]
        res[n] = dict(n_abn=int(abn.sum()), in_hot_post=int((tree.query(np.nan_to_num(posts[n]))[0] < 20).sum()) if abn.any() else 0, sigma=float(sig))
    ov = {}
    for x, y in itertools.combinations(names[1:], 2): ov[f'{x}&{y}'] = len(sets[x] & sets[y])
    allk = names[1:]; union = set().union(*sets.values()); cnt = pd.Series([sum(i in sets[n] for n in allk) for i in union]).value_counts().sort_index().to_dict()
    nvalid = len(np.load(NEEDLE / 'repro_pair_6_6-1.npz')['ctr'])
    exp = {f'{x}&{y}': len(sets[x]) * len(sets[y]) / nvalid for x, y in itertools.combinations(allk, 2)}
    out['needle_series'] = dict(per_pair=res, pairwise_overlap=ov, expected_by_chance=exp, n_mounts_per_union_pillar=cnt, union=len(union))
    json.dump(out, open(V85 / 'E3_repro.json', 'w'), indent=1, ensure_ascii=False, default=float); print(json.dumps(out, indent=0, ensure_ascii=False, default=float)[:5000])


if __name__ == '__main__':
    main()
