"""v85: camera-fixed hotspots in the POST frame.  Abnormal pillars cluster tightly in the post-image coordinates (median 4.4 px from the centre vs 21 px in the pre frame, new data) -> the defect is a fixed
camera-coordinate region where the pillar fit fails.  Here: (1) post-frame position of each pre pillar = ctr @ Llin.T + tL (v70 fields2) for all 632 fields; (2) hotspots (post frame) from non-blank non-outlier
fields only; (3) mask = pillars whose pre position OR post position is within R px of a hotspot centre; (4) effect on the 69 blank fields (independent) and on the 24 new -0/-3 and -0/-1 fields (independent).
usage: python v85_hotspot_post.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
for f in ('Yu Gothic', 'Meiryo', 'MS Gothic'):
    if any(f in x.name for x in font_manager.fontManager.ttflist): plt.rcParams['font.family'] = f; break
plt.rcParams['axes.unicode_minus'] = False
V70 = ROOT / 'data/results/v70_ledger_center_fit'; V84 = ROOT / 'data/results/v84_1fM_strategies'; V85 = ROOT / 'data/results/v85_abnormal_pillars'; R83 = ROOT / 'data/results/v83_new_shots_261008'
VAULT = Path(r'W:\GoogleDrive\chuya2816\Obsidian Vault\ラボノート\05_解析\261008_【未読】偽陽性ゼロ化_原因確定と標準経路\図と表')
N_MIN_FIELDS = 10


def cluster(points):
    X = points[['x', 'y']].values; tr = cKDTree(X); pr = tr.query_pairs(10.0, output_type='ndarray'); n = len(X)
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)); _, comp = connected_components(g, directed=False); sizes = np.bincount(comp)
    q = points.assign(c=np.where(sizes[comp] >= 8, comp, -1)); q = q[q.c >= 0]
    cl = q.groupby('c').agg(x=('x', 'mean'), y=('y', 'mean'), n=('x', 'size'), nf=('fid', 'nunique'), sdx=('x', 'std'), sdy=('y', 'std')).reset_index()
    return cl[cl.nf >= N_MIN_FIELDS].reset_index(drop=True)


def main():
    t = pd.read_csv(V85 / 'E4_counts_all632.csv', dtype={'date': str, 'board': str}); bm = t.groupby(['date', 'board']).abn6_out.mean().rename('bmean').reset_index(); t = t.merge(bm, on=['date', 'board'])
    meta = t.set_index('fid'); pts_post = []; pts_pre = []
    cache = {}
    for r in t.itertuples():
        z = np.load(V84 / 'cache' / f'{r.fid}.npz'); z2 = np.load(V70 / 'fields2' / f'{r.fid}.npz'); ctr = z['ctr'].astype(np.float64)
        post = ctr @ z2['Llin'].T + z2['tL']; okP = np.isfinite(z['z5p']); abn = z['abn'] & okP & z['excl'] & (np.abs(z['z5p']) > 6)
        cache[r.fid] = (ctr, post)
        for i in np.where(abn)[0]: pts_post.append((r.fid, post[i, 0], post[i, 1])); pts_pre.append((r.fid, ctr[i, 0], ctr[i, 1]))
    pp = pd.DataFrame(pts_post, columns=['fid', 'x', 'y']); pr_ = pd.DataFrame(pts_pre, columns=['fid', 'x', 'y'])
    good = t[t.bmean < 50]; nb = good[good.group != 'blank']
    hs_post = cluster(pp[pp.fid.isin(nb.fid)]); hs_pre = cluster(pr_[pr_.fid.isin(nb.fid)])
    print('hotspots post', len(hs_post), 'pre', len(hs_pre)); hs_post.to_csv(V85 / 'E3_hotspots_post_from_nonblank.csv', index=False)
    print(hs_post.round(1).to_string())
    # spread of the members around the centre: post vs pre
    out = dict(n_hot_post=len(hs_post), n_hot_pre=len(hs_pre), median_sd_post=float(np.nanmedian(np.hypot(hs_post.sdx, hs_post.sdy))), median_sd_pre=float(np.nanmedian(np.hypot(hs_pre.sdx, hs_pre.sdy))))
    tree = cKDTree(hs_post[['x', 'y']].values)
    res = {}
    for R in (15, 20, 30, 40):
        rows = []
        for r in t[t.group == 'blank'].itertuples():
            z = np.load(V84 / 'cache' / f'{r.fid}.npz'); ctr, post = cache[r.fid]; ex = z['excl']; z5 = z['z5'].astype(np.float64); zp = z['z5p'].astype(np.float64); okL = np.isfinite(z5); okP = np.isfinite(zp); abn = z['abn'] & okP
            inh = (tree.query(ctr)[0] < R) | (tree.query(post)[0] < R)
            rows.append(dict(n_valid=int((okL & ex).sum()), n_masked=int((okL & ex & inh).sum()), abn6_in=int((abn & ex & inh & (np.abs(zp) > 6)).sum()), abn6_out=int((abn & ex & ~inh & (np.abs(zp) > 6)).sum()),
                             fp_pos_k4_before=int((okL & ex & (z5 > 4)).sum()), fp_pos_k4_after=int((okL & ex & ~inh & (z5 > 4)).sum()), fp_pos_k8_before=int((okL & ex & (z5 > 8)).sum()), fp_pos_k8_after=int((okL & ex & ~inh & (z5 > 8)).sum()),
                             ev6_in=int((okL & ex & inh & (np.abs(z5) > 6)).sum()), ev6_out=int((okL & ex & ~inh & (np.abs(z5) > 6)).sum()), ev8_in=int((okL & ex & inh & (np.abs(z5) > 8)).sum()), ev8_out=int((okL & ex & ~inh & (np.abs(z5) > 8)).sum()),
                             zpos=z5[okL & ex], zmask=inh[okL & ex]))
        b = pd.DataFrame(rows)
        def kcal(side):
            def cnt(k, masked): return np.array([int((r.zpos * side > k)[(~r.zmask) if masked else slice(None)].sum()) for r in b.itertuples()])
            kb = ka = 30.0
            for k in np.arange(3, 30.01, 0.5):
                if np.percentile(cnt(k, False), 95) <= 2: kb = float(k); break
            for k in np.arange(3, 30.01, 0.5):
                if np.percentile(cnt(k, True), 95) <= 2: ka = float(k); break
            return kb, ka
        kp, kn = kcal(1), kcal(-1)
        res[f'R{R}'] = dict(mask_area_frac=float(b.n_masked.sum() / b.n_valid.sum()), abn6_per_field_in=float(b.abn6_in.mean()), abn6_per_field_out=float(b.abn6_out.mean()), share_abn_inside=float(b.abn6_in.sum() / (b.abn6_in.sum() + b.abn6_out.sum())),
                            share_ev6_inside=float(b.ev6_in.sum() / (b.ev6_in.sum() + b.ev6_out.sum())), share_ev8_inside=float(b.ev8_in.sum() / (b.ev8_in.sum() + b.ev8_out.sum())),
                            fp_pos_k4=(float(b.fp_pos_k4_before.mean()), float(b.fp_pos_k4_after.mean())), fp_pos_k8=(float(b.fp_pos_k8_before.mean()), float(b.fp_pos_k8_after.mean())), k_P95le2_pos=kp, k_P95le2_neg=kn)
    out['blank_effect'] = res
    # new data (posP available): hotspots from the dev data (post frame), independent
    Rn = 20; rows = []
    for b_ in (1, 2, 3):
        for p_ in range(1, 9):
            for kind in (1, 3):
                z = np.load(R83 / 'pairs' / f'{b_}-{p_}_{kind}.npz'); S5 = z['S5'].astype(float); S5P = z['S5P'].astype(float); ctr = z['ctr_xy'].astype(float); pos = z['posP'].astype(float); sd = z['sd_periods']; ex = np.isfinite(sd) & (sd > 2)
                okL = np.isfinite(S5); okP = np.isfinite(S5P); v = S5[okL & ex]; med = np.median(v); sig = 1.4826 * np.median(abs(v - med)); zz = (S5 - med) / sig; zpp = (S5P - med) / sig; abn = okP & ~okL
                inh = (tree.query(ctr)[0] < Rn) | (tree.query(np.nan_to_num(pos))[0] < Rn)
                rows.append(dict(board=b_, pos=p_, kind=f'-0/-{kind}', abn6_in=int((abn & ex & inh & (abs(zpp) > 6)).sum()), abn6_out=int((abn & ex & ~inh & (abs(zpp) > 6)).sum()), n=int((okL & ex).sum()), masked=int((okL & ex & inh).sum()),
                                 **{f'two{k}_before': int((okL & ex & (abs(zz) > k)).sum()) for k in (4, 6, 8, 10.5)}, **{f'two{k}_after': int((okL & ex & ~inh & (abs(zz) > k)).sum()) for k in (4, 6, 8, 10.5)},
                                 **{f'pos{k}_before': int((okL & ex & (zz > k)).sum()) for k in (4, 6, 8, 10.5)}, **{f'pos{k}_after': int((okL & ex & ~inh & (zz > k)).sum()) for k in (4, 6, 8, 10.5)},
                                 **{f'abn_all{k}_before': int((abn & ex & (abs(zpp) > k)).sum()) for k in (6, 10.5)}, **{f'abn_all{k}_after': int((abn & ex & ~inh & (abs(zpp) > k)).sum()) for k in (6, 10.5)}))
    n = pd.DataFrame(rows); n.to_csv(V85 / 'E3_new_hotspot_post_effect.csv', index=False)
    for kind, g in n.groupby('kind'):
        out[f'new_{kind}_R{Rn}'] = {c: float(g[c].mean()) for c in g.columns if c not in ('board', 'pos', 'kind')}
        out[f'new_{kind}_R{Rn}']['masked_area_frac'] = float(g.masked.sum() / g.n.sum())
        out[f'new_{kind}_R{Rn}']['fields_le2_two10.5_after'] = float((g['two10.5_after'] <= 2).mean()); out[f'new_{kind}_R{Rn}']['fields_le2_two10.5_before'] = float((g['two10.5_before'] <= 2).mean())
    json.dump(out, open(V85 / 'E3_hotspot_post_effect.json', 'w'), indent=1, ensure_ascii=False, default=float); print(json.dumps(out, indent=0, ensure_ascii=False, default=float)[:6000])
    # figure: hotspot centres (post frame), with members
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.6))
    for a, (pts, hs, ttl) in zip(ax, ((pr_[pr_.fid.isin(nb.fid)], hs_pre, '前の像の座標'), (pp[pp.fid.isin(nb.fid)], hs_post, '後ろの像の座標'))):
        a.scatter(pts.x, pts.y, s=3, c='#d62728', alpha=.35); a.scatter(hs.x, hs.y, s=90, facecolors='none', edgecolors='k'); a.set_xlim(0, 2048); a.set_ylim(2044, 0); a.set_aspect('equal'); a.set_title(f'異常ピラー(6σ超、傷の外)の位置:{ttl}')
    plt.tight_layout(); plt.savefig(VAULT / 'E3_固定欠陥_前後の座標.png', dpi=100); plt.close()


if __name__ == '__main__':
    main()
