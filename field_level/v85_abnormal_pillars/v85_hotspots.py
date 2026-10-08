"""v85: abnormal pillars sit at the SAME camera coordinates across fields/boards/days.  Cluster the pooled positions (blank fields, then all non-outlier boards) with DBSCAN, count in how many fields
each hotspot appears, the fraction of abnormal pillars inside hotspots, and compare the hotspot centres with the hot pixels / fixed pattern of the dark frames (15:18, 15:55, 2026-09-04) and with the pre-image
brightness at those positions.   usage: python v85_hotspots.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd, tifffile, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
for f in ('Yu Gothic', 'Meiryo', 'MS Gothic'):
    if any(f in x.name for x in font_manager.fontManager.ttflist): plt.rcParams['font.family'] = f; break
plt.rcParams['axes.unicode_minus'] = False
V85 = ROOT / 'data/results/v85_abnormal_pillars'; RAW = Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu')
VAULT = Path(r'W:\GoogleDrive\chuya2816\Obsidian Vault\ラボノート\05_解析\261008_【未読】偽陽性ゼロ化_原因確定と標準経路\図と表')


def main():
    p = pd.read_csv(V85 / 'abn_points_all632_k6_out.csv', dtype={'date': str}); t = pd.read_csv(V85 / 'E4_counts_all632.csv', dtype={'date': str, 'board': str})
    bm = t.groupby(['date', 'board']).abn6_out.mean().rename('bmean').reset_index(); t = t.merge(bm, on=['date', 'board']); good = t[t.bmean < 50]
    pts = p[p.fid.isin(good.fid)]; blank = pts[pts.group == 'blank']; print('non-outlier fields', len(good), 'points', len(pts), 'blank points', len(blank))
    out = {}
    for nm, q, nf in (('blank', blank, int((good.group == 'blank').sum())), ('all_nonoutlier', pts, len(good))):
        X = q[['x', 'y']].values; tr = cKDTree(X); pr = tr.query_pairs(15.0, output_type='ndarray'); n_ = len(X)
        g_ = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n_, n_)); _, comp = connected_components(g_, directed=False)
        sizes = np.bincount(comp); lab = np.where(sizes[comp] >= 8, comp, -1); q = q.assign(c=lab)
        cl = []
        for c in sorted(set(lab) - {-1}):
            g = q[q.c == c]; cl.append(dict(cluster=c, x=g.x.mean(), y=g.y.mean(), n=len(g), n_fields=g.fid.nunique(), n_dates=g.date.nunique(), radius_p90=float(np.percentile(np.hypot(g.x - g.x.mean(), g.y - g.y.mean()), 90))))
        cl = pd.DataFrame(cl).sort_values('n', ascending=False); cl.to_csv(V85 / f'E3_hotspots_{nm}.csv', index=False)
        out[nm] = dict(n_points=len(q), n_clusters=len(cl), frac_in_clusters=float((lab >= 0).mean()), n_fields_total=nf,
                       top10=cl.head(10).round(1).to_dict('records'), median_fields_per_cluster=float(cl.n_fields.median()), median_dates_per_cluster=float(cl.n_dates.median()))
        # abnormal pillars per field in/out of hotspots
        if nm == 'all_nonoutlier': hot = cl; pts = q
    # hotspot membership of all points in non-outlier fields (using clusters from non-outlier pooled)
    hx = hot[['x', 'y']].values; tree = cKDTree(hx); d, _ = tree.query(pts[['x', 'y']].values); pts = pts.assign(d_hot=d); inhot = pts.d_hot < 25
    per_field = pts.assign(inhot=inhot).groupby('fid').inhot.agg(['sum', 'count']); per_field['out'] = per_field['count'] - per_field['sum']
    out['per_field_mean_in_hotspot'] = float(per_field['sum'].reindex(good.fid).fillna(0).mean()); out['per_field_mean_outside_hotspot'] = float(per_field['out'].reindex(good.fid).fillna(0).mean())
    blk = good[good.group == 'blank'].fid; out['blank_per_field_in_hot'] = float(per_field['sum'].reindex(blk).fillna(0).mean()); out['blank_per_field_out_hot'] = float(per_field['out'].reindex(blk).fillna(0).mean())
    # dark frames: hot pixels and fixed pattern at the hotspot positions
    ims = {n: tifffile.imread(RAW / '261008_ligjt' / f).astype(np.float64) for n, f in (('d1518', '1-1.tif'), ('d1518b', '1-2.tif'), ('d1555', '2-1.tif'), ('d1555b', '2-2.tif'))}
    ims['d0904'] = tifffile.imread(RAW / '260904_p50_暗時' / '1.tif').astype(np.float64); ims['d0904b'] = tifffile.imread(RAW / '260904_p50_暗時' / '2.tif').astype(np.float64)
    dk = np.mean([ims['d1518'], ims['d1518b'], ims['d1555'], ims['d1555b'], ims['d0904'], ims['d0904b']], axis=0)
    dkb = cv2.GaussianBlur(dk, (0, 0), 3) - cv2.GaussianBlur(dk, (0, 0), 30); sd = 1.4826 * np.median(abs(dkb - np.median(dkb)))
    z_dark = []
    for r in hot.itertuples():
        x, y = int(round(r.x)), int(round(r.y)); win = dkb[max(y - 12, 0):y + 13, max(x - 12, 0):x + 13]; z_dark.append(float(np.abs(win).max() / sd))
    hot = hot.assign(dark_absmax_over_sd=z_dark)
    # control: random positions
    rng = np.random.default_rng(3); ctrl = []
    for _ in range(2000):
        x, y = rng.integers(20, 2028), rng.integers(20, 2024); win = dkb[y - 12:y + 13, x - 12:x + 13]; ctrl.append(float(np.abs(win).max() / sd))
    hot['dark_pct_vs_random'] = [float((np.array(ctrl) < v).mean()) for v in hot.dark_absmax_over_sd]
    hot.to_csv(V85 / 'E3_hotspots_vs_dark.csv', index=False)
    out['dark_check'] = dict(n_hot=len(hot), median_pct_vs_random=float(hot.dark_pct_vs_random.median()), frac_above_95pct=float((hot.dark_pct_vs_random > 0.95).mean()), n_above_95=int((hot.dark_pct_vs_random > 0.95).sum()),
                             dark_sd=float(sd), random_median=float(np.median(ctrl)), random_p95=float(np.percentile(ctrl, 95)))
    # mean pre image brightness structure at hotspots (is there a visible object?): use the 2026-10-08 pre images (-0) mean as a camera-coordinate stack
    stack = np.mean([tifffile.imread(RAW / '261008_p50_z' / f'{b}-{pp}-0.tif').astype(np.float32) for b in (1, 2, 3) for pp in range(1, 9)], axis=0)
    sb = cv2.GaussianBlur(stack, (0, 0), 2) - cv2.GaussianBlur(stack, (0, 0), 25); ssd = 1.4826 * np.median(abs(sb - np.median(sb)))
    zs = []
    for r in hot.itertuples():
        x, y = int(round(r.x)), int(round(r.y)); win = sb[max(y - 12, 0):y + 13, max(x - 12, 0):x + 13]; zs.append(float(np.abs(win).max() / ssd))
    hot['stack_absmax_over_sd'] = zs; hot.to_csv(V85 / 'E3_hotspots_vs_dark.csv', index=False)
    ctrl2 = [float(np.abs(sb[y - 12:y + 13, x - 12:x + 13]).max() / ssd) for x, y in zip(rng.integers(20, 2028, 2000), rng.integers(20, 2024, 2000))]
    out['stack_check'] = dict(median_hot=float(np.median(zs)), median_random=float(np.median(ctrl2)), frac_hot_above_random_p95=float((np.array(zs) > np.percentile(ctrl2, 95)).mean()))
    json.dump(out, open(V85 / 'E3_hotspots.json', 'w'), indent=1, ensure_ascii=False, default=float)
    # figure: hotspot map over the dark-frame fixed pattern and the stack of pre images
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.6))
    axs[0].scatter(pts.x, pts.y, s=3, c='#d62728', alpha=.4); axs[0].scatter(hot.x, hot.y, s=60, facecolors='none', edgecolors='k'); axs[0].set_title(f'異常ピラー(外れ値の基板を除く{len(good)}視野、{len(pts)}個)とホットスポット{len(hot)}か所(黒丸)')
    for a_, im, ttl in ((axs[1], dkb, '暗画像6枚の平均(バンドパス)'), (axs[2], sb, '今日の-0画像24枚の平均(バンドパス)')):
        lo, hi = np.percentile(im, [1, 99]); a_.imshow(im, cmap='gray', vmin=lo, vmax=hi, extent=(0, 2048, 2044, 0)); a_.scatter(hot.x, hot.y, s=70, facecolors='none', edgecolors='#ff1744', lw=1); a_.set_title(ttl + '(赤丸=ホットスポット)')
    for a_ in axs[:1]: a_.set_xlim(0, 2048); a_.set_ylim(2044, 0); a_.set_aspect('equal')
    plt.tight_layout(); plt.savefig(VAULT / 'E3_ホットスポットと暗画像.png', dpi=100); plt.close()
    print(json.dumps(out, indent=0, ensure_ascii=False, default=float)[:5000])
    print(hot.head(20).round(2).to_string())


if __name__ == '__main__':
    main()
