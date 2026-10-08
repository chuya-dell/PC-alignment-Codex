"""v85 maps: (1) per-field map of abnormal pillars (fixed-29 + blank = 97 fields) with the cross-scar mask, the bright band and the field edge zone drawn on top;
(2) for the same fields, positives (|z|>4, scar-excluded) NOT counting abnormal pillars (S5) next to COUNTING them (S5P) (k=4).
Output PNGs in the vault: 図と表/異常ピラー地図/{fid}.png and 図と表/陽性比較_k4/{fid}.png, plus contact sheets.   usage: python v85_maps_fields.py [--workers K]"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd, cv2, tifffile, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ('Yu Gothic', 'Meiryo', 'MS Gothic'):
    if any(f in x.name for x in font_manager.fontManager.ttflist): plt.rcParams['font.family'] = f; break
plt.rcParams['axes.unicode_minus'] = False
V84 = ROOT / 'data/results/v84_1fM_strategies'; V85 = ROOT / 'data/results/v85_abnormal_pillars'
V57 = Path(r'C:\Users\chuya\PC-alignment-localcorr\data\results\v57_false_positive_facts_20261006\stepD')
VAULT = Path(r'W:\GoogleDrive\chuya2816\Obsidian Vault\ラボノート\05_解析\261008_【未読】偽陽性ゼロ化_原因確定と標準経路\図と表')
D1 = VAULT / '異常ピラー地図'; D2 = VAULT / '陽性比較_k4'
PER = 7.37; H, W = 2044, 2048
V70 = ROOT / 'data/results/v70_ledger_center_fit'
HOT = pd.read_csv(V85 / 'E3_hotspots_post_from_nonblank.csv')


def unpack(p):
    with np.load(p) as z:
        shp = tuple(z['shape']); return np.unpackbits(z['mask'])[:np.prod(shp)].reshape(shp).astype(bool)


def contours(m, ds=2):
    mm = cv2.resize(m.astype(np.uint8), (W // ds, H // ds), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(mm, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE); return [c[:, 0, :].astype(float) * ds for c in cs if len(c) > 20]


def base_axes(ax, pre, fid, title):
    lo, hi = np.percentile(pre, [0.5, 99.5]); g = np.clip((pre.astype(np.float32) - lo) / (hi - lo), 0, 1)
    ax.imshow(g, cmap='gray', vmin=0, vmax=1, extent=(0, W, H, 0), interpolation='nearest'); ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(title, fontsize=9); ax.tick_params(labelsize=6)


def overlays(ax, mask, band, edge_periods=4):
    for c in contours(mask): ax.plot(c[:, 0], c[:, 1], color='#00e676', lw=0.9)
    if band is not None:
        for c in contours(band): ax.plot(c[:, 0], c[:, 1], color='#ffeb3b', lw=0.9)
    e = edge_periods * PER; ax.plot([e, W - e, W - e, e, e], [e, e, H - e, H - e, e], color='#00bcd4', lw=0.8, ls='--')


def one(rec):
    fid = rec['fid']
    try:
        z = np.load(V84 / 'cache' / f'{fid}.npz'); ctr = z['ctr']; z5 = z['z5']; z5p = z['z5p']; sd = z['sd']; excl = z['excl']; abn = z['abn']
        pre = tifffile.imread(rec['pre_path']); mask = unpack(V57 / f'{fid}_mask.npz')
        bp = V85 / 'band' / f'{fid}_band.npz'; band = unpack(bp) if bp.exists() else None
        z2 = np.load(V70 / 'fields2' / f'{fid}.npz'); post = ctr.astype(np.float64) @ z2['Llin'].T + z2['tL']
    except Exception as e:
        return fid, repr(e)
    sig_note = f"{fid}  群:{rec['group']}  位置{rec['field']}"
    # ---- (1) abnormal map ----
    okP = np.isfinite(z5p); cand = abn & okP; big = cand & (np.abs(z5p) > 6); mid = cand & (np.abs(z5p) > 4) & ~(np.abs(z5p) > 6)
    out_ = excl
    fig, ax = plt.subplots(figsize=(9.2, 9.2)); base_axes(ax, pre, fid, f"{sig_note}\n異常ピラー(後ろ側の当てはめ失敗かつ |差|>6σ):傷の外 {int((big & out_).sum())} 個、傷の中・2周期以内 {int((big & ~out_).sum())} 個(4〜6σ: {int(mid.sum())} 個)")
    overlays(ax, mask, band)
    s = big & out_; ax.scatter(ctr[s, 0], ctr[s, 1], s=70, facecolors='#ff1744', edgecolors='white', lw=0.8, zorder=5)
    s = big & ~out_; ax.scatter(ctr[s, 0], ctr[s, 1], s=36, facecolors='none', edgecolors='#ff1744', lw=1.2, zorder=5)
    s = mid; ax.scatter(ctr[s, 0], ctr[s, 1], s=14, facecolors='#ff9100', edgecolors='none', zorder=4)
    for hx, hy in zip(HOT.x, HOT.y): ax.add_patch(plt.Circle((hx, hy), 20, fill=False, ec='#e040fb', lw=1.1, ls='--', zorder=3))
    for i in np.where(big & out_)[0]: ax.plot([ctr[i, 0], post[i, 0]], [ctr[i, 1], post[i, 1]], color='white', lw=0.6, alpha=.8, zorder=4); ax.scatter([post[i, 0]], [post[i, 1]], s=26, marker='x', c='#e040fb', lw=1.2, zorder=6)
    ax.text(12, H - 12, '赤丸=傷の外の異常ピラー(6σ超、前の像の位置) 紫の×と白線=その後ろの像での位置 紫破線の円=カメラ固定の欠陥(13か所)\n赤枠=傷の中 橙点=4〜6σ 緑線=十字傷マスク 黄線=明部帯 青破線=端から4周期', color='white', fontsize=6.5, va='bottom', bbox=dict(facecolor='black', alpha=.55, pad=2))
    fig.tight_layout(); fig.savefig(D1 / f'{fid}.png', dpi=100); plt.close(fig)
    # ---- (2) positives k=4, S5 (abnormal not counted) vs S5P (abnormal counted) ----
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.3))
    pos5 = excl & np.isfinite(z5) & (np.abs(z5) > 4); posP = excl & okP & (np.abs(z5p) > 4)
    for ax, sel, ttl in ((axs[0], pos5, f'異常ピラーを数えない(S5): 陽性 {int(pos5.sum())} 個'), (axs[1], posP, f'異常ピラーを数える(S5P): 陽性 {int(posP.sum())} 個(うち異常 {int((posP & abn).sum())})')):
        base_axes(ax, pre, fid, f'{fid}  k=4  {ttl}'); overlays(ax, mask, band)
        pp = sel & (z5p > 0) if ax is axs[1] else sel & (z5 > 0); nn = sel & ~pp
        zz = z5p if ax is axs[1] else z5
        ax.scatter(ctr[sel & (zz > 0) & ~abn, 0], ctr[sel & (zz > 0) & ~abn, 1], s=22, c='#ff1744', marker='o', lw=0)
        ax.scatter(ctr[sel & (zz < 0) & ~abn, 0], ctr[sel & (zz < 0) & ~abn, 1], s=22, c='#2979ff', marker='o', lw=0)
        if ax is axs[1]: ax.scatter(ctr[sel & abn, 0], ctr[sel & abn, 1], s=70, facecolors='none', edgecolors='#ffea00', lw=1.4, zorder=6)
    axs[1].text(12, H - 12, '赤=後ろが暗い(減光側)/青=明るい(増光側)/黄の輪=異常ピラー(右だけに出る)', color='white', fontsize=8, va='bottom', bbox=dict(facecolor='black', alpha=.55, pad=2))
    fig.tight_layout(); fig.savefig(D2 / f'{fid}.png', dpi=90); plt.close(fig)
    return fid, 'ok'


if __name__ == '__main__':
    D1.mkdir(parents=True, exist_ok=True); D2.mkdir(parents=True, exist_ok=True)
    led = pd.read_csv(ROOT / 'data/results/v70_ledger_center_fit/ledger.csv', dtype={'date': str, 'board': str}); fs = pd.read_csv(V85 / 'field_summary.csv')
    sel = led[led.fid.isin(fs.fid)].to_dict('records')
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 4
    from multiprocessing import Pool
    with Pool(workers) as p:
        res = p.map(one, sel, chunksize=2)
    bad = [r for r in res if r[1] != 'ok']; print('done', len(res) - len(bad), 'failed', bad[:5])
