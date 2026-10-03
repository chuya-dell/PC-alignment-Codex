"""Read-only cached-field analysis under the unchanged round-one predictions.

All scientific outputs and checkpoints go to data/results/v28_band_origin_round1.
No input imports or source-directory writes (including bytecode) are performed.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import fft

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'results' / 'v28_band_origin_round1'
LOCAL = ROOT / 'data' / 'inputs_local'
SEED = 20261004
METHODS = ['平均標準偏差', '中央値絶対偏差']

def plotting():
    os.environ['MPLCONFIGDIR'] = str(OUT/'matplotlib_config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt

def correlate_full(a, b):
    shape = tuple(x+y-1 for x, y in zip(a.shape, b.shape))
    return fft.irfftn(fft.rfftn(a, shape)*fft.rfftn(b, shape), shape)

def local_peaks(a):
    return np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]))+1

def half_width(a, peak):
    height = a[peak]/2
    left, right = peak, peak
    while left > 0 and a[left] > height:
        left -= 1
    while right < len(a)-1 and a[right] > height:
        right += 1
    xl = left+(height-a[left])/(a[left+1]-a[left]) if a[left+1] != a[left] else left
    xr = right-1+(height-a[right-1])/(a[right]-a[right-1]) if a[right] != a[right-1] else right
    return xr-xl

def discover():
    children = list(LOCAL.iterdir())
    source = [p for p in children if p.is_dir() and p.name.startswith('2026') and 'digital_judgment' in p.name]
    if len(source) != 1:
        raise RuntimeError('解析入力の起点を一意に確認できない')
    return source[0]

def verify_prediction():
    recorded = pd.read_csv(OUT / 'prediction_sha256.csv').iloc[0]
    p = Path(recorded['Path'])
    got = hashlib.sha256(p.read_bytes()).hexdigest().upper()
    if got != recorded['Hash']:
        raise RuntimeError('元の予測の指紋が一致しない')
    return got

def write_csv(rows, name):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')

def fields():
    f = pd.read_csv(OUT / 'all_fields_existing_definition.csv', dtype={'日程': str, '基板': str})
    f['key'] = f['日程'] + '_' + f['基板'] + '_' + f['視野番号'].astype(str)
    return f

def load(r):
    # Path comes from the previously enumerated actual cache inventory.
    p = Path(r['保存データ実在パス'])
    if LOCAL.resolve() not in p.resolve().parents:
        raise RuntimeError('入力起点外の保存データ')
    with np.load(p, allow_pickle=False) as z:
        return z['delta'].copy(), z['xy'].copy(), z['ids'].copy()

def maps_stage():
    f = fields()
    thresholds = []
    for date, g in f.groupby('日程'):
        blank = np.concatenate([load(r)[0] for _, r in g[g['ブランク']].iterrows()])
        med = np.median(blank)
        mad = np.median(np.abs(blank - med))
        thresholds.append(dict(日程=date, ブランク視野数=int(g['ブランク'].sum()), ブランクピラー数=len(blank),
                               平均=float(blank.mean()), 標準偏差=float(blank.std()), 中央値=float(med),
                               中央値絶対偏差=float(mad), 平均標準偏差=float(blank.mean()+3*blank.std()),
                               中央値絶対偏差閾値=float(med+3*1.4826*mad)))
    write_csv(thresholds, 'round1_thresholds.csv')
    t = {r['日程']: [r['平均標準偏差'], r['中央値絶対偏差閾値']] for r in thresholds}
    density = np.full((len(f), 2, 64, 64), np.nan, np.float32)
    count = np.zeros((len(f), 64, 64), np.int32)
    rows, inventory = [], []
    for i, r in f.iterrows():
        d, xy, ids = load(r)
        if not np.isfinite(d).all() or not np.isfinite(xy).all():
            raise RuntimeError(f'非有限入力 {r.key}')
        cell = (xy[:, 1] // 32).astype(int)*64+(xy[:, 0]//32).astype(int)
        n = np.bincount(cell, minlength=4096)
        count[i] = n.reshape(64, 64)
        fit = np.linalg.lstsq(np.column_stack([ids, np.ones(len(ids))]), xy, rcond=None)[0]
        lattice_angle = float(np.degrees(np.arctan2(fit[0, 1], fit[0, 0])) % 60)
        for mi, method in enumerate(METHODS):
            positive = d > t[r['日程']][mi]
            pos = np.bincount(cell, weights=positive, minlength=4096)
            dm = np.divide(pos, n, out=np.full(4096, np.nan), where=n > 0).reshape(64, 64)
            density[i, mi] = dm
            edge = np.minimum.reduce([xy[:, 0], 2048-xy[:, 0], xy[:, 1], 2044-xy[:, 1]])
            row = dict(key=r.key, 日程=r['日程'], 基板=r['基板'], 視野番号=int(r['視野番号']), 位置番号=int(r['位置番号']),
                       濃度=r['濃度'], ブランク=bool(r['ブランク']), 定義=method, 閾値=t[r['日程']][mi],
                       ピラー数=len(d), 外れ値ピラー数=int(positive.sum()), 外れ値割合=float(positive.mean()),
                       固定29視野=bool(r['外れ値視野3percent以上']), この定義3percent以上=bool(positive.mean() >= .03),
                       六方格子第一軸度=lattice_angle, 格子座標近似最大誤差=float(np.max(np.abs(np.column_stack([ids, np.ones(len(ids))]) @ fit-xy))),
                       既存割合差=float(positive.mean()-r['外れ値ピラー割合']) if mi == 0 else np.nan)
            for width in (200, 300):
                e = edge < width
                row[f'縁{width}内割合'] = float(positive[e].mean()) if e.any() else np.nan
                row[f'縁{width}外割合'] = float(positive[~e].mean()) if (~e).any() else np.nan
                row[f'外れ値の縁{width}内占有率'] = float(positive[e].sum()/positive.sum()) if positive.any() else np.nan
                row[f'ピラーの縁{width}内占有率'] = float(e.mean())
            rows.append(row)
        p = Path(r['保存データ実在パス'])
        inventory.append(dict(key=r.key, 日程=r['日程'], 基板=r['基板'], 視野番号=int(r['視野番号']), 実在パス=str(p),
                              内容指紋=hashlib.sha256(p.read_bytes()).hexdigest(), ピラー数=len(d)))
        if i % 100 == 0:
            print('density', i, r.key, flush=True)
    np.savez_compressed(OUT/'density_maps.npz', density=density, counts=count, keys=f.key.to_numpy(dtype=str), methods=np.array(METHODS))
    # Leave-one-field-out common profile is saved for later explanation models.
    valid = np.isfinite(density)
    total = np.nansum(density, axis=0)
    number = valid.sum(axis=0)
    common = np.divide(total[None]-np.nan_to_num(density), number[None]-valid,
                       out=np.full_like(density, np.nan), where=(number[None]-valid)>0)
    np.savez_compressed(OUT/'common_profile_residuals.npz', residual=density-common, common=common, keys=f.key.to_numpy(dtype=str))
    write_csv(rows, 'round1_fields_both_definitions.csv')
    write_csv(inventory, 'round1_cache_usage_sha256.csv')
    cutoffs = []
    for method, g in pd.DataFrame(rows).groupby('定義'):
        for cut in (.01, .02, .03, .04, .05):
            cutoffs.append(dict(定義=method, 基準割合=cut, 視野数=int((g['外れ値割合'] >= cut).sum())))
    write_csv(cutoffs, 'round1_cutoff_counts.csv')

def angular_difference(a, b):
    return np.abs((np.asarray(a)-b+90) % 180-90)

def geometry(shape):
    h, w = shape
    fy = fft.fftfreq(h, d=32)[:, None]
    fx = fft.rfftfreq(w, d=32)[None, :]
    radius = np.hypot(fy, fx)
    band = (radius >= 1/1024) & (radius <= 1/128)
    axis = (np.degrees(np.arctan2(np.broadcast_to(fy, radius.shape), np.broadcast_to(fx, radius.shape)))+90) % 180
    angles = np.arange(0, 180, 10)
    # Equal +/-5 degree bins define the direction search; +/-15 determines concentration.
    direction_bin = np.floor(((axis+5) % 180)/10).astype(int)
    sectors = np.array([band & (direction_bin == j) for j in range(len(angles))])
    multiplicity = np.full(radius.shape, 2.)
    multiplicity[:, 0] = 1
    if w % 2 == 0:
        multiplicity[:, -1] = 1
    window = np.outer(np.hanning(h), np.hanning(w))
    return band, axis, angles, sectors, multiplicity, window, radius

def linear_autocorrelation(a):
    mask = np.isfinite(a)
    centered = np.where(mask, a-np.nanmean(a), 0)
    numerator = correlate_full(centered, centered[::-1, ::-1])
    overlap = correlate_full(mask.astype(float), mask[::-1, ::-1].astype(float))
    ac = np.divide(numerator, overlap, out=np.full_like(numerator, np.nan), where=overlap > .5)
    var = np.nanvar(a)
    if var > 0:
        ac /= var
    else:
        ac[:] = np.nan
    return ac, overlap

def projection(a, axis):
    # Average along the band axis, leaving a one-dimensional normal coordinate.
    yy, xx = np.indices(a.shape)
    normal = np.radians(axis-90)
    coord = xx*np.cos(normal)+yy*np.sin(normal)
    bins = np.rint(coord-coord.min()).astype(int)
    mask = np.isfinite(a)
    n = np.bincount(bins[mask], minlength=bins.max()+1)
    val = np.bincount(bins[mask], weights=a[mask], minlength=len(n))
    p = np.divide(val, n, out=np.full(len(n), np.nan), where=n > 0)
    keep = n >= max(3, .2*n.max())
    if not keep.any():
        return np.array([])
    lo, hi = np.flatnonzero(keep)[[0, -1]]
    return p[lo:hi+1]

def projection_metrics(a, axis, spectrum_period):
    p = projection(a, axis)
    if len(p) < 12 or not np.isfinite(p).all() or np.var(p) == 0:
        return np.nan, np.nan, False
    c = p-p.mean()
    ac = correlate_full(c, c[::-1])[len(c)-1:]/np.arange(len(c), 0, -1)
    ac /= ac[0]
    peaks = local_peaks(ac)
    candidates = peaks[(peaks >= 4) & (peaks <= 32) & (peaks <= len(p)//2) & (ac[peaks] > 0)]
    period = float(candidates[0]*32) if len(candidates) else np.nan
    positive = p-np.median(p)
    hills = local_peaks(positive)
    hills = hills[positive[hills] > 0]
    width = float(np.median([half_width(positive, h) for h in hills])*32) if len(hills) else np.nan
    agree = bool(np.isfinite(period) and abs(period-spectrum_period)/spectrum_period <= .2)
    return period, width, agree

def analyze_map(a, rng, repetitions=1000):
    band, axis, angles, sectors, mult, window, radius = geometry(a.shape)
    mask = np.isfinite(a)
    mean = np.nanmean(a)
    c = np.where(mask, a-mean, 0)
    power = abs(fft.rfft2(c*window))**2*mult
    weights = np.einsum('ahw,hw->a', sectors, power)
    rank = np.argsort(weights)[::-1]
    first = float(angles[rank[0]])
    total = power[band].sum()
    strongest = float(weights[rank[0]]/total) if total > 0 else 0.
    concentration = float(power[band & (angular_difference(axis, first) <= 15)].sum()/total) if total > 0 else 0.
    second_candidates = [int(k) for k in rank if 30 <= angular_difference(angles[k], first) <= 150]
    second = float(angles[second_candidates[0]]) if second_candidates else np.nan
    conc2 = float(power[band & (angular_difference(axis, second) <= 15)].sum()/total) if total > 0 else 0.
    def spectral_period(direction):
        select = band & (angular_difference(axis, direction) <= 15)
        if not select.any() or total == 0:
            return np.nan
        best = np.argmax(np.where(select, power, -1))
        return float(1/radius.ravel()[best])
    sp1, sp2 = spectral_period(first), spectral_period(second)
    per1, width, agree1 = projection_metrics(a, first, sp1)
    per2, _, agree2 = projection_metrics(a, second, sp2)
    values = c[mask]
    null = []
    # Vectorized batches: re-use frequency and direction masks for all shuffles.
    for start in range(0, repetitions, 50):
        size = min(50, repetitions-start)
        batch = np.zeros((size, *a.shape), dtype=np.float32)
        batch[:, mask] = rng.permuted(np.broadcast_to(values, (size, len(values))), axis=1)
        bp = abs(fft.rfft2(batch*window, axes=(-2, -1), workers=2))**2*mult
        sums = np.einsum('ahw,bhw->ba', sectors.astype(float), bp, optimize=True)
        totals = bp[:, band].sum(axis=1)
        null.extend(np.divide(sums.max(axis=1), totals, out=np.zeros(size), where=totals>0))
    critical = float(np.percentile(null, 95))
    periodic = strongest > critical
    category = 'その他'
    if periodic and concentration >= .45 and agree1:
        category = '帯状候補'
    elif periodic and concentration >= .20 and conc2 >= .20 and concentration+conc2 >= .60 and agree1 and agree2:
        category = 'うろこ状候補'
    ac, overlap = linear_autocorrelation(a)
    row = dict(分類=category, 周期候補=bool(periodic), 第一帯軸度=first, 第二帯軸度=second,
               第一方向集中度=concentration, 第二方向集中度=conc2, 最大方向パワー割合=strongest,
               並替95百分位=critical, 分類用比較確率=float((1+np.sum(np.array(null)>=strongest))/(repetitions+1)),
               第一スペクトル周期画素=sp1, 第二スペクトル周期画素=sp2, 第一自己相関周期画素=per1,
               第二自己相関周期画素=per2, 帯幅画素=width if agree1 else np.nan,
               第一周期一致=agree1, 第二周期一致=agree2,
               カメラ軸最小角度差=float(min(angular_difference(first, 0), angular_difference(first, 90))))
    return row, ac, overlap, power

def classify_stage(limit=0):
    f = fields()
    with np.load(OUT/'density_maps.npz') as z:
        dm = z['density']
    checkpoints = OUT/'classification_checkpoints'
    checkpoints.mkdir(exist_ok=True)
    spatial = OUT/'spatial_arrays'
    spatial.mkdir(exist_ok=True)
    lattice = pd.read_csv(OUT/'round1_fields_both_definitions.csv').set_index(['key', '定義'])
    end = min(len(f), limit) if limit else len(f)
    for i in range(end):
        r = f.iloc[i]
        p = checkpoints/(r.key+'.json')
        if p.exists():
            continue
        rows = []
        arrays = {}
        for mi, method in enumerate(METHODS):
            for margin in (0, 200, 300):
                # Cell-center crop: first 32-square cells with centers >= margin.
                keepx = (np.arange(64)*32+16 >= margin) & (np.arange(64)*32+16 <= 2048-margin)
                keepy = (np.arange(64)*32+16 >= margin) & (np.arange(64)*32+16 <= 2044-margin)
                a = dm[i, mi][np.ix_(keepy, keepx)]
                rng = np.random.default_rng(SEED+int.from_bytes(hashlib.sha256(f'{r.key}/{method}/{margin}'.encode()).digest()[:4], 'little'))
                stats, ac, overlap, power = analyze_map(a, rng)
                stats.update(key=r.key, 定義=method, 縁除外画素=margin, 地図行数=a.shape[0], 地図列数=a.shape[1],
                             六方格子最小角度差=float(min(angular_difference(stats['第一帯軸度'], lattice.loc[(r.key, method), '六方格子第一軸度']+j*60) for j in range(3))))
                rows.append(stats)
                prefix = f'm{mi}_e{margin}'
                arrays[prefix+'_autocorrelation'] = ac.astype(np.float32)
                arrays[prefix+'_overlap'] = overlap.astype(np.float32)
                arrays[prefix+'_power'] = power.astype(np.float32)
        np.savez_compressed(spatial/(r.key+'.npz'), **arrays)
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=True), encoding='utf-8')
        if i % 10 == 0 or limit:
            print('classification', i+1, '/', end, r.key, flush=True)
    rows = []
    for p in sorted(checkpoints.glob('*.json')):
        rows.extend(json.loads(p.read_text(encoding='utf-8')))
    write_csv(rows, 'round1_spatial_classification.csv')

def atlas_stage():
    plt = plotting()
    f = fields()
    with np.load(OUT/'density_maps.npz') as z:
        dm = z['density']
    dest = OUT/'atlases'
    dest.mkdir(exist_ok=True)
    plt.rcParams['font.family'] = 'Yu Gothic'
    for mi, method in enumerate(METHODS):
        for start in range(0, len(f), 12):
            page = dest/f'{method}_{start//12+1:03d}.png'
            if page.exists():
                continue
            n = min(12, len(f)-start)
            fig, axes = plt.subplots(n, 3, figsize=(12, n*2.4), squeeze=False)
            for j in range(n):
                i = start+j
                r = f.iloc[i]
                a = dm[i, mi]
                with np.load(OUT/'spatial_arrays'/(r.key+'.npz')) as z:
                    ac = z[f'm{mi}_e0_autocorrelation']
                    power = z[f'm{mi}_e0_power']
                axes[j, 0].imshow(a, origin='upper', vmin=0, vmax=max(.01, np.nanpercentile(a, 99)), extent=(0, 2048, 2044, 0))
                axes[j, 0].set_title(r.key+' '+str(r['濃度']), fontsize=9)
                axes[j, 1].imshow(ac, origin='upper', vmin=-.4, vmax=.8, cmap='coolwarm', extent=(-2016, 2016, 2016, -2016))
                axes[j, 1].set_title('重なり補正自己相関', fontsize=9)
                # Display native unpadded frequency grid, explicitly retaining quantization.
                axes[j, 2].imshow(np.log1p(np.fft.fftshift(power, axes=0)), origin='upper', aspect='auto')
                axes[j, 2].set_title('窓付きスペクトル（片側）', fontsize=9)
                for ax in axes[j]:
                    ax.tick_params(labelsize=6)
            fig.tight_layout()
            fig.savefig(page, dpi=120)
            plt.close(fig)
            print('atlas', page.name, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['maps', 'classify', 'atlas'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    print('prediction unchanged', verify_prediction(), flush=True)
    start = time.time()
    if args.stage == 'maps':
        maps_stage()
    elif args.stage == 'classify':
        classify_stage(args.limit)
    else:
        atlas_stage()
    print('completed', args.stage, time.time()-start, flush=True)

if __name__ == '__main__':
    main()
