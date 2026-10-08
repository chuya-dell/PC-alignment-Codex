"""v87 D3: hypothesis 4(b): the steep intensity gradient at the scar edges turns a small reading-position error into a large difference.
Per-pillar local background gradient G = |grad(Gaussian_4(pre/65535))| (contrast units per pixel; the pillar lattice (7.3 px) is removed by sigma 4), sampled at the reading position, for the 69 blank fields.
A reading-position error d changes an area-9 aperture sum by ~ 9*G*d.  Measured: (1) G by distance class from the scar mask, (2) |delta - median| (S0 current standard at its std positions; S5 C1) in bins of G,
(3) the slope of the median |delta| versus G in the high-G range, converted to an effective position error d_eff = slope/9 (px), (4) share of the |z|>4 events in the high-G pillars.
usage: python v87_d3_edge_gradient.py [--workers K]"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit'))
import numpy as np, pandas as pd, cv2, tifffile
from scipy import ndimage
V70 = ROOT / 'data/results/v70_ledger_center_fit'; V72 = ROOT / 'data/results/v72_real_field_readout'; V84 = ROOT / 'data/results/v84_1fM_strategies'; OUT = ROOT / 'data/results/v87_part_d'; PER = 7.37
EDG = [-1e9, 0, 2, 4, 8, 1e9]; LAB = ['マスク内', '外0-2', '2-4', '4-8', '>8']


def one(rec):
    cv2.setNumThreads(1); fid = rec['fid']
    pre = tifffile.imread(rec['pre_path']).astype(np.float32) / 65535; B = cv2.GaussianBlur(pre, (0, 0), 4)
    gx = cv2.Sobel(B, cv2.CV_32F, 1, 0, ksize=3) / 8.0; gy = cv2.Sobel(B, cv2.CV_32F, 0, 1, ksize=3) / 8.0; G = np.hypot(gx, gy)
    z72 = np.load(V72 / 'fields' / f'{fid}.npz'); c = np.load(V84 / 'cache' / f'{fid}.npz')
    # S5 at actual centres (pre)
    ctr = z72['ctr_xy'].astype(float); g5 = ndimage.map_coordinates(G, [ctr[:, 1], ctr[:, 0]], order=1, mode='nearest'); S5 = z72['S5'].astype(float); sd5 = c['sd'].astype(float)
    ok5 = np.isfinite(S5); med5 = np.median(S5[ok5 & (sd5 > 2)]); sig5 = 1.4826 * np.median(abs(S5[ok5 & (sd5 > 2)] - med5))
    # S0 at standard positions
    xy = z72['std_xy'].astype(float); g0 = ndimage.map_coordinates(G, [xy[:, 1], xy[:, 0]], order=1, mode='nearest'); S0 = z72['S0'].astype(float); sd0 = np.where(z72['in_mask_std'], -1.0, z72['dist_std'] / PER)
    ok0 = np.isfinite(S0); med0 = np.median(S0[ok0 & (sd0 > 2)]); sig0 = 1.4826 * np.median(abs(S0[ok0 & (sd0 > 2)] - med0))
    return dict(fid=fid, g5=g5[ok5].astype(np.float32), a5=np.abs(S5[ok5] - med5).astype(np.float32), z5=((S5[ok5] - med5) / sig5).astype(np.float32), sd5=sd5[ok5].astype(np.float32),
                g0=g0[ok0].astype(np.float32), a0=np.abs(S0[ok0] - med0).astype(np.float32), z0=((S0[ok0] - med0) / sig0).astype(np.float32), sd0=sd0[ok0].astype(np.float32), sig5=sig5, sig0=sig0)


def main():
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 6
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}); sel = led[(led.group == 'blank')]
    from multiprocessing import Pool
    with Pool(workers) as p: R = p.map(one, sel.to_dict('records'), chunksize=1)
    out = {}
    for tag in ('5', '0'):
        g = np.concatenate([r['g' + tag] for r in R]); a = np.concatenate([r['a' + tag] for r in R]); z = np.concatenate([r['z' + tag] for r in R]); sd = np.concatenate([r['sd' + tag] for r in R])
        sig = float(np.median([r['sig' + tag] for r in R]))
        res = dict(sigma_median=sig)
        # (1) G by distance class
        res['G_by_dist'] = {l: dict(median=float(np.median(g[(sd > lo) & (sd <= hi)])), p95=float(np.percentile(g[(sd > lo) & (sd <= hi)], 95)), n=int(((sd > lo) & (sd <= hi)).sum())) for l, lo, hi in zip(LAB, EDG[:-1], EDG[1:])}
        # (2) |delta| and tail in G bins (pooled over distance classes; outside-scar only and all)
        qs = np.quantile(g, [0, .5, .8, .9, .95, .98, .99, 1.0]); bins = []
        for i in range(len(qs) - 1):
            s = (g >= qs[i]) & (g <= qs[i + 1]); bins.append(dict(G_lo=float(qs[i]), G_hi=float(qs[i + 1]), n=int(s.sum()), median_abs_delta=float(np.median(a[s])), p95_abs_delta=float(np.percentile(a[s], 95)), frac_z_gt4=float((np.abs(z[s]) > 4).mean()), frac_z_gt8=float((np.abs(z[s]) > 8).mean()), share_in_mask=float((sd[s] <= 0).mean())))
        res['G_bins'] = bins
        out_ = sd > 2; qo = np.quantile(g[out_], [0, .5, .8, .9, .95, .98, .99, 1.0]); bo = []
        for i in range(len(qo) - 1):
            s = out_ & (g >= qo[i]) & (g <= qo[i + 1]); bo.append(dict(G_lo=float(qo[i]), G_hi=float(qo[i + 1]), n=int(s.sum()), median_abs_delta=float(np.median(a[s])), frac_z_gt4=float((np.abs(z[s]) > 4).mean()), frac_z_gt8=float((np.abs(z[s]) > 8).mean())))
        res['G_bins_outside_scar'] = bo
        # (3) slope: median |delta| vs G above the 80th percentile (outside scar), bin-wise linear fit; d_eff = slope / 9
        hi = out_ & (g >= qo[2]); gg = g[hi]; aa = a[hi]; edges = np.quantile(gg, np.linspace(0, 1, 11)); xs = []; ys = []
        for i in range(10):
            s = (gg >= edges[i]) & (gg <= edges[i + 1]); xs.append(np.median(gg[s])); ys.append(np.median(aa[s]))
        slope = float(np.polyfit(xs, ys, 1)[0]); res['slope_median_abs_delta_per_G'] = slope; res['d_eff_px'] = slope / 9.0
        # (4) share of |z|>4 and >8 events in the top 5% G (all), and its excess factor
        top = g >= np.quantile(g, 0.95); res['share_events_z4_in_top5pctG'] = float((np.abs(z[top]) > 4).sum() / max((np.abs(z) > 4).sum(), 1)); res['share_events_z8_in_top5pctG'] = float((np.abs(z[top]) > 8).sum() / max((np.abs(z) > 8).sum(), 1))
        out[tag] = res
    json.dump(out, open(OUT / 'D3_edge_gradient.json', 'w'), indent=1, ensure_ascii=False, default=float)
    for tag, nm in (('0', 'S0 現行標準'), ('5', 'S5 C1')):
        r = out[tag]; print('==', nm, 'sigma', round(r['sigma_median'], 4), '| d_eff_px', round(r['d_eff_px'], 4), '| z4 share top5%G', round(r['share_events_z4_in_top5pctG'], 3), 'z8', round(r['share_events_z8_in_top5pctG'], 3))
        print(' G by dist:', {k: (round(v['median'], 5), round(v['p95'], 5)) for k, v in r['G_by_dist'].items()})
        print(' outside-scar G bins:', [(round(b['G_hi'], 5), round(b['median_abs_delta'], 4), round(b['frac_z_gt4'], 5), round(b['frac_z_gt8'], 6)) for b in r['G_bins_outside_scar']])


if __name__ == '__main__':
    main()
