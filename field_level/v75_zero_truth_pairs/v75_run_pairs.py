"""v75 step B: zero-truth pairs (burst, small stage steps, synthetic shifts) x readout schemes x thresholds.
Synthetic pair generation: burst image 2 -> 4x sinc (FFT zero padding) -> exact 1/4-px shifts on the fine grid -> 4x4 area integration
(rotations: Lanczos4 warp on the fine grid -> area integration).  The analysis never interpolates this way (it rounds or uses bilinear).
usage: python v75_run_pairs.py real | synth"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, cv2
from scipy import fft as sfft
import v75_pair_lib as P
import field_control_common as M60
OUT = P.ROOT / 'data/results/v75_zero_truth_pairs'
RAW = M60.RAW
KS = [3, 4, 5, 6, 8]
READS = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S3A', 'S4A']


def folder(n): return M60.folder(n)


def metrics_for(res, label, thr_ref=None):
    rows = []
    for s in READS:
        d = res[s].astype(float); xy = P.positions(res, s) if s not in ('S3A', 'S4A') else res['ctr_xy']
        ok = np.isfinite(d)
        if ok.sum() < 5000: continue
        med = float(np.median(d[ok])); mad = 1.4826 * float(np.median(abs(d[ok] - med)))
        for thr_name, t in (('each', float(d[ok].mean() + 3 * d[ok].std())), ) + ((('pool', thr_ref[s]),) if thr_ref and s in thr_ref else ()):
            met = M60.band_metrics(xy, d, t, False)
            rows.append(dict(label=label, readout=s, thr=thr_name, threshold=t, n_valid=int(ok.sum()), positive=met['positive'], rate=met['rate'], count91k=met['rate'] * 91000,
                             sd=float(d[ok].std()), mad_sigma=mad, band_power=met['band_power'], strength=met['strength'], density_sd=met['density_sd'], axis_deg=met['axis_deg'], period_px=met['period_px'],
                             **{f'k{k}_count91k': float((d[ok] > med + k * mad).mean() * 91000) for k in KS}))
    return rows


def local_thresh_metrics(res, label):
    import v72_analyze as A72
    rows = []
    for s in READS:
        d = res[s].astype(float); xy = P.positions(res, s) if s not in ('S3A', 'S4A') else res['ctr_xy']
        ok = np.isfinite(d)
        if ok.sum() < 5000: continue
        v = A72.local_resid(xy, d); t = float(v[ok].mean() + 3 * v[ok].std())
        met = M60.band_metrics(xy, v, t, False)
        rows.append(dict(label=label, readout=s, thr='local', threshold=t, n_valid=int(ok.sum()), positive=met['positive'], rate=met['rate'], count91k=met['rate'] * 91000,
                         sd=float(v[ok].std()), band_power=met['band_power'], strength=met['strength'], density_sd=met['density_sd'], axis_deg=met['axis_deg'], period_px=met['period_px']))
    return rows


def real_pairs():
    pairs = []
    p = folder('260904_p50_repeat'); pairs.append(('burst_1_2', p / '1.tif', p / '2.tif', 'burst'))
    m = folder('261006-p50-ステッピングモーター_test')
    names = ['-0.6965', '-0.6960', '-0.6955', '-0.6950', '-0.6945', '-0.6940', '-0.6935']
    for i in range(6): pairs.append((f'step_{names[i]}_{names[i + 1]}', m / f'{names[i]}.tif', m / f'{names[i + 1]}.tif', 'step'))
    return pairs


def run_real():
    (OUT / 'real').mkdir(parents=True, exist_ok=True)
    rows = []; results = {}
    for label, pa, pb, kind in real_pairs():
        t0 = time.time(); a = M60.read(pa); b = M60.read(pb)
        res = P.analyse_pair(a, b); results[label] = res
        np.savez_compressed(OUT / 'real' / f'{label}.npz', **{k: v for k, v in res.items() if isinstance(v, np.ndarray) and k not in ('pre_dict',)})
        rows.append(dict(label=label, kind=kind, A=json.dumps(res['A'].tolist()), corner_disagree_px=res['corner_disagree_px'], frac_corr_px=res['frac_corr_px'], U=json.dumps(res['U'].tolist()),
                         sub_pre=res['sub_pre'], sub_post=res['sub_post'], ok_frac=float(res['ok'].mean()), pitch_pre=res['pitch_pre'], pitch_post=res['pitch_post'], seconds=time.time() - t0))
        print('done', label, f'{time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'real_pairs_geometry.csv', index=False)
    # series-pool threshold for the 6 step pairs (per readout): pooled mean+3SD
    pool = {}
    for s in READS:
        v = np.concatenate([results[l][s][np.isfinite(results[l][s])] for l in results if l.startswith('step_')]); pool[s] = float(v.mean() + 3 * v.std())
    allrows = []
    for l, res in results.items():
        allrows += metrics_for(res, l, pool if l.startswith('step_') else None); allrows += local_thresh_metrics(res, l)
    t = pd.DataFrame(allrows); t.to_csv(OUT / 'real_pairs_metrics.csv', index=False)
    return t


# ---------- synthetic pairs ----------
def upsample4(img):
    h, w = img.shape
    F = sfft.fft2(img.astype(np.float32))
    Fp = np.zeros((4 * h, 4 * w), np.complex64)
    hh, ww = h // 2, w // 2
    Fp[:hh, :ww] = F[:hh, :ww]; Fp[:hh, -ww:] = F[:hh, -ww:]; Fp[-hh:, :ww] = F[-hh:, :ww]; Fp[-hh:, -ww:] = F[-hh:, -ww:]
    return (sfft.ifft2(Fp).real * 16).astype(np.float32)


def integrate4(fine):
    """Area-integrate the 4x fine grid back to original pixels, centred on the pixel: pixel n covers positions n-0.5..n+0.5 = fine indices 4n-2..4n+2
    (end samples at half weight).  Earlier versions used fine indices 4n..4n+3 (a constant -0.375 px offset; found by the Codex audit) and then 4n-2..4n+1
    (+0.125 px); this symmetric version has zero offset."""
    wts = {-2: .5, -1: 1., 0: 1., 1: 1., 2: .5}
    t = sum(wk * np.roll(fine, -k, axis=0) for k, wk in wts.items()) / 4.0
    t = t[::4, :]
    t = sum(wk * np.roll(t, -k, axis=1) for k, wk in wts.items()) / 4.0
    return t[:, ::4]


def synth_shift(fine, dx, dy):
    """Shift image content by (dx,dy) original pixels (multiples of 0.25) on the fine grid: fine index shift (4dx,4dy)."""
    sx, sy = int(round(dx * 4)), int(round(dy * 4))
    return np.roll(np.roll(fine, sy, axis=0), sx, axis=1)


def synth_rotate(fine, deg):
    h, w = fine.shape; M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(fine, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def run_synth():
    (OUT / 'synth').mkdir(parents=True, exist_ok=True)
    p = folder('260904_p50_repeat'); a = M60.read(p / '1.tif'); b = M60.read(p / '2.tif')
    fine = upsample4(b)
    rows = []; geo = []
    jobs = [(f'shift_{dx:.2f}_{dy:.2f}', dx, dy, 0.0) for dx in (0, .25, .5, .75) for dy in (0, .25, .5, .75)] + [('rot_0.1', 0, 0, 0.1), ('rot_0.3', 0, 0, 0.3)]
    for label, dx, dy, rot in jobs:
        t0 = time.time()
        f2 = synth_shift(fine, dx, dy) if rot == 0 else synth_rotate(fine, rot)
        bsyn = integrate4(f2).astype(np.float32)
        res = P.analyse_pair(a, bsyn); np.savez_compressed(OUT / 'synth' / f'{label}.npz', **{k: v for k, v in res.items() if isinstance(v, np.ndarray) and k not in ('pre_dict',)})
        geo.append(dict(label=label, dx=dx, dy=dy, rot=rot, A=json.dumps(res['A'].tolist()), corner_disagree_px=res['corner_disagree_px'], frac_corr_px=res['frac_corr_px'], ok_frac=float(res['ok'].mean())))
        rows += metrics_for(res, label); rows += local_thresh_metrics(res, label)
        print('done', label, f'{time.time() - t0:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'synth_pairs_metrics.csv', index=False); pd.DataFrame(geo).to_csv(OUT / 'synth_pairs_geometry.csv', index=False)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    {'real': run_real, 'synth': run_synth}[sys.argv[1]]()
