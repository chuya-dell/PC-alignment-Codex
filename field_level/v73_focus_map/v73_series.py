"""v73 step D (part 1): focus series 260901_p50_Zstep (31 images).  Per-image sharpness metrics (Laplacian variance, lattice fundamental amplitude,
fitted pillar amplitude and width) and the 30 adjacent-pair false-positive counts under S0/S3/S5.  Motor command values are NOT used as true moves;
the file name is used only as the ordinal axis.  usage: python v73_series.py [metrics|pairs]"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v75_zero_truth_pairs'))
import numpy as np, pandas as pd, cv2
import v75_pair_lib as P
import v75_run_pairs as RP
import field_control_common as M60
OUT = ROOT / 'data/results/v73_focus_map'


def files():
    p = M60.folder('260901_p50_Zstep')
    return sorted([f for f in p.iterdir() if f.suffix.lower() == '.tif'], key=lambda f: float(f.stem))


def image_metrics(im):
    f = im.astype(np.float32) / 65535
    lap = cv2.Laplacian(cv2.GaussianBlur(f, (0, 0), 1.0), cv2.CV_32F)
    c = f - cv2.GaussianBlur(f, (51, 51), 0)
    win = np.outer(np.hanning(f.shape[0]), np.hanning(f.shape[1]))
    F = np.abs(np.fft.fftshift(np.fft.fft2(c * win)))
    fy = np.fft.fftshift(np.fft.fftfreq(f.shape[0])); fx = np.fft.fftshift(np.fft.fftfreq(f.shape[1])); rad = np.hypot(fy[:, None], fx[None, :])
    tg = 2 / (np.sqrt(3) * 7.35); ring = (rad > tg * .94) & (rad < tg * 1.06)
    sharp_ring = float(F[ring].max())
    try:
        cf = P.C.centers_smooth(im)
        ok = cf['amp'] > 0.4 * cf['aloc']
        amp = float(np.median(cf['amp'][ok])); wid = float(np.median(cf['wid'][ok])); n_ok = float(ok.mean())
    except Exception:
        amp = wid = n_ok = np.nan
    return dict(lap_var=float(lap.var()), contrast_sd=float(c.std()), lattice_peak=sharp_ring, pillar_amp=amp, pillar_width=wid, ok_fraction=n_ok)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fl = files(); mode = sys.argv[1] if len(sys.argv) > 1 else 'metrics'
    if mode == 'metrics':
        rows = []
        for f in fl:
            im = M60.read(f); r = image_metrics(im); r.update(file=f.name, index=float(f.stem), mtime=f.stat().st_mtime); rows.append(r); print('metrics', f.name, flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'zstep_image_metrics.csv', index=False)
    else:
        rows = []; geo = []
        prev = M60.read(fl[0]); pool = {}
        results = []
        for i in range(1, len(fl)):
            b = M60.read(fl[i]); t0 = time.time()
            try:
                res = P.analyse_pair(prev, b, A_override=np.array([[1., 0, 0], [0, 1., 0]]))
                results.append((fl[i - 1].stem, fl[i].stem, res))
                geo.append(dict(a=fl[i - 1].stem, b=fl[i].stem, corner_disagree_px=res['corner_disagree_px'], ok_frac=float(res['ok'].mean()), sub_pre=res['sub_pre'], sub_post=res['sub_post'], seconds=time.time() - t0))
                rows += RP.metrics_for(res, f'{fl[i - 1].stem}_{fl[i].stem}')
            except Exception as e:
                geo.append(dict(a=fl[i - 1].stem, b=fl[i].stem, error=repr(e)))
            print('pair', fl[i - 1].stem, fl[i].stem, f'{time.time() - t0:.0f}s', flush=True)
            prev = b
            pd.DataFrame(rows).to_csv(OUT / 'zstep_pairs_metrics.csv', index=False); pd.DataFrame(geo).to_csv(OUT / 'zstep_pairs_geometry.csv', index=False)


if __name__ == '__main__':
    main()
