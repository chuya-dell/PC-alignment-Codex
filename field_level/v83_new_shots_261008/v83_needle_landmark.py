"""v83 part A-4 (3/3): residual displacement of the scar landmarks after alignment.  The scar was NOT an input of the 'masked' alignment (v83_needle_align.py).
For each pair (i earlier, j later) and map M (C1 lattice map L, or the feature affine A): image j is warped into the frame of i (Lanczos4, inverse map), both are band-passed (DoG sigma 3 - 30 px;
the pillar lattice, period 7.3 px, is suppressed), and for each landmark window the residual translation delta is found by Lucas-Kanade (translation, cubic sampling, phase-correlation start).
delta = position of the landmark in the warped j minus its position in i (pixels, frame of i).  Landmarks: needle gouge (9 tiles + whole + tip), cross-scar bands (left vertical, bottom horizontal;
the component along a line is weakly constrained).
Validation: synthetic known sub-pixel shifts of 6.tif with independent read noise, through the same chain.
usage: python v83_needle_landmark.py [synth|pairs|all]"""
import sys, json, itertools
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, cv2, tifffile, pandas as pd
RAW = Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu\261008_p50_傷'); OUT = ROOT / 'data/results/v83_new_shots_261008/needle'
NAMES = ['6', '6-1', '6-2', '6-3', '6-4']
H, W = 2044, 2048


def bandpass(im):
    f = im.astype(np.float32) / 65535
    return (cv2.GaussianBlur(f, (0, 0), 3) - cv2.GaussianBlur(f, (0, 0), 30)).astype(np.float32)


def warp_into_i(imj, M):
    """dst(x) = imj(M x) with M = [L | t] mapping i->j coordinates."""
    return cv2.warpAffine(imj, np.asarray(M, np.float64), (W, H), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def lk_translation(ia, ib, box, iters=30):
    """box=(x0,y0,x1,y1) in the frame of ia.  Find d with ia(x) ~ ib(x+d).  Returns dict(dx, dy, se_x, se_y, n, ncc) or None."""
    x0, y0, x1, y1 = box
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    t = ia[y0:y1, x0:x1].astype(np.float64)
    if t.std() < 1e-4:
        return None
    pad = 40; ya, yb, xa, xb = max(y0 - pad, 0), min(y1 + pad, H), max(x0 - pad, 0), min(x1 + pad, W)
    d0 = np.zeros(2)
    try:
        sa = ia[ya:yb, xa:xb]; sb = ib[ya:yb, xa:xb]
        win = cv2.createHanningWindow((sa.shape[1], sa.shape[0]), cv2.CV_32F)
        (sx, sy), _ = cv2.phaseCorrelate(sa.astype(np.float32), sb.astype(np.float32), win)
        d0 = np.array([sx, sy])
    except Exception:
        pass
    gx = cv2.Sobel(ib, cv2.CV_32F, 1, 0, ksize=3) / 8.0; gy = cv2.Sobel(ib, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    best = None
    for start in (d0, -d0, np.zeros(2)):
        dd = start.copy()
        for it in range(iters):
            mx = xs + np.float32(dd[0]); my = ys + np.float32(dd[1])
            if mx.min() < 2 or my.min() < 2 or mx.max() > W - 3 or my.max() > H - 3:
                dd = None; break
            s = cv2.remap(ib, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float64)
            sgx = cv2.remap(gx, mx, my, cv2.INTER_CUBIC).astype(np.float64); sgy = cv2.remap(gy, mx, my, cv2.INTER_CUBIC).astype(np.float64)
            e = t - s
            G = np.array([[np.sum(sgx * sgx), np.sum(sgx * sgy)], [np.sum(sgx * sgy), np.sum(sgy * sgy)]]); b = np.array([np.sum(sgx * e), np.sum(sgy * e)])
            step = np.linalg.solve(G + 1e-12 * np.eye(2), b)
            dd = dd + step
            if np.abs(step).max() < 1e-4:
                break
        if dd is None:
            continue
        mx = xs + np.float32(dd[0]); my = ys + np.float32(dd[1])
        s = cv2.remap(ib, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float64); e = t - s
        sgx = cv2.remap(gx, mx, my, cv2.INTER_CUBIC).astype(np.float64); sgy = cv2.remap(gy, mx, my, cv2.INTER_CUBIC).astype(np.float64)
        G = np.array([[np.sum(sgx * sgx), np.sum(sgx * sgy)], [np.sum(sgx * sgy), np.sum(sgy * sgy)]])
        rms = np.sqrt(np.mean(e ** 2)); cov = np.linalg.inv(G + 1e-12 * np.eye(2)) * rms ** 2
        ncc = np.corrcoef(t.ravel(), s.ravel())[0, 1]
        if best is None or rms < best[0]:
            best = (rms, dd, cov, ncc)
    if best is None:
        return None
    rms, dd, cov, ncc = best
    return dict(dx=float(dd[0]), dy=float(dd[1]), se_x=float(np.sqrt(cov[0, 0])), se_y=float(np.sqrt(cov[1, 1])), n=int(t.size), ncc=float(ncc))


def landmarks(gouge_bbox):
    x0, y0, x1, y1 = gouge_bbox
    L = [('gouge_whole', 'gouge', (x0 + 10, y0 + 10, x1 - 10, y1 - 10))]
    tw = (x1 - x0) // 3; th = (y1 - y0) // 3
    for r in range(3):
        for c in range(3):
            L.append((f'gouge_t{r}{c}', 'gouge', (x0 + c * tw + 8, y0 + r * th + 8, x0 + (c + 1) * tw - 8, y0 + (r + 1) * th - 8)))
    L.append(('gouge_tip', 'gouge', (x1 - 190, y0 - 6, x1 + 10, y0 + 150)))
    for k, yy in enumerate(range(100, 1700, 300)):
        L.append((f'vline_{k}', 'vline', (4, yy, 140, yy + 300)))
    for k, xx in enumerate(range(200, 1900, 350)):
        L.append((f'hline_{k}', 'hline', (xx, 1790, xx + 350, 1940)))
    return L


def measure(ia, ib, lms):
    rows = []
    for name, cls, box in lms:
        box = tuple(int(v) for v in box)
        if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
            continue
        r = lk_translation(ia, ib, box)
        if r:
            rows.append(dict(landmark=name, cls=cls, cx=(box[0] + box[2]) / 2, cy=(box[1] + box[3]) / 2, **r))
    return rows


def run_pairs():
    info = json.load(open(OUT / 'align_mask_info.json'))
    bp = {n: bandpass(tifffile.imread(RAW / f'{n}.tif').astype(np.float32)) for n in NAMES}
    rows = []
    for i, j in itertools.combinations(range(5), 2):
        for var in ('masked', 'plain'):
            f = OUT / 'align' / f'{NAMES[i]}__{NAMES[j]}__{var}.npz'
            if not f.exists():
                continue
            z = np.load(f); lms = landmarks(info[NAMES[i]]['gouge_bbox'])
            for mapname, M in (('L', np.hstack([z['Llin'], z['tL'][:, None]])), ('A', z['A'])):
                wj = warp_into_i(bp[NAMES[j]], M)
                for r in measure(bp[NAMES[i]], wj, lms):
                    rows.append(dict(i=NAMES[i], j=NAMES[j], variant=var, map=mapname, **r))
            print('done', NAMES[i], NAMES[j], var, flush=True)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'landmark_residuals.csv', index=False)
    return t


def run_synth():
    """Known shift of 6.tif (exact, Fourier), independent read noise on both; mapping = true shift + error e.  Expected measured delta = -e (checks the whole chain incl. the Lanczos warp)."""
    info = json.load(open(OUT / 'align_mask_info.json')); a = tifffile.imread(RAW / '6.tif').astype(np.float32)
    rng = np.random.default_rng(1); rows = []
    lms = landmarks(info['6']['gouge_bbox']); sig = 358.0   # read-noise SD in ADU (dark frames)
    ia = bandpass(a + rng.normal(0, sig, a.shape).astype(np.float32))
    F = np.fft.fft2(a); fy = np.fft.fftfreq(a.shape[0])[:, None]; fx = np.fft.fftfreq(a.shape[1])[None, :]
    for (sx, sy, ex, ey) in [(0.0, 0.0, 0.0, 0.0), (3.37, -2.81, 0.0, 0.0), (3.37, -2.81, 0.05, -0.03), (10.62, 4.18, 0.0, 0.0), (10.62, 4.18, -0.2, 0.1), (-20.25, 7.9, 0.01, 0.01)]:
        b = np.fft.ifft2(F * np.exp(-2j * np.pi * (fx * sx + fy * sy))).real.astype(np.float32)
        ib = bandpass(b + rng.normal(0, sig, a.shape).astype(np.float32))
        M = np.array([[1, 0, sx + ex], [0, 1, sy + ey]], float)
        wj = warp_into_i(ib, M)
        for r in measure(ia, wj, lms):
            rows.append(dict(sx=sx, sy=sy, ex=ex, ey=ey, **r))
    t = pd.DataFrame(rows); t.to_csv(OUT / 'landmark_synthetic.csv', index=False)
    return t


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    pd.set_option('display.width', 250)
    if which in ('synth', 'all'):
        s = run_synth(); s['err_x'] = s.dx + s.ex; s['err_y'] = s.dy + s.ey
        print(s.groupby(['sx', 'sy', 'ex', 'ey', 'cls']).agg(dx_med=('dx', 'median'), dy_med=('dy', 'median'), errx_med=('err_x', 'median'), erry_med=('err_y', 'median'), errx_sd=('err_x', 'std'), erry_sd=('err_y', 'std'), n=('dx', 'size')).round(4).to_string())
    if which in ('pairs', 'all'):
        run_pairs()
