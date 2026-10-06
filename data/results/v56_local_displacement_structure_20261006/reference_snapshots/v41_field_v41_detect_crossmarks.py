"""v41 step 0: detect crossmark (dark long straight lines) in the pre-wash image of every accepted field.
Method: 2x area-downsample, local-contrast normalisation (divide by Gaussian sigma=20 background, minus 1),
take absolute value (dark core + bright flanks), then for a family of tilts shear the map and average along the line
direction; the 1-D profile peak (max over tilt) gives the line position and tilt. Robust z = (peak-median)/(1.4826*MAD).
Does NOT re-run registration."""
import sys
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2, tifffile
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import field_v41_common as C
DS = 2
SL = np.linspace(-0.10, 0.10, 81)

def energy_map(path):
    a = tifffile.imread(path).astype(np.float32)
    a = cv2.resize(a, (C.W // DS, C.H // DS), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (0, 0), 1.5)
    bg = cv2.GaussianBlur(a, (0, 0), 20)
    return np.abs(a / bg - 1.0)

def shear_profile(m, s):
    h, w = m.shape; yc = (h - 1) / 2
    # x' = x - s*(y-yc)  : a line x = x0 + s*(y-yc) becomes x' = x0
    M = np.float32([[1, -s, s * yc], [0, 1, 0]])
    wm = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    wo = cv2.warpAffine(np.ones_like(m), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    cnt = wo.sum(0)
    p = np.where(cnt > 0.8 * h, wm.sum(0) / np.maximum(cnt, 1), np.nan)
    return p

def best_line(m):
    def score(s):
        p = shear_profile(m, s); k = np.nan_to_num(cv2.blur(p.reshape(1, -1), (5, 1)).ravel(), nan=0)
        return k.max(), p
    sc = [(score(s)[0], s) for s in SL]
    s0 = max(sc)[1]
    fine = np.arange(s0 - 0.0025, s0 + 0.00251, 0.0005)
    sc = [(score(s)[0], s) for s in fine]
    s1 = max(sc)[1]
    p = shear_profile(m, s1)
    return s1, p

def analyse_profile(p):
    pv = p[np.isfinite(p)]
    med = np.median(pv); mad = 1.4826 * np.median(np.abs(pv - med)) + 1e-9
    k = np.nan_to_num(cv2.blur(p.reshape(1, -1), (5, 1)).ravel(), nan=-1)
    x0 = int(np.argmax(k)); pk = k[x0]
    z = (pk - med) / mad
    # cluster: contiguous extent around the peak above half height (relative to median)
    thr = med + 0.5 * (pk - med)
    lo = x0
    while lo > 0 and k[lo - 1] > thr: lo -= 1
    hi = x0
    while hi < len(k) - 1 and k[hi + 1] > thr: hi += 1
    # local maxima >= half height, spaced >= 5 ds px (10 px original), within +-60 ds px
    cand = [i for i in range(max(1, x0 - 60), min(len(k) - 1, x0 + 61)) if k[i] >= k[i - 1] and k[i] > k[i + 1] and k[i] > thr]
    keep = []
    for i in sorted(cand, key=lambda i: -k[i]):
        if all(abs(i - j) >= 5 for j in keep): keep.append(i)
    # sub-pixel centroid of the peak (+-2 ds px)
    sl = slice(max(0, x0 - 2), min(len(k), x0 + 3)); w = np.clip(k[sl] - med, 0, None)
    xc = float((np.arange(len(k))[sl] * w).sum() / w.sum()) if w.sum() > 0 else float(x0)
    return dict(pos=(xc + 0.5) * DS - 0.5, peak=float(pk), median=float(med), mad=float(mad), z=float(z),
                width_px=float((hi - lo + 1) * DS), n_peaks=len(keep))

def segment_coverage(m, s, x0ds, vertical, nseg=8):
    """fraction of segments along the line whose local mean energy exceeds the field median + 3*MAD-like margin."""
    h, w = m.shape; yc = (h - 1) / 2
    ys = np.arange(h); xs = x0ds + s * (ys - yc)
    vals = []
    for seg in np.array_split(np.arange(h), nseg):
        v = []
        for y in seg[::2]:
            x = int(round(xs[y]))
            if 2 <= x < w - 2: v.append(m[y, x - 2:x + 3].mean())
        vals.append(np.mean(v) if v else np.nan)
    return np.array(vals)

def detect(row):
    try:
        m = energy_map(row['pre_path'])
        res = dict(field_id=row['field_id'])
        bg = float(np.median(m))
        for name, mm, vert in (('v', m, True), ('h', np.ascontiguousarray(m.T), False)):
            s, p = best_line(mm)
            r = analyse_profile(p)
            segs = segment_coverage(mm, s, (r['pos'] + 0.5) / DS - 0.5, vert)
            segthr = r['median'] + 0.5 * (r['peak'] - r['median'])
            cov = float(np.nanmean(segs > r['median'] + 0.3 * (r['peak'] - r['median'])))
            for k_, v_ in r.items(): res[f'{name}_{k_}'] = v_
            res[f'{name}_slope'] = float(s); res[f'{name}_coverage'] = cov
        # geometry in full-res image coordinates (x right, y down)
        # vertical: x = v_pos + v_slope*(y-yc) ; horizontal: y = h_pos + h_slope*(x-xc)
        res['ok'] = True
        return res
    except Exception as e:
        return dict(field_id=row['field_id'], ok=False, error=repr(e))

if __name__ == '__main__':
    q = C.load_fields()
    rows = q[['field_id', 'pre_path']].to_dict('records')
    with Pool(4) as pool: out = pool.map(detect, rows, chunksize=4)
    df = pd.DataFrame(out)
    df.to_csv(C.OUT / 'cache' / 'crossmark_raw_detection.csv', index=False)
    print(df.ok.value_counts()); print(df[['v_z', 'h_z']].describe())
