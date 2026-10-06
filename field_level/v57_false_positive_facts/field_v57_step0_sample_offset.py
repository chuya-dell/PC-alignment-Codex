"""v57 step 0-2d: direct image check. For 5x5 windows (128 px), mean of (img-gauss51) at stored xy + delta (delta on a 0.5-px grid, +-5 px)
-> the location of the extremum nearest the origin tells where the real pillar pattern sits relative to the stored sample points."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, cv2, tifffile
from scipy.ndimage import map_coordinates
import field_v57_common as C
t = pd.read_csv(C.OUT/'step0'/'fields_inventory.csv', encoding='utf-8-sig', dtype={'board': str})
D = np.arange(-5, 5.01, 0.5)
def run(fid):
    r = t[t.fid == fid].iloc[0]
    d, xy, ids = C.load_cache(fid)
    im = tifffile.imread(r.pre_path).astype(np.float32)/65535.0
    c = im - cv2.GaussianBlur(im, (51, 51), 0); c = cv2.GaussianBlur(c, (0, 0), 0.6)
    H, W = c.shape; out = np.zeros((5, 5, 2)); amp = np.zeros((5, 5))
    for bj, yc in enumerate(np.linspace(150, H-150, 5)):
        for bi, xc in enumerate(np.linspace(150, W-150, 5)):
            sel = (np.abs(xy[:, 0]-xc) < 64) & (np.abs(xy[:, 1]-yc) < 64); p = xy[sel]
            m = np.zeros((len(D), len(D)))
            for j, dy in enumerate(D):
                for i, dx in enumerate(D):
                    m[j, i] = map_coordinates(c, [p[:, 1]+dy, p[:, 0]+dx], order=1, mode='nearest').mean()
            m0 = m-np.median(m)
            sgn = 1 if m0.max() > -m0.min() else -1   # pillar polarity from the larger excursion
            z = sgn*m0
            # local maxima of z; take the one nearest the origin among those with z > 0.6*max
            from scipy.ndimage import maximum_filter
            mx = (z == maximum_filter(z, size=5)) & (z > 0.6*z.max())
            jj, ii = np.nonzero(mx); dist = np.hypot(D[ii], D[jj]); k = np.argmin(dist)
            out[bj, bi] = (D[ii[k]], D[jj[k]]); amp[bj, bi] = z.max()
    return out
for fid in ['260922_7_2', '260825_0_1', '260927_9_2']:
    o = run(fid)
    print(fid, '(offset of nearest real pillar centre from stored sample point, px; rows=y windows top->bottom, cols=x left->right)')
    for bj in range(5): print('   ', '  '.join('(%+.1f,%+.1f)' % tuple(o[bj, bi]) for bi in range(5)))
    print('   |offset| centre %.2f, mean of 4 corners %.2f' % (np.hypot(*o[2, 2]), np.mean([np.hypot(*o[a, b]) for a in (0, 4) for b in (0, 4)])))
