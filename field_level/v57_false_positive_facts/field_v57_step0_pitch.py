"""v57 step 0-2c: actual lattice periodicity of every pre image versus the stored ideal lattice (pitch 7.286).
Zoomed DFT of (img - gauss51) around the three nominal first-order harmonics; the peak position gives the actual k_j.
Output: per field, actual nearest-neighbour pitch from |k_j|, mismatch vs 7.286, and the lattice phase slip across the image
(px that a stored sample point slides relative to the real pillar pattern between the image centre and the corners)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2, tifffile
import field_v57_common as C

NOM = 7.286
GRID = np.arange(-0.080, 0.0801, 0.002)  # rad/px search half-width around the nominal harmonic (~4 percent of |k| ~ 1)

def zoom_dft(c, kx0, ky0, w):
    H, W = c.shape
    x = np.arange(W, dtype=np.float64); y = np.arange(H, dtype=np.float64)
    kxs = kx0 + GRID; kys = ky0 + GRID
    Ex = np.exp(-1j*np.outer(x, kxs)).astype(np.complex64)      # W x n
    Ey = np.exp(-1j*np.outer(y, kys)).astype(np.complex64)      # H x n
    T = (c*w).astype(np.float32).astype(np.complex64) @ Ex      # H x n
    S = Ey.T @ T                                                 # n_ky x n_kx
    return np.abs(S), kxs, kys

def refine(A, kxs, kys):
    j, i = np.unravel_index(np.argmax(A), A.shape)
    def par(m1, m0, p1):
        d = m1-2*m0+p1
        return 0.0 if d == 0 else 0.5*(m1-p1)/d
    di = par(A[j, i-1], A[j, i], A[j, i+1]) if 0 < i < A.shape[1]-1 else 0.0
    dj = par(A[j-1, i], A[j, i], A[j+1, i]) if 0 < j < A.shape[0]-1 else 0.0
    step = GRID[1]-GRID[0]
    return kxs[i]+di*step, kys[j]+dj*step, A[j, i], (i, j)

def one(r):
    fid, path, a1x, a1y, a2x, a2y = r
    B = np.array([[a1x, a2x], [a1y, a2y]]); K = 2*np.pi*np.linalg.inv(B)   # rows b1,b2 with b_i . a_j = 2 pi delta_ij
    cand = sorted([K[0], K[1], K[0]+K[1], K[0]-K[1]], key=np.linalg.norm)[:3]
    im = tifffile.imread(path).astype(np.float32)/65535.0
    c = im - cv2.GaussianBlur(im, (51, 51), 0)
    H, W = c.shape
    w = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    out = dict(fid=fid)
    ks = []
    for j, k in enumerate(cand, 1):
        A, kxs, kys = zoom_dft(c, k[0], k[1], w)
        kx, ky, amp, (i, jj) = refine(A, kxs, kys)
        edge = (i in (0, A.shape[1]-1)) or (jj in (0, A.shape[0]-1))
        ks.append((kx, ky))
        out[f'k{j}_nom'] = np.linalg.norm(k); out[f'k{j}_act'] = np.hypot(kx, ky)
        out[f'dkx{j}'] = kx-k[0]; out[f'dky{j}'] = ky-k[1]; out[f'amp{j}'] = amp; out[f'edge{j}'] = bool(edge)
    # actual nearest-neighbour pitch from the mean harmonic length; hex: pitch = 4 pi / (sqrt(3) |k|)
    kn = np.mean([out[f'k{j}_nom'] for j in (1, 2, 3)]); ka = np.mean([out[f'k{j}_act'] for j in (1, 2, 3)])
    out['pitch_nom'] = 4*np.pi/(np.sqrt(3)*kn); out['pitch_act'] = 4*np.pi/(np.sqrt(3)*ka)
    out['pitch_ratio'] = out['pitch_act']/out['pitch_nom']
    # phase slip of the stored lattice relative to the actual pattern, evaluated at the four image corners
    # relative to the image centre: slip_j = (dk_j . (x - x_centre)) / |k_j|  (px along the harmonic's wave vector)
    corners = np.array([[-W/2, -H/2], [W/2, -H/2], [-W/2, H/2], [W/2, H/2]])
    slip = []
    for j, k in enumerate(cand):
        dk = np.array([out[f'dkx{j+1}'], out[f'dky{j+1}']])
        slip.append(np.abs(corners@dk)/np.linalg.norm(k))
    out['slip_px_corner_max'] = float(np.max(slip))
    out['beat_len_px'] = float(2*np.pi/np.max([np.hypot(out[f'dkx{j}'], out[f'dky{j}']) for j in (1, 2, 3)]) ) if np.max([np.hypot(out[f'dkx{j}'], out[f'dky{j}']) for j in (1, 2, 3)]) > 0 else np.inf
    return out

if __name__ == '__main__':
    t = pd.read_csv(C.OUT/'step0'/'fields_inventory.csv', encoding='utf-8-sig', dtype={'board': str})
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(t)
    args = [(r.fid, r.pre_path, r.a1x, r.a1y, r.a2x, r.a2y) for r in t.iloc[:n].itertuples()]
    with Pool(min(8, os.cpu_count() or 2)) as p:
        res = p.map(one, args, chunksize=4)
    o = pd.DataFrame(res).merge(t[['fid', 'date', 'board', 'field']], on='fid')
    o.to_csv(C.OUT/'step0'/('actual_pitch_by_field.csv' if n == len(t) else 'actual_pitch_test.csv'), index=False)
    print(o.groupby('date').agg(pitch_act_med=('pitch_act', 'median'), pitch_act_min=('pitch_act', 'min'), pitch_act_max=('pitch_act', 'max'),
                                ratio_med=('pitch_ratio', 'median'), slip_med=('slip_px_corner_max', 'median'), slip_max=('slip_px_corner_max', 'max')).round(4))
    print('edge flags (peak at search boundary):', int(o[[f'edge{j}' for j in (1, 2, 3)]].any(axis=1).sum()), 'of', len(o))
