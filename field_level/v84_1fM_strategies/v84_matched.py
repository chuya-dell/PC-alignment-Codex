"""v84 part B (c): matched filter vs the area aperture (S5P-type readout: pre at the actual centre, post at the P position), on the 48 reshoot pairs of 2026-10-08.
MF = Gaussian-weighted mean of the background-subtracted contrast image (weights sigma_w = 1.0 / 1.3 / 1.6 px; the pillar image width is 1.28-1.30 px) at the same positions as the aperture.
Both are scaled to the same response to a unit-amplitude Gaussian pillar image (sigma_s=1.3, centred, numerically), so the delta SD is directly the inverse SNR.
Output: the SD ratio, the analytic white-noise bound, tail counts.   usage: python v84_matched.py"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v72_real_field_readout', 'field_level/v70_ledger_center_fit', 'field_level/v60_band_scar_controls'):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd, tifffile, cv2
from scipy import ndimage
import v72_readout as RD
RAW = Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu\261008_p50_z'); R = ROOT / 'data/results/v83_new_shots_261008'; OUT = ROOT / 'data/results/v84_1fM_strategies'
SW = (1.0, 1.3, 1.6)


def sample(c, xy):
    return ndimage.map_coordinates(c, [xy[:, 1], xy[:, 0]], order=3, mode='nearest')


def responses():
    """response of aperture (area integral, as RD.aper) and Gaussian-weighted mean to a unit Gaussian pillar image sigma_s=1.3 centred"""
    n = 41; yy, xx = np.mgrid[-20:21, -20:21].astype(np.float32); im = np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.3 ** 2)).astype(np.float32)
    ap = RD.aper(im, np.array([[20.0, 20.0]]))[0]
    mf = {s: float(cv2.GaussianBlur(im, (0, 0), s)[20, 20]) for s in SW}
    return float(ap), mf


def main():
    ap_resp, mf_resp = responses(); print('response to unit pillar: aperture', ap_resp, 'MF', mf_resp)
    rows = []
    for b in (1, 2, 3):
        for p in range(1, 9):
            im = {k: tifffile.imread(RAW / f'{b}-{p}-{k}.tif').astype(np.float32) for k in (0, 1, 3)}
            con = {k: RD.plain_contrast(v) for k, v in im.items()}
            for kind in (1, 3):
                z = np.load(R / 'pairs' / f'{b}-{p}_{kind}.npz'); ctr = z['ctr_xy'].astype(np.float64) if 'ctr_xy' in z.files else None
                ctr = z['ctr_xy'].astype(np.float64); pos = z['posP'].astype(np.float64); okP = z['okP']; sd = z['sd_periods']; ex = np.isfinite(sd) & (sd > 2) & okP
                ok2 = ex & np.all(np.isfinite(pos), axis=1)
                dap = RD.aper(con[0], ctr) - RD.aper(con[kind], pos)
                row = dict(board=b, pos=p, kind=f'-0/-{kind}', n=int(ok2.sum()))
                def stat(d, nm):
                    v = d[ok2]; v = v[np.isfinite(v)]; med = np.median(v); s = 1.4826 * np.median(abs(v - med)); row[nm + '_sig'] = s
                    for k in (4, 6, 8): row[f'{nm}_n{k}'] = int((abs(v - med) > k * s).sum())
                stat(dap, 'ap')
                for s in SW:
                    c0 = cv2.GaussianBlur(con[0], (0, 0), s); ck = cv2.GaussianBlur(con[kind], (0, 0), s)
                    d = (sample(c0, ctr) - sample(ck, pos)) * (ap_resp / mf_resp[s]); stat(d, f'mf{s}')
                rows.append(row)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'B1_matched_filter.csv', index=False)
    for s in SW: t[f'ratio{s}'] = t[f'mf{s}_sig'] / t.ap_sig
    pd.set_option('display.width', 250)
    print(t.groupby('kind')[['ap_sig'] + [f'mf{s}_sig' for s in SW] + [f'ratio{s}' for s in SW]].median().round(4).to_string())
    print(t.groupby('kind')[[f'ap_n4'] + [f'mf{s}_n4' for s in SW] + ['ap_n6'] + [f'mf{s}_n6' for s in SW]].mean().round(2).to_string())
    # white-noise bound for the SNR gain (Gaussian pillar sigma_s = 1.3): aperture (disc 1.69) vs matched
    sig_s = 1.3; r = 1.69
    snr_ap = 2 * np.pi * sig_s ** 2 * (1 - np.exp(-r ** 2 / (2 * sig_s ** 2))) / np.sqrt(np.pi * r ** 2); snr_mf = np.sqrt(np.pi) * sig_s
    print('white-noise SNR gain bound MF/aperture:', round(float(snr_mf / snr_ap), 3))


if __name__ == '__main__':
    main()
