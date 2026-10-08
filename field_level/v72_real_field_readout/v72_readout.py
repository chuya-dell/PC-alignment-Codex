"""v72 step C (data part): re-read the real fields with several reading schemes (same pillars, same images).

Readouts (delta = pre - post, contrast units of the current standard: 3x3 sum minus 3x3 background sum):
  S0_std        current standard (stored by v70 run: std_delta)
  S1_idealL     ideal grid, integer rounding, post position through the lattice-implied physical map (L)
  S2_idealL_bil ideal grid, bilinear on the box contrast, post through L
  S3_ctr_round  actual fitted pillar centres, integer rounding
  S4_ctr_bil    actual centres, bilinear on box contrast
  S5_ctr_aper   actual centres, area-weighted disc aperture (r=1.69 px = 9 px area; 0.25 px sub-grid) on background-subtracted image
Centres and the physical numbering come from v70 (fields2/).  usage: python v72_readout.py [--workers K] [--limit N]"""
from __future__ import annotations
import sys, json, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit'))
import numpy as np, pandas as pd, cv2
from scipy import ndimage
import v70_lib as L
import v70_centers2 as C
V70 = L.ROOT / 'data/results/v70_ledger_center_fit'
OUT = ROOT / 'data/results/v72_real_field_readout'
MASKDIR = Path(r'C:\Users\chuya\PC-alignment-localcorr\data\results\v57_false_positive_facts_20261006\stepD')   # read-only use (v57 provisional scar masks)
H, W = 2044, 2048
_off = np.mgrid[-1.7:1.7001:0.25, -1.7:1.7001:0.25].reshape(2, -1).T
_off = _off[np.hypot(_off[:, 0], _off[:, 1]) <= 1.69]; AP_OFF = _off; AP_W = 0.25 ** 2


def box_contrast(im):
    f = im.astype(np.float32) / 65535
    return cv2.boxFilter(f - cv2.GaussianBlur(f, (51, 51), 0), -1, (3, 3), normalize=False)


def plain_contrast(im):
    f = im.astype(np.float32) / 65535
    return f - cv2.GaussianBlur(f, (51, 51), 0)


def bil(c, xy):
    v = np.full(len(xy), np.nan, np.float32)
    ok = (xy[:, 0] >= 1) & (xy[:, 0] < W - 2) & (xy[:, 1] >= 1) & (xy[:, 1] < H - 2)
    v[ok] = ndimage.map_coordinates(c, [xy[ok, 1], xy[ok, 0]], order=1, mode='nearest')
    return v


def rnd(c, xy):
    v = np.full(len(xy), np.nan, np.float32)
    ok = (xy[:, 0] >= 1) & (xy[:, 0] < W - 2) & (xy[:, 1] >= 1) & (xy[:, 1] < H - 2)
    q = np.rint(xy[ok]).astype(int); v[ok] = c[q[:, 1], q[:, 0]]
    return v


def aper(c, xy):
    v = np.full(len(xy), np.nan, np.float32)
    ok = (xy[:, 0] >= 3) & (xy[:, 0] < W - 4) & (xy[:, 1] >= 3) & (xy[:, 1] < H - 4)
    p = xy[ok]; acc = np.zeros(len(p), np.float64)
    for dy, dx in AP_OFF:
        acc += ndimage.map_coordinates(c, [p[:, 1] + dy, p[:, 0] + dx], order=1, mode='nearest')
    v[ok] = (acc * AP_W).astype(np.float32); return v


def load_mask(fid):
    p = MASKDIR / f'{fid}_mask.npz'
    if not p.exists(): return None
    with np.load(p) as z:
        shape = tuple(z['shape']); m = np.unpackbits(z['mask'])[:np.prod(shape)].reshape(shape).astype(bool)
    return m


def one(rec):
    fid = rec['fid']; dst = OUT / 'fields' / f'{fid}.npz'
    if dst.exists(): return dict(fid=fid, ok=True, cached=True)
    t0 = time.time()
    try:
        z1 = np.load(V70 / 'fields' / f'{fid}.npz'); z2 = np.load(V70 / 'fields2' / f'{fid}.npz')
        pre = L.read(rec['pre_path']); post = L.read(rec['post_path'])
        ca, cb = box_contrast(pre), box_contrast(post)
        pa, pb = plain_contrast(pre), plain_contrast(post)
        Llin, tL = z2['Llin'], z2['tL']
        xy = z1['std_xy'].astype(np.float64); xyL = xy @ Llin.T + tL
        S1 = rnd(ca, xy) - rnd(cb, xyL); S2 = bil(ca, xy) - bil(cb, xyL)
        pre_d = dict(ctr=z2['pre_ctr'].astype(float), pred=z2['pre_pred'], amp=z2['pre_amp'], wid=z2['pre_wid'], aloc=z2['pre_aloc'])
        post_d = dict(ctr=z2['post_ctr'].astype(float), pred=z2['post_pred'], amp=z2['post_amp'], wid=z2['post_wid'], aloc=z2['post_aloc'])
        Sp = C.sub_flags(pre_d, 'main'); So = C.sub_flags(post_d, 'main'); jL = z2['jL']
        expL = z2['pre_pred_lin'] @ Llin.T + tL
        ov = (expL[:, 0] >= 8) & (expL[:, 0] < W - 8) & (expL[:, 1] >= 8) & (expL[:, 1] < H - 8)
        ok = (~Sp) & (~So[jL]) & z2['consL'] & ov
        a = pre_d['ctr']; b = post_d['ctr'][jL]
        S3 = rnd(ca, a) - rnd(cb, b); S4 = bil(ca, a) - bil(cb, b); S5 = aper(pa, a) - aper(pb, b)
        for S in (S3, S4, S5): S[~ok] = np.nan
        mask = load_mask(fid)
        if mask is not None:
            dt = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
            def dist(xy_):
                q = np.clip(np.rint(xy_).astype(int), [0, 0], [W - 1, H - 1]); return dt[q[:, 1], q[:, 0]].astype(np.float32)
            d_std = dist(xy); d_ctr = dist(a); in_std = mask[np.clip(np.rint(xy[:, 1]).astype(int), 0, H - 1), np.clip(np.rint(xy[:, 0]).astype(int), 0, W - 1)]
            in_ctr = mask[np.clip(np.rint(a[:, 1]).astype(int), 0, H - 1), np.clip(np.rint(a[:, 0]).astype(int), 0, W - 1)]
        else:
            d_std = np.full(len(xy), np.nan, np.float32); d_ctr = np.full(len(a), np.nan, np.float32); in_std = np.zeros(len(xy), bool); in_ctr = np.zeros(len(a), bool)
        np.savez_compressed(dst, std_xy=xy.astype(np.float32), S0=z1['std_delta'], S1=S1, S2=S2, ctr_xy=a.astype(np.float32), S3=S3, S4=S4, S5=S5, ok=ok,
                            dist_std=d_std, dist_ctr=d_ctr, in_mask_std=in_std, in_mask_ctr=in_ctr, has_mask=mask is not None)
        return dict(fid=fid, ok=True, seconds=time.time() - t0, n_ok=int(ok.sum()), has_mask=mask is not None)
    except Exception as e:
        return dict(fid=fid, ok=False, error=repr(e), trace=traceback.format_exc()[-1200:])


def main():
    (OUT / 'fields').mkdir(parents=True, exist_ok=True)
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str})
    recs = led.to_dict('records')
    if '--limit' in sys.argv: recs = recs[::max(1, len(recs) // int(sys.argv[sys.argv.index('--limit') + 1]))][:int(sys.argv[sys.argv.index('--limit') + 1])]
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 4
    if workers <= 1: res = [one(r) for r in recs]
    else:
        from multiprocessing import Pool
        with Pool(workers) as p:
            res = []
            for r in p.imap_unordered(one, recs, chunksize=1):
                res.append(r); print('DONE' if r['ok'] else 'FAIL', r['fid'], r.get('seconds', ''), r.get('error', ''), flush=True)
    pd.DataFrame(res).to_csv(OUT / 'run_status.csv', index=False)
    print('finished', sum(r['ok'] for r in res), '/', len(res))


if __name__ == '__main__':
    main()
