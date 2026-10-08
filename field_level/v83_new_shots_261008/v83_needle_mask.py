"""v83 part A-4 (1/3): scar masks for the needle-scar series 261008_p50_傷 (6, 6-1..6-4).  Cross-scar lines: v57 detector (unchanged).  Needle gouge: strong scar-energy components
(v57 energy E, >= median+8 sigma, area >= 300 ds-px^2), filled, dilated by 2 lattice periods.  mask_align = union, used to neutralise the scars for the alignment
(masked pixels replaced by a smooth background: normalised convolution sigma 20).  usage: python v83_needle_mask.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / 'field_level/v57_false_positive_facts'))
import numpy as np, cv2, tifffile
import field_v57_detect_scars as D
D.Z_MIN = 40.0     # the v57 default (10) accepts a spurious bridged vertical line through the gouge in 6-3 (z=31); real cross-scar lines here have z>=50
RAW = Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu\261008_p50_傷'); OUT = ROOT / 'data/results/v83_new_shots_261008/needle'; OUT.mkdir(parents=True, exist_ok=True)
NAMES = ['6', '6-1', '6-2', '6-3', '6-4']


def gouge_mask(im):
    E, a = D.energy(im / 65535.0); med = np.median(E); sig = 1.4826 * np.median(abs(E - med)) + 1e-6
    strong = (E >= med + 8 * sig).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
    keep = np.zeros_like(strong)
    H, W = E.shape
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        if ar >= 300 and x >= 300 and y > 20 and x + w < W - 20 and y + h < H - 200:     # the needle gouge only (not the cross-scar flanks at the left/bottom borders)
            keep[lab == i] = 1
    k = int(round(2 * 7.286 / 2)); ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    m = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); filled = np.zeros_like(m)
    for c in cnts: cv2.fillPoly(filled, [cv2.convexHull(c)], 1)
    filled = cv2.dilate(filled, ker)
    return cv2.resize(filled, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


if __name__ == '__main__':
    info = {}
    for n in NAMES:
        im = tifffile.imread(RAW / f'{n}.tif').astype(np.float32)
        r = D.process((f'n{n}', str(RAW / f'{n}.tif'), str(OUT), True))
        with np.load(OUT / f'n{n}_mask.npz') as z: shp = tuple(z['shape']); mc = np.unpackbits(z['mask'])[:np.prod(shp)].reshape(shp).astype(bool)
        mg = gouge_mask(im); m = mc | mg
        np.savez_compressed(OUT / f'align_mask_{n}.npz', cross=np.packbits(mc), gouge=np.packbits(mg), shape=np.array(mc.shape))
        ys, xs = np.where(mg)
        info[n] = dict(cross_frac=float(mc.mean()), gouge_frac=float(mg.mean()), gouge_bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None, union_frac=float(m.mean()),
                       lines=json.loads(r.get('lines_json', '[]')))
        lo, hi = np.percentile(im, [0.5, 99.8]); g = np.clip((im - lo) / (hi - lo), 0, 1); g = cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for mm, col in ((mc, (0, 255, 0)), (mg, (0, 0, 255))):
            cs, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(g, cs, -1, col, 2)
        cv2.imwrite(str(OUT / f'mask_overlay_{n}.jpg'), cv2.resize(g, (1024, 1024), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(n, {k: v for k, v in info[n].items() if k != 'lines'}, flush=True)
    json.dump(info, open(OUT / 'align_mask_info.json', 'w'), indent=1)
