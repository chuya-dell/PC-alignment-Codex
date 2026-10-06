"""v57 step D1/D2: detect the cross-scar lines in the pre-wash image of each field and build a complete mask.
Image-only (no use of stored differences / exceedance rates). Does not re-detect pillars and does not touch registration.

Method
 1. 2x area downsample, Gaussian sigma 1.5 (suppresses the pillar lattice), E = |a / Gaussian_sigma20(a) - 1| (scar energy: dark core + bright flanks).
 2. Per family (near-vertical lines; near-horizontal lines = transposed image): shear-projection over slopes |s|<=0.10 gives a 1-D profile;
    best line = maximum smoothed profile peak. Per-segment (32 ds px) ridge positions are found near the line; visible segments (ridge z>=4)
    feed a robust straight-line fit (Theil-Sen + trimming); invisible segments are bridged by the fitted line (extension).
    A line is accepted if profile z >= Z_MIN and the visible fraction >= VIS_MIN.  Up to 3 lines per family; neighbours < 80 ds px merged.
 3. Mask = pixel-level hysteresis region of E (seeds E>=T_HI near the line, growth E>=T_LO inside a corridor) united with a synthetic band
    of the measured median half-width along bridged parts, dilated by 1 lattice period (7.286 px) on every side.
 4. Numerical check: dark scar pixels (relative darkness <= -DARK_TAU after sigma 1.5 smoothing) 8-connected to the mask boundary
    outside the mask must be zero pixels; isolated dark objects (dust) are counted separately.
All coordinates are in the full-resolution pre-image frame (x right, y down, 2048 x 2044)."""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, cv2, tifffile
from scipy.stats import theilslopes
import field_v57_common as C

DS = 2
PERIOD = 7.286
Z_MIN = 10.0          # profile peak z for a line to be accepted
VIS_MIN = 0.35        # visible fraction of along-line segments
VIS_RESCUE_Z, VIS_RESCUE_MIN = 40.0, 0.20   # a very strong profile peak is accepted with a lower visible fraction (bridged by the fitted line)
SEG = 32              # segment length, downsampled px
SLOPES = np.linspace(-0.30, 0.30, 61)   # tilt |s|<=0.30 (~17 deg) of near-vertical / near-horizontal lines
T_HI_K, T_LO_K = 20.0, 10.0    # in robust sigmas of E (sigma ~0.003)
CORR_SEED = 60        # ds px: seeds searched within this distance of the fitted line
CORR_GROW = 110       # ds px: growth corridor half-width
DARK_TAU = 0.08

def energy(im):
    H, W = im.shape
    a = cv2.resize(im, (W//DS, H//DS), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (0, 0), 1.5)
    bg = cv2.GaussianBlur(a, (0, 0), 20)
    return np.abs(a/np.maximum(bg, 1e-6)-1.0).astype(np.float32), a

def shear(E, s):
    h, w = E.shape; yc = (h-1)/2
    M = np.float32([[1, -s, s*yc], [0, 1, 0]])
    return cv2.warpAffine(E, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0), M

_CNT = {}
def profile_of(E, s):
    Ws, _ = shear(E, s); h = E.shape[0]
    key = (E.shape, round(float(s), 6))
    if key not in _CNT:
        ones, _ = shear(np.ones(E.shape, np.float32), s); _CNT[key] = ones.sum(0)
    cnt = _CNT[key]
    p = np.where(cnt > 0.8*h, Ws.sum(0)/np.maximum(cnt, 1e-6), np.nan)
    return p, Ws

def _peak(E, s, med_e):
    p, _ = profile_of(E, s)
    k = cv2.blur(np.nan_to_num(p-med_e, nan=0.0).astype(np.float32).reshape(1, -1), (5, 1)).ravel()
    return k.max(), int(np.argmax(k)), p

def best_line(E, med_e):
    """return (s, x0, z, profile) of the strongest near-vertical line. coarse slope search on a 2x coarser copy, fine search at full ds resolution"""
    E4 = cv2.resize(E, (E.shape[1]//2, E.shape[0]//2), interpolation=cv2.INTER_AREA)
    sc = [(_peak(E4, s, med_e)[0], s) for s in SLOPES]
    s0 = max(sc)[1]
    fine = np.arange(s0-0.005, s0+0.00501, 0.001)
    sc = [(_peak(E, s, med_e)[0], s) for s in fine]
    s1 = max(sc)[1]
    pk, x0, p = _peak(E, s1, med_e)
    pv = p[np.isfinite(p)]
    med = np.median(pv); mad = 1.4826*np.median(np.abs(pv-med))+1e-9
    k = np.nan_to_num(cv2.blur(p.reshape(1, -1).astype(np.float32), (5, 1)).ravel(), nan=-1)
    z = (k[x0]-med)/mad
    return s1, x0, z, p

def segment_fit(E, s, x0, noise_sig, med_e):
    """per-segment ridge near the line (in sheared frame) -> robust fit on visible segments"""
    Ws, _ = shear(E, s); h, w = E.shape; yc = (h-1)/2
    rows = np.arange(0, h-SEG+1, SEG); pos = []; ys = []; vis = []
    for r in rows:
        q = Ws[r:r+SEG].mean(0)
        lo = max(0, x0-45); hi = min(w, x0+46)
        win = q[lo:hi]
        ks = cv2.blur(win.astype(np.float32).reshape(1, -1), (5, 1)).ravel()
        j = int(np.argmax(ks)); pkv = ks[j]
        rest = np.delete(ks, np.arange(max(0, j-8), min(len(ks), j+9)))
        bg = np.median(rest) if len(rest) else med_e
        sig = 1.4826*np.median(np.abs(rest-bg))+1e-4 if len(rest) else noise_sig
        z = (pkv-bg)/max(sig, noise_sig*0.5)
        pos.append(lo+j); ys.append(r+SEG/2); vis.append(z >= 4.0 and pkv-bg > 2.5*noise_sig)
    pos = np.array(pos, float); ys = np.array(ys, float); vis = np.array(vis)
    return pos, ys, vis, (yc, Ws)

def robust_line(ys, xs, init_x0, init_s):
    """x' = a + b*(y-yc) in the sheared frame is the residual of the shear; Theil-Sen on visible points, then trim 20%"""
    if len(ys) < 3: return None
    b, a, lo, hi = theilslopes(xs, ys)
    for _ in range(2):
        res = xs-(a+b*ys); thr = max(3.0, np.quantile(np.abs(res), 0.8))
        keep = np.abs(res) <= thr
        if keep.sum() < 3: break
        b, a, lo, hi = theilslopes(xs[keep], ys[keep])
    res = xs-(a+b*ys)
    return a, b, float(np.median(np.abs(res)))

def detect_family(E, noise_sig, med_e, tag):
    """returns list of line dicts in downsampled coordinates of E (E rows = along-line axis)"""
    h, w = E.shape; yc = (h-1)/2
    Ecur = E.copy(); lines = []
    for it in range(3):
        s, x0, z, p = best_line(Ecur, med_e)
        rec = dict(family=tag, iter=it, s_init=float(s), x0_init=float(x0), z=float(z))
        if z < Z_MIN:
            rec['status'] = 'z_below_threshold'; lines.append(rec); break
        pos, ys, vis, (yc_, Ws) = segment_fit(Ecur, s, x0, noise_sig, med_e)
        rec['visible_frac'] = float(vis.mean())
        if vis.sum() < 3 or not (vis.mean() >= VIS_MIN or (z >= VIS_RESCUE_Z and vis.mean() >= VIS_RESCUE_MIN)):
            rec['status'] = 'visible_fraction_low'; lines.append(rec)
            # suppress and look for another candidate
            Ecur = suppress(Ecur, s, x0, 40, med_e); continue
        fit = robust_line(ys[vis], pos[vis], x0, s)
        a, b, resid = fit
        # line in original (ds) coords: sheared x' = x - s*(y-yc)  (+ nothing).  line: x'(y) = a + b*y  ->  x = a + b*y + s*(y-yc)
        # parametrise as x = X0 + S*(y-yc): S = s + b ; X0 = a + b*yc  (since x'(yc) = a + b*yc)
        S = s+b; X0 = a+b*yc
        rec.update(status='accepted', X0=float(X0), S=float(S), fit_resid_med=resid, n_vis=int(vis.sum()), n_seg=int(len(vis)),
                   seg_y=ys.tolist(), seg_x=(pos+0).tolist(), seg_vis=vis.tolist(), s_shear=float(s))
        lines.append(rec)
        Ecur = suppress(Ecur, s, x0, 40, med_e)
    return lines

def suppress(E, s, x0, hw, med_e):
    out = E.copy(); h, w = E.shape; yc = (h-1)/2
    ys = np.arange(h); xs = x0+s*(ys-yc)
    for y, x in zip(ys, xs):
        a = int(max(0, x-hw)); b = int(min(w, x+hw+1))
        if b > a: out[y, a:b] = med_e
    return out

def build_mask(E, lines_v, lines_h, noise_sig, med_e):
    h, w = E.shape
    T_hi = med_e+T_HI_K*noise_sig; T_lo = med_e+T_LO_K*noise_sig
    yy, xx = np.mgrid[0:h, 0:w]
    seed = np.zeros((h, w), bool); grow = np.zeros((h, w), bool); synth = np.zeros((h, w), bool)
    per_line = []
    for fam, lines in (('v', lines_v), ('h', lines_h)):
        for ln in lines:
            if ln.get('status') != 'accepted': continue
            if fam == 'v':
                d = xx-(ln['X0']+ln['S']*(yy-(h-1)/2))      # signed distance across (px), line family near-vertical
                along = yy
            else:
                # lines found on E.T: E.T[x, y]; line in transposed frame: X' = X0 + S*(Y'-yc') with X'=y(orig), Y'=x(orig)
                d = yy-(ln['X0']+ln['S']*(xx-(w-1)/2))
                along = xx
            ln['_d'] = d
            seed |= (np.abs(d) <= CORR_SEED); grow |= (np.abs(d) <= CORR_GROW)
    M_hi = seed & (E >= T_hi); M_lo = grow & (E >= T_lo)
    # hysteresis: components of M_lo containing at least one M_hi pixel
    n, lab = cv2.connectedComponents(M_lo.astype(np.uint8), connectivity=8)
    keep_ids = np.unique(lab[M_hi]); keep_ids = keep_ids[keep_ids > 0]
    M0 = np.isin(lab, keep_ids)
    # per-line, per-segment half extents (both sides) from M0 within the corridor; bridge invisible parts with the median half width
    info = []
    for fam, lines in (('v', lines_v), ('h', lines_h)):
        for ln in lines:
            if ln.get('status') != 'accepted': continue
            d = ln.pop('_d')
            near = M0 & (np.abs(d) <= CORR_GROW)
            L = h if fam == 'v' else w
            segs = np.arange(0, L, SEG)
            left = []; right = []; present = []
            for a in segs:
                b = min(L, a+SEG)
                sl = near[a:b, :] if fam == 'v' else near[:, a:b]
                dd = d[a:b, :] if fam == 'v' else d[:, a:b]
                if sl.any():
                    dv = dd[sl]
                    left.append(-dv.min()); right.append(dv.max()); present.append(True)
                else:
                    left.append(np.nan); right.append(np.nan); present.append(False)
            left = np.array(left); right = np.array(right); present = np.array(present)
            hw_med = float(np.nanmedian(np.where(present, (np.nan_to_num(left)+np.nan_to_num(right))/2, np.nan))) if present.any() else 8.0
            ln['half_width_median_ds'] = hw_med
            ln['present_frac'] = float(present.mean())
            for k, a in enumerate(segs):
                if present[k]: continue
                b = min(L, a+SEG)
                sel = (np.zeros((h, w), bool))
                if fam == 'v': sel[a:b, :] = True
                else: sel[:, a:b] = True
                synth |= sel & (np.abs(d) <= max(hw_med, 6.0))
            ln['n_bridged_segments'] = int((~present).sum()); ln['n_segments'] = int(len(present))
            ln['thickness_px_full_median'] = float(2*hw_med*DS)
    mask_ds = M0 | synth
    return mask_ds, M0, synth

def process(args):
    fid, path, out_dir, save_overlay = args
    res = dict(fid=fid, path=path)
    try:
        im = tifffile.imread(path).astype(np.float32)/65535.0
        H, W = im.shape
        E, a_ds = energy(im)
        med_e = float(np.median(E)); noise_sig = float(1.4826*np.median(np.abs(E-med_e)))+1e-6
        lines_v = detect_family(E, noise_sig, med_e, 'vertical')
        lines_h_raw = detect_family(E.T.copy(), noise_sig, med_e, 'horizontal')
        # merge near-duplicates (< 80 ds px at image centre)
        def merge(lines, h):
            acc = [l for l in lines if l.get('status') == 'accepted']
            acc.sort(key=lambda l: -l['z'])
            out = []
            for l in acc:
                if all(abs(l['X0']-o['X0']) > 80 for o in out): out.append(l)
            return out, [l for l in lines if l.get('status') != 'accepted']
        hv = E.shape[0]; hh = E.T.shape[0]
        accv, rejv = merge(lines_v, hv); acch, rejh = merge(lines_h_raw, hh)
        res['lines_rejected'] = json.dumps([{k: r[k] for k in ('family', 'iter', 'z', 'status') if k in r} | ({'visible_frac': r['visible_frac']} if 'visible_frac' in r else {}) for r in rejv+rejh])
        mask_ds, M0, synth = build_mask(E, accv, acch, noise_sig, med_e)
        # scars clipped by the image border (line not recoverable): absorb long strong components that touch the border
        T_hi = med_e+T_HI_K*noise_sig; T_lo = med_e+T_LO_K*noise_sig
        md = cv2.dilate(mask_ds.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        nS, labS, stS, _ = cv2.connectedComponentsWithStats(((E >= T_hi) & ~md).astype(np.uint8), connectivity=8)
        hE, wE = E.shape; seeds_b = []
        for i in range(1, nS):
            x_, y_, w_, h_, a_ = stS[i]
            if max(w_, h_) >= 60 and a_ >= 400 and (x_ <= 1 or y_ <= 1 or x_+w_ >= wE-1 or y_+h_ >= hE-1): seeds_b.append(i)
        res['border_clipped_components_absorbed'] = len(seeds_b)
        if seeds_b:
            nL, labL = cv2.connectedComponents((E >= T_lo).astype(np.uint8), connectivity=8)
            ids = np.unique(labL[np.isin(labS, seeds_b)]); ids = ids[ids > 0]
            mask_ds = mask_ds | (np.isin(labL, ids) & ~md | np.isin(labS, seeds_b))
        res['n_lines_v'] = len(accv); res['n_lines_h'] = len(acch)
        res['z_max'] = float(max([l['z'] for l in accv+acch], default=np.nan))
        # strongest rejected candidate (for failure reasons)
        res['z_best_rejected'] = float(max([r['z'] for r in rejv+rejh if 'z' in r], default=np.nan))
        if not (accv or acch):
            res['status'] = 'no_line'
            res['reason'] = ('傷が写っていない(候補の最大z=%.1f < %.0f)' % (res['z_best_rejected'], Z_MIN)) if (np.isnan(res['z_best_rejected']) or res['z_best_rejected'] < Z_MIN) else \
                            ('直線が当てはまらない(z=%.1f だが見える区間の割合が%.2f未満)' % (res['z_best_rejected'], VIS_MIN))
            if save_overlay: write_overlay(out_dir, fid, a_ds, None, accv, acch, res)
            return res
        # full-resolution mask: upsample, absorb dark scar components touching it, dilate by 1 period, absorb again
        m_full = cv2.resize(mask_ds.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        c = cv2.GaussianBlur(im, (0, 0), 1.5); bgf = cv2.GaussianBlur(im, (0, 0), 20)
        dark = (c/np.maximum(bgf, 1e-6)-1.0) <= -DARK_TAU
        def absorb(m):
            added = 0
            for _ in range(8):
                left = dark & ~m
                ring = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                n, lab = cv2.connectedComponents(left.astype(np.uint8), connectivity=8)
                ids = np.unique(lab[ring & left]); ids = ids[ids > 0]
                if len(ids) == 0: break
                add = np.isin(lab, ids); added += int(add.sum()); m = m | add
            return m, added
        m_full, add1 = absorb(m_full)
        k = int(round(PERIOD))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
        m_full = cv2.dilate(m_full.astype(np.uint8), ker) > 0
        m_full, add2 = absorb(m_full)
        left = dark & ~m_full
        ring = cv2.dilate(m_full.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0   # within 3 px of the mask
        n, lab, stats, _ = cv2.connectedComponentsWithStats(left.astype(np.uint8), connectivity=8)
        touch_ids = np.unique(lab[ring & left]); touch_ids = touch_ids[touch_ids > 0]
        res['dark_px_in_mask'] = int((dark & m_full).sum())
        res['dark_absorbed_px_before_margin'] = add1; res['dark_absorbed_px_after_margin'] = add2
        res['dark_leftover_total_px'] = int(left.sum())
        res['dark_leftover_touching_mask_px'] = int(np.isin(lab, touch_ids).sum())
        res['dark_isolated_components'] = int(n-1-len(touch_ids))
        res['mask_fraction'] = float(m_full.mean())
        # QC: strong scar-like energy components left outside the mask (long: bbox >= 60 ds px, area >= 400 ds px^2)
        m_ds2 = cv2.resize(m_full.astype(np.uint8), (E.shape[1], E.shape[0]), interpolation=cv2.INTER_NEAREST)
        m_ds2 = cv2.dilate(m_ds2, np.ones((5, 5), np.uint8)) > 0
        strong = (E >= med_e+T_HI_K*noise_sig) & ~m_ds2
        nn, lab2, st2, _ = cv2.connectedComponentsWithStats(strong.astype(np.uint8), connectivity=8)
        big = [(int(st2[i, cv2.CC_STAT_LEFT]*DS), int(st2[i, cv2.CC_STAT_TOP]*DS), int(st2[i, cv2.CC_STAT_WIDTH]*DS), int(st2[i, cv2.CC_STAT_HEIGHT]*DS), int(st2[i, cv2.CC_STAT_AREA]*DS*DS))
               for i in range(1, nn) if max(st2[i, cv2.CC_STAT_WIDTH], st2[i, cv2.CC_STAT_HEIGHT]) >= 60 and st2[i, cv2.CC_STAT_AREA] >= 400]
        res['leftover_scar_like_components'] = len(big); res['leftover_scar_like_area_px'] = int(sum(b[4] for b in big)); res['leftover_scar_like_bboxes'] = json.dumps(big)
        res['status'] = 'ok'
        # line geometry in full-res coordinates
        geo = []
        for fam, lines in (('vertical', accv), ('horizontal', acch)):
            for l in lines:
                geo.append(dict(family=fam, X0_full=l['X0']*DS, S=l['S'], z=l['z'], visible_frac=l['visible_frac'], fit_resid_med_px=l['fit_resid_med']*DS,
                                thickness_px=l.get('thickness_px_full_median'), bridged=l.get('n_bridged_segments'), segments=l.get('n_segments'),
                                present_frac=l.get('present_frac')))
        res['lines_json'] = json.dumps(geo)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, f'{fid}_mask.npz'), mask=np.packbits(m_full), shape=np.array(m_full.shape))
        if save_overlay: write_overlay(out_dir, fid, a_ds, m_full, accv, acch, res)
    except Exception as e:
        res['status'] = 'error'; res['reason'] = repr(e)
    return res

def write_overlay(out_dir, fid, a_ds, m_full, accv, acch, res):
    os.makedirs(os.path.join(out_dir, 'overlays'), exist_ok=True)
    lo, hi = np.percentile(a_ds, [1, 99.5]); g = np.clip((a_ds-lo)/(hi-lo+1e-9), 0, 1)
    img = cv2.cvtColor((g*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    h, w = a_ds.shape
    if m_full is not None:
        m = cv2.resize(m_full.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
    for l in accv:
        y = np.arange(h); x = l['X0']+l['S']*(y-(h-1)/2)
        pts = np.c_[x, y].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (0, 0, 255), 1)
    for l in acch:
        x = np.arange(w); y = l['X0']+l['S']*(x-(w-1)/2)
        pts = np.c_[x, y].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (255, 0, 0), 1)
    txt = f"{fid} {res.get('status')} V{res.get('n_lines_v')} H{res.get('n_lines_h')} zmax={res.get('z_max', float('nan')):.0f}"
    cv2.putText(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, 'overlays', f'{fid}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 88])

def worker(args):
    cv2.setNumThreads(1)
    fid, path, out_dir, save_overlay = args
    jp = os.path.join(out_dir, 'results', f'{fid}.json')
    if os.path.exists(jp) and os.environ.get('V57_FORCE') != '1':
        return json.load(open(jp, encoding='utf-8'))
    r = process(args)
    os.makedirs(os.path.join(out_dir, 'results'), exist_ok=True)
    json.dump(r, open(jp, 'w', encoding='utf-8'), ensure_ascii=False, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else (o.item() if hasattr(o, 'item') else str(o)))
    return r

if __name__ == '__main__':
    from multiprocessing import Pool
    inv = pd.read_csv(C.OUT/'step0'/'fields_inventory.csv', encoding='utf-8-sig', dtype={'board': str})
    ids = sys.argv[1:] if len(sys.argv) > 1 else None
    tag = 'all' if ids is None or ids == ['all'] else 'test'
    out_dir = str(C.OUT/'stepD') if tag == 'all' else str(C.OUT/'stepD_test')
    sub = inv if tag == 'all' else inv[inv.fid.isin(ids)]
    args = [(r.fid, r.pre_path, out_dir, True) for r in sub.itertuples()]
    rs = []
    with Pool(int(os.environ.get('V57_PROCS', '10'))) as p:
        for i, r in enumerate(p.imap_unordered(worker, args, chunksize=1), 1):
            rs.append(r)
            if i % 25 == 0: print(i, 'of', len(args), flush=True)
    o = pd.DataFrame(rs).sort_values('fid')
    o.to_csv(os.path.join(out_dir, 'detection_results.csv'), index=False, encoding='utf-8-sig')
    print(o.status.value_counts().to_dict())
