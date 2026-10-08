"""v75 step B library: analyse an image pair with every readout scheme (S0..S5 and A-map variants).
Reuses v70 (lattice, centres, physical numbering) and v72 (readouts).  Works on arrays, not on the 632-field tables."""
from __future__ import annotations
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit')); sys.path.insert(0, str(ROOT / 'field_level/v72_real_field_readout'))
sys.path.insert(0, str(ROOT / 'field_level/v60_band_scar_controls'))
import numpy as np, cv2
from scipy.spatial import cKDTree
import v70_lib as L
import v70_centers2 as C
import v72_readout as R
import v70_period_fix as PF
import field_control_common as M60
from scipy import ndimage as _ndi
W, H = 2048, 2044
c0 = np.array([W / 2, H / 2])


def analyse_pair(a, b, A_override=None, shift_hint=None):
    """a (pre), b (post): float32 raw images.  Returns dict.  A_override: use this 2x3 matrix instead of the standard estimate."""
    std = L.standard_run(a, b, matrix=A_override)
    A = std['matrix']
    ca_ = C.centers_smooth(a); cb_ = C.centers_smooth(b)
    Bpre = ca_['model'][1:].T; Bpost = cb_['model'][1:].T
    U = np.rint(np.linalg.inv(Bpost) @ A[:, :2] @ Bpre).astype(int)
    Llin = Bpost @ U @ np.linalg.inv(Bpre)
    tL = (A[:, :2] @ c0 + A[:, 2]) - Llin @ c0
    q = cb_['pred_lin']
    expL = ca_['pred_lin'] @ Llin.T + tL
    _, j0 = cKDTree(q).query(expL)
    f = (q[j0] - expL) @ np.linalg.inv(Bpost).T
    s = np.angle(np.exp(2j * np.pi * f).mean(axis=0)) / (2 * np.pi)
    frac = Bpost @ s; tL = tL + frac
    # integer lattice period: decided by pillar-value correlation (v70_period_fix), NOT by the feature-based affine (independent critique, 2026-10-08)
    a_hp = a - cv2.GaussianBlur(a, (0, 0), 8); b_hp = b - cv2.GaussianBlur(b, (0, 0), 8)
    pcen = ca_['ctr'].astype(float); va = _ndi.map_coordinates(a_hp, [pcen[:, 1], pcen[:, 0]], order=1, mode='nearest')
    basrows = cb_['model'][1:]
    res_s = PF.scan(va, pcen, b_hp, Llin, tL, basrows, PF.R); best_s = max(res_s, key=res_s.get); c_zero = res_s[(0, 0)]; c_best = res_s[best_s]
    if c_best < 0.5 or best_s[0] in (-PF.R, PF.R) or best_s[1] in (-PF.R, PF.R):
        res2 = PF.scan(va, pcen, b_hp, Llin, tL, basrows, PF.RBIG); b2 = max(res2, key=res2.get)
        if res2[b2] > c_best: best_s, c_best = b2, res2[b2]
    shift = best_s if (c_best - c_zero > 0.1 and c_best >= 0.3) else (0, 0)
    tL = tL + shift[0] * basrows[0] + shift[1] * basrows[1]
    expL = ca_['pred_lin'] @ Llin.T + tL
    dL, jL = cKDTree(q).query(expL)
    expA = ca_['pred_lin'] @ A[:, :2].T + A[:, 2]
    dA, jA = cKDTree(q).query(expA)
    pre_d = dict(ctr=ca_['ctr'].astype(float), pred=ca_['pred'], amp=ca_['amp'], wid=ca_['wid'], aloc=ca_['aloc'])
    post_d = dict(ctr=cb_['ctr'].astype(float), pred=cb_['pred'], amp=cb_['amp'], wid=cb_['wid'], aloc=cb_['aloc'])
    Sp = C.sub_flags(pre_d, 'main'); So = C.sub_flags(post_d, 'main')
    ovL = (expL[:, 0] >= 8) & (expL[:, 0] < W - 8) & (expL[:, 1] >= 8) & (expL[:, 1] < H - 8)
    okL = (~Sp) & (~So[jL]) & (dL < 2.0) & ovL
    ovA = (expA[:, 0] >= 8) & (expA[:, 0] < W - 8) & (expA[:, 1] >= 8) & (expA[:, 1] < H - 8)
    okA = (~Sp) & (~So[jA]) & (dA < 2.0) & ovA
    boxa, boxb = R.box_contrast(a), R.box_contrast(b)
    pa, pb = R.plain_contrast(a), R.plain_contrast(b)
    xy = std['xy'].astype(float); xyL = xy @ Llin.T + tL
    out = dict(A=A, Llin=Llin, tL=tL, U=U, frac_corr_px=float(np.linalg.norm(frac)), corner_disagree_px=float(np.linalg.norm((A[:, :2] - Llin) @ (np.array([[0, 0], [W, 0], [0, H], [W, H]], float) - c0).T, axis=0).max()),
               std_xy=xy, ctr_xy=pre_d['ctr'], ok=okL, okA=okA, n_nodes=len(Sp), pitch_pre=ca_['info']['pitch_px'], pitch_post=cb_['info']['pitch_px'],
               sub_pre=float(Sp.mean()), sub_post=float(So.mean()), period_shift=np.array(shift), period_corr_zero=c_zero, period_corr_best=c_best)
    out['S0'] = std['delta']; out['xy0'] = std['xy'].astype(float)
    out['S1'] = R.rnd(boxa, xy) - R.rnd(boxb, xyL)
    out['S2'] = R.bil(boxa, xy) - R.bil(boxb, xyL)
    ac, bc = pre_d['ctr'], post_d['ctr'][jL]
    out['S3'] = R.rnd(boxa, ac) - R.rnd(boxb, bc); out['S4'] = R.bil(boxa, ac) - R.bil(boxb, bc); out['S5'] = R.aper(pa, ac) - R.aper(pb, bc)
    for k in ('S3', 'S4', 'S5'): out[k][~okL] = np.nan
    bcA = post_d['ctr'][jA]
    out['S3A'] = R.rnd(boxa, ac) - R.rnd(boxb, bcA); out['S4A'] = R.bil(boxa, ac) - R.bil(boxb, bcA)
    for k in ('S3A', 'S4A'): out[k][~okA] = np.nan
    # P readouts: pre-centre based; post at L-mapped + local offset (validity from the PRE image only)
    okP = (~Sp) & ovL & (dL < 2.0)
    expC = ca_['ctr'].astype(float) @ Llin.T + tL            # actual pre centres mapped by L (the pillar does not move)
    posP = local_offset_positions(expC, post_d['ctr'][jL], okL)
    out['okP'] = okP; out['posP'] = posP
    out['S3P'] = R.rnd(boxa, ac) - R.rnd(boxb, posP); out['S4P'] = R.bil(boxa, ac) - R.bil(boxb, posP); out['S5P'] = R.aper(pa, ac) - R.aper(pb, posP)
    for k in ('S3P', 'S4P', 'S5P'): out[k][~okP] = np.nan
    # second-order information for local thresholds / maps
    out['pre_dict'] = pre_d
    return out


def local_offset_positions(expC, post_ctr_matched, ok, block=64):
    """Post reading positions: the pre pillar centre mapped by the lattice map (expC) plus a smooth local offset (64-px block median of fitted-post minus mapped-pre over
    trusted nodes).  No per-pillar post fit is used, so a pillar whose brightness changed is still read at its physical position."""
    off = post_ctr_matched - expC
    gx = np.clip((expC[:, 0] // block).astype(int), 0, W // block - 1); gy = np.clip((expC[:, 1] // block).astype(int), 0, H // block - 1)
    ny, nx = H // block + 1, W // block
    cell = gy * nx + gx
    medx = np.full(ny * nx, np.nan); medy = np.full(ny * nx, np.nan)
    sel = ok & (np.linalg.norm(off, axis=1) < 3.0)
    order = np.argsort(cell[sel]); cs = cell[sel][order]; ox = off[sel, 0][order]; oy = off[sel, 1][order]
    b_ = np.flatnonzero(np.diff(cs)) + 1
    for c_, sx, sy in zip(np.split(cs, b_), np.split(ox, b_), np.split(oy, b_)):
        if len(sx) >= 20: medx[c_[0]] = np.median(sx); medy[c_[0]] = np.median(sy)
    gm = (np.nanmedian(medx), np.nanmedian(medy))
    mx = np.where(np.isfinite(medx), medx, gm[0]).reshape(ny, nx); my = np.where(np.isfinite(medy), medy, gm[1]).reshape(ny, nx)
    mx = _ndi.median_filter(mx, size=3, mode='nearest'); my = _ndi.median_filter(my, size=3, mode='nearest')
    px = expC[:, 0] / block - .5; py = expC[:, 1] / block - .5
    ox_ = _ndi.map_coordinates(mx, [py, px], order=1, mode='nearest'); oy_ = _ndi.map_coordinates(my, [py, px], order=1, mode='nearest')
    return expC + np.column_stack([ox_, oy_])


READS_STD = ['S0', 'S1', 'S2']
def positions(res, key):
    return res['xy0'] if key in ('S0', 'S1', 'S2') else res['ctr_xy']
