"""v70 step A (revised): per-pillar centers with a smooth (cubic) lattice model, physical numbering by the
lattice-implied linear map (translation fixed from the registration at the image centre, checked against
non-periodic landmarks), local amplitude reference.  Output: fields2/<fid>.npz (+json).
usage: python v70_centers2.py [--pilot N] [--workers K]"""
from __future__ import annotations
import sys, json, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
import v70_lib as L
OUT = L.ROOT / 'data/results/v70_ledger_center_fit'
F1 = OUT / 'fields'; F2 = OUT / 'fields2'
W, H = 2048, 2044


def poly_terms(ids, deg=3):
    n = ids / 300.; i, j = n[:, 0], n[:, 1]
    cols = [i ** a * j ** b for a in range(deg + 1) for b in range(deg + 1 - a)]
    return np.column_stack(cols)


def local_aref(pos, amp, block=64):
    gx = np.clip((pos[:, 0] // block).astype(int), 0, W // block - 1); gy = np.clip((pos[:, 1] // block).astype(int), 0, H // block - 1)
    nb = (W // block) * (H // block + 1)
    cell = gy * (W // block) + gx
    order = np.argsort(cell); cs = cell[order]; a = amp[order]
    med = np.full((H // block + 1, W // block), np.nan)
    bounds = np.flatnonzero(np.diff(cs)) + 1
    for seg_c, seg in zip(np.split(cs, bounds), np.split(a, bounds)):
        if len(seg) >= 30:
            c = seg_c[0]; med[c // (W // block), c % (W // block)] = np.median(seg[seg > 0]) if (seg > 0).any() else np.nan
    glob = np.nanmedian(med) if np.isfinite(med).any() else 1.0
    med = np.where(np.isfinite(med), med, glob)
    # robust smoothing across blocks: median filter then bilinear sample at node position
    sm = ndimage.median_filter(med, size=3, mode='nearest')
    xs = pos[:, 0] / block - .5; ys = pos[:, 1] / block - .5
    return ndimage.map_coordinates(sm, [ys, xs], order=1, mode='nearest'), float(glob)


def centers_smooth(im):
    h, w = im.shape
    model, quad, info = L.fit_lattice(im)
    ids, pred_lin = L.nodes_in_image(model, quad, w, h)
    cb = L.contrast_plain(im)
    ctr, amp, wid = L.fit_centers(cb, pred_lin)
    pred = pred_lin
    for rnd in range(3):
        aloc, glob = local_aref(ctr, amp)
        good = (amp > 0.5 * aloc) & (np.linalg.norm(ctr - pred, axis=1) < 2.2)
        T = poly_terms(ids)
        sel = good.copy()
        for _ in range(3):
            coef = np.linalg.lstsq(T[sel], ctr[sel], rcond=None)[0]
            res = np.linalg.norm(ctr - T @ coef, axis=1)
            sel = good & (res < max(0.4, 3 * np.median(res[sel])))
        pred = T @ coef
        ctr, amp, wid = L.fit_centers(cb, pred, iters=4)
    aloc, glob = local_aref(ctr, amp)
    dist = np.linalg.norm(pred - pred_lin, axis=1)
    info.update(poly_vs_linear_median=float(np.median(dist)), poly_vs_linear_p95=float(np.percentile(dist, 95)), poly_vs_linear_max=float(dist.max()), poly_fit_nodes=int(sel.sum()),
                poly_resid_median=float(np.median(np.linalg.norm(ctr - pred, axis=1)[sel])))
    return dict(ids=ids, pred=pred, pred_lin=pred_lin, ctr=ctr, amp=amp, wid=wid, aloc=aloc, model=model, quad=quad, info=info)


CRIT2 = {'main': dict(shift_max=0.25 * L.PITCH, amp_frac=0.40),
         'strict': dict(shift_max=0.15 * L.PITCH, amp_frac=0.55),
         'loose': dict(shift_max=0.35 * L.PITCH, amp_frac=0.25)}


def sub_flags(cf, crit):
    c = CRIT2[crit]
    shift = np.linalg.norm(cf['ctr'] - cf['pred'], axis=1)
    bad = (shift > c['shift_max']) | (cf['amp'] < c['amp_frac'] * cf['aloc']) | (cf['wid'] < 0.4)
    for a, b in cKDTree(cf['ctr']).query_pairs(2.0):
        bad[a if cf['amp'][a] < cf['amp'][b] else b] = True
    return bad


def one(rec):
    fid = rec['fid']; dst = F2 / f'{fid}.npz'; js = F2 / f'{fid}.json'
    if dst.exists() and js.exists() and json.loads(js.read_text(encoding='utf8')).get('ok') and 'frac_corr_px' in json.loads(js.read_text(encoding='utf8')):
        return json.loads(js.read_text(encoding='utf8'))
    t0 = time.time(); row = dict(fid=fid, ok=False)
    try:
        j1 = json.loads((F1 / f'{fid}.json').read_text(encoding='utf8')); z1 = np.load(F1 / f'{fid}.npz')
        A = z1['matrix']
        pre = L.read(rec['pre_path']); post = L.read(rec['post_path'])
        cpre = centers_smooth(pre); cpost = centers_smooth(post)
        Bpre = cpre['model'][1:].T; Bpost = cpost['model'][1:].T
        U = np.rint(np.linalg.inv(Bpost) @ A[:, :2] @ Bpre).astype(int)
        Llin = Bpost @ U @ np.linalg.inv(Bpre)
        c0 = np.array([W / 2, H / 2]); tL = (A[:, :2] @ c0 + A[:, 2]) - Llin @ c0
        # use smooth predictions (distortion-following) for the geometry of the map: linear map on lattice coords
        postpred_lin = cpost['pred_lin']
        # fractional (sub-period) translation from the lattices themselves; integer period from the registration centre
        expL = cpre['pred_lin'] @ Llin.T + tL
        _, j0 = cKDTree(postpred_lin).query(expL)
        f = (postpred_lin[j0] - expL) @ np.linalg.inv(Bpost).T
        sfrac = np.angle(np.exp(2j * np.pi * f).mean(axis=0)) / (2 * np.pi)
        frac_corr = Bpost @ sfrac
        tL = tL + frac_corr
        expL = cpre['pred_lin'] @ Llin.T + tL
        d_L, jL = cKDTree(postpred_lin).query(expL)
        expA = cpre['pred_lin'] @ A[:, :2].T + A[:, 2]
        d_A, jA = cKDTree(postpred_lin).query(expA)
        di = cpost['ids'][jL] - cpre['ids'] @ U.T
        vals, cnt = np.unique(di[d_L < 1.8], axis=0, return_counts=True); mode = vals[np.argmax(cnt)]
        consL = np.all(di == mode, axis=1) & (d_L < 2.0)
        diA = cpost['ids'][jA] - cpre['ids'] @ U.T
        valsA, cntA = np.unique(diA[d_A < 1.8], axis=0, return_counts=True); modeA = valsA[np.argmax(cntA)]
        consA = np.all(diA == modeA, axis=1) & (d_A < 2.0)
        np.savez_compressed(dst, pre_ids=cpre['ids'].astype(np.int32), pre_pred=cpre['pred'].astype(np.float32), pre_pred_lin=cpre['pred_lin'].astype(np.float32),
                            pre_ctr=cpre['ctr'].astype(np.float32), pre_amp=cpre['amp'].astype(np.float32), pre_wid=cpre['wid'].astype(np.float32), pre_aloc=cpre['aloc'].astype(np.float32),
                            post_ids=cpost['ids'].astype(np.int32), post_pred=cpost['pred'].astype(np.float32), post_pred_lin=cpost['pred_lin'].astype(np.float32),
                            post_ctr=cpost['ctr'].astype(np.float32), post_amp=cpost['amp'].astype(np.float32), post_wid=cpost['wid'].astype(np.float32), post_aloc=cpost['aloc'].astype(np.float32),
                            jL=jL.astype(np.int32), dL=d_L.astype(np.float32), consL=consL, jA=jA.astype(np.int32), dA=d_A.astype(np.float32), consA=consA,
                            U=U, Llin=Llin, tL=tL, A=A, frac_corr=frac_corr, pre_model=cpre['model'], post_model=cpost['model'])
        corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], float) - c0
        row.update(ok=True, pre_info=cpre['info'], post_info=cpost['info'], U=U.tolist(), index_modeL=[int(mode[0]), int(mode[1])], index_modeA=[int(modeA[0]), int(modeA[1])],
                   consistent_L=float(consL.mean()), frac_corr_px=float(np.linalg.norm(frac_corr)), dL_median=float(np.median(d_L)), dL_p95=float(np.percentile(d_L,95)), consistent_A=float(consA.mean()),
                   corner_disagree_px=float(np.linalg.norm((A[:, :2] - Llin) @ corners.T, axis=0).max()),
                   scale_A=float(np.sqrt(abs(np.linalg.det(A[:, :2])))), scale_L=float(np.sqrt(abs(np.linalg.det(Llin)))),
                   pitch_pre=cpre['info']['pitch_px'], pitch_post=cpost['info']['pitch_px'], seconds=time.time() - t0)
    except Exception as e:
        row.update(ok=False, error=repr(e), trace=traceback.format_exc()[-1500:], seconds=time.time() - t0)
    js.write_text(json.dumps(row, ensure_ascii=False, default=str), encoding='utf8')
    print('DONE' if row['ok'] else 'FAIL', fid, f"{row['seconds']:.0f}s", row.get('consistent_L', ''), row.get('error', ''), flush=True)
    return row


def main():
    F2.mkdir(parents=True, exist_ok=True)
    led = pd.read_csv(OUT / 'ledger.csv', dtype={'date': str, 'board': str})
    recs = led[led.pre_exists & led.post_exists].to_dict('records')
    workers = 3
    if '--workers' in sys.argv: workers = int(sys.argv[sys.argv.index('--workers') + 1])
    if '--pilot' in sys.argv:
        n = int(sys.argv[sys.argv.index('--pilot') + 1]); recs = [r for r in recs if r['fid'] in ('260825_7_2', '260827_5_3', '260825_0_1', '260828_7_8')][:n]
    if workers <= 1:
        res = [one(r) for r in recs]
    else:
        from multiprocessing import Pool
        with Pool(workers) as p: res = p.map(one, recs, chunksize=1)
    print('finished', sum(r['ok'] for r in res), '/', len(res))


if __name__ == '__main__':
    main()
