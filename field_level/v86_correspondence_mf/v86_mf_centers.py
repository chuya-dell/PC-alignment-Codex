"""v86 (part C): matched-filter pillar fit to raise the correspondence rate (fraction of pillars that are NOT substituted).
Baseline (v70_centers2): Gaussian-weighted centroid within 3 px of the smooth-lattice prediction; amplitude = centre sample - 20th percentile of the 9x9 patch; substitution if shift > 0.25 pitch, amp < 0.40*local median
amplitude, width < 0.4, or two nodes on one pillar.  The pairing pre->post (jL, consL), the lattice and the substitution THRESHOLDS are kept; only the estimator changes:
 MF  : response R = Gaussian(sigma_w) * contrast; centre = sub-pixel maximum of R within +-2 px of the prediction (0.25-px search, parabolic refinement); amplitude = R(peak) - 20th percentile of R in the 9x9 patch.
Rate = n(ok)/n(overlap) as in v70_summary2 (okL = ~sub_pre & ~sub_post[jL] & consL & overlap).   usage: python v86_mf_centers.py [--sigma 1.3] [--workers K] [--set fixed29blank|all]"""
import sys, json, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit'))
import numpy as np, pandas as pd, cv2
from scipy import ndimage
import v70_lib as L
import v70_centers2 as C
V70 = ROOT / 'data/results/v70_ledger_center_fit'; OUT = ROOT / 'data/results/v86_correspondence_mf'; (OUT / 'fields').mkdir(parents=True, exist_ok=True)
W, H = 2048, 2044
OFF = np.arange(-2.0, 2.0001, 0.25)
_OY, _OX = np.meshgrid(OFF, OFF, indexing='ij'); _OY = _OY.ravel(); _OX = _OX.ravel()
_PY, _PX = np.mgrid[-4:5, -4:5]; _PY = _PY.ravel().astype(np.float64); _PX = _PX.ravel().astype(np.float64)


def sample(R, x, y):
    return ndimage.map_coordinates(R, [y, x], order=1, mode='nearest')


def mf_fit(im, pred, sigma_w=1.3):
    """returns ctr, amp, wid (wid from the old second-moment estimate at the new centre)"""
    cb = L.contrast_plain(im); R = cv2.GaussianBlur(cb, (0, 0), sigma_w).astype(np.float32)
    n = len(pred); best = np.full(n, -np.inf); bi = np.zeros(n, int)
    vals = np.empty((len(_OX), n), np.float32)
    for k, (dy, dx) in enumerate(zip(_OY, _OX)):
        vals[k] = sample(R, pred[:, 0] + dx, pred[:, 1] + dy)
    bi = vals.argmax(axis=0); bestv = vals[bi, np.arange(n)]
    ctr = pred + np.column_stack([_OX[bi], _OY[bi]])
    # parabolic refinement on the 0.25-px grid
    g = len(OFF)
    iy, ix = np.divmod(bi, g)
    def at(dy, dx):
        jy = np.clip(iy + dy, 0, g - 1); jx = np.clip(ix + dx, 0, g - 1); return vals[jy * g + jx, np.arange(n)]
    c0 = bestv; cxm, cxp, cym, cyp = at(0, -1), at(0, 1), at(-1, 0), at(1, 0)
    dxs = 0.25 * 0.5 * (cxm - cxp) / (cxm - 2 * c0 + cxp - 1e-12); dys = 0.25 * 0.5 * (cym - cyp) / (cym - 2 * c0 + cyp - 1e-12)
    ctr = ctr + np.column_stack([np.clip(dxs, -.25, .25), np.clip(dys, -.25, .25)])
    # amplitude: R(peak) - 20th percentile of R in the 9x9 patch around the new centre
    xs = ctr[:, 0][:, None] + _PX[None]; ys = ctr[:, 1][:, None] + _PY[None]
    patch = ndimage.map_coordinates(R, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(n, -1)
    base = np.percentile(patch, 20, axis=1)
    amp = sample(R, ctr[:, 0], ctr[:, 1]) - base
    # width: same second-moment estimator as the baseline, on the contrast image at the new centre
    v = ndimage.map_coordinates(cb, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(n, -1)
    b2 = np.percentile(v, 20, axis=1)[:, None]; r2 = _PX[None] ** 2 + _PY[None] ** 2
    w2 = np.maximum(v - b2, 0) * (r2 <= L.R ** 2); s2 = w2.sum(axis=1) + 1e-12
    mx = (w2 * _PX[None]).sum(axis=1) / s2; my = (w2 * _PY[None]).sum(axis=1) / s2
    var = ((w2 * ((_PX[None] - mx[:, None]) ** 2 + (_PY[None] - my[:, None]) ** 2)).sum(axis=1) / s2) / 2.0
    return ctr, amp, np.sqrt(np.maximum(var, 0))


def rate_from(z, pre, post):
    ovL_exp = z['pre_pred_lin'] @ z['Llin'].T + z['tL']; ovL = (ovL_exp[:, 0] >= 8) & (ovL_exp[:, 0] < W - 8) & (ovL_exp[:, 1] >= 8) & (ovL_exp[:, 1] < H - 8)
    out = {}
    for crit in ('main', 'strict', 'loose'):
        Sp = C.sub_flags(pre, crit); So = C.sub_flags(post, crit); okL = (~Sp) & (~So[z['jL']]) & z['consL'] & ovL
        out[crit] = (float(okL.sum() / max(ovL.sum(), 1)), okL, Sp, So, ovL)
    return out


def one(args):
    rec, sigma = args; fid = rec['fid']; dst = OUT / 'fields' / f'{fid}_s{sigma}.json'
    if dst.exists(): return json.loads(dst.read_text(encoding='utf8'))
    cv2.setNumThreads(1); t0 = time.time(); row = dict(fid=fid, sigma_w=sigma, ok=False)
    try:
        z = np.load(V70 / 'fields2' / f'{fid}.npz')
        pre_b = dict(ctr=z['pre_ctr'].astype(float), pred=z['pre_pred'], amp=z['pre_amp'], wid=z['pre_wid'], aloc=z['pre_aloc'])
        post_b = dict(ctr=z['post_ctr'].astype(float), pred=z['post_pred'], amp=z['post_amp'], wid=z['post_wid'], aloc=z['post_aloc'])
        base = rate_from(z, pre_b, post_b)
        pre_im = L.read(rec['pre_path']); post_im = L.read(rec['post_path'])
        res = {}
        for nm, im, pr in (('pre', pre_im, z['pre_pred'].astype(np.float64)), ('post', post_im, z['post_pred'].astype(np.float64))):
            ctr, amp, wid = mf_fit(im, pr, sigma); aloc, _ = C.local_aref(ctr, amp); res[nm] = dict(ctr=ctr, pred=z[f'{nm}_pred'], amp=amp, wid=wid, aloc=aloc)
        new = rate_from(z, res['pre'], res['post'])
        for crit in ('main', 'strict', 'loose'):
            row[f'rate_base_{crit}'] = base[crit][0]; row[f'rate_mf_{crit}'] = new[crit][0]
        okb, Spb, Sob, ov = base['main'][1], base['main'][2], base['main'][3], base['main'][4]
        okn, Spn, Son = new['main'][1], new['main'][2], new['main'][3]
        row.update(sub_pre_base=float(Spb[ov].mean()), sub_pre_mf=float(Spn[ov].mean()), sub_post_base=float(Sob.mean()), sub_post_mf=float(Son.mean()), n_overlap=int(ov.sum()),
                   shift_med_base=float(np.median(np.linalg.norm(pre_b['ctr'] - pre_b['pred'], axis=1))), shift_med_mf=float(np.median(np.linalg.norm(res['pre']['ctr'] - res['pre']['pred'], axis=1))),
                   amp_cv_base=float(np.std(pre_b['amp'] / pre_b['aloc'])), amp_cv_mf=float(np.std(res['pre']['amp'] / res['pre']['aloc'])), consL=float(z['consL'][ov].mean()))
        # per-pillar failing reasons (new): amplitude / shift / duplicate / index
        c = C.CRIT2['main']
        for nm, d in (('pre', res['pre']), ('post', res['post'])):
            shift = np.linalg.norm(d['ctr'] - d['pred'], axis=1); row[f'fail_amp_{nm}_mf'] = float((d['amp'] < c['amp_frac'] * d['aloc'])[ov if nm == 'pre' else slice(None)].mean())
            row[f'fail_shift_{nm}_mf'] = float((shift > c['shift_max'])[ov if nm == 'pre' else slice(None)].mean())
        np.savez_compressed(OUT / 'fields' / f'{fid}_s{sigma}.npz', ok_mf=okn, ok_base=okb, ov=ov, sub_pre=Spn, sub_post_at_pre=Son[z['jL']], pre_ctr=res['pre']['ctr'].astype(np.float32), pre_amp=res['pre']['amp'].astype(np.float32), pre_aloc=res['pre']['aloc'].astype(np.float32))
        row.update(ok=True, seconds=time.time() - t0)
    except Exception as e:
        row.update(error=repr(e), trace=traceback.format_exc()[-800:])
    dst.write_text(json.dumps(row, ensure_ascii=False, default=float), encoding='utf8'); return row


if __name__ == '__main__':
    sigma = float(sys.argv[sys.argv.index('--sigma') + 1]) if '--sigma' in sys.argv else 1.3
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 6
    which = sys.argv[sys.argv.index('--set') + 1] if '--set' in sys.argv else 'fixed29blank'
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str})
    sel = led[led.pre_exists & led.post_exists]
    if which == 'fixed29blank': sel = sel[sel.fixed29 | (sel.group == 'blank')]
    if '--limit' in sys.argv: sel = sel.iloc[:int(sys.argv[sys.argv.index('--limit') + 1])]
    from multiprocessing import Pool
    with Pool(workers) as p: rows = p.map(one, [(r, sigma) for r in sel.to_dict('records')], chunksize=1)
    t = pd.DataFrame(rows); t.to_csv(OUT / f'rates_{which}_s{sigma}.csv', index=False)
    ok = t[t.ok == True]; print('fields', len(ok), 'failed', int((t.ok != True).sum()))
    for crit in ('main', 'strict', 'loose'):
        print(crit, 'base median %.4f min %.4f | MF median %.4f min %.4f' % (ok[f'rate_base_{crit}'].median(), ok[f'rate_base_{crit}'].min(), ok[f'rate_mf_{crit}'].median(), ok[f'rate_mf_{crit}'].min()))
