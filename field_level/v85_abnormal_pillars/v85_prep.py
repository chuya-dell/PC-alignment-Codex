"""v85 prep: abnormal-pillar table for the fixed-29 and blank fields of the development set (and the reference distance distribution of all valid pillars).
Abnormal pillar (C1, tentative rule 'not counted as signal, reported separately'): post-side fit failed (S5 invalid) while the P readout (S5P) is valid, and |z| > K_ABN = 6
(z = (S5P - field median)/(1.4826 MAD), same normalisation as C1; 6 sigma gives ~4.7 per blank field, i.e. the previous report's 'about 5'; 4 and 8 are kept as sensitivity columns).
Distances (periods = 7.37 px): to the cross-scar core (fitted centre lines of the v57 detector), to the bright band (shared.image_qc.bright_band_mask of the pre image; signed, negative inside), to the field edge.
Scar mask distance (sd_periods) as in C1.  Output: data/results/v85_abnormal_pillars/abn_pillars.csv, ref_hist.csv, field_summary.csv, band/{fid}_band.npz.  usage: python v85_prep.py [--workers K]"""
import sys, json, traceback
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, cv2, tifffile
from shared.image_qc import bright_band_mask
V84 = ROOT / 'data/results/v84_1fM_strategies'; OUT = ROOT / 'data/results/v85_abnormal_pillars'; (OUT / 'band').mkdir(parents=True, exist_ok=True)
V57 = Path(r'C:\Users\chuya\PC-alignment-localcorr\data\results\v57_false_positive_facts_20261006\stepD')
PER = 7.37; H, W = 2044, 2048; K_ABN = 6.0
EDGES = {'core': [0, 2, 4, 8, 16, 32, 1e9], 'band': [-1e9, 0, 2, 4, 8, 16, 1e9], 'edge': [0, 2, 4, 8, 16, 32, 1e9]}


def core_distance(lines, x, y):
    d = np.full(len(x), np.inf)
    for l in lines:
        if l['family'] == 'vertical':
            xl = l['X0_full'] + l['S'] * (y - (H - 2) / 2); dd = np.abs(x - xl) / np.sqrt(1 + l['S'] ** 2)
        else:
            yl = l['X0_full'] + l['S'] * (x - (W - 2) / 2); dd = np.abs(y - yl) / np.sqrt(1 + l['S'] ** 2)
        d = np.minimum(d, dd)
    return d / PER


def hist(vals, edges):
    return np.histogram(vals[np.isfinite(vals)], bins=edges)[0]


def one(rec):
    fid = rec['fid']
    try:
        z = np.load(V84 / 'cache' / f'{fid}.npz'); ctr = z['ctr'].astype(np.float64); z5p = z['z5p'].astype(np.float64); abn = z['abn']; sd = z['sd'].astype(np.float64)
        okP = np.isfinite(z5p)
        j = json.load(open(V57 / 'results' / f'{fid}.json', encoding='utf8')); lines = json.loads(j.get('lines_json', '[]'))
        core = core_distance(lines, ctr[:, 0], ctr[:, 1]) if lines else np.full(len(ctr), np.nan)
        pre = tifffile.imread(rec['pre_path'])
        bm = bright_band_mask(pre)
        if bm.any():
            outd = cv2.distanceTransform((~bm).astype(np.uint8), cv2.DIST_L2, 5); ind = cv2.distanceTransform(bm.astype(np.uint8), cv2.DIST_L2, 5)
            sdist = np.where(bm, -ind, outd) / PER; q = np.clip(np.rint(ctr).astype(int), [0, 0], [W - 1, H - 1]); band = sdist[q[:, 1], q[:, 0]]
            np.savez_compressed(OUT / 'band' / f'{fid}_band.npz', mask=np.packbits(bm), shape=np.array(bm.shape))
        else:
            band = np.full(len(ctr), np.nan)
        edge = np.minimum.reduce([ctr[:, 0], W - 1 - ctr[:, 0], ctr[:, 1], H - 1 - ctr[:, 1]]) / PER
        cand = abn & okP
        rows = []
        for i in np.where(cand & (np.abs(z5p) > 4))[0]:
            rows.append(dict(fid=fid, x=ctr[i, 0], y=ctr[i, 1], z=z5p[i], sd=sd[i], core=core[i], band=band[i], edge=edge[i]))
        ref = dict(fid=fid, n_valid=int(okP.sum()))
        for nm, v in (('core', core), ('band', band), ('edge', edge)):
            ref[f'h_{nm}'] = hist(v[okP], EDGES[nm]).tolist()
        okO = okP & (sd > 2)
        for nm, v in (('core', core), ('band', band), ('edge', edge)):
            ref[f'ho_{nm}'] = hist(v[okO], EDGES[nm]).tolist()
        ref['n_valid_out'] = int(okO.sum()); ref['has_band'] = bool(bm.any())
        ref['h_sd'] = hist(sd[okP], [-1e9, 0, 2, 4, 8, 1e9]).tolist()
        return rows, ref, dict(fid=fid, has_band=bool(bm.any()), band_frac=float(bm.mean()), n_lines=len(lines), n_abn_cand=int((cand & (sd > 2)).sum()),
                               n_abn_k4=int((cand & (np.abs(z5p) > 4) & (sd > 2)).sum()), n_abn_k6=int((cand & (np.abs(z5p) > 6) & (sd > 2)).sum()), n_abn_k8=int((cand & (np.abs(z5p) > 8) & (sd > 2)).sum()),
                               n_abn_k6_all=int((cand & (np.abs(z5p) > 6)).sum()), ok=True)
    except Exception as e:
        return [], None, dict(fid=fid, ok=False, error=repr(e), trace=traceback.format_exc()[-500:])


if __name__ == '__main__':
    led = pd.read_csv(ROOT / 'data/results/v70_ledger_center_fit/ledger.csv', dtype={'date': str, 'board': str}); meta = pd.read_csv(V84 / 'field_meta.csv', dtype={'date': str, 'board': str})
    sel = led[led.fid.isin(meta[meta.ok & ((meta.group == 'blank') | meta.fixed29)].fid)]
    print('fields', len(sel), 'fixed29', int(sel.fixed29.sum()), 'blank', int((sel.group == 'blank').sum()))
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 6
    from multiprocessing import Pool
    A, Rr, S = [], [], []
    with Pool(workers) as p:
        for rows, ref, summ in p.imap_unordered(one, sel.to_dict('records'), chunksize=1):
            A += rows; S.append(summ)
            if ref: Rr.append(ref)
    pd.DataFrame(A).to_csv(OUT / 'abn_pillars.csv', index=False)
    pd.DataFrame(S).to_csv(OUT / 'field_summary.csv', index=False)
    pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in Rr]).to_csv(OUT / 'ref_hist.csv', index=False)
    s = pd.DataFrame(S); print(s.ok.value_counts().to_dict()); print(s[s.ok][['n_abn_k4', 'n_abn_k6', 'n_abn_k8', 'n_abn_k6_all', 'has_band']].describe().round(2))
