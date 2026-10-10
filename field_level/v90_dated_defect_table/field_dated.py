"""Codexによる日付別欠陥表の事前登録第3版の開発用再評価。"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import beta, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(r'C:\Users\chuya\PC-alignment-fp\data\results')
OUT = ROOT / 'data/results/v90_dated_defect_table'
VAULT = Path(r'W:\GoogleDrive\chuya2816\Obsidian Vault')
PREREG = VAULT / '_依頼記録/偽陽性ゼロ化_C1D/261010_日付別欠陥表(iv)_設計と合否基準_事前登録.md'
SEED = 26101090
GRID = np.arange(3., 15.01, .5)
NS = (8, 12, 16, 24, 32, 48)
AUDIT = []
ALLOWED = set()
LABEL = '事後の再評価（独立な検証ではない）'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(name, rows, columns=None):
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig', mode='x')
    return df


def load(folder, fid):
    assert fid in ALLOWED and fid[:6] not in ('261008', '260925')
    p = SRC / folder / (fid + '.npz')
    assert 'v83_new_shots_261008' not in str(p)
    AUDIT.append(dict(fid=fid, path=str(p), purpose='development_only'))
    return np.load(p, allow_pickle=False)


def component_table(xy, field):
    if not len(xy):
        return np.empty((0, 4))
    pairs = cKDTree(xy).query_pairs(10., output_type='ndarray')
    g = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy)))
    nc, lab = connected_components(g, directed=False)
    counts = np.bincount(lab, minlength=nc)
    candidates = np.flatnonzero(counts >= 8)
    if not len(candidates):
        return np.empty((0, 4))
    sx = np.bincount(lab, weights=xy[:, 0], minlength=nc)
    sy = np.bincount(lab, weights=xy[:, 1], minlength=nc)
    nf = np.bincount(np.unique(np.column_stack((lab, field)), axis=0)[:, 0], minlength=nc)
    return np.column_stack((sx[candidates]/counts[candidates], sy[candidates]/counts[candidates], counts[candidates], nf[candidates]))


def combine(fields, key='points'):
    sizes = [len(f[key]) for f in fields]
    if not sum(sizes):
        return np.empty((0, 2)), np.empty(0, int)
    return np.concatenate([f[key] for f in fields]), np.repeat(np.arange(len(fields)), sizes)


def build(fields, minimum):
    xy, fi = combine(fields)
    t = component_table(xy, fi)
    return t[t[:, 3] >= minimum, :2]


def union(*tables):
    # Union of circles, not centroid merging: moving a centroid alters exclusion.
    nonempty = [t for t in tables if len(t)]
    return np.unique(np.concatenate(nonempty), axis=0) if nonempty else np.empty((0, 2))


def covered(points, centers, radius=20):
    return cKDTree(centers).query(points)[0] < radius if len(centers) and len(points) else np.zeros(len(points), bool)


def match(a, b):
    # Maximum one-to-one matching prevents one center from reproducing two.
    from scipy.sparse.csgraph import maximum_bipartite_matching
    if not len(a) or not len(b):
        return np.zeros(len(a), bool)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = np.where(d <= 5)
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(a), len(b))).tocsr()
    return maximum_bipartite_matching(g, perm_type='column') >= 0


def cp_upper(k, n):
    return 1. if k == n else float(beta.ppf(.95, k+1, n-k))


def ring_indices(xy, center, n):
    """Equal-count circular nearest-neighbor controls, entirely inside 30..60px."""
    if n == 0:
        return []
    dist = np.linalg.norm(xy-center, axis=1)
    ann = np.flatnonzero((dist >= 30) & (dist <= 60))
    if len(ann) < n:
        return []
    candidates = center + 45*np.column_stack((np.cos(np.arange(16)*2*np.pi/16), np.sin(np.arange(16)*2*np.pi/16)))
    result, seen = [], set()
    for anchor in candidates:
        dd = np.linalg.norm(xy[ann]-anchor, axis=1)
        ix = ann[np.argsort(dd, kind='stable')[:n]]
        rr = float(np.max(np.linalg.norm(xy[ix]-anchor, axis=1)))
        ac = np.linalg.norm(anchor-center)
        if ac-rr < 30 or ac+rr > 60:
            continue
        sig = tuple(sorted(ix))
        if sig not in seen:
            seen.add(sig)
            result.append((ix, rr))
    return result


def bootstrap(values, rng, reps=2000):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan, 0
    means = v[rng.integers(len(v), size=(reps, len(v)))].mean(axis=1)
    return float(v.mean()), *np.quantile(means, [.025, .975]).tolist(), len(v)


def local_row(f, z, a, c, defs):
    xy = f['xy']
    finite = np.isfinite(z['z5p'])
    exc = z['excl'].astype(bool)
    zabs = np.abs((a['S5'].astype(float)-f['median'])/f['scale'])
    # All finite pillars are the registered primary local population.
    result, img = [], []
    for d in defs.itertuples():
        center = np.array([d.x_camera, d.y_camera])
        ids = np.flatnonzero(np.linalg.norm(xy-center, axis=1) <= 10)
        rings = ring_indices(xy, center, len(ids))
        row = dict(fid=f['fid'], date=f['date'], group='blank' if f['group']=='blank' else 'nonblank', defect_id=d.defect_id,
                   n_center=len(ids), n_controls=len(rings))
        vv = zabs[ids]; vv = vv[np.isfinite(vv)]
        mx = float(vv.max()) if len(vv) else np.nan
        row['center_max_abs_z'] = mx
        for k in (3., 4.5, 6.):
            control = []
            for ix, _ in rings:
                v = zabs[ix]; v = v[np.isfinite(v)]
                if len(v):
                    control.append(float(v.max() > k))
            row[f'center_gt{k:g}'] = float(mx > k) if np.isfinite(mx) else np.nan
            row[f'control_gt{k:g}'] = float(np.mean(control)) if control else np.nan
            row[f'difference_gt{k:g}'] = row[f'center_gt{k:g}']-row[f'control_gt{k:g}']
        for name, values in [('excl_false', ~exc), ('z5p_nonfinite', ~finite)]:
            row[name+'_center'] = int(values[ids].sum())
            row[name+'_control'] = float(np.mean([values[ix].sum() for ix, _ in rings])) if rings else np.nan
            row[name+'_difference'] = row[name+'_center']-row[name+'_control']
        pp = f['points'][np.linalg.norm(f['points']-center, axis=1) <= 20]
        row['n_detected'] = len(pp)
        row['detected_x'] = float(pp[:, 0].mean()) if len(pp) else np.nan
        row['detected_y'] = float(pp[:, 1].mean()) if len(pp) else np.nan
        result.append(row)
        for side in ('pre', 'post'):
            pxy = c[side+'_ctr'].astype(float)
            ok = np.isfinite(pxy).all(axis=1)
            pxy = pxy[ok]
            ci = np.flatnonzero(np.linalg.norm(pxy-center, axis=1) <= 10)
            ri = ring_indices(pxy, center, len(ci))
            ir = dict(fid=f['fid'], date=f['date'], group='blank' if f['group']=='blank' else 'nonblank', defect_id=d.defect_id, side=side,
                      n_center=len(ci), n_controls=len(ri))
            for metric in ('wid', 'amp'):
                vals = c[side+'_'+metric][ok]
                ir[metric+'_center'] = float(np.nanmean(vals[ci])) if len(ci) else np.nan
                ir[metric+'_control'] = float(np.nanmean([np.nanmean(vals[ii]) for ii, _ in ri])) if ri else np.nan
                ir[metric+'_difference'] = ir[metric+'_center']-ir[metric+'_control']
            # Equal counts require density per circle area, not the count ratio.
            control_density = np.mean([len(ii)/(np.pi*rr*rr) for ii, rr in ri if rr > 0]) if ri else np.nan
            ir['density_ratio'] = len(ci)/(np.pi*100)/control_density if control_density > 0 else np.nan
            img.append(ir)
    return result, img


def prepare(meta, defs):
    fields, locals_, image, scales = [], [], [], []
    for idx, r in enumerate(meta.itertuples()):
        with load('v84_1fM_strategies/cache', r.fid) as z, load('v70_ledger_center_fit/fields2', r.fid) as c, load('v72_real_field_readout/fields', r.fid) as a:
            z = {k:z[k] for k in ('ctr','excl','abn','z5p')}
            c = {k:c[k] for k in ('pre_ctr','pre_amp','pre_wid','post_ctr','post_amp','post_wid','jL','Llin','tL')}
            a = {k:a[k] for k in ('ctr_xy','S5')}
            xy = z['ctr'].astype(float) @ c['Llin'].T + c['tL']
            assert np.allclose(a['ctr_xy'], z['ctr'], equal_nan=True)
            valid = np.isfinite(a['S5']) & z['excl']
            delta = a['S5'].astype(float)
            med = float(np.median(delta[valid])); scale = float(1.4826*np.median(np.abs(delta[valid]-med)))
            assert scale > 0 and valid.sum() >= 5000
            hit = z['abn'] & np.isfinite(z['z5p']) & z['excl'] & (np.abs(z['z5p']) > 6)
            f = dict(fid=r.fid, date=r.date, group=r.group, points=xy[hit], xy=xy,
                     valid_xy=xy[z['excl'] & np.isfinite(xy).all(axis=1)].astype(np.float32),
                     n_valid=int(valid.sum()), median=med, scale=scale)
            lr, ir = local_row(f, z, a, c, defs)
            locals_.extend(lr); image.extend(ir)
            scales.append(dict(fid=r.fid, date=r.date, group='blank' if r.group=='blank' else 'nonblank', median=med, mad_scale=scale,
                               n_valid=int(valid.sum()), sigma=r.sigma))
            if r.group == 'blank':
                post = c['post_ctr'][c['jL']].astype(float)
                fallback = ~np.isfinite(delta) | ~np.isfinite(post).all(axis=1)
                post[fallback] = xy[fallback]
                f.update(delta=delta[valid], pre=z['ctr'][valid].astype(float), post=post[valid],
                         fixed_z=np.abs((delta[valid]-med)/scale))
                f['pre_tree'] = cKDTree(f['pre']); f['post_tree'] = cKDTree(f['post'])
            else:
                # Same pre/post exclusion union as v88 and blank evaluation.
                post = c['post_ctr'][c['jL']].astype(float)
                fallback = ~np.isfinite(delta) | ~np.isfinite(post).all(axis=1)
                post[fallback] = xy[fallback]
                f['pre_tree'] = cKDTree(z['ctr'][valid].astype(float))
                f['post_tree'] = cKDTree(post[valid])
            del f['xy']
            fields.append(f)
        if idx % 50 == 0:
            print(f'Codex: loaded development fields {idx}/{len(meta)}', flush=True)
    return fields, pd.DataFrame(locals_), pd.DataFrame(image), pd.DataFrame(scales)


def null_thresholds(bydate):
    rows, diagnostics, thresholds = [], [], {}
    joint_vectors = {}
    for di, (date, fields) in enumerate(bydate.items()):
        sizes = sorted(set([len(fields), len(fields)//2, len(fields)-len(fields)//2]+[n for n in NS if n <= len(fields)]))
        distributions = {(kind, n): [] for kind in ('resample', 'translate') for n in sizes}
        trees = [cKDTree(f['valid_xy']) for f in fields]
        domains = [(f['valid_xy'].min(axis=0),f['valid_xy'].max(axis=0),float(np.median(t.query(f['valid_xy'][::max(1,len(f['valid_xy'])//256)],k=2)[0][:,1]))/np.sqrt(2)) for f,t in zip(fields,trees)]
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 1, di]))
        removed, total = 0, 0
        for rep in range(1000):
            order = rng.permutation(len(fields))
            nulls = {'resample': [], 'translate': []}
            for f, tree, domain in zip(fields, trees, domains):
                p = f['points']; valid = f['valid_xy']
                nulls['resample'].append(valid[rng.integers(len(valid), size=len(p))].astype(float))
                if len(p):
                    # One common displacement per field; no torus wrapping.
                    lo = domain[0]-p.min(axis=0)
                    hi = domain[1]-p.max(axis=0)
                    shift = rng.uniform(np.minimum(lo, hi), np.maximum(lo, hi))
                    dist, ix = tree.query(p+shift)
                    # Points beyond one local half-diagonal are outside effective lattice.
                    keep = dist <= domain[2]
                    snapped = valid[ix[keep]].astype(float)
                    removed += int((~keep).sum()); total += len(p)
                else:
                    snapped = np.empty((0, 2))
                nulls['translate'].append(snapped)
            for kind, sets in nulls.items():
                for n in sizes:
                    selected = [sets[i] for i in order[:n]]
                    counts = [len(p) for p in selected]
                    xy = np.concatenate(selected) if sum(counts) else np.empty((0, 2))
                    comp = component_table(xy, np.repeat(np.arange(n), counts))
                    distributions[kind, n].append(comp[:, 3].astype(np.int16))
            if rep % 200 == 0:
                print(f'Codex: null date {date}, iteration {rep}/1000', flush=True)
        for (kind, n), vals in distributions.items():
            joint_vectors[date, kind, n] = np.array([int(v.max()) if len(v) else 0 for v in vals])
            for purpose in ('table', 'E5'):
                chosen = None
                for m in range(3, n+2):
                    numbers = np.array([np.count_nonzero(v >= m) for v in vals])
                    k = int(np.count_nonzero(numbers))
                    p = k/1000; upper = cp_upper(k, 1000)
                    # Nine-table family under independent date-specific null draws.
                    family_p = 1-(1-p)**9
                    family_upper = 1-(1-upper)**9
                    passes = family_upper <= .05 if purpose == 'table' else numbers.mean() <= .05
                    rows.append(dict(date=date, N=n, null=kind, purpose=purpose, m=m,
                                     iterations=1000, mean_false_centers=float(numbers.mean()),
                                     probability_any=p, probability_upper95=upper,
                                     nine_table_probability=family_p, nine_table_upper95=family_upper,
                                     passes=bool(passes), seed=SEED, stream=f'1,{di}'))
                    if passes and chosen is None:
                        chosen = m
                assert chosen is not None
                thresholds[date, n, purpose, kind] = chosen
        diagnostics.append(dict(date=date, translated_points=total, removed_outside_effective_domain=removed,
                                removed_fraction=removed/total if total else 0))
        del trees
    full = write('null_threshold_scan.csv', rows)
    write('null_translation_diagnostics.csv', diagnostics)
    # Registered family: nine actual date-specific tables, one joint event
    # per independent Monte Carlo iteration; CP upper is on that joint event.
    joint_rows = []
    joint_threshold = {}
    for family in ('full', 'half_low', 'half_high'):
        for kind in ('resample', 'translate'):
            vectors = []
            for date, fields in bydate.items():
                n = len(fields) if family == 'full' else len(fields)//2 if family == 'half_low' else len(fields)-len(fields)//2
                vectors.append(joint_vectors[date, kind, n])
            maxima = np.max(vectors, axis=0)
            selected = None
            for m in range(3, max(map(len, bydate.values()))+2):
                k = int((maxima >= m).sum())
                upper = cp_upper(k, 1000)
                joint_rows.append(dict(family=family, null=kind, m=m, iterations=1000,
                                       nine_table_events=k, nine_table_probability=k/1000,
                                       nine_table_upper95=upper, passes=upper <= .05))
                if upper <= .05 and selected is None:
                    selected = m
            joint_threshold[family, kind] = selected
    write('null_nine_table_scan.csv', joint_rows)
    for date, fields in bydate.items():
        for kind in ('resample', 'translate'):
            for family, n in [('full', len(fields)), ('half_low', len(fields)//2), ('half_high', len(fields)-len(fields)//2)]:
                thresholds[date, n, 'table', kind] = joint_threshold[family, kind]
    adopted = []
    minimum = {}
    for date, fields in bydate.items():
        sizes = sorted({n for d, n, purpose, kind in thresholds if d == date})
        for n in sizes:
            for purpose in ('table', 'E5'):
                a, b = [thresholds[date, n, purpose, kind] for kind in ('resample', 'translate')]
                m = max(a, b); minimum[date, n, purpose] = m
                adopted.append(dict(date=date, N=n, purpose=purpose, m_resample=a, m_translate=b, m_adopted=m,
                                    iterations=1000, family_rule='joint_nine_date_CP_upper95<=.05' if purpose == 'table' and n in [len(fields), len(fields)//2, len(fields)-len(fields)//2] else 'conditional_conservative_nine_tables' if purpose=='table' else 'mean<=.05',
                                    impossible_nonempty_table=m > n))
    write('m_of_N.csv', adopted)
    return minimum, full


def exclusion_ids(tree, centers):
    if not len(centers):
        return np.empty(0, int)
    q = tree.query_ball_point(centers, 20.)
    ids = [i for ll in q for i in ll]
    if not ids:
        return np.empty(0, int)
    ids = np.unique(ids)
    # Exact v88 boundary: distance>=20 remains valid.
    return ids[covered(tree.data[ids], centers)]


def field_exclusion_ids(f, centers):
    return np.union1d(exclusion_ids(f['pre_tree'],centers),exclusion_ids(f['post_tree'],centers))


def evaluate_blank(f, centers, arm, rep=-1, n=0):
    ids = field_exclusion_ids(f, centers)
    cache=f.setdefault('evaluation_cache',{})
    key=ids.tobytes()
    if key not in cache:
        keep = np.ones(len(f['delta']), bool); keep[ids] = False
        v=f['delta'][keep]
        med=np.median(v);scale=1.4826*np.median(np.abs(v-med))
        assert len(v)>=5000 and scale>0
        counts=[]
        for zz in [np.abs((v-med)/scale),f['fixed_z'][keep]]:
            zz=np.sort(zz)
            counts.append(len(zz)-np.searchsorted(zz,GRID,side='right'))
        cache[key]=(len(v),float(med),float(scale),counts)
    nv,med,scale,counts=cache[key]
    rows=[]
    for norm,cc in zip(['recomputed','fixed'],counts):
        for k,count in zip(GRID,cc):
            rows.append(dict(arm=arm,repetition=rep,n=n,fid=f['fid'],date=f['date'],normalization=norm,
                             k=k,count=int(count),fp91k=float(count/nv*91000),n_valid=nv,n_before=len(f['delta']),
                             lost_fraction=len(ids)/len(f['delta']),median=med,scale=scale))
    return rows


def m3_rows(fields, centers, arm, evaluation, rep=-1, n=0):
    return [dict(arm=arm, evaluation=evaluation, repetition=rep, n=n, fid=f['fid'], date=f['date'],
                 n_abnormal=len(f['points']), n_covered=int(covered(f['points'], centers).sum()),
                 coverage=float(covered(f['points'], centers).mean()) if len(f['points']) else np.nan) for f in fields]


def summaries(m1):
    keys = ['arm', 'repetition', 'n', 'normalization', 'k']
    agg = dict(n_fields=('fid', 'size'), mean_fp91k=('fp91k', 'mean'), p95_fp91k=('fp91k', lambda x:x.quantile(.95)),
               total_count=('count', 'sum'), mean_count=('count', 'mean'), max_lost_fraction=('lost_fraction', 'max'))
    daily = m1.groupby(keys+['date']).agg(**agg).reset_index()
    all_ = m1.groupby(keys).agg(**agg).reset_index(); all_['date'] = 'all'
    return pd.concat([daily, all_], ignore_index=True)


def extra_checks():
    """Additional registered controls and prediction checks, from v90 tables only."""
    local=pd.read_csv(OUT/'local_signal_fields.csv',dtype={'date':str})
    defs=pd.read_csv(ROOT/'data/raw/v88_camera_defects_unapproved.csv')
    controls=defs[~defs.defect_id.isin([8,40,157,309])].defect_id.tolist()
    rng=np.random.default_rng(np.random.SeedSequence([SEED,4]))
    rows=[]
    for boundary in ['260922','260924']:
        for (group,did),g in local[local.defect_id.isin(controls)].groupby(['group','defect_id']):
            for metric in ['difference_gt3','difference_gt4.5','difference_gt6']:
                sides=[]
                for side,q in [('early',g[g.date<=boundary]),('late',g[g.date>boundary])]:
                    chunks=[v[metric].dropna().to_numpy(float) for _,v in q.groupby('date')]
                    chunks=[v for v in chunks if len(v)]
                    nv=sum(map(len,chunks))
                    if nv:
                        draws=sum(v[rng.integers(len(v),size=(2000,len(v)))].sum(axis=1) for v in chunks)/nv
                        avg=sum(v.sum() for v in chunks)/nv
                        lo,hi=np.quantile(draws,[.025,.975])
                    else:avg=lo=hi=np.nan;draws=np.full(2000,np.nan)
                    rows.append(dict(group=group,defect_id=did,boundary=boundary,metric=metric,side=side,
                                     mean=avg,ci95_low=lo,ci95_high=hi,n_fields=nv))
                    sides.append((avg,draws))
                change=sides[1][1]-sides[0][1]
                lo,hi=np.nanquantile(change,[.025,.975]) if np.any(np.isfinite(change)) else (np.nan,np.nan)
                rows.append(dict(group=group,defect_id=did,boundary=boundary,metric=metric,side='late_minus_early',
                                 mean=sides[1][0]-sides[0][0],ci95_low=lo,ci95_high=hi,n_fields=len(g)))
    write('stable_nine_local_boundaries.csv',rows)
    centers=pd.read_csv(OUT/'dated_centers.csv',dtype={'date':str})
    dates=sorted(local.date.unique())
    predictions=[]
    for did in [8,157,309]:
        origin=defs[defs.defect_id==did][['x_camera','y_camera']].to_numpy()[0]
        for date in dates:
            for arm in ['E3','E4','E7']:
                if arm in ['E4','E7'] and date==dates[0]:continue
                ct=centers[(centers.date==date)&(centers.arm==arm)][['x','y']].to_numpy()
                recovered=bool(match(origin[None,:],ct)[0])
                expected=None
                if arm=='E3':expected=date<='260922' if did==8 else date>='260926'
                if arm=='E4' and did==8 and date=='260923':expected=True
                if arm=='E4' and did in [157,309] and date=='260926':expected=False
                predictions.append(dict(defect_id=did,date=date,arm=arm,recovered=recovered,expected=expected,
                                        matches_prediction=recovered==expected if expected is not None else None))
    write('preregistered_predictions.csv',predictions)
    # Include empty tables explicitly, so all date/arm combinations are auditable.
    sizes=[]
    for date in dates:
        for arm in ['E0','E1','E2','E3','E3u','E4','E4u','E7','E8']:
            available=not(arm in ['E4','E4u','E7'] and date==dates[0])
            sizes.append(dict(date=date,arm=arm,available=available,
                              n_centers=int(((centers.date==date)&(centers.arm==arm)).sum()) if available else np.nan))
    write('dated_table_sizes.csv',sizes)
    m1=pd.read_csv(OUT/'M1.csv',dtype={'date':str})
    worse=[]
    for norm in ['fixed','recomputed']:
        for k in [6,8,10]:
            q=m1[(m1.normalization==norm)&(m1.k==k)]
            base=q[q.arm=='E2'].set_index('fid')
            for arm in ['E3','E3u','E4','E4u','E7']:
                t=q[q.arm==arm].set_index('fid')
                b=base.loc[t.index]
                for fid in t.index[t['count']>b['count']]:
                    worse.append(dict(arm=arm,normalization=norm,k=k,fid=fid,date=t.loc[fid,'date'],
                                      count_arm=int(t.loc[fid,'count']),count_E2=int(b.loc[fid,'count'])))
    write('worsened_blank_counts_all_arms.csv',worse,columns=['arm','normalization','k','fid','date','count_arm','count_E2'])
    return predictions


def main():
    global ALLOWED
    start = time.perf_counter(); utc = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    assert all(p.name.startswith('run_started') for p in OUT.iterdir()), 'Codexは既存の第90版結果を上書きしない。'
    inputs = [PREREG, ROOT/'data/raw/v88_camera_defects_unapproved.csv', ROOT/'field_level/v88_defect_exclusion/field_stage1.py',
              ROOT/'field_level/v89_defect_cause/field_cause.py', SRC/'v85_abnormal_pillars/E4_counts_all632.csv',
              ROOT/'data/results/v88_defect_exclusion/blank_counts_by_k.csv']
    hashes = {str(p):sha(p) for p in inputs}
    (OUT/('run_started_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')).write_text(json.dumps(dict(start=utc, seed=SEED, task='v90_only', inputs=hashes), ensure_ascii=False, indent=2), encoding='utf-8')
    meta = pd.read_csv(inputs[4], dtype={'date':str, 'board':str})
    assert len(meta) == 632 and meta.fid.is_unique
    assert not meta.date.isin(['261008', '260925']).any()
    mean = meta.groupby(['date', 'board']).abn6_out.transform('mean')
    nb = meta[(meta.group != 'blank') & (mean < 50)]
    blank = meta[meta.group == 'blank']
    assert len(nb) == 547 and len(blank) == 69
    selected = pd.concat([nb, blank]).sort_values(['date', 'fid'])
    ALLOWED = set(selected.fid)
    defs = pd.read_csv(inputs[1]); assert len(defs) == 13 and defs.radius_px.eq(20).all()
    fields, local, image, scales = prepare(selected, defs)
    write('array_read_audit.csv', AUDIT)
    write('input_boundary.csv', [dict(n_ledger=632, n_nonblank=547, n_blank=69, prohibited_arrays_opened=0)])
    write('local_signal_fields.csv', local); write('image_indicator_fields.csv', image); write('noise_scale_fields.csv', scales)
    dates = sorted(nb.date.unique())
    bydate = {d:[f for f in fields if f['date'] == d and f['group'] != 'blank'] for d in dates}
    blanks = {d:[f for f in fields if f['date'] == d and f['group'] == 'blank'] for d in dates}
    minimum, null_scan = null_thresholds(bydate)
    e1 = defs[['x_camera', 'y_camera']].to_numpy()
    e2 = defs[~defs.defect_id.isin([8, 40, 157, 309])][['x_camera', 'y_camera']].to_numpy()
    e3 = {d:build(fs, minimum[d, len(fs), 'table']) for d, fs in bydate.items()}
    centers_rows, m1_rows, m3, m4, m5, sampling, e3h = [], [], [], [], [], [], []
    # Streaming prevents millions of calibration rows accumulating in memory.
    repeat_path = OUT/'M1_E5_all_k.csv'
    repeat_file = open(repeat_path, 'x', encoding='utf-8-sig', newline='')
    repeat_writer = None
    m4_repeat_file=open(OUT/'M4_E5_fields.csv','x',encoding='utf-8-sig',newline='')
    m4_repeat_writer=csv.DictWriter(m4_repeat_file,fieldnames=['n','repetition','fid','date','group','n_before','n_valid','lost_fraction'])
    m4_repeat_writer.writeheader()
    e5_summary = []
    prev, cumulative = np.empty((0, 2)), np.empty((0, 2))
    for di, (date, fs) in enumerate(bydate.items()):
        q = image[(image.date == date) & (image.group != 'blank') & (image.side == 'pre')]
        ii = q.groupby('defect_id')[['wid_difference', 'density_ratio']].mean()
        ids8 = ii.index[(ii.wid_difference > .02) & (ii.density_ratio < .9)]
        e8 = defs[defs.defect_id.isin(ids8)][['x_camera', 'y_camera']].to_numpy()
        arms = {'E0':np.empty((0, 2)), 'E1':e1, 'E2':e2, 'E3':e3[date], 'E3u':union(e2,e3[date]),
                'E8':e8}
        if di:
            arms.update(E4=prev, E4u=union(e2,prev), E7=cumulative)
        # No first-day past table: E4u, E4 and E7 share comparable dates.
        for arm, centers in arms.items():
            for cid, (x,y) in enumerate(centers):
                centers_rows.append(dict(arm=arm, date=date, center_index=cid, x=x, y=y, n_centers=len(centers), radius=20))
            for f in blanks[date]:
                m1_rows.extend(evaluate_blank(f, centers, arm))
            m3.extend(m3_rows(fs, centers, arm, 'inside' if arm in ('E3','E3u','E8') else 'outside_or_reference'))
            for f in fs:
                lost = len(field_exclusion_ids(f, centers))
                m4.append(dict(arm=arm, fid=f['fid'], date=date, group='nonblank', lost=lost, n_before=f['n_valid'], lost_fraction=lost/f['n_valid']))
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, di]))
        reproduction = np.zeros((200, len(e3[date])), bool)
        split_rates = []
        for rep in range(200):
            order = rng.permutation(len(fs)); cut = len(fs)//2
            fa, fb = [fs[i] for i in order[:cut]], [fs[i] for i in order[cut:]]
            ca = build(fa, minimum[date,len(fa),'table']); cb = build(fb,minimum[date,len(fb),'table'])
            reproduction[rep] = match(e3[date],ca) & match(e3[date],cb)
            matches = int(match(ca,cb).sum())
            rate = 2*matches/(len(ca)+len(cb)) if len(ca)+len(cb) else np.nan
            split_rates.append(rate)
            m5.append(dict(date=date, repetition=rep, center_index=-1, x=np.nan, y=np.nan, matched=matches,
                           n_a=len(ca), n_b=len(cb), split_match_rate=rate, full_center_rate=float(reproduction[rep].mean()) if len(e3[date]) else np.nan))
            for train, test, table_, side in [(fa,fb,ca,'a_to_b'),(fb,fa,cb,'b_to_a')]:
                e3h.extend(m3_rows(train,table_,'E3h','inside_'+side,rep))
                e3h.extend(m3_rows(test,table_,'E3h','outside_'+side,rep))
        for ci, (x,y) in enumerate(e3[date]):
            didx = int(np.argmin(np.linalg.norm(e1-[x,y],axis=1)))
            did = int(defs.iloc[didx].defect_id) if np.linalg.norm(e1[didx]-[x,y]) <= 5 else -1
            m5.append(dict(date=date, repetition=-1, center_index=ci, defect_id=did,x=x,y=y,
                           matched=int(reproduction[:,ci].sum()), n_a=200,n_b=200,
                           split_match_rate=float(reproduction[:,ci].mean()), full_center_rate=float(reproduction[:,ci].mean())))
        for n in NS:
            if n > len(fs):
                sampling.append(dict(date=date,n=n,repetition=-1,status='insufficient_fields',available=len(fs), recall=np.nan,false_centers=np.nan))
                continue
            for rep in range(200):
                order = rng.permutation(len(fs))
                train, test = [fs[i] for i in order[:n]], [fs[i] for i in order[n:]]
                ct = build(train,minimum[date,n,'E5'])
                recall = float(match(e3[date],ct).mean()) if len(e3[date]) else np.nan
                false = int((~match(ct,e3[date])).sum())
                sampling.append(dict(date=date,n=n,repetition=rep,status='computed',available=len(fs),recall=recall,false_centers=false,
                                     n_centers=len(ct),train_ids=json.dumps([f['fid'] for f in train]), minimum=minimum[date,n,'E5']))
                m3.extend(m3_rows(test,ct,'E5','outside',rep,n))
                for f in fs:
                    lost=len(field_exclusion_ids(f,ct))
                    m4_repeat_writer.writerow(dict(n=n,repetition=rep,fid=f['fid'],date=date,group='nonblank',n_before=f['n_valid'],n_valid=f['n_valid']-lost,lost_fraction=lost/f['n_valid']))
                erows = []
                for f in blanks[date]:
                    erows.extend(evaluate_blank(f,ct,'E5',rep,n))
                if repeat_writer is None:
                    repeat_writer = csv.DictWriter(repeat_file, fieldnames=list(erows[0])); repeat_writer.writeheader()
                repeat_writer.writerows(erows)
                for r in erows:
                    if r['k']==6 and r['normalization']=='fixed':
                        m4_repeat_writer.writerow({k:r[k] for k in ['n','repetition','fid','date','n_before','n_valid','lost_fraction']}|{'group':'blank'})
                e5_summary.extend(summaries(pd.DataFrame(erows)).query("date != 'all'").to_dict('records'))
            print(f'Codex: E5 {date}, n={n}, 200 repetitions complete',flush=True)
        prev = e3[date]; cumulative = union(cumulative,e3[date])
        # Null support trees are rebuilt one date at a time; valid arrays no longer needed.
        for f in fs:
            f.pop('valid_xy',None)
    repeat_file.close();m4_repeat_file.close()
    m1 = write('M1_all_k.csv',m1_rows)
    write('M1.csv',m1[m1.k.isin([6.,8.,10.])])
    summary = summaries(m1)
    es = pd.DataFrame(e5_summary)
    # Weighted mean across dates, paired repetition indices fixed across days.
    for key,g in es.groupby(['arm','repetition','n','normalization','k']):
        total = g.n_fields.sum()
        e5_summary.append(dict(zip(['arm','repetition','n','normalization','k'],key),date='all',n_fields=total,
                               mean_fp91k=float(np.average(g.mean_fp91k,weights=g.n_fields)),total_count=int(g.total_count.sum()),
                               max_lost_fraction=float(g.max_lost_fraction.max()),p95_fp91k=np.nan,mean_count=g.total_count.sum()/total))
    write('M1_summary.csv',summary)
    write('M1_E5_summary.csv',e5_summary)
    write('dated_centers.csv',centers_rows,columns=['arm','date','center_index','x','y','n_centers','radius'])
    m3 = write('M3_fields.csv',m3)
    write('M3_summary.csv',m3.groupby(['arm','evaluation','n','date']).agg(n_records=('fid','size'),n_abnormal=('n_abnormal','sum'),n_covered=('n_covered','sum')).assign(coverage=lambda t:t.n_covered/t.n_abnormal).reset_index())
    eh = write('M3_E3h_fields.csv',e3h)
    write('M3_E3h_summary.csv',eh.groupby(['evaluation','date']).agg(n_records=('fid','size'),n_abnormal=('n_abnormal','sum'),n_covered=('n_covered','sum')).assign(coverage=lambda t:t.n_covered/t.n_abnormal).reset_index())
    b4=m1[(m1.k==6)&(m1.normalization=='fixed')].drop_duplicates(['arm','fid'])
    m4.extend(dict(arm=r.arm,fid=r.fid,date=r.date,group='blank',lost=r.n_before-r.n_valid,n_before=r.n_before,lost_fraction=r.lost_fraction) for r in b4.itertuples())
    m4=write('M4_fields.csv',m4)
    write('M4_summary.csv',m4.groupby(['arm','group']).lost_fraction.agg(['mean','max','size']).reset_index())
    m5=write('M5.csv',m5)
    sample=write('E5_sampling.csv',sampling)
    write('E5_sample_size_summary.csv',sample.groupby(['n','date']).agg(mean_recall=('recall','mean'),mean_false_centers=('false_centers','mean'),n_repetitions=('recall','count')).reset_index())
    m2=[]
    for (arm,norm),g in summary[summary.date=='all'].groupby(['arm','normalization']):
        passing=g[g.p95_fp91k<=2]
        m2.append(dict(arm=arm,normalization=norm,k=float(passing.k.min()) if len(passing) else np.nan,reached=bool(len(passing)),grid='3..15 step .5',evaluation='inside_blank_calibration'))
    # E5 M2 requires the pooled 69 field quantile, not a mean of daily quantiles.
    repeat=pd.read_csv(repeat_path)
    e5_cal=[]
    for (n,rep,norm),g in repeat.groupby(['n','repetition','normalization']):
        p95=g.groupby('k').fp91k.quantile(.95); passing=p95[p95<=2]
        e5_cal.append(dict(arm='E5',n=n,repetition=rep,normalization=norm,k=float(passing.index.min()) if len(passing) else np.nan,reached=bool(len(passing)),n_fields=g.fid.nunique(),evaluation='inside_blank_calibration'))
    write('M2.csv',m2);write('M2_E5.csv',e5_cal)
    # E5 per-field M4 is directly recoverable for every draw from exact M1 denominators.
    e5m4=pd.read_csv(OUT/'M4_E5_fields.csv')
    write('M4_E5_summary.csv',e5m4.groupby(['n','repetition','group']).lost_fraction.agg(['mean','max','size']).reset_index())
    del e5m4
    # Complete global quantiles for E5 summaries, using actual field values.
    pooled=summaries(repeat)
    write('M1_E5_pooled_summary.csv',pooled)
    del repeat
    local_summ=[]; image_summ=[]
    rng=np.random.default_rng(np.random.SeedSequence([SEED,3]))
    for (group,did,date),g in local.groupby(['group','defect_id','date']):
        for metric in ['center_gt3','control_gt3','difference_gt3','center_gt4.5','control_gt4.5','difference_gt4.5','center_gt6','control_gt6','difference_gt6','excl_false_difference','z5p_nonfinite_difference']:
            avg,lo,hi,n=bootstrap(g[metric],rng)
            local_summ.append(dict(group=group,defect_id=did,date=date,metric=metric,mean=avg,ci95_low=lo,ci95_high=hi,n_fields=n))
    for (group,did,date,side),g in image.groupby(['group','defect_id','date','side']):
        for metric in ['wid_difference','amp_difference','density_ratio','n_center']:
            avg,lo,hi,n=bootstrap(g[metric],rng)
            image_summ.append(dict(group=group,defect_id=did,date=date,side=side,metric=metric,mean=avg,ci95_low=lo,ci95_high=hi,n_fields=n))
    write('local_signal_by_date.csv',local_summ);write('image_indicators_by_date.csv',image_summ)
    noise=scales.groupby(['group','date'])[['mad_scale','n_valid','sigma']].median().reset_index();write('noise_scale_by_date.csv',noise)
    centroid=[]
    for (group,did,date),g in local.groupby(['group','defect_id','date']):
        n=g.n_detected.sum()
        centroid.append(dict(group=group,defect_id=did,date=date,n_detected=n,
                             x=float((g.detected_x.fillna(0)*g.n_detected).sum()/n) if n else np.nan,
                             y=float((g.detected_y.fillna(0)*g.n_detected).sum()/n) if n else np.nan))
    write('detected_centroids_by_date.csv',centroid)
    classification=[]; boundaries=[]
    for did,boundary,absent_early in [(8,'260922',False),(157,'260924',True),(309,'260924',True)]:
        for group,g in local[local.defect_id==did].groupby('group'):
            early=g[g.date<=boundary];late=g[g.date>boundary]
            a0,a1=(early,late) if absent_early else (late,early)
            # Stratified within each date, pooled with original date weights.
            def stratified(q,metric):
                chunks=[v[metric].dropna().to_numpy(float) for _,v in q.groupby('date')]
                chunks=[v for v in chunks if len(v)]
                if not chunks:return np.nan,np.nan,np.nan,0
                sums=sum(v[rng.integers(len(v),size=(2000,len(v)))].sum(axis=1) for v in chunks)
                n=sum(map(len,chunks));return sum(v.sum() for v in chunks)/n,*np.quantile(sums/n,[.025,.975]),n
            v0,lo0,hi0,n0=stratified(a0,'difference_gt3');v1,lo1,hi1,n1=stratified(a1,'difference_gt3')
            ns=scales[(scales.group==group)]
            s0=ns[ns.date<=boundary];s1=ns[ns.date>boundary]
            ratios={metric:float(s1[metric].median()/s0[metric].median()) for metric in ['mad_scale','n_valid','sigma']}
            change=any(abs(r-1)>=.2 for r in ratios.values())
            cl='判定不能'
            if hi0<=.2*v1 and lo0<=0<=hi0 and v0<=.1*v1:cl='局所信号の増加/減少：低信号側がほぼ無い'
            if v0>=.5*v1 and lo0>0:cl='検出上の変化：低信号側にも存在'
            if change:cl='尺度の変化あり：分類保留'
            classification.append(dict(group=group,defect_id=did,boundary=boundary+'|'+dates[dates.index(boundary)+1],
                                       low_signal_side='early' if absent_early else 'late',a_low=v0,ci_low_lo=lo0,ci_low_hi=hi0,
                                       a_high=v1,ci_high_lo=lo1,ci_high_hi=hi1,n_low=n0,n_high=n1,classification=cl,
                                       scale_change=change,**{k+'_ratio_late_early':v for k,v in ratios.items()}))
            for side,q in [('early',early),('late',late)]:
                for k in ['3','4.5','6']:
                    avg,lo,hi,n=stratified(q,'difference_gt'+k)
                    boundaries.append(dict(group=group,defect_id=did,boundary=boundary,side=side,k=k,mean_difference=avg,ci95_low=lo,ci95_high=hi,n_fields=n))
    write('signal_classification.csv',classification);write('local_signal_boundaries.csv',boundaries)
    # Mechanical v3 application to both normalization variants; no criterion changes.
    pf=[]; worsened=[]
    def add(arm,norm,criterion,value,threshold,passed,detail=''):
        pf.append(dict(arm=arm,normalization=norm,criterion=criterion,value=value,threshold=threshold,passes=bool(passed),detail=detail))
    for norm in ('recomputed','fixed'):
        for arm in ('E3','E3u'):
            for k in (6.,8.):
                q=m1[(m1.normalization==norm)&(m1.k==k)]
                base=q[q.arm=='E2'].set_index('fid');one=q[q.arm=='E1'].set_index('fid');t=q[q.arm==arm].set_index('fid')
                reduction=1-t.fp91k.mean()/base.fp91k.mean();ratio=t.fp91k.mean()/one.fp91k.mean()
                add(arm,norm,f'reduction_vs_E2_k{k:g}',reduction,.1,reduction>=.1)
                add(arm,norm,f'noninferiority_vs_E1_k{k:g}',ratio,1.1,ratio<=1.1)
                ids=t.index[t.date.isin(['260926','260927'])]
                excess=int(t.loc[ids,'count'].sum()-base.loc[ids,'count'].sum())
                add(arm,norm,f'late_total_count_vs_E2_k{k:g}',excess,0,excess<=0)
                delta=t.fp91k-base.fp91k
                p=float(wilcoxon(delta,alternative='less',zero_method='wilcox').pvalue) if np.any(delta!=0) else 1.
                add(arm,norm,f'Wilcoxon_descriptive_k{k:g}',p,np.nan,True,'descriptive_only')
                for fid in t.index[t['count']>base['count']]:
                    worsened.append(dict(arm=arm,normalization=norm,k=k,fid=fid,date=t.loc[fid,'date'],count_arm=t.loc[fid,'count'],count_E2=base.loc[fid,'count']))
            loss=m4[m4.arm==arm].lost_fraction.max();limit=m4[m4.arm=='E1'].lost_fraction.max()+.005
            add(arm,norm,'max_lost_fraction_vs_E1',loss,limit,loss<=limit)
            relevant=[r['passes'] for r in pf if r['arm']==arm and r['normalization']==norm and not r['criterion'].startswith('Wilcoxon')]
            add(arm,norm,'overall_iv',int(all(relevant)),1,all(relevant))
        for arm in ('E4','E4u','E7'):
            q=m1[(m1.normalization==norm)&(m1.k==6)]
            t=q[q.arm==arm].set_index('fid');same=q[q.arm=='E3'].set_index('fid').loc[t.index]
            diff=abs(t.fp91k.mean()/same.fp91k.mean()-1)
            add(arm,norm,'carryover_within_10percent_of_E3',diff,.1,diff<=.1,'same dates; first date excluded; one observed change')
    splits=m5[(m5.repetition>=0)&(m5.center_index==-1)]
    rr=splits.split_match_rate.mean();add('E3','not_applicable','M5_split_match_mean',rr,.8,rr>=.8)
    required=[]
    for n,g in sample[sample.status=='computed'].groupby('n'):
        recall=g.recall.mean();false=g.false_centers.mean();complete=g.date.nunique()==9
        add('E5','not_applicable',f'reference_n{n}',recall,.9,complete and recall>=.9 and false<=.5,f'mean_false_centers={false}; dates={g.date.nunique()}/9; descriptive')
        if complete and recall>=.9 and false<=.5:required.append(n)
    passfail=write('pass_fail.csv',pf)
    write('worsened_blank_counts.csv',worsened,columns=['arm','normalization','k','fid','date','count_arm','count_E2'])
    # Meaningful regression: v88 E1 exact counts and denominators at every registered k.
    old=pd.read_csv(inputs[5],dtype={'date':str})
    old=old[(old.radius==20)&(old.side=='two_sided')]
    new=m1[(m1.arm=='E1')&(m1.normalization=='recomputed')]
    compare=new.merge(old,on=['fid','k'],suffixes=('_new','_old'))
    assert len(compare)==69*len(GRID)
    assert np.array_equal(compare.count_new,compare.count_old)
    assert np.array_equal(compare.n_valid_new,compare.n_valid_old)
    assert np.allclose(compare.fp91k_new,compare.fp91k_old)
    assert len(AUDIT)==616*3 and all(r['fid'] in ALLOWED for r in AUDIT)
    after={str(p):sha(p) for p in inputs};assert hashes==after
    write('input_integrity.csv',[dict(path=p,sha256_before=h,sha256_after=after[p],unchanged=True) for p,h in hashes.items()])
    verification=dict(evaluation=LABEL,preregistration='v3',nonblank_fields=547,blank_fields=69,null_iterations=1000,
                      E5_repetitions=200,split_repetitions=200,seed=SEED,prohibited_arrays_opened=0,
                      array_reads=len(AUDIT),v88_E1_exact_count_match=True,input_integrity=True,
                      first_day_excluded=['E4','E4u','E7'],unavailable_E5=[dict(date=d,n=n,available=len(fs)) for d,fs in bydate.items() for n in NS if n>len(fs)],
                      minimum_reference_n=min(required) if required else None, elapsed_seconds=time.perf_counter()-start)
    (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['---','確認: 未確認','状態: 未読','---','# 【未読】日付別欠陥表の開発用再評価','',
            'Codexは事前登録第3版に従い、547非ブランク視野と69ブランク視野を再評価した。Codexは本結果を事後の再評価（独立な検証ではない）として扱う。',
            'Codexは未確認の事前登録・調査ノート・第88版の欠陥候補表を入力に使った。Codexは候補の承認状態と標準経路を変更しなかった。','',
            '|条件|標準化|判定項目|値|基準|合否|','|---|---|---|---:|---:|---|']
    for r in passfail.itertuples():
        if r.criterion.startswith('Wilcoxon'):continue
        report.append(f'|{r.arm}|{r.normalization}|{r.criterion}|{r.value:.6g}|{r.threshold:.6g}|'+('合格' if r.passes else '不合格')+'|')
    report+=['','CodexはE2の固定9中心とE1を、全期間の情報を使った後知恵の参照として扱う。CodexはE7を直前までの撮影日の和集合とし、当日の表を含めない。',
             'CodexはE4・E4u・E7から最初の撮影日を除き、繰り越し比較の分母を同じ日付にそろえた。Codexは繰り越しの結論が260926の1回の変化に強く依存する点を残した。',
             f'Codexは同日分割の表同士の一致率を{rr:.6g}と記録した。Codexは中心ごとの両側再現率をM5.csvに分けた。',
             f'Codexは全9日で適用可能な必要参照視野数を{min(required) if required else "基準を満たす数なし"}と記録した。',
             'Codexは260827の47視野に対する48視野抽出を実行不能と記録し、置換抽出へ変更しなかった。Codexは板番号を文字列で保持し、260922を54視野として選別した。','',
             '## 中心の変化と尺度','']
    for r in classification:
        report.append(f'Codexは中心{r["defect_id"]}の{r["group"]}層を「{r["classification"]}」とした。Codexは低信号側の差を{r["a_low"]:.6g}、高信号側の差を{r["a_high"]:.6g}と記録した。')
    report+=['Codexは境を中心8で260922と260923の間、中心157・309で260924と260926の間に限定した。Codexはカメラ・光学・汚れ・試料などの物理的原因を断定していない。','',
             '## 実装の細部（仮置き）','',
             'Codexは2種類の帰無を日付ごと・視野数ごとに1000回計算し、必要最小視野数の大きい方を採った。Codexは同じ反復番号の9日分を合わせ、1つでも偽中心が出る事象を1000反復で数えた。Codexはその事象の片側95%上限が0.05以下になる最小の視野数を全日表・半分の表について採った。Codexは採用値と走査結果をm_of_N.csvとnull_nine_table_scan.csvに保存した。',
             'Codexは集合を保つ帰無で、視野全体の点集合に同じ平行移動を与え、座標範囲内に収め、最寄り有効ピラーへ丸めた。Codexは丸め距離が局所ピッチの半対角を超える点を有効域外として除いた。Codexは除いた数をnull_translation_diagnostics.csvに記録した。',
             'Codexは表の和集合で重なる中心を平均せず、円の和集合を維持した。Codexは再現率で5画素以内の一対一最大対応を使った。',
             'Codexは環状対照で半径45画素上の16候補円を試し、円全体が30〜60画素の環内に収まる同数ピラーの円を採った。Codexは当てはめ密度を円面積当たりで計算した。Codexは対照が作れない視野を欠測とし、視野数を併記した。',
             'CodexはE8を非ブランクの滴下前画像の13候補中心の日付別平均から作った。Codexは幅の差>0.02かつ密度比<0.9を仮置き閾値として使い、合否に使わなかった。',
             'Codexは局所の主指標を除外なしのS5中央値・中央絶対偏差で標準化した全有限値ピラーから計算した。Codexは日付内2000回の視野単位再抽出を使い、境の比較でも日付ごとの視野数を保った。',
             'Codexは指標1と指標4の除外を第88版と同じ前後座標の円の和集合とし、有限S5かつexclのピラーを有効数の分母にした。Codexは指標3を変換後の検出座標の円で評価した。','',
             '## 保存先と検証','',
             'Codexはdata/results/v90_dated_defect_table/に全表、NOTES.md、verification.json、v90_manifest.csvを保存した。Codexは第88版の13中心・再計算標準化の全69視野×25閾値で個数・有効数・換算値の一致を検証した。',
             'Codexは1848件の配列読込をarray_read_audit.csvに記録した。Codexは禁止日付の配列と画像を開かず、入力ファイルの実行前後の内容の不変を検証した。','',
             '## 基準への異議','',
             'Codexは点集合の平行移動で有効域への丸めと点の除去が集合の形を変える限界を記録した。Codexは全日表の9日同時事象に対する最小の共通視野数閾値を使い、日付別の偽中心数も報告した。',
             'CodexはE5の48視野が260827で実行不能である点を記録し、基準と視野数を変えなかった。Codexは両分割表が空のときの一致率を未定義とし、空表の一致を再現成功に数えなかった。',
             'Codexは局所信号の分類が尺度変化、欠測、日付と試料の交絡を解消しない点を残した。CodexはE8が既存13候補中心に限った探索であり、未知の画像欠陥を探索する表ではない点を記録した。']
    (OUT/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (OUT/'NOTES.md').write_text('# 第90版の結果記録\n\nCodexは第88版を変更せず、日付別表、2帰無、2標準化、全比較条件と指標を追加した。\nCodexは結果をdata/results/v90_dated_defect_table/に保存した。Codexは合否をpass_fail.csvとreport.mdに記録した。\nCodexは実行不能な48視野抽出、後知恵の参照、平行移動帰無の丸め、同日内依存を既知の問題として残した。\nCodexは最新の書込範囲指定に従い、共有ノートとモデル選択の記録を変更しなかった。\n',encoding='utf-8')
    extra_checks()
    elapsed=time.perf_counter()-start
    manifest=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file():manifest.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,seed=SEED,command='python -B field_level/v90_dated_defect_table/field_dated.py',utc_start=utc,elapsed_seconds=elapsed))
    for p in (Path(__file__),Path(__file__).with_name('NOTES.md')):
        manifest.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,seed=SEED,command='python -B field_level/v90_dated_defect_table/field_dated.py',utc_start=utc,elapsed_seconds=elapsed))
    write('v90_manifest.csv',manifest)
    print(json.dumps(verification,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
