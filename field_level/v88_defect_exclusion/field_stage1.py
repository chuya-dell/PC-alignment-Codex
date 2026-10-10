"""Codexが開発用だけを計算する：事後の再評価（独立な検証ではない）。"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(r'C:\Users\chuya\PC-alignment-fp\data\results')
OUT = ROOT / 'data/results/v88_defect_exclusion'
LABEL = '事後の再評価（独立な検証ではない）'
GRID = np.arange(3., 15.01, .5)
READ_LOG = []

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def table(name, rows):
    t = pd.DataFrame(rows)
    t.insert(0, 'evaluation', LABEL)
    t.to_csv(OUT / name, index=False, encoding='utf-8-sig')
    return t

def load_dev(folder, fid):
    assert fid in ALLOWED and not fid.startswith('261008')
    p = SRC / folder / f'{fid}.npz'
    READ_LOG.append(dict(path=str(p), fid=fid, purpose='development_only'))
    return np.load(p, allow_pickle=False)

def clusters(points):
    if not len(points):
        return pd.DataFrame(columns=['x', 'y', 'n', 'nf'])
    x = points[['x', 'y']].to_numpy()
    pairs = cKDTree(x).query_pairs(10., output_type='ndarray')
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
    _, c = connected_components(g, directed=False)
    sizes = np.bincount(c)
    q = points.assign(component=c)
    q = q[sizes[c] >= 8]
    q = q.groupby('component').agg(x=('x', 'mean'), y=('y', 'mean'), n=('x', 'size'), nf=('fid', 'nunique')).reset_index()
    return q[q.nf >= 10].reset_index(drop=True)

def main():
    global ALLOWED
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'stage1_summary.json').exists(), 'Codexは既存結果を上書きしない。'
    meta = pd.read_csv(SRC / 'v85_abnormal_pillars/E4_counts_all632.csv', dtype={'date': str, 'board': str})
    assert len(meta) == 632 and meta.fid.is_unique
    assert not meta.date.eq('261008').any()
    ALLOWED = set(meta.fid)
    # Codexは新撮影の識別子だけを既存集計表から読む。Codexは配列を読まない。
    hist = {}
    for name in ['E3_new_hotspot_effect.csv', 'E3_new_hotspot_post_effect.csv', 'E3_repro_new.csv']:
        p = SRC / 'v85_abnormal_pillars' / name
        hist[name] = pd.read_csv(p) if p.exists() else None
    h = hist['E3_repro_new.csv']
    forbidden = {f'261008_{int(r.board)}_{int(r.pos)}' for r in h.itertuples()} if h is not None else set()
    assert ALLOWED.isdisjoint(forbidden)
    bm = meta.groupby(['date', 'board']).abn6_out.mean().rename('bmean').reset_index()
    meta = meta.merge(bm, on=['date', 'board'])
    nb = meta[(meta.group != 'blank') & (meta.bmean < 50)]
    blanks = meta[meta.group == 'blank']
    assert len(blanks) == 69
    dates = sorted(nb.date.unique())
    groups = {'early': dates[:(len(dates)+1)//2], 'late': dates[(len(dates)+1)//2:]}
    table('date_split.csv', [dict(group=g, date=d, n_fields=int(nb.date.eq(d).sum())) for g, ds in groups.items() for d in ds])
    table('input_boundary.csv', [dict(n_ledger=632, n_nonblank=len(nb), n_blank=69, n_new_ids=len(forbidden), overlap=len(ALLOWED & forbidden), arrays_261008_opened=0)])
    defs = pd.read_csv(ROOT / 'data/raw/v88_camera_defects_unapproved.csv')
    assert len(defs) == 13 and defs.radius_px.eq(20).all()
    centers = defs[['x_camera', 'y_camera']].to_numpy()
    points = []
    for i, r in enumerate(nb.itertuples()):
        with load_dev('v84_1fM_strategies/cache', r.fid) as z, load_dev('v70_ledger_center_fit/fields2', r.fid) as c:
            xy = z['ctr'].astype(float) @ c['Llin'].T + c['tL']
            keep = z['abn'] & np.isfinite(z['z5p']) & z['excl'] & (np.abs(z['z5p']) > 6)
            points.extend((r.fid, r.date, x, y) for x, y in xy[keep])
        if i % 100 == 0: print('Codex development centers', i, '/', len(nb), flush=True)
    pts = pd.DataFrame(points, columns=['fid', 'date', 'x', 'y'])
    rebuilt = {}
    for g, ds in groups.items():
        rebuilt[g] = clusters(pts[pts.date.isin(ds)])
        table(f'centers_{g}.csv', rebuilt[g])
    full = clusters(pts)
    table('centers_full.csv', full)
    repro = []
    for r, xy in zip(defs.itertuples(), centers):
        row = dict(defect_id=r.defect_id, x=xy[0], y=xy[1])
        for g, c in rebuilt.items():
            dist = float(cKDTree(c[['x','y']]).query(xy)[0]) if len(c) else float('inf')
            row[f'{g}_distance_px'] = dist
            row[f'{g}_reproduced'] = dist <= 5
        row['both_reproduced'] = row['early_reproduced'] and row['late_reproduced']
        repro.append(row)
    rp = table('center_reproducibility.csv', repro)
    full_dist = cKDTree(full[['x','y']]).query(centers)[0] if len(full) else np.full(13, np.inf)
    tree = cKDTree(centers)
    count_rows = []
    for i, r in enumerate(blanks.itertuples()):
        with load_dev('v72_real_field_readout/fields', r.fid) as a, load_dev('v84_1fM_strategies/cache', r.fid) as z, load_dev('v70_ledger_center_fit/fields2', r.fid) as c:
            delta = a['S5'].astype(float)
            ctr = z['ctr'].astype(float)
            assert np.allclose(ctr, a['ctr_xy'], equal_nan=True)
            valid = np.isfinite(delta) & z['excl']
            post = c['post_ctr'][c['jL']].astype(float)
            fallback = ~np.isfinite(delta) | ~np.isfinite(post).all(axis=1)
            post[fallback] = (ctr @ c['Llin'].T + c['tL'])[fallback]
            distances = np.minimum(tree.query(ctr)[0], tree.query(post)[0])
            for radius in [0, 10, 15, 20, 25, 30]:
                mask = valid & (distances >= radius)
                v = delta[mask]
                assert len(v) >= 5000
                med = np.median(v); scale = 1.4826 * np.median(np.abs(v-med))
                assert np.isfinite(scale) and scale > 0
                zz = (v-med)/scale
                for side, vals in [('positive', zz), ('negative', -zz), ('two_sided', np.abs(zz))]:
                    for k in GRID:
                        n = int((vals > k).sum())
                        count_rows.append(dict(fid=r.fid, date=r.date, radius=radius, side=side, k=k, count=n, fp91k=n/len(v)*91000, n_valid=len(v), n_before=int(valid.sum()), median=med, scale=scale))
        if i % 10 == 0: print('Codex development blanks', i, '/ 69', flush=True)
    counts = table('blank_counts_by_k.csv', count_rows)
    cal = []
    for (radius, side), g in counts.groupby(['radius', 'side']):
        candidates = g.groupby('k').fp91k.quantile(.95)
        passing = candidates[candidates <= 2]
        reached = len(passing) > 0
        k = float(passing.index[0]) if reached else 15.
        q = g[g.k == k]
        cal.append(dict(radius=int(radius), side=side, k=k, reached=reached, n_fields=len(q), p95_fp91k=float(q.fp91k.quantile(.95)), p95_count=float(q['count'].quantile(.95)), mean_count=float(q['count'].mean()), max_count=int(q['count'].max()), max_lost_fraction=float((1-q.n_valid/q.n_before).max())))
    ct = table('calibration_k.csv', cal)
    sensitivity = []
    for side in ['positive','negative','two_sided']:
        base = ct[(ct.radius == 0) & (ct.side == side)].iloc[0]
        for radius in [10,15,20,25,30]:
            q = counts[(counts.radius == radius) & (counts.side == side) & (counts.k == base.k)]
            p95 = float(q.fp91k.quantile(.95)); actual = float(q['count'].quantile(.95))
            sensitivity.append(dict(radius=radius, side=side, common_k=base.k, baseline_reached=bool(base.reached), baseline_p95_fp91k=base.p95_fp91k, p95_fp91k=p95, baseline_p95_count=base.p95_count, p95_count=actual, worsened=(p95 > base.p95_fp91k or actual > base.p95_count)))
    st = table('radius_sensitivity.csv', sensitivity)
    frozen = {'evaluation':LABEL, 'radius':20, 'calibration':ct[ct.radius.isin([0,20])].to_dict('records'), 'positive':'後ろ像の減光', 'negative':'後ろ像の増光'}
    (OUT/'frozen_k.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    historical = []
    for name, t in hist.items():
        if t is None:
            historical.append(dict(source=name, status='見つからなかった')); continue
        design = '前像・半径40' if name == 'E3_new_hotspot_effect.csv' else '後ろ像・半径20'
        for kind, g in t.groupby('kind') if 'kind' in t else [('existing_summary',t)]:
            for col in g.select_dtypes(include='number').columns:
                if col in ['board','pos']: continue
                historical.append(dict(source=name, design=design, kind=kind, metric=col, n_fields=len(g), mean=float(g[col].mean()), total=float(g[col].sum()), p95=float(g[col].quantile(.95)), recomputed_from_arrays=False))
        table('history_'+name, t)
    table('historical_design_summary.csv', historical)
    table('array_read_audit.csv', READ_LOG)
    summary = dict(evaluation=LABEL, split_dates=groups, nonblank_fields=len(nb), blank_fields=69, reconstructed_centers={g:len(c) for g,c in rebuilt.items()}, full_centers=len(full), full_max_distance=float(np.max(full_dist)), reproduced_both=int(rp.both_reproduced.sum()), reproducibility_pass=bool(rp.both_reproduced.sum()>=10), radius_failures=st[st.worsened][['radius','side']].to_dict('records'), prohibited_arrays_opened=0)
    (OUT/'stage1_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    main()
