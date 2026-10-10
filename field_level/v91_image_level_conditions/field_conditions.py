"""Codexが第91版の探索的画像条件調査を再現する。"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(r'C:\Users\chuya\PC-alignment-fp\data\results')
VAULT = Path(r'W:\GoogleDrive\chuya2816\Obsidian Vault')
OUT = ROOT / 'data/results/v91_image_level_conditions'
SEED = 26101091
STABLE = {0, 1, 2, 3, 4, 5, 6, 7, 42}
AUDIT = []
ALLOWED = set()
LABEL = '探索的・事後の再評価（独立な検証ではない）'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def table(name, rows):
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.insert(0, 'evaluation', LABEL)
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig', mode='x')
    return df


def load(folder, fid):
    assert fid in ALLOWED
    assert not fid.startswith(('261008', '260925'))
    assert folder in ('v70_ledger_center_fit/fields2', 'v84_1fM_strategies/cache')
    path = SRC / folder / (fid + '.npz')
    AUDIT.append(dict(fid=fid, path=str(path), purpose='547_development_nonblank_only',
                      sha256=sha(path)))
    return np.load(path, allow_pickle=False)


def average(x):
    x = np.asarray(x)
    return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan


def median(x):
    x = np.asarray(x)
    return float(np.median(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan


def area(xy, radius):
    """Circle intersected with the 2048 by 2044 pixel image footprint."""
    x, y = xy
    left, right = max(-.5, x-radius), min(2047.5, x+radius)
    if left >= right:
        return 0.
    if x-radius >= -.5 and x+radius <= 2047.5 and y-radius >= -.5 and y+radius <= 2043.5:
        return float(np.pi * radius**2)
    def height(t):
        half = np.sqrt(max(0., radius**2-(t-x)**2))
        return max(0., min(2043.5, y+half)-max(-.5, y-half))
    return float(quad(height, left, right, epsabs=1e-6)[0])


def controls(tree, center, n):
    """Deterministic 24 angular locations at radius 45; adapt circle radius."""
    result = []
    if not n:
        return result
    for angle_id, angle in enumerate(np.arange(24) * 2*np.pi/24):
        q = center + 45*np.array([np.cos(angle), np.sin(angle)])
        if not (-.5 <= q[0] <= 2047.5 and -.5 <= q[1] <= 2043.5):
            continue
        dist, idx = tree.query(q, k=n+1)
        dist, idx = np.atleast_1d(dist), np.atleast_1d(idx)
        if not np.isfinite(dist).all() or dist[n-1] >= dist[n]:
            continue
        radius = float((dist[n-1]+dist[n])/2)
        if radius > 15:  # Entire circle is inside the 30--60 annulus.
            continue
        ids = np.asarray(tree.query_ball_point(q, radius), int)
        assert len(ids) == n
        assert np.all(np.linalg.norm(tree.data[ids]-center, axis=1) >= 30-1e-9)
        assert np.all(np.linalg.norm(tree.data[ids]-center, axis=1) <= 60+1e-9)
        result.append((angle_id, q, radius, ids, area(q, radius)))
    return result


def summarize(df, metrics, group_cols, rng, repetitions):
    rows = []
    for key, g in df.groupby(group_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        x = g[metrics].to_numpy(float)
        finite = np.isfinite(x)
        counts = finite.sum(axis=0)
        sums = np.where(finite, x, 0.)
        mean = np.divide(sums.sum(axis=0), counts, out=np.full(len(metrics), np.nan), where=counts>0)
        reps = np.zeros((repetitions, len(metrics)))
        denominators = np.zeros_like(reps)
        # Each draw resamples entire fields within each date, preserving phase/metric pairing.
        date_values = g.date.to_numpy()
        for date in sorted(set(date_values)):
            ids = np.flatnonzero(date_values == date)
            sampled = ids[rng.integers(0, len(ids), size=(repetitions, len(ids)))]
            reps += sums[sampled].sum(axis=1)
            denominators += finite[sampled].sum(axis=1)
        reps = np.divide(reps, denominators, out=np.full_like(reps, np.nan), where=denominators>0)
        for j, metric in enumerate(metrics):
            vals = reps[:, j]; vals = vals[np.isfinite(vals)]
            lo, hi = np.quantile(vals, [.025, .975]) if len(vals) else (np.nan, np.nan)
            rows.append(dict(zip(group_cols, key), metric=metric, n_fields=len(g),
                             n_finite_fields=int(counts[j]), mean=mean[j], median=median(x[:, j]),
                             ci95_low=lo, ci95_high=hi, bootstrap_repetitions=repetitions))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    assert args.bootstrap >= 1000
    start = time.perf_counter(); utc = datetime.now(timezone.utc).isoformat()
    assert not OUT.exists(), 'Codexは既存結果を上書きしない。'
    ledger_path = SRC / 'v85_abnormal_pillars/E4_counts_all632.csv'
    defs_path = ROOT / 'data/raw/v88_camera_defects_unapproved.csv'
    note = VAULT / 'ラボノート/05_解析/261008_【未読】偽陽性ゼロ化_原因確定と標準経路/【未読】9月22〜26日前後の変化_操作履歴と撮影・解析条件の切り分け.md'
    prereg = VAULT / '_依頼記録/偽陽性ゼロ化_C1D/261010_日付別欠陥表(iv)_設計と合否基準_事前登録.md'
    inputs = [ledger_path, defs_path, note, prereg,
              ROOT/'field_level/v88_defect_exclusion/field_stage1.py',
              ROOT/'field_level/v89_defect_cause/field_cause.py',
              ROOT/'data/results/v89_defect_cause/field_table.csv']
    initial = {str(p): sha(p) for p in inputs}
    meta = pd.read_csv(ledger_path, dtype={'fid':str, 'date':str, 'board':str})
    assert len(meta) == 632 and meta.fid.is_unique
    assert not meta.fid.str.startswith(('261008', '260925')).any()
    bm = meta.groupby(['date', 'board']).abn6_out.transform('mean')
    nb = meta[(meta.group != 'blank') & (bm < 50)].copy()
    assert len(nb) == 547 and nb.fid.is_unique and meta.group.eq('blank').sum() == 69
    global ALLOWED
    ALLOWED = set(nb.fid)
    defs = pd.read_csv(defs_path)
    assert len(defs) == 13 and defs.defect_id.is_unique
    assert set(defs.defect_id)-STABLE == {8,40,157,309}
    OUT.mkdir(parents=True)
    global_rows, local_rows, control_rows, centroids, transforms, detection_rows = [], [], [], [], [], []
    for fi, r in enumerate(nb.itertuples()):
        with load('v70_ledger_center_fit/fields2', r.fid) as c, load('v84_1fM_strategies/cache', r.fid) as z:
            xy_z = z['ctr'].astype(float) @ c['Llin'].T + c['tL']
            assert np.allclose(z['ctr'], c['pre_ctr'], equal_nan=True)
            excl = z['excl'].astype(bool); finite_z = np.isfinite(z['z5p'])
            detected = z['abn'].astype(bool) & excl & finite_z & (np.abs(z['z5p']) > 6)
            transforms.append(dict(fid=r.fid, date=r.date, fixed29=r.fixed29,
                                   **{f'Llin_{i}{j}':float(c['Llin'][i,j]) for i in range(2) for j in range(2)},
                                   tx=float(c['tL'][0]), ty=float(c['tL'][1])))
            grow = dict(fid=r.fid, date=r.date, board=r.board, group=r.group,
                        fixed29=r.fixed29, sigma=r.sigma, n_valid=r.n_valid)
            for phase in ('pre', 'post'):
                xy = c[phase+'_ctr'].astype(float)
                finite_xy = np.isfinite(xy).all(axis=1)
                original_ids = np.flatnonzero(finite_xy)
                tree = cKDTree(xy[finite_xy])
                vals = {metric:c[phase+'_'+metric].astype(float) for metric in ('amp','wid')}
                status_xy = z['ctr'].astype(float) if phase=='pre' else xy_z
                assert np.isfinite(status_xy).all()
                status_tree=cKDTree(status_xy)
                for metric, v in vals.items():
                    grow[phase+'_'+metric+'_median'] = median(v)
                    grow[phase+'_'+metric+'_mean'] = average(v)
                grow[phase+'_fitted_count'] = int(finite_xy.sum())
                if phase == 'post':
                    assert c['jL'].min() >= 0 and c['jL'].max() < len(xy)
                for d in defs.itertuples():
                    center = np.array([d.x_camera,d.y_camera])
                    ti = np.array(tree.query_ball_point(center,10.), int)
                    ids = original_ids[ti]
                    all60 = np.array(tree.query_ball_point(center,60.), int)
                    ann_ti = all60[np.linalg.norm(tree.data[all60]-center,axis=1) >= 30]
                    ann_ids = original_ids[ann_ti]
                    a10 = area(center,10.); aring = area(center,60.)-area(center,30.)
                    matches = controls(tree,center,len(ids))
                    row = dict(fid=r.fid,date=r.date,board=r.board,fixed29=r.fixed29,
                               defect_id=int(d.defect_id),stable_control=d.defect_id in STABLE,phase=phase,
                               n_center=len(ids),n_annulus=len(ann_ids),n_control_circles=len(matches),
                               area_center=a10,area_annulus=aring,
                               center_density=len(ids)/a10,annulus_density=len(ann_ids)/aring)
                    row['legacy_density_ratio'] = row['center_density']/row['annulus_density'] if len(ann_ids) else np.nan
                    densities=[]; control_vals={m:[] for m in vals}
                    for aid,q,radius,tids,aa in matches:
                        cids=original_ids[tids]
                        densities.append(len(cids)/aa)
                        cr=dict(fid=r.fid,date=r.date,defect_id=int(d.defect_id),phase=phase,
                                angle_id=aid,x=q[0],y=q[1],radius=radius,n_pillars=len(cids),area=aa,
                                density=len(cids)/aa)
                        for m,v in vals.items():
                            cr[m+'_mean']=average(v[cids]); control_vals[m].append(cr[m+'_mean'])
                        control_rows.append(cr)
                    row['control_density']=average(densities)
                    row['density_diff']=row['center_density']-row['control_density']
                    row['density_ratio']=row['center_density']/row['control_density'] if row['control_density']>0 else np.nan
                    for m,v in vals.items():
                        row[m+'_center']=average(v[ids]); row[m+'_control']=average(control_vals[m])
                        row[m+'_diff']=row[m+'_center']-row[m+'_control']
                        row['legacy_'+m+'_diff']=average(v[ids])-average(v[ann_ids])
                        row['legacy_'+m+'_median_diff']=median(v[ids])-median(v[ann_ids])
                        row[m+'_center_nonfinite']=int((~np.isfinite(v[ids])).sum())
                    # Failure counts are available on the pre-indexed v84 lattice only.
                    # Pre: native coordinates; post: transformed pre coordinates, matching v89.
                    sids=np.asarray(status_tree.query_ball_point(center,10.),int)
                    row['status_center_pillars']=len(sids)
                    row['status_excl_false']=int((~excl[sids]).sum())
                    row['status_z5p_nonfinite']=int((~finite_z[sids]).sum())
                    fail_excl=[]; fail_z=[]
                    for _,q,radius,_,_ in matches:
                        sctrl=np.asarray(status_tree.query_ball_point(q,radius),int)
                        fail_excl.append(int((~excl[sctrl]).sum())); fail_z.append(int((~finite_z[sctrl]).sum()))
                    row['status_excl_false_diff']=row['status_excl_false']-average(fail_excl)
                    row['status_z5p_nonfinite_diff']=row['status_z5p_nonfinite']-average(fail_z)
                    local_rows.append(row)
                    if phase=='post':
                        for radius in (10,20):
                            near=np.linalg.norm(xy_z-center,axis=1)<=radius
                            points=xy_z[near & detected]
                            centroids.extend(dict(fid=r.fid,date=r.date,defect_id=int(d.defect_id),
                                                  radius=radius,x=x,y=y) for x,y in points)
                            valid=near & excl & finite_z
                            detection_rows.append(dict(fid=r.fid,date=r.date,defect_id=int(d.defect_id),
                                                       radius=radius,fixed29=r.fixed29,n_opportunity_pillars=int(valid.sum()),
                                                       opportunity=bool(valid.any()),detected=bool((near & detected).any()),
                                                       n_detection_points=len(points)))
            grow['post_pre_amp_ratio']=grow['post_amp_median']/grow['pre_amp_median'] if grow['pre_amp_median'] else np.nan
            global_rows.append(grow)
        if fi%50==0:
            print(f'Codex: {fi}/547 fields',flush=True)
    global_df=pd.DataFrame(global_rows); local=pd.DataFrame(local_rows); ctr=pd.DataFrame(control_rows)
    assert len(local)==547*13*2 and not local.duplicated(['fid','defect_id','phase']).any()
    assert ctr.n_pillars.eq(ctr.merge(local[['fid','defect_id','phase','n_center']],on=['fid','defect_id','phase']).n_center).all()
    table('global_fields.csv',global_df); table('local_fields.csv',local); table('control_circles.csv',ctr)
    table('coordinate_transforms.csv',transforms)
    rng=np.random.default_rng(SEED)
    metrics=['amp_diff','wid_diff','density_diff','density_ratio','legacy_amp_diff','legacy_wid_diff',
             'legacy_density_ratio','legacy_amp_median_diff','legacy_wid_median_diff',
             'status_excl_false_diff','status_z5p_nonfinite_diff','center_density',
             'amp_center','wid_center','n_center','n_control_circles']
    daily=summarize(local,metrics,['date','defect_id','phase'],rng,args.bootstrap)
    table('daily_local_conditions.csv',daily)
    global_metrics=['pre_amp_median','post_amp_median','pre_wid_median','post_wid_median','sigma','post_pre_amp_ratio']
    gd=summarize(global_df,global_metrics,['date'],rng,args.bootstrap)
    table('daily_global_conditions.csv',gd)
    bounds=[]; changes=[]
    for before,after in [('260922','260923'),('260924','260926')]:
        b=before+'|'+after
        q=local.copy(); q['side']=np.where(q.date<=before,'before','after'); q['boundary']=b
        bs=summarize(q,metrics,['boundary','side','defect_id','phase'],rng,args.bootstrap)
        bounds.append(bs)
        for (did,phase),g in q.groupby(['defect_id','phase']):
            a=g[g.side=='before']; z=g[g.side=='after']
            x=a[metrics].to_numpy(float); y=z[metrics].to_numpy(float)
            reps=[]
            for part in [a,z]:
                p=part[metrics].to_numpy(float); ff=np.isfinite(p); ss=np.where(ff,p,0)
                sums=np.zeros((args.bootstrap,len(metrics))); ns=np.zeros_like(sums)
                for date,inds in part.groupby('date').indices.items():
                    inds=np.asarray(inds); sample=inds[rng.integers(0,len(inds),(args.bootstrap,len(inds)))]
                    sums+=ss[sample].sum(axis=1); ns+=ff[sample].sum(axis=1)
                reps.append(np.divide(sums,ns,out=np.full_like(sums,np.nan),where=ns>0))
            diff=reps[1]-reps[0]
            for j,m in enumerate(metrics):
                v=diff[:,j]; v=v[np.isfinite(v)]
                lo,hi=np.quantile(v,[.025,.975]) if len(v) else (np.nan,np.nan)
                changes.append(dict(boundary=b,defect_id=did,stable_control=did in STABLE,phase=phase,metric=m,
                                    before_mean=average(x[:,j]),after_mean=average(y[:,j]),
                                    change=average(y[:,j])-average(x[:,j]),ci95_low=lo,ci95_high=hi,
                                    before_fields=len(a),after_fields=len(z),bootstrap_repetitions=args.bootstrap))
    boundary_df=pd.concat(bounds,ignore_index=True)
    table('boundary_conditions.csv',boundary_df)
    changes=pd.DataFrame(changes); table('boundary_changes.csv',changes)
    table('stable9_boundary_changes.csv',changes[changes.stable_control])
    points=pd.DataFrame(centroids,columns=['fid','date','defect_id','radius','x','y'])
    table('detection_points.csv',points)
    centroid=points.groupby(['date','defect_id','radius']).agg(x=('x','mean'),y=('y','mean'),
                        n_points=('x','size'),n_fields=('fid','nunique'),x_std=('x','std'),y_std=('y','std')).reset_index()
    allkeys=pd.MultiIndex.from_product([sorted(nb.date.unique()),defs.defect_id,[10,20]],names=['date','defect_id','radius']).to_frame(index=False)
    centroid=allkeys.merge(centroid,how='left'); centroid[['n_points','n_fields']]=centroid[['n_points','n_fields']].fillna(0).astype(int)
    centroid=centroid.merge(defs[['defect_id','x_camera','y_camera']],on='defect_id')
    centroid['distance_to_fixed_center']=np.hypot(centroid.x-centroid.x_camera,centroid.y-centroid.y_camera)
    table('daily_detection_centroids.csv',centroid)
    det=pd.DataFrame(detection_rows); table('detection_fields.csv',det)
    fixed=det.groupby(['date','defect_id','radius','fixed29']).agg(n_fields=('fid','size'),
                      opportunity_fields=('opportunity','sum'),detected_fields=('detected','sum'),n_points=('n_detection_points','sum')).reset_index()
    table('fixed29_detection.csv',fixed)
    old=pd.read_csv(inputs[-1],dtype={'fid':str,'date':str})
    expected=old[['fid','defect_id','missing_excl_false_10','missing_z5p_nonfinite_10']]
    actual=local[local.phase=='post'].merge(expected,on=['fid','defect_id'],validate='one_to_one')
    assert np.array_equal(actual.status_excl_false,actual.missing_excl_false_10)
    assert np.array_equal(actual.status_z5p_nonfinite,actual.missing_z5p_nonfinite_10)
    # Published rounded provisional point values; each phase is tested independently.
    refs=[(157,'260924|260926','wid_diff','pre',-.001,.064,.001),
          (157,'260924|260926','wid_diff','post',-.001,.064,.001),
          (157,'260924|260926','density_ratio','pre',1.,.82,.005),
          (157,'260924|260926','density_ratio','post',1.,.82,.005),
          (309,'260924|260926','wid_diff','pre',-.001,.027,.001),
          (309,'260924|260926','wid_diff','post',-.001,.027,.001),
          (309,'260924|260926','amp_diff','pre',.011,.076,.001),
          (309,'260924|260926','amp_diff','post',.007,.048,.001),
          (8,'260922|260923','amp_diff','pre',.098,.032,.001),
          (8,'260922|260923','amp_diff','post',.084,.017,.001),
          (8,'260922|260923','wid_diff','pre',.018,.006,.001),
          (8,'260922|260923','wid_diff','post',.020,.009,.001)]
    comp=[]
    for did,b,m,phase,v1,v2,tol in refs:
        for method,metric in [('matched_count_circles',m),('legacy_whole_annulus','legacy_'+m)]:
            rr=changes[(changes.defect_id==did)&(changes.boundary==b)&(changes.phase==phase)&(changes.metric==metric)].iloc[0]
            for side,value in [('before',v1),('after',v2)]:
                got=rr[side+'_mean']
                comp.append(dict(defect_id=did,boundary=b,phase=phase,metric=m,method=method,side=side,
                                 provisional=value,recomputed=got,absolute_difference=abs(got-value),
                                 rounding_tolerance=tol,status='一致' if abs(got-value)<=tol else '不一致'))
    comparison=pd.DataFrame(comp); table('provisional_comparison.csv',comparison)
    table('array_read_audit.csv',AUDIT)
    integrity=[dict(path=p,sha256_before=h,sha256_after=sha(Path(p))) for p,h in initial.items()]
    assert all(r['sha256_before']==r['sha256_after'] for r in integrity)
    table('input_integrity.csv',integrity)
    verify=dict(n_fields=547,n_centers=13,n_phases=2,local_rows=len(local),
                n_control_circles=len(ctr),all_controls_exact_count=True,
                no_matched_controls=int(local.n_control_circles.eq(0).sum()),
                single_matched_control=int(local.n_control_circles.eq(1).sum()),
                n_array_reads=len(AUDIT),prohibited_arrays_opened=0,
                v89_missing_count_matches=True,input_integrity=True,seed=SEED,
                bootstrap_repetitions=args.bootstrap,development_date_counts=nb.date.value_counts().sort_index().to_dict())
    assert len(AUDIT)==1094
    (OUT/'verification.json').write_text(json.dumps(verify,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=['# Codexによる第91版の探索的調査', '',
            'Codexは全547非ブランク視野・13中心・滴下前後の幅、振幅、密度を再計算した。Codexは開発用の事後評価として結果を扱い、物理的原因と欠陥承認を確定しない。', '',
            '## Codexが算出した境の前後の値', '',
            '|中心|境|像|指標|前の平均|後の平均|後−前の95%区間|', '|---|---|---|---|---|---|---|']
    for did,b in [(8,'260922|260923'),(157,'260924|260926'),(309,'260924|260926')]:
        for r in changes[(changes.defect_id==did)&(changes.boundary==b)&changes.metric.isin(['wid_diff','amp_diff','density_ratio'])].itertuples():
            report.append(f'|{did}|{b.replace(chr(124),chr(92)+chr(124))}|{r.phase}|{r.metric}|{r.before_mean:.6g}|{r.after_mean:.6g}|[{r.ci95_low:.6g}, {r.ci95_high:.6g}]|')
    report+=['','## Codexが暫定値と照合した結果','',
             'Codexはprovisional_comparison.csvで暫定値を滴下前後別に照合した。Codexは幅・振幅の許容差を0.001、密度比の許容差を0.005と仮置きし、数値を一致させるための調整を行わなかった。']
    for method,g in comparison.groupby('method'):
        report.append(f'Codexは{method}の24件中{g.status.eq("一致").sum()}件を一致、{g.status.eq("不一致").sum()}件を不一致と記録した。')
    report+=['Codexは暫定ノートの「両方で同じ」を両像への同じ点推定値として照合したが、ノートは滴下後の詳細値を省略している。Codexは中心40の列を像別に分解できず、境も明記されていないため、中心40の数値一致を判定しなかった。Codexは中心40の両境・両像の再計算値をboundary_conditions.csvに保存した。', '',
             '## Codexが使った定義と補正（未指定の細部は仮置き）', '',
             'Codexはpre_ctrとpost_ctrを各画像固有のカメラ座標として別々に使い、幅・振幅の有限値の算術平均を各視野で計算した。Codexは中心から10画素以内の全当てはめピラーを用い、異常判定による選別を加えなかった。',
             'Codexは中心から45画素の円周上に15度間隔で24個の対照候補を置いた。Codexは各候補の第N近傍と第N+1近傍の距離の中点を円の半径とし、中心円と同数Nのピラーを含む円だけを採用した。Codexは半径15画素以下と画像内の候補中心を要求し、円全体を30〜60画素の環内に保った。Codexは対照円の重複を許し、各円の平均を等重みで平均した。',
             'Codexはピラー数を円と画像の交差面積で割り、密度を画素面積当たりの当てはめピラー数と定義した。Codexは同数の円を使うため半径が変わることをcontrol_circles.csvに残した。Codexは密度差と密度比の両方を保存した。Codexは固定10画素・同数の対照では密度比が常に1になる問題を避けるため、対照円の面積を可変とした。',
             f'Codexは対照円を作れない{verify["no_matched_controls"]}行を欠測として残し、中心円のピラーが0本でも中心密度0を保存した。Codexは対照円が1個だけの{verify["single_matched_control"]}行も個数を記録した。Codexは欠測を0に置換せず、各表に有限視野数を残した。',
             'Codexは補正前の環全体を使った算術平均差、中央値差、面積補正密度比をlegacy列に残した。Codexは元の一時スクリプトが保存されていないため、旧集約方法の完全一致を保証していない。',
             f'Codexは種{SEED}で視野全体を日付内で{args.bootstrap}回再抽出し、平均差の百分位95%区間を計算した。Codexは境の前後で各日付の視野数を固定して再抽出し、日付別の全視野を視野数で重み付けした。Codexは全過去日対全後続日を境で分けた。Codexは安定9中心を同じ2境で計算し、stable9_boundary_changes.csvに保存した。',
             'Codexはdaily_global_conditions.csvのmedian列に視野ごとのピラー中央値の日別中央値を記録した。Codexはpost/pre振幅比を視野ごとに計算した。Codexはsigmaを第85版の既存尺度として使い、画像別のsigmaと解釈しなかった。',
             'Codexは検出点を第89版と同じ変換座標、abnかつexclかつ有限z5pかつ絶対値6超で選び、10画素と20画素の両近傍の重心を日付別に保存した。Codexは検出0点の重心を欠測とし、点数と視野数を併記した。Codexはfixed29有無の機会数と検出数、視野ごとのLlinとtLも保存した。',
             'Codexは除外条件が偽の数と非有限z5pの数を当てはめ密度から分けた。Codexはこの状態配列が滴下前ピラーに索引付けされるため、滴下後では変換した滴下前座標を使った。Codexは対照円内の状態ピラー数が当てはめピラー数と一致するとは仮定しなかった。Codexは滴下後中心の欠けた数が第89版の表に完全一致することを検証した。', '',
             '## Codexが再現した節3・節5の範囲', '',
             'Codexは節3の日別振幅・幅・sigma・滴下後対滴下前比をdaily_global_conditions.csvに、節5のfixed29別の検出数をfixed29_detection.csvに保存した。Codexは配列生成時刻とコード版の不変性を推定せず、Llin・tLの値と入力の内容の指紋を保存した。Codexは露光・焦点・日付と試料の交絡を数値だけで除去していない。', '',
             '## 基準への異議', '',
             'Codexは同数の対照円を作る半径の選択が格子位相に依存する点と、中心が0本の視野では同数対照の密度比が定義できない点を記録した。Codexはこれらを探索的な仮置きとして明示し、第90版の合否基準を変更しなかった。Codexは物理的原因と全く信号が無いことを断定しなかった。']
    (OUT/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    notes='# 第91版の記録\n\n## 実装内容・旧版からの変更点\n\nCodexは13中心の画像別の同数対照円、日付内の視野単位再抽出、旧環状比較、欠測分離、座標変換と検出重心を追加した。\n\n## 結果概要と保存先\n\nCodexはdata/results/v91_image_level_conditions/に全547視野の表、報告、監査、検証結果、実行記録を保存した。\n\n## 既知の問題・未解決事項\n\nCodexは対照円の半径・位置と集約方法を仮置きとし、暫定値との不一致を残した。Codexは独立検証と物理的原因の特定を行っていない。\n'
    (OUT/'NOTES.md').write_text(notes,encoding='utf-8')
    code_notes=Path(__file__).parent/'NOTES.md'
    if not code_notes.exists():
        code_notes.write_text(notes,encoding='utf-8')
    elapsed=time.perf_counter()-start
    command='python -B field_level/v91_image_level_conditions/field_conditions.py --bootstrap '+str(args.bootstrap)
    manifest=[]
    for p in sorted(OUT.iterdir())+[Path(__file__),code_notes]:
        if p.is_file():
            manifest.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,
                                 command=command,elapsed_seconds=elapsed,utc_start=utc,seed=SEED,
                                 bootstrap_repetitions=args.bootstrap))
    table('v91_manifest.csv',manifest)
    print(json.dumps(dict(verify,elapsed_seconds=elapsed),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
