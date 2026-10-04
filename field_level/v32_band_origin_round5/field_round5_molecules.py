"""Date-stratified field label permutation and independent group-map prediction."""
from field_round5_common import *
from field_round5_methods import cv_score, controls
import argparse
METRICS=['外れ値割合','帯状うろこ状候補','双方向平均決定係数','残差分散']
def manifest():
    a=table(OLD/'round1_fields_both_definitions.csv')
    cl=table(OLD/'round1_spatial_classification.csv').query('縁除外画素 == 0')
    a=a.merge(cl[['key','定義','分類']],on=['key','定義'],validate='one_to_one')
    a['帯状うろこ状候補']=a['分類'].isin(['帯状候補','うろこ状候補']).astype(float)
    a['帯状候補']=(a['分類']=='帯状候補').astype(float);a['うろこ状候補']=(a['分類']=='うろこ状候補').astype(float)
    noise=table(PREV/'round4_noise_ceiling.csv').query("分割 == '格子行偶奇'")
    noise=noise.groupby(['key','定義'],as_index=False)['双方向平均決定係数'].first()
    a=a.merge(noise,on=['key','定義'],validate='one_to_one')
    with np.load(OLD/'common_profile_residuals.npz') as z:res=z['residual'].copy();keys=z['keys'].tolist()
    index={k:i for i,k in enumerate(keys)}
    a['残差分散']=[float(np.nanvar(res[index[r.key],METHODS.index(r['定義'])])) for _,r in a.iterrows()]
    assert len(a)==1264 and a['ブランク'].sum()==138
    assert not a[(a['日程'].isin(['260922','260923']))&(a['基板']=='01')]['ブランク'].any()
    assert a[(a['日程']=='260926')&(a['基板']=='01')]['ブランク'].all()
    csv(a,'round5_molecule_field_metrics.csv');return a,res,keys
def permutations(g,columns,reps=10000):
    rng=np.random.default_rng(20261004);observed=[];null=np.zeros((reps,len(columns)))
    for date,h in g.groupby('日程'):
        y=h[columns].to_numpy(float);blank=h['ブランク'].to_numpy(bool)
        assert np.isfinite(y).all() and blank.any() and (~blank).any()
        observed.append(y[~blank].mean(axis=0)-y[blank].mean(axis=0))
        nb=int(blank.sum());n=len(h)
        # Fixed group counts within date; the field remains the resampling unit.
        for start in range(0,reps,500):
            sz=min(500,reps-start);pick=np.argsort(rng.random((sz,n)),axis=1)[:,:nb]
            bs=y[pick].mean(axis=1);ms=(y.sum(axis=0)-bs*nb)/(n-nb)
            null[start:start+sz]+=ms-bs
    return np.mean(observed,axis=0),null/len(observed)
def tests():
    verify();a,_,_=manifest();rows=[];dates=[]
    for method,g in a.groupby('定義'):
        obs,null=permutations(g,METRICS)
        for j,col in enumerate(METRICS):
            p=(1+int((null[:,j]>=obs[j]-1e-15).sum()))/10001
            critical=float(np.quantile(null[:,j],1-.05/8))
            rows.append(dict(定義=method,指標=col,分子あり視野数=int((~g['ブランク']).sum()),ブランク視野数=int(g['ブランク'].sum()),日程数=g['日程'].nunique(),同重み日程平均差=obs[j],補正前有意確率=p,ボンフェローニ補正後=min(1,p*8),補正後帰無臨界差=critical,加算効果80パーセント検出近似=critical-float(np.quantile(null[:,j],.2))))
        for date,h in g.groupby('日程'):
            for col in METRICS+['帯状候補','うろこ状候補']:
                b=h.loc[h['ブランク'],col];m=h.loc[~h['ブランク'],col]
                dates.append(dict(定義=method,日程=date,指標=col,ブランク視野数=len(b),分子あり視野数=len(m),ブランク平均=b.mean(),分子あり平均=m.mean(),分子あり引くブランク=m.mean()-b.mean(),ブランク中央値=b.median(),分子あり中央値=m.median()))
    csv(rows,'round5_J_tests.csv');csv(dates,'round5_J_by_date.csv');print('molecule tests completed',flush=True)
def score(y,X):
    X=X[:,:,None];valid=np.isfinite(y)&np.isfinite(X).all(axis=2)
    full,_=cv_score(y,X,valid);ctl=controls(X);common=valid.copy()
    for c in ctl:common &= np.isfinite(c).all(axis=2)
    real,_=cv_score(y,X,common);values=[cv_score(y,c,common)[0] for c in ctl];median=float(np.nanmedian(values))
    return dict(全面決定係数=full,対照共通領域決定係数=real,対照決定係数中央値=median,対照との差=real-median,全面升目数=int(valid.sum()),対照共通升目数=int(common.sum()),**{f'対照{j+1}決定係数':v for j,v in enumerate(values)})
def explain(limit=0):
    verify();a,res,keys=manifest();index={k:i for i,k in enumerate(keys)}
    f=a[a['定義']==METHODS[0]].set_index('key').loc[keys].reset_index()
    dest=OUT/'J_checkpoints';dest.mkdir(exist_ok=True)
    with np.load(OLD/'density_maps.npz') as z:all_dm=z['density'].copy()
    for i,r in f.iloc[:limit or len(f)].iterrows():
        p=dest/(r.key+'.json')
        if p.exists():continue
        same_sub=(f['日程']==r['日程'])&(f['基板']==r['基板'])
        select=(~same_sub)&(f['日程']==r['日程'])&(f['ブランク']==r['ブランク'])
        fallback=not select.any()
        if fallback:select=(~same_sub)&(f['ブランク']==r['ブランク'])
        assert select.any()
        rows=[]
        for mi,method in enumerate(METHODS):
            # Source density, rather than source residual, avoids the target's
            # tiny contribution through leave-one-field-out shared averages.
            dm=all_dm[:,mi]
            train_mean=np.nanmean(dm[~same_sub],axis=0)
            predictor=np.nanmean(dm[select],axis=0)-train_mean
            rows.append(dict(key=r.key,日程=r['日程'],基板=r['基板'],ブランク=bool(r['ブランク']),固定29視野=bool(r['固定29視野']),定義=method,同日同群学習不能=fallback,学習視野数=int(select.sum()),モデル='分子群の他基板平均地図',**score(res[i,mi],predictor)))
        dump(rows,p)
        if i%100==0:print('J explanation',i,r.key,flush=True)
    csv([r for p in sorted(dest.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))],'round5_J_explained_fraction.csv')
def pilot():
    verify();a,_,_=manifest();g=a[a['定義']==METHODS[0]]
    obs,null=permutations(g,['帯状うろこ状候補'],200)
    assert null.shape==(200,1)
    allzero=g.copy();allzero['z']=0;ob,n=permutations(allzero,['z'],20);assert ob[0]==0 and not n.any()
    explain(2)
    dump(dict(視野数=len(g),二定義視野数=len(a),日程数=g['日程'].nunique(),零効果検算=True,並替小規模形状=list(null.shape),基板01日程別確認=True),OUT/'round5_J_pilot.json')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','tests','explain']);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    if args.stage=='explain':explain(args.limit)
    else:globals()[args.stage]()
