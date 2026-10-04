"""Reevaluate saved maps and cheap deterministic previous predictors."""
from field_round6_common import *
import argparse

def plane():
    x=(xx+.5)/32-1;y=(yy+.5)/32-1
    return np.stack([x,y,x*x,y*y,x*y],axis=-1)
def mean_map(a):
    good=np.isfinite(a);count=good.sum(axis=0)
    return np.divide(np.nansum(a,axis=0),count,out=np.full(a.shape[1:],np.nan),where=count>0)
def direction(angle):
    theta=np.deg2rad(angle+90);u=((xx+.5)*32-1024)*np.cos(theta)+((yy+.5)*32-1024)*np.sin(theta)
    return np.stack([u/1024,(u/1024)**2]+[fn(2*np.pi*u/p) for p in (256,512,1024) for fn in (np.sin,np.cos)],axis=-1)
def feature_index():
    folders=[OLD/'bf_models',R2/'raw_features',R3/'raw_candidates',OLD/'bf_models',R5/'F_features',RESUME/'F_features']
    return [{p.stem:p for p in d.iterdir() if p.suffix=='.npz'} for d in folders]
def reevaluate(limit=0,pilot=False):
    verify();started=time.monotonic();f=fields();keys,k,n,q,res,dm=common_data();assert f.key.tolist()==keys
    idx={key:i for i,key in enumerate(keys)};features=feature_index()
    directions=table(R3/'round3_orientation_predictions.csv').set_index(['key','定義'])
    audited=table(RESUME/'round5_F_explained_fraction.csv')
    elig={r.key for _,r in audited.iterrows() if r['モデル']=='F' and r['非周期目印検証'] and r['F支持条件']}
    ref=set(table(RESUME/'round5_reference_blank_fields.csv').key)
    choose=f
    if pilot:choose=f[f.key.isin([keys[0],'260926_01_8','260926_7_8',next(iter(sorted(elig)))])]
    elif limit:choose=f.head(limit)
    dest=OUT/'model_checkpoints';dest.mkdir(exist_ok=True)
    for j,(_,r) in enumerate(choose.iterrows()):
        path=dest/(r.key+'.json')
        if path.exists():continue
        i=idx[r.key];models={'G座標二次面':plane()};notes={}
        for source,names in ((features[0],['B']),(features[1],['G','I']),(features[2],['H'])):
            if r.key in source:
                a=arrays(source[r.key])
                for name in names:
                    if name in a:models[name]=a[name]
        if r.key in elig:
            pp=next((fs[r.key] for fs in features[3:][::-1] if r.key in fs),None)
            if pp is not None:
                a=arrays(pp)
                if 'F' in a:models['F']=a['F']
        rows=[]
        for mi,method in enumerate(METHODS):
            mod=models.copy();d=directions.loc[(r.key,method)]
            if d['C利用可能']:mod['C']=direction(d['予測帯軸度'])
            if r.key not in ref:
                sameboard=(f['日程']==r['日程'])&(f['基板']==r['基板'])
                select=(~sameboard)&(~f.key.isin(ref))&(f['日程']==r['日程'])&(f['ブランク']==r['ブランク'])
                fallback=not select.any()
                if fallback:select=(~sameboard)&(~f.key.isin(ref))&(f['ブランク']==r['ブランク'])
                assert select.any()
                mod['J']=mean_map(dm[select,mi])-mean_map(dm[~sameboard,mi])
                notes.update(J他日程へ切替=bool(fallback),J学習視野数=int(select.sum()))
            for name,X in mod.items():
                vals=score(k[i,mi],n[i],q[i,mi],X)
                rows.append(dict(**meta(r),定義=method,モデル=name,採用区分='画像由来記述のみ' if name in ('G','I','H') else '確認的モデル整合性',**notes,**vals))
        dump(rows,path)
        if j%10==0:print('models',j,r.key,'rows',len(rows),'seconds',round(time.monotonic()-started,1),flush=True)
    csv([r for p in sorted(dest.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))],'model_field_metrics.csv')
    print('models saved',len(list(dest.glob('*.json'))),flush=True)

def summary():
    a=pd.read_csv(OUT/'model_field_metrics.csv',dtype={'日程':str,'基板':str});rows=[];tests=[]
    cols=['全面逸脱度改善','全面共通除去逸脱度改善','全面順位相関','全面共通除去順位相関','共通領域逸脱度改善','共通領域共通除去逸脱度改善','対照中央値逸脱度改善','対照差逸脱度改善','対照中央値共通除去逸脱度改善','対照差共通除去逸脱度改善','共通領域共通除去順位相関','対照中央値共通除去順位相関','対照差共通除去順位相関']
    for (name,method),g in a.groupby(['モデル','定義']):
        for label,h in [('全利用可能',g),('固定29利用可能',g[g['固定29視野']]),('主ブランク',g[g['ブランク']])]:
            for c in cols:
                v=h[c];rows.append(dict(モデル=name,定義=method,対象=label,指標=c,視野数=len(h),有効視野数=int(v.notna().sum()),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75),負値数=int((v<0).sum())))
    for name in ('B','F','C','G座標二次面','J'):
        for method in METHODS:
            g=a[(a['モデル']==name)&(a['定義']==method)]
            for label,h in [('全利用可能',g),('固定29利用可能',g[g['固定29視野']])]:
                for c in ('対照差逸脱度改善','対照差共通除去順位相関'):
                    ob,p,nv=signflip(h[c]);_,bp,_=signflip(h[c],h['日程']+'_'+h['基板'])
                    tests.append(dict(モデル=name,定義=method,対象=label,指標=c,視野数=nv,平均対照差=ob,補正前有意確率=p,ボンフェローニ補正後=min(1,p*40) if np.isfinite(p) else np.nan,日程基板共通符号有意確率=bp,状態='実施' if nv else '利用可能なし'))
    csv(rows,'model_summary.csv');csv(tests,'model_tests.csv')
    check=dict(保存視野数=a.key.nunique(),保存行数=len(a),未収束全面分割数=int(a['全面未収束分割数'].sum()),未収束共通分割数=int(a['共通領域未収束分割数'].sum()),未収束対照分割数=int(a['対照未収束分割数'].sum()),確認的検定枠数=len(tests))
    dump(check,OUT/'model_verification.json');print(check,flush=True)

def pilot():
    verify();random=rng('pilot');n=np.full((64,64),21.);q=np.full((64,64),.01)
    X=np.cos(2*np.pi*xx/12);truth=expit(logit(q)+1.5*X)
    k=random.binomial(n.astype(int),truth)
    a=cv_count(k,n,q,X[:,:,None],np.ones((64,64),bool),True)
    assert a['逸脱度改善']>0 and a['共通除去順位相関']>0 and a['未収束分割数']==0
    for tr,te in SPLITS:
        assert not np.intersect1d(tr,te).size
        mask=np.zeros((64,64),'uint8');mask.ravel()[te]=1
        assert not cv2.dilate(mask,np.ones((5,5),'uint8')).ravel()[tr].any()
    assert deviance(np.array([0.,21.]),np.array([21.,21.]),np.array([EPS,1-EPS]))>=0
    mat=np.column_stack([random.normal(size=500),random.normal(size=500)]);trials=np.full(500,20.);cnt=random.binomial(20,expit(-3+mat@np.array([.5,-.3])))
    coef,conv,_=fit_binomial(mat,cnt,trials,np.full(500,-3.))
    from scipy.optimize import minimize
    opt=minimize(lambda c:np.sum(trials*np.logaddexp(0,-3+mat@c)-cnt*(-3+mat@c))+.5*c@c,np.zeros(2),method='BFGS',tol=1e-8)
    assert np.max(abs(coef-opt.x))<1e-5
    a.pop('予測地図');a.pop('基準地図')
    dump(dict(既知二項波= a,緩衝領域確認=True,独立最適化係数最大差=float(np.max(abs(coef-opt.x)))),OUT/'pilot_verification.json')
    reevaluate(pilot=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','run','summary']);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    if args.stage=='run':reevaluate(args.limit)
    else:globals()[args.stage]()
