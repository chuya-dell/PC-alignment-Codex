"""Field-unit comparisons, preregistered scores and 40 multiplicity slots."""
from field_round7_common import *
from field_round7_scores import cv_score,score
import argparse

def regression_pair(y,X):
    X=X[:,:,None] if X.ndim==2 else X
    valid=np.isfinite(y)&np.isfinite(X).all(axis=2)
    full,_=cv_score(y,X,valid);ctl=controls(X);shared=valid.copy()
    for c in ctl:shared&=np.isfinite(c).all(axis=2)
    main,_=cv_score(y,X,shared)
    vals=[cv_score(y,c,shared)[0] for c in ctl]
    median=float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
    return dict(全面決定係数=full,共通領域決定係数=main,対照中央値決定係数=median,対照差決定係数=main-median,**{f'対照{i+1}決定係数':v for i,v in enumerate(vals)})

def evaluate_key(key,s,commons):
    folder=OUT/'experiment_checkpoints';files={p.name:p for p in folder.iterdir()}
    with np.load(files[key+'.npz']) as z:data={k:z[k] for k in z.files}
    rec=json.loads(files[key+'.json'].read_text(encoding='utf-8'));xy=data['xy'];actual=data['actual_delta']
    f=s['f'].set_index('key');meta={k:f.loc[key,k] for k in ('日程','基板','ブランク','固定29視野','必須対象','通常選択','指定3視野')}
    i=commons['keys'].index(key);thresholds=table(s['r1files']['round1_thresholds.csv']).set_index('日程')
    real_signed,n=binned(xy,actual);rows=[];signed=[];maps=dict(real_signed=real_signed)
    for variant in ['主帰無','既知変換','N0','N1','N2','N3']:
        simulated=data[variant+'_delta'];ok=np.isfinite(simulated)&np.isfinite(actual)
        if ok.sum()<100:
            signed.append(dict(key=key,**meta,条件=variant,状態='標本化または位置合わせ失敗'));continue
        # Matched validity for every comparison, including map denominators.
        real_onmatch=np.where(ok,actual,np.nan);sim_onmatch=np.where(ok,simulated,np.nan)
        realmap,validn=binned(xy,real_onmatch);simmap,_=binned(xy,sim_onmatch)
        paired=paired_map(realmap,simmap);spectral=spectral_pair(realmap,simmap)
        interior=(xx*32+16>=300)&(xx*32+16<2048-300)&(yy*32+16>=300)&(yy*32+16<2044-300)
        row=dict(key=key,**meta,条件=variant,状態='成功',有効ピラー数=int(ok.sum()),相関=paired['全面'],共通相関=paired['共通領域'],対照相関中央値=paired['対照中央値'],相関対照差=paired['対照差'],縁300除外相関=corr(np.where(interior,realmap,np.nan),np.where(interior,simmap,np.nan)),**spectral,**variance_metrics(realmap,simmap))
        for j,v in enumerate(paired['対照値']):row[f'対照{j+1}相関']=v
        if variant=='N3':row['双一次実測対模擬相関']=corr(binned(xy,data['actual_bilinear_delta'])[0],simmap)
        signed.append(row);maps[variant+'_signed']=simmap
        for j,method in enumerate(METHODS):
            threshold=float(thresholds.loc[meta['日程'],'平均標準偏差' if j==0 else '中央値絶対偏差閾値'])
            realpos=actual>threshold;simpos=simulated>threshold
            rd,_=binned(xy,np.where(ok,realpos.astype(float),np.nan));sd,_=binned(xy,np.where(ok,simpos.astype(float),np.nan))
            maps[f'{variant}_density_{j}']=sd
            if variant=='主帰無':maps[f'real_density_{j}']=rd
            union=np.count_nonzero((realpos|simpos)&ok);intersection=np.count_nonzero(realpos&simpos&ok)
            ov=paired_map(rd,sd,overlap)
            q=commons['common'][i,j];y=rd-q;X=np.stack([simmap,sd],axis=2)
            basic=dict(key=key,**meta,条件=variant,定義=method,閾値=threshold,有効ピラー数=int(ok.sum()),実測外れ値割合=float(realpos[ok].mean()),模擬外れ値割合=float(simpos[ok].mean()),外れ値交わり=intersection,外れ値和=union,ジャカード係数=intersection/union if union else np.nan,密度重なり=ov['全面'],共通密度重なり=ov['共通領域'],対照密度重なり中央値=ov['対照中央値'],密度重なり対照差=ov['対照差'])
            if variant=='N3':basic['双一次実測外れ値割合']=float(np.mean(data['actual_bilinear_delta'][ok]>threshold))
            # All control conditions have descriptive full-domain scores.
            if variant=='主帰無':
                basic.update(regression_pair(y,X))
                countk=np.rint(np.nan_to_num(rd)*validn).astype(int)
                basic.update(score(countk,validn,q,X))
                # Reconcile one valid construction with old n and old densities.
                if ok.all():
                    assert np.array_equal(validn,commons['counts'][i])
                    assert np.nanmax(abs(rd-commons['density'][i,j]))<1e-7
            else:
                valid=np.isfinite(y)&np.isfinite(X).all(axis=2)
                basic['全面決定係数']=cv_score(y,X,valid)[0]
            rows.append(basic)
    main=next((r for r in signed if r['条件']=='主帰無'),None);n1=next((r for r in signed if r['条件']=='N1'),None)
    if main is not None:
        # Require identical shared spatial support for primary-versus-donor comparison.
        if '主帰無_signed' in maps and 'N1_signed' in maps:
            a=maps['主帰無_signed'];b=maps['N1_signed'];valid=np.isfinite(a)&np.isfinite(b)&np.isfinite(real_signed)
            mr=corr(np.where(valid,real_signed,np.nan),np.where(valid,a,np.nan));dr=corr(np.where(valid,real_signed,np.nan),np.where(valid,b,np.nan))
            main['主対N1共通相関差']=mr-dr;main['主対N1主相関']=mr;main['主対N1供給元相関']=dr
    dest=OUT/'evaluation_checkpoints';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/(key+'.npz'),**maps)
    dump(dict(key=key,signed=signed,outliers=rows),dest/(key+'.json'))
    print('evaluated',key,flush=True)

def load_common(s):
    density=arrays(s['r1files']['density_maps.npz']);common=arrays(s['r1files']['common_profile_residuals.npz'])
    assert density['keys'].tolist()==common['keys'].tolist()
    return dict(keys=density['keys'].tolist(),density=density['density'].astype(float),counts=density['counts'],common=common['common'].astype(float))

def evaluate(limit=0):
    verify();s=setup();commons=load_common(s)
    completed=sorted(p.stem for p in (OUT/'experiment_checkpoints').glob('*.json'))
    if limit:completed=completed[:limit]
    for key in completed:
        dest=OUT/'evaluation_checkpoints'
        if (dest/(key+'.json')).exists():continue
        evaluate_key(key,s,commons)
    collect()

def watch():
    verify();s=setup();commons=load_common(s);expected=set(s['f'].loc[s['f']['回復成功'],'key'])
    done=set(p.stem for p in (OUT/'evaluation_checkpoints').glob('*.json'))
    while True:
        completed=set(p.stem for p in (OUT/'experiment_checkpoints').glob('*.json'))
        for key in sorted(completed-done):
            try:evaluate_key(key,s,commons)
            except json.JSONDecodeError:continue
            done.add(key)
        collect()
        if expected<=done:break
        errors=OUT/'execution_errors'
        failed=set(p.stem for p in errors.glob('*.json')) if errors.exists() else set()
        if expected <= (done|failed):break
        time.sleep(10)
    tests()
    print('watch finished',len(done),flush=True)

def collect():
    signed=[];outliers=[]
    for p in sorted((OUT/'evaluation_checkpoints').glob('*.json')):
        a=json.loads(p.read_text(encoding='utf-8'));signed+=a['signed'];outliers+=a['outliers']
    pd.DataFrame(signed).to_csv(OUT/'signed_field_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(outliers).to_csv(OUT/'outlier_field_metrics.csv',index=False,encoding='utf-8-sig')

def tests():
    verify();collect();signed=pd.read_csv(OUT/'signed_field_metrics.csv',dtype={'日程':str,'基板':str});density=pd.read_csv(OUT/'outlier_field_metrics.csv',dtype={'日程':str,'基板':str})
    allsigned=signed[signed['条件']=='主帰無'].copy();allcounts=density[density['条件']=='主帰無'].copy()
    signed=allsigned[allsigned['必須対象']];density=allcounts[allcounts['必須対象']]
    groups={'必須全体':lambda f:np.ones(len(f),bool),'固定29回復':lambda f:f['固定29視野'],'ブランク':lambda f:f['ブランク'],'通常選択':lambda f:f['通常選択'],'指定3視野':lambda f:f['指定3視野']}
    tests=[]
    for group,choose in groups.items():
        rows=signed[choose(signed)]
        for name in ('相関対照差','主対N1共通相関差'):
            a,p,n=signflip(rows[name]);_,pg,ng=signflip(rows[name],rows['日程']+'_'+rows['基板'])
            tests.append(dict(集合=group,指標=name,定義='符号付き差',視野数=n,平均対照差=a,補正前有意確率=p,補正後有意確率=min(1,p*40),基板一括補正前有意確率=pg,基板一括補正後有意確率=min(1,pg*40)))
        for method in METHODS:
            rows=density[choose(density)&(density['定義']==method)]
            for name in ('対照差決定係数','対照差共通除去逸脱度改善','密度重なり対照差'):
                a,p,n=signflip(rows[name]);_,pg,ng=signflip(rows[name],rows['日程']+'_'+rows['基板'])
                tests.append(dict(集合=group,指標=name,定義=method,視野数=n,平均対照差=a,補正前有意確率=p,補正後有意確率=min(1,p*40),基板一括補正前有意確率=pg,基板一括補正後有意確率=min(1,pg*40)))
    assert len(tests)==40
    pd.DataFrame(tests).to_csv(OUT/'confirmatory_tests.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    for group,choose in groups.items():
        for frame,cols in [(signed,['相関','相関対照差','主対N1共通相関差','分散比','無調整分散再現','相関二乗']), (density,['実測外れ値割合','模擬外れ値割合','ジャカード係数','密度重なり','密度重なり対照差','全面決定係数','共通領域決定係数','対照中央値決定係数','対照差決定係数','全面逸脱度改善','全面共通除去逸脱度改善','対照中央値共通除去逸脱度改善','対照差共通除去逸脱度改善','全面共通除去順位相関'])]:
            sub=frame[choose(frame)]
            parts=[('符号付き差',sub)] if frame is signed else list(sub.groupby('定義'))
            for method,g in parts:
                for col in cols:
                    v=g[col].dropna()
                    summaries.append(dict(集合=group,定義=method,指標=col,視野数=len(v),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75),最小=v.min(),最大=v.max(),負値数=int((v<0).sum())))
    for group,extra_only in [('回復全体（補助）',False),('追加視野（補助）',True)]:
        for frame,cols in [(allsigned,['相関','相関対照差','主対N1共通相関差','分散比','無調整分散再現','相関二乗']), (allcounts,['実測外れ値割合','模擬外れ値割合','ジャカード係数','密度重なり','密度重なり対照差','全面決定係数','共通領域決定係数','対照中央値決定係数','対照差決定係数','全面逸脱度改善','全面共通除去逸脱度改善','対照中央値共通除去逸脱度改善','対照差共通除去逸脱度改善','全面共通除去順位相関'])]:
            sub=frame[~frame['必須対象']] if extra_only else frame
            parts=[('符号付き差',sub)] if frame is allsigned else list(sub.groupby('定義'))
            for method,g in parts:
                for col in cols:
                    v=g[col].dropna();summaries.append(dict(集合=group,定義=method,指標=col,視野数=len(v),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75),最小=v.min(),最大=v.max(),負値数=int((v<0).sum())))
    pd.DataFrame(summaries).to_csv(OUT/'metric_summary.csv',index=False,encoding='utf-8-sig')
    fixed=signed[signed['固定29視野']];matches=int(fixed['三成分一致'].sum())
    t=pd.DataFrame(tests)
    significant=False
    for group in ['必須全体','固定29回復']:
        p=t[(t['集合']==group)&t['指標'].isin(['相関対照差','主対N1共通相関差'])]['補正後有意確率']
        significant|=len(p)==2 and bool((p<.05).all())
    dump(dict(確認的検定枠=40,固定29回復評価数=len(fixed),三成分一致数=matches,相関二検定の支持=significant,主機構支持条件=bool(significant and matches>len(fixed)/2)),OUT/'decision.json')
    print('tests complete',len(tests),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['evaluate','tests','watch']);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    if a.action=='evaluate':evaluate(a.limit)
    else:globals()[a.action]()
