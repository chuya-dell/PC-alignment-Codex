"""Field-unit comparisons, preregistered scores and 40 multiplicity slots."""
from field_round8_common import *
from field_round8_scores import cv_score,score
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
    for variant in ['主帰無','N0','N1','N2','N3半分','N3二倍','N4','局所追随']:
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
            if variant in ['主帰無','N0','N1','N2','N4']:
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
        for control in ['N0','N1','N2','N4']:
            if control+'_signed' in maps:
                a=maps['主帰無_signed'];b=maps[control+'_signed'];valid=np.isfinite(a)&np.isfinite(b)&np.isfinite(real_signed)
                main['主対'+control+'共通相関差']=corr(np.where(valid,real_signed,np.nan),np.where(valid,a,np.nan))-corr(np.where(valid,real_signed,np.nan),np.where(valid,b,np.nan))
        # Require identical shared spatial support for primary-versus-donor comparison.
        if '主帰無_signed' in maps and 'N1_signed' in maps:
            a=maps['主帰無_signed'];b=maps['N1_signed'];valid=np.isfinite(a)&np.isfinite(b)&np.isfinite(real_signed)
            mr=corr(np.where(valid,real_signed,np.nan),np.where(valid,a,np.nan));dr=corr(np.where(valid,real_signed,np.nan),np.where(valid,b,np.nan))
            main['主対N1共通相関差']=mr-dr;main['主対N1主相関']=mr;main['主対N1供給元相関']=dr
    dest=OUT/'evaluation_checkpoints';dest.mkdir(exist_ok=True)
    for row in rows:
        if row['条件']=='主帰無':
            baseline=next((b for b in rows if b['条件']=='N0' and b['定義']==row['定義']),None)
            if baseline is not None:
                for col in ['全面決定係数','全面共通除去逸脱度改善','密度重なり','模擬外れ値割合']:
                    row['主対N0'+col+'差']=row.get(col,np.nan)-baseline.get(col,np.nan)
    np.savez_compressed(dest/(key+'.npz'),**maps)
    dump(dict(key=key,signed=signed,outliers=rows),dest/(key+'.json'))
    print('evaluated',key,flush=True)

def load_common(s):
    density=arrays(s['r1files']['density_maps.npz']);common=arrays(s['r1files']['common_profile_residuals.npz'])
    assert density['keys'].tolist()==common['keys'].tolist()
    return dict(keys=density['keys'].tolist(),density=density['density'].astype(float),counts=density['counts'],common=common['common'].astype(float))

def collect():
    signed=[];density=[]
    for p in sorted((OUT/'evaluation_checkpoints').glob('*.json')):
        a=json.loads(p.read_text(encoding='utf-8'));signed+=a['signed'];density+=a['outliers']
    pd.DataFrame(signed).to_csv(OUT/'signed_field_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(density).to_csv(OUT/'outlier_field_metrics.csv',index=False,encoding='utf-8-sig')

def evaluate():
    verify();s=setup();commons=load_common(s)
    for p in sorted((OUT/'experiment_checkpoints').glob('*.json')):
        if not (OUT/'evaluation_checkpoints'/(p.stem+'.json')).exists():evaluate_key(p.stem,s,commons)
    collect()

GROUPS={'必須全体':lambda f:f['必須対象'],'固定29該当':lambda f:f['必須対象']&f['固定29視野'],'主ブランク':lambda f:f['必須対象']&f['ブランク'],'通常選択':lambda f:f['通常選択'],'指定該当':lambda f:f['指定3視野']}

def tests():
    verify();collect();a=pd.read_csv(OUT/'signed_field_metrics.csv',dtype={'日程':str,'基板':str});b=pd.read_csv(OUT/'outlier_field_metrics.csv',dtype={'日程':str,'基板':str})
    signed=a[a['条件']=='主帰無'];density=b[b['条件']=='主帰無'];rows=[]
    for group,choose in GROUPS.items():
        g=signed[choose(signed)]
        for col in ['主対N0共通相関差','主対N1共通相関差','主対N2共通相関差','主対N4共通相関差']:
            mean,p,n=signflip(g[col]);_,pb,_=signflip(g[col],g['日程']+'_'+g['基板'])
            rows.append(dict(集合=group,定義='符号付き差',指標=col,視野数=n,平均差=mean,補正前有意確率=p,補正後有意確率=min(1,p*50),基板一括補正前有意確率=pb,基板一括補正後有意確率=min(1,pb*50)))
        for method in METHODS:
            g=density[choose(density)&density['定義'].eq(method)]
            for col in ['対照差決定係数','対照差共通除去逸脱度改善','主対N0全面決定係数差']:
                mean,p,n=signflip(g[col]);_,pb,_=signflip(g[col],g['日程']+'_'+g['基板'])
                rows.append(dict(集合=group,定義=method,指標=col,視野数=n,平均差=mean,補正前有意確率=p,補正後有意確率=min(1,p*50),基板一括補正前有意確率=pb,基板一括補正後有意確率=min(1,pb*50)))
    assert len(rows)==50;pd.DataFrame(rows).to_csv(OUT/'confirmatory_tests.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    selectors=dict(GROUPS);selectors['全適格実施（補助）']=lambda f:np.ones(len(f),bool)
    for group,choose in selectors.items():
        for frame,cols in [(a,['相関','相関対照差','分散比','無調整分散再現','相関二乗','三成分一致','主対N0共通相関差','主対N1共通相関差','主対N2共通相関差','主対N4共通相関差']), (b,['実測外れ値割合','模擬外れ値割合','ジャカード係数','密度重なり','全面決定係数','共通領域決定係数','対照中央値決定係数','対照差決定係数','全面共通除去逸脱度改善','対照差共通除去逸脱度改善','主対N0全面決定係数差'])]:
            g=frame[choose(frame)]
            for labels,h in g.groupby(['条件','定義'] if frame is b else ['条件']):
                variant=labels[0];method=labels[1] if frame is b else '符号付き差'
                for col in cols:
                    if col not in h:continue
                    v=pd.to_numeric(h[col],errors='coerce').dropna().astype(float)
                    summaries.append(dict(集合=group,条件=variant,定義=method,指標=col,視野数=len(v),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75),負値数=int((v<0).sum())))
    pd.DataFrame(summaries).to_csv(OUT/'metric_summary.csv',index=False,encoding='utf-8-sig')
    print('tests saved 50',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['evaluate','tests']);args=a.parse_args();globals()[args.action]()
