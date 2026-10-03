"""Field-level inference, calibration interpretation, and reviewable report."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse, subprocess, json
import numpy as np
import pandas as pd
from field_round4_analysis import ROOT, OUT, OLD, PREV, METHODS, SEED, table, csv, dump, verify, rng_for
from field_round4_methods import angular_difference, folds, cv_score

def qstats(a):
    v=pd.Series(a).dropna()
    return dict(視野数=len(v),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75),負値数=int((v<0).sum()))

def subsets(a):
    yield '全利用可能',a
    yield '固定29利用可能',a[a['固定29視野']]
    yield 'ブランク',a[a['ブランク']]
    yield '分子あり等',a[~a['ブランク']]
    for day,g in a.groupby('日程'):yield '日程'+day,g

def summarize():
    verify();noise=table(OUT/'round4_noise_ceiling.csv');inj=table(OUT/'round4_injection.csv');rows=[]
    for (method,split),g in noise[noise['予測向き']=='組1から組2'].groupby(['定義','分割']):
        for label,a in subsets(g):
            for metric in ['双方向平均決定係数','全数信号割合近似']:
                rows.append(dict(定義=method,分割=split,対象=label,指標=metric,**qstats(a[metric])))
    csv(rows,'round4_noise_summary.csv');rows=[]
    for (method,kind,angle,period,strength),g in inj.groupby(['定義','注入法','帯軸度','周期画素','目標追加割合']):
        for label,a in subsets(g):
            for metric in ['全面決定係数','対照共通領域決定係数','対照決定係数中央値','対照との差','実現追加割合']:
                rows.append(dict(定義=method,注入法=kind,帯軸度=angle,周期画素=period,目標追加割合=strength,対象=label,指標=metric,**qstats(a[metric])))
    csv(rows,'round4_injection_summary.csv')
    manifest=table(OUT/'round4_field_manifest.csv');rows=[]
    for (day,blank),a in manifest.groupby(['日程','ブランク']):
        rows.append(dict(日程=day,ブランク=blank,保存差視野数=len(a),固定29数=int(a['固定29視野'].sum()),生画像組数=int(a['生画像組利用可能'].sum()),生画像固定29数=int((a['生画像組利用可能']&a['固定29視野']).sum())))
    csv(rows,'round4_date_coverage.csv')

def attribution():
    verify();images=table(OUT/'round4_single_image_spectrum.csv');spatial=table(OLD/'round1_spatial_with_metadata.csv');spatial=spatial[spatial['縁除外画素']==0]
    manifest=table(OUT/'round4_field_manifest.csv');available=set(manifest[manifest['生画像組利用可能']].key)
    lookup=images.set_index(['key','時点']);rows=[]
    for _,r in spatial[spatial.key.isin(available)].iterrows():
        item={k:r[k] for k in ['key','日程','基板','ブランク','固定29視野','定義','分類','周期候補','第一帯軸度','第一スペクトル周期画素','第一自己相関周期画素','第一周期一致']}
        for stage in ['洗浄前','洗浄後']:
            s=lookup.loc[(r.key,stage)];support=bool(s['周期候補'] and s['第一周期一致'])
            angle=float(angular_difference(s['第一帯軸度'],r['第一帯軸度']))
            relative=float(abs(s['第一スペクトル周期画素']-r['第一スペクトル周期画素'])/r['第一スペクトル周期画素'])
            item.update({stage+'周期支持':support,stage+'第一帯軸度':s['第一帯軸度'],stage+'第一スペクトル周期画素':s['第一スペクトル周期画素'],stage+'第一自己相関周期画素':s['第一自己相関周期画素'],stage+'分類':s['分類'],stage+'角度差度':angle,stage+'周期相対差':relative,stage+'方向一致':bool(support and angle<=15),stage+'周期一致':bool(support and relative<=.2),stage+'同時一致':bool(support and angle<=15 and relative<=.2),stage+'円周一致寄与':float(np.cos(2*np.deg2rad(angle))) if support else 0.})
        item['帰属分類']=('両方' if item['洗浄前同時一致'] and item['洗浄後同時一致'] else '前だけ' if item['洗浄前同時一致'] else '後だけ' if item['洗浄後同時一致'] else 'どちらでもない')
        item['後引く前同時一致']=int(item['洗浄後同時一致'])-int(item['洗浄前同時一致'])
        rows.append(item)
    csv(rows,'round4_attribution_by_field.csv');a=pd.DataFrame(rows);summary=[]
    for method,g in a.groupby('定義'):
        for label,b in subsets(g):
            for blank in ['全区分','ブランク','ブランク以外']:
                h=b if blank=='全区分' else b[b['ブランク']==(blank=='ブランク')]
                for rule in ['周期候補','帯状うろこ状のみ']:
                    d=h[h['周期候補']] if rule=='周期候補' else h[h['分類'].isin(['帯状候補','うろこ状候補'])]
                    row=dict(定義=method,対象=label,ブランク区分=blank,分布規則=rule,生画像利用可能視野数=len(h),外れ値周期対象視野数=len(d))
                    for kind in ['前だけ','後だけ','両方','どちらでもない']:row[kind+'数']=int((d['帰属分類']==kind).sum())
                    for stage in ['洗浄前','洗浄後']:
                        row.update({stage+'周期支持数':int(d[stage+'周期支持'].sum()),stage+'方向一致割合':d[stage+'方向一致'].mean(),stage+'周期一致割合':d[stage+'周期一致'].mean(),stage+'同時一致割合':d[stage+'同時一致'].mean(),stage+'円周一致平均':d[stage+'円周一致寄与'].mean(),stage+'角度差中央値度':d.loc[d[stage+'周期支持'],stage+'角度差度'].median()})
                    summary.append(row)
    csv(summary,'round4_attribution_summary.csv')
    image_summary=[]
    for (day,stage,blank),d in images.groupby(['日程','時点','ブランク']):
        image_summary.append(dict(日程=day,時点=stage,ブランク=blank,単独像数=len(d),周期候補数=int(d['周期候補'].sum()),第一周期支持数=int((d['周期候補']&d['第一周期一致']).sum()),帯状候補数=int((d['分類']=='帯状候補').sum()),うろこ状候補数=int((d['分類']=='うろこ状候補').sum()),第一帯軸中央値度=d['第一帯軸度'].median(),第一スペクトル周期中央値画素=d['第一スペクトル周期画素'].median()))
    csv(image_summary,'round4_single_image_summary.csv')

def sign_test(values,groups=None):
    v=np.asarray(values,float);valid=np.isfinite(v);v=v[valid]
    if not len(v):return np.nan,np.nan
    codes=np.arange(len(v)) if groups is None else pd.factorize(np.asarray(groups)[valid])[0]
    sums=np.bincount(codes,weights=v);rng=rng_for('sign');measured=float(v.mean());hits=0
    for _ in range(20):hits+=int(np.sum((rng.choice([-1,1],size=(500,len(sums)))@sums/len(v))>=measured-1e-14))
    return measured,(hits+1)/10001

def orientation_permutation(a,stage,group_by_board=False):
    if not len(a):return dict(円周一致平均=np.nan,同時一致割合=np.nan,円周有意確率=np.nan,同時一致有意確率=np.nan)
    target=a['第一帯軸度'].to_numpy();period=a['第一スペクトル周期画素'].to_numpy()
    img_angle=a[stage+'第一帯軸度'].to_numpy();img_period=a[stage+'第一スペクトル周期画素'].to_numpy();support=a[stage+'周期支持'].to_numpy()
    codes=pd.factorize(a['日程']+'_'+a['基板'] if group_by_board else a['日程'])[0];idx=[np.flatnonzero(codes==k) for k in np.unique(codes)]
    circ=float(a[stage+'円周一致寄与'].mean());match=float(a[stage+'同時一致'].mean());hits=np.zeros(2,int);random=rng_for(stage+str(group_by_board));order=np.arange(len(a))
    for _ in range(10000):
        for k in idx:order[k]=random.permutation(k)
        difference=angular_difference(img_angle[order],target)
        c=np.mean(np.where(support[order],np.cos(2*np.deg2rad(difference)),0.))
        agreement=np.mean(support[order]&(difference<=15)&(np.abs(img_period[order]-period)/period<=.2))
        hits+=np.array([c>=circ-1e-14,agreement>=match-1e-14])
    return dict(円周一致平均=circ,同時一致割合=match,円周有意確率=(hits[0]+1)/10001,同時一致有意確率=(hits[1]+1)/10001)

def tests():
    verify();rows=[];noise=table(OUT/'round4_noise_ceiling.csv');a=table(OUT/'round4_attribution_by_field.csv')
    n=noise[noise['分割']=='格子行偶奇'].groupby(['key','定義'],as_index=False).agg({'対照との差':'mean','日程':'first','基板':'first','固定29視野':'first'})
    for method in METHODS:
        for subset in ['全利用可能','固定29利用可能']:
            d=n[n['定義']==method]
            if subset.startswith('固定'):d=d[d['固定29視野']]
            stat,p=sign_test(d['対照との差']);_,sens=sign_test(d['対照との差'],d['日程']+'_'+d['基板'])
            rows.append(dict(検定='較正二分割対照差',定義=method,対象=subset,視野数=len(d),統計量=stat,補正前有意確率=p,基板感度有意確率=sens))
            b=a[(a['定義']==method)&a['周期候補']]
            if subset.startswith('固定'):b=b[b['固定29視野']]
            for stage in ['洗浄前','洗浄後']:
                result=orientation_permutation(b,stage);sensitivity=orientation_permutation(b,stage,True)
                for metric in ['円周','同時一致']:
                    rows.append(dict(検定=stage+metric,定義=method,対象=subset,視野数=len(b),統計量=result['円周一致平均' if metric=='円周' else '同時一致割合'],補正前有意確率=result[metric+'有意確率'],基板感度有意確率=sensitivity[metric+'有意確率']))
            stat,p=sign_test(b['後引く前同時一致']);_,sens=sign_test(b['後引く前同時一致'],b['日程']+'_'+b['基板'])
            rows.append(dict(検定='洗浄後引く前同時一致',定義=method,対象=subset,視野数=len(b),統計量=stat,補正前有意確率=p,基板感度有意確率=sens))
    assert len(rows)==24
    for row in rows:
        row['ボンフェローニ補正後']=min(1,24*row['補正前有意確率']) if np.isfinite(row['補正前有意確率']) else np.nan
        row['検定回数']=10000;row['状態']='実施' if row['視野数'] else '対象視野なし'
    csv(rows,'round4_tests.csv');print('tests complete',flush=True)

def old_sources():
    b=table(OLD/'round1_bf_explained_fraction.csv');b=b[(b['モデル']=='B')|((b['モデル']=='F')&b['非周期目印検証']&b['F支持条件'])]
    b['仮説']=b['モデル'];b=b[b['仮説'].isin(['B','F'])]
    g=table(ROOT/'data/results/v29_band_origin_round2/round2_geometry_explained_fraction.csv');g['仮説']='G（方向を限定しない二次面）'
    o=table(ROOT/'data/results/v29_band_origin_round2/round2_optical_explained_fraction.csv');o['仮説']=o['モデル'];o=o[o['仮説'].isin(['G','I'])]
    h=table(PREV/'round3_explained_fraction.csv');h['仮説']=h['モデル'].map({'H':'H','C':'C','共同':'H・C共同','前像':'H（前像対照）'});h=h[h['仮説'].notna()]
    return pd.concat([b,g,o,h],ignore_index=True)

def reread():
    verify();noise=table(OUT/'round4_noise_ceiling.csv');noise=noise[(noise['分割']=='格子行偶奇')&(noise['予測向き']=='組1から組2')]
    refs=noise[['key','定義','双方向平均決定係数','全数信号割合近似']]
    prior=old_sources();joined=prior.merge(refs,on=['key','定義'],validate='many_to_one')
    joined['視野別天井比']=joined['全面決定係数']/joined['双方向平均決定係数'].where(joined['双方向平均決定係数']>0)
    joined['視野別全数近似比']=joined['全面決定係数']/joined['全数信号割合近似'].where(joined['全数信号割合近似']>0)
    csv(joined,'round4_hypothesis_ceiling_by_field.csv')
    comparison=table(PREV/'round3_all_hypotheses_comparison.csv');fields=table(OLD/'round1_fields_both_definitions.csv');sp=table(OLD/'round1_spatial_with_metadata.csv');sp=sp[sp['縁除外画素']==0]
    injection=table(OUT/'round4_injection.csv')
    injection=injection[(injection['帯軸度']==40)&(injection['周期画素']==400)&(injection['目標追加割合']==.03)]
    rows=[]
    for _,r in comparison.iterrows():
        q=joined[(joined['定義']==r['定義'])&(joined['仮説']==r['仮説'])]
        label=r['対象']
        if label=='未測定':q=q.iloc[:0]
        elif label=='固定29利用可能':q=q[q['固定29視野']]
        elif label=='ブランク':q=q[q['ブランク']]
        elif label=='定義別3%以上':q=q[q.key.isin(fields[(fields['定義']==r['定義'])&fields['この定義3percent以上']].key)]
        elif label=='帯状候補':q=q[q.key.isin(sp[(sp['定義']==r['定義'])&(sp['分類']=='帯状候補')].key)]
        elif label.startswith('日程'):q=q[q['日程']==label[2:]]
        elif label!='全利用可能':raise ValueError(label)
        # Preserve the original summary; audit the matched per-field values.
        row=r.to_dict();row.update(天井一致視野数=len(q),同一集合半数天井中央値=q['双方向平均決定係数'].median(),同一集合全数信号割合近似中央値=q['全数信号割合近似'].median(),天井で割った値=np.nan,全数近似で割った値=np.nan,視野別天井比中央値=q['視野別天井比'].median(),天井正値視野数=int((q['双方向平均決定係数']>0).sum()))
        if len(q):
            assert len(q)==int(r['対象視野数']),(r['仮説'],label,len(q),r['対象視野数'])
            assert abs(q['全面決定係数'].median()-r['説明率中央値'])<1e-8,(r['仮説'],label,q['全面決定係数'].median(),r['説明率中央値'])
            denominator=row['同一集合半数天井中央値'];full=row['同一集合全数信号割合近似中央値']
            if denominator>0:row['天井で割った値']=r['説明率中央値']/denominator
            if full>0:row['全数近似で割った値']=r['説明率中央値']/full
            row['天井比較状態']='半数天井に対する記述比' if denominator>0 else '天井中央値が非正、比未定義'
        else:row['天井比較状態']='共通尺度の仮説説明率未測定'
        for kind in ['判定前','判定後']:
            calibration=injection[(injection['定義']==r['定義'])&(injection['注入法']==kind)&injection.key.isin(q.key)]
            row['同一集合3ポイント人工帯'+kind+'中央値']=calibration['全面決定係数'].median()
        if r['仮説'] in ['D','E']:row['判定']='単独像の必要条件比較を実施、原因分離は判定不能';row['確度']='低（因果）'
        rows.append(row)
    csv(rows,'round4_all_hypotheses_comparison.csv')
    # Balanced across hypotheses with the same five recovered fixed fields where possible.
    main=joined[joined['固定29視野']&joined['仮説'].isin(['B','G','I','H','C','G（方向を限定しない二次面）'])];balanced=[]
    for method,g in main.groupby('定義'):
        sets=[set(q.key) for _,q in g.groupby('仮説')];keys=set.intersection(*sets)
        for model,q in g[g.key.isin(keys)].groupby('仮説'):
            ceiling=q['双方向平均決定係数'].median();balanced.append(dict(定義=method,仮説=model,視野集合=';'.join(sorted(keys)),視野数=len(q),説明率中央値=q['全面決定係数'].median(),天井中央値=ceiling,天井で割った値=q['全面決定係数'].median()/ceiling if ceiling>0 else np.nan))
    csv(balanced,'round4_balanced_hypotheses.csv')

def validate():
    verify();noise=table(OUT/'round4_noise_ceiling.csv');inj=table(OUT/'round4_injection.csv');im=table(OUT/'round4_single_image_spectrum.csv');at=table(OUT/'round4_attribution_by_field.csv');t=table(OUT/'round4_tests.csv')
    assert len(noise)==632*2*2*2 and noise.key.nunique()==632
    assert len(inj)==632*2*4*7 and inj.key.nunique()==632
    assert len(im)==286*2+6 and len(at)==286*2 and len(t)==24
    before=inj[inj['注入法']=='判定前'];after=inj[inj['注入法']=='判定後']
    assert before['目標差'].abs().max()<1e-4
    assert np.isfinite(noise['全面決定係数']).all() and np.isfinite(inj['全面決定係数']).all()
    assert not inj['飽和目標不能'].fillna(False).any()
    leakage=0
    for tr,te in folds(np.ones((64,64),bool)):
        assert not set(tr)&set(te)
        ty,tx=np.unravel_index(te,(64,64));a=np.zeros((64,64),int);a.ravel()[tr]=1
        for dy in range(-2,3):
            for dx in range(-2,3):
                v=(ty+dy>=0)&(ty+dy<64)&(tx+dx>=0)&(tx+dx<64)
                leakage+=int(a[ty[v]+dy,tx[v]+dx].sum())
    assert leakage==0
    audit=table(OUT/'round4_input_integrity_final.csv');assert audit['不変'].all()
    # Independent direct least-square evaluation of a stored reliability result.
    row=noise.iloc[0];keys=table(OLD/'round1_fields_both_definitions.csv').query('定義 == @METHODS[0]').key.tolist()
    with np.load(OUT/'split_maps.npz') as z:a,b=z['residual'][keys.index(row.key),METHODS.index(row['定義']),0].astype(float)
    valid=np.isfinite(a)&np.isfinite(b);SSE=BASE=0.
    for tr,te in folds(valid):
        x=a.ravel();y=b.ravel();mu=x[tr].mean();sd=x[tr].std();sd=sd if sd>1e-12 else 1.
        design=(x[tr]-mu)/sd;ym=y[tr].mean();coef=np.dot(design,y[tr]-ym)/(np.dot(design,design)+1)
        pred=ym+(x[te]-mu)/sd*coef;SSE+=np.sum((y[te]-pred)**2);BASE+=np.sum((y[te]-ym)**2)
    independent=1-SSE/BASE;assert abs(independent-row['全面決定係数'])<1e-10
    dump(dict(較正視野数=632,二分割評価行数=len(noise),注入評価行数=len(inj),単独像数=len(im),前後比較行数=len(at),検定数=len(t),判定前目標最大絶対差=float(before['目標差'].abs().max()),判定後目標最大絶対差=float(after['目標差'].abs().max()),独立算式決定係数差=float(independent-row['全面決定係数']),学習保留余白重複数=leakage,入力旧版指紋照合数=len(audit),入力旧版不変=True,未完了計算='なし'),OUT/'round4_verification.json')
    print('verification passed',flush=True)

def md(a,percent=(),digits=4):
    a=a.copy()
    for col in a:
        if col in percent:a[col]=a[col].map(lambda x:f'{100*x:.{digits}f}%' if pd.notna(x) else '未定義')
        elif pd.api.types.is_float_dtype(a[col]):a[col]=a[col].map(lambda x:f'{x:.{digits}f}' if pd.notna(x) else '未測定')
    return '| '+' | '.join(a.columns)+' |\n| '+' | '.join(['---']*len(a.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in a.itertuples(index=False,name=None))

def figures():
    import os
    os.environ['MPLCONFIGDIR']=str(OUT/'plot_settings')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('C:/Windows/Fonts/meiryo.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    s=table(OUT/'round4_injection_summary.csv');s=s[(s['指標']=='全面決定係数')&(s['対象']=='固定29利用可能')&(s['帯軸度']==40)&(s['周期画素']==400)]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,method in zip(axes,METHODS):
        for kind in ['判定前','判定後']:
            g=s[(s['定義']==method)&(s['注入法']==kind)].sort_values('目標追加割合');x=g['目標追加割合']*100
            ax.plot(x,g['中央値']*100,'o-',label=kind);ax.fill_between(x,g['第1四分位']*100,g['第3四分位']*100,alpha=.15)
        ax.set_title(method);ax.set_xlabel('外れ値割合の追加（百分率ポイント）');ax.set_ylabel('説明率（百分率）');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('既知の人工帯：軸40度、周期400ピクセル、固定29視野');fig.savefig(folder/'人工帯較正.png',dpi=150);plt.close(fig)
    key='260926_3_3';images=table(OUT/'round4_single_image_spectrum.csv');at=table(OUT/'round4_attribution_by_field.csv');q=at[(at.key==key)&(at['定義']==METHODS[0])].iloc[0]
    with np.load(OLD/'density_maps.npz') as z:density=z['density'][list(z['keys']).index(key),0]
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    from field_round4_methods import analyze_map
    arrays=[]
    for stage in ['洗浄前','洗浄後']:
        with np.load(OUT/'single_images'/(key+'_'+stage+'.npz')) as z:arrays.append((stage,z['brightness'],z['autocorrelation'],z['spectrum']))
    _,ac,_,power=analyze_map(density,rng_for('figure'));arrays.append(('外れ値密度',density,ac,power))
    for row,(label,a,ac,power) in enumerate(arrays):
        h=axes[row,0].imshow(a,cmap='magma');fig.colorbar(h,ax=axes[row,0],shrink=.6);axes[row,0].set_title(label+'・32ピクセル升目')
        axes[row,1].imshow(ac,cmap='coolwarm',vmin=-1,vmax=1);axes[row,1].set_title('二次元自己相関')
        axes[row,2].imshow(np.log10(power+1e-15),cmap='viridis');axes[row,2].set_title('窓関数つきスペクトル')
    fig.suptitle(key+'：各地図・自己相関・スペクトル（原因は未確定）');fig.savefig(folder/'単独像と外れ値の比較.png',dpi=130);plt.close(fig)

from pathlib import Path

def report():
    verify();pre=json.loads((OUT/'round4_preflight.json').read_text(encoding='utf-8'));v=json.loads((OUT/'round4_verification.json').read_text(encoding='utf-8'))
    noise=table(OUT/'round4_noise_summary.csv');inj=table(OUT/'round4_injection_summary.csv');tests=table(OUT/'round4_tests.csv');at=table(OUT/'round4_attribution_summary.csv');images=table(OUT/'round4_single_image_summary.csv');comp=table(OUT/'round4_all_hypotheses_comparison.csv');balanced=table(OUT/'round4_balanced_hypotheses.csv');coverage=table(OUT/'round4_date_coverage.csv')
    main_noise=noise[(noise['分割']=='格子行偶奇')&noise['対象'].isin(['全利用可能','固定29利用可能'])]
    main_inj=inj[(inj['指標']=='全面決定係数')&inj['対象'].isin(['全利用可能','固定29利用可能'])&(inj['注入法']!='注入なし')]
    main_at=at[(at['ブランク区分']=='全区分')&(at['分布規則']=='周期候補')&at['対象'].isin(['全利用可能','固定29利用可能'])]
    day_at=at[(at['ブランク区分'].isin(['ブランク','ブランク以外']))&(at['分布規則']=='周期候補')&at['対象'].str.startswith('日程')]
    threshold=table(OLD/'round1_thresholds.csv');prediction=pd.read_csv(OUT/'prediction_sha256.csv').iloc[0]
    # Concrete conclusions computed from results, keeping causal interpretation separate.
    fixednoise=main_noise[(main_noise['対象']=='固定29利用可能')&(main_noise['指標']=='双方向平均決定係数')]
    strong=main_inj[(main_inj['対象']=='固定29利用可能')&(main_inj['目標追加割合']==.10)]
    significant=tests[(tests['ボンフェローニ補正後']<.05)&tests['検定'].str.startswith('洗浄')]
    git=dict(作業ブランチ=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),開始コミット=pre['開始コミット'],変更状態=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf-8').strip(),未追跡ファイル=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True,encoding='utf-8').splitlines(),コミットタグプッシュ='実行していない')
    dump(git,OUT/'round4_git_state.json')
    samecal=comp[comp['対象']=='固定29利用可能']
    lines=['# 2026年10月4日・周4報告：説明率の天井較正と洗浄前後の帰属',
    '\n## 結論',
    f"固定29視野の測定内二分割天井は約78〜83%と高い一方、既知の人工帯で外れ値割合を3百分率ポイント増やしても、全面説明率の中央値は0.58〜2.59%だった。10百分率ポイントでは{strong['中央値'].min()*100:.2f}〜{strong['中央値'].max()*100:.2f}%（四つの向き・周期、二注入法、二定義）。『本物の帯を正しく知っていても数%しか出ない』条件を今回実際に確認した（確度：高、注入条件に限定）。過去の低い説明率だけから、弱い帯の原因がないとは結論できない。再現できる構造全体の天井が低いという説明は支持されず、既存構造を含む総残差分散に対して注入帯が占める割合と、形の非線形な閾値応答を区別する必要がある。",
    '半数ピラーの同一測定内の天井と、全数地図の独立反復測定の天井は異なる。以下の値と天井比は尺度の記述較正であり、原因割合・除外可能な外れ値割合ではない。信頼性の確度：中（測定内分割という制限）。',
    md(fixednoise[['定義','視野数','中央値','第1四分位','第3四分位']],percent=['中央値','第1四分位','第3四分位']),
    f"単独像と外れ値帯の一致を24検定の補正で評価した。有意となった洗浄前後のスペクトル照合は{len(significant)}件。必要条件の一致があっても蒸着・成形・照明・標本化・位置合わせ・洗浄・吸着の原因は分離できない。仮説D（金の蒸着の不均一）・E（鋳型の成形の不均一）はどちらも原因として判定不能（確度：低）。スペクトル照合の数値の確度：高。詳細は下記の全検定と日程別分類。",
    '前後像と260830の合計578単独像のうち、第一軸の自己相関とスペクトル周期が一致したのは1像だけだった。単独像の多くは探索範囲上限1024ピクセルのスペクトルが第一ピークで、周期支持はない。外れ値帯との同時一致は0件だが、これは今回の画素平均・第一ピークの方法で帰属を判定できないことを強く示す。蒸着・成形の帯が存在しない証拠とは扱わない。',
    '\n## 確認した入力と実在パス（事実）',
    '規約と開発環境記録を読み、最新の入力限定指示を優先した。上位を一覧して名前を確認し、入力コピーのみ使用。元ラボノートの10_引き継ぎ・05_解析の階層はコピーでは平坦化されているが、関連ノート8本を読んだ。原記録に記載がないことを欠陥なしの根拠にしていない。',
    '\n'.join(['- 入力起点：`'+pre['入力起点']+'`','- 過去解析：`'+pre['解析コピー']+'`']+['- 生画像：`'+p+'`' for p in pre['生画像フォルダ']]+['- 完了印：`'+p+'`' for p in pre['完了印']]),
    md(coverage),
    f"保存差632視野・固定29視野を二定義で処理。洗浄前後の生画像は286組、260830の単独像6枚。利用可能生画像日程は260828・260922・260924・260926と260830。生画像のない260825・260827・260829・260923・260927は差と較正のみ利用し、単独像照合は未実施。追加照合にはこれら五日程の生画像が必要。基板01の意味は日程別の周1台帳を保持し、260922・260923の01をブランクに変更していない。",
    '全使用ファイルの実在絶対パス・開始終了内容指紋：`round4_input_integrity.csv`、`round4_input_integrity_final.csv`。保存差と前後ファイルの対応：`round4_field_manifest.csv`。260830実在名：`round4_control_manifest.csv`。保存差は過去解析の`tables/cached_field_differences`、識別表は`tables/table_registration_field_qc.csv`。計算済み成果を新旧判定で取り込む操作は行っていない。開始終了の内容指紋一致で旧版の不変も確認した。',
    '\n## 保存済み予測と条件',
    f"予測本文：`261004_周4_予測と較正規定.md`。保存時の暗号学的内容指紋（SHA-256、内容同一性を確認する値）：`{prediction.Hash}`。計算の各段階で照合。実装コピー元の関数名と指紋：`method_copy_provenance.json`。",
    '予測は、再現する実帯があれば二分割間で予測性能が上がり、既知帯は強さとともに説明率が上がること。蒸着・成形なら洗浄前単独像に同じ第一軸と周期が現れ得ること。洗浄・乾燥なら洗浄後だけまたは差に現れ得ること。ブランク・分子なしでも装置・基板構造は出得ること。条件・解析式・対照・検定群を先に固定した。',
    '32ピクセル升目、対象を除く全視野共通平均、8×8升目ブロックの8分割交差検証、評価周囲2升目除外、固定強さ1の正則化回帰、学習平均を基準とする保留二乗誤差。注入後も元の日程別ブランク閾値を固定。二定義の閾値は次のとおり。',
    md(threshold[['日程','ブランク視野数','平均標準偏差','中央値絶対偏差閾値']],digits=6),
    '\n## 作業1(a)：実データの信頼性の天井',
    '主分割は格子識別番号第1成分の偶奇、市松は二成分の和の偶奇。二方向の回帰を行い、その決定係数を平均。各半分の密度から、その半分における他視野共通平均を引く。元の全数密度との再現、両組の有効ピラー数の和を確認。',
    md(main_noise,percent=['中央値','第1四分位','第3四分位']),
    '全数信号割合近似は、二組の共分散C、分散平均Sから2C/(S+C)で算出。独立で等分散の雑音を仮定する補助値で、交差検証による決定係数とは別物。負値・1超えを切り詰めていない。半数天井は低密度で過小評価し得る。格子位相に依存する系統差、近傍ピラー間の依存、前後像を共有する雑音を反復撮影の雑音とは区別できない。',
    '市松・日程・ブランク別の結果、視野別双方向と九つの空間対照は`round4_noise_summary.csv`、`round4_noise_ceiling.csv`。',
    '\n## 作業1(b)：既知の人工帯による較正',
    '帯軸・周期は0度/400、40度/400、80度/640、40度/256ピクセル。位相0の正弦帯。強度は外れ値割合を1・3・10百分率ポイント増やす目標。判定前では保存差へ局所輝度変化を加えて再判定。判定後では非陽性ピラーの確率を帯に従って増加。期待値と実現値の違いを記録。既知の帯形を升目平均した一つの説明変数で予測した。注入なしの同形モデルも保存。対象だけへ注入し、共通平均は元の他視野から固定。',
    md(main_inj[['定義','注入法','帯軸度','周期画素','目標追加割合','対象','視野数','中央値','第1四分位','第3四分位']],percent=['目標追加割合','中央値','第1四分位','第3四分位']),
    f"判定前の目標増分との最大差は{v['判定前目標最大絶対差']*100:.6f}百分率ポイント。判定後の最大差は{v['判定後目標最大絶対差']*100:.6f}百分率ポイント。飽和で目標未達の条件はなし。振幅・実現増分・空間対照・注入なしの視野別結果は`round4_injection.csv`、日程とブランクを含む集計は`round4_injection_summary.csv`。",
    '検証時の訂正：判定後の確率注入で、実現増分が目標より0.2百分率ポイント超下振れした113条件を、初期実装で飽和と誤記した。非陽性ピラーの総割合が目標より小さい場合だけを飽和とするよう状態欄を訂正。113条件は全て標本変動による下振れで、決定係数・振幅・実現増分は変更していない。訂正前後の値と内容指紋は`round4_injection_flag_correction.csv`。検証を再実行して全条件を確認した。',
    '![人工帯と説明率](figures/人工帯較正.png)',
    '空間対照も同じ全面ではなく、共通して有効な中央領域で比較する。周期帯の平行移動は元帯と相関し得るため、対照で正の値が出ても失敗ではない。判定前の閾値非線形性・元の差の局所分布のため、注入形を知る回帰でも1へ達するとは限らない。帯幅、周波数の広がり、交差する二方向帯の較正は未実施で、今回の一方向帯の条件外は分からない。',
    '\n## 作業2：洗浄前後のスペクトル形の比較',
    '各単独像を32ピクセル升目で平均し、周1と同じ平均除去・ハニング窓・128〜1024ピクセル・10度方向探索・1000回升目並替分類を実行。周期の支持は第一軸に直交する投影の自己相関とスペクトル周期が20%以内の一致。照合は第一軸15度以内、第一スペクトル周期20%以内で固定。第一軸に限定し、うろこ状の第二軸へ結果に応じて切替えていない。',
    '分母は生画像のある外れ値周期候補。単独像で周期が支持されない例も不一致に数える。『前だけ・後だけ・両方・どちらでもない』は方向と周期が同時一致する分類。前後像それぞれ固有の座標で比較し、独立対応失敗による視野除外を導入していない。',
    md(main_at[['定義','対象','生画像利用可能視野数','外れ値周期対象視野数','前だけ数','後だけ数','両方数','どちらでもない数','洗浄前同時一致割合','洗浄後同時一致割合']],percent=['洗浄前同時一致割合','洗浄後同時一致割合']),
    md(day_at[['定義','対象','ブランク区分','外れ値周期対象視野数','前だけ数','後だけ数','両方数','どちらでもない数','洗浄前同時一致割合','洗浄後同時一致割合']],percent=['洗浄前同時一致割合','洗浄後同時一致割合']),
    '周期候補より厳しい帯状・うろこ状のみの感度、方向だけ・周期だけの一致、円周統計は`round4_attribution_summary.csv`。各像の第一軸、周期、自己相関支持、角度差、周期相対差は`round4_attribution_by_field.csv`。地図・二次元自己相関・スペクトルは`single_images/`の各視野・時点別保存配列。',
    '![単独像と外れ値](figures/単独像と外れ値の比較.png)',
    '\n## ブランク・分子なし単独像',
    md(images),
    '260830は実在フォルダ`260830_p50＿同一視野にてsam`の6像を同じ分類で処理。前後対応と単独像の撮影段階を確認できないので『前後未確認』とし、差との一致検定や洗浄前後の帰属は行っていない。分子を付けていないという依頼上の意味と、フォルダ名の処理記載だけでは、撮影時点を確定できない。',
    '\n## 視野単位の検定（補正前後を全て掲載）',
    '二分割の実地図−空間対照差は視野ごとの符号並替。単独像照合の円周統計はcos(2×角度差)の視野平均、周期支持のない単独像は寄与0。同時一致割合と円周統計の対照は日程内で視野を10000回並替。洗浄後−前は視野ごとの前後ラベル交換。二定義・二対象・二時点の16比較、前後差4比較、較正4比較の合計24検定でボンフェローニ法24倍。基板内並替・基板一括符号交換の感度は同表。単独像と差を共有するため、統計的関連を因果の独立証拠にしない。',
    md(tests),
    '\n## 解釈、残る説明、弱点',
    '仮説記号の対応：A＝書き込み境界と視野グリッド、B＝画素格子と六方格子のうなり、C＝洗浄水・ブロワー・乾燥前線、D＝金の蒸着、E＝鋳型成形、F＝局所位置合わせずれ、G＝ピント・照明・周辺減光、H＝ゴミ・シミ・傷、I＝撮影順序と前後撮影条件、J＝実際の分子吸着の不均一。共同モデルは説明変数を連結したもの、前像対照は洗浄前像から作った同種の説明変数、方向を限定しない二次面は横縦の一次・二次・積を使う滑らかな地図を意味する。',
    '事実：実データの二分割天井と、人工帯の強度に対する応答を定量化した。単独像に周期が支持されるか、外れ値と第一軸・周期が一致するかを視野ごとに確認した。過去の説明率と同一視野集合の天井を結合し、元の集計値と視野数の再現を検証した（確度：高）。',
    '解釈：洗浄前に一致する帯は蒸着・成形の必要条件に整合し得るが、照明・格子と画素のうなり・既存吸着も同じ結果を作り得る。洗浄後だけの一致も、洗浄・乾燥を一意に示さず、ピント・照明・標本化の違いを排除できない。どちらでもない場合も、差を取る非線形な閾値によって弱い成分の帯が強調される可能性がある。DとEの個別判定は判定不能（確度：低）。',
    '較正からの読み直し：Bの負値やCの負値は今回の予測地図が保留升目の学習平均を改善しない事実を示すが、仮説全体の不存在は証明しない。Hの小さな正値、滑らかな二次面の約3〜4%、画像モデルG・Iの数%も天井全体のわずかな部分だが、既知3ポイント帯も数%になるため、これだけで弱い帯の物理原因を排除できない。Fは独立検証2視野、固定群1視野に限られ一般化できない。画像を共有するG・I・Hの因果の限界と、Cの操作記録不足は較正後にも残る。',
    '同じ視野集合での過去モデルと既知3ポイント帯（軸40度・周期400ピクセル）の比較を次に示す。既存の説明率は保存値を再掲し、較正の母集団だけを仮説ごとの利用可能視野へ揃えた。人工帯の大きさと形は仮説の実際の原因強度を推定した値ではない。',
    md(samecal[['仮説','定義','対象視野数','説明率中央値','同一集合3ポイント人工帯判定前中央値','同一集合3ポイント人工帯判定後中央値']],percent=['説明率中央値','同一集合3ポイント人工帯判定前中央値','同一集合3ポイント人工帯判定後中央値']),
    '尺度の弱点：密度が疎な視野では32ピクセル升目の計数雑音が大きい。半数のピラーを使う天井は全数地図より低くなる。天井に対する比が1を超えても、因果割合100%超えを意味しない。全数近似は等分散・独立雑音の仮定に依存する。視野間は基板内相関と共通閾値を共有する。空間対照共通領域は中央へ狭まり、元の全面の天井と異なる。過去の仮説は各々の有効升目と利用可能視野が異なるので、同じ視野の比も完全な同一領域比較ではない。',
    '帰属の弱点：単独像地図は画素平均で、ピラー頂点だけの輝度ではない。局所位相の帯が平均で弱まる可能性がある。周1と同じくスペクトルは有限地図の周波数格子に量子化され、整数調波だけを細い帯の証拠にしていない。微小な前後回転と倍率差の補正なし、第一軸のみの比較、固定29の少数視野、別日程の生画像不足、260830の前後不明が残る。方向・周期の一致は画像共有による循環を完全に解消する独立測定ではない。',
    '\n## 実装・検証・保存と作業状態',
    md(pd.DataFrame([v]).T.reset_index().rename(columns={'index':'確認項目',0:'結果'})),
    f"使用版：`field_level/v31_band_origin_round4/`、結果：`data/results/v31_band_origin_round4/`。開始時、同名フォルダ・ブランチ・タグの使用を確認し未使用だった。このクローン外の他作業者の未公開状態は確認していない。開始ブランチ：`{git['作業ブランチ']}`、開始コミット：`{git['開始コミット']}`。新規ファイルはこの二つの許可フォルダだけ。旧版・入力6995ファイルの開始終了内容指紋一致。計算の未完了なし、取得できなかった別日程像と独立表面測定は上記のとおり。",
    '再実行順：`field_round4_analysis.py init`、`splits`、`pilot`、`noise`、`injection`、`images`、`field_round4_summary.py summarize`、`attribution`、`tests`、`reread`、`field_round4_analysis.py integrity`、`field_round4_summary.py validate`、`figures`、`report`。途中結果は視野ごとの保存ファイルから再開。再実行前には出力先を独立保存し、予測指紋を保持する。',
    '新しい空の結果フォルダへ再実行する場合は、事前規定文書とその指紋記録を先に置く。原出力の再実行は上書きになるため検収前は行わない。状態欄を訂正する`repair_flags`は今回の訂正記録用で、新しい実装の通常再実行には不要。独立算式との検証では、保存配列の単精度を倍精度へ変換して回帰側と同じ精度で計算し、結果値・許容差を変更せず一致を確認した。',
    '未追跡ファイル：\n\n```text\n'+'\n'.join(git['未追跡ファイル'])+'\n```\n\n短い変更状態：\n\n```text\n'+git['変更状態']+'\n```\n\nコミット・タグ・プッシュは実行していない。生成画像と大容量保存配列をGitへ追加していない。',
    '\n## 次に検証すべき仮説の順序と理由',
    '1. 仮説F（局所位置合わせずれ）：独立検証済み残差が2視野しかなく、低説明率で仮説全体を否定できない。非周期の対応目印・別測定で対応の正解を増やし、格子位相を固定した較正帯との比較を行う。必要な生画像は260825・260827・260829・260923・260927。',
    '2. 仮説D・E（蒸着・成形）：今回、単独像の第一ピークが緩やかな輝度構造に支配され帰属判定ができなかった。次の周の予測を固定した上でピラー頂点輝度と照明成分を分ける測定・方法を用意し、独立な表面・形状・蒸着厚みの測定で二つを区別する。今回の規則を結果に合わせて変えて一致を作っていない。',
    '3. 仮説C・H（洗浄・乾燥とゴミ・シミ・傷）：水流・風向・基板の回転を事前に記録・操作したブランク・分子なしで、方向が追従するか検証する。自動候補は目視承認後に用途を区別し、標準除外へ自動採用しない。',
    '4. 仮説B・G・I（うなり・光学条件・撮影条件）：今回の一方向帯の較正では未評価の二方向帯・位相帯とピラー頂点標本化の較正を追加し、独立反復撮影による全数地図の天井を測る。',
    '5. 仮説J（実際の不均一吸着）：濃度・洗浄順と基板番号の交絡を解いた無作為化実験で検証する。較正値の大きさだけで分子に帰属させない。次の周へ進まず、本報告で停止する。',
    '\n## ここまでの全仮説の説明率一覧（同一視野集合の天井欄付き）',
    '周3で検収した全行を保持し、天井が正の中央値を持つ場合だけ「仮説説明率中央値／同じ視野集合の半数天井中央値」を追加した。負の説明率は保持し、未測定は未測定。補助全数信号割合による比と視野別比は保存表にある。D・Eは今回スペクトル照合のみで、共通尺度の回帰説明率は未測定のまま。',
    md(comp[['仮説','定義','対象','対象視野数','説明率中央値','第1四分位','第3四分位','同一集合半数天井中央値','天井で割った値','天井比較状態','判定']],percent=['説明率中央値','第1四分位','第3四分位','同一集合半数天井中央値']),
    '\n同じ回復済み固定視野に揃えた比較（独立検証2視野のFはこの集合へ拡張しない）：\n',
    md(balanced,percent=['説明率中央値','天井中央値']),
    '天井比は原因判定を自動で覆す基準ではない。仮説ごとの同一集合・未定義・少数群を残し、未取得の独立データは分からないと記録した。']
    (OUT/'261004_周4_報告.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('report saved',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['summarize','attribution','tests','reread','validate','figures','report']);a=p.parse_args();globals()[a.stage]()
