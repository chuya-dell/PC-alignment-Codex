"""Additional completed raw copies, with initial results preserved."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from field_round2_analysis import OUT,OLD,LOCAL,METHODS,table,csv,digest,verify,discover,raw,time_features,fields,association,field_cv

def init():
    verify();children,source,tables,root,roots=discover()
    markers=[p for n,p in roots.items() if p.is_file() and 'copy_done_2' in n]
    assert len(markers)>0
    a=table(OUT/'round2_field_files_times.csv');used=set(markers)
    actual={n:{p.name:p for p in folder.iterdir() if p.is_file()} for n,folder in roots.items() if folder.is_dir() and n.startswith(('260828','260922','260924'))}
    caches={p.name:p for p in tables['cached_field_differences'].iterdir() if p.is_file()}
    for j,r in a[a['日程'].isin(['260828','260922','260924'])].iterrows():
        names=[n for n in actual if n.startswith(r['日程']) and r['ファイル'] in actual[n]]
        if r.get('フォルダ') in names:names=[r['フォルダ']]
        assert len(names)==1, (r.key,names)
        p=actual[names[0]][r['ファイル']]
        assert p.stat().st_size==r['元バイト数']
        a.loc[j,'生画像パス']=str(p);a.loc[j,'回復']=False
        used.add(p);used.add(caches[r.key+'.npz'])
    csv(a,'round2_field_files_times_completed.csv')
    csv([dict(実在パス=str(p),バイト数=p.stat().st_size,開始内容指紋=digest(p)) for p in sorted(used)],'round2_completion_input_integrity.csv')
    (OUT/'round2_completion_preflight.json').write_text(json.dumps(dict(追加生画像フォルダ=[str(roots[n]) for n in actual],完了印=[str(p) for p in markers],追加組数=int(a[a['日程'].isin(['260828','260922','260924'])].key.nunique()),使用追加ファイル数=len(used)),ensure_ascii=False,indent=2),encoding='utf-8')
    print('completion preflight',len(used),flush=True)

def pilot():
    # raw(limit=2) sees the two earliest newly enumerated 260828 pairs.
    raw(limit=2)

def tests():
    verify();timea=time_features();f=fields();a=timea.merge(f[['key','定義','外れ値割合']],on='key')
    tc=['洗浄前順位','洗浄後順位','撮影間隔時間'];ch=[k+'前後対数比' for k in ('平均輝度','コントラスト','鮮明さ')]
    extra=[]
    for subset in ('全4日程','追加3日程'):
        aa=a if subset=='全4日程' else a[a['日程']!='260926']
        for m in METHODS:
            s=aa[aa['定義']==m]
            for left,right,label in [(tc,['外れ値割合'],'時刻と外れ値割合'),(ch,['外れ値割合'],'画像条件変化と外れ値割合'),(tc,ch,'時刻と画像条件変化')]:
                r=association(s,left,right,'追加完了_'+subset+'_'+label,m);r['検定群']='追加完了後12';extra.append(r)
                print('completion test',subset,label,m,flush=True)
    csv(extra,'round2_completion_tests.csv')
    initial=table(OUT/'round2_confirmatory_tests_initial.csv');initial['検定群']='初回20'
    alltests=pd.concat([initial,pd.DataFrame(extra)],ignore_index=True)
    alltests['初回20比較補正']=alltests['ボンフェローニ補正後']
    alltests['ボンフェローニ補正後']=np.minimum(1,32*alltests['補正前有意確率'])
    assert len(alltests)==32;csv(alltests,'round2_confirmatory_tests.csv')
    cvrows=[];predrows=[]
    for subset in ('全4日程','追加3日程'):
        aa=a if subset=='全4日程' else a[a['日程']!='260926']
        for m in METHODS:
            s=aa[aa['定義']==m];gc=['洗浄前コントラスト','洗浄前鮮明さ','洗浄前周辺中央比']
            ss=s.dropna(subset=gc+ch+tc).reset_index(drop=True);base=field_cv(ss,[]);yy=ss['外れ値割合'].to_numpy();bsse=((yy-base)**2).sum();total=((yy-yy.mean())**2).sum();rows=[]
            for name,features in {'G':gc,'I':ch+tc,'共同':gc+ch+tc,'I時刻':tc}.items():
                pred=field_cv(ss,features);sse=((yy-pred)**2).sum();row=dict(定義=m,対象='追加完了_'+subset,モデル=name,視野数=len(ss),基準追加決定係数=1-sse/bsse,全分散決定係数=1-sse/total,基準全分散決定係数=1-bsse/total);rows.append(row)
                for j,r in ss.iterrows():predrows.append(dict(key=r.key,定義=m,対象=subset,モデル=name,実測=yy[j],予測=pred[j],基準予測=base[j]))
            d={r['モデル']:r['基準追加決定係数'] for r in rows}
            for r in rows:
                if r['モデル']!='I時刻':r.update(G固有=d['共同']-d['I'],I固有=d['共同']-d['G'],共通=d['G']+d['I']-d['共同'])
            cvrows+=rows
    csv(cvrows,'round2_completion_field_rate_cv.csv');csv(predrows,'round2_completion_field_rate_predictions.csv')
    initialcv=table(OUT/'round2_field_rate_cv_summary_initial.csv');csv(pd.concat([initialcv,pd.DataFrame(cvrows)],ignore_index=True),'round2_field_rate_cv_summary.csv')

def validate():
    verify();g=table(OUT/'round2_geometry_explained_fraction.csv');o=table(OUT/'round2_optical_explained_fraction.csv');im=table(OUT/'round2_global_image_features.csv')
    # Partition labels belong to G/I/joint, not the separate time-only model.
    for name in ('round2_completion_field_rate_cv.csv','round2_field_rate_cv_summary.csv'):
        t=table(OUT/name);t.loc[t['モデル']=='I時刻',['G固有','I固有','共通']]=np.nan;csv(t,name)
    assert len(g)==1264 and len(o)==204 and len(im)==286
    assert im.key.nunique()==286 and im['回復'].sum()==34
    errors=[]
    for domain in ('全面決定係数','対照共通領域決定係数'):
        z=o[o['モデル']=='共同'];errors.append(float(np.max(np.abs(z[domain+'_G固有']+z[domain+'_I固有']+z[domain+'_共通']-z[domain]))))
    assert max(errors)<1e-12
    assert np.isfinite(g['全面決定係数']).all() and np.isfinite(o['全面決定係数']).all()
    assert (g['対照共通升目数']<=g['全面升目数']).all()
    a=table(OUT/'round2_completion_input_integrity.csv');a['終了内容指紋']=[digest(Path(p)) for p in a['実在パス']];a['不変']=a['開始内容指紋']==a['終了内容指紋'];assert a['不変'].all();csv(a,'round2_completion_input_integrity_final.csv')
    initial=table(OUT/'round2_confirmatory_tests_initial.csv');final=table(OUT/'round2_confirmatory_tests.csv')
    assert np.array_equal(initial['補正前有意確率'].to_numpy(),final.iloc[:20]['補正前有意確率'].to_numpy())
    out=dict(二次面条件数=len(g),画像モデル条件数=len(o),原画像組数=len(im),回復組数=int(im['回復'].sum()),共通分解恒等最大差=max(errors),初回未補正有意確率不変=True,追加入力不変数=len(a),追加12検定保存=True)
    (OUT/'round2_completion_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','pilot','tests','validate']);args=p.parse_args();globals()[args.stage]()
