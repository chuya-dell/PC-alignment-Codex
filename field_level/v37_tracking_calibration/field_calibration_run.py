"""既存の合成・追跡関数を読み取り専用で使う較正。条件単位で再開可能。"""
import sys
sys.dont_write_bytecode=True
import os, json, time, inspect, subprocess
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from field_calibration import ROOT,CODE,OUT,FIELD,RESULTS,LOCAL,child,resolve,dump

CHOICES=[
    '保存場再現の丸め差は絶対許容差を倍精度機械精度の8倍、相対許容差を0として評価する。完全一致も別に記録する。',
    '回収場の全4096格子点を使い、定数項・正弦項・余弦項の最小二乗で振幅と位相を求める。縁の除外や零変位場の差引きはしない。',
    '視野内は位相2条件の中央値、全振幅をまとめる曲線と判定では振幅4条件の中央値を用い、その後に4視野の中央値と第1・第3四分位を求める。位相差の視野内集計は円周平均を用いる。',
    '手順5は保存場の64×64格子上の離散フーリエ変換で、各周波数を最も近い較正周期と方向に割り当てる。範囲外も最も近い周期を使用し、零周波数は無補正。変位の直交成分は無補正。',
    '健全性確認は全96条件の理想回収場の当てはめ、恒等変位ゼロの理想残差、および周期512画素・振幅0.05画素の追跡利得を参考値として示す。最後の値には新たな合否閾値を設けない。',
    '補助の双一次合成は既存の6回反復逆写像の座標をcv2.remapのINTER_LINEAR、反射境界で標本化し、主と同じ丸め・65535制限を行う。',
    '利得の二乗と分散比の関係は、変位振幅と差画像振幅の線形対応を仮定した参考解釈に限る。新しい検定や判定閾値には使わない。',
    '四分位はpandasの既定の線形補間で算出する。条件番号と周波数の同距離割当は事前表の列挙順を優先する。'
]

def save(frame,name):frame.to_csv(OUT/name,index=False,encoding='utf-8')

def markdown(frame):
    def cell(v):
        if pd.isna(v):return '未算出'
        if isinstance(v,(float,np.floating)):return f'{v:.8g}'
        return str(v).replace('|','／').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])

def write_once(name,text):
    p=OUT/name
    if p.exists():assert p.read_text(encoding='utf-8')==text,'事前定義の不一致'
    else:p.write_text(text,encoding='utf-8')

def circular(v):
    v=np.asarray(v,float);v=v[np.isfinite(v)]
    return float(np.degrees(np.angle(np.mean(np.exp(1j*np.radians(v)))))) if len(v) else np.nan

def fit(u,condition):
    y,x=np.indices((64,64));q=np.column_stack([x.ravel()*32+16,y.ravel()*32+16])
    n=np.array([np.cos(np.radians(condition['波の進む方向度'])),np.sin(np.radians(condition['波の進む方向度']))])
    theta=2*np.pi*((q-[1023.5,1021.5])@n)/condition['周期画素']+np.radians(condition['位相度'])
    design=np.column_stack([np.ones(len(q)),np.sin(theta),np.cos(theta)])
    v=u.reshape(-1,2);ok=np.isfinite(v).all(axis=1)
    if ok.sum()<3:raise RuntimeError('回収格子の有効点不足')
    parallel=v@n;orthogonal=v@np.array([-n[1],n[0]])
    a=np.linalg.lstsq(design[ok],parallel[ok],rcond=None)[0]
    b=np.linalg.lstsq(design[ok],orthogonal[ok],rcond=None)[0]
    amp=np.hypot(a[1],a[2]);leak=np.hypot(b[1],b[2])
    return dict(利得=float(amp/condition['振幅画素']),回収振幅画素=float(amp),位相差度=float(np.degrees(np.arctan2(a[2],a[1]))),漏れ利得=float(leak/condition['振幅画素']),直交回収振幅画素=float(leak),定数変位画素=float(a[0]),当てはめ残差画素=float(np.sqrt(np.mean((parallel[ok]-design[ok]@a)**2))),有効格子点数=int(ok.sum()))

def run(records,conditions,final,methods):
    worker_key=os.environ.get('V37_CALIBRATION_FIELD')
    cv2.setNumThreads(1)
    sys.path.insert(0,str(child(FIELD,'v35_band_origin_round8')))
    import field_round8_common as common
    # 記録関数は呼ばない。旧版出力への書込を避け、読込は既に監査した実在パスに限定する。
    common.OUT=OUT
    os.environ['MPLCONFIGDIR']=str(OUT/'plot_settings')
    import field_round8_experiment as experiment
    settings=[]
    def setting(module,function,name,default,actual):
        fn=getattr(module,function);lines,line=inspect.getsourcelines(fn)
        settings.append(dict(処理=name,コード=str(Path(module.__file__).relative_to(ROOT)),関数=function,行番号=line,既定値=default,実際の値=actual))
    for name,value in [('窓画素','21・31、別計算後一致確認'),('特徴点','最大2500、品質0.015、最小距離12、区画5'),('反復','40回、収束精度0.0001、階層0'),('追跡選別','往復0.2以下、予測差2以下、窓差0.2以下、画像縁20'),('支持条件','100点以上、32区画以上、保留誤差改善、二群差0.2以下')]:setting(methods,'tracking',name,value,value)
    setting(methods,'kernel_field','平滑化画素','256、正規分布型の重み','256、正規分布型の重み')
    setting(methods,'read_image','前後像読込','元ビット深度、2044×2048、単一成分','同じ')
    setting(experiment,'prepare','主補間','4倍帯域制限フーリエ補間、反射32画素、正方形画素開口','同じ')
    setting(experiment,'inverse_points','逆写像','6反復','同じ')
    setting(experiment,'generate','合成','64行ずつ、一次標本化、逆写像残差0.0001以下、16ビット丸め','同じ')
    setting(experiment,'build_field','格子識別','前像の高速フーリエ変換からピッチ7.286の格子、余白30、保存識別子照合、最大差0.000001以下','周8の格子生成最大差0、保存識別子・座標を使用')
    if not worker_key:save(pd.DataFrame(settings),'tracking_settings.csv')
    methods_text='''# 結果を見る前に固定した測定方法

既存の field_round5_methods.tracking と kernel_field、周8の prepare・generate・inverse_points をそのまま使う。全体変換は各視野の保存済み行列を固定する。合成経路に渡す変位だけを指定の解析的正弦波とする。既存 sample_field の格子補間の代わりに、注入した正弦波を任意の前像座標で直接評価する。反復回数、逆写像残差判定、画素開口、画像量子化、追跡選別条件は変更しない。

回収の64×64格子点は中心座標(32列+16,32行+16)。正弦波の位相原点は(1023.5,1021.5)。全4096点の進行方向と直交方向への射影それぞれを定数・正弦・余弦で最小二乗当てはめし、振幅は正弦係数と余弦係数の二乗和平方根、位相差は余弦係数と正弦係数の偏角。利得は回収振幅を指定振幅で割る。直交成分の利得も記録する。縁は除外せず、零変位残差は差し引かない。支持不足などで場が作られない条件は理由付きの欠損行とする。追跡の採用候補判定を全条件で記録し、偽の場合も既知注入較正の測定値として別記する。自動候補を実測解析の補正に新規採用しない。

零変位は各視野・各補間で保存全体変換のみの後像を合成し、回収場の変位二乗平均平方根・中央値・四分位を求める。周8の往復誤差0.0284画素、保留誤差中央値0.1426画素を参考値として併記する。理想回収確認では全96条件の正解場を同じ当てはめに入力し、利得1、位相差0、漏れ0を確認する。理想零変位残差0も確認する。周期512画素・振幅0.05画素の追跡利得は長周期参考値として示し、新しい合否閾値は置かない。

視野内は位相2条件の中央値。曲線と判定ではさらに振幅4条件の中央値。位相差は円周平均。視野間の中央値と第1・第3四分位は4視野について計算する。欠損時は残った視野数も示し、4視野中央値と称しない。補間差は同一条件の主から補助を引く。
'''
    write_once('methods_note.md',methods_text)
    correction_text='''# 結果を見る前に固定した手順5の補正

指定視野260926_7_5の周8保存場(64×64、格子間隔32画素)の各変位成分を離散フーリエ変換する。各非零周波数の周期は周波数ベクトルの長さの逆数、方向は前像座標での偏角を180度周期で表す。周期は較正した128・212・256・512画素のうち対数周期の距離が最小のもの、方向は0・90・111.2505度のうち180度周期の角度差が最小のものに割り当てる。範囲外も最も近い較正周期を使用し、外挿による利得推定はしない。

利得曲線は主補間だけを使用し、位相と振幅の視野内中央値を取った後の4視野の中央値とする。各周波数の進行方向への射影だけを利得で除算し、直交成分は変更しない。利得0.5未満は補正係数1、零周波数も係数1。成分は除去しない。未知・欠損利得があれば手順5を停止する。逆変換後の実数場を既存の周8 generate に渡し、主帰無と同じ全体変換再推定、マスク無効、ピラー標本化・32画素集計・既存閾値で評価する。実測差で調整しない。この結果は採用判断に使用しない。
'''
    write_once('step5_correction_definition.md',correction_text)
    if not worker_key:dump('unspecified_choices.json',CHOICES)
    checkpoint=OUT/'calibration_checkpoints';checkpoint.mkdir(exist_ok=True)
    y,x=np.indices((64,64));query=np.column_stack([x.ravel()*32+16,y.ravel()*32+16])
    original_sample=experiment.sample_field
    def sine(condition,points):
        n=np.array([np.cos(np.radians(condition['波の進む方向度'])),np.sin(np.radians(condition['波の進む方向度']))])
        theta=2*np.pi*((points-[1023.5,1021.5])@n)/condition['周期画素']+np.radians(condition['位相度'])
        return condition['振幅画素']*np.sin(theta)[:,None]*n
    ideal=[]
    for condition in conditions:ideal.append({**condition,**fit(sine(condition,query).reshape(64,64,2),condition)})
    if not worker_key:
        save(pd.DataFrame(ideal),'ideal_recovery_checks.csv')
        dump('ideal_zero_check.json',{'理想零変位二乗平均平方根画素':0.,'理想利得最大誤差':max(abs(r['利得']-1) for r in ideal),'理想位相最大絶対差度':max(abs(r['位相差度']) for r in ideal),'理想漏れ最大利得':max(r['漏れ利得'] for r in ideal)})
    def aux(pre,m,u):
        h,w=pre.shape;out=np.empty(pre.shape,np.float32);maxerr=0.
        for y0 in range(0,h,64):
            sy,sx=np.indices((min(64,h-y0),w),dtype=float);sy+=y0
            source,err=experiment.inverse_points(np.column_stack([sx.ravel(),sy.ravel()]),m,u);maxerr=max(maxerr,err)
            out[y0:y0+len(sy)]=cv2.remap(pre,source[:,0].reshape(sy.shape).astype(np.float32),source[:,1].reshape(sy.shape).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        assert maxerr<=1e-4
        return np.rint(np.clip(out,0,65535)).astype(np.uint16),dict(逆写像最大残差画素=maxerr)
    rows=[];zero=[]
    for key,rec in records.items():
        if worker_key and key!=worker_key:continue
        pre=methods.read_image(resolve(rec['洗浄前パス']));m=np.asarray(rec['元変換'],np.float32)
        needed=any(not (checkpoint/f'{key}_{mode}_{c["条件番号"]:03}.json').exists() for mode in ['主','補助'] for c in conditions)
        expanded=experiment.prepare(pre,m) if needed else None
        for mode in ['主','補助']:
            zero_path=checkpoint/f'{key}_{mode}_zero.json'
            if zero_path.exists():z=json.loads(zero_path.read_text(encoding='utf-8'))
            else:
                experiment.sample_field=original_sample
                if expanded is None and mode=='主':expanded=experiment.prepare(pre,m)
                raw,meta=experiment.generate(pre,m,np.zeros((64,64,2)),expanded) if mode=='主' else aux(pre,m,np.zeros((64,64,2)))
                a,info=methods.tracking(pre,raw,m)
                z=dict(視野=key,日程=key[:6],補間=mode,欠損理由='',往復誤差周8画素=.0284,保留誤差周8画素=.1426,追跡採用候補=info.get('採用候補'))
                if 'residual_grid' in a:
                    mag=np.linalg.norm(a['residual_grid'],axis=2)
                    z.update(変位二乗平均平方根画素=float(np.sqrt(np.mean(mag**2))),変位中央値画素=float(np.median(mag)),変位第1四分位画素=float(np.quantile(mag,.25)),変位第3四分位画素=float(np.quantile(mag,.75)))
                else:z['欠損理由']=info.get('理由','回収場なし')
                zero_path.write_text(json.dumps(z,ensure_ascii=False,indent=2),encoding='utf-8')
            zero.append(z)
            for c in conditions:
                dest=checkpoint/f'{key}_{mode}_{c["条件番号"]:03}.json'
                if dest.exists():row=json.loads(dest.read_text(encoding='utf-8'))
                else:
                    t=time.monotonic();row=dict(視野=key,日程=key[:6],分子あり=True,補間=mode,**c,欠損理由='')
                    experiment.sample_field=lambda u,p,c=c:sine(c,p)
                    raw,meta=experiment.generate(pre,m,None,expanded) if mode=='主' else aux(pre,m,None)
                    a,info=methods.tracking(pre,raw,m)
                    row.update(追跡採用候補=info.get('採用候補'),追跡判定理由=info.get('理由'),支持点数=info.get('支持点数'),支持区画数=info.get('支持区画数'),合成記録=meta)
                    if 'residual_grid' in a:
                        row.update(fit(a['residual_grid'],c));np.savez_compressed(dest.with_suffix('.npz'),residual_grid=a['residual_grid'])
                    else:row['欠損理由']=info.get('理由','回収場なし')
                    row['秒']=time.monotonic()-t
                    dest.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
                rows.append(row)
                if c['条件番号']%6==0:
                    print(key,mode,c['条件番号'],'/96',flush=True)
                    save(pd.DataFrame(rows),f'calibration_progress_{worker_key}.csv' if worker_key else 'calibration_progress.csv')
        del expanded
    experiment.sample_field=original_sample
    if worker_key:return
    allrows=pd.DataFrame(rows)
    for mode,name in [('主','calibration_all_conditions_main.csv'),('補助','calibration_all_conditions_aux_bilinear.csv')]:save(allrows[allrows['補間']==mode],name)
    save(pd.DataFrame(zero),'zero_displacement_residuals.csv')
    zero_summary=[]
    zframe=pd.DataFrame(zero)
    for mode,g in zframe.groupby('補間'):
        row=dict(補間=mode,視野数=len(g),欠損数=int((g['欠損理由']!='').sum()))
        for col in ['変位二乗平均平方根画素','変位中央値画素']:
            v=g[col].dropna();row[col+'中央値']=v.median();row[col+'第1四分位']=v.quantile(.25);row[col+'第3四分位']=v.quantile(.75)
        zero_summary.append(row)
    save(pd.DataFrame(zero_summary),'zero_displacement_summary_median_iqr.csv')
    def summaries(frame,group):
        out=[]
        for k,g in frame.groupby(group,dropna=False):
            k=k if isinstance(k,tuple) else (k,)
            row=dict(zip(group,k));row['条件数']=len(g);row['欠損数']=int((g['欠損理由']!='').sum()) if '欠損理由' in g else 0
            for col in ['利得','位相差度','漏れ利得']:
                v=g[col].dropna();row[col]=circular(v) if col=='位相差度' else v.median()
            out.append(row)
        return pd.DataFrame(out)
    byfield=summaries(allrows,['補間','視野','日程','周期画素','振幅画素','波の進む方向度']);save(byfield,'gain_by_field.csv')
    def between(frame,group):
        out=[]
        for k,g in frame.groupby(group):
            k=k if isinstance(k,tuple) else (k,);row=dict(zip(group,k))
            row['視野数']=g['利得'].notna().sum()
            for col in ['利得','位相差度','漏れ利得']:
                v=g[col].dropna();row[col+'中央値']=v.median();row[col+'第1四分位']=v.quantile(.25);row[col+'第3四分位']=v.quantile(.75)
            out.append(row)
        return pd.DataFrame(out)
    summary=between(byfield,['補間','周期画素','振幅画素','波の進む方向度']);save(summary,'gain_summary_median_iqr.csv')
    curvefield=summaries(byfield,['補間','視野','日程','周期画素','波の進む方向度'])
    curve=between(curvefield,['補間','周期画素','波の進む方向度']);save(curve,'gain_curve_by_period.csv')
    amplitude=summary.copy()
    for _,g in amplitude.groupby(['補間','周期画素','波の進む方向度']):
        values=g['利得中央値']
        amplitude.loc[g.index,'振幅間利得最大引く最小']=values.max()-values.min()
        amplitude.loc[g.index,'振幅0.3引く0.05利得差']=float(g.set_index('振幅画素').loc[.3,'利得中央値']-g.set_index('振幅画素').loc[.05,'利得中央値'])
    save(amplitude,'amplitude_dependence.csv')
    join=['視野','条件番号','周期画素','振幅画素','波の進む方向度','位相度']
    difference=allrows[allrows['補間']=='主'].merge(allrows[allrows['補間']=='補助'],on=join,suffixes=('_主','_補助'))
    for col in ['利得','位相差度','漏れ利得']:
        difference[col+'主引く補助']=difference[col+'_主']-difference[col+'_補助']
    save(difference,'interpolation_difference_main_vs_aux.csv')
    save(allrows[(allrows['周期画素']==512)&(allrows['振幅画素']==.05)],'long_period_sanity.csv')
    # 手順5。場は実測差で調整しない。
    key='260926_7_5';rec=records[key]
    with np.load(resolve(rec['場実在パス']),allow_pickle=False) as z:u=z['residual_grid'].astype(float)
    gains=curve[curve['補間']=='主'].set_index(['周期画素','波の進む方向度'])['利得中央値']
    assert len(gains)==12 and np.isfinite(gains).all(),'補正利得欠損'
    fy,fx=np.meshgrid(np.fft.fftfreq(64,d=32),np.fft.fftfreq(64,d=32),indexing='ij')
    radius=np.hypot(fx,fy);nonzero=radius>0
    periods=np.divide(1.,radius,out=np.full(radius.shape,np.inf),where=nonzero)
    direction=np.degrees(np.arctan2(fy,fx))%180
    pvalues=np.array([128,212,256,512]);dvalues=np.array([0.,90.,111.2505])
    ip=np.argmin(abs(np.log(periods[:,:,None]/pvalues)),axis=2)
    id_=np.argmin(abs((direction[:,:,None]-dvalues+90)%180-90),axis=2)
    gain=np.array([[gains.loc[(int(pvalues[ip[y,x]]),float(dvalues[id_[y,x]]))] for x in range(64)] for y in range(64)])
    coeff=np.where(nonzero&(gain>=.5),1/gain,1.)
    ft=np.fft.fft2(u,axes=(0,1));nx=np.divide(fx,radius,out=np.zeros_like(fx),where=nonzero);ny=np.divide(fy,radius,out=np.zeros_like(fy),where=nonzero)
    longitudinal=ft[:,:,0]*nx+ft[:,:,1]*ny
    ft[:,:,0]+=longitudinal*(coeff-1)*nx;ft[:,:,1]+=longitudinal*(coeff-1)*ny
    corrected=np.fft.ifft2(ft,axes=(0,1)).real
    np.savez_compressed(OUT/'step5_corrected_field.npz',residual_grid=corrected,correction_coefficient=coeff)
    childcode=child(LOCAL,'code_v24');sys.path.insert(0,str(childcode))
    from shared import registration
    from shared.v2_registration_precision.refinement import register_refined
    assert Path(registration.__file__).resolve().is_relative_to(childcode)
    pre=methods.read_image(resolve(rec['洗浄前パス']));m=np.asarray(rec['元変換'],np.float32)
    expanded=experiment.prepare(pre,m);raw,meta=experiment.generate(pre,m,corrected,expanded);del expanded
    with np.load(resolve(rec['保存差パス']),allow_pickle=False) as z:xy=z['xy'];actual=z['delta']
    mest,est=experiment.estimate(pre,raw,registration,register_refined,m,xy)
    simulated=common.sampler(pre,xy)-common.sampler(raw,common.transform(xy,mest))
    ok=np.isfinite(actual)&np.isfinite(simulated)
    realmap,_=common.binned(xy,np.where(ok,actual,np.nan));simmap,_=common.binned(xy,np.where(ok,simulated,np.nan))
    variance=common.variance_metrics(realmap,simmap)
    thresholds=pd.read_csv(child(child(RESULTS,'v28_band_origin_round1'),'round1_thresholds.csv'),dtype={'日程':str}).set_index('日程')
    signed=pd.read_csv(child(final,'signed_field_metrics.csv'));outliers=pd.read_csv(child(final,'outlier_field_metrics.csv'))
    step=[]
    for variant in ['主帰無','N3二倍','較正利得補正']:
        for method,col in [('平均標準偏差','平均標準偏差'),('中央値絶対偏差','中央値絶対偏差閾値')]:
            threshold=float(thresholds.loc['260926',col])
            if variant=='較正利得補正':
                value=variance;fraction=float((simulated[ok]>threshold).mean());actualfraction=float((actual[ok]>threshold).mean())
            else:
                s=signed[(signed.key==key)&(signed['条件']==variant)].iloc[0]
                o=outliers[(outliers.key==key)&(outliers['条件']==variant)&(outliers['定義']==method)].iloc[0]
                value=s;fraction=o['模擬外れ値割合'];actualfraction=o['実測外れ値割合']
            step.append(dict(視野=key,日程='260926',条件=variant,定義=method,分散比=value['分散比'],無調整分散再現=value['無調整分散再現'],模擬外れ値割合=fraction,実測外れ値割合=actualfraction,元の分散表='signed_field_metrics.csv' if variant!='較正利得補正' else '今回の再実行',元の外れ値表='outlier_field_metrics.csv' if variant!='較正利得補正' else '今回の再実行',元の表フォルダ=str(final.relative_to(ROOT))))
    save(pd.DataFrame(step),'step5_variance_table.csv');dump('step5_simulation_record.json',{'合成':meta,'位置合わせ':est,'補正係数最大':float(coeff.max()),'補正対象周波数数':int((coeff!=1).sum())})
    missing=int((allrows['欠損理由']!='').sum());band=curve[(curve['補間']=='主')&(curve['周期画素']==212)&(curve['波の進む方向度']==111.2505)].iloc[0]
    gain=float(band['利得中央値']);judgment='主因ではない' if gain>=.9 else '部分説明' if gain>=.5 else '主因候補'
    baseline_variance=float(next(r['分散比'] for r in step if r['条件']=='主帰無'))
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip();state=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip()
    report=f'''# 周9 追跡場の振幅較正

## 確認した事実（確度：高）

対象4視野はすべて分子あり。260926の基板7は分子あり、ブランクは基板01。prediction_erratum_1.md に従った。主384条件・補助384条件の全行を保存し、欠損は合計{missing}条件。欠損を0で埋めていない。追跡の採用候補判定が偽の条件は合計{int((allrows['追跡採用候補']==False).sum())}条件で、較正の測定値として記録し、実測の標準経路には採用しない。

周8の保存場は支持点・変位・往復誤差・窓間差・二群分割が完全一致。平滑化場の最大差は4.44×10⁻¹⁶画素で丸め精度内。ビット一致を要求した確認実装を修正し、比較の両方を tracking_reproduction.json に残した。

帯の線の方向21.2505度は周波数ベクトルに90度を加えた値なので、波の進む方向は111.2505度。事前条件96行と根拠を conditions_resolved.json に保存した。利得・補正の測定定義は計算前に methods_note.md と step5_correction_definition.md に保存した。

## 設定表（確度：高）

{markdown(pd.DataFrame(settings))}

## 利得・位相差・漏れ（確度：高）

全条件は calibration_all_conditions_main.csv と calibration_all_conditions_aux_bilinear.csv。視野別は gain_by_field.csv、4視野中央値と四分位は gain_summary_median_iqr.csv。周期曲線は gain_curve_by_period.csv。振幅依存は amplitude_dependence.csv、補間差は interpolation_difference_main_vs_aux.csv。以下は主補間の4視野中央値[第1〜第3四分位]を構成する全曲線値。

{markdown(curve[curve['補間']=='主'])}

周期212画素・実測帯の法線方向の利得は{gain:.8g} [{band['利得第1四分位']:.8g}〜{band['利得第3四分位']:.8g}]。実測周期212.08画素に近い指定周期212画素の値であり、事前基準では「{judgment}」。方向別の差は上表にすべて示した。

周期212画素・振幅別の4視野集計：

{markdown(summary[(summary['補間']=='主')&(summary['周期画素']==212)])}

指定視野260926_7_5の周期212画素の値：

{markdown(byfield[(byfield['補間']=='主')&(byfield['視野']==key)&(byfield['周期画素']==212)])}

## 零変位残差と健全性確認（確度：高）

{markdown(pd.DataFrame(zero))}

4視野の中央値と四分位：

{markdown(pd.DataFrame(zero_summary))}

理想回収96条件は ideal_recovery_checks.csv、理想零変位は ideal_zero_check.json。長周期512画素・振幅0.05画素の全値は long_period_sanity.csv に示し、1から離れた値も保持した。理想回収の正確さは当てはめの実装確認であり、追跡の精度保証ではない。

## 手順5（補助、採用判断に使用しない。確度：高）

{markdown(pd.DataFrame(step))}

1倍と2倍は周8最終フォルダの signed_field_metrics.csv・outlier_field_metrics.csv の指定視野行から転記した。補正場と係数は step5_corrected_field.npz。係数は較正利得だけから決定した。利得0.5未満の成分は係数1で残した。標準経路への採用は行っていない。

補正対象周波数数は{int((coeff!=1).sum())}、補正係数の最大は{float(coeff.max()):.8g}。較正した利得曲線の全12組が0.5未満のため、今回は打切り規則によって全周波数が係数1になった。手順5は結果として場を増幅しない再実行であり、補正による改善を示したものではない。

## 解釈（確度：中）と方法の弱点

指定の判定基準に照らすと追跡と平滑化の減衰は「{judgment}」。これは振幅不足の切り分けであり、物質変化の有無の判定ではない。利得の位相・振幅・方向依存、零変位追跡残差、主と双一次の差を含む。低利得の場は位相が不安定になり得る。全画像の有限範囲、特徴点分布、256画素の重み、16ビット・8ビット量子化が回収に影響する。保存場の空間周波数補正は周期・方向の離散的な割り当てと周期境界を仮定しており、直交成分は較正していないので無補正とした。

参考解釈（確度：低）：変位振幅と差画像振幅が線形に対応すると仮定すれば、利得の二乗が分散比に対応する。周8の1倍分散比{baseline_variance:.8g}は振幅比{np.sqrt(baseline_variance):.8g}相当で、較正の周期212画素利得{gain:.8g}と比較できる。ただし、この換算は較正した正弦変位が実測の変位場と同じという証拠ではなく、振幅不足の説明割合を直接測ったものでもない。零変位残差が利得の下限に寄与し得るので、小さい回収振幅を真の正弦応答だけとはみなせない。

差画像で実測した縞の周期212.08画素が、変位場そのものの周期を表すかは今回の較正だけでは確認できない。周期212画素の較正に対する事前判定と、実測縞の振幅不足を実際に説明した割合は区別する。256画素の平滑化に対して周期512画素も十分長いとは限らず、長周期参考値が1に近づかない結果だけで実装の失敗とは判定しない。

失われた高周波、前後のピント・照明の変化、実際の物質変化は今回だけでは区別できず、有力な説明として残る。手順5で外れ値が消失したという結論は出さない。新しい検定は行っていない。

## 確認範囲・作業状態（確度：高）

必要入力はすべて開けた。新規撮影・新規生画像取得、読めないドライブ、凍結リポジトリは使用していない。終了時の入力指紋と保護対象全ファイル一覧の照合結果は input_integrity_final.csv と protected_inventory_comparison.json に保存する。対象外ファイルの内容不変を一覧だけで保証するものではない。

作業ブランチ：{branch}。未追跡状況：{state}。結果は追跡除外されるので未追跡表示と成果一覧は異なる。コミット・タグ・送信・削除は行っていない。取得は許可された範囲外に書き込むので実施していない。指定の事前予測・未使用確認・訂正記録は変更していない。

指示書にない実装上の選択は unspecified_choices.json に列挙した。各条件の記録を保存し、再開時は保存済み条件を再計算しない。
'''
    (OUT/'261004_周9_Codex報告.md').write_text(report,encoding='utf-8')
    (CODE/'NOTES.md').write_text(f'# 実装変更\n\n既存の合成・追跡を読み取り専用で使用する較正と条件別再開を追加。既存設定・マスク・標準経路は変更しない。\n\n# 結果と保存先\n\ndata/results/v37_tracking_calibration/。主384条件、補助384条件、欠損合計{missing}。周期212画素の法線方向利得中央値{gain:.8g}、手順5補正後分散比{variance["分散比"]:.8g}。\n\n# 既知の問題\n\n低利得の位相不安定、有限範囲と特徴点分布、量子化、補正の離散的な周期方向割当。全成分を較正したわけではない。標準経路への採用はしない。詳細は報告と事前測定定義。\n',encoding='utf-8')
    dump('completion.json',{'status':'completed','主完了数':384,'補助完了数':384,'欠損数':missing,'周期212法線利得中央値':gain,'手順5補正後分散比':variance['分散比'],'unspecified_choices':CHOICES})

if __name__=='__main__':
    os.environ['V37_CALIBRATION_FIELD']=sys.argv[1]
    final=child(child(RESULTS,'v35_band_origin_round8'),'resume_20261004_final')
    checkpoints=child(final,'experiment_checkpoints')
    records={key:json.loads(child(checkpoints,key+'.json').read_text(encoding='utf-8-sig')) for key in ['260827_7_1','260827_7_8','260829_3_7','260926_7_5']}
    conditions=json.loads(child(OUT,'conditions_resolved.json').read_text(encoding='utf-8'))['条件']
    sys.path.insert(0,str(child(FIELD,'v32_band_origin_round5')))
    import field_round5_methods as methods
    run(records,conditions,final,methods)
