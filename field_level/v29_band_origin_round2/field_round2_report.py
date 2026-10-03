"""Summaries, descriptive image checks, integrity verification, round-two report."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import json, os, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from field_round2_analysis import ROOT,LOCAL,OLD,OUT,METHODS,P,table,csv,digest,verify,plane,energy,time_features,file_records

def correlation(a,b):
    v=np.isfinite(a)&np.isfinite(b)
    if v.sum()<20 or np.std(a[v])<1e-12 or np.std(b[v])<1e-12:return np.nan
    return float(np.corrcoef(a[v],b[v])[0,1])

def descriptive():
    f=table(OLD/'round1_fields_both_definitions.csv');index={k:i for i,k in enumerate(f[f['定義']==METHODS[0]].key)}
    with np.load(OLD/'common_profile_residuals.npz') as z:res=z['residual']
    rows=[];single=[]
    for p in sorted((OUT/'raw_features').glob('*.json')):
        info=json.loads(p.read_text(encoding='utf-8'))
        with np.load(p.with_suffix('.npz')) as z:
            if p.stem.startswith('control_'):
                for k,name in enumerate(('平均輝度','コントラスト','鮮明さ')):
                    a=z['maps'][:,:,k];pr=z['planes'][:,:,k]
                    single.append(dict(key=info['key'],指標=name,二次面の記述分散割合=1-np.nanvar(a-pr)/np.nanvar(a),周期帯エネルギー残存率=energy(a-pr)/energy(a)))
                continue
            key=info['key']
            for stage,arr in [('洗浄前',z['pre_maps']),('洗浄後',z['post_native_maps'])]:
                for k,name in enumerate(('平均輝度','コントラスト','鮮明さ')):
                    a=arr[:,:,k];pr=plane(a)
                    single.append(dict(key=key,指標=stage+name,二次面の記述分散割合=1-np.nanvar(a-pr)/np.nanvar(a),周期帯エネルギー残存率=energy(a-pr)/energy(a)))
            for mi,m in enumerate(METHODS):
                row=dict(key=key,定義=m,回復=info['回復'],固定29視野=info['固定29視野'],ブランク=info['ブランク'])
                for k,name in enumerate(('平均輝度','コントラスト','鮮明さ')):
                    row['洗浄前'+name+'地図相関']=correlation(res[index[key],mi],z['pre_maps'][:,:,k])
                    if info['回復']:
                        row['洗浄後'+name+'地図相関']=correlation(res[index[key],mi],z['post_maps'][:,:,k])
                        row[name+'前後差地図相関']=correlation(res[index[key],mi],z['pre_maps'][:,:,k]-z['post_maps'][:,:,k])
                rows.append(row)
    csv(rows,'round2_single_image_descriptive_correlations.csv');csv(single,'round2_single_image_smoothness.csv')
    single=pd.DataFrame(single);single['日程']=single.key.str[:6]
    smoothsummary=single.groupby(['日程','指標'])['二次面の記述分散割合'].agg(視野数='count',中央値='median',第1四分位=lambda a:a.quantile(.25),第3四分位=lambda a:a.quantile(.75)).reset_index()
    csv(smoothsummary,'round2_single_image_smoothness_summary.csv')
    timea=time_features();sp=table(OLD/'round1_spatial_with_metadata.csv');sp=sp[(sp['縁除外画素']==0)&(sp['分類']=='帯状候補')]
    selected=sp[['key','定義','日程','基板','視野番号','外れ値割合','固定29視野','第一帯軸度','第一スペクトル周期画素','第一自己相関周期画素']]
    csv(selected.merge(timea[['key','洗浄前元更新時刻','洗浄後元更新時刻','洗浄前順位','洗浄後順位','撮影間隔時間']],on='key',how='left'),'round2_band_candidates_chronology_descriptive.csv')
    meta=json.loads((OUT/'round2_image_metadata.json').read_text(encoding='utf-8'))
    files=file_records();original=table(LOCAL/'raw_file_times_original.csv');mr=[]
    for r in meta:
        p=Path(r['画像パス']);t=original[(original.folder==p.parent.name)&(original.file==p.name)]
        image_date=r['画像内日時'];parsed=pd.to_datetime(image_date,format='%Y:%m:%d %H:%M:%S',errors='coerce')
        diff=(parsed[0]-pd.Timestamp(t.iloc[0].modified_local)).total_seconds() if len(t)==1 and len(parsed) and not pd.isna(parsed[0]) else np.nan
        mr.append(dict(画像パス=str(p),ファイル=p.name,元時刻一致件数=len(t),元更新時刻=t.iloc[0].modified_local if len(t)==1 else None,画像内時刻=';'.join(image_date),露光記録=';'.join(r['露光記録']),照合秒差=diff,時刻照合可能=np.isfinite(diff)))
    csv(mr,'round2_metadata_time_audit.csv')
    control=pd.DataFrame([r for r in mr if Path(r['画像パス']).parent.name.startswith('260830')]);control=control.sort_values('元更新時刻')
    control['順序候補']=np.arange(1,len(control)+1);control['洗浄前後対応']='未確定。番号と更新時刻は順序の候補だけ。確認的検定未使用。'
    csv(control,'round2_control_chronology_candidates.csv')
    day=[]
    for d,g in files.groupby('日程'):
        keys=set(g[g['時刻一意']].key)
        both=sum(g.groupby('key')['時刻一意'].all())
        available=g['生画像パス'].notna().all()
        day.append(dict(日程=d,保存差視野数=g.key.nunique(),両時刻一意視野数=int(both),生画像実施=bool(available),画像内時刻確認=bool(available)))
    csv(day,'round2_date_coverage.csv')

def summaries():
    geom=table(OUT/'round2_geometry_explained_fraction.csv');opt=table(OUT/'round2_optical_explained_fraction.csv')
    old=table(OLD/'round1_bf_explained_fraction.csv')
    old=old[(old['モデル']=='B')|((old['モデル']=='F')&old['非周期目印検証'])].copy()
    old['モデル']='周1_'+old['モデル'];old['分類']='周1参照'
    allrows=pd.concat([geom,opt,old],ignore_index=True)
    f=table(OLD/'round1_fields_both_definitions.csv')
    allrows=allrows.merge(f[['key','定義','この定義3percent以上']],on=['key','定義'],how='left')
    metrics=['全面決定係数','対照共通領域決定係数','対照決定係数中央値','対照との差','全面決定係数_G固有','全面決定係数_I固有','全面決定係数_共通','対照共通領域決定係数_G固有','対照共通領域決定係数_I固有','対照共通領域決定係数_共通','周期帯エネルギー残存率']
    rows=[]
    for (method,model),g in allrows.groupby(['定義','モデル']):
        subsets=[('全利用可能',g),('固定29利用可能',g[g['固定29視野']]),('定義別3%以上',g[g['この定義3percent以上']]),('ブランク',g[g['ブランク']]),('帯状候補',g[g['分類']=='帯状候補'])]
        subsets += [('日程'+str(d),a) for d,a in g.groupby('日程')]
        for label,a in subsets:
            for metric in metrics:
                v=a[metric].dropna() if metric in a else pd.Series(dtype=float)
                if not len(v):continue
                rows.append(dict(定義=method,モデル=model,対象=label,指標=metric,視野数=len(v),中央値=float(v.median()),第1四分位=float(v.quantile(.25)),第3四分位=float(v.quantile(.75)),負値視野数=int((v<0).sum())))
    csv(rows,'round2_explanation_summary.csv')
    csv(allrows,'round2_cross_round_comparison.csv')
    # Signed partitions saved once per field, without triple-counting G/I/joint.
    csv(opt[opt['モデル']=='共同'],'round2_shared_unique_fractions.csv')
    return pd.DataFrame(rows)

def plots():
    os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Yu Gothic'
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    geom=table(OUT/'round2_geometry_explained_fraction.csv');opt=table(OUT/'round2_optical_explained_fraction.csv');old=table(OLD/'round1_bf_explained_fraction.csv')
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    for mi,m in enumerate(METHODS):
        for sj,outliers in enumerate((False,True)):
            ax=axes[mi,sj];data=[];labels=[]
            for name,df in [('二次面',geom),('G',opt),('I',opt),('共同',opt),('B（周1）',old),('F（周1）',old)]:
                model=name[0] if '周1' in name else name
                g=df[(df['定義']==m)&(df['モデル']==model)]
                if model=='F':g=g[g['非周期目印検証']]
                if outliers:g=g[g['固定29視野']]
                vals=g['全面決定係数'].dropna().to_numpy()*100
                if not len(vals):continue
                data.append(vals);labels.append(name+'\n'+str(len(vals))+'視野')
            ax.boxplot(data,tick_labels=labels,showfliers=True);ax.axhline(0,color='gray',lw=.8)
            ax.set_ylabel('交差検証による決定係数（%）');ax.set_title(m+'：'+('固定29視野の利用可能部分' if outliers else '全利用可能部分'))
    fig.tight_layout();fig.savefig(dest/'explained_fraction_comparison.png',dpi=140);plt.close(fig)
    f=table(OLD/'round1_fields_both_definitions.csv');index={k:i for i,k in enumerate(f[f['定義']==METHODS[0]].key)}
    with np.load(OLD/'common_profile_residuals.npz') as z:res=z['residual']
    # Five fields identified in the inherited lab note, without reclassifying.
    requested=['260926_01_8','260926_3_3','260926_5_3','260926_7_5','260926_7_8']
    for mi,m in enumerate(METHODS):
        fig,axes=plt.subplots(5,6,figsize=(19,16))
        for j,key in enumerate(requested):
            with np.load(OUT/'raw_features'/(key+'.npz')) as z:pm=z['pre_maps'];qm=z['post_native_maps']
            panels=[pm[:,:,0],qm[:,:,0],res[index[key],mi]]
            ev=OUT/'optical_evaluation'/(key+'.npz')
            if ev.exists():
                with np.load(ev) as z:panels += [z[m+'_'+name+'_prediction'] for name in ('G','I','共同')]
            else:panels += [None,None,None]
            for k,(a,label) in enumerate(zip(panels,['洗浄前の平均輝度','洗浄後の平均輝度（元座標）','外れ値密度の残差','Gの保留予測','Iの保留予測','共同の保留予測'])):
                ax=axes[j,k];ax.set_title(label)
                if a is None:ax.text(.5,.5,'局所前後対応が\n回復できず欠測',ha='center',va='center',transform=ax.transAxes)
                else:
                    if k<2:low,high=np.nanpercentile(a,[1,99]);cmap='gray'
                    else:low,high=np.nanpercentile(res[index[key],mi],[1,99]);cmap='magma'
                    ax.imshow(a,cmap=cmap,vmin=low,vmax=high,extent=[0,2048,2044,0])
                ax.set_xticks([0,1024,2048]);ax.set_yticks([0,1022,2044])
                ax.set_xlim(0,2048);ax.set_ylim(2044,0)
                if k==0:ax.set_ylabel(key+'\n縦位置（ピクセル）')
        fig.suptitle(m+'；32ピクセル四方の升目。平均像2枚の濃淡は各像で独立、密度と予測は行内で同じ濃淡。',fontsize=15)
        fig.tight_layout();fig.savefig(dest/('five_fields_'+str(mi)+'.png'),dpi=120);plt.close(fig)
    raw=table(OUT/'round2_field_time_features.csv');f=table(OLD/'round1_fields_both_definitions.csv');a=raw.merge(f[['key','定義','外れ値割合']],on='key')
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for mi,m in enumerate(METHODS):
        s=a[a['定義']==m]
        for k,c in enumerate(['洗浄前順位','平均輝度前後対数比','鮮明さ前後対数比']):
            for date,g in s.groupby('日程'):axes[mi,k].scatter(g[c],g['外れ値割合']*100,s=14,alpha=.65,label=date)
            axes[mi,k].set_xlabel(c);axes[mi,k].set_ylabel(m+'：外れ値割合（%）')
    axes[0,0].legend(fontsize=8);fig.suptitle('単純散布図は記述用。検定は日程・基板・位置を条件付ける。');fig.tight_layout();fig.savefig(dest/'chronology_and_image_changes.png',dpi=140);plt.close(fig)

def md(a):
    if not len(a):return '該当なし。'
    cols=list(a.columns);lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,r in a.iterrows():
        vals=[]
        for v in r:
            if pd.isna(v):s='未測定'
            elif isinstance(v,(float,np.floating)):s=f'{v:.7g}'
            else:s=str(v)
            vals.append(s.replace('|','／').replace('\n',' '))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def audit():
    verify();a=table(OUT/'round2_input_integrity.csv')
    a['終了内容指紋']=[digest(Path(p)) for p in a['実在パス']];a['不変']=a['開始内容指紋']==a['終了内容指紋']
    csv(a,'round2_input_integrity_final.csv');assert a['不変'].all()
    # Report the repository's actual status, including generated artifacts if unignored.
    def git(*args):return subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True).stdout.strip()
    state=dict(ブランチ=git('branch','--show-current'),コミット=git('rev-parse','HEAD'),変更=git('status','--short'),未追跡=git('ls-files','--others','--exclude-standard'),追跡済変更=git('diff','--name-only'),ステージ変更=git('diff','--cached','--name-only'))
    (OUT/'round2_git_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');return a,state

def report():
    verify();descriptive();summary=summaries();plots();integrity,state=audit()
    tests=table(OUT/'round2_confirmatory_tests.csv');cv=table(OUT/'round2_field_rate_cv_summary.csv');coverage=table(OUT/'round2_date_coverage.csv');meta=table(OUT/'round2_metadata_time_audit.csv');pre=json.loads((OUT/'round2_preflight.json').read_text(encoding='utf-8'))
    def percentage_table(metric,targets=('全利用可能','固定29利用可能')):
        a=summary[(summary['指標']==metric)&summary['対象'].isin(targets)].copy()
        if '_G固有' in metric or '_I固有' in metric or '_共通' in metric:a=a[a['モデル']=='共同']
        for c in ('中央値','第1四分位','第3四分位'):a[c]*=100
        return md(a.drop(columns=['指標']))
    def val(method,model,target,metric='全面決定係数'):
        a=summary[(summary['定義']==method)&(summary['モデル']==model)&(summary['対象']==target)&(summary['指標']==metric)]
        return f"{float(a.iloc[0]['中央値'])*100:.3f}%" if len(a) else '未測定'
    prediction=pd.read_csv(OUT/'prediction_sha256.csv').iloc[0]
    globalf=table(OUT/'round2_global_image_features.csv');opt=table(OUT/'round2_optical_explained_fraction.csv');geom=table(OUT/'round2_geometry_explained_fraction.csv')
    lines=['# 2026年10月4日・外れ値の帯状分布：周2報告','',
           '## 結論と範囲','',
           '仮説G（ピント・照明・周辺減光）：滑らかな勾配の記述には部分的な支持があるが、今回の低次面・画像モデルを帯状分布の主因とする支持は得られない。光学原因全体は判定不能（確度：中）。仮説I（撮影順序・前後撮影条件）：今回の予測・対象では帯の主因として支持されない（確度：中）。時刻・画像条件との関連が下の検定で支持されず、因果機構全体の否定はできない。画像構造との整合性だけではドリフトを支持しない。撮影条件・試料変化・局所位置合わせの切り分けは判定不能（確度：低〜中）。',
           '',f"二次面の固定29視野での説明率中央値は、平均と標準偏差の定義で{val(METHODS[0],'二次面','固定29利用可能')}、中央値と絶対偏差の定義で{val(METHODS[1],'二次面','固定29利用可能')}。画像Gは同順に{val(METHODS[0],'G','固定29利用可能')}、{val(METHODS[1],'G','固定29利用可能')}。画像Iは{val(METHODS[0],'I','固定29利用可能')}、{val(METHODS[1],'I','固定29利用可能')}。共同モデルは{val(METHODS[0],'共同','固定29利用可能')}、{val(METHODS[1],'共同','固定29利用可能')}。画像モデルは260926の回復34視野（固定29中5視野）の数値である。確度：高（計算値）。",
           '', 'これらは外れ値残差の分散を予測する説明率であり、分子由来の割合・補正可能な陽性ピラーの割合・因果の寄与率ではない。負値を保持し、未測定を0%にしていない。今回の比較を標準解析・マスク・除外に取り込んでいない。',
           '', '## 実在確認、運用、対象','',
           f"入力起点は `{LOCAL}`。解析コピーは `{pre['解析フォルダ']}`。生画像の実在フォルダは次のとおり。",
           '', *['- `'+p+'`' for p in pre['生画像フォルダ']],
           '', '260830の名前には全角の下線「＿」が含まれる。列挙した綴りを使い、似た名前を組み立てていない。開始時の完了印は '+', '.join('`'+s+'`' for s in pre['完了印'])+'。初回計算時に `copy_done_2` はなかったが、最終確認中に `_copy_done_2.txt` を確認し、追加規定と指紋を先に保存して260922・260924・260828の216組を追加した。初回結果を別名で保持した。最終解析は全4日程の286組、原画像572枚と260830の6枚。',
           '', *['- `'+p+'`' for p in json.loads((OUT/'round2_completion_preflight.json').read_text(encoding='utf-8'))['追加生画像フォルダ']],
           '',md(coverage),
           '',f"632視野の保存済み密度・残差と二定義の閾値をそのまま利用。局所前後対応は周1で差が回復した{int(globalf['回復'].sum())}視野に限定。260926の残り36視野と追加216視野は局所共同説明率が欠測だが、全体の明るさ・鮮明さ・時刻の検定には利用できる。仮説B・Fを再実行して追加日程の変換を回復することはしていない。", 
           '', '周1のコードは読み取りで数値規定を確認し、必要な交差検証関数を新しい版へコピーした。周1の分類・方向・仮説B・Fを再計算していない。ラボノート8本を読んだ。`10_引き継ぎ` と `05_解析` の元フォルダ構造はコピーに残っていないため、平坦化された8本を内容で確認した。別日程の基板01は洗浄ブランクと推定せず、周1のブランク列を保持する。',
           '', '版確認：開始時、`field_level/` と `data/results/`、解析コピー直下、現在のブランチ一覧・タグ一覧にv29なし。v25〜v27は解析コピー、v28はコード・結果・タグで使用済み。新規配置は `field_level/v29_band_origin_round2/` と `data/results/v29_band_origin_round2/` のみ。古い環境文書の入力指定より今回の `data/inputs_local/` 指定を優先した。作業開始時の追跡済み変更は空。読取り専用の `.git` と今回の書込み範囲の制限に従い、取得による `.git` 更新を伴う `git pull --ff-only` は実行していない。ローカルに存在するブランチ・タグでの版確認であり、取得していない遠隔の未取得版までは確認できない。',
           '', '## 予測の先行保存','',
           f"予測書：`{prediction.Path}`。保存時刻：{prediction.RecordedAt}。セキュアハッシュアルゴリズム二百五十六ビットの内容指紋：`{prediction.Hash}`。全計算前に記録し、各段階で照合した。予測方向・初回20検定・10%の主因判定基準を変更していない。",
           '', '追加規定は `261004_周2_追加コピー完了後の実行規定.md`。追加画像指標の計算前に保存し、内容指紋 `A5DD33A429F7E4853D5652A945926811B20D76E9636F90E77BE6FBF4F98B7799`、記録時刻2026-10-04T05:49:04.5841154+09:00。追加は同じ方法の12検定で、初回20＋追加12＝32比較補正。初回未補正有意確率は保持し、初回20比較補正も別列に保持。初回の260926の画像指標・結果は既に見た後の追加であり、この事情を伏せない。',
           '', 'Gの予測：滑らかな照明・鮮明さは二次面で説明できるが、周期帯は引いた後も80%以上残る。洗浄前のみ・後のみで低次の空間変化があり、両像に共通なら差で弱まり、前後で変われば差と関連する。ブランク・260830にも生じ得る。Iの予測：元時刻の順序と前後の明るさ・鮮明さの変化、さらに視野割合が日程・基板内でも関連する。順序だけは一定なので帯を作れず、局所像との相互作用が必要。260830の前後は推測であり確認的検定にしない。完全な変数・検定規定は予測書に保存。',
           '', '## 方法と検定','',
           '差は洗浄前から洗浄後を引く。平均＋標準偏差の三倍、中央値＋1.4826倍した中央値絶対偏差の三倍という周1の日程別ブランク閾値を維持。32ピクセル四方の外れ値密度から、自身以外の631視野の平均地図を引く。',
           '', '二次面は横・縦・それぞれ二乗・積の5変数。Gは洗浄前の平均・コントラスト・鮮明さ・ピーク背景比、前後の平均・コントラスト・鮮明さ差を画像だけで二次面へ投影した7地図。Iは洗浄前の局所平均・コントラスト・鮮明さ3地図に全体前後対数比を掛ける。全体の前後対応は台帳の指定、局所の洗浄後は回復行列で32ピクセル統計地図を洗浄前座標へ双一次補間する。これは標準のピラー標本値を変更する処理ではない。局所統計の補間は境界や格子位相に影響され、正確な光学モデルではない。',
           '', '鮮明さは隣接画素差の二乗平均の平方根を平均で割る代理量で、露光・ピントを直接測定していない。ピーク対背景比は洗浄前の格子点を整数丸めし、三×三画素和を51画素のガウス背景和で割って1を引く。',
           '', '交差検証は周1と同じ8分割・256ピクセルブロック・周囲64ピクセルの学習余白・学習内標準化・正則化係数1。各保留部分の学習平均を基準に決定係数を計算。非循環の横縦±256・±512ピクセル移動8地図と90度回転1地図を比較し、全9対照が測れる共通領域を実地図にも適用。全面と共通領域を混ぜない。',
           '', '初回20検定と追加12検定へ、合計32比較のボンフェローニ法を適用。空間説明率は視野単位の片側符号反転、時刻・画像変化は日程と基板の指示変数・位置の指示変数を除いた最大絶対相関の両側並べ替え。基板内で視野行全体を並べ替えて再残差化し、多変数から最大値を選ぶ操作も毎回含める。10,000回、乱数種20261004。有意確率は（実測以上＋1）／10001。追加の日程内訳は全4日程286視野と追加3日程216視野。',
           '',md(tests[['検定','定義','視野数','統計量','補正前有意確率','ボンフェローニ補正後','日程基板反転有意確率','状態']]),
           '', '基板内の残差並べ替えは位置効果を推定した近似的な検定で、完全な無作為実験ではない。同じ基板の依存も残る。日程・基板ごとの符号反転は空間検定の感度として併記した。基板間の処理順と濃度は分離しない。時刻と条件変化の検定は二定義で同一の検定であり、新しい独立証拠として重ねて数えない。全4日程の時刻と割合も初回の時刻286視野検定と同じである。',
           '', '## 周1と同じ尺度の空間説明率','',
           '以下は百分率。第1・第3四分位を別列で示す。母集団は全632・固定29だが、画像G・I・共同・周1Bは260926の34／5視野、周1Fは独立検証2／1視野のみ。F候補だけの地図を混ぜない。',
           '',percentage_table('全面決定係数'),
           '', '空間対照と同じ領域での説明率：', '',percentage_table('対照共通領域決定係数'),
           '', '空間対照の説明率中央値：', '',percentage_table('対照決定係数中央値'),
           '', '実地図と空間対照との差：', '',percentage_table('対照との差'),
           '', '画像Gの二次面は移動・回転しても同じ関数空間を含み、移動対照との差に識別力が乏しい。正則化と地図の共線性で小さい差は生じる。G・二次面の対照比較の不成立を、ピント・照明原因全体の否定には使えない。Iは局所像の構造を持つので対照は比較可能だが、全体変化の非ゼロ係数は学習時の標準化で消える。Iの空間説明率は撮影順序の説明率と呼ばない。',
           '', '固定29かつ帯状候補での、二次面を引いた周期エネルギー残存率（%）：',
           '',md(geom[(geom['固定29視野'])&(geom['分類']=='帯状候補')].groupby('定義')['周期帯エネルギー残存率'].agg(視野数='count',中央値=lambda a:a.median()*100,第1四分位=lambda a:a.quantile(.25)*100,第3四分位=lambda a:a.quantile(.75)*100).reset_index()),
           '', '帯のエネルギーは面を引いても残る。100%を超える値は保留予測の誤差や窓の周波数混合で増えた成分を含み、エネルギーが物理的に発生したことを意味しない。Iの全面説明率に対して中央の共通領域の説明率が低い点は、広い明暗・周辺構造への依存を示唆する（解釈、確度：中）。',
           '', '## G・I固有と共通部分','',
           '同じ視野・有効領域・分割で、G固有＝共同−I、I固有＝共同−G、共通＝G＋I−共同。これは符号付きの交差検証分解。負の共通部分や負の固有部分を保持し、共有する物理原因の割合とは解釈しない。',
           '', 'G固有（全面、%）：','',percentage_table('全面決定係数_G固有'),
           '', 'I固有（全面、%）：','',percentage_table('全面決定係数_I固有'),
           '', '共通部分（全面、%）：','',percentage_table('全面決定係数_共通'),
           '', '共通領域の同じ分解も `round2_shared_unique_fractions.csv` と `round2_explanation_summary.csv` に全て保存。二次面と画像G・Iを混ぜて分解していない。',
           '', '## 視野割合と撮影順序・前後条件','',
           '1視野ずつ保留した予測。日程・基板・位置の基準モデルに対する追加決定係数と、全体の視野割合分散に対する決定係数を区別する。以下は割合単位（1が100%）。空間説明率とは異なる応答・尺度であり、周1の表とは混ぜない。',
           '',md(cv),
           '', '時間モデルは洗浄前・後の基板内順位と前後間隔。Gの視野モデルは洗浄前の全体コントラスト・鮮明さ・周辺中央比、Iの視野モデルは時間と全体前後対数比。基板と濃度が1対1なので基板間の濃度効果・処理順効果は推定しない。各日程・基板内の時刻幅や順序は `round2_field_time_features.csv` に保存。帯の向き・間隔と順序は `round2_band_candidates_chronology_descriptive.csv` に記述のみで保存し、少数候補から有意な関係を主張しない。',
           '', '## 単独像、ブランク、260830、時刻の確かさ','',
           f"{len(meta)}画像中、認識できる画像内撮影日時のある画像は{int(meta['画像内時刻'].notna().sum())}、露光記録のある画像は{int(meta['露光記録'].notna().sum())}、元更新時刻と画像内日時を照合できた画像は{int(meta['時刻照合可能'].sum())}。一致の程度は測定不能で、更新時刻を撮影時刻と断定しない。時刻表の作成時刻はコピー等の時刻を含み、撮影順序に使わない。更新時刻の撮影時刻としての妥当性は未確認（確度：中、利用者提供の代理記録）。",
           '', 'ファイル名は台帳で洗浄前末尾0、洗浄後末尾1を対応させる。a-b-c-d形式では濃度の接頭辞・基板文字列・視野番号・前後符号の順。三要素形式のブランクなどもあり、接頭辞の数字だけで濃度を再解釈しない。基板01と基板1を別の文字列として維持。台帳に採用されない倍率違いの `50-...` や無接頭辞画像・モザイクを今回の286組に加えない。初回の前後名と元時刻は `round2_field_files_times.csv`、追加完了後の実在画像パスは `round2_field_files_times_completed.csv`。',
           '', '洗浄前のみ・後のみの平均・コントラスト・鮮明さ地図と、その二次面の記述分散割合を `round2_single_image_smoothness.csv`、外れ値残差との地図相関を `round2_single_image_descriptive_correlations.csv` に保存。これらは記述統計で、同一地図が繰り返される組の検定ではない。単独像の明暗模様と外れ値の模様の完全一致を必須としていない。ブランクの空間説明率は要約表の対象「ブランク」、全体像の前後差は全体画像表で確認可能。',
           '', '単独像の滑らかさ（日程・指標別、割合単位。二次面で記述される地図内分散であり、外れ値の交差検証説明率ではない）：',
           '',md(table(OUT/'round2_single_image_smoothness_summary.csv')),
           '', '指定5視野の図を実際に開いて確認した。平均輝度の広い明暗と境界の明部は洗浄前後で共通して見え、外れ値の周期帯を低次のG予測が再現していない（図で確認した事実、確度：中）。この図は32ピクセル平均で、ピラーを直接切り出した焦点判定図ではない。260830も単独像に滑らかな成分があるが、同じ周期の外れ値が出るかは対応不明のため検証できない。',
           '',md(table(OUT/'round2_control_chronology_candidates.csv')[['ファイル','元更新時刻','順序候補','洗浄前後対応']]),
           '', '260830は6画像の局所像・滑らかさを記述した。上の時刻順は推測の候補であり、洗浄前後の対応を一つにも確定していない。前後差や外れ値の確認的検定に使っていない。洗浄対応・露光・ピントの一次記録が必要。',
           '', '## 数値動作確認、図、入力不変','',
           md(pd.DataFrame([json.loads((OUT/'round2_numerical_verification.json').read_text(encoding='utf-8'))])),
           '',md(pd.DataFrame([json.loads((OUT/'round2_completion_verification.json').read_text(encoding='utf-8'))])),
           '', '最初に2視野で動作確認し、その後全体へ拡張。人工像の結果は実データの説明率の代用ではない。独立の最小二乗実装と正則化解の一致、学習余白と分割、非循環移動を検証。G・I固有＋共通が共同に戻る恒等式を数値検証した。',
           '', '- `figures/explained_fraction_comparison.png`：二定義、全利用可能・固定29利用可能、周1B・Fとの比較。',
           '- `figures/five_fields_0.png` と `figures/five_fields_1.png`：指定5視野の洗浄前後像の平均地図、外れ値密度残差、G・I・共同の保留予測。',
           '- `figures/chronology_and_image_changes.png`：時刻と画像変化の記述散布図。',
           '', f"開始前後の{len(integrity)}入力・周1参照ファイルと追加649ファイルの内容指紋は全て不変。`round2_input_integrity_final.csv` と `round2_completion_input_integrity_final.csv` に実在絶対パス・サイズ・開始終了指紋を保存。入力は読み取りのみ、凍結中の別リポジトリは参照も実行もしていない。新しい他チャット成果のコピーは行わず、利用者が用意した読取り専用入力スナップショットを参照した。", 
           '', '## 解釈と別の説明、弱点','',
           '事実は上の計算値・有意確率・撮影時刻記録の欠如・入力不変。解釈は、滑らかな光学むらは外れ値の広い勾配の一部に整合するが、周期帯の主因を確定しない、というもの（確度：中）。Iの局所像モデルが説明しても、局所像を光学変化が変調したのか、洗浄が変えたのか、残差位置合わせが変えたのかは区別できない（確度：低〜中）。',
           '', '洗浄・乾燥の残渣（仮説C）、ゴミ・シミ・傷の周辺（仮説H）、独立検証不足の局所位置合わせ（仮説F）が残る。金の厚み（仮説D）・鋳型構造（仮説E）・実吸着（仮説J）も否定していない。視野内の低次面というモデルの狭さ、回復34視野への選択、32ピクセル升目の分解能、隣接画素鮮明さのノイズ依存、全体条件係数の標準化、時刻代理の未検証、同基板の依存、位置残差並べ替えの近似、濃度交絡が弱点。二次面で説明しないことは、細かい焦点むらを含むG全体の否定ではない。',
           '', '## 未実施・必要な追加入力','',
           '260922・260924・260828の生画像単独指標と全体前後比較は追加完了したが、元の最終変換がないため局所前後の空間比較は未実施。632視野のうち原画像がまだない日程260825・260827・260829・260923・260927の前後原画像が、日程再現の検証に必要。元更新時刻も表にない日程では必要。露光・ゲイン・ピント・撮影順序の装置記録、260830の前後対応、撮影時刻と更新時刻の照合元が必要。260926の36視野と追加216視野の元の最終変換があれば局所比較の欠測を埋められる。仮説B・Fの再検証は周2では行っていない。',
           '', '## 作業状態と全出力','',f"ブランチ：`{state['ブランチ']}`。開始・終了コミット：`{state['コミット']}`。コミット・タグ・プッシュなし。既存の追跡済み変更・ステージ変更なし。未追跡は下記（報告生成時点）。生成画像・数値配列・キャッシュをGitへ追加していない。",
           '', '```text',state['未追跡'],'```','', '全出力のパス・サイズ・内容指紋は `round2_output_inventory.csv`、実装の内容指紋は `round2_code_provenance.csv` に記録。完了範囲は表と上の未実施一覧のとおり。この報告で周2を止め、次周には進まない。',
           '', '## 次に検証すべき仮説の順序（提案。決定はClaude Code）','',
           '1. 仮説H：指定外れ値の局所像との対応が傷・シミ・明部近傍に集中するかを、目視承認した位置情報と距離で検証する。画像Iの説明が撮影条件なのか既存構造なのかの切り分けに直結する。自動欠陥候補をマスクや除外に採用しない。',
           '2. 仮説C：洗浄後に増える局所模様、洗浄水流・風向き・乾燥前線の一次記録から、予測する向きや前後の差を先に固定して検証する。撮影時刻の関連だけでは除けない洗浄変化を分離する。',
           '3. 仮説Aの境界との幾何学的関係：位置固定の再検定ではなく、実際の書き込み境界・十字構造から距離を予測する。既知の位置6・7の構造と外れ値の局所像との関係を検証する。',
           '4. 仮説D・E：金蒸着・鋳型構造の独立観察と製作記録が得られた段階で検証。仮説Jは光学・洗浄・傷の説明を限定し、分子なし対照の対応を確定してから評価する。仮説Fは必要な独立位置目印・元変換を得た後に別周として判断する。']
    (OUT/'261004_周2_報告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    csv([dict(実在パス=str(p),内容指紋=digest(p)) for p in sorted((ROOT/'field_level'/'v29_band_origin_round2').glob('*')) if p.is_file()],'round2_code_provenance.csv')
    csv([dict(実在パス=str(p),バイト数=p.stat().st_size,内容指紋=digest(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='round2_output_inventory.csv'],'round2_output_inventory.csv')
    print('report',OUT/'261004_周2_報告.md',flush=True)

if __name__=='__main__':report()
