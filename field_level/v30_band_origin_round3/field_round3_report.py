"""Write reviewable scientific figures and the round-three report."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import os,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from field_round3_analysis import ROOT,OUT,OLD,PREVIOUS,METHODS,table,csv,read_image,dump,digest,verify,spatial
os.environ['MPLCONFIGDIR']=str(OUT/'plot_settings')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font=Path('C:/Windows/Fonts/meiryo.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False

def md(a,percent=()):
    a=a.copy()
    for c in a:
        if c in percent:a[c]=a[c].map(lambda x:f'{100*x:.4f}%' if pd.notna(x) else '未測定')
        elif pd.api.types.is_float_dtype(a[c]):a[c]=a[c].map(lambda x:f'{x:.6g}' if pd.notna(x) else '未測定')
        else:a[c]=a[c].map(lambda x:str(x).replace('|','／').replace('\n',' ') if pd.notna(x) else '未測定')
    return '| '+' | '.join(a.columns)+' |\n| '+' | '.join(['---']*len(a.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in a.itertuples(index=False,name=None))

def figures():
    verify();dest=OUT/'figures';dest.mkdir(exist_ok=True)
    with np.load(OLD/'density_maps.npz') as z:maps=z['density'];keys=list(z['keys'])
    fields=['260926_3_3','260926_5_3','260926_7_5','260926_7_8','260926_01_8']
    for key in fields:
        info=json.loads((OUT/'raw_candidates'/(key+'.json')).read_text(encoding='utf-8'))
        with np.load(OUT/'raw_candidates'/(key+'.npz')) as z:pc=z['pre_mask'];qc=z['post_mask'];pz=z['pre_z'];qz=z['post_z']
        pre=read_image(Path(info['洗浄前パス']))[::4,::4];post=read_image(Path(info['洗浄後パス']))[::4,::4]
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        lo,hi=np.quantile(np.concatenate([pre.ravel(),post.ravel()]),[.005,.995])
        for ax,im,mask,label in [(axes[0,0],pre,pc,'洗浄前・候補輪郭'),(axes[0,1],post,qc,'洗浄後・候補輪郭')]:
            ax.imshow(im,cmap='gray',vmin=lo,vmax=hi,extent=(0,2048,2044,0))
            if mask.any():ax.contour(np.arange(512)*4,np.arange(511)*4,mask>0,levels=[.5],colors=['#ff6550'],linewidths=.5)
            ax.set_title(label)
        for ax,z,title in [(axes[1,0],pz,'洗浄前単独像・局所残差'),(axes[1,1],qz,'洗浄後単独像・局所残差')]:
            ax.imshow(z,cmap='coolwarm',vmin=-10,vmax=10,extent=(0,2048,2044,0));ax.set_title(title)
        for mi,ax in enumerate(axes[:,2]):
            image=ax.imshow(maps[keys.index(key),mi]*100,cmap='magma',extent=(0,2048,2044,0));ax.set_title(METHODS[mi]+'・外れ値密度');fig.colorbar(image,ax=ax,label='百分率')
        for ax in axes.ravel():ax.set_xlabel('横位置（ピクセル）');ax.set_ylabel('縦位置（ピクセル）')
        fig.suptitle(key+'：自動候補の記述比較（承認・除外・マスクには使わない）')
        fig.savefig(dest/(key+'_候補と外れ値.png'),dpi=140);plt.close(fig)
    a=table(OUT/'round3_distance_density.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,method in zip(axes,METHODS):
        for label,color in [('後全候補','#bf442b'),('前全候補','#287bb1'),('後新規移動候補','#a477b8'),('後既存候補','#559c6f')]:
            q=a[(a['定義']==method)&(a['候補区分']==label)]
            q=q.pivot(index='key',columns='殻番号',values='外れ値密度')*100
            med=q.median();lower=q.quantile(.25);upper=q.quantile(.75)
            ax.plot(med.index,med.values,marker='o',label=label,color=color);ax.fill_between(med.index,lower.values,upper.values,alpha=.12,color=color)
        ax.set_xticks(range(7),['0～16','16～32','32～64','64～128','128～256','256～512','512以上'],rotation=30)
        ax.set_xlabel('候補からの距離（ピクセル）');ax.set_ylabel('外れ値密度の視野中央値（百分率）');ax.set_title(method);ax.legend(fontsize=9)
    fig.suptitle('距離別の記述：影は視野間の第1～第3四分位、画像を共有する関連')
    fig.savefig(dest/'距離別密度の視野中央値.png',dpi=150);plt.close(fig)
    s=spatial();fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,method in zip(axes,METHODS):
        q=s[(s['定義']==method)&s['周期候補']]
        for day,g in q.groupby('日程'):
            counts,bins=np.histogram(g['第一帯軸度'],bins=np.arange(0,181,10))
            ax.plot((bins[:-1]+bins[1:])/2,counts,label=day)
        ax.set_xlabel('帯軸角度（度、180度周期）');ax.set_ylabel('周期候補視野数');ax.set_title(method);ax.legend(fontsize=8,ncol=3)
    fig.savefig(dest/'日程別の帯軸方向.png',dpi=150);plt.close(fig)
    return fields

def supplemental():
    a=table(OUT/'round3_image_features.csv');density=table(OUT/'round3_distance_density.csv');rows=[]
    for stage in ('洗浄前','洗浄後'):
        for label,q in [('全画像組',a),('ブランク',a[a['ブランク']]),('固定29利用可能',a[a['固定29視野']]),('位置6・7',a[a['位置番号'].isin([6,7])]),('位置6・7以外',a[~a['位置番号'].isin([6,7])])]:
            for c in ['候補数','線状候補数','候補画素割合','影響半径代理画素','単独像方向集中度','単独像周期帯パワー']:
                v=q[stage+c].dropna()
                rows.append(dict(時点=stage,対象=label,指標=c,視野数=len(v),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75)))
    csv(rows,'round3_image_feature_summary.csv')
    shellsummary=[]
    for cols,g in density.groupby(['定義','候補区分','殻番号']):
        v=g['外れ値密度'].dropna()
        shellsummary.append(dict(定義=cols[0],候補区分=cols[1],殻番号=cols[2],視野数=len(v),密度中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75)))
    csv(shellsummary,'round3_distance_density_summary.csv')
    s=spatial();angles=[]
    for method,g in s.groupby('定義'):
        for label,a in [('全周期候補',g[g['周期候補']]),('固定29周期候補',g[g['周期候補']&g['固定29視野']]),('帯状うろこ状のみ',g[g['分類'].isin(['帯状','うろこ状'])])]:
            for c in ['カメラ軸最小角度差','六方格子最小角度差','十字線最小角度差']:
                v=a[c].dropna()
                angles.append(dict(定義=method,対象=label,基準=c,視野数=len(v),角度差中央値度=v.median(),十度以内割合=(v<=10).mean() if len(v) else np.nan))
    csv(angles,'round3_axis_agreement_descriptive.csv')
    # Candidate-proportion and directions in each single image are descriptive;
    # their changes are not independent confirmation of the difference response.
    changes=[]
    a=table(OUT/'round3_image_features.csv')
    for _,r in a.iterrows():
        angle=abs((r['洗浄後単独像帯軸度']-r['洗浄前単独像帯軸度']+90)%180-90)
        changes.append(dict(key=r.key,日程=r['日程'],基板=r['基板'],ブランク=r['ブランク'],固定29視野=r['固定29視野'],候補数後引く前=r['洗浄後候補数']-r['洗浄前候補数'],単独像前後軸角度差度=angle,周期帯パワー後前比=r['洗浄後単独像周期帯パワー']/r['洗浄前単独像周期帯パワー']))
    csv(changes,'round3_single_image_change_descriptive.csv')
    evaluated=table(OUT/'round3_explained_fraction.csv');numeric=[]
    for (method,model),g in evaluated.groupby(['定義','モデル']):
        for metric in ['全面決定係数','対照共通領域決定係数','対照決定係数中央値']:
            v=g[metric].dropna()
            numeric.append(dict(定義=method,モデル=model,指標=metric,視野数=len(v),最小値=v.min(),中央値=v.median(),負値数=int((v<0).sum()),負の一未満数=int((v < -1).sum())))
    csv(numeric,'round3_numerical_extremes.csv')
    # Diagnose the worst H common-domain result without altering the locked model.
    worst=evaluated[evaluated['モデル']=='H'].nsmallest(1,'対照共通領域決定係数').iloc[0]
    key=worst.key;method=worst['定義'];d=table(OUT/'round3_orientation_predictions.csv')
    with np.load(OUT/'raw_candidates'/(key+'.npz')) as z:H=z['H'];before=z['前像']
    from field_round3_analysis import c_features,controls,folds,cv_score
    with np.load(OLD/'common_profile_residuals.npz') as z:response=z['residual'][list(z['keys']).index(key),METHODS.index(method)]
    models=[H,before];r=d[(d.key==key)&(d['定義']==method)].iloc[0]
    if r['C利用可能']:models.append(c_features(r['予測帯軸度']))
    valid=np.isfinite(response)
    for X in models:valid &= np.isfinite(X).all(axis=2)
    for X in models:
        for c in controls(X):valid &= np.isfinite(c).all(axis=2)
    score,pred=cv_score(response,H,valid);assert abs(score-worst['対照共通領域決定係数'])<1e-6
    entries=[]
    flat=H.reshape(4096,-1)
    for fold,(tr,te) in enumerate(folds(valid)):
        if not len(te):continue
        mu=flat[tr].mean(axis=0);sigma=flat[tr].std(axis=0);sigma[sigma<1e-12]=1
        standardized=(flat[te]-mu)/sigma
        entries.append(dict(key=key,定義=method,保留分割番号=fold,学習升目数=len(tr),保留升目数=len(te),最大標準化保留説明変数=float(np.max(abs(standardized))),保留残差最大絶対値=float(np.max(abs(response.ravel()[te]))),保留予測最大絶対値=float(np.max(abs(pred.ravel()[te])))))
    csv(entries,'round3_extrapolation_diagnostic.csv')

def report():
    verify();supplemental();drawn=figures()
    preflight=json.loads((OUT/'round3_preflight.json').read_text(encoding='utf-8'));validation=json.loads((OUT/'round3_verification.json').read_text(encoding='utf-8'))
    e=table(OUT/'round3_explanation_summary.csv');tests=table(OUT/'round3_tests.csv');image=table(OUT/'round3_image_features.csv');coverage=table(OUT/'round3_date_coverage.csv');comparison=table(OUT/'round3_all_hypotheses_comparison.csv');prediction=pd.read_csv(OUT/'prediction_sha256.csv').iloc[0]
    cmodel=[]
    for method in METHODS:
        r=e[(e['定義']==method)&(e['モデル']=='C')&(e['対象']=='固定29利用可能')&(e['指標']=='全面決定係数')]
        n=int(r.iloc[0]['視野数']) if len(r) else 0;median=float(r.iloc[0]['中央値']) if len(r) else np.nan
        q=tests[(tests['定義']==method)&tests['検定'].isin(['C基板内方向一致_全周期候補','C空間対照差_全利用可能'])]
        support=len(q)==2 and (q['ボンフェローニ補正後']<.05).all() and median>=.1
        cmodel.append(dict(定義=method,固定29利用可能視野数=n,固定29説明率中央値=median,規定した方向機構支持条件=bool(support)))
    csv(cmodel,'round3_C_decision.csv')
    comparison.loc[comparison['仮説']=='H','判定']='局所画像との関連は記述、原因は判定不能'
    comparison.loc[comparison['仮説']=='H','確度']='低（因果）、高（計算値）'
    for method,r in zip(METHODS,cmodel):
        v=(comparison['仮説']=='C')&(comparison['定義']==method)
        comparison.loc[v,'判定']='固定方向モデルに限定して支持、洗浄原因は判定不能' if r['規定した方向機構支持条件'] else '固定方向モデルの支持条件を満たさない、洗浄原因は判定不能'
        comparison.loc[v,'確度']='中（モデル評価）、低（洗浄原因）'
    descriptions={'A':'書き込み境界と視野グリッドの位置固定構造','B':'画素・六方格子の標本化と位相によるうなり','F':'独立非周期目印で検証した局所位置合わせずれ','G（方向を限定しない二次面）':'横縦の一次・二次・積による滑らかな勾配','G':'画像の光学指標を二次面へ投影した地図','I':'前像の局所指標と前後の全体条件差の相互作用','H':'後候補への距離の五つの指数減衰関数','H（前像対照）':'前候補への距離の五つの指数減衰関数','C':'対象以外の基板平均帯軸による固定周期・勾配地図','H・C共同':'HとCの説明変数を連結、同一視野・升目で検証','D':'金の蒸着の角度・影・厚みの不均一','E':'鋳型柱の倒れ・直径・欠損の不均一','J':'分子の実際の不均一吸着'}
    comparison['仮説内容または説明変数']=comparison['仮説'].map(descriptions)
    csv(comparison,'round3_all_hypotheses_comparison.csv')
    git=dict(作業ブランチ=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True,encoding='utf-8').strip(),作業開始時点のコミット=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),短い状態=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf-8').strip(),未追跡ファイル=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True,encoding='utf-8').splitlines())
    dump(git,OUT/'round3_git_state.json')
    selected=e[(e['指標']=='全面決定係数')&(e['対象']=='固定29利用可能')&e['モデル'].isin(['H','前像','C','共同'])]
    lines=['# 2026年10月4日・周3報告：ゴミ・シミ・傷と洗浄・乾燥',
    '\n## 結論',
    '仮説H（局所のゴミ・シミ・傷の影響）は、候補近傍との関連を記述できるが、原因としては判定不能（確度：低）。単独画像から候補を作る手順は保存差から独立でも、差の片側の画像を共有するため、説明率や有意な関連を因果の証拠にしない。今回の自動候補の承認、採用、除外、マスクは一切行っていない。',
    '仮説C（洗浄水・ブロワー・乾燥前線）の今回の固定方向モデルは、二定義とも支持条件を満たさなかった（確度：中）。洗浄・乾燥そのものの原因判定は、独立した操作記録がないため判定不能（確度：低）。帯の向きが揃っても撮影軸や六方格子、成形、蒸着、傷、位置合わせの方向依存が残る。',
    md(pd.DataFrame(cmodel),percent=['固定29説明率中央値']),
    '固定29視野における利用可能部分の全面説明率は次のとおり。説明率は残差密度地図の保留升目の分散予測を表す。外れ値ピラーの原因割合、分子の割合、除外可能割合ではない。負値は保持し、欠測を0に置き換えていない。数値の確度：高。',
    md(selected[['定義','モデル','視野数','中央値','第1四分位','第3四分位','負値視野数']],percent=['中央値','第1四分位','第3四分位']),
    '\n## 確認した入力と実行範囲（事実）',
    '入力の起点と実在名を上位から列挙して確認した。全角下線を含む260830フォルダは列挙した綴りをそのまま使用した。入力はdata/inputs_localのコピーと周1・周2の保存成果のみ。アクセス不能な外部ドライブへ推測で進んでいない。',
    '- 起点：`'+preflight['入力起点']+'`\n- 解析コピー：`'+preflight['解析コピー']+'`\n'+'\n'.join('- 生画像：`'+p+'`' for p in preflight['生画像フォルダ'])+'\n'+'\n'.join('- コピー完了印：`'+p+'`' for p in preflight['完了印']),
    md(coverage),
    f"632視野の保存差を二定義で保持し、286組の生画像すべてと260830の単独像6枚を処理した。局所距離の対応利用可能は{int(image['対応採用'].sum())}組、保存差再現済み行列34組と独立画像対応{int((image['対応採用']&~image['保存差再現群']).sum())}組。追加対応失敗は{int((~image['対応採用']).sum())}組で、候補は記述したが局所説明率を欠測とした。対応失敗はround3_image_features.csvに理由と相関、移動、行列式を保存。利用できない別日程の生画像は260825・260827・260829・260923・260927。これらの差・方向は利用したがHは未測定。",
    '使用した全ファイルの実在する絶対パス、バイト数、開始・終了内容指紋はround3_input_integrity.csvとround3_input_integrity_final.csv。実験画像の組、元更新時刻、一意性、差との識別子対応はround3_field_manifest.csv。保存差は解析コピーのtables/cached_field_differences内の632ファイル。table_registration_field_qc.csvの実在ファイル名を列挙した生画像に照合し、日程により基板01の意味が変わる点は周1のブランク列を保持。',
    'ラボノートのコピー8本を読み、解析コピー内のマークダウン文書も検索した。元の10_引き継ぎ・05_解析のフォルダ構造は平坦化されている。水流、風向き、乾燥前線の実測操作記録は今回のコピー内で見つからない。記載の欠如は操作や欠陥がなかった証拠ではない。位置6・7の書き込み境界と視野グリッドの構造は既存記録の確定済み知見として保持する。元の位置6の画像鑑識専用成果・260916原記録は入力コピー内では見つからず、詳細の独立確認はできない。検索で見つけた文と行番号はround3_record_search.csv。',
    '視野番号と位置番号の対応は既存表で確認したが、基板上の横縦座標、基板端からの距離、洗浄に対する基板の回転を示す座標表は確認できない。既存v25報告も番号の隣接が物理的位置の隣接を保証しないと明記する。端から内側への勾配の検定は未実施。番号を端距離と仮定していない。',
    '\n## 先に保存した予測',
    '- 予測書：`'+str(OUT/'261004_周3_予測と検定規定.md')+'`\n- 記録時刻：`'+prediction.RecordedAt+'`\n- セキュアハッシュアルゴリズム二百五十六ビットの内容指紋：`'+prediction.Hash+'`',
    'H：後だけの明点・暗部・線状候補近傍で外れ値密度が増え、距離で減る。半径は後画像の構造のみで推定。前にもある候補と新規または移動候補を分け、ブランクにも関連を予測。260830は候補が検出され得るが単独像なので外れ値の集中や前後変化は測れない。陽性差は前−後のため暗化が直接結合し、明部近傍の陽性には別作用が必要。',
    'C：同じ基板の帯軸が日程全体を超えて揃い、対象視野を除いた平均方向による地図が空間対照より良く予測する。後単独像で構造が増える可能性があり、前像にもあれば成形・蒸着・傷・撮影が残る。ブランクにも発生し得る。基板端からの勾配は座標記録がある場合だけ検証する。全予測・候補閾値・二十比較の補正・判定条件は予測書に固定し、全体処理後も指紋を照合した。',
    '\n## 方法',
    '候補は格子周期を抑える標準偏差3ピクセル平滑化、4分の1縮小、64ピクセル背景の除去、画像内中央値と換算済み絶対偏差で標準化し、絶対値6以上・連結面積256画素以上で検出した。線状は長軸128ピクセル以上・長短軸比4以上。傷・残渣・書き込み境界・画像縁・飽和を分類だけで区別しない。候補一覧はround3_candidate_objects.csv。新規または移動は前候補との距離32ピクセル超とした参考分類で、誤検出や対応ずれを含む。',
    'Hは候補領域への距離を16・32・64・128・256ピクセルで指数減衰させる五変数。後像の絶対局所残差が距離殻の二区間連続で1.5未満になる半径を後像だけの影響半径代理とした。後・前の距離、明暗、新規・既存の距離殻別密度はround3_distance_density.csv。半径単独の説明率はround3_radius_explained_fraction.csv。',
    'Cは周1の周期候補の第一帯軸を二倍角の円周統計で扱う。対象を除いた基板内の他候補二視野以上、集中度0.2以上を必要とした。平均軸に直交する座標の一次・二次項と周期256・512・1024ピクセルの正弦・余弦の八変数。対象自身の方向・周期・位相は選んでいない。方向集中度の倍率は標準化回帰で消えるため、集中度と説明率は別に順位相関で検定。方向由来説明率を洗浄の因果寄与率と呼ばない。',
    '両定義は周1の日程別閾値を保持。32ピクセル密度から自己以外の631視野の平均を引いた残差を使用。8分割・256ピクセル空間ブロック、64ピクセルの学習余白、学習内標準化、正則化係数1。横縦±256・±512の非循環移動八つと90度回転一つの対照を用い、全九対照と実地図が有限な共通領域で比較した。周期基底の整数調波は細い帯の証拠としない。前像、H、C、共同は同一視野の同じ有効升目で比較する。',
    '\n## 視野単位の検定結果',
    '各視野を一度だけ数え、組単位の検定は使わない。方向は基板平均方向一致から日程平均方向一致を引き、日程内で方向を基板スロットに並べ替える。空間説明率と近遠密度差は片側符号反転、日程基板一括反転を依存の感度として併記。集中度は日程内順位並べ替え。乱数種20261004、10000回、二定義合わせて20比較のボンフェローニ法。検定名のH・Cは仮説記号。画像を共有するH検定は、補正後有意でも局所画像との関連の記述である。',
    md(tests),
    '方向の事実：全周期候補の基板内一致は平均と標準偏差の定義313視野で補正後0.0220、中央値と絶対偏差の定義386視野で0.00200。ただし固定29の周期候補で基板内他候補数を満たすのは各6視野のみで、補正後はいずれも1。Cの空間対照差の補正後も両定義1で、基板内方向一致だけを帯の説明へ置き換えることはできない。',
    '距離の事実：後候補0～64ピクセルと256～512ピクセルの外れ値密度差は267視野で平均約1.184／1.277百分率点、補正後はいずれも0.00200。ブランク30視野では約0.877／0.930百分率点で、視野ごと反転の補正後0.00400／0.0280。ただし日程基板一括反転では両定義約0.0628で、同じ基板内の依存に弱い。既存の前候補でも近傍密度が高く、後の新規または移動候補だけに限定した因果効果とは判定しない。',
    '周期候補は分類上の帯状・うろこ状より広い。周1の第一方向が弱いその他も含まれるので、方向一致には分類の弱い視野が混じる。全第一軸の感度はround3_direction_sensitivity.csv、帯状・うろこ状のみの基準軸角度差はround3_axis_agreement_descriptive.csv。感度結果を確認的な20検定へ追加して根拠を増やしていない。',
    '\n## 全面と空間対照の説明率',
    md(e[e['指標'].isin(['全面決定係数','対照共通領域決定係数','対照決定係数中央値','対照との差'])&e['対象'].isin(['全利用可能','固定29利用可能'])][['定義','モデル','対象','指標','視野数','中央値','第1四分位','第3四分位','負値視野数']],percent=['中央値','第1四分位','第3四分位']),
    '全視野の個別値と九対照はround3_explained_fraction.csv。保存差再現群34と追加独立対応群の別要約はround3_explanation_summary.csv。Cは生画像のない視野にも予測可能なため、HとC単独の全利用可能集団は同じではない。個別分解を用いた共同モデルの比較だけが同じ視野・升目の比較。',
    '\n## HとCの固有部分・共通部分',
    'H固有＝共同−C、C固有＝共同−H、共通＝H＋C−共同を各視野で計算してから要約した。負の共通部分は抑制効果・予測誤差・正則化の違いを示し得る。因果配分ではない。二つの中央値を直接引いていない。',
    md(e[(e['モデル']=='共同')&e['指標'].isin(['全面決定係数_H固有','全面決定係数_C固有','全面決定係数_共通'])&e['対象'].isin(['全利用可能','固定29利用可能'])][['定義','対象','指標','視野数','中央値','第1四分位','第3四分位']],percent=['中央値','第1四分位','第3四分位']),
    '\n## 洗浄前・洗浄後・ブランク・分子なしの記述',
    md(table(OUT/'round3_image_feature_summary.csv').query("対象 in ['全画像組','ブランク','固定29利用可能'] and 指標 in ['候補数','線状候補数','影響半径代理画素','単独像方向集中度']")),
    md(table(OUT/'round3_control_features.csv')[['key','候補数','明候補数','暗候補数','線状候補数','影響半径代理画素','単独像帯軸度','単独像方向集中度']]),
    '画像の事実：286組で前候補9203個、後候補9315個。画像のみで推定した影響半径代理の中央値は前後とも64ピクセル。前からある構造が多く、検出した後候補の数が多いこと自体を新規残渣の証拠にしない。260926基板3視野3の比較図では右側と下側の線状構造が前後とも検出され、後像の下部には異なる局所構造も見える。粗視化した図の所見に限定し、ゴミや乾燥痕との材料同定や候補の承認は行っていない。',
    '候補数や単独像の方向は前後の双方に測れ、後だけの候補の存在だけで乾燥痕とは判定しない。ブランクでも候補があるが、近傍集中は上の視野単位検定を参照。260830は6枚に同じ候補検出を適用した単独像の記述であり、洗浄前後のラベルを推測せず外れ値率を捏造しない。',
    '位置6・7と他位置の候補・方向の記述比較はround3_image_feature_summary.csvに保存。書き込み境界が自動候補になり得るため、Hは仮説Aや画像周辺構造も拾う。軸一致をCの独立証拠として重ねない。',
    md(table(OUT/'round3_axis_agreement_descriptive.csv'),percent=['十度以内割合']),
    '\n## 図と目視の限界',
    '\n'.join('- `figures/'+key+'_候補と外れ値.png`：洗浄前後の生画像、候補輪郭、局所残差、二定義の密度。' for key in drawn),
    '- `figures/距離別密度の視野中央値.png`：距離殻別の視野中央値と第1～第3四分位。\n- `figures/日程別の帯軸方向.png`：180度周期の方向分布。',
    '図は比較のための自動候補の可視化で、欠陥の目視承認ではない。洗浄前後の局所構造は画像全体の粗視化による表示なので、細い傷やピラーの欠陥の確定には原寸の別確認が必要。独立対応は平滑画像相関によるアフィン写像で、局所位置合わせの残差や表面変化を分離できない。保存差再現済み34以外は差の再現を主張しない。',
    '\n## 不都合な結果・未実施・弱点・解釈',
    '事実：全286組の処理と632視野の評価を完了したが、局所距離は対応失敗した視野と生画像未コピーの日程で欠測。端勾配、水流・ブロワー方向との実一致、乾燥前線の時間変化は未測定。対応失敗を隠さず、説明率の負値と対照の方が高い場合を全て表に残した。',
    '解釈：候補の近傍が高密度でも、暗化という差の定義との結合、周辺の明線、飽和、位置合わせのずれ、ピント差が同じ関連を作り得る。Hの関連が弱ければ検出閾値6・256画素以上では拾えない広い薄い残渣も残る。洗浄前像の説明率が同程度なら後の新規付着に限定できない。確度：中（方法上の限界）、低（具体的な原因）。',
    '解釈：基板内方向一致が有意でも、水流の独立方向記録がなく、カメラ・六方格子・十字傷の軸との交絡が残る。第一帯軸はうろこ状の二方向で選択不安定。対象以外の方向による予測も同じ基板の撮影条件や閾値を共有する。方向一致から基板をまたぐ洗浄法の因果効果へ飛躍しない。基板番号・洗浄順・濃度は分離できない。',
    '方法上の弱点：全632視野共通プロファイルは保留対象の他視野の差を使用する。8分割の交差検証は視野内空間への汎化であり、未撮影基板への汎化ではない。推定位置合わせ行列と画像内の全体中央値・絶対偏差、方向は保留升目を含む画像または他視野から作る。視野内空間依存と日程基板内依存は残り、反転検定は無作為実験ではない。対照共通領域は中央約1024ピクセル幅へ縮むので、全面値と数値差を原因寄与と解釈しない。',
    '不都合な数値：H・前像・共同モデルの中央の対照共通領域で極端な負の決定係数が生じた。候補が縁に多く、短い指数減衰関数が学習領域でほぼ一定となる一方、保留領域に近い候補があると学習内標準化後の値が非常に大きくなる外挿が確認された。全面Hの最小値は約−0.064、共通領域Hの最小値は約−132797であり、全面の説明率と混同しない。距離モデルはこの中央領域への予測に失敗している。閾値・標準化・正則化を事後変更せず、全負値を保持した。平均差の反転検定は極端値の影響を強く受ける。round3_numerical_extremes.csvとround3_extrapolation_diagnostic.csvに量級と保留予測の診断を保存。',
    '\n## 実装・検証・作業状態',
    '新規コードはfield_level/v30_band_origin_round3のみ。解析関数と交差検証の数値規定を読み取りで確認し、新しい版へ必要箇所をコピーした。旧版・標準解析・既定マスクを実行変更せず、入力へ書き込んでいない。初回小範囲で実装の変数名衝突を修正して再実行し、計算規定は変更していない。小範囲の追加対応失敗は相関0.895で規定0.90未満だったためそのまま欠測。',
    f"入力と旧版成果{validation['入力指紋照合数']}ファイルの開始・終了指紋を照合し不変。学習保留と64ピクセル余白の重複0、既知移動の方向・縮小倍率、保存密度の再計算、分解恒等式を確認。分解恒等式の最大差{validation['分解恒等最大差']:.3g}。小範囲確認はround3_pilot_verification.json、全体確認はround3_verification.json。",
    f"分解恒等式の検証で絶対許容差1兆分の1は、約十万の負値の丸め誤差を過剰判定したため、値の量級に対する許容差1兆分の1へ検証条件だけを修正。量級相対最大差{validation['分解恒等量級相対最大差']:.3g}。モデル、予測、説明率、検定の値は変更していない。",
    '版確認：開始時にfield_level、data/results、解析コピー直下、ローカルブランチとタグにv30なし。v25～v27は解析コピー、v28・v29はコード・結果・タグで使用済み。遠隔の未取得版は確認できない。古い環境文書より今回の入力・書込み制限を優先し、.gitを更新する取得操作は実施していない。凍結中の別リポジトリは参照も実行もしていない。コミット・タグ・プッシュは行っていない。',
    '- 作業ブランチ：`'+git['作業ブランチ']+'`\n- 作業開始時点のコミット：`'+git['作業開始時点のコミット']+'`\n- 状態：\n```text\n'+git['短い状態']+'\n```\n- 未追跡ファイル：\n'+'\n'.join('  - `'+p+'`' for p in git['未追跡ファイル']),
    '生成画像・大容量の途中配列・入力コピーは既存の追跡除外規則に従い、Gitへ追加していない。結果全体はdata/results/v30_band_origin_round3にあり、未追跡コードとは別。全出力一覧はround3_output_inventory.csv、実行コード指紋はround3_code_provenance.csv。',
    '\n## ここまでの全仮説の説明率一覧',
    '全て同じ32ピクセル密度残差と空間交差検証による決定係数。対象視野数と説明変数は異なる。定義名は平均＋標準偏差三倍と中央値＋換算済み絶対偏差三倍を略している。未検証の仮説は未測定であり0%ではない。Gの二次面は方向を限定しない滑らかな勾配、G・Iの画像モデルは同じ画像由来の記述値。Fは独立非周期目印で検証できた2視野のみ。Aの旧位置固定検定は共通尺度の説明率を測っていない。',
    md(comparison[['仮説','仮説内容または説明変数','定義','対象','対象視野数','説明率中央値','第1四分位','第3四分位','判定','確度']],percent=['説明率中央値','第1四分位','第3四分位']),
    '\n## 次に検証すべき仮説の順序と理由（決定はClaude Code）',
    '1. 仮説F：局所位置合わせ残差。独立検証2視野という不足をまず埋める。全4日程で非周期目印または別測定による対応の正解を確保し、画像を共有するH・C・G・Iの記述関連から切り分ける。生画像未コピーの260825・260827・260829・260923・260927が必要。追加独立対応をそのまま標準補正に採用しない。',
    '2. 仮説D・E：蒸着と鋳型成形。洗浄前に存在する構造との区別に、同一基板・同一場所の独立な表面像、ピラー形状・欠損・蒸着厚みの記録を使う。単独光学画像だけを差の因果説明にしない。',
    '3. 仮説C・Hの独立再検証：水流・ブロワー方向、基板の回転、端座標、乾燥前線の動画または時系列を先に記録し、方向を操作したブランク・分子なしで予測を検定する。自動候補の前後原寸レビューを経て、欠陥承認を解析採用と切り離して記録する。',
    '4. 仮説J：実際の不均一吸着。装置・基板・局所位置合わせの交絡を減らした後、処理順と濃度を無作為化した独立基板・日程の分子あり対照で検証する。現データでは基板番号と濃度・洗浄順の交絡から分子効果を確定しない。周3の報告をここで保存し、次の周へ自動では進まない。']
    (OUT/'261004_周3_報告.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    csv([dict(実在パス=str(p),内容指紋=digest(p)) for p in sorted((ROOT/'field_level'/'v30_band_origin_round3').iterdir()) if p.is_file()],'round3_code_provenance.csv')
    csv([dict(実在パス=str(p),バイト数=p.stat().st_size,内容指紋=digest(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='round3_output_inventory.csv'],'round3_output_inventory.csv')
    print('report written',flush=True)

if __name__=='__main__':report()
