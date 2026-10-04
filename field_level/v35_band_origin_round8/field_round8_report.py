"""Round-eight tables, figures, verification and self-contained Japanese report."""
from field_round8_common import *
from field_round8_evaluate import collect,tests,GROUPS

def md(frame):
    if not len(frame):return '該当する値なし。'
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6g}' if np.isfinite(v) else '未定義'
        return str(v).replace('|','／').replace('\n',' ')
    columns=frame.columns.tolist();rows=['| '+' | '.join(map(str,columns))+' |','| '+' | '.join(['---']*len(columns))+' |']
    rows+=['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def figures():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Yu Gothic';dest=OUT/'figures';dest.mkdir(exist_ok=True);manifest=[]
    for key in CORE:
        p=OUT/'evaluation_checkpoints'/(key+'.npz')
        if not p.exists():continue
        with np.load(p) as z:data={k:z[k] for k in z.files}
        names=['real_signed','N0_signed','主帰無_signed','N1_signed'];labels=['実際の差','周7：全体変換のみ','周8：測定局所変位を追加','別視野の局所変位']
        images=[data[n] for n in names];vmax=max(float(np.nanpercentile(abs(a),99.5)) for a in images)
        fig,axes=plt.subplots(1,4,figsize=(16,4.7),layout='constrained')
        for ax,a,label in zip(axes,images,labels):
            im=ax.imshow(a,cmap='RdBu_r',vmin=-vmax,vmax=vmax,extent=[0,2048,2044,0]);ax.set_title(label,fontsize=12);ax.set_xlabel('横位置（画素）')
        axes[0].set_ylabel('縦位置（画素）');fig.colorbar(im,ax=axes,shrink=.8,label='背景除去9画素和：前から後を引く')
        fig.suptitle(key+'：四面同じ色範囲');fp=dest/(key+'_null_comparison.png');fig.savefig(fp,dpi=140);plt.close(fig)
        manifest.append(dict(key=key,path=str(fp),vmin=-vmax,vmax=vmax,同一色範囲=True))
        fig,axes=plt.subplots(2,4,figsize=(16,8.8),layout='constrained')
        for j,method in enumerate(METHODS):
            den=[data[f'real_density_{j}'],data[f'N0_density_{j}'],data[f'主帰無_density_{j}'],data[f'N1_density_{j}']];vm=max(float(np.nanpercentile(a,99.5)) for a in den)
            for ax,a,label in zip(axes[j],den,labels):
                im=ax.imshow(a,cmap='magma',vmin=0,vmax=vm,extent=[0,2048,2044,0]);ax.set_title(label);ax.set_xlabel('横位置（画素）')
            axes[j,0].set_ylabel(method+'\n縦位置（画素）');fig.colorbar(im,ax=axes[j],shrink=.8,label='外れ値割合')
        fig.suptitle(key+'：各定義の四面は同じ色範囲');fp=dest/(key+'_outlier_comparison.png');fig.savefig(fp,dpi=130);plt.close(fig)
        manifest.append(dict(key=key,path=str(fp),同一色範囲=True))
    dump(manifest,OUT/'figure_manifest.json')

def derived(s):
    signed=pd.read_csv(OUT/'signed_field_metrics.csv',dtype={'日程':str,'基板':str});density=pd.read_csv(OUT/'outlier_field_metrics.csv',dtype={'日程':str,'基板':str})
    quality=[];amplitudes=[];verification=[];paths=[]
    for p in sorted((OUT/'experiment_checkpoints').glob('*.json')):
        r=json.loads(p.read_text(encoding='utf-8'));key=r['key'];fi=r['追跡検証'];a=r['場振幅'];marker=r['目印検証']
        if key not in set(signed.key):continue
        quality.append(dict(key=key,日程=r['日程'],基板=r['基板'],固定29視野=r['固定29視野'],必須対象=r['必須対象'],支持点数=fi['支持点数'],支持区画数=fi['支持区画数'],補正前誤差二群平均画素=float(np.mean(fi['補正前誤差中央値'])),補正後誤差二群平均画素=float(np.mean(fi['補正後誤差中央値'])),二群地図差中央値画素=fi['二群地図差中央値'],往復誤差中央値画素=r['往復誤差中央値画素'],窓差中央値画素=r['窓差中央値画素'],目印差距離画素=marker['差距離画素'],目印相関=marker['十字線相関最大値'],**a))
        center=np.array([marker.get('目印中心横',np.nan),marker.get('目印中心縦',np.nan)],float)
        qrow=quality[-1]
        if np.isfinite(center).all():
            with np.load(p.with_suffix('.npz')) as z:u=z['local_field']
            point_u=np.array([map_coordinates(u[:,:,j],[[(center[1]-16)/32],[(center[0]-16)/32]],order=1,mode='nearest',prefilter=False)[0] for j in range(2)])
            native=np.linalg.solve(np.asarray(r['元変換'])[:,:2],point_u)
            offset=np.array([marker['横差画素'],marker['縦差画素']],float)
            qrow.update(目印位置局所場横画素=native[0],目印位置局所場縦画素=native[1],目印位置場との差画素=float(np.linalg.norm(native-offset)))
        else:qrow.update(目印位置局所場横画素=np.nan,目印位置局所場縦画素=np.nan,目印位置場との差画素=np.nan)
        g=signed[signed.key.eq(key)].set_index('条件');cols={}
        for var in ['N3半分','主帰無','N3二倍']:
            cols[var+'相関']=g.loc[var,'相関'];cols[var+'分散比']=g.loc[var,'分散比'];cols[var+'無調整分散再現']=g.loc[var,'無調整分散再現']
        cor=np.array([cols[v+'相関'] for v in ['N3半分','主帰無','N3二倍']]);vr=np.array([cols[v+'分散比'] for v in ['N3半分','主帰無','N3二倍']])
        ratio=cols['主帰無分散比'];factor=1/np.sqrt(ratio) if ratio>0 else np.nan
        ampl=dict(key=key,日程=r['日程'],基板=r['基板'],固定29視野=r['固定29視野'],必須対象=r['必須対象'],測定場中心化振幅画素=a['中心化二乗平均平方根画素'],参考必要倍率=factor,参考必要振幅画素=factor*a['中心化二乗平均平方根画素'],一倍相関が最大から002以内=bool(cor[1]>=np.nanmax(cor)-.02),一倍分散比が最も一に近い=bool(np.argmin(abs(np.log(np.maximum(vr,1e-12))))==1),半分対一倍分散比=vr[0]/vr[1],二倍対一倍分散比=vr[2]/vr[1],**cols)
        amplitudes.append(ampl)
        verification.append(dict(key=key,格子生成最大差=r['格子生成最大差'],実測差再現最大差=r['実測差再現最大差'],N0周7差最大差=r['条件']['N0']['周7既知変換差最大差'],最大逆写像残差=max(v.get('逆写像最大残差画素',np.nan) for v in r['条件'].values()),成功条件数=sum(v['状態']=='成功' for v in r['条件'].values()),回転対照整合済み=r['条件']['N2'].get('回転整合確認済み',False)))
        paths.append({k:r[k] for k in ['key','場実在パス','供給元','供給元場パス','洗浄前パス','洗浄後パス','保存差パス','回復記録パス']})
    for rows,name in [(quality,'field_quality.csv'),(amplitudes,'amplitude_sensitivity.csv'),(verification,'field_verification.csv'),(paths,'field_input_paths.csv')]:pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    q=pd.DataFrame(quality);summ=[]
    for label,g in [('全実施',q),('必須全体',q[q['必須対象']]),('固定29該当',q[q['固定29視野']])]:
        for col in ['補正前誤差二群平均画素','補正後誤差二群平均画素','二群地図差中央値画素','往復誤差中央値画素','窓差中央値画素','目印差距離画素','目印位置場との差画素','中心化二乗平均平方根画素','局所勾配最大']:
            v=g[col].dropna();summ.append(dict(集合=label,指標=col,視野数=len(v),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75)))
    pd.DataFrame(summ).to_csv(OUT/'quality_summary.csv',index=False,encoding='utf-8-sig')
    main=signed[signed['条件'].eq('主帰無')];fixed=main[main['固定29視野']];strong=[]
    for r in fixed.to_dict('records'):
        ds=density[density.key.eq(r['key'])&density['条件'].eq('主帰無')];ratio=ds['模擬外れ値割合']/ds['実測外れ値割合']
        strong.append(dict(key=r['key'],三成分一致=bool(r['三成分一致']),相関07以上=r['相関']>=.7,分散比適合=.5<=r['分散比']<=2,二定義率適合=bool(ratio.between(.5,2).all()),主因再現条件=bool(r['三成分一致'] and r['相関']>=.7 and .5<=r['分散比']<=2 and ratio.between(.5,2).all())))
    st=pd.DataFrame(strong);st.to_csv(OUT/'fixed_field_support.csv',index=False,encoding='utf-8-sig')
    t=pd.read_csv(OUT/'confirmatory_tests.csv');significant=False
    for name in ['必須全体','固定29該当']:
        h=t[t['集合'].eq(name)&t['定義'].eq('符号付き差')];significant|=len(h)==4 and bool(h['補正後有意確率'].lt(.05).all())
    decision=dict(確認的検定枠=50,相関四比較支持=significant,固定29評価数=len(st),固定29三成分一致数=int(st['三成分一致'].sum()),固定29主因再現条件数=int(st['主因再現条件'].sum()),主因支持=bool(significant and st['主因再現条件'].sum()>len(st)/2))
    dump(decision,OUT/'decision.json')
    return signed,density,q,pd.DataFrame(amplitudes),pd.DataFrame(summ),pd.DataFrame(verification),decision

def build():
    verify();tests();s=setup();signed,density,q,amps,quality,v,decision=derived(s);figures()
    completed=set(signed.key);f=s['f'];un=f[f['適格']&~f.key.isin(completed)].copy();un['理由']='この周の計算未実施または失敗';un.to_csv(OUT/'unfinished_fields.csv',index=False,encoding='utf-8-sig')
    eligibility=f[f['固定29視野']|f.key.isin(['260926_01_8','260926_3_3','260926_5_3','260926_7_5','260926_7_8'])][['key','日程','基板','視野番号','ブランク','外れ値割合','適格']]
    eligibility.to_csv(OUT/'fixed29_eligibility.csv',index=False,encoding='utf-8-sig')
    tt=pd.read_csv(OUT/'confirmatory_tests.csv');summary=pd.read_csv(OUT/'metric_summary.csv');ms=summary[summary['条件'].eq('主帰無')&summary['集合'].isin(GROUPS)]
    sshow=ms[ms['指標'].isin(['相関','分散比','無調整分散再現','実測外れ値割合','模擬外れ値割合','全面決定係数','共通領域決定係数','対照中央値決定係数','対照差決定係数','全面共通除去逸脱度改善','対照差共通除去逸脱度改善'])]
    base=summary[summary['集合'].eq('固定29該当')&summary['条件'].isin(['主帰無','N0','N1','N2','N3半分','N3二倍','N4'])&summary['指標'].isin(['相関','分散比','無調整分散再現','模擬外れ値割合','全面決定係数'])]
    allmain=summary[summary['集合'].eq('全適格実施（補助）')&summary['条件'].eq('主帰無')&summary['指標'].isin(['相関','分散比','無調整分散再現','実測外れ値割合','模擬外れ値割合','全面決定係数','対照差決定係数','全面共通除去逸脱度改善','対照差共通除去逸脱度改善'])]
    state=dict(ブランチ=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),状態=subprocess.check_output(['git','status','--short','--untracked-files=all'],cwd=ROOT,text=True).strip(),コミット=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());dump(state,OUT/'git_state.json')
    prior=table(s['results']['v34_band_origin_round7']/'all_hypotheses_round1_to_round7.csv')
    # Keep the seven-round judgments, append this round's limited inference.
    prior['周8後判定']=['帯状差の部分的生成能力は支持／中。測定場1倍の今回モデルによる帯全体再現は否定（非支持）／中。機構全体の主因帰属は判定不能／低' if str(x).startswith('F') else '周7判定を保持' for x in prior.iloc[:,0]]
    prior['周8の理由']=['測定局所場は一部の帯状差を物質変化なしに生成したが、振幅・計数・全体対応が未再現。全体変換のみの旧実験と区別する。場の誤差と対象選別が残る。' if str(x).startswith('F') else '局所場と整数標本化の組合せに部分整合。画素格子機構を独立に因果分離した結果ではない。' if str(x).startswith('B') else 'この周で追加の原因検定を行っていない。' for x in prior.iloc[:,0]]
    prior.to_csv(OUT/'all_hypotheses_round1_to_round8.csv',index=False,encoding='utf-8-sig')
    integrity();iv=pd.read_csv(OUT/'input_integrity_final.csv');dups=not signed.duplicated(['key','条件']).any() and not density.duplicated(['key','条件','定義']).any()
    verification=dict(予測指紋一致=True,実施視野数=len(completed),適格視野数=int(f['適格'].sum()),必須視野数=int(f['必須対象'].sum()),必須未完了数=int((f['必須対象']&~f.key.isin(completed)).sum()),格子最大差=float(v['格子生成最大差'].max()),実測差最大差=float(v['実測差再現最大差'].max()),N0周7最大差=float(v['N0周7差最大差'].max()),逆写像最大残差=float(v['最大逆写像残差'].max()),入力旧成果指紋照合件数=len(iv),入力旧成果不変=bool(iv['不変'].all()),確認的検定数=len(tt),補正算術一致=bool(np.allclose(tt['補正後有意確率'],np.minimum(1,50*tt['補正前有意確率']))),重複行なし=dups,主因支持=decision['主因支持'])
    verification.update(符号付き評価行数=len(signed),外れ値評価行数=len(density),全条件評価行数一致=bool(len(signed)==8*len(completed) and len(density)==16*len(completed)),条件成功数最小=int(v['成功条件数'].min()))
    verification['全実施視野の回転対照整合']=bool(v['回転対照整合済み'].all())
    dump(verification,OUT/'verification.json')
    core=signed[signed.key.isin(CORE)][['key','条件','相関','分散比','無調整分散再現','実測周期','模擬周期','方向差','第一軸方向差','位相差','三成分一致']]
    coreden=density[density.key.isin(CORE)][['key','条件','定義','実測外れ値割合','模擬外れ値割合','ジャカード係数','密度重なり','全面決定係数','全面共通除去逸脱度改善']]
    errors=list((OUT/'execution_errors').glob('*.json')) if (OUT/'execution_errors').exists() else []
    ca=amps[amps.key.eq(CORE[0])].iloc[0];cq=q[q.key.eq(CORE[0])].iloc[0]
    amplitude_text=(f'指定該当視野の測定場の中心化振幅は{ca["測定場中心化振幅画素"]:.3f}画素、分散比からの粗い参考必要振幅は{ca["参考必要振幅画素"]:.3f}画素（約{ca["参考必要倍率"]:.2f}倍）。1倍と一致しない。'
        f'相関は1倍{ca["主帰無相関"]:.3f}から2倍{ca["N3二倍相関"]:.3f}へ、分散比は{ca["主帰無分散比"]:.3f}から{ca["N3二倍分散比"]:.3f}へ上がり、1倍はこの三条件の最大相関から0.02以内ではない。'
        f'二群地図差中央値{cq["二群地図差中央値画素"]:.3f}画素、二群平均の保留追跡誤差{cq["補正後誤差二群平均画素"]:.3f}画素は真の誤差の上限でなく、約2倍が測定誤差内で妥当かは確認できない。平滑化による減衰の独立較正もない。'
        'したがって、実測振幅での不足は示すが、2倍化した場を真の変位として採用したり、仮説F全体を否定したりする根拠にはしない。2倍でも三成分・振幅・計数の全域再現を確認した結果ではない。')
    fixed_density=density[density['固定29視野']&density['条件'].eq('主帰無')]
    quantitative=[]
    for method,g in fixed_density.groupby('定義'):
        baseline=density[density['固定29視野']&density['条件'].eq('N0')&density['定義'].eq(method)]
        quantitative.append(f'{method}：説明率中央値{100*g["全面決定係数"].median():.3f}%（全体変換のみ{100*baseline["全面決定係数"].median():.3f}%）、模擬外れ値率中央値{100*g["模擬外れ値割合"].median():.3f}%／実測{100*g["実測外れ値割合"].median():.3f}%')
    counts=f[f['適格']].groupby('日程').agg(適格数=('key','size'),必須数=('必須対象','sum'),主ブランク数=('ブランク','sum'),通常選択数=('通常選択','sum'),固定29該当数=('固定29視野','sum')).reset_index();counts['実施数']=[int(signed[signed['日程'].eq(d)]['key'].nunique()) for d in counts['日程']]
    lines=['# 2026年10月4日・周8報告',
        '## 結論（事実と解釈を区別）',
        '測定した局所変位を加えると説明が改善する成分はあるが、実際の大きな帯の主因として採用する条件は満たさない。仮説F（局所位置合わせずれ）の判定は、帯状差を部分的に生成できることは支持（確度：中）、測定場1倍の今回モデルによる実測帯全体の再現は否定（非支持、確度：中）、機構全体の主因帰属は判定不能（確度：低）。物質変化の採用には進めない。必要性は検証していない。',
        f'実施{len(completed)}／適格179視野、必須未完了{verification["必須未完了数"]}／47。固定29の適格4視野、指定5の適格1視野、主ブランク19視野、通常選択24視野。追加視野は補助集計で、確認的検定集合を変更しない。数値照合の確度：高。',
        '固定29の適格4視野での主結果（数値の確度：高）：'+'。'.join(quantitative)+'。'+f'50検定中、補正後0.05未満は{int(tt["補正後有意確率"].lt(.05).sum())}件。形・振幅・計数の事前主因再現条件を満たしたのは{decision["固定29主因再現条件数"]}／4視野。',
        '## 先に固定した予測・方法',
        f'予測全文：[261004_周8_予測と検定規定.md](261004_周8_予測と検定規定.md)。内容指紋：`{PRED_HASH}`。保存時刻と実在パスは prediction_sha256.json。模擬計算前に保存し、終了時も照合した。',
        '主因なら周期・向き・位相・振幅・計数が全体変換のみより大きく近づき、別視野・回転・移動した場では対応が壊れる。前後単独像に帯が見える必要はなく、ブランクにも働く。分子なし260830は前後対応と適格測定場がなく未検証。',
        '保存した64×64の局所残差場を変更せず、前像位置xから後像への写像 A x + t + u(x) に注入。残差は後像座標の変位。格子・背景除去9画素和・整数丸め・前から後を引く向き・日程別二閾値を保持。模擬前後像組の全体位置合わせを読み取り専用code_v24で一度ずつ推定した。標準経路・既定値・マスクを変更していない。',
        '周7と同じ帯域制限・4倍フーリエ補間・正方形画素開口。局所場の開口内変化は省く局所並進近似であり、全ての局所光学効果を精密再現した実験ではない。逆写像は6回反復、補間は保存場間の双一次補間。N0はゼロ場を生成し周7の既知変換差との完全一致を照合後、周7の再位置合わせ済み差を再利用した。',
        'N1＝適格179視野を89だけ巡回した別視野の場。N2＝空間配置とベクトルを90度回転。N3半分・N3二倍＝振幅0.5・2倍の事前感度。N4＝横512・縦256画素の循環移動。「局所追随」は注入写像に沿って標本化する補助で、主解析は全体変換のみで標本化する。内部列「指定3視野」は周7互換の列名で、この周は指定適格1視野を意味する。',
        '## 対象・選別（事実、確度：高）',md(counts),
        '固定29の未適格25視野を含む一覧は fixed29_eligibility.csv。指定5の残り4視野は今回の条件を満たさず模擬に使わなかった。保存場が存在するだけでは採用しない。回復・支持・非周期目印確認の全条件を要求。参考ブランクを通常分子付き選択から除いた。適格179の対象選別は、帯を代表する標本にならない。260922・260923の基板01はミスマッチ配列、260926の01はブランク。基板番号と処理順・濃度の交絡は残る。',
        f'適格179の中の参考ブランクは{int((f["適格"]&f["参考ブランク"]).sum())}視野。主ブランク19に移さず、通常分子付き選択から除いた。全適格補助集計には含むため、その集合の主ブランク以外を全て分子付きとは呼ばない。',
        '## 指定該当視野の結果（事実、確度：高）',md(core),md(coreden),
        '割合は0から1（0.01＝1%）。分散比は模擬分散／実測分散。無調整分散再現は1−分散(実測−模擬)／実測分散で、相関二乗や交差検証決定係数と異なる。周波数探索の下限64画素に集まる模擬ピークには境界効果があり、周波数格子の整数調波を細い帯の証明にしない。',
        '![指定該当視野の差地図](figures/260926_7_5_null_comparison.png)',
        '![二定義の外れ値密度](figures/260926_7_5_outlier_comparison.png)',
        '## 説明できた割合と全体変換のみとの差',
        '二定義は、日程ごとのブランク平均＋標準偏差の3倍を超えるピラー、およびブランク中央値＋中央値からの絶対偏差の中央値を1.4826倍した値の3倍を超えるピラー。周1の閾値をそのまま使用し、模擬のブランクから閾値を作り直さない。',
        '32画素升目の外れ値密度から対象自身を除く全632視野の共通平均を引き、模擬符号付き差と模擬密度の二変数で空間8分割交差検証。検証周囲2升目を学習から除き、学習残差平均を基準にした決定係数。負値は保留平均より悪いことを示す。計数指標は二項モデルの共通平均基準への逸脱度改善。説明率の係数は保留データの予測用に学習するが、像の生成や場を実測差に合わせて調整しない。',md(sshow[['集合','定義','指標','視野数','中央値','第1四分位','第3四分位','負値数']]),
        '固定29該当4視野の対照・感度を含む比較：',md(base[['条件','定義','指標','視野数','中央値','第1四分位','第3四分位']]),
        '全適格実施視野の補助集計（確認的検定を追加しない）：',md(allmain[['定義','指標','視野数','中央値','第1四分位','第3四分位','負値数']]),
        '視野別全条件・対照9個の値は signed_field_metrics.csv と outlier_field_metrics.csv、集計は metric_summary.csv。全条件の欠測、負値、不都合な対照結果も保持した。密度重なり、集合交わり／和、計数順位相関も保存表に含む。',
        '## 検定結果（全50枠）',
        '片側の視野単位符号反転。18視野以下は全列挙、他は10000回と足し1補正。補正はボンフェローニ法で50倍、補正前後を併記。基板一括符号の列は同日程同基板への依存の感度。組・升目・ピラーを独立標本としていない。重複する集合にも同じ50枠の補正をかけ、後から集合や指標を選ばない。',md(tt),
        '固定4視野の最小片側有意確率は0.0625、指定1視野は0.5なので補正後有意にならない。非有意を不存在の証拠にしない。主因の判定は相関だけでなく形・分散・二定義の計数再現条件も要求した。',md(pd.read_csv(OUT/'fixed_field_support.csv')),
        '## 局所場の誤差と必要振幅（作業2）',md(quality),
        '支持二群の保留誤差、地図差、往復追跡、窓差は誤差の目安で、真の変位誤差の上限ではない。非周期目印は格子一周期の取り違えを抑える全体位相の確認であり、視野全域の局所場との一致を保証しない。追跡の保存記録に非周期検証が偽と書かれる視野も、後で確定した目印監査と採用表を照合して採用している。',
        '目印の中心座標が保存されている視野では、その位置の局所場を前像座標へ戻し、既存の整数画素の目印移動との差も記述した。整数探索・平滑化された目印なのでサブピクセル精度の独立較正ではない。周1由来の260926の2適格視野は監査表に目印中心座標がなく、点位置での場との一致を確認できない。位相確認の距離だけを局所場精度の証明にしない。',
        '指定該当視野の誤差・振幅：',md(q[q.key.isin(CORE)]),
        '固定29該当の事前振幅感度：',md(amps[amps['固定29視野']]),
        amplitude_text,
        '参考必要倍率は1倍分散比の逆平方根、参考必要振幅はそれに測定場の中心化二乗平均平方根を掛けた値。線形・場のみの変化を仮定する粗い大きさの目安で、全体変換と丸め非線形を無視する。これを使って像や場を作り直していない。相関が弱いと、振幅だけを合わせても位相は一致しない。半分対1倍・2倍対1倍の分散比が0.25・4から外れることは単純比例の限界を示す。全視野の感度は amplitude_sensitivity.csv。',
        '過去の依頼Eの約0.30画素相当という空間相関成分を、今回の中心化場振幅や保留追跡誤差と同じ量とはみなしていない。今回の必要振幅との比較は同じ保存場の振幅を基準にする。',
        '## 実装確認と模擬の限界', '```json\n'+json.dumps(verification,ensure_ascii=False,indent=2)+'\n```',
        '指定該当視野の開口積分・8倍補間の確認：',
        '```json\n'+json.dumps({k:json.loads((OUT/'experiment_checkpoints'/(CORE[0]+'.json')).read_text(encoding='utf-8'))[k] for k in ['開口感度','8倍感度']},ensure_ascii=False,indent=2)+'\n```',
        '人工余弦の座標・開口積分確認は pilot_geometry.json。局所開口変化の省略を含む実像の固定512画素面積積分との差は上記のとおり。これは元の真の連続像の回復を証明しない。撮像済み前像の失われた高周波、光学変化、実際の独立ノイズ、256画素平滑化による細かな変位の減衰が残る。場は保存差への当てはめではないが、同じ前後像の物質・照明変化が追跡を偏らせる可能性は残る。',
        '指定該当視野の図を目視確認した。四面の色範囲を統一しており、局所場を加えた像には帯状の差が生じる一方、実測帯の位置と強さを全域では再現せず、外れ値密度も低い。この目視は検定の代用にしない。',
        f'実行失敗記録は{len(errors)}件。未完了適格視野は{len(un)}件で unfinished_fields.csv に保存。未実施とモデルで再現できなかった視野を区別する。260830は既存適格場・対応がないため実測比較なし。新しい日程コピーは今回要求しない。対象不足を埋めるには指定残り4視野・固定29未適格25視野の独立測定場が必要で、未承認場を自動採用しない。',
        '運用上の処理エラーは operational_issues.json に記録。計算前に周1由来の目印監査の読込不足を修正し、小規模確認を通してから本計算した。途中報告生成で評価中の視野を参照したエラーも修正し、計算・予測・保存済み視野結果は変更していない。履歴取得の書込拒否も同記録に含む。',
        '実装レビューで90度回転対照の空間配置とベクトルの回転方向が逆になっていた誤りを検出した。既知の単位横ベクトルと一点配置で整合を確認後、この対照だけを全実施視野で再生成・再位置合わせし、評価・検定を再計算した。原予測は共に90度回す規定のまま保全し、結果に合わせた調整ではない。修正前の対照差・変換・旧記録指紋は rotation_control_before_correction/、修正記録は 261004_周8_回転対照の実装修正.md、検証は rotation_correction_verification.json。主模擬・他条件の差配列は保持した。',
        '## 解釈・別の説明・次の周への提案',
        '局所場を加えた模擬が改善する成分は、局所幾何学ずれと整数標本化の組合せが物質変化なしに差を作り得ることと整合する。ただし実測の大きな帯・外れ値数の全体を説明したとはいえない。洗浄・乾燥痕、光学条件、蒸着・成形の不均一、物質変化は区別できず残る。未再現分を分子吸着の証拠にしない。',
        '次に進むなら、前後独立の非周期目印を増やした局所場と、その誤差を独立に較正した模擬、複数画素位相の撮像による連続像モデルの検証を優先する。場の2倍化を補正として採用しない。必要な測定・対象を検収してから次周を設計する。この周では補正・除外規則・標準マスクへ導入せず止まる。',
        '## 実在パスと保全',
        '入力起点は `C:\\Users\\chuya\\dev\\PC-alignment-Codex\\data\\inputs_local`。直下と各上位を一覧して実在名を使用した。生画像直下：'+', '.join('`'+n+'`' for n in s['raw'])+'。完了印3件あり。ラボノートは平坦化された8ファイルを先に読んだ。凍結リポジトリを参照・実行していない。',
        '場・生画像・保存差・回復記録・供給元を含む各視野の絶対実在パスは field_input_paths.csv。全利用入力・旧コード・旧報告の開始終了内容指紋は input_integrity_final.csv。場は周5で集約された周1保存場・周5原結果・周5再開結果の三保存先から読み取り、作成日時で新旧を判定していない。依存コードのコピー出所は code_copy_provenance.json。',
        md(pd.read_csv(OUT/'field_input_paths.csv').query('key in @CORE')),
        '## 作業ブランチ・未追跡状況',
        f'ブランチ：`{state["ブランチ"]}`、開始・終了コミット：`{state["コミット"]}`。コミット・タグ・プッシュなし。開始時は変更・未追跡なし。版35は作業・結果・入力解析直下と全ブランチ・タグで未使用。履歴更新の確認は .git/FETCH_HEAD の書込み拒否で未実施。現在の取得済み履歴で確認した。書込みは周8のコード・結果配下だけ。結果・図・配列は既存の無視設定で通常の状態表示に出ず、Gitへ追加していない。',
        '```text\n'+state['状態']+'\n```',
        '## 周1から周8の全仮説の判定と、採用する原因（または決着しない理由）',md(prior),
        '採用する原因：大きな外れ値帯全体の主因として確定採用するものはない。局所変位による部分成分の十分性と整合する結果を、全原因の必要性に読み替えない。適格固定4・指定1という選別、測定場の誤差、連続像モデルの限界、未再現の振幅と計数が決着しない理由である。']
    report=OUT/'261004_周8_報告.md';report.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    notes=ROOT/'field_level/v35_band_origin_round8/NOTES.md'
    notes.write_text('# 周8：局所変位場を注入する帰無実験\n\n実装内容：周7の撮像・元処理・評価を新しい比較専用コードとして継承。周5で集約された検証済み保存場を固定注入し、場ゼロ・別視野・回転・振幅半分／二倍・移動の事前対照を比較。回転対照の実装誤りを既知座標・ベクトルで修正し、旧対照値を保存して全実施視野の対照と関連評価を再計算した。原予測は保全。標準経路と旧版を変更しない。\n\n結果概要：'+f'適格179中{len(completed)}、必須47中{47-verification["必須未完了数"]}を実施。報告は data/results/v35_band_origin_round8/261004_周8_報告.md。部分整合があるが帯主因の支持条件は未達。\n\n既知の問題：固定29の適格4・指定5の適格1という偏り。場の誤差、平滑化、帯域制限、局所開口変化の省略、独立撮像ノイズの限界。260830の適格場・前後対応なし。Git履歴取得は書込拒否で未実施。未実施一覧は unfinished_fields.csv。\n',encoding='utf-8')
    inventory=[dict(実在パス=str(p),バイト数=p.stat().st_size) for p in OUT.rglob('*') if p.is_file()];pd.DataFrame(inventory).to_csv(OUT/'output_inventory.csv',index=False,encoding='utf-8-sig')
    code=ROOT/'field_level/v35_band_origin_round8';pd.DataFrame([dict(実在パス=str(p),内容指紋=digest(p)) for p in code.iterdir() if p.is_file()]).to_csv(OUT/'round8_code_sha256.csv',index=False,encoding='utf-8-sig')
    import scipy
    dump(dict(Python=sys.version,数値配列=np.__version__,表処理=pd.__version__,科学計算=scipy.__version__,画像処理=cv2.__version__,論理処理器数=os.cpu_count()),OUT/'environment.json')
    print('report saved',report,flush=True)

if __name__=='__main__':build()
