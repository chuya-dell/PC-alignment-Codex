"""Audit, descriptive summaries, standalone figures, and final round-five report."""
from field_round5_common import *
from field_round5_local import plotting
from field_round5_scales import BIN_NAMES
import argparse
def subsets(g):return [('全利用可能',g),('固定29利用可能',g[g['固定29視野']]),('ブランク',g[g['ブランク']]),('分子あり',g[~g['ブランク']])]
def summaries():
    verify();rows=[]
    for hyp,file in [('J','round5_J_explained_fraction.csv'),('F','round5_F_explained_fraction.csv')]:
        a=table(OUT/file)
        for method,g in a.groupby('定義'):
            categories=[('他基板群地図の記述',g)] if hyp=='J' else [('非周期目印確認済み',g[g['非周期目印検証']]),('候補全て・補助',g)]
            for category,h in categories:
                for target,s in subsets(h):
                    for metric in ['全面決定係数','対照共通領域決定係数','対照決定係数中央値','対照との差']:
                        v=s[metric].dropna();rows.append(dict(仮説=hyp,定義=method,採用区分=category,対象=target,指標=metric,視野数=len(v),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75),負値数=int((v<0).sum())))
    csv(rows,'round5_explanation_summary.csv')
    old=table(OLD/'round1_bf_v24_recovery.csv');new=table(OUT/'round5_v24_recovery.csv');old['新規試行']=False;new['新規試行']=True
    r=pd.concat([old,new],ignore_index=True);assert not r.key.duplicated().any();csv(r,'round5_recovery_combined.csv')
    tracks=pd.concat([table(OLD/'round1_bf_tracking.csv').assign(新規試行=False),table(OUT/'round5_F_tracking.csv').assign(新規試行=True)],ignore_index=True)
    tracks=tracks.set_index('key')
    markers=table(OUT/'round5_F_explained_fraction.csv').groupby('key')['非周期目印検証'].first()
    r['F支持条件']=[bool(tracks.loc[k,'採用候補']) if k in tracks.index else False for k in r.key]
    r['F非周期目印確認']=[bool(markers.loc[k]) if k in markers.index else False for k in r.key]
    csv(r,'round5_recovery_and_F_gates.csv')
    coverage=[]
    for (date,blank,outlier),g in r.groupby(['日程','ブランク','固定29視野']):
        coverage.append(dict(日程=date,ブランク=blank,固定29視野=outlier,試行視野数=len(g),回復視野数=int(g['回復'].sum()),支持条件視野数=int(g['F支持条件'].sum()),目印確認視野数=int(g['F非周期目印確認'].sum())))
    csv(coverage,'round5_recovery_coverage.csv')
def comparison():
    verify();a=table(PREV/'round4_all_hypotheses_comparison.csv')
    a=a[a['対象'].isin(['全利用可能','固定29利用可能'])|a['仮説'].isin(['A','D','E'])].copy()
    a['周5補足']='周4値を保存。低い値は原因不存在の根拠にしない。'
    # Preserve previous F rows and distinguish the extended model population.
    a.loc[a['仮説']=='F','仮説']='F（周1モデル集計）'
    a=a[a['仮説']!='J']
    extra=[];summary=table(OUT/'round5_explanation_summary.csv')
    noise=table(PREV/'round4_noise_ceiling.csv').query("分割 == '格子行偶奇'").groupby(['key','定義'],as_index=False)['双方向平均決定係数'].first()
    injection=table(PREV/'round4_injection.csv')
    for hyp,file,cat in [('J','round5_J_explained_fraction.csv','他基板群地図の記述'),('F（周5確認済み）','round5_F_explained_fraction.csv','非周期目印確認済み'),('F（候補全て・補助）','round5_F_explained_fraction.csv','候補全て・補助')]:
        data=table(OUT/file)
        if cat=='非周期目印確認済み':data=data[data['非周期目印検証']]
        for method,g in data.groupby('定義'):
            for target,h in subsets(g)[:2]:
                v=h['全面決定係数'];keys=set(h.key)
                n=noise[(noise['定義']==method)&noise.key.isin(keys)]['双方向平均決定係数']
                inj=injection[(injection['定義']==method)&injection.key.isin(keys)&(injection['目標追加割合']==.03)]
                inj_pre=inj[inj['注入法']=='判定前']['全面決定係数'];inj_post=inj[inj['注入法']=='判定後']['全面決定係数']
                extra.append(dict(仮説=hyp,定義=method,対象=target,対象視野数=len(h),説明率中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75),判定='判定不能。群平均の予測性能のみ' if hyp=='J' else '局所残差の利用可能部分で検定。候補だけは補助',確度='低',独立性='対象基板を除く群平均。因果帰属は不能' if hyp=='J' else '周1の独立支持・目印条件を保持',仮説内容または説明変数='日程・分子ラベル別の他基板平均地図' if hyp=='J' else '局所変位二成分・二乗・斜面との積',天井一致視野数=len(n),同一集合半数天井中央値=n.median(),天井で割った値=v.median()/n.median() if len(n) and n.median()!=0 else np.nan,同一集合3ポイント人工帯判定前中央値=inj_pre.median(),同一集合3ポイント人工帯判定後中央値=inj_post.median(),周5補足=cat))
    a=pd.concat([a,pd.DataFrame(extra)],ignore_index=True);csv(a,'round5_all_hypotheses_comparison.csv')
def verification():
    verify();a=table(OUT/'round5_molecule_field_metrics.csv');j=table(OUT/'round5_J_explained_fraction.csv');sc=table(OUT/'round5_structure_by_field.csv');bins=table(OUT/'round5_scale_by_field.csv')
    assert len(a)==1264 and len(j)==1264 and len(sc)==632*8 and len(bins)==632*8*6
    assert not j.duplicated(['key','定義']).any()
    assert len(table(OUT/'round5_J_tests.csv'))==8
    cluster=table(OUT/'round5_cluster_fields.csv')
    five={'260926_01_8':1414,'260926_3_3':203,'260926_5_3':408,'260926_7_5':497,'260926_7_8':1477}
    for key,n in five.items():
        got=int(cluster[(cluster.key==key)&(cluster['定義']==METHODS[0])]['最大陽性塊'].iloc[0]);assert got==n,(key,got,n)
    rec=table(OUT/'round5_v24_recovery.csv')
    for _,r in rec[rec['回復']].iterrows():assert r['ピラー識別一致'] and r['洗浄前座標一致'] and r['差最大絶対差']<=1e-6
    primary=sc[(sc['分割']=='格子行偶奇')&(sc['窓']=='ハニング窓')]
    assert np.isfinite(primary['全共通成分']).all()
    fs=table(OUT/'round5_F_explained_fraction.csv');assert not fs.duplicated(['key','定義']).any()
    gates=table(OUT/'round5_recovery_and_F_gates.csv');confirmed=set(fs[fs['非周期目印検証']].key)
    assert confirmed<=set(gates[gates['F支持条件']&gates['F非周期目印確認']].key)
    dump(dict(二定義632視野一致=True,八層別検定=True,説明率全視野保存=True,波長分解全視野全条件保存=True,五視野最大陽性塊過去値一致=five,回復基準違反数=0,新規回復試行数=len(rec),新規回復数=int(rec['回復'].sum()),F確認的採用の支持条件一致=True,高周波負値保持数=int((bins['共通成分']<0).sum()),一次分解全共通成分非正数=int((primary['全共通成分']<=0).sum())),OUT/'round5_verification.json')
def plots():
    verify();plt=plotting();dest=OUT/'figures';dest.mkdir(exist_ok=True)
    a=table(OUT/'round5_scale_summary.csv').query("分割 == '格子行偶奇' and 窓 == 'ハニング窓'")
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharey=True)
    for ax,target in zip(axes.flat,['全視野','固定29視野','ブランク','分子あり']):
        for mi,method in enumerate(METHODS):
            g=a[(a['対象']==target)&(a['定義']==method)].set_index('波長区間').loc[BIN_NAMES]
            center=g['中央値'].to_numpy()*100;err=np.stack([center-g['第1四分位'].to_numpy()*100,g['第3四分位'].to_numpy()*100-center])
            ax.errorbar(np.arange(6)+(mi-.5)*.10,center,yerr=err,marker='o',capsize=3,label=method)
        ax.axhline(0,color='gray',lw=.7);ax.set_xticks(range(6),BIN_NAMES,rotation=20);ax.set_title(target);ax.set_ylabel('共通成分の割合（%）：視野中央値と四分位');ax.set_xlabel('波長（ピクセル）')
    axes[0,0].legend();fig.tight_layout();fig.savefig(dest/'波長別共通成分.png',dpi=160);plt.close(fig)
    s=table(OUT/'round5_structure_summary.csv').query("分割 == '格子行偶奇' and 窓 == 'ハニング窓' and 対象 == '固定29視野'")
    labels=['周期近傍割合','周期支持区間割合','波長512超割合','塊関連寄与割合'];fig,ax=plt.subplots(figsize=(10,5))
    for mi,method in enumerate(METHODS):
        g=s[s['定義']==method].set_index('指標').loc[labels];center=g['中央値'].to_numpy()*100;err=np.stack([center-g['第1四分位'].to_numpy()*100,g['第3四分位'].to_numpy()*100-center]);ax.errorbar(np.arange(4)+(mi-.5)*.12,center,yerr=err,marker='o',capsize=4,label=method)
    ax.set_xticks(range(4),['周期ピーク近傍','周期支持の広い区間','波長512超','492本以上の塊関連']);ax.set_ylabel('共通成分に対する割合（%）');ax.set_title('固定29視野：中央値と四分位。区分間の重なりあり');ax.legend();ax.axhline(0,color='gray',lw=.7);fig.tight_layout();fig.savefig(dest/'周期と大塊の比較.png',dpi=160);plt.close(fig)
def md(df,columns=None,percent=None):
    df=df.copy() if columns is None else df[columns].copy()
    for c in percent or []:df[c]=df[c].map(lambda x:f'{100*x:.3f}%' if pd.notna(x) else '未算出')
    def fmt(x):
        if pd.isna(x):return '未算出'
        if isinstance(x,(float,np.floating)):return f'{x:.6g}'
        return str(x).replace('|','／').replace('DNA','デオキシリボ核酸').replace('True','はい').replace('False','いいえ')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
def report():
    verify();pre=json.loads((OUT/'round5_preflight.json').read_text(encoding='utf-8'));jt=table(OUT/'round5_J_tests.csv');es=table(OUT/'round5_explanation_summary.csv');ft=table(OUT/'round5_F_tests.csv');rec=table(OUT/'round5_recovery_and_F_gates.csv');ss=table(OUT/'round5_structure_summary.csv');bs=table(OUT/'round5_scale_summary.csv');allhyp=table(OUT/'round5_all_hypotheses_comparison.csv')
    pred=table(OUT/'prediction_sha256.csv').iloc[0];lines=[]
    def add(x):lines.append(x)
    add('# 2026年10月4日・周5報告\n')
    add('## 結論と判定\n\n仮説J（分子の実際の不均一吸着）は、今回の「分子ありで四指標が高い」という予測を支持する結果が得られなかった。八検定はいずれもボンフェローニ補正後1。吸着という原因全体は**判定不能、確度：低**。日程内視野単位の計算結果の確度は高。濃度依存性は検定していない。ブランクにも再現する構造があるため、分子吸着だけによる説明は十分ではない。')
    significant=ft[ft['ボンフェローニ補正後']<=.05]
    add('仮説F（位置合わせの局所的な位相ずれ）の追加回復・独立支持・目印確認を下記の段階別に実施した。'+('今回の線形予測モデルは空間対照を有意に上回らなかった（モデルの判定：支持されない、確度：中）。原因全体は判定不能、確度：低。' if significant.empty else '採用集合の一部で空間対照より改善した。該当する定義・対象は作業2の検定表に示す。原因全体への帰属は判定不能、確度：低。')+'低い説明率だけで原因を否定しない。今回のモデルが保留升目の平均を改善するかと、原因そのものの支持を分ける。')
    add('波長分解は原因判定を目的としない記述である。固定29視野の共通成分は128〜512ピクセルに多い。周期の狭い近傍、周期を含む広い区間、大塊の寄与は重なり、足して100%にすることはできない。大塊の影響は二つの外れ値定義で大きく異なる。')
    add('## 確認した入力・運用（事実、確度：高）\n')
    add('`AGENTS.md`、`docs/REPOSITORY_CONVENTIONS.md`、`docs/LOCAL_ENVIRONMENT_20260926.md`を読み、最新の入力限定・書き込み限定指示を優先した。入力起点の直下を一覧し、実在名を用いた。ラボノートのコピーは元の10_引き継ぎ・05_解析の階層が平坦化されており、関連する八ノートを参照。大塊基準は解析結果内の9月29日報告書と再現資料を参照し、492本と六近傍を保持した。欠陥の未記載を欠陥なしと判断していない。')
    raw=json.loads((OUT/'round5_raw_final_inventory.json').read_text(encoding='utf-8'))
    add('終了確認時の生画像直下の実在名：\n\n'+ '\n'.join('- `'+n+'`' for n in raw['生画像直下']))
    add('開始時点の第三コピー完了印：'+('あり' if pre['第三コピー完了'] else '**なし**')+'。初めに260828・260922・260924を実施し、終了確認中に`_copy_done_3.txt`が現れたため、予測文に先に固定した条件に従い260923・260927・260829・260825・260827を追加対象とした。260926の周1結果は読み取りで併用した。条件成立時の記録は`round5_copy3_activation.json`。完了印だけで完全とは扱わず、元台帳の実在ファイル名と前後両像の存在を視野別に照合した。')
    add('前後画像の所在（確認時点の記述）：\n\n'+md(table(OUT/'round5_raw_pair_summary.csv'))+'\n\n不足する正確なファイル名と実在日程フォルダは`round5_raw_pair_inventory.csv`。前後の片方のみ存在する場合や、派生画像しかない場合は回復に使わない。コピー到着中に存在数が変わり得るため、回復試行済み集合と最後の所在確認集合を区別する。')
    source=Path(pre['解析入力']);add('使用した主な実在絶対パス：\n\n'+ '\n'.join('- `'+str(p)+'`' for p in [LOCAL,source,source/'tables/cached_field_differences',source/'tables/table_registration_field_qc.csv',source/'tables/table_alignment_quality_features_and_sensitivity_by_field.csv',source/'20260929_外れ値解析/20260929_報告書.md',source/'20260929_外れ値解析/20260929_再現資料/20260929_依頼H集計.py',LOCAL/'code_v24',LOCAL/'lab_notes',OLD/'round1_fields_both_definitions.csv',OLD/'round1_spatial_classification.csv',OLD/'common_profile_residuals.npz',PREV/'split_maps.npz',PREV/'round4_noise_ceiling.csv',PREV/'round4_injection.csv']))
    add('全使用候補ファイルの実在絶対パス、開始内容指紋は`round5_input_integrity.csv`、終了照合は`round5_input_integrity_final.csv`。第三コピー到着時の追加指紋は`round5_copy3_input_integrity.csv`、その終了照合は`round5_copy3_input_integrity_final.csv`。さらに遅く到着したファイルは`round5_late_copy_inventory.csv`に別記し、開始時不変の検証対象と混同しない。原コードの関数コピーと指紋は`method_copy_provenance.json`。生画像前後の実在パスと差保存先は視野別回復記録、視野全体の対応は周4の`round4_field_manifest.csv`にある。全指紋表の全てを解析に用いたわけではなく、入力と旧版の保全監査範囲も含む。')
    add('## 先に保存した予測と規定\n\n予測本文：`261004_周5_予測と検定規定.md`。256ビット暗号学的内容指紋：`'+pred.Hash+'`。全段階で指紋照合した。')
    add('予測J：同日程の分子ありで、外れ値割合、帯状・うろこ状候補割合、二分割再現構造、残差分散が高い。濃度単調増加は予測しない。前像のみの構造は吸着と整合し得るが輝度だけでは分子と同定できず、洗浄後にも残れば差は弱まり得る。ブランクや260830に構造があれば他原因が残る。')
    add('予測F：独立追跡で再現した局所変位と前像斜面から作る地図が外れ値地図を予測し、空間対照より改善する。ブランクにも生じ得る。単独像だけでは変位差を測れない。260830は単独像六枚で前後対応不明のため今回の差検定に含めず、周4の記述だけを保持した。')
    add('仮説Jの八検定群と仮説Fの周1と同じ補正倍率8を別々に固定した。波長分解・説明率の群集計は記述であり、有意確率を増やしていない。')
    add('## 作業1：分子あり対ブランク\n\n全632視野、ブランク69、分子あり563、九日程。ミスマッチ配列も分子あり。260922・260923の基板01はミスマッチ、260926の01はブランクとして台帳を照合した。二定義の閾値は周1の固定値を保持した。周1全面分類の帯状候補またはうろこ状候補を合計した二値指標を主比較とし、個別割合も保存。二分割再現性は周4の格子行偶奇・双方向平均決定係数。残差分散は対象を除く全視野平均を引いた32ピクセル密度地図の升目分散。')
    add('各日程の群平均差を等重みで平均し、日程内で視野ラベルを10,000回並べ替えた片側検定。群数固定。補正前後と帰無分布の臨界値を全て示す。差の正は分子ありで高い。割合は下表では0〜1、例えば0.001は0.1百分率ポイント。\n\n'+md(jt))
    for method in METHODS:
        d=table(OUT/'round5_J_by_date.csv');d=d[d['定義']==method]
        z=d.pivot(index='日程',columns='指標',values='分子あり引くブランク').reset_index()
        add('### '+method+'：日程ごとの差（記述）\n\n'+md(z))
    add('日程別の両群平均・中央値・視野数は`round5_J_by_date.csv`、全視野の割合・分類・二分割再現性・分散は`round5_molecule_field_metrics.csv`。日程別比較は検定を行っていない。')
    add('### 仮説Jの説明できた割合\n\n対象基板全視野を学習から外し、日程・ラベル別の他基板平均密度から学習群共通平均を引いた予測地図を作った。同日同群を学習できない主ブランクは他日程の同ラベルを使うため、日程固有のブランク形状は独立検証できない。学習地図に対象の差を混入しない。空間8分割、8×8升目のブロック、評価周囲2升目除外、正則化強さ1、学習平均基準の決定係数。九空間対照は±256・±512ピクセルの横縦非循環移動と90度回転。同じ有効領域で対照との差を比較した。\n\n'+md(es[es['仮説']=='J'],percent=['中央値','第1四分位','第3四分位']))
    add('視野別全数値は`round5_J_explained_fraction.csv`。群平均で説明された変動を分子起源の因果割合とみなせない。ブランクの再利用による閾値依存と、基板番号・濃度・洗浄順の交絡が残る。検出感度の参考加算シフトは帰無臨界値から20百分位を引いた量で、等分散の加算近似である。二値割合の実験的検出力を保証しない。人工帯較正は後述。')
    add('## 作業2：元位置合わせの回復と局所残差\n\n元の`code_v24`の呼出しを保持し、粗いアフィン推定、微小位置合わせ、前像格子生成、整数丸めによる標本化を一度だけ実行。出力を合わせる探索はしていない。識別完全一致・前座標最大差0.000001以内・全ピラー差最大絶対差0.000001以内の三条件を保持。診断値は独立監査列とし採用基準を緩めなかった。明部マスク無効の元呼出しでも診断に非ゼロ割合が出る元コードの挙動を変更していない。')
    counts=rec.groupby('日程').agg(試行視野数=('回復','size'),回復視野数=('回復','sum'),固定29試行=('固定29視野','sum'),F支持条件視野数=('F支持条件','sum'),F目印確認視野数=('F非周期目印確認','sum')).reset_index()
    add(md(counts))
    add('ブランク・固定29による偏り：\n\n'+md(table(OUT/'round5_recovery_coverage.csv')))
    add('21・31画素の二窓、前後逆追跡誤差0.2画素以下、窓間差0.2画素以下、最終変換から2画素以下、100点以上、8×8空間区画の半分以上を保持。交互の支持点二群から帯域幅256画素の残差場を作り、他群への誤差改善と地図差中央値0.2画素以内を要求。新規の非周期十字参照線は全景・拡大を目視し、周1と同じ4画素平滑化後の対応誤差が半周期3.643画素未満であることを要求した。目印をマスクや除外には使っていない。')
    add('目視記録は`manual_marker_locations.json`、採用監査は`round5_F_marker_audit.csv`、全景と拡大は`marker_sheets/`・`marker_crops/`。目印未確認は補助に限定。支持条件を通過した全候補の説明率と、目印確認済みだけの説明率を分離した。')
    add('視野単位10,000回符号並べ替えで、実地図と九対照中央値との差を検定。基板・日程共通符号は感度比較である。\n\n'+md(ft))
    add('仮説Fの説明率・全ての負値を含む集計：\n\n'+md(es[es['仮説']=='F'],percent=['中央値','第1四分位','第3四分位']))
    add('判定は確認的利用可能集合に限定する。目印確認は縁の一点であり、視野全域の位相の正しさを保証しない。回復可能性・支持条件・目印の有無による選別がある。母集団632視野と固定29視野全体で完了したとは呼ばない。元差と一致した丸め後標本値から、丸め前座標が一意と断言しない。')
    add('## 作業3：再現構造の大きさ別分解（記述、数値の確度：高）\n\n各半分の他視野共通平均を除いた残差を平均除去し、二次元ハニング窓で高速フーリエ変換した。二地図の複素スペクトル積の実部を共通成分とし、直流以外を六区間へ分けた。実数変換の多重度を補正。符号付き成分を用い、負の区間を零に切り詰めない。比は独立反復撮影の信頼性ではない。\n\n![波長別共通成分](figures/波長別共通成分.png)')
    primary=bs[(bs['分割']=='格子行偶奇')&(bs['窓']=='ハニング窓')]
    add(md(primary[['定義','対象','波長区間','視野数','中央値','第1四分位','第3四分位','共通成分合計比','負の区間共通成分数']],percent=['中央値','第1四分位','第3四分位','共通成分合計比']))
    add('### 周期と大きな塊\n\n周期は周1の分類と自己相関・スペクトル周期一致を必要とし、両半分の同じ近傍でピークが残る軸だけを数えた。ピーク近傍は軸±15度・逆周期±基本周波数一格子幅。周波数格子への量子化や整数調波だけを帯の根拠にしていない。広い支持区間割合は周期以外を含む上限的な範囲であり、周期近傍も窓漏れ・塊の周波数成分を含み得る。\n\n![周期と大塊](figures/周期と大塊の比較.png)')
    q=ss[(ss['分割']=='格子行偶奇')&(ss['窓']=='ハニング窓')]
    add(md(q[['定義','対象','指標','視野数','中央値','第1四分位','第3四分位','共通成分加重合計比','共通周期支持視野数','負値数','一超数']],percent=['中央値','第1四分位','第3四分位','共通成分加重合計比']))
    add('大塊は六方格子六近傍で連結した陽性ピラー492本以上。塊だけの密度の共通平均も他視野から引いた。塊関連寄与は「全体の二分割共通成分−塊除去後の二分割共通成分」を全体で割る。塊と残部の交差寄与を含むので塊単独パワーと異なり、零未満もあり得る。小数の外れ値視野に大塊が偏るため、視野中央値とエネルギー合計比の差が大きい。中央値定義では492基準の独立較正をしていない。周1既知五視野の最大塊1414・203・408・497・1477本は完全再現した。標準経路で塊を除外していない。')
    sens=ss[(ss['対象']=='固定29視野')&ss['指標'].isin(['周期近傍割合','周期支持区間割合','波長512超割合','塊関連寄与割合'])]
    add('固定29視野の窓なし・市松感度比較：\n\n'+md(sens[['定義','分割','窓','指標','中央値','第1四分位','第3四分位']],percent=['中央値','第1四分位','第3四分位']))
    add('32ピクセル升目では軸方向の64ピクセル未満を分解できない。斜め周波数の64以下区間は約45〜64ピクセルで、ピラー周期7.286ピクセルそのものを測る区間ではない。ハニング窓は縁を弱め、窓なしは不連続な画像境界による漏れが増える。周波数区間は純粋な形の分離ではない。極低周波が主部であると一括して断言できず、128〜512の成分に周期と非周期の両方が入る。')
    add('全視野別の区間数値と負値は`round5_scale_by_field.csv`、周期・塊の視野別値は`round5_structure_by_field.csv`、全四群・二分割・窓の集計は`round5_scale_summary.csv`と`round5_structure_summary.csv`。')
    add('## 較正・弱点・残る説明\n\n周4の固定29視野の二分割決定係数天井は平均標準偏差78.45%、中央値絶対偏差82.54%。既知の形でも、3百分率ポイント追加の人工帯の説明率中央値は条件ごと0.58〜2.59%、10ポイントは7.25〜19.92%。今回の数%の説明率だけで原因を否定しない。この較正は空間モデルの尺度で、日程内群間検定の検出力とは別。群間検定の条件付き検出感度参考は作業1の表に示した。')
    add('基板番号は洗浄順と1対1、濃度とも交絡し、ブランク基板は少ない。視野単位の並べ替えは依頼どおりだが、基板内依存のため視野交換可能性は保証されない。ブランク由来閾値をブランク自身へ使う再利用、共通平均の学習、二分割で撮影を共有する雑音も残る。原因Jの群地図と原因Fの追跡は異なる範囲の形を説明する。説明率は欠陥除外可能な本数の割合ではない。')
    add('別の説明として、丸め誤差と局所残差の非線形な相互作用、前後の光学条件差、局所的な乾燥・残渣・傷の影響を残す。蒸着・成形は周4単独像帰属の方法限界のため判定不能。ラボノートの過去の「分子由来ではない」という解釈は記録として参照するが、今回の群間結果だけでその確定度を引き上げない。')
    pairs=table(OUT/'round5_raw_pair_inventory.csv');attempted=set(rec.key)
    missing=pairs[~pairs['前後組存在']]
    not_attempted=pairs[pairs['前後組存在']&~pairs.key.isin(attempted)]
    add('## 確認できなかった項目と実行上の問題\n\n終了時点で前後画像組が不足する視野は'+str(len(missing))+'。日程別件数と必要ファイル名は上記所在表と`round5_raw_pair_inventory.csv`に記録。最後の所在照合で組が存在していて回復未試行の視野は'+str(len(not_attempted))+'。260830の前後対応、分子そのものの光学同定、独立反復撮影の天井、基板番号と洗浄順・濃度の因果分離は確認できていない。旧版の単独像帰属の限界を解消する処理は今回の対象外。')
    add('小規模確認でコピーした監査関数の読み込み漏れが二件発生し、失敗記録を`pilot_execution_errors/`に残して修正後に同じ推定条件で再実行した。波長スクリプトの構文誤りは計算開始前に修正。群地図の空の縁升目に対する警告は有効領域で除外されており、数値を埋めていない。探索したファイル名が存在しなかった場合は実在一覧で修正し、元入力を推測で代用していない。小規模確認と全体監査は`round5_J_pilot.json`・`round5_F_pilot.json`・`round5_scale_pilot.json`・`round5_verification.json`に保存した。')
    add('## 作業状態・保存先\n\n新規コードは`field_level/v32_band_origin_round5/`、全生成物は`data/results/v32_band_origin_round5/`。開始時に解析入力直下、版フォルダ、ブランチ・タグを確認しv32未使用。作業ブランチ：`'+pre['作業ブランチ']+'`。開始コミット：`'+pre['開始コミット']+'`。コミット・タグ・プッシュは実施していない。書き込み範囲を固定するため取得による履歴更新も実施していない。')
    status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True);dump(dict(ブランチ=pre['作業ブランチ'],状態=status),OUT/'round5_git_state.json')
    add('未追跡・変更状況：\n\n```text\n'+status.strip()+'\n```\n\n結果は追跡除外の設定があるため、上記一覧だけで生成物の有無を判断しない。旧版と入力の終了内容指紋照合の結果は保全監査表を参照。')
    add('## 次に検証する順序と理由\n\n1. 仮説B・F（丸め・局所残差）の非線形閾値応答と相互作用。今回回復できた別日程と独立支持を利用し、次の予測を別周で固定する。天井や人工帯の較正から、線形モデルの低い説明率だけでは判断できないため。\n2. 仮説G・I（光学条件と撮影条件）。露光・ピント記録または反復撮影で条件を独立に測り、画像を共有する説明変数の自己説明を減らす。\n3. 仮説C・H（乾燥・残渣・傷）。大塊関連寄与と定義感度が大きいため、承認済み傷位置と洗浄方向の一次記録を集めて空間的予測を作る。\n4. 仮説D・E（蒸着・成形）。単独像の第一ピークだけでの帰属が難しく、独立した形状・厚みの測定が必要。\n5. 仮説J（実吸着）。基板・洗浄順の交絡を解く分子あり／なしの無作為配置または独立実験で再検証する。仮説A（位置固定）は縁構造の対照として維持し、帯全体の因果帰属を再主張しない。')
    add('## 全仮説の説明率一覧（天井・人工帯の欄付き）\n\n仮説の記号は原因の区別：A＝位置関係、B＝画素格子と六方格子のうなり、C＝洗浄・乾燥、D＝蒸着、E＝成形、F＝局所位置ずれ、G＝ピント・照明、H＝ゴミ・シミ・傷、I＝撮影順・条件、J＝実際の分子吸着。複数行のG・H・Fは異なる予測モデルまたは採用範囲。周4の値は保持し、判定文は周4検収の「今回のモデルで保留平均を改善しない」という意味に限定する。')
    add(md(allhyp[['仮説','定義','対象','対象視野数','説明率中央値','第1四分位','第3四分位','同一集合半数天井中央値','同一集合3ポイント人工帯判定前中央値','同一集合3ポイント人工帯判定後中央値']],percent=['説明率中央値','第1四分位','第3四分位','同一集合半数天井中央値','同一集合3ポイント人工帯判定前中央値','同一集合3ポイント人工帯判定後中央値']))
    add('詳細な判定・独立性・同一集合天井比は`round5_all_hypotheses_comparison.csv`。天井比は記述比であり原因の寄与率ではない。未算出を零とみなさない。ここで周5を終了し、次周の処理は開始しない。')
    (OUT/'261004_周5_報告.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('report written',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['summaries','comparison','verification','plots','report']);a=ap.parse_args();globals()[a.stage]()
