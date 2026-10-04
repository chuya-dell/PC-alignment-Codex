"""Publish a self-contained final report while retaining the interrupted report."""
from field_round8_resume import configure,RESUME,PRIOR,ROOT,CODE,verify_preserved
c=configure()
from field_round8_common import *
from field_round8_resume_report import md

def finish():
    signed=pd.read_csv(RESUME/'signed_field_metrics.csv',dtype={'日程':str,'基板':str})
    outliers=pd.read_csv(RESUME/'outlier_field_metrics.csv',dtype={'日程':str,'基板':str})
    metrics=['相関','共通相関','相関対照差','分散比','無調整分散再現','実測周期','模擬周期','方向差','位相差','三成分一致']
    main=signed[signed['条件'].eq('主帰無')][['key','日程','基板','固定29視野','必須対象']+metrics]
    baseline=signed[signed['条件'].eq('N0')][['key']+metrics]
    improvements=main.merge(baseline,on='key',suffixes=('_局所場追加','_全体変換のみ'),validate='one_to_one')
    for col in ['相関','共通相関','相関対照差','分散比','無調整分散再現']:
        improvements[col+'改善差']=improvements[col+'_局所場追加']-improvements[col+'_全体変換のみ']
    improvements.to_csv(RESUME/'global_vs_local_signed_by_field.csv',index=False,encoding='utf-8-sig')
    cols=['実測外れ値割合','模擬外れ値割合','ジャカード係数','密度重なり','全面決定係数','対照差決定係数','全面共通除去逸脱度改善','対照差共通除去逸脱度改善']
    left=outliers[outliers['条件'].eq('主帰無')][['key','日程','基板','定義','固定29視野','必須対象']+cols]
    right=outliers[outliers['条件'].eq('N0')][['key','定義']+cols]
    comparison=left.merge(right,on=['key','定義'],suffixes=('_局所場追加','_全体変換のみ'),validate='one_to_one')
    for col in cols[1:]:comparison[col+'改善差']=comparison[col+'_局所場追加']-comparison[col+'_全体変換のみ']
    comparison.to_csv(RESUME/'global_vs_local_outliers_by_field.csv',index=False,encoding='utf-8-sig')
    state=json.loads((RESUME/'resume_initial_state.json').read_text(encoding='utf-8'))
    correction=json.loads((RESUME/'rotation_correction_verification.json').read_text(encoding='utf-8'))
    verification=json.loads((RESUME/'verification.json').read_text(encoding='utf-8'))
    rotationtests=pd.read_csv(RESUME/'rotation_before_after_tests.csv')
    rotationfields=pd.read_csv(RESUME/'rotation_before_after_field_metrics.csv')
    tests=pd.read_csv(RESUME/'confirmatory_tests.csv')
    group=signed[signed['必須対象']&signed['条件'].eq('主帰無')]
    group_summary=[]
    for condition in ['主帰無','N0','N1','N2','N4']:
        g=signed[signed['必須対象']&signed['条件'].eq(condition)]
        for col in ['相関','共通相関','相関対照差','分散比','無調整分散再現']:
            v=g[col];group_summary.append(dict(条件=condition,指標=col,視野数=len(v),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75)))
    condition_summary=pd.DataFrame(group_summary)
    condition_summary.to_csv(RESUME/'required_signed_conditions_summary.csv',index=False,encoding='utf-8-sig')
    summary=pd.read_csv(RESUME/'metric_summary.csv')
    all_signed=signed[signed['条件'].eq('主帰無')]
    amps=pd.read_csv(RESUME/'amplitude_sensitivity.csv')
    amplitude_summary=[]
    for title,frame in [('必須47視野',amps[amps['必須対象']]),('全適格179視野',amps)]:
        amplitude_summary.append(dict(集合=title,視野数=len(frame),一倍相関が三条件最大から002以内数=int(frame['一倍相関が最大から002以内'].sum()),一倍分散比が最も一に近い数=int(frame['一倍分散比が最も一に近い'].sum()),測定中心化振幅中央値画素=frame['測定場中心化振幅画素'].median(),参考必要倍率中央値=frame['参考必要倍率'].median(),参考必要振幅中央値画素=frame['参考必要振幅画素'].median(),半分対一倍分散比中央値=frame['半分対一倍分散比'].median(),二倍対一倍分散比中央値=frame['二倍対一倍分散比'].median()))
    pd.DataFrame(amplitude_summary).to_csv(RESUME/'amplitude_sensitivity_summary.csv',index=False,encoding='utf-8-sig')
    # An independent check of the unchanged pilot and the corrected rotation axes.
    from field_round8_experiment import rotate_field, inverse_points, sample_field
    unit=np.zeros((64,64,2));unit[:,:,0]=1
    assert np.array_equal(rotate_field(unit),np.broadcast_to([0.,-1.],unit.shape))
    marker=np.zeros((64,64,2));marker[10,20]=[1.,2.]
    assert np.array_equal(rotate_field(marker)[43,10],[2.,-1.])
    assert np.array_equal(np.load(RESUME/'experiment_checkpoints'/(CORE[0]+'.npz'))['local_field'],np.load(PRIOR/'experiment_checkpoints'/(CORE[0]+'.npz'))['local_field'])
    verify_preserved()
    preserved=pd.read_csv(RESUME/'preserved_artifacts_final.csv')
    verification.update(再開時保全ファイル数=len(preserved),再開時旧成果すべて不変=bool(preserved.unchanged.all()),再開計算視野数=len(all_signed)-state['開始実験完了数'],回転修正前後比較視野数=correction['比較視野数'],回転修正前後有意判定変更数=correction['有意判定変更数'])
    c.dump(verification,RESUME/'verification.json')
    prefix=[
        '## 再開時の完了・未完了確認と今回の完了範囲（事実、確度：高）',
        f'途中版 `261004_周8_報告.md` は77視野を記載するが、保存済み実験配列・評価記録を直接確認すると、再開時に170視野の全7生成条件と8評価条件が完了していた。回転対照も170視野すべて修正済みで、修正前配列と変換記録も170視野分保存されていた。未計算は9視野、未評価だけの視野と未修正の回転対照は0視野だった。旧報告とその集計表の視野数だけから未完了範囲を判断していない。',
        '再開時の未計算9視野は、日程260927の基板03視野7・8、基板6視野2・7、基板8視野7、基板9視野2・4・5・7。今回この9視野だけを、保全した予測と修正済み実装で追加計算した。完了済み170視野の像、場、位置合わせを再計算していない。途中の9視野には完成した視野記録がないため、その7条件を最後まで実行した。',
        f'最終的に適格179／179視野、必須47／47視野の全条件を完了。科学的な未検証対象は260830、指定5視野のうち未適格4視野、固定29視野のうち未適格25視野であり、適格179の計算未完了とは区別する。',
        f'旧成果{len(preserved)}ファイルの開始・終了内容指紋がすべて一致した。予測の指紋 `{PRED_HASH}` と回転実装修正記録の指紋も不変。旧コード、旧途中報告、旧予測・規定、旧配列・評価、周1から周7、読み取り専用入力を変更していない。再開用出力は `resume_20261004_final/`、元の途中版は直下のまま保全した。保全一覧は [preserved_artifacts_final.csv](resume_20261004_final/preserved_artifacts_final.csv)。',
        '再計算したのは、追加9視野を含む集計と検定、および保存済み修正前対照の評価・関連5検定の再構成。旧報告の77視野集計と実在する170視野配列の不一致を解消し、対照修正前後を比較するために必要だった。確認的対象47視野と50枠は変更していない。',
        '本文中の表・配列・検証記録のファイル名は、別記した元入力を除き `data/results/v35_band_origin_round8/resume_20261004_final/` 配下を指す。再開開始状態は `resume_initial_state.json`、再開中の進捗は `261004_周8_再開進捗.md`。識別列 `key` は「日程_基板_視野番号」。写像式の A は全体の回転・倍率・せん断を表す二行二列の行列、t は全体並進、x は前像座標、u(x) はその位置の保存済み局所変位ベクトル。',
        '## 回転対照の修正前後で判定が変わったか（事実、確度：高）',
        '空間配置とベクトルの回転方向を一致させる理由と修正前保全は、元の `261004_周8_回転対照の実装修正.md` のまま保持した。修正前の保存配列から評価のみを再構成し、同じ47視野と50倍補正で比較した。再開した9視野は修正済み実装だけを実行し、存在しない修正前結果を作っていない。170視野に必須47視野はすべて含まれるため、確認的比較に対象欠落はない。単位横変位と一点配置の既知正解も再確認した。',
        md(rotationtests),
        f'関連5検定の補正後有意・非有意の判定変更は{correction["有意判定変更数"]}件。修正前・修正後とも相関四比較の支持条件は未達で、固定4視野の主因再現条件は0／4、帯全体の主因としての判定は変わらない。補正前の有意確率と対照の相関・位相は変化するため、最終判定には修正後を使う。全50枠の修正前再構成は `confirmatory_tests_rotation_before.csv`、視野別前後は `rotation_before_after_field_metrics.csv`。旧途中の保存評価と、回転対照以外の{correction["旧途中評価との他条件数値照合行数"]}行の数値は一致した。修正前保存には他条件全配列の内容指紋がないため、前回修正時の他条件全配列不変を独立に照合することはできない。今回の再開以降については旧成果全ファイル不変を内容指紋で確認した。',
        '指定該当視野（日程260926、基板7、視野5）の回転対照だけの修正前後：',
        md(rotationfields[rotationfields.key.eq(CORE[0])][['key','対照版','相関','分散比','無調整分散再現','模擬周期','方向差','位相差','三成分一致']]),
        '## 有意な支持が得られなかった結果と振幅感度',
        f'確認的50枠すべてで補正後0.05未満は{int(tests["補正後有意確率"].lt(.05).sum())}件。必須47視野の主帰無と全体変換のみの共通領域相関差の平均は{tests.iloc[0]["平均差"]:.6g}、補正前{tests.iloc[0]["補正前有意確率"]:.6g}、補正後{tests.iloc[0]["補正後有意確率"]:.6g}。この集合では全域相関中央値も局所場追加0.287194、全体変換のみ0.305504で改善していない。固定4視野や指定1視野の部分改善を、全対象での改善に読み替えない。この不都合な結果も保持した。',
        md(condition_summary),
        '1倍が相関最大に近いか、および分散比が最も1に近いかの全視野集計。数値は事前指定の半分・1倍・2倍だけの比較で、最適振幅の探索ではない。',
        md(pd.DataFrame(amplitude_summary)),
        '分散を合わせるための参考必要振幅は線形近似の粗い目安で、真の変位の推定値ではない。1倍分散比に全体変換・丸め・局所ずれが混ざるため、局所場だけの必要振幅を厳密に分離できない。半分・二倍の分散比が0.25・4から外れることもその限界。位相と相関が合わない視野は、振幅を増やすだけでは再現できない。',
        '視野ごとの全体変換のみとの差は [global_vs_local_signed_by_field.csv](resume_20261004_final/global_vs_local_signed_by_field.csv) と [global_vs_local_outliers_by_field.csv](resume_20261004_final/global_vs_local_outliers_by_field.csv) に、両条件の数値と改善差を並べた。説明率・計数の対照値・負値も元評価表に保持する。',
    ]
    reportpath=RESUME/'261004_周8_最終集計原稿.md'
    report=reportpath.read_text(encoding='utf-8')
    report=report.replace('# 2026年10月4日・周8報告', '# 2026年10月4日・周8報告・最終版',1)
    report=report.replace('## 結論（事実と解釈を区別）','\n\n'.join(prefix)+'\n\n## 結論（事実と解釈を区別）',1)
    report=report.replace('開始時は変更・未追跡なし。版35は作業・結果・入力解析直下と全ブランチ・タグで未使用。','今回の再開開始時には `field_level/v35_band_origin_round8/` が未追跡で存在した。版35は前回の周8で使用中であり、今回は指定どおり同版内へ再開用ファイルを追加した。取得済みブランチ・タグには版35の別用途なし。')
    report=report.replace('検証は rotation_correction_verification.json。','再開時に修正前後を照合した検証は rotation_correction_verification.json。')
    report=report.replace('主模擬・他条件の差配列は保持した。','前回の修正手続きでは主模擬・他条件の差配列を保持する規定であり、今回の再開では旧成果を変更していない。前回修正時の全配列不変の独立照合範囲は冒頭に記した。')
    issues=json.loads((RESUME/'operational_issues.json').read_text(encoding='utf-8'))
    issues.extend([
        dict(stage='resume_environment',issue='標準のpythonコマンドは登録なし。実在する.venv/Scripts/python.exeで実行。管理用の実行中プロセス照会はアクセス拒否。コンピュータ名THINKING・ユーザー名chuya・認識ドライブC/G/Hは環境変数とドライブ一覧で確認。入力コピーは読み取り可能。'),
        dict(stage='resume_rotation_audit_print',issue='回転前後の全表と検証記録を保存後、数値型の画面表示で直列化エラー。整数型への変換を修正して監査を再実行し成功。模擬の再生成なし。',log='resume_rotation_audit.log'),
    ])
    dump(issues,RESUME/'operational_issues.json')
    report=report.replace('運用上の処理エラーは operational_issues.json に記録。','運用上の処理エラーは operational_issues.json に記録。再開中は管理用のプロセス照会がアクセス拒否となり、回転比較結果を保存後の画面表示に数値型の変換エラーが出た。前者は環境変数とドライブ一覧で必要情報を確認し、後者は変換を修正して監査を再実行して成功した。生画像入力へのアクセスと模擬計算に失敗はなかった。')
    report=report.replace('![指定該当視野の差地図](figures/', '![指定該当視野の差地図](resume_20261004_final/figures/')
    report=report.replace('![二定義の外れ値密度](figures/', '![二定義の外れ値密度](resume_20261004_final/figures/')
    report=report.replace('[261004_周8_予測と検定規定.md](261004_周8_予測と検定規定.md)', '[261004_周8_予測と検定規定.md](261004_周8_予測と検定規定.md)')
    # Keep the report's embedded verification synchronized with the extra resume audit.
    start=report.index('```json\n',report.index('## 実装確認と模擬の限界'))
    end=report.index('\n```',start)+4
    report=report[:start]+'```json\n'+json.dumps(verification,ensure_ascii=False,indent=2)+'\n```'+report[end:]
    final=PRIOR/'261004_周8_報告_最終版.md'
    assert not final.exists(), 'Do not overwrite an already published final report'
    final.write_text(report,encoding='utf-8')
    notes=(CODE/'NOTES_resume.md').read_text(encoding='utf-8')
    notes+='\n再開記録：旧NOTES.mdを含む前回成果をすべて保全。実在保存記録170視野を再利用し、残り9視野だけを追加、適格179視野と必須47視野を完了。修正前後170視野の比較は主因判定を変えなかった。再開出力は data/results/v35_band_origin_round8/resume_20261004_final/。旧途中報告は上書きしない。\n'
    (CODE/'NOTES_resume.md').write_text(notes,encoding='utf-8')
    progress=RESUME/'261004_周8_再開進捗.md'
    progress.write_text(progress.read_text(encoding='utf-8')+'\n最終状態：179／179視野・必須47／47視野完了。旧成果保全照合・回転前後比較・二定義・図確認・最終報告完了。未検証科学対象は最終報告に記載。\n',encoding='utf-8')
    print('final report',final,flush=True)

if __name__=='__main__':finish()
