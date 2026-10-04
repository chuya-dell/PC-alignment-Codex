"""Review figures and self-contained Japanese round report."""
from field_round7_common import *
import argparse

def markdown(df):
    def cell(v):
        if isinstance(v,(float,np.floating)):return '未定義' if not np.isfinite(v) else f'{v:.6g}'
        return str(v).replace('|','／').replace('\n',' ')
    cols=list(df.columns)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available={f.name for f in font_manager.fontManager.ttflist}
    for name in ['Yu Gothic','Meiryo','MS Gothic']:
        if name in available:plt.rcParams['font.family']=name;break
    plt.rcParams['axes.unicode_minus']=False
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    evals={p.stem:p for p in (OUT/'evaluation_checkpoints').iterdir() if p.suffix=='.npz'}
    manifest=[]
    for key in CORE:
        if key not in evals:continue
        with np.load(evals[key]) as z:a={k:z[k] for k in z.files}
        variants=['real_signed','主帰無_signed','N0_signed','N1_signed']
        titles=['実際の差','模擬の差（再推定）','動かさない対照','別視野の位相運動']
        field=key.split('_');title=f'日程{field[0]}・基板{field[1]}・視野{field[2]}（32画素四方の平均）'
        scale=max(float(np.nanmax(abs(a[n]))) for n in variants)
        fig,axes=plt.subplots(1,4,figsize=(15.5,4.6),layout='constrained')
        for ax,n,t in zip(axes,variants,titles):
            im=ax.imshow(a[n],cmap='RdBu_r',vmin=-scale,vmax=scale,extent=[0,2048,2048,0])
            ax.set_title(t);ax.set_xlabel('横座標（画素）');ax.set_ylabel('縦座標（画素）')
        fig.colorbar(im,ax=axes,shrink=.75,label='背景除去後9画素和の差（前−後）')
        fig.suptitle(title);path=dest/(key+'_null_comparison.png');fig.savefig(path,dpi=150);plt.close(fig)
        manifest.append(dict(key=key,実在パス=str(path),色範囲=[-scale,scale]))
        # Secondary plots reveal small synthetic structures without substituting
        # their different colour range for the required common-scale comparison.
        variants=['real_signed','主帰無_signed','既知変換_signed','N2_signed','N3_signed']
        titles=['実際の差','模擬・再推定','模擬・既知変換','少量ノイズ','双一次補間']
        fig,axes=plt.subplots(1,5,figsize=(18,4.5),layout='constrained')
        for ax,n,t in zip(axes,variants,titles):
            lim=max(float(np.nanquantile(abs(a[n]),.995)),1e-6)
            im=ax.imshow(a[n],cmap='RdBu_r',vmin=-lim,vmax=lim,extent=[0,2048,2048,0]);ax.set_title(t)
            ax.set_xlabel('横座標（画素）');fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(title+'・各面の色範囲は異なる（形の確認用）')
        path=dest/(key+'_shape_and_controls.png');fig.savefig(path,dpi=130);plt.close(fig)
        manifest.append(dict(key=key,実在パス=str(path),色範囲='各面別。振幅比較には使わない'))
    dump(manifest,OUT/'figure_manifest.json')
    print('figures',len(manifest),flush=True)

def compact_summary(summary,group,methods,cols):
    rows=[]
    for method in methods:
        rec=dict(定義=method)
        for col in cols:
            v=summary[(summary['集合']==group)&(summary['定義']==method)&(summary['指標']==col)]
            if len(v):
                a=v.iloc[0];rec[col]=f"{a['中央値']:.6g} [{a['第1四分位']:.6g}, {a['第3四分位']:.6g}]（{int(a['視野数'])}視野）"
        rows.append(rec)
    return markdown(pd.DataFrame(rows))

def report():
    verify();s=setup();pre=json.loads((OUT/'preflight.json').read_text(encoding='utf-8'))
    signed=pd.read_csv(OUT/'signed_field_metrics.csv',dtype={'日程':str,'基板':str})
    density=pd.read_csv(OUT/'outlier_field_metrics.csv',dtype={'日程':str,'基板':str})
    summary=pd.read_csv(OUT/'metric_summary.csv');tests=pd.read_csv(OUT/'confirmatory_tests.csv')
    decision=json.loads((OUT/'decision.json').read_text(encoding='utf-8'))
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'experiment_checkpoints').glob('*.json'))]
    complete={r['key'] for r in records};f=s['f'];missing=f[f['必須対象']&~f.key.isin(complete)]
    status=subprocess.check_output(['git','status','--short','--untracked-files=all'],cwd=ROOT,text=True,encoding='utf-8')
    dump(dict(ブランチ=pre['ブランチ'],未追跡変更状態=status,コミットタグプッシュ=False),OUT/'git_state.json')
    main=signed[signed['条件']=='主帰無'];fixed=main[main['固定29視野']];core=main[main['指定3視野']]
    text=['# 2026年10月4日・周7報告：物質変化なしの帰無実験','']
    def add(t):text.extend([t,''])
    stage='周7の報告を保存し、ここで停止する。' if len(complete)==266 else '全回復視野への拡張は進行中で、本報告は途中保全記録。必須集合の検定枠は変更しない。'
    add(f"必須{pre['必須視野数']}視野のうち{len(complete & set(f.loc[f['必須対象'],'key']))}視野を完了。全回復266視野のうち{len(complete)}視野を実施した。未完了の必須対象は{len(missing)}視野。"+stage)
    support=decision['主機構支持条件']
    add('## 主な事実と判定')
    add(('事前に定めた主機構の支持条件を満たした。' if support else '事前に定めた主機構の支持条件を満たさなかった。')+f"固定29視野の回復成功部分{len(fixed)}視野のうち、周期・帯方向・位相の三成分一致は{decision['三成分一致数']}視野。下記の相関、振幅、二定義の外れ値率、空間対照、説明率を併せて判断する。数値の確度：高（今回の入力・規定・実装に限定）。")
    add('選択ラベルの訂正を含む：当初の通常27視野には参考ブランク1視野が混入していた。元の65視野・40検定を保全し、参考除外64・通常分子付き26視野の感度集計を追加した。固定29回復11視野と指定3視野の結果には影響しない。詳細は対象選択とラベル監査の節。')
    headline=[]
    for method in METHODS:
        g=density[(density['条件']=='主帰無')&density['固定29視野']&(density['定義']==method)]
        headline.append(dict(定義=method,視野数=len(g),実測外れ値割合中央値百分率=g['実測外れ値割合'].median()*100,模擬外れ値割合中央値百分率=g['模擬外れ値割合'].median()*100,旧尺度説明率中央値百分率=g['全面決定係数'].median()*100,共通基準計数改善中央値百分率=g['全面共通除去逸脱度改善'].median()*100))
    add(markdown(pd.DataFrame(headline)))
    p=tests[(tests['集合']=='必須全体')&tests['指標'].isin(['相関対照差','主対N1共通相関差'])]
    add('必須全65視野では相関の空間対照差と別位相対照差がともに有意である。一方、固定29回復部分の別位相対照差は補正後有意でなく、三成分一致も半数に達しない。この部分整合を無視せず、全体の小さい解析由来成分への整合と大きな外れ値帯の再現不足を分ける。')
    add(markdown(p[['集合','指標','視野数','平均対照差','補正前有意確率','補正後有意確率']]))
    add('今回の模擬は、撮像済み前像の帯域制限近似と回復済みの全体変換で、物質変化なしの標本化・位置合わせ処理を評価する。実際の後像や保存差へ合わせた変換探索、振幅・位相の調整を行っていない。物質変化なしに帯が生じる可能性の実証は、生成能力・十分性を示し得るが、解析機構が必要だという証明や実際の全原因の証明ではない。')
    add('## 入力、実在名、版、保全')
    add('規約、開発環境記録、関連ラボノート八本、周1から周6の報告と予測を確認した。元ノートの10_引き継ぎ・05_解析の階層はコピーでは平坦化されている。開発環境記録の旧利用者・旧ドライブ配置より最新の入力限定指示を優先し、実験入力を data/inputs_local のみに限定した。入力・旧版・旧予測へ書き込まず、凍結リポジトリを参照・実行していない。標準経路、既定値、シミ・ゴミのマスクは変更していない。新しい成果の取り込みや新旧関係の変更は行っていない。')
    add('実在する入力起点：`'+str(LOCAL)+'`。直下：`'+'`、`'.join(pre['入力直下'])+'`。解析フォルダ：`'+str(s['src'])+'`。直下：`'+'`、`'.join(pre['解析直下'])+'`。')
    add('生画像直下の実在名：`'+'`、`'.join(pre['生画像直下'])+'`。コピー完了印：`'+'`、`'.join(pre['コピー完了印'])+'`。260830のフォルダ名 `260830_p50＿同一視野にてsam` の下線は全角。バックスラッシュが下線の前に入る架空名を使っていない。')
    add('開始時、field_level、data/results、入力解析直下、全ブランチ・タグでv34未使用を確認した。v25からv27は入力解析直下、v28からv33は既存フォルダ・タグで使用済み。作業ブランチ：`'+pre['ブランチ']+'`。開始コミット：`'+pre['開始コミット']+'`。書き込み範囲の限定により取得による作業ツリー更新は行っていない。')
    add('主な実在パス（個別の全入力と開始・終了指紋は input_integrity_final.csv）：')
    paths=[s['tabs']['table_registration_field_qc.csv'],s['r1files']['round1_fields_both_definitions.csv'],s['r1files']['round1_thresholds.csv'],s['r1files']['density_maps.npz'],s['r1files']['common_profile_residuals.npz'],s['resume']['round5_recovery_combined.csv'],s['results']['v33_band_origin_round6']/'all_hypotheses_comparison.csv',s['children']['code_v24']]
    for p in paths:add('- `'+str(p)+'`')
    for r in records:
        if r['key'] in CORE:
            add('指定視野 '+r['key']+'：\n\n'+'\n'.join('- `'+r[k]+'`' for k in ('洗浄前パス','洗浄後パス','保存差パス','回復記録パス')))
    add('## 事前予測と実験方法')
    add('予測ファイル：`'+str(OUT/'261004_周7_予測と検定規定.md')+'`。256ビット暗号学的内容指紋：`'+PRED_HASH+'`。保存時刻は prediction_sha256.json。実行前と終了時に一致を照合した。')
    add('予測：画素格子と六方格子のうなり（仮説B）・位置合わせ位相（仮説F）が帯を作るなら、物質の変化なしの模擬差と実測差の周期・向き・位相が一致し、同位置相関と外れ値分布の重なりが移動・回転対照より高い。動かさない対照ではゼロ、別視野の位相運動では対応が弱まる。ブランクにも可能。単独の前像・後像に目立つ帯を要求しない。260830の実測前後対応・回復済み変換は未確定で、実測との確認的検定はできない。')
    add('前像を32画素鏡映拡張。単位正方形画素開口をフーリエ領域で外し、逆全体変換に対応する後像画素の開口を掛けた。各軸4倍のフーリエ補間、逆変換した後像画素中心で高解像度格子を双一次補間、16ビットへ整数化。点広がり・回折のうち撮像済み前像に含まれる帯域を保持する。失われた高周波は作らない。指定3視野は8倍でも感度評価した。')
    add('読み取り専用 code_v24 の元処理で格子生成、シミマスク無効の全体位置合わせ、subpixel段階の調整、最近傍整数丸め、51画素背景除去後9画素和、前から後を引く差を実施。模擬像で位置合わせを一度ずつ再推定し、既知変換で読む補助も保存。組込み明部の位置合わせ除外は元処理のまま。模擬生成・再推定後に初めて実際の後像を読み、生成に流用していない。')
    add('内部呼称：N0は恒等変換、N1は識別名順の回復266視野を133視野巡回移動した供給元の回転・倍率・せん断と中心変位の端数を使う別位相対照、N2は全幅0.1%の独立ガウスノイズ、N3はピラーの9画素を前後とも双一次補間へ替える標本化対照。N1では対象の中心変位の整数部を保持する。N3での実測双一次補間値も別列に保存。N3の主説明率は同一の旧実測密度を予測する感度比較である。')
    add('対象選択は旧割合だけ。固定29の回復部分、回復済みブランク全部、各日程で通常分子付き視野の中央値に近い3視野、指定3視野の和集合。選択表は field_selection.csv。')
    audit=json.loads((OUT/'label_audit.json').read_text(encoding='utf-8'))
    add('**選択実装の訂正：主ブランクの真偽だけで分子付き通常視野を選んだため、参考ブランク260827の基板8視野6が通常選択へ1視野混入した。分子付き通常27視野という説明は誤りで、分子付き26視野と参考ブランク1視野だった。** 当初65視野・40枠を保全し、参考ブランクを除く64視野と通常26視野の感度記述を追加した。固定29回復11視野と指定3視野に混入はない。260827でも通常分子付き2視野が残る。対象の実測適合度を使った除外ではなく、周5の独立したラベル台帳との照合による訂正である。')
    add('補助規定：`261004_周7_ラベル監査と補助集計規定.md`。内容指紋：`'+audit['補助予測指紋']+'`。今回の結果を見た後のラベル監査・感度集計であるため新しい確認的検定は行っていない。元の40検定は当初選択のラベル不整合を含む集合の結果として解釈する。主閾値用ブランクと参考ブランクは混ぜない。元台帳の blank_reference は参考ブランクを表す内部ラベル。修正済みラベル一覧は field_selection_label_audited.csv。')
    add('表の key は「日程_基板_視野番号」の識別名。基板01と基板1を区別し、日程を省略しない。「subpixel」は元コードの微小位置合わせ調整段階名。識別に必要なファイル名・関数名を原表記で記載している。')
    add(markdown(pd.read_csv(OUT/'selection_by_date.csv',dtype={'日程':str})))
    add('日程別の「固定29視野」列は元632視野中の全固定29件数であり、必須対象中の回復件数ではない。今回の回復利用数は以下と field_selection.csv で示す。')
    coverage=[]
    for date,g in f.groupby('日程'):
        coverage.append(dict(日程=date,元固定29数=int(g['固定29視野'].sum()),回復固定29数=int((g['固定29視野']&g['回復成功']).sum()),回復ブランク数=int((g['ブランク']&g['回復成功']).sum()),実施数=int(g.key.isin(complete).sum())))
    add(markdown(pd.DataFrame(coverage)))
    add('## 符号付き差、周期・方向・位相')
    add('32画素四方の平均地図。同位置相関の対照差は横・縦±256、±512画素と90度回転の9対照と共通有効領域で比較。周期は平面勾配除去・ハニング窓・8倍余白付加、64から1024画素の最強ピーク。方向差15度以内、周期比0.8から1.25、位相差45度以内を三成分一致とした。余白付加は周波数の補間であり実質的分解能を増やさない。整数調波を細い帯の証拠にしていない。')
    add(markdown(core[['key','相関','相関対照差','主対N1共通相関差','実測周期','模擬周期','方向差','周期比','位相差','三成分一致','分散比','無調整分散再現']]))
    add('指定3視野の各対照と既知変換での成分別比較：')
    add(markdown(signed[signed['指定3視野']][['key','条件','相関','方向差','周期比','位相差','三成分一致','分散比']]))
    add('指定3視野の主帰無では、基板7視野5は位相差だけが規定内で周期・向きが合わず、基板7視野8と基板01視野8は三成分とも規定外。既知変換で読む補助では弱い実測ピーク位置での位相差が規定内に入るが、周期と向きは合わない。双一次補間では基板7視野5の周期・向き・位相が規定内、基板7視野8は周期・向きだけが規定内になる一方、模擬の分散比は小さい。これを主帰無の大きな帯の再現や原因確定に読み替えない。')
    axes=pd.read_csv(OUT/'spectral_first_axes.csv')
    add('スペクトル全パワーの二次モーメントによる第一軸の絶対方向と、実測の最大ピーク位置における模擬パワーの強さも保存した（追加検定なし）。ピーク対最大パワー比は、実測ピーク位置の模擬パワーを模擬自身の最大ピークパワーで割った値。小さい場合、位相差だけの一致を強い証拠としない。全視野は spectral_first_axes.csv。')
    add(markdown(axes[axes.key.isin(CORE)]))
    for group in ['必須全体','固定29回復','ブランク','通常選択','回復全体（補助）','追加視野（補助）']:
        add('### '+group+'：中央値［第1四分位、第3四分位］')
        add(compact_summary(summary,group,['符号付き差'],['相関','相関対照差','主対N1共通相関差','分散比','無調整分散再現','相関二乗']))
    add('全視野別の周期・帯方向・第一軸・位相・縁300画素除外・各空間対照は signed_field_metrics.csv。以下は主帰無の視野別値。')
    add(markdown(main[['key','相関','相関対照差','主対N1共通相関差','方向差','周期比','位相差','三成分一致']]))
    add('## 外れ値率、重なり、説明できた割合')
    add('二定義はブランクの平均＋標準偏差の3倍、中央値＋絶対偏差中央値の1.4826倍の3倍。周1の日程別閾値を保持し、模擬閾値を再推定しない。表の平均標準偏差・中央値絶対偏差はこの二定義の内部列名。ジャカード係数は同一ピラー外れ値集合の交わりを和集合で割った値、密度重なりは升目ごとの小さい率の和の2倍を両率の総和で割った値。両集合が空なら未定義。')
    for group in ['必須全体','固定29回復','ブランク','通常選択','指定3視野','回復全体（補助）','追加視野（補助）']:
        add('### '+group+'：中央値［第1四分位、第3四分位］')
        add(compact_summary(summary,group,METHODS,['実測外れ値割合','模擬外れ値割合','ジャカード係数','密度重なり対照差']))
        add(compact_summary(summary,group,METHODS,['全面決定係数','共通領域決定係数','対照中央値決定係数','対照差決定係数']))
        add(compact_summary(summary,group,METHODS,['全面逸脱度改善','全面共通除去逸脱度改善','対照中央値共通除去逸脱度改善','対照差共通除去逸脱度改善','全面共通除去順位相関']))
    add('説明変数は模擬の符号付き差平均と模擬外れ値密度の2つ。非線形な標本化と閾値応答を模擬内部で計算し、実測との対応に合わせて周期・向き・位相を調整していない。当該視野を除く全631視野の共通平均を使用。周1・周5と同じ空間8分割・周囲2升目除外・係数二乗罰則強さ1。旧尺度は共通平均除去後の密度残差に対する保留升目の決定係数。周6の二項計数指標は平坦基準と共通構造基準への逸脱度改善、残差順位相関。全面と9対照共通領域を分ける。負値はそのまま保存。')
    add('これらは分散・計数の予測改善割合であり、分子由来割合・除外できるピラーの割合・因果寄与割合ではない。単純な模擬／実測分散比は振幅の目安であり、真の説明率や厳密な上限ではない。未調整分散再現量は1−分散(実測−模擬)/分散(実測)、負値は悪化を表す。相関二乗は実測で線形調整した記述上の整合の目安であり、交差検証説明率と区別する。')
    add('回復全体・追加視野は事前規定どおり補助記述であり、確認的検定に追加していない。仮説Bと仮説Fの標本化・全体位置合わせを組み合わせた経路の評価であり、二つの機構の因果寄与を単独に分離する実験ではない。')
    add('主帰無の視野別二定義値（率と説明率は0から1の比、負値も保持）：')
    corrected=pd.read_csv(OUT/'reference_excluded_metric_summary.csv')
    add('### ラベル監査後の参考ブランク除外感度（記述のみ）')
    for group in ['参考除外必須（補助）','参考除外通常（補助）','参考除外回復全体（補助）']:
        add(group)
        add(compact_summary(corrected,group,['符号付き差'],['相関','相関対照差','主対N1共通相関差']))
        add(compact_summary(corrected,group,METHODS,['実測外れ値割合','模擬外れ値割合','全面決定係数','対照差決定係数','全面共通除去逸脱度改善','対照差共通除去逸脱度改善']))
    add(markdown(density[density['条件']=='主帰無'][['key','定義','実測外れ値割合','模擬外れ値割合','ジャカード係数','全面決定係数','対照中央値決定係数','全面共通除去逸脱度改善','対照差共通除去逸脱度改善']]))
    add('## 全確認的検定：補正前後')
    add('8指標×5集合の40枠。視野ごとの対照差の片側符号反転、18視野以下は全符号列挙、他は10000回＋足し1。ボンフェローニ倍率40。ピラー・升目・同じ地図が何度も入る組を検定単位にしていない。同日程同基板の一括符号反転は感度として併記。指定3視野の同一基板2視野は完全独立ではなく、小標本で有意性は出にくい。')
    add('符号反転は帰無で対照差の符号が交換可能であることを仮定する。視野単位でも同基板・同日程の依存や分布の非対称性は残り、一括符号反転との違いを保持する。空間8分割の周囲2升目除外より広い空間相関が残る可能性もあり、交差検証は独立した撮像反復による検証とは異なる。')
    add(markdown(tests))
    add('## 動かさない、別位相、ノイズ、双一次補間の対照')
    variants=signed.groupby('条件').agg(視野数=('key','count'),相関中央値=('相関','median'),分散比中央値=('分散比','median'),未調整分散再現中央値=('無調整分散再現','median')).reset_index()
    add(markdown(variants))
    controls_summary=density.groupby(['条件','定義']).agg(視野数=('key','count'),模擬外れ値割合中央値=('模擬外れ値割合','median'),全面決定係数中央値=('全面決定係数','median')).reset_index()
    add(markdown(controls_summary))
    add('N0の既知恒等変換は厳密ゼロ。模擬像組で再推定したN0の微小差は位置合わせ経路自身の誤差として保存する。別位相で相関が落ちない場合は不都合な結果として保持する。ノイズ・双一次補間の結果は事前規定どおり記述で、追加の確認的検定を行わない。各変換と既知変換誤差、失敗、各分散評価は experiment_checkpoints と experiment_fields.csv。')
    registration=[]
    for r in records:
        if r['key'] in CORE:
            a=r['条件']['主帰無'].get('位置合わせ',{})
            registration.append(dict(key=r['key'],既知変換誤差中央値画素=a.get('既知変換誤差中央値',np.nan),既知変換誤差最大画素=a.get('既知変換誤差最大',np.nan)))
    add('指定3視野の模擬像再位置合わせは注入した変換を小さい誤差で回収した。既知変換で読む場合も大きな実測帯は再現されず、模擬像の再位置合わせ失敗だけでは再現不足を説明できない。これは実像の変換が真の動きと一致したという意味ではない。')
    add(markdown(pd.DataFrame(registration)))
    add('## 模擬撮像の限界と像の鋭さ')
    fidelity=[]
    for r in records:
        a=r['限界'];row=dict(key=r['key'],後像ピラー相関=a['後像ピラー相関'],後像差平均=a['後像差平均'],後像差標準偏差=a['後像差標準偏差'])
        for quantile,value in zip(['第2点5百分位','第25百分位','第50百分位','第75百分位','第97点5百分位'],a['後像差分位点']):row['後像差'+quantile]=value
        for v,c in [('前像','前像鋭さ'),('実測後像','実測後像鋭さ'),('模擬後像','模擬後像鋭さ')]:
            row[v+'1画素低下率']=a[c]['低下率中央値'];row[v+'低下率第1四分位']=a[c]['低下率第1四分位'];row[v+'低下率第3四分位']=a[c]['低下率第3四分位']
        fidelity.append(row)
    fidelity=pd.DataFrame(fidelity);fidelity.to_csv(OUT/'imaging_fidelity.csv',index=False,encoding='utf-8-sig')
    add('1画素低下率はピラー付近3×3の局所最大の背景除去値から、横縦4方向に1画素ずらした平均値へ下がる割合。各像5000本以内の固定等間隔、正の中心値だけ。前像・実測後像・模擬後像でそれぞれ局所最大を取り直す。9画素和の低下率ではなく像の鋭さの補助量である。背景値のため1を超える低下率もあり得る。')
    add(markdown(fidelity[fidelity.key.isin(CORE)]))
    cols=[c for c in fidelity.columns if c!='key'];add(markdown(pd.DataFrame([dict(指標=c,視野数=int(fidelity[c].notna().sum()),中央値=fidelity[c].median(),第1四分位=fidelity[c].quantile(.25),第3四分位=fidelity[c].quantile(.75)) for c in cols])))
    sensitivity=[]
    for r in records:
        if '8倍補間' in r['限界']:sensitivity.append(dict(key=r['key'],**{k:v for k,v in r['限界']['8倍補間'].items() if not isinstance(v,dict)}))
    add('4倍から8倍の補間感度：');add(markdown(pd.DataFrame(sensitivity)))
    camera=json.loads((OUT/'analytic_camera_verification.json').read_text(encoding='utf-8'))
    add(f"変換と画素開口の実装を、答えのある人工余弦波でも検証。回転・倍率・せん断・端数並進を含め、振幅に対する二乗平均平方根誤差は{camera['振幅相対二乗平均平方根誤差']*100:.4g}%。恒等変換の数値再構成は全画素一致した。これは実装の照合であり実際の連続像の再現性を保証しない。")
    add('指定3視野の同色範囲の図を全て目視した。実測に大きな斜めの帯がある一方、模擬の振幅は小さく、動かさない対照は白いゼロ地図。別位相でも小さな構造が残る。この目視は数値検定の代わりに使わない。最強周期が探索下限64画素付近へ集まる模擬では周期推定の境界効果が残り、弱い実測ピーク位置での位相の数値だけを位相一致の証拠としない。')
    add('鋭さが近くても同じ連続像を復元できた証明ではない。単一の低下率やピラーコントラストは点広がり全体、回折、超ナイキスト周波数の折り返しを同定しない。前像ノイズが後像へ変形され共有され、実際の独立した撮像ノイズを再現しない。模擬／実測後像の差には光学変化、物質変化、局所変形、位置合わせ誤差が混ざる。')
    add('回復した全体変換をそのまま真の動きと仮定すると、同じ全体変換を使った像組は自己整合的になる。実際の前後像に残った局所非アフィンずれや、最終変換の真の誤差を模擬へ入れた実験ではない。したがって今回の帰無モデルが再現できない部分を物質変化とみなすことはできない。模擬の不完全さ、局所ずれ、未知の光学変化、真の物質変化は区別できない。残差分散のうち未再現分はこの限界を含む。')
    add('## 解釈、残る説明、次の周への提案')
    add(('今回の帯域制限モデルでも物質変化なしの対応が支持されるため、解析機構が少なくとも一部の帯を生成し得る。確度：中。実際の全帯の原因は確定しない。' if support else '今回の帯域制限・全体変換の帰無モデルを、実際の帯の主因として採用する根拠は得られない。確度：中（このモデルに限定）。画素格子のうなりや局所位相ずれの機構全体は判定不能。確度：低。')+'結果の一部だけが対応した場合は、周期・向き・位相と振幅の欄を分けて評価する。')
    add('必要なら次の周は、前像だけからピラーの形と位置を推定する画素開口込みの連続像モデル、複数位相の独立撮像による高周波の検証、独立した非周期目印による局所変形を事前固定して注入する実験を優先する。模擬の推定に実測差を使って対応を作らない。物質・洗浄・蒸着・成形原因を確定するには独立した実験条件や形態観察が必要。今回は次の周を実行せず、標準除外・補正・マスクへ導入しない。')
    add('## 完了範囲、弱点、確認不能、検証')
    add(f"必須対象の未完了：{len(missing)}。全266の未実施：{266-len(complete)}。固定29のうち元変換非回復の視野は{int((f['固定29視野']&~f['回復成功']).sum())}視野。回復成功は選択された部分であり632視野全体への無条件の一般化はできない。260830は前後対応と実測変換が未確定で未検定。新しい日程の生画像コピー要求はない。実際の連続像・点広がり・局所真変形・独立ノイズは確認できない。")
    if len(missing):add(markdown(missing[['key','日程','基板','必須対象']]))
    errdir=OUT/'execution_errors';errors=list(errdir.glob('*.json')) if errdir.exists() else []
    add('実行失敗記録数：'+str(len(errors))+'。失敗記録は execution_errors。管理インターフェースでの物理メモリ・論理処理器情報取得はアクセス拒否だったが、Windowsの別の読取方法で確認できた。入力データの読み取りは成功した。描画の初回に利用者共通の字体キャッシュへの保存が拒否され、その後の描画設定・キャッシュは本周の結果配下へ限定した。入力・旧版への書き込みはない。')
    add('開けなかった記録位置：周5再開の bf_v24_recovery 配下に260926基板7視野5の記録は存在しなかった。上位から実在一覧を確認して、周1の bf_v24_recovery にある同視野の回復記録を使った。指定3視野の使用記録は上記の実在パスと内容指紋で固定した。')
    check=OUT/'verification.json'
    if check.exists():add('検証結果：\n\n```json\n'+check.read_text(encoding='utf-8')+'\n```')
    add('開始・終了内容指紋は input_integrity_final.csv。生成画像・模擬配列・キャッシュをGitへ追加していない。図は figures に保存。指定3視野の実際／模擬／N0／N1は全4面同じ色範囲。shape_and_controls は形を見る補助図で色範囲が各面別であり振幅比較には使わない。')
    for key in CORE:add('!['+key+' 帰無実験](figures/'+key+'_null_comparison.png)')
    add('## 作業ブランチと未追跡ファイル')
    add('ブランチ：`'+pre['ブランチ']+'`。コミット・タグ・プッシュなし。開始時は変更・未追跡なし。終了時の状態：\n\n```text\n'+status.strip()+'\n```')
    add('結果配下は既存の無視設定で通常の状態表示には出ない。実在ファイルの一覧は output_inventory.csv。今回の未追跡コードと NOTES.md は field_level/v34_band_origin_round7 のみ。既存コード・周1から周6の結果と予測の指紋を照合した。')
    add('## 周1から周7の全仮説の判定と、採用する原因（または決着しない理由）')
    source=s['results']['v33_band_origin_round6'];files={p.name:p for p in source.iterdir()};previous=table(files['all_hypotheses_comparison.csv'])
    hypotheses=[]
    for h,g in previous.groupby('仮説',sort=False):
        name=h.split('：')[0]
        old='；'.join(dict.fromkeys(g['周6判定と確度']))
        if name in ('B','F'):
            new='今回の帰無モデルは支持／中、実際の全原因は判定不能／低' if support else '今回モデルを大きな帯の主因とする説明は否定／中、機構全体は判定不能／低'
            reason='小さい成分への整合あり。撮像済み像の帯域制限、全体変換の自己整合性、実際の局所ずれ未注入。三成分・振幅・計数を併せて評価。'
        else:new=old;reason={'A':'位置6・7の系統的明部は事実だが外れ値帯の位置固定説明は非支持。','C':'流れ・乾燥の独立記録なし。固定方向モデルのみ非支持。','D':'蒸着の独立形態・厚み測定なし。','E':'鋳型の独立形態測定なし。','G':'滑らかな二次面は記述できるが帯の主因を特定しない。','H':'自動候補は未承認。画像由来の循環を含む記述のみ。','I':'時刻・条件の独立因果証拠不足。画像由来モデルは循環の恐れ。','J':'濃度単調性未確認。ブランクの帯と非分子原因が残る。'}.get(name,'原因分離不能。')
        if name=='A':new='位置固定を帯主因とする説明は否定／中、構造全体の原因分離は判定不能／低'
        if name in ('H','I'):new='原因は判定不能／低（画像由来指標は記述に限定）'
        hypotheses.append(dict(仮説=h,周1から周6判定=old,周7後判定=new,採用または決着しない理由=reason))
    pd.DataFrame(hypotheses).to_csv(OUT/'all_hypotheses_round1_to_round7.csv',index=False,encoding='utf-8-sig');add(markdown(pd.DataFrame(hypotheses)))
    add('採用する原因：'+('解析側の生成能力を部分的に採用するが、実際の全外れ値の原因割合は未確定。' if support else '全外れ値の主因として確定採用する原因はない。今回の帰無モデルの非支持だけで物質変化を採用することもできない。'))
    (OUT/'261004_周7_報告.md').write_text('\n'.join(text),encoding='utf-8')
    notes='''# 周7：物質変化なしの帰無実験

## 実装内容・旧版からの変更点

比較専用の前像変形・画素開口・フーリエ補間、読み取り専用code_v24による模擬像の再位置合わせ、既知変換対照、二定義の外れ値率・重なり、周1・5の決定係数と周6の二項計数評価を追加。標準経路・旧版・入力は変更しない。予測指紋を実行前・終了時に照合する。

## 結果概要と保存先

data/results/v34_band_origin_round7/261004_周7_報告.md。視野ごとの実験・評価チェックポイント、全指標、40枠の補正前後検定、同色範囲の指定3視野の図、全仮説一覧を保存。詳しい数値と完了範囲は報告と verification.json。

## 既知の問題・未解決事項

撮像済み前像の帯域制限近似は失われた高周波・独立撮像ノイズ・実際の局所非アフィンずれを再現しない。全体変換は自己整合的な動きとして注入する。模擬の生成能力と実際の原因寄与を区別し、未再現残差を物質変化と断定しない。260830の前後対応は未確定。今回の解析を標準補正・除外へ導入しない。コミット・タグ・プッシュなし。
'''
    (ROOT/'field_level/v34_band_origin_round7/NOTES.md').write_text(notes,encoding='utf-8')
    with (ROOT/'field_level/v34_band_origin_round7/NOTES.md').open('a',encoding='utf-8') as stream:
        stream.write(f'\n実施{len(complete)}／266視野、必須65完了。主機構支持条件：{support}。固定29回復11視野の三成分一致1視野。旧尺度説明率中央値は平均と標準偏差で−0.0446%、中央値と絶対偏差で0.0843%。計数改善は共通基準で1.29%、1.02%。\n\n選択実装で通常27に参考ブランク260827基板8視野6が1視野混入したため、当初65・40検定を保全し、ラベル監査規定と指紋を別保存して参考除外64・通常26の補助記述を追加。固定29・指定3視野に混入なし。詳細は報告。\n')
        stream.write('''
## 再実行の順序

リポジトリを作業ディレクトリとし、各コマンドに `.venv/Scripts/python.exe -B` を付ける。予測ファイルと指紋の照合を必須とし、実験・評価は既存の視野別チェックポイントを再利用する。

1. `field_round7_experiment.py init`、`pilot`、`run`、必要に応じ `all --workers 2`。
2. `field_round7_verify_model.py`、`field_round7_evaluate.py evaluate`、`tests`。
3. `field_round7_label_audit.py`、`field_round7_spectral_audit.py`。
4. `field_round7_report.py figures`、`verify`、`report`。

上記ファイルは全てこの版のフォルダ内。生成画像と配列は data/results 配下のみ。通常選択の原集合を再利用し、ラベル訂正は別の監査済み選択表と補助集計に保持する。
''')
    inventory()
    print('report saved',flush=True)

def verify_results():
    verify();s=setup();signed=pd.read_csv(OUT/'signed_field_metrics.csv');density=pd.read_csv(OUT/'outlier_field_metrics.csv');tests=pd.read_csv(OUT/'confirmatory_tests.csv')
    records=[json.loads(p.read_text(encoding='utf-8')) for p in (OUT/'experiment_checkpoints').glob('*.json')]
    complete={r['key'] for r in records};required=set(s['f'].loc[s['f']['必須対象'],'key']);gridmax=max(r['格子生成最大差'] for r in records);auditmax=max(r['限界']['元差再現最大差'] for r in records)
    n0max=0.;oracle0max=0.;allvalid=True
    for p in (OUT/'experiment_checkpoints').glob('*.npz'):
        with np.load(p) as z:
            oracle0max=max(oracle0max,float(np.nanmax(abs(z['N0_known_delta']))));n0max=max(n0max,float(np.nanmax(abs(z['N0_delta']))))
        metrics=json.loads((OUT/'evaluation_checkpoints'/(p.stem+'.json')).read_text(encoding='utf-8'))
        for r in metrics['outliers']:
            assert 0<=r['実測外れ値割合']<=1 and 0<=r['模擬外れ値割合']<=1
            assert r['外れ値交わり']<=r['外れ値和']<=r['有効ピラー数']
    assert oracle0max==0 and gridmax<=1e-6 and auditmax<=1e-6
    assert len(tests)==40 and not density.duplicated(['key','条件','定義']).any()
    assert all(k in complete for k in CORE)
    assert ((tests['補正後有意確率']-np.minimum(1,tests['補正前有意確率']*40)).abs()<1e-10).all()
    convcols=[c for c in density if '未収束' in c]
    maxnonconv=int(density[convcols].max().max()) if convcols else 0
    dump(dict(予測指紋一致=True,実施視野数=len(complete),必須視野数=len(required),必須未完了数=len(required-complete),主帰無成功数=int(((signed['条件']=='主帰無')&(signed['状態']=='成功')).sum()),格子最大差=gridmax,実測保存差再現最大差=auditmax,恒等既知変換最大差=oracle0max,恒等再推定最大差=n0max,確認的検定数=len(tests),補正算術照合=True,重複行なし=True,計数モデル最大未収束分割数=maxnonconv),OUT/'verification.json')
    integrity()
    inventory()
    print('verified',len(complete),flush=True)

def inventory():
    paths=[dict(実在パス=str(p.resolve()),バイト数=p.stat().st_size) for p in OUT.rglob('*') if p.is_file() and p.name!='output_inventory.csv']
    pd.DataFrame(paths).to_csv(OUT/'output_inventory.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['figures','report','verify']);a=ap.parse_args()
    {'figures':figures,'report':report,'verify':verify_results}[a.action]()
