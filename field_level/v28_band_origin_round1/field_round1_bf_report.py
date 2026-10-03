"""New round-one review report; previous reports remain untouched."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import json, subprocess, hashlib, platform, getpass
from pathlib import Path
import numpy as np
import pandas as pd
from field_round1_analysis import ROOT,OUT,LOCAL,METHODS,fields,load
from field_round1_bf_recovery import DEST,CODE,collect,verify,digest,csv,pairs_for
from field_round1_bf_models import MD,tests

REPORT=OUT/'261004_周1_報告_B_F検定編.md'

def table(df):
    if len(df)==0:return '該当なし。'
    def fmt(v):
        if isinstance(v,(float,np.floating)):
            return '欠測' if not np.isfinite(v) else f'{v:.7g}'
        return str(v).replace('|','／').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(fmt(x) for x in r)+' |' for r in df.itertuples(index=False,name=None)])

def read(name):
    return pd.read_csv(OUT/name,dtype={'日程':str,'基板':str})

def interval(v):
    v=pd.Series(v).dropna()
    return f'{v.median()*100:.4f}%（第1〜第3四分位：{v.quantile(.25)*100:.4f}〜{v.quantile(.75)*100:.4f}%、{len(v)}視野）' if len(v) else '欠測'

def audit():
    verify();records=collect();rows=[]
    for r in records:
        for col,h in [('洗浄前パス','洗浄前指紋'),('洗浄後パス','洗浄後指紋'),('保存差パス','保存差指紋')]:
            p=Path(r[col]);now=digest(p)
            rows.append(dict(key=r['key'],種類=col,実在パス=str(p),開始指紋=r[h],終了指紋=now,不変=now==r[h]))
    assert all(r['不変'] for r in rows)
    csv(rows,'round1_bf_input_integrity.csv')
    tag='v24_registration_residual_spatial_correlation_20260928'
    sources=[]
    for rel in ['shared/registration.py','shared/v2_registration_precision/refinement.py','shared/lattice_indexing.py','shared/image_qc.py']:
        p=CODE/rel
        raw=p.read_bytes();gitraw=subprocess.check_output(['git','show',tag+':'+rel],cwd=ROOT)
        sources.append(dict(実在パス=str(p),内容指紋=digest(p),タグ=tag,タグ内容指紋=hashlib.sha256(gitraw).hexdigest(),
                            バイト一致=raw==gitraw,改行正規化一致=raw.replace(b'\r\n',b'\n')==gitraw.replace(b'\r\n',b'\n')))
    csv(sources,'round1_bf_code_provenance.csv')
    return rows,sources

def gradient_table():
    spatial=read('round1_spatial_classification.csv');rows=[]
    for r in collect():
        if not r['回復']:continue
        m=np.array(r['最終行列']);g=m[:,:2]-np.eye(2)
        for comp in (0,1):
            v=g[comp];norm=np.linalg.norm(v)
            period=1/norm if norm else np.nan
            axis=(np.degrees(np.arctan2(v[1],v[0]))+90)%180
            s=spatial[(spatial.key==r['key'])&(spatial['定義']=='平均標準偏差')&(spatial['縁除外画素']==0)].iloc[0]
            rows.append(dict(key=r['key'],日程=r['日程'],固定29視野=r['固定29視野'],変位成分='横' if comp==0 else '縦',
                             横勾配=v[0],縦勾配=v[1],一画素変化の予測間隔画素=period,予測帯軸度=axis,
                             観測第一軸度=s['第一帯軸度'],観測自己相関周期画素=s['第一自己相関周期画素'],
                             観測分類=s['分類'],軸差度=abs((axis-s['第一帯軸度']+90)%180-90),
                             注意='変位勾配の一画素等位相間隔。格子標本化による追加のうなりとは区別。'))
    csv(rows,'round1_bf_gradient_predictions.csv')
    return pd.DataFrame(rows)

def output_content(p):
    n=p.name;parent=p.parent.name
    if parent=='bf_v24_recovery':return '視野別：最終変換、格子と保存差の照合、生画像・保存差の指紋' if p.suffix=='.json' else '視野別：再計算差、元座標・洗浄後座標、最終変換、双一次補間差、周期像の基底'
    if parent=='bf_models':return '視野別：局所追跡の支持・独立検証条件と採用候補状態' if p.suffix=='.json' else '視野別：Bの13変数地図、支持点・変位・候補Fの5変数、周期像係数'
    if parent=='bf_model_evaluation':return '視野別：二定義、全面・対照共通領域の決定係数と分解' if p.suffix=='.json' else '視野別：学習外予測、共通有効領域、交差検証分割番号'
    if parent=='bf_marker_review':return '洗浄前・変換した洗浄後・十字交点・重ね合わせの目視確認図'
    if parent=='bf_overlay':return '外れ値密度と丸め・周期像・利用可能な残差予測地図の重ね合わせ'
    named={
      'round1_bf_v24_recovery.csv':'全再試行視野の回復状態、差の大きさ、格子・マスク・診断・濃度の監査',
      'round1_bf_tracking.csv':'全回復視野の局所追跡品質と支持条件',
      'round1_bf_explained_fraction.csv':'各視野・各定義・各モデルの説明率、九対照、共通部分・固有分・相互作用',
      'round1_bf_confirmatory_tests.csv':'視野単位の八検定、補正前後、日程・基板単位の感度比較',
      'round1_bf_explanation_summary.csv':'説明率・対照・分解の中央値と第1・第3四分位',
      'round1_bf_sampling_thresholds.csv':'回復ブランクによる方式別再校正。完全性を明示',
      'round1_bf_sampling_comparison.csv':'整数丸めと双一次補間の外れ値割合・密度分散・周期域パワー',
      'round1_bf_phase_correlations.csv':'回復外れ値視野の丸め端数地図との記述的相関係数',
      'round1_bf_marker_phase_audit.csv':'目視済み十字線の独立相関による一周期違いの監査',
      'round1_bf_numerical_verification.json':'既知の説明変数、無関係な対照、学習余白、非循環移動、ゼロ変換の検証',
      'round1_bf_input_integrity.csv':'使用した生画像・保存差の処理前後指紋と不変判定',
      'round1_bf_code_provenance.csv':'過去コードコピーと指定タグのバイト・改行正規化比較',
      'round1_bf_gradient_predictions.csv':'回復最終変換から予測する等位相の間隔・軸と観測軸',
      'bf_marker_review.json':'実画像を開いて確認した非周期目印の採用状態と限界',
      'bf_execution_sha256.csv':'追加指示の実行整理文書の内容指紋',
      '261004_B_F検定編_実行整理.md':'元規定を変更せず、最新指示の適用と未指定の数値実装を記録',
      REPORT.name:'今回の新しい検収報告。前の報告は保持',
      'round1_bf_output_inventory.csv':'今回の成果物全ファイルの実在パス、内容、容量',
      'round1_bf_environment.json':'実行環境、ブランチ・履歴・未追跡状態、コピー完了状態',
    }
    if n in named:return named[n]
    if n.startswith('field_round1_bf_'):return '今回追加した比較専用実行コード'
    if n=='NOTES.md':return '版の実装内容、保存先、既知の問題。今回の追記を含む'
    return '前回の成果物。詳細は前回完了版報告と round1_output_inventory.csv'

def generate():
    tests();integrity,sources=audit();gradient=gradient_table()
    recovery=read('round1_bf_v24_recovery.csv');ex=read('round1_bf_explained_fraction.csv');test=read('round1_bf_confirmatory_tests.csv')
    tracking=read('round1_bf_tracking.csv');sampling=read('round1_bf_sampling_comparison.csv');thresholds=read('round1_bf_sampling_thresholds.csv')
    phase=read('round1_bf_phase_correlations.csv'); marker=read('round1_bf_marker_phase_audit.csv')
    raw=next(p for p in LOCAL.iterdir() if p.name=='raw_readonly');children=list(raw.iterdir())
    extra=any(p.is_file() and 'copy_done_2' in p.name for p in children)
    status=subprocess.check_output(['git','status','--short','--untracked-files=all'],cwd=ROOT,text=True,encoding='utf-8')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tagcommit=subprocess.check_output(['git','rev-parse','v24_registration_residual_spatial_correlation_20260928'],cwd=ROOT,text=True).strip()
    environment=dict(computer=platform.node(),user=getpass.getuser(),python=sys.version,branch=branch,head=head,code_tag_commit=tagcommit,
                     git_status=status,raw_children=[p.name for p in children],additional_copy_complete=extra,
                     source_paths=[str(p) for p in LOCAL.iterdir()],read_only_inputs=True,no_git_mutation=True)
    (OUT/'round1_bf_environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
    recovered=recovery[recovery['回復']];bad=recovery[~recovery['回復']]
    verified=ex[(ex['非周期目印検証'])&(ex['F支持条件'])]
    joint=verified[verified['モデル']=='BF']
    lines=['# 2026年10月4日・周1報告：仮説B・F検定編','',
      '## 判定と範囲','',
      f'260926の70視野を元の呼び出しで再試行し、保存済み差の全ピラー最大絶対差0.000001以内で34視野を回復した（全て最大差0）。外れ値11視野のうち回復5視野。回復できない36視野は差を合わせる探索をせず不一致として残した。これらは260926のみの数値で、後述する他日程が利用可能な場合は別集計を示す。確度：高。',
      '', '仮説B：今回の事前に決めた予測変数・集計・利用可能視野では支持されない。確度：中。整数丸めという機構が存在しないとの否定ではなく、今回のモデルが帯の主要因を説明する証拠が得られない、という判定。',
      '', '仮説F：判定不能。確度：低。確認条件を満たす260926の2視野（外れ値は1視野）では説明率を計算できたが、補正後の有意な支持を得ていない。局所残差候補だけの他視野を確認的検定に混ぜない。',
      '', '全632視野・固定29視野という元の母集団の検証は完結していない。最新追加指示に基づき、その回復・独立検証可能部分で八検定を実施した。説明率が欠測の視野を0%と数えていない。前回の手順1〜3は採用済みとして保持し、分類・予測を変更していない。',
      '', '## 予測・指紋・今回の追加指示','',
      'Bの事前予測：標本化座標の端数・丸め誤差と周期像の予測差が外れ値地図に対応し、双一次補間でその構造が弱まる。洗浄前だけ・洗浄後だけ、ブランク・260830でも生じ得る。単独画像と差の位置・符号の一致は必須でない。',
      '', 'Fの事前予測：生画像から独立に求めた局所残差と像の斜面が、Bを条件付けた後にも外れ値密度を説明する。補間を変えても局所残差が残れば影響が残る。ブランク・260830でも生じ得る。単独画像と差の同じ帯は必須でない。',
      '', '元予測と追加実装規定を保持した内容指紋：',
      '', table(pd.concat([pd.read_csv(OUT/'prediction_sha256.csv'),pd.read_csv(OUT/'supplement_prediction_sha256.csv')],ignore_index=True).rename(columns={'Algorithm':'指紋算法','Hash':'内容指紋','Path':'実在パス'}).replace({'指紋算法':{'SHA256':'セキュアハッシュアルゴリズム二百五十六ビット'}})),
      '', '追加の実行整理は `261004_B_F検定編_実行整理.md`、指紋は `bf_execution_sha256.csv`。回復の初期試行後、説明率・残差・検定の計算前に保存した。元の二文書を書き換えていない。今回の最新指示による回復条件・利用可能部分検定の変更と、未指定の回帰分割・残差平滑化の実装を明示した。',
      '', '## 回復の方法と事実','',
      f'読取り専用コードコピー：`{CODE}`。指定タグのコミットは `{tagcommit}`。当時の実行コミットは未確定。今回のコピーを標準解析へ組み込んでいない。',
      '', table(pd.DataFrame(sources)),
      '', '元スクリプト `scripts_and_config/field_run_digital_judgment.py` の compute の手順を比較用スクリプトへコピーした。粗いアフィン位置合わせを `mask_stains=False, return_qc=True`、続いて `register_refined(stage="subpixel", initial=coarse, mask_stains=False)` で呼ぶ。洗浄前の高速フーリエ変換からピッチ7.286画素・余白30画素の格子を再計算する。その点を最終行列で洗浄後に写し、`sample_contrast` の整数丸め・三×三画素和・五十一画素ガウス背景差をそのまま使った。洗浄前から洗浄後を引いた。',
      '', '既存の明部マスクは元コードの位置合わせ内部の既定値であり、ゴミ・シミのマスクは無効。今回の密度・外れ値の標本から候補欠陥を除外していない。元の明部マスク割合と今回の割合は全再試行視野で一致する。格子識別は全視野一致、洗浄前座標の最大差は計算機の丸め程度。前回の無マスク粗い推定との違いは、この既存マスク・空間支持に基づく推定候補選択・微小位置合わせを正しく呼んだ点。',
      '', '最新指示では差の一致を回復条件とするため、粗い診断値の差や微小位置合わせの採用有無で回復視野を除外しない。診断値が一致しても丸め境界をまたぐと一部ピラーの差が変わり得る。差の一致も、当時の丸め前変換行列を数学的に一意に証明するものではない。',
      '', table(recovery.groupby(['日程','回復','固定29視野']).size().rename('視野数').reset_index()),
      '', '濃度・ブランクの回復の偏り：',
      '', table(recovery.groupby(['日程','濃度','ブランク','回復']).size().rename('視野数').reset_index()),
      '', '不一致視野と差の大きさ（全例）：',
      '', table(bad[['key','日程','基板','外れ値割合','差最大絶対差','差平均絶対差','差一致ピラー率','診断最大差','洗浄前座標最大差']]),
      '', '不一致の原因は確定していない。明部マスク割合・洗浄前格子の再現だけで説明できない。元の実行コミット、画像処理ライブラリの版・特徴点推定の丸め、乱数状態、微小位置合わせ採否や当時の前処理が候補として残る。保存差を使って変換・マスク・乱数を調整する探索は行っていない。確度：低。',
      '', '## 検定と説明率の方法','',
      '密度地図は32画素四方。対象を除く残り631視野の共通平均地図を引いた残差を目的変数にした。元の二閾値を用いる。八×八升目の空間ブロックを八分割し、評価ブロックの周囲二升目を学習から外す。標準化・回帰係数は学習部分だけで推定し、正則化強度は1。基準は学習残差の平均。決定係数は1−モデル二乗誤差／基準二乗誤差。負値はそのまま残した。',
      '', 'Bは丸め誤差の二成分・二乗・洗浄前後の差と二乗・洗浄前像から作った周期像の予測差、計13変数。Fは独立局所残差の二成分・二乗・洗浄前像の斜面との積の差予測、計5変数。両者のモデルと、両予測差の積を追加した相互作用モデルを同じ有効領域・分割で評価した。',
      '', '対照は横・縦に±256・±512画素の8移動と90度回転の計9通り。移動を循環させず、実地図・全対照が共通して有効な中央部分で決定係数を再計算した。全面の説明率と対照共通領域の説明率を区別する。中央部分だけでは全面と符号が変わることがある。',
      '', '検定は各視野の「実地図の共通領域決定係数−九対照の中央値」を対象とする片側符号並べ替え10,000回。乱数種20261004、加算1の有意確率。八比較を一群としてボンフェローニ法の八倍補正。日程・基板の組を同符号にする感度比較と日程同符号の比較も併記する。組相関を独立標本とした検定は使わない。',
      '', table(test[['仮説','定義','対象','視野数','日程数','日程基板数','差平均','差中央値','補正前有意確率','ボンフェローニ補正後','日程基板符号有意確率','日程符号有意確率']]),
      '', '## 説明できた割合と共通部分','',
      '以下の百分率は交差検証の決定係数に100を掛けた値。因果寄与の百分率ではない。負値は学習平均より予測が悪かったことを示す。']
    for method in METHODS:
        b=ex[(ex['定義']==method)&(ex['モデル']=='B')]
        ff=verified[(verified['定義']==method)&(verified['モデル']=='F')]
        jj=joint[joint['定義']==method]
        lines += ['',f'### {method}','',f'B・全面：{interval(b["全面決定係数"])}。固定29視野の回復部分：{interval(b[b["固定29視野"]]["全面決定係数"])}。',
                  '',f'B・対照共通領域：{interval(b["対照共通領域決定係数"])}。九対照中央値：{interval(b["対照決定係数中央値"])}。',
                  '',f'F・独立検証済み全面：{interval(ff["全面決定係数"])}。両者の全面：{interval(jj["全面決定係数"])}。',
                  '',f'共通部分：{interval(jj["全面決定係数_共通"])}。B固有：{interval(jj["全面決定係数_B固有"])}。F固有：{interval(jj["全面決定係数_F固有"])}。相互作用の改善：{interval(jj["全面決定係数_相互作用改善"])}。']
    lines += ['', '独立検証済み視野の全値：','',table(verified[verified['モデル'].isin(['B','F','BF','BF_interaction'])][['key','定義','モデル','全面決定係数','対照共通領域決定係数','対照決定係数中央値','全面決定係数_B固有','全面決定係数_F固有','全面決定係数_共通','全面決定係数_相互作用改善']]),
              '', '全回復視野のB説明率、F説明率が得られない理由：','', table(ex[ex['モデル']=='B'][['key','定義','全面決定係数','対照共通領域決定係数','対照決定係数中央値','対照との差']]),
              '', table(tracking[['key','支持点数','支持区画数','採用候補','理由']]),
              '', 'Fの候補地図だけの説明率・分解も `round1_bf_explained_fraction.csv` と `round1_bf_explanation_summary.csv` に全値を保存した。目印検証がない候補を確認的Fの数値に混ぜていない。全体の統計・日程別・ブランク別の中央値・四分位も同表にある。',
              '', '## Fの独立検証と限界','',
              '二十一・三十一画素窓のルーカス・カナデ追跡を使い、逆追跡誤差・窓間差0.2画素以内、最終変換から二画素以内を条件とした。百点以上、八×八区画の半分以上の支持、交互二群で推定した残差場の相互検証改善と地図差中央値0.2画素以内を必要とする。条件を事後に緩めなかった。',
              '', '260926の候補支持条件を通過した30視野のうち、非周期の目印も検証できたのは基板01視野6と基板7視野5の2視野。基板01視野8は支持区画27、基板7視野8は31で、必要な32区画に達しない。目印が一致しても全面の局所残差の検証を通過したとは数えなかった。',
              '', '前回目視済み十字線の交点について、今回も洗浄前後・重ね合わせ図を実際に開いて確認した。四画素ガウス平滑化で格子周期を弱めた十字線の独立相関でも、ずれ0〜1画素で一周期7.286画素より小さい。相関から変換を修正していない。',
              '', table(marker),
              '', '目印は縁にあり、視野中央の対応の一意性を直接保証しない。二画素の追跡範囲と独立支持検証を併用しても、局所残差が撮影像の変化・斜面・汚れに引かれる可能性は残る。Fの2視野検定、とくに外れ値1視野の検定は検出力が極めて低い。',
              '', '## 整数丸めと双一次補間の比較','',
              '方式別ブランク校正を実施したが、260926の元ブランク8視野中、回復できた7視野だけの再校正である。基板01視野7が欠ける。したがってこの比較は補助であり、元の全ブランクによる標準閾値の再現でも、方式変更の確認的効果判定でもない。補間で陰性になると仮定していない。整数丸めの補助割合が前回の割合と違うのは再校正に使うブランク集合の違いによる。',
              '',table(thresholds),
              '',table(sampling[sampling['固定29視野']][['key','定義','標本化','外れ値割合','密度分散','周期域窓付きパワー','完全校正']]),
              '', '260926の回復外れ値5視野では二定義とも、再校正後の双一次補間で外れ値割合・周期域パワーが増加した。Bの「補間で弱まる」という予測と逆方向であり、不都合な結果として載せる。ただし閾値自体の低下を含むので、それだけでBを否定しない。',
              '', '## 端数地図の補助相関と等位相の予測','',
              '回復した外れ値視野の保存済み洗浄前格子点を最終変換で写し、丸め前座標の端数から予測地図を作った。外れ値差を用いて変換を探索していない。重ね合わせ図は `bf_overlay/`。升目を独立標本とする検定は行わず、記述的相関として示す。',
              '',table(phase),
              '',table(gradient[gradient['固定29視野']][['key','変位成分','一画素変化の予測間隔画素','予測帯軸度','観測第一軸度','観測自己相関周期画素','軸差度','観測分類']]),
              '', '等位相の予測間隔は最終変位勾配の大きさの逆数であり、六方格子の標本化が加えるうなり全てを表す値ではない。勾配の各成分を使い、回転角・平均倍率だけから元座標を再構成していない。観測の周波数格子・32画素升目の分解能を考慮し、整数調波への一致を細い帯の証拠としない。',
              '', '## 生画像コピーと別日程・単独画像','',
              f'報告作成時の生画像直下の実在名：`{raw}`。'+', '.join('`'+p.name+'`' for p in children)+f'。追加日程完了印：{extra}。',
              '', ('別日程の回復・B検定は利用可能なコピーについて別日程集計を出した。Fは非周期目印を検証できた範囲だけを採用する。' if recovery['日程'].nunique()>1 else '追加日程の `copy_done_2` 完了印を確認できなかったため、260922・260924・260828の別日程の再現は未実施。コピー途中の画像を完全な入力とみなしていない。'),
              '', '260830の実在フォルダ名は `260830_p50＿同一視野にてsam`（全角の下線を保持）。六つの単独画像は前回確認済み。撮影時点・洗浄前後の対応表がないため、番号からペアを推測していない。B・Fの同一分布という予測の確認的検定は未実施。洗浄前だけ・洗浄後だけの固定座標による前回の図は、洗浄後に追従していないため、同一ピラーの差の代わりにしない。',
              '', '## 事実と解釈、残る説明','',
              '事実（確度：高）：260926で保存差・格子を回復34視野、未回復36視野。Bの予測性能は平均予測を大半で上回らず、補正後の検定は有意でない。Fの独立検証可能視野は2視野で、その外れ値基板7視野5では全面説明率が平均・標準偏差定義で約2.31%、中央値・絶対偏差定義で約3.85%。共通部分とB固有分は小さく、負の固有分もある。',
              '', '解釈（確度：中〜低）：この条件で単純な丸め誤差・第一殻周期像だけが帯の主因との説明は弱い。Fは一視野で小さい説明力があるが、全外れ値視野に一般化できない。Bの局所的なピラー像の不均一・高次の周期・像と格子の局所位相差をこの低次周期モデルが取り切れていない可能性も残る。',
              '', '別の有力な説明：洗浄・乾燥の残渣、ピント・照明の空間変化、洗浄前後の撮影条件、ゴミ・シミの周辺影響。基板番号と処理順・濃度が交絡し、回復可否と画質・マスクが関連する可能性もある。分子の真の不均一吸着も、今回の非有意だけでは排除しない。',
              '', '## 方法の弱点と未確認事項','',
              '- 回復視野は無作為標本ではなく、外れ値視野と高倍率条件の偏りを伴う。全632視野、固定29視野への推論は制限される。',
              '- 保存差の完全一致は丸め後標本値の強い監査だが、丸め前座標や当時の実行コミットの一意性を保証しない。',
              '- Bの周期像は洗浄前全体の第一殻だけであり、局所形状・高次成分を省く。方向・位相を結果に合わせて最適化していない。',
              '- Fの追跡場は平滑化帯域幅256画素に依存する。追跡は像の輝度変化の影響を受け得る。検証用の非周期目印は縁に偏る。',
              '- 視野は同じ基板・同じ閾値を共有する。日程・基板の符号感度比較を示したが、少数日程では依存を十分評価できない。',
              '- 空間移動対照の共通領域は中央だけ。全面の説明率との違いを残した。負値・欠測・不一致・逆方向の結果を切り捨てていない。',
              '- 完全な補間方式別ブランク校正、局所残差補正後の完全な方式別校正、260830の前後対応、他日程の再現は未確認の部分を明示した。',
              '', '## 入力・出力・作業状態','',
              f'作業ブランチ：`{branch}`。作業開始時・報告時のコミット：`{head}`。v28は前回使用済みの指定フォルダを継続利用する最新指示に従った。解析結果の既存v25〜v27、ブランチ、タグを確認した。v28の別ブランチ・タグは見つからなかった。旧スクリプト・旧報告・予測文書は保持した。標準経路・既定値は変更していない。コミット・タグ・プッシュはしていない。書込みをコードの指定フォルダと生成物の保存先に限定するため、作業中の履歴更新も行っていない。',
              '', f'使用した生画像{len(integrity)//3*2}ファイルと保存差{len(integrity)//3}ファイルについて、各視野の実行前後の指紋一致を確認した。実在パスと含まれる視野は `round1_bf_input_integrity.csv` と `round1_bf_v24_recovery.csv` に全件記録する。保存差の読取り専用起点は `'+str(LOCAL/'20260928_digital_judgment_current_alignment'/'tables'/'cached_field_differences')+'`。',
              '', '今回参照したラボノートは `'+str(LOCAL/'lab_notes')+'` の実在八ノート。`偽陽性の帯状分布.md` と `261002_偽陽性の帯状分布_` で始まる七ノートを読み、仮説Aの再実験はしなかった。前回の縁・位置6・7・十字線の数値は前回完了版報告を採用する。',
              '', '未追跡ファイルの状況：','', '```text',status.rstrip(),'```',
              '', '## 次の周に進む場合の提案','',
              '1. 撮影順序・洗浄前後の撮影条件（仮説I）とピント・照明の変化（仮説G）。Bの補間で構造が弱まらず、単独像にも明暗の広い構造があるため、撮影記録と対応した差の変動を先に切り分けたい。',
              '2. 洗浄・乾燥痕（仮説C）。基板番号と処理順の交絡を日程内で扱い、洗浄・乾燥の操作記録と分布の向きを照合する。',
              '3. ゴミ・シミの周辺影響（仮説H）。目視承認された領域だけを比較用に使い、影響範囲を距離で測る。標準マスクへの採用は別の意思決定とする。',
              '4. 仮説Fの拡充と仮説Bの局所像モデル。未回復視野の当時行列・実行コミット・依存版が得られれば回復監査を先行し、独立した非周期目印の支持を増やす。現在の結果に合わせた変換探索は行わない。',
              '5. 金蒸着・鋳型の不均一（仮説D・E）、その後に分子の不均一吸着（仮説J）。洗浄前後の撮影・残渣の成分を分けた後に、別基板・分子なしコントロールで検証する。',
              '', '次の仮説・追加コピー・測定の採否はClaude Codeの検収で決める。この報告で止まる。',
              '', '## 本文とは別の出力一覧','',
              '個別ファイルのパス・内容は以下と `round1_bf_output_inventory.csv`。前回の全出力一覧は変更せず `round1_output_inventory.csv` と前回完了版報告に保持した。今回追加分の一覧に続けて、読取りに用いた全入力パスを載せる。']
    paths=sorted([p for p in OUT.rglob('*') if p.is_file() and ('bf_' in str(p.relative_to(OUT)) or 'B_F検定編' in p.name)],key=str)
    paths+=sorted((ROOT/'field_level'/'v28_band_origin_round1').glob('field_round1_bf_*'))
    paths+=[ROOT/'field_level'/'v28_band_origin_round1'/'NOTES.md',REPORT,OUT/'round1_bf_output_inventory.csv']
    paths=sorted(set(paths),key=str)
    inventory=[dict(実在パス=str(p),内容=output_content(p),容量バイト=p.stat().st_size if p.exists() else np.nan) for p in paths]
    csv(inventory,'round1_bf_output_inventory.csv')
    lines += ['',table(pd.DataFrame(inventory)[['実在パス','内容']]),'', '## 使用した全生画像・保存差の実在パスと視野','',table(pd.DataFrame(integrity)[['key','種類','実在パス','不変']])]
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    inventory=[dict(実在パス=str(p),内容=output_content(p),容量バイト=p.stat().st_size if p.exists() else np.nan) for p in paths]
    csv(inventory,'round1_bf_output_inventory.csv')
    print(str(REPORT),flush=True)

if __name__=='__main__':generate()
