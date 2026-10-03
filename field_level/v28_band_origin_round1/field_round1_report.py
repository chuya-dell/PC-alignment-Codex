"""Aggregate completed field outputs and write the requested review handoff."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import itertools,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
from field_round1_analysis import ROOT,OUT,METHODS,fields,write_csv,verify_prediction

def md_table(df):
    columns=list(df.columns)
    def fmt(x):
        if pd.isna(x): return '未測定'
        if isinstance(x,(float,np.floating)): return f'{x:.6g}'
        return str(x).replace('|','／').replace('\n',' ')
    return '| '+' | '.join(map(str,columns))+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def summary(g,col):
    a=g[col].dropna().to_numpy()
    return (len(a),float(np.median(a)),float(np.quantile(a,.25)),float(np.quantile(a,.75))) if len(a) else (0,np.nan,np.nan,np.nan)

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True,encoding='utf-8').strip()

def main():
    f=fields()
    meta=pd.read_csv(OUT/'round1_fields_both_definitions.csv',dtype={'日程':str,'基板':str})
    c=pd.read_csv(OUT/'round1_spatial_classification.csv')
    assert len(c)==632*2*3
    assert not c.duplicated(['key','定義','縁除外画素']).any()
    assert c.key.nunique()==632
    joined=c.merge(meta,on=['key','定義'],validate='many_to_one')
    markers=pd.read_csv(OUT/'round1_visually_confirmed_marker_geometry.csv')
    for _,r in markers.iterrows():
        sel=joined.key==r.key
        axis=joined.loc[sel,'第一帯軸度'].to_numpy()
        from field_round1_analysis import angular_difference
        joined.loc[sel,'十字線最小角度差']=np.minimum(angular_difference(axis,r['水平線角度']),angular_difference(axis,r['垂直線角度']))
    write_csv(joined,'round1_spatial_with_metadata.csv')
    for method,g in meta.groupby('定義'):
        write_csv(g[g['この定義3percent以上']],f'round1_outlier_fields_{method}.csv')
    counts=[]
    statistics=[]
    for (method,margin),g in joined.groupby(['定義','縁除外画素']):
        for group,sub in [('全632視野',g),('固定29視野',g[g['固定29視野']]),('定義別3%以上',g[g['この定義3percent以上']])]:
            for kind in ['帯状候補','うろこ状候補','その他']:
                counts.append(dict(定義=method,縁除外画素=margin,対象=group,分類=kind,視野数=int((sub['分類']==kind).sum()),対象視野数=len(sub)))
            for col in ['カメラ軸最小角度差','六方格子最小角度差','外れ値の縁200内占有率','外れ値の縁300内占有率']:
                n,med,q1,q3=summary(sub,col)
                statistics.append(dict(定義=method,縁除外画素=margin,対象=group,指標=col,測定視野数=n,中央値=med,第一四分位数=q1,第三四分位数=q3))
            candidate=sub[sub['分類']!='その他']
            for col in ['カメラ軸最小角度差','六方格子最小角度差','十字線最小角度差']:
                n,med,q1,q3=summary(candidate,col)
                statistics.append(dict(定義=method,縁除外画素=margin,対象=group+'の帯・うろこ候補のみ',指標=col,測定視野数=n,中央値=med,第一四分位数=q1,第三四分位数=q3))
    write_csv(counts,'round1_classification_counts.csv')
    write_csv(statistics,'round1_descriptive_summary.csv')
    # Every requested coefficient is explicitly missing rather than zero-filled.
    unavailable=[]
    for _,r in meta.iterrows():
        unavailable.append(dict(key=r.key,日程=r['日程'],基板=r['基板'],視野番号=r['視野番号'],定義=r['定義'],固定29視野=r['固定29視野'],
                                丸めうなり説明率=np.nan,局所位置合わせ説明率=np.nan,両者説明率=np.nan,
                                丸め固有分=np.nan,局所固有分=np.nan,共通部分=np.nan,相互作用=np.nan,
                                空間移動対照中央値=np.nan,状態='未検定',理由='元洗浄後標本化座標・最終変換が未回復、独立検証済み局所残差がない'))
    write_csv(unavailable,'round1_explained_fraction_unavailable.csv')
    tests=[]
    for hypothesis,method,group in itertools.product(['B：画素格子と丸め','F：局所位置合わせ'],METHODS,['全632視野','固定29視野']):
        tests.append(dict(仮説=hypothesis,定義=method,対象=group,予定視野数=632 if group.startswith('全') else 29,
                          検定可能視野数=0,未補正有意確率=np.nan,ボンフェローニ法8倍有意確率=np.nan,
                          状態='未検定',注意='有意確率1や説明率0で補わない。分類用並替確率を代用しない'))
    write_csv(tests,'round1_confirmatory_test_status.csv')
    th=pd.read_csv(OUT/'round1_thresholds.csv',dtype={'日程':str})
    cutoff=pd.read_csv(OUT/'round1_cutoff_counts.csv')
    rec=pd.read_csv(OUT/'round1_transform_recovery.csv')
    semi=pd.read_csv(OUT/'round1_semisynthetic_mechanism.csv').rename(columns={'密度地図分散':'差平均地図分散'})
    semi.to_csv(OUT/'round1_semisynthetic_mechanism.csv',index=False,encoding='utf-8-sig')
    near=pd.read_csv(OUT/'round1_marker_proximity.csv')
    wanted=['260926_3_3','260926_5_3','260926_7_5','260926_7_8','260926_01_8']
    basic=meta[meta.key.isin(wanted)][['key','定義','外れ値割合','縁200内割合','縁200外割合','外れ値の縁200内占有率']]
    basic=basic.copy()
    for col in basic.columns[2:]: basic[col]*=100
    basic=basic.rename(columns={col:col+'（百分率）' for col in basic.columns[2:]})
    spatial=joined[(joined.key.isin(wanted)) & (joined['縁除外画素']==0)][['key','定義','分類','第一帯軸度','第一自己相関周期画素','第一スペクトル周期画素','帯幅画素','第一方向集中度','第二方向集中度','六方格子最小角度差','十字線最小角度差']]
    day=meta.groupby(['日程','定義']).agg(全視野=('key','count'),外れ値視野=('この定義3percent以上','sum'),ブランク視野=('ブランク','sum')).reset_index()
    write_csv(day,'round1_day_counts.csv')
    branch=git('branch','--show-current')
    status=git('status','--short','--untracked-files=all')
    commit=git('rev-parse','HEAD')
    text=f'''# 2026年10月4日・周1報告・再開後の検収用完了版

**手順1〜3の632視野の計算と補助検証を完了した。仮説B（画素格子のうなり・整数丸め）と仮説F（局所位置合わせ）は、元の最終変換・洗浄後標本化座標が回復できず、実データの確認的検定・説明率・共通部分が未完了である。周1全体の科学的検証が完了したという意味ではない。** 必要入力と未完了箇所を明示したこの報告を保存して停止する。前回の `261004_周1_報告.md` は環境停止記録として保持した。

## 1. 事実と判定

確度：高（計算・保存内容について）。原因の判定は別に示す。

- Python 3.12.14と必要ライブラリの基本読込みが成功した。生画像コピー完了を示す `_copy_done.txt` を確認した。
- 全632視野の保存差を読み、既存の平均＋標準偏差の三倍の外れ値割合を再現した。最大絶対差は {meta[meta['定義']==METHODS[0]]['既存割合差'].abs().max():.3g}。これは丸め表示の差の範囲であり、基板・日程の対応も保存した。ブランクは69視野。260922・260923の基板01をブランクにしていない。
- 従来定義の3%以上は29視野。中央値＋3×1.4826×中央値絶対偏差では104視野。定義変更の影響が大きく、同じ固定29視野の比較と定義別対象の比較を両方保存した。
- 二つの定義それぞれについて、全視野・縁200画素除外・縁300画素除外の計3,792地図を分類した。地図と重なり補正自己相関、窓付き高速フーリエ変換のスペクトルを数値保存し、全632視野の全面の図を図録にまとめた。
- 従来定義の全面分類は帯状候補62、うろこ状候補18、その他552視野。指定5視野では帯状候補3、その他2視野で、目視でうろこ状と呼ばれた基板3・5の視野3は、それぞれその他・帯状候補になった。固定した数値分類は目視印象を完全には再現しない。これを理由に分類規則を後から変えていない。
- 260926の元解析対象70視野・140画像を読み、マスクなしの粗いアフィン変換を再推定した。事前に決めた診断最大絶対差0.000001と全保存差最大絶対差0.000001の両条件に合うものは0視野。差だけが全ピラーで完全一致したのは基板01の視野5・8の2視野であり、診断条件には合わなかった。事後に許容幅を広げて採用していない。
- 仮説B・Fとも判定不能。確度：高（必要変数の欠落が理由で判定できない点について）。仮説が正しい／誤りの確度を示すものではない。

## 2. 事前予測・規則の保持

元の `261004_予測と分類規則_未実行.md` を変更せず、内容指紋 `{verify_prediction()}` の一致を解析前と解析後に確認した。元の文書は過去の知見と対象表を読んだ後、今回の地図・検定を見る前に保存されたもの。

仮説Bの予測：実際の洗浄前後標本化座標の端数から作る丸め誤差地図が外れ値分布を予測し、位置合わせが完全な人工像でも回転・倍率変化と整数丸めで構造が生じ得る。双一次補間で弱まる。洗浄前・洗浄後の単独像には位相に対応する変化が生じ得るが、差と同じ位置・符号であることは要求しない。ブランク・分子を付けない260830でも生じ得る。

仮説Fの予測：外れ値を見ずに測った局所残差変位とピラー像の斜面が、Bを条件付けた後にも分布を予測する。双一次補間にしても局所残差を残すと影響が残り、局所残差補正で弱まる。単独像の同じ帯は必須でなく、ブランク・260830でも残差と斜面があれば生じ得る。

説明率は、対象視野を含めない全視野共通平均を引いた32画素四方の外れ値密度地図の残差を、空間ブロック8分割・評価周囲2升目を学習から外す交差検証で予測する決定係数。負値を切り捨てない。Bのみ、Fのみ、両者、横縦に±256・±512画素動かす対照8通りと90度回転を同一有効範囲で比べる。固有分と共通部分は決定係数の差で分ける。予定の確認的検定は2仮説×2定義×2対象群の8検定、視野単位10,000回の符号並べ替え、ボンフェローニ法で8倍する。今回は必要変数がなく、この回帰・対照・有意検定は実行していない。

不足入力を確認した段階で、別文書 `261004_追加実装規定_再開時.md` と追加の指紋を保存した。元の予測・分類閾値・確認的検定群を変更していない。追加は変換の再現照合、人工像、分類実装の詳細の規定である。十字線から64画素の記述的重なりは生画像を見て追加した参考測定であり、事前の仮説検定ではない。

## 3. 手順1・対象と閾値

日程ごとの全ブランクピラーをプールした平均・母標準偏差、中央値・中央値絶対偏差を使う。視野単位平均の標準偏差ではない。統計比較の単位は視野。差は洗浄前−洗浄後、閾値を厳密に超えたピラーを外れ値とした。

{md_table(th)}

{md_table(cutoff)}

{md_table(day)}

指定5視野の比較。識別子 `key` は「日程_基板_視野番号」で、基板の先頭ゼロを保持する。全て260926。元表の濃度表記は、ナノモル毎リットル、ピコモル毎リットル、フェムトモル毎リットル等の単位を省略した原表記として保存する。

{md_table(basic)}

全視野の基板・日程・位置・濃度・ブランク・割合は `round1_fields_both_definitions.csv`。従来定義29視野と別定義104視野の個別一覧も保存した。

## 4. 手順2・3・分類と向き

結果を見る前の元規則を使った。波長128〜1024画素、方向10度区分、第一軸±15度の方向集中度、各視野内の升目の1,000回並べ替えによる最大方向パワーの95百分位を用いた。帯状候補は集中度0.45以上と実空間再ピークの周期一致20%以内を要求する。うろこ状候補は帯の条件に当たらず、離れた二方向の集中度が各0.20以上・合計0.60以上で、両方向の周期一致を要求する。分類用の並べ替え確率を原因の有意確率として使わない。

{md_table(pd.DataFrame(counts))}

指定5視野の全面結果。向きはカメラ横軸から時計回り、画像縦軸は下向き。周期の実空間測定分解能は32画素。第一スペクトル周期は元の周波数格子の最大ピークであり、整数調波の一致を帯の証拠にしない。自己相関で一致しない場合は、帯幅の確定値を示さない。主分類は数値条件を満たした候補に限り、目視のうろこ状・帯状の印象と一致しない結果も残す。

{md_table(spatial)}

周期構造の候補に限定した方向差の集計。十字線は目視確認した7視野のみなので、測定視野数が小さい。その他の視野の最強方向を帯の方向として混ぜていない。

{md_table(pd.DataFrame(statistics).query('縁除外画素 == 0').loc[lambda a: a['対象'].str.endswith('候補のみ')])}

全面スペクトルの最強方向と格子三方向・カメラ軸の比較、縁200・300画素の感度結果は `round1_spatial_with_metadata.csv` と `round1_descriptive_summary.csv` に全視野分ある。自己相関はゼロ詰めによる線形相関を有効な重なり数で補正した。自己相関の短い重なりは不安定なので、大ラグを原因判定には使わない。分類は欠陥の承認でも、標準解析からの除外基準でもない。

図録の密度色域は視野ごとの表示用に調整したので、色の濃さを視野間の外れ値割合の比較に使わない。右のスペクトルの横軸は片側離散周波数の列番号、縦軸は中央入れ替え後の行番号。画素単位の周期・向きの数値は上表と全条件表で示した。スペクトルを比較する時も、図の色域だけで強さを判断しない。

## 5. 仮説Aに関連する記述的重なり

仮説Aの相関・位置固定性の再検定はしていない。固定29視野のうち位置6・7は3視野。位置6・7の既知構造を直接確認した今回の生画像は260926ブランク基板01の視野6・7。確認した5視野を加えた7視野に、ほぼ水平・垂直の十字状の線を目視で確認した。これはマーカーの幾何測定であり、欠陥なし・ありの目視承認やマスク採用ではない。

{md_table(markers[['key','水平線角度','垂直線角度','水平近似誤差中央値画素','垂直近似誤差中央値画素']])}

近似誤差が大きい線は方向・位置の精度が低い。特に基板5視野3の垂直線と基板01視野8の水平線は、単純な明部・勾配追跡で隣の構造が混ざり得るので確度：低。それ以外は確度：中。線から64画素近傍の割合は参考値であり、線の不確かさを含む。

{md_table(near)}

外れ値の縁200画素内占有率は、指定5視野の従来定義で28.15〜42.10%。縁を外した内側でも外れ値割合は3.85〜4.65%であり、縁の構造だけではこれらの視野の外れ値を尽くさない（記述的事実、確度：高）。これは帯の原因が仮説Aではないことの新しい検定ではない。位置6・7の構造との角度一致も、カメラ軸との一致とほぼ同じ情報なので独立の原因証拠にしない。

## 6. 仮説B・F・回復照合と未完了の検定

元コード `scripts_and_config/field_run_digital_judgment.py` は、洗浄前の理論格子座標を最終アフィン行列で洗浄後へ写し、洗浄前後それぞれ `sample_contrast` で読んで差を保存していた。現在の `shared/registration.py` 329行の関数は、343行の最近傍整数丸めを用い、中心の周囲3×3画素の和から51×51画素ガウス背景の同じ3×3画素和を引く。画像境界が無効判定に入り、欠陥マスクは引数が指定された時だけ適用する。しかし元の追加位置合わせの実装はない。診断表の一部には `mask_stains=False` にもかかわらずマスク割合が正の行があり、当時の共有関数の版・動作を現在の関数で代用できない。

現在の粗い推定は比較専用に必要部分をコピーし、マスクなしで70視野に実行した。元診断の中央並進・回転・倍率・異方性・行列式と全保存差を照合した。差の一致率の中央値は {rec['差一致ピラー率'].median():.6g}、第一四分位数 {rec['差一致ピラー率'].quantile(.25):.6g}、第三四分位数 {rec['差一致ピラー率'].quantile(.75):.6g}。粗い推定の微小な差でも整数丸めの読み先が変わる。差だけの完全一致2視野も、丸め前の座標の一意性を証明しない。許容幅・推定値を事後調整せず、不一致も `round1_transform_recovery.csv` に全件保存した。

{md_table(pd.DataFrame(tests))}

実データの視野ごとのB・F・両者・固有分・共通部分・相互作用・移動対照は `round1_explained_fraction_unavailable.csv` の1,264行で欠測として明示した。中央値・四分位範囲も算出不能。**欠測は0%ではない。** 元変換を再現した視野がないため、局所残差の候補追跡を先へ進めなかった。一周期違いの対応を独立検証しない残差場や、単なる全体回転・倍率をFとして回帰しない。

## 7. 単独像・260830・半合成の補助結果

指定5視野と位置6・7の洗浄前後画像、中央200画素四方の拡大、洗浄前の固定座標で読んだ単独輝度地図を保存し目視確認した。いくつかの単独輝度地図に帯が見えるが、洗浄後地図は元の変換に追従しておらず、差の帯と同じ位置・符号であるという検証に使えない。生画像中央の縮小・拡大像だけから残渣・乾燥痕・実吸着を同定していない。確度：中（表示上の観察）、原因：判定不能。

`260830_p50＿同一視野にてsam` の6画像は読めており、単独画像の図を保存した。`1.tif`〜`6.tif` と撮影時点・洗浄条件の対応表がないため、番号だけで3組等に分けず、洗浄前後差・ブランク閾値・B/F検定を作っていない。利用者へ対応情報を依頼したが、この報告時点では確認できていない。したがって「コントロールで同じ外れ値の分布が出る」は未検証。

人工データは260926ブランク基板01視野1の洗浄前画像のみから固定格子の第一殻の三方向と定数の周期像を最小二乗で推定した。既知の回転0.14度、振幅0.30画素・周期448画素の局所横変位を独立に与え、整数丸めと双一次補間、局所変位未補正と既知補正を比べた。

{md_table(semi)}

ゼロ変換対照は両標本化方式とも差0。回転だけの差の標準偏差は整数丸め0.0003952、双一次補間0.0003129。局所変位未補正では0.0005742と0.0003251、既知補正では0.0004066と0.0003283。双一次補間や局所補正で常に全指標が改善したとは言えず、差平均地図分散を含め全条件を掲載した。実データの閾値0.187609を借りた参考超過率は全条件0%。像の周期成分分散は元生画像分散の約0.0795%しかなく、全視野で単一の固定周期に合わせた弱い像である。この人工像の陰性は、実データのB/F否定でも、実データの4〜5%を再現した結果でもない。方式ごとのブランク再校正をしていない参考値であり、確認的検定には使わない。実際のB/Fの分離・共通部分は依然未計算。

## 8. 検証・実装上の弱点

既知448画素の水平帯で事前分類が帯状候補を返し、自己相関周期448画素を検出すること、欠測付き自己相関のゼロラグと重なり数、方向区分が周波数点を漏らさず重複しないことを確認した。全632保存差と260926の140画像の内容指紋を計算後に再照合し、不一致0。元予測の指紋も不変。詳細は `round1_verification.json`。

実装初期に、科学計算ライブラリの信号処理の遅延読込みで動的ライブラリが実行方針に拒否された。信号処理の広い依存読込みを使わず、高速フーリエ変換と数値配列だけで線形自己相関・山の検出を実装して継続した。描画ライブラリが作業外の利用者キャッシュへ書こうとしたため、全描画キャッシュを結果フォルダへ明示した。依存先の地図がまだ生成されていない段階の分類起動1回も失敗した。その後は地図生成の完了を確認して進めた。

実行中プロセスの一覧取得もアクセス拒否だったため、一覧からの操作は行わず、自分の計算セッションへの中断で試行を停止した。入力コピーの読込みへのアクセス拒否はなく、認識ドライブ一覧を報告して入力不足と同一視する停止条件には該当しなかった。

小規模確認で、粗い変換診断の中央座標と回転の定義を現在の共有関数に合わせる実装修正、方向区分境界の点が抜ける実装不備の修正を行った。予測の閾値は変えていない。修正前の2視野の診断と、分類途中の試行ファイルは別フォルダで残した。最終の集計には入れず、図・表は修正後の全視野計算から作った。

主な弱点：升目値の並べ替えは空間相関を壊すので分類用候補化のみである。10度方向区分、32画素周期分解能、端の投影支持不足、閾値変更による候補群変更、日程・基板内依存、ブランク閾値共有がある。全面の最強方向は縁や勾配の影響を受ける。窓関数を使っても、ピークの格子量子化は消えない。縁の地図は升目中心で選び、厳密な個々のピラーの縁距離集計と最大16画素の差がある。残差地図の平均は実装どおり対象だけを除いた全視野平均だが、ブランク・日程・条件の混合による基準の影響は未検証。

## 9. 使用した実在パスと必要な追加入力

入力は全て `C:\\Users\\chuya\\dev\\PC-alignment-Codex\\data\\inputs_local` 以下。上位の一覧で実在名を確認した。全角の「＿」を含む生画像フォルダ名は原表記のまま扱った。Gドライブ・Fドライブ・凍結中の別リポジトリは参照・実行していない。

- 過去解析：`{str(next((ROOT/'data'/'inputs_local').glob('2026*digital_judgment*')))}`。直下のv25・v26・v27は存在確認済みで、読み取りと必要コードの確認のみ。
- 全632差保存ファイルの実在パスと含まれる日程・基板・視野・ピラー数・指紋：`round1_cache_usage_sha256.csv`。元の保存台帳 `tables/table_difference_cache_manifest_by_field.csv` の歴史的なドライブパスは今回の読込みに使っていない。
- 主な元表：`tables/table_registration_field_qc.csv`、`tables/table_alignment_quality_features_and_sensitivity_by_field.csv`、`20261002_外れ値視野_追加検証_v27/tables/全視野と外れ値判定.csv`。元実行コード：`scripts_and_config/field_run_digital_judgment.py`、`field_recompute_from_cached_differences.py`、`field_quality_sensitivity_checkpointed.py`。
- ラボノート：`data/inputs_local/lab_notes` の全8文書を先に読んだ。`偽陽性の帯状分布.md` と `261002_偽陽性の帯状分布_` で始まる7文書。コピーに元の `10_引き継ぎ`・`05_解析` の階層はなく、原本階層は確認していない。
- 生画像：`data/inputs_local/raw_readonly/260926-p50-dna`、`data/inputs_local/raw_readonly/260830_p50＿同一視野にてsam`。使った全70視野の画像パスは回復照合表、単独画像のパスと指紋は `round1_raw_single_image_audit.csv` に列挙した。

追加が必要：最優先は元解析の各632視野の最終アフィン行列、洗浄後標本化座標、当時の `shared/registration.py` と `shared/v2_registration_precision/refinement.py` 及び依存コード・パラメータ・採用マスクの情報。Bの丸め前座標と、Fの独立した残差検証を回復するために必要。元の標準経路にマスクを新しく入れる依頼ではない。全日程を検定するなら260825・260827・260828・260829・260922・260923・260924・260927の生画像も必要。固定29視野だけでなく同日ブランクと通常視野を含める。260830は画像追加より先に1〜6の撮影条件・時点・同一視野対応の記録が必要。これらの不足を解消しないまま、別の変換で元の説明率を報告しない。

## 10. 作業状態

作業ブランチ `{branch}`、コミット `{commit}`。v28は前回のこの作業が新設した版を、今回の再開指示どおり継続使用した。過去解析の直下、作業の版一覧、保存済みブランチ・タグに別のv28の衝突を認めなかった。遠隔の最新状態は取得していない。今回の書き込み先を二か所に限定したため、Git管理領域を更新する取得・取り込みは行わなかった。

追跡ファイル変更なし。未追跡はこのv28のコード・備考だけ。生成物は既存の除外設定により通常の変更一覧に出ない。コミット・タグ・プッシュ・追加なし。元予測、前回停止報告、旧版、入力、標準処理・既定値は保持した。

```text
{status}
```

## 11. 本文と別の出力一覧

以下は出力の各実在パスと内容。大量の視野別保存も省略せず列挙する。別ファイル `round1_output_inventory.csv` にも同じ一覧と容量を保存した。`m0` は平均・標準偏差、`m1` は中央値・絶対偏差、`e0/e200/e300` は縁除外0・200・300画素。配列の `power` は片側スペクトル、`autocorrelation` は重なり補正自己相関、`overlap` は有効重なり数。図録の番号は全632視野を元一覧順に12視野ずつ分けたページ番号。各ページの左から密度地図・自己相関・片側スペクトル。

'''
    # Annotate all artifacts, including pilot outputs, separately from conclusions.
    def describe(p):
        rel=p.relative_to(OUT) if OUT in p.parents else p.relative_to(ROOT)
        s=str(rel).replace('\\','/')
        if s.startswith('classification_checkpoints/'): return '修正後の視野別分類6条件の途中保存。'+p.stem
        if s.startswith('spatial_arrays/'): return '修正後の自己相関・重なり・スペクトル6条件の数値配列。'+p.stem
        if s.startswith('atlases/'): return '全視野の密度・自己相関・スペクトル図録のページ'
        if s.startswith('pilot_'): return '実装確認途中の保存。最終集計には不使用'
        if 'pilot_diagnostic_definition' in s: return '診断定義の実装修正前の2視野。最終集計には不使用'
        if s.startswith('transform_recovery/'): return '視野別変換再現照合、画像指紋、候補行列、不一致の記録'
        if s.startswith('raw_review/'): return '単独生画像・未位置合わせ固定座標輝度地図・拡大の比較図または配列'
        if s.startswith('matplotlib_config/'): return 'この結果内に限定した描画用キャッシュ'
        mapping={
            'density_maps.npz':'632視野×2定義の32画素密度地図、有効数、視野識別子',
            'common_profile_residuals.npz':'対象自身を除く共通平均と外れ値密度残差の全視野配列',
            'round1_explained_fraction_unavailable.csv':'1,264視野定義行の未検定説明率・共通部分・対照を欠測で明示',
            'round1_confirmatory_test_status.csv':'予定8検定の未実施、有意確率欠測、必要変数不足',
            'round1_spatial_with_metadata.csv':'3,792分類条件の結果と日程・基板・位置・格子角度・縁割合',
            'round1_spatial_classification.csv':'3,792条件の分類・集中度・周期・幅・向き',
            'round1_classification_counts.csv':'定義・縁幅・対象群別の分類件数',
            'round1_descriptive_summary.csv':'方向差・縁占有率の中央値と四分位範囲',
            'round1_thresholds.csv':'9日程のブランク統計と2閾値',
            'round1_fields_both_definitions.csv':'632視野×2定義の外れ値割合・格子角度・縁内外率',
            'round1_cache_usage_sha256.csv':'全632保存差の実在パス・含まれる視野・内容指紋',
            'round1_day_counts.csv':'定義と日程別の対象件数',
            'round1_cutoff_counts.csv':'二つの定義で1・2・3・4・5%以上の視野数',
            'round1_transform_recovery.csv':'260926の70視野の変換再現照合結果',
            'round1_raw_single_image_audit.csv':'確認した単独生画像と260830の実在パス・画素統計・指紋',
            'round1_visually_confirmed_marker_geometry.csv':'目視確認した7視野の十字状マーカーの線近似と誤差',
            'round1_marker_proximity.csv':'十字状マーカー64画素近傍との記述的重なり',
            'round1_semisynthetic_mechanism.csv':'8人工条件の差の強さ・地図分散・参考超過率',
            'semisynthetic_maps.npz':'人工条件の差平均地図・像の係数・既知行列',
            'semisynthetic_comparison.png':'8人工条件の比較図',
            'semisynthetic_template_limitations.json':'人工周期像の小さな振幅と代表性不足',
            'round1_verification.json':'既知周期・自己相関・方向分割・入力不変の検証結果',
            '261004_周1_報告.md':'前回の環境停止記録。今回保持',
            '261004_予測と分類規則_未実行.md':'前回の事前予測と分類・検定規則。内容不変',
            'prediction_sha256.csv':'元事前予測の内容指紋。内容不変',
            '261004_追加実装規定_再開時.md':'必要入力不足を受けた追加実装の事前規定',
            'supplement_prediction_sha256.csv':'追加実装規定の内容指紋',
            'round1_output_inventory.csv':'全作業成果の個別実在パス・容量・内容一覧',
            '261004_周1_報告_完了版.md':'今回の検収報告。未完了の仮説検定と説明率も明示',
        }
        if p.name in mapping:return mapping[p.name]
        if p.suffix=='.py':return '今回の比較専用の実行コード。実装と再実行手順は備考に記載'
        if p.name=='NOTES.md':return '版の実装・結果・既知の問題・再実行手順'
        if p.name=='field_round1_preflight.ps1':return '前回の入力監査スクリプト。今回保持'
        if p.name.startswith('round1_outlier_fields_'):return '定義別に3%以上の視野の個別一覧'
        return '前回の入力監査・既存対象表・出所または内容指紋の記録。保持'
    final=OUT/'261004_周1_報告_完了版.md'
    inventory_path=OUT/'round1_output_inventory.csv'
    paths=sorted([p for p in OUT.rglob('*') if p.is_file()]+[p for p in (ROOT/'field_level'/'v28_band_origin_round1').iterdir() if p.is_file()]+[final,inventory_path],key=str)
    paths=list(dict.fromkeys(paths))
    inventory=[dict(実在パス=str(p),バイト数=p.stat().st_size if p.exists() else 0,内容=describe(p)) for p in paths]
    text+=md_table(pd.DataFrame(inventory)[['実在パス','内容']])
    text+='''

## 12. 次に検証する順序の提案（決定はClaude Code）

1. **仮説BとFの必要入力回復を最優先にし、同じ周1の実データ検定を完結する。** 帯が単独の固定座標標本化地図にも見えること、元の回復条件が満たせないことから、別仮説へ移る前に丸め前座標・当時の処理・独立した局所残差をそろえる必要がある。BとFは同じ評価範囲・分割で並行して分離し、どちらかの説明率を先に原因とみなさない。
2. **仮説G（ピント・照明）とI（撮影条件・順序）。** 現物の低周波照明変動と、洗浄前後の輝度・位相差が残る。撮影時点と条件が確定した260830、同日ブランク、単独像の比較が使える。
3. **仮説H（ゴミ・シミ・傷の周辺）。** 中央にも明暗の斑点が見えるが、欠陥としての利用者承認はなく、候補を解析除外に使わない。承認された原画像領域と外れ値の距離を調べる。
4. **仮説C（洗浄・乾燥）。** B・F・撮影の説明分を先に把握し、基板番号と処理順・濃度の交絡を記録でほどいたうえで検証する。
5. **仮説D・E（蒸着・成形）、続いてJ（実吸着）。** 製作工程の独立情報・形態観察が必要。差の地図だけから実吸着を結論しない。仮説Aは今回再実験せず、既存の位置固定性結果を維持する。
'''
    final.write_text(text,encoding='utf-8')
    for row,p in zip(inventory,paths):
        if p.exists():row['バイト数']=p.stat().st_size
    write_csv(inventory,'round1_output_inventory.csv')
    for _ in range(3):
        for row,p in zip(inventory,paths):
            if p.exists():row['バイト数']=p.stat().st_size
        write_csv(inventory,'round1_output_inventory.csv')
    print('report saved',len(paths),'artifacts',flush=True)

if __name__=='__main__':
    main()
