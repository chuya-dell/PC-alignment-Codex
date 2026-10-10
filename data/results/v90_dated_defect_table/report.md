---
確認: 未確認
状態: 未読
---
# 【未読】日付別欠陥表の開発用再評価

Codexは事前登録第3版に従い、547非ブランク視野と69ブランク視野を再評価した。Codexは本結果を事後の再評価（独立な検証ではない）として扱う。
Codexは未確認の事前登録・調査ノート・第88版の欠陥候補表を入力に使った。Codexは候補の承認状態と標準経路を変更しなかった。

|条件|標準化|判定項目|値|基準|合否|
|---|---|---|---:|---:|---|
|E3|recomputed|reduction_vs_E2_k6|0.160791|0.1|合格|
|E3|recomputed|noninferiority_vs_E1_k6|1.1501|1.1|不合格|
|E3|recomputed|late_total_count_vs_E2_k6|-36|0|合格|
|E3|recomputed|reduction_vs_E2_k8|0.276236|0.1|合格|
|E3|recomputed|noninferiority_vs_E1_k8|1.08742|1.1|合格|
|E3|recomputed|late_total_count_vs_E2_k8|-22|0|合格|
|E3|recomputed|max_lost_fraction_vs_E1|0.00574369|0.0127781|合格|
|E3|recomputed|overall_iv|0|1|不合格|
|E3u|recomputed|reduction_vs_E2_k6|0.271163|0.1|合格|
|E3u|recomputed|noninferiority_vs_E1_k6|0.998838|1.1|合格|
|E3u|recomputed|late_total_count_vs_E2_k6|-45|0|合格|
|E3u|recomputed|reduction_vs_E2_k8|0.335245|0.1|合格|
|E3u|recomputed|noninferiority_vs_E1_k8|0.998763|1.1|合格|
|E3u|recomputed|late_total_count_vs_E2_k8|-24|0|合格|
|E3u|recomputed|max_lost_fraction_vs_E1|0.00685627|0.0127781|合格|
|E3u|recomputed|overall_iv|1|1|合格|
|E4|recomputed|carryover_within_10percent_of_E3|0.0754153|0.1|合格|
|E4u|recomputed|carryover_within_10percent_of_E3|0.0212052|0.1|合格|
|E7|recomputed|carryover_within_10percent_of_E3|0.0505966|0.1|合格|
|E3|fixed|reduction_vs_E2_k6|0.1482|0.1|合格|
|E3|fixed|noninferiority_vs_E1_k6|1.15109|1.1|不合格|
|E3|fixed|late_total_count_vs_E2_k6|-34|0|合格|
|E3|fixed|reduction_vs_E2_k8|0.276236|0.1|合格|
|E3|fixed|noninferiority_vs_E1_k8|1.08742|1.1|合格|
|E3|fixed|late_total_count_vs_E2_k8|-22|0|合格|
|E3|fixed|max_lost_fraction_vs_E1|0.00574369|0.0127781|合格|
|E3|fixed|overall_iv|0|1|不合格|
|E3u|fixed|reduction_vs_E2_k6|0.260867|0.1|合格|
|E3u|fixed|noninferiority_vs_E1_k6|0.998833|1.1|合格|
|E3u|fixed|late_total_count_vs_E2_k6|-43|0|合格|
|E3u|fixed|reduction_vs_E2_k8|0.335245|0.1|合格|
|E3u|fixed|noninferiority_vs_E1_k8|0.998763|1.1|合格|
|E3u|fixed|late_total_count_vs_E2_k8|-24|0|合格|
|E3u|fixed|max_lost_fraction_vs_E1|0.00685627|0.0127781|合格|
|E3u|fixed|overall_iv|1|1|合格|
|E4|fixed|carryover_within_10percent_of_E3|0.0694811|0.1|合格|
|E4u|fixed|carryover_within_10percent_of_E3|0.0149396|0.1|合格|
|E7|fixed|carryover_within_10percent_of_E3|0.0508998|0.1|合格|
|E3|not_applicable|M5_split_match_mean|0.882875|0.8|合格|
|E5|not_applicable|reference_n8|0.159884|0.9|不合格|
|E5|not_applicable|reference_n12|0.290154|0.9|不合格|
|E5|not_applicable|reference_n16|0.431111|0.9|不合格|
|E5|not_applicable|reference_n24|0.589144|0.9|不合格|
|E5|not_applicable|reference_n32|0.68696|0.9|不合格|
|E5|not_applicable|reference_n48|0.849314|0.9|不合格|

CodexはE2の固定9中心とE1を、全期間の情報を使った後知恵の参照として扱う。CodexはE7を直前までの撮影日の和集合とし、当日の表を含めない。
CodexはE4・E4u・E7から最初の撮影日を除き、繰り越し比較の分母を同じ日付にそろえた。Codexは繰り越しの結論が260926の1回の変化に強く依存する点を残した。
Codexは同日分割の表同士の一致率を0.882875と記録した。Codexは中心ごとの両側再現率をM5.csvに分けた。
Codexは全9日で適用可能な必要参照視野数を基準を満たす数なしと記録した。
Codexは260827の47視野に対する48視野抽出を実行不能と記録し、置換抽出へ変更しなかった。Codexは板番号を文字列で保持し、260922を54視野として選別した。

## 中心の変化と尺度

Codexは中心8のblank層を「検出上の変化：低信号側にも存在」とした。Codexは低信号側の差を0.3125、高信号側の差を0.381579と記録した。
Codexは中心8のnonblank層を「検出上の変化：低信号側にも存在」とした。Codexは低信号側の差を0.321285、高信号側の差を0.497805と記録した。
Codexは中心157のblank層を「局所信号の増加/減少：低信号側がほぼ無い」とした。Codexは低信号側の差を-0.00761369、高信号側の差を0.625と記録した。
Codexは中心157のnonblank層を「局所信号の増加/減少：低信号側がほぼ無い」とした。Codexは低信号側の差を-0.000763741、高信号側の差を0.866751と記録した。
Codexは中心309のblank層を「判定不能」とした。Codexは低信号側の差を-0.0162037、高信号側の差を0.85と記録した。
Codexは中心309のnonblank層を「局所信号の増加/減少：低信号側がほぼ無い」とした。Codexは低信号側の差を-0.000577367、高信号側の差を0.649671と記録した。
Codexは境を中心8で260922と260923の間、中心157・309で260924と260926の間に限定した。Codexはカメラ・光学・汚れ・試料などの物理的原因を断定していない。

## 実装の細部（仮置き）

Codexは2種類の帰無を日付ごと・視野数ごとに1000回計算し、必要最小視野数の大きい方を採った。Codexは同じ反復番号の9日分を合わせ、1つでも偽中心が出る事象を1000反復で数えた。Codexはその事象の片側95%上限が0.05以下になる最小の視野数を全日表・半分の表について採った。Codexは採用値と走査結果をm_of_N.csvとnull_nine_table_scan.csvに保存した。
Codexは集合を保つ帰無で、視野全体の点集合に同じ平行移動を与え、座標範囲内に収め、最寄り有効ピラーへ丸めた。Codexは丸め距離が局所ピッチの半対角を超える点を有効域外として除いた。Codexは除いた数をnull_translation_diagnostics.csvに記録した。
Codexは表の和集合で重なる中心を平均せず、円の和集合を維持した。Codexは再現率で5画素以内の一対一最大対応を使った。
Codexは環状対照で半径45画素上の16候補円を試し、円全体が30〜60画素の環内に収まる同数ピラーの円を採った。Codexは当てはめ密度を円面積当たりで計算した。Codexは対照が作れない視野を欠測とし、視野数を併記した。
CodexはE8を非ブランクの滴下前画像の13候補中心の日付別平均から作った。Codexは幅の差>0.02かつ密度比<0.9を仮置き閾値として使い、合否に使わなかった。
Codexは局所の主指標を除外なしのS5中央値・中央絶対偏差で標準化した全有限値ピラーから計算した。Codexは日付内2000回の視野単位再抽出を使い、境の比較でも日付ごとの視野数を保った。
Codexは指標1と指標4の除外を第88版と同じ前後座標の円の和集合とし、有限S5かつexclのピラーを有効数の分母にした。Codexは指標3を変換後の検出座標の円で評価した。

## 保存先と検証

Codexはdata/results/v90_dated_defect_table/に全表、NOTES.md、verification.json、v90_manifest.csvを保存した。Codexは第88版の13中心・再計算標準化の全69視野×25閾値で個数・有効数・換算値の一致を検証した。
Codexは1848件の配列読込をarray_read_audit.csvに記録した。Codexは禁止日付の配列と画像を開かず、入力ファイルの実行前後の内容の不変を検証した。

## 基準への異議

Codexは点集合の平行移動で有効域への丸めと点の除去が集合の形を変える限界を記録した。Codexは全日表の9日同時事象に対する最小の共通視野数閾値を使い、日付別の偽中心数も報告した。
CodexはE5の48視野が260827で実行不能である点を記録し、基準と視野数を変えなかった。Codexは両分割表が空のときの一致率を未定義とし、空表の一致を再現成功に数えなかった。
Codexは局所信号の分類が尺度変化、欠測、日付と試料の交絡を解消しない点を残した。CodexはE8が既存13候補中心に限った探索であり、未知の画像欠陥を探索する表ではない点を記録した。
