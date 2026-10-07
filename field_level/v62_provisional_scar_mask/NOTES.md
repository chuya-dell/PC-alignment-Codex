# v62 十字傷マスクの承認前仮置き

## 実装内容・旧版からの変更点
本人の明示許可に従いv57で凍結した632視野のpixel maskを承認前の仮置き感度分析に使用。C:/Users/chuya/PC-alignment-localcorr/data/results/v57_false_positive_facts_20261006/stepD/のマスクを読み、保存済みpillar_distのin_maskと全要素一致を確認。再検出・マスク変更・生データ書込みなし。日程別の主ブランクだけから、マスク外の平均＋母標準偏差3倍を再計算。
同一の632視野で陽性率・日程内濃度対ブランクの視野単位片側Mann–Whitney（asymptotic）を比較。本番と同じHolm補正をこのn=3の60比較全体に適用（前／後を別に補正）。本番の9閾値倍率を含む540比較とは補正家族が異なる。

## 結果概要と保存先
data/results/v62_provisional_scar_mask/field_metrics.csv、blank_thresholds.csv、concentration_comparison.csv、dose_rank_correlation.csv、input_audit.csv、mask_effect.png、固定29視野の前後図。
ピラー残存89.2607%。ブランクの視野陽性率中央値0.2599→0.0668%、視野平均0.6093→0.6541%。分子あり中央値0.2972→0.0516%、平均0.7102→0.6887%。全60濃度比較のHolm後p<.05は前後とも0件。

## 既知の問題・未解決事項
自動傷マスクは本人未承認。数値は仮置き対策比較に限り、原因結論・正式解析には使わない。中央値が下がっても平均は一様に下がらない。除外による分母と閾値の変更を含むため、除いた傷の効き目だけに帰属できない。視野単位の指定に従ったが同一基板内の依存は残る。濃度依存性がないことを証明した結果ではない。原registryの他作業行はcommitに含めない。
