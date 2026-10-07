# v61 実中心サンプリング・精密化と層別集計

## 実装内容・旧版からの変更点
v59を複製し旧版を残した。連続周波数のFourier精密化を実行。threadpoolctlは任意依存（未導入時nullcontext）。前後別格子と必要な二次歪みを推定し、実中心対応・整数・bilinear・0.25画素16位置を再計算。対応率80%未満／以上を別集計。比較は対応済みの共通ピラー支持（rerun_round_center_valid）を基準にする。offset_relationships.csvに軸変化・基準周波数の位相変化・法線方向への換算位置変化を出力。

## 結果概要と保存先
data/results/v61_actual_centers_sampling/：90視野（固定29＋同日程blank62、重複1）の全入力指紋・samples_fourier・audits_fourier・fit_audit.csv・sampling_metrics_stratified.csv・stratum_summary.csv・common_support_paired_comparison.csv・offset_relationships.csv・step1_stratified.png・step2_offsets.png・atlas/。
精密化後も対応率80%未満46視野、以上44視野。固定29は全て80%未満で、>=80%固定29比較はn=0。この欠測を隠さない。
>=80%の44視野（全て同日程ブランク）では共通支持の陽性率中央値0.2708→実中心bilinear0.2118%、方向パワー割合0.1075→0.0952。固定29の低対応層では共通支持陽性率2.8885→2.0094%、方向パワー割合0.1358→0.3549、帯域パワー中央値2573.07→90.93。方向比率と絶対振幅は異なる。低対応群の欠測は偏りが残り、有効性を確定しない。

## 既知の問題・未解決事項
FFT精密化は依存不足で止めないが中心対応の欠測は解消しない。中心位置で読む効果の断定には高対応の固定29が必要。各方法の閾値は全同日程ブランクから作る（対応高層だけに再定義しない）。共通支持比較でも方法ごとの再閾値を含む。savedとrerunの登録行列差があり、savedとの変化を読取法だけに帰属しない。全再計算のstored相関最小値は結果監査参照。基準位相の換算位置は単一周波数モードの指標で画像全体の剛体移動ではない。
