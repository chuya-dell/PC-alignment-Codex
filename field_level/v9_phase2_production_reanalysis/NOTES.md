# Phase 2 (QC-gated) production concentration-series reanalysis

## 目的

これまでの濃度依存性解析（`PC-alignment-anti`の`run_canonical_theoretical_grid.py`が
呼ぶ`align_and_match_dataframes(local_refinement=False)`、Phase 1）は、固定ROI内の
最小輝度点を検証なしに十字傷として採用する構造的欠陥を持ち（
[`docs/PHASE1_PRODUCTION_USAGE_VERIFICATION_20260916.md`](../../docs/PHASE1_PRODUCTION_USAGE_VERIFICATION_20260916.md)）、
824〜829の6日程・382組中68%で画像中心変位10px以上の手法間差分を生んでいた。標準方式として
確定したPhase 2（`register_image_pair_affine`、QCゲート込み、
[`docs/PHASE2_TRANSFORM_QC_GATE_20260918.md`](../../docs/PHASE2_TRANSFORM_QC_GATE_20260918.md)）を
使った本番品質の一気通貫解析（FFT格子生成→pillarごとのコントラスト抽出→Blank閾値決定→
FOV単位exact Mann-Whitney集計）は、これまで位置合わせだけを切り出した使い捨て比較に
留まっていたため、本バージョンで新設する。

## 実装

`field_run_phase2_production_reanalysis.py`が新しいエントリポイント。移植元と移植内容は
スクリプト冒頭のdocstringに記載（`PC-alignment-anti`は参照のみ、コード変更なし）。

- FFT格子生成・直接コントラストサンプリングの流れ: `run_canonical_theoretical_grid.py`
  （本リポジトリの`shared/lattice_indexing.py`・`shared/theoretical_grid_evaluation.py`は
  この移植元と同一ロジックが既に移植済みだった）。
- pre格子点→post画像座標への投影: Phase 1の`inverse_transform_points`ではなく、Phase 2の
  アフィン行列を`cv2.transform`で順方向適用する。Phase 2の`warp`はpre画像の点をpost画像の
  点へ写像する（post画像点をpre画像点へではない）ことを、溝ランドマーク座標への順・逆適用の
  比較で実データにて確認した（260824 SHC6OH 1-1: pre溝(79.9,125.0)を順方向適用→
  (67.4,113.5)、実測post溝(70.0,112.5)と近い。post溝を逆行列で適用→(82.6,123.9)、
  実測pre溝(79.9,125.0)と近い）。
- Blank閾値（プールしたBlank格子点コントラスト差分の`mean - 3*std`）とFOV単位exact
  Mann-Whitney: `investigation/260907_pipeline_rebuild/scripts/run_fov_level_direct_sampling.py`
  の`summarize()`・`fov_mw`。`shared/concentration_series_stats.py`に書き直して実装
  （出典はモジュールdocstringに明記）。片側`alternative="greater"`検定を追加し、
  有意差の方向を明示できるようにした。

`AffineTransformQCError`が送出されたFOV/サンプル対は`registration_qc_rejected`として記録し、
以降のBlank閾値・Mann-Whitney計算から除外する。

## 入力

382組の対象ペア（pre/postパス、dataset/sample/position/concentration/is_blank）は、
2026-09-17のPhase 1/Phase 2全数比較で生成された
`pair_transform_comparison_with_flags_20260917.csv`（Google Drive/Obsidian保存のデータ
成果物であり、`PC-alignment-anti`のソースコードではない）をそのまま使う。これにより
今回の再実行は2026-09-17の使い捨てMann-Whitney再計算と直接比較可能になる。

## 検証（少数ペアでの一致確認、および入力ペア一覧の訂正）

260824 SHC6OH Sample1のPosition 1〜5について、`n_grid_points`
（85793, 85794, 85795, 85795, 85793）と`delta_mean`が、2026-09-17の使い捨て再計算の
`phase2_recompute_fov_exceeds_rate_20260917.csv`とほぼ完全一致した
（例: Position 1の`delta_mean` -0.002230576 vs 参照 -0.002227；ORB/RANSACの
非決定性に起因する程度の差）。

**Mann-Whitney検定はFOVごとの`delta_mean`ではなく`exceeds_rate`（閾値超過率）で
行う。** 当初`delta_mean`で検定を組んだところ、260825_DNAの両側p値が0.710となり、
参照値0.05303と大きく食い違った。個々のFOVの`delta_mean`自体は参照と一致していた
ため、参照側が検定統計量として`exceeds_rate`を使っていたと判明し（`delta_mean`と
`exceeds_rate`をそれぞれ検定して0.05303030303030303に一致したのが`exceeds_rate`側
だった）、`shared/concentration_series_stats.py`の`fov_exceeds_rate_mann_whitney`に
訂正した。

**入力ペア一覧は382行の`..._with_flags_20260917.csv`ではなく、409行の
`pair_transform_comparison_20260917.csv`を使う。** 前者は「2026-09-17比較実行時に
Phase 1側も成功した」行だけを残したファイルで、Phase 2単独なら登録できるペア
（例: 260827_p50_dna sample12のposition 3・5）を暗黙に落としてしまう——まさに
本タスクがPhase 1依存から切り離したい対象そのもの。409行版を入力にして初めて、
6つの正本日程すべてで2026-09-17のPhase 2代入Mann-Whitney再計算
（`phase2_recompute_mannwhitney_summary_20260917.csv`）とp値が完全一致した。

## 結果概要と保存先

実行結果は`data/results/v9_phase2_production_reanalysis/`（Git管理外）に、Google Drive・
Obsidian `06_解析`にも保存する。数値の詳細は
[`docs/PHASE2_PRODUCTION_REANALYSIS_20260918.md`](../../docs/PHASE2_PRODUCTION_REANALYSIS_20260918.md)
を参照。
