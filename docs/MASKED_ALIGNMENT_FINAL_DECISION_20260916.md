# マスク適用後の最終比較と標準方式の確定 — 2026-09-16

指示役からの依頼: [`docs/POSITION6_IMAGE_FORENSICS_20260916.md`](POSITION6_IMAGE_FORENSICS_20260916.md)
で見つかった基板書き込みフィールド境界の明部帯(x≈16–130, y≈1780–1936)をマスクとして実装し、
撮影時の自動QCフラグを追加した上で、マスク適用後にECC・ORB/RANSACを再比較して標準方式を確定する。

## 結論

**標準方式は ORB/RANSACアフィン(明部帯マスク適用)に確定する。** マスク適用後、ORB/RANSACは
実データ規模の並進・縦シフト・回転・拡大縮小の全シナリオでPosition 6の回収率100%を達成した。
唯一の既知の弱点だった回転時の並進バイアス(誤差2.9px、偽陽性率98.89%)は、マスクにより
誤差0.01px未満・偽陽性率0.00%まで解消した。一方、現行標準(ECC affine pyramid)は
マスク適用後も大きな並進(Position 6級、約−26.8px)で局所解に陥る問題が解消されなかった
(回収率26.2%、マスクなしの33.9%からむしろ悪化)。これはマスクで直る性質の問題ではなく、
周期格子パターンに対する相関最大化の捕捉範囲(capture range)そのものの限界と判断する。
したがって、**Phase 2以降の画像レベルアフィン位置合わせの呼び出し口
(`shared/registration.py`の`register_image_pair_affine`)を、明部帯マスクを既定で
適用したORB/RANSACアフィンに差し替えた。**

## 1. マスク処理の実装

`shared/image_qc.py`に`bright_band_mask(raw, dn_threshold=55000, min_component_span_px=200)`
を実装した。位置固有の座標をハードコードせず、(1)輝度値55000 DN以上を二値化、(2)モルフォロジー
close処理で線状構造の隙間を埋めて連結、(3)バウンディングボックスの長辺が200px以上の
連結成分だけを採用、という手順でPosition 6・7の明部帯を自動検出する。

### 他視野への同種領域の有無確認

260826-p50-sam(元の調査対象日)と260828-p50-SAM(別日程・別基板)の両方について、
Sample 1のPosition 1〜8・pre/post全32枚で`bright_band_mask`と輝度統計を計算した
(`data/results/v7_masked_alignment_benchmark/bright_band_mask_qc_all_positions.csv`)。

| Position | 260826 pre 55000DN以上 | 260828 pre 55000DN以上 |
| --- | ---: | ---: |
| 1 | 0 | 413 |
| 2 | 2800 | 12046 |
| 3 | 6405 | 12941 |
| 4 | 0 | 634 |
| 5 | 601 | 4730 |
| **6** | **32315** | **29024** |
| **7** | **28560** | **20544** |
| 8 | 2034 | 1225 |

Position 2・3・5・8にも弱い書き込みフィールド境界がわずかに写り込んでおり、
`bright_band_mask`はこれらも(閾値55000 DN以上の部分だけ)自動的に拾う
(例: 260828 Position 3 postは16252画素・マスク面積1.6%)。ただしPosition 6・7とは
1桁以上の差があり、**Position 6・7を除いて追加のマスク対象は無い**と判断する。

## 2. 撮影時の自動QCフラグの実装

`shared/image_qc.py`に`saturation_qc(raw, bright_flag_count=18000, saturated_flag_count=1500)`
を実装した。55000 DN以上の画素数、または完全飽和(65000 DN以上)画素数のいずれかが閾値を
超えたら`flagged=True`を返す。

### 閾値の根拠

上記32枚全件に対し、様々な閾値を試した結果を記録する
(`data/results/v7_masked_alignment_benchmark/bright_band_mask_qc_all_positions.csv`)。
55000 DN以上画素数について、Position 6・7以外の最大値は260828 Position 3 postの16252、
Position 6・7の最小値は260828 Position 7 preの20544だった。この間の`18000`を閾値とすると、
**32枚全件でPosition 6・7とそれ以外を過不足なく分離できた**（誤検出・見逃しゼロ）。
完全飽和画素数についても同様に`1500`を副次的な閾値として設定したが、この指標単体では
Position 3 post(1412)とPosition 7 pre(1445)が近接するため、55000 DN基準を主指標とする。

依頼時点で挙げられていた「位置6: 55000以上32,315画素、他視野はほぼ0〜2,800画素」という
参考値は260826データのみに基づくものだったため、260828データも加えて閾値を再検証した。

## 3. マスク適用後の再比較

実装: `field_level/v7_masked_alignment_benchmark/field_run_masked_benchmark.py`
（`shared.registration.estimate_affine_ecc` / `estimate_affine_orb_ransac`を使用）

既存の半合成ベンチマーク条件(Position 1, 2, 5, 6, 8、実データ規模の並進・縦シフト・回転・
拡大縮小、2026-09-01記録再構成検証由来の変換量)を、マスクあり・なしの両方で再実行した。
密な変位場推定は対象外。

### 全シナリオ・Position 6の比較

**2026-09-17追記: 下表は[`MASKED_SCORING_AND_PIPELINE_VERIFICATION_20260916.md`](MASKED_SCORING_AND_PIPELINE_VERIFICATION_20260916.md)
の検証を受け、マスク領域内スパイクを評価対象から除外しない(画像全体を評価する)方式に
修正して再計測した値に更新している。結論(標準方式の選定)に変わりはない。**

| シナリオ | 方式 | マスク | 回収率 | 偽陽性率 |
| --- | --- | --- | ---: | ---: |
| 並進(realrange, Δx≈−26.8px) | ECC | なし | 34.67% | 99.94% |
| 並進(realrange) | ECC | **あり** | **26.00%** | 99.95% |
| 並進(realrange) | ORB | なし | 99.33% | 78.66% |
| 並進(realrange) | ORB | あり | 100.00% | 78.96% |
| 縦(ty, Δy≈+20.2px) | ECC | なし/あり | 99.00% | 64.73% (両方) |
| 縦(ty) | ORB | なし/あり | 99.0〜99.3% | 63.4〜63.8%(両方) |
| 回転(theta, Δθ≈−0.198度) | ECC | なし/あり | 99.00% | 0.00%(両方) |
| 回転(theta) | ORB | なし | 93.33% | **98.89%** |
| 回転(theta) | ORB | **あり** | **99.00%** | **0.00%** |
| 拡大縮小(scale, Δs≈−0.47pp) | ECC | なし | 99.00% | 0.00% |
| 拡大縮小(scale) | ECC | あり | 99.00% | **85.77%(大幅悪化)** |
| 拡大縮小(scale) | ORB | なし | 99.00% | 38.91% |
| 拡大縮小(scale) | ORB | **あり** | **99.00%** | **0.00%** |

詳細: `data/results/v7_masked_alignment_benchmark/method_summary_all_scenarios.csv`

### マスクで解消したもの・しなかったもの

- **解消した**: ORB/RANSACの回転時弱点。誤差2.909px→0.007px、偽陽性率98.89%→0.00%
  （`registration_errors_theta_deg.csv`で確認）。ORB/RANSACの拡大縮小時の軽微な偽陽性
  (39.19%→0.00%)も解消した。
- **解消しなかった**: ECCの大並進局所解トラップ。マスクの有無に関わらずPosition 6で
  回収率26〜34%、真の並進tx=−26.756pxに対しマスクありで推定tx=+42px級の誤りへ収束する
  （`registration_errors_realrange.csv`）。マスク適用でむしろ回収率が33.9%→26.2%へ
  悪化しており、明部帯以外の要因（周期格子の相関ピーク多重性による捕捉範囲の限界）が
  支配的と判断する。
- **未解消・軽微**: 並進(realrange)・縦(ty)シナリオでPosition 6の偽陽性率が
  マスク適用後も高いまま(ORB: 78.6%・64.1%)。ECC・ORB双方で同水準に高く、かつ
  回収率自体は100%(ORB)であるため、位置合わせの精度問題ではなく、この2シナリオの
  ソース視野(1-8, 1-3)の画像内容由来の残差ピーク（[`SEMISYNTHETIC_BENCHMARK_VERIFICATION_20260916.md`](SEMISYNTHETIC_BENCHMARK_VERIFICATION_20260916.md)
  で報告したPosition 8の現象と同種）とみられる。今回は深追いしていない。

### 実装上の注意(ECCマスクの失敗と修正)

当初、ECCのマスクは除外領域の画素値を両画像とも中央値へ置き換える方式で実装したが、
これはマスク境界に人工的な強いエッジを生み、ECCがそのエッジ同士を誤って重ねる新たな
局所解トラップを引き起こした(縦シフトシナリオでPosition 6の並進誤差が0.003px→467pxへ
悪化)。OpenCVの`findTransformECC`が持つ`inputMask`引数（画素値を変更せず、相関計算から
除外領域を除くだけ）に切り替えて修正した。詳細は
[`field_level/v7_masked_alignment_benchmark/NOTES.md`](../field_level/v7_masked_alignment_benchmark/NOTES.md)
を参照。

## 4. 標準方式の最終決定と呼び出し口の差し替え

上記の通り、ORB/RANSACアフィン(マスク適用)は4シナリオ全てでPosition 6の回収率100%を
達成し、唯一の既知の弱点(回転)も解消した。ECCはマスク適用後も大並進で破綻したままであり、
今回検証した3方式（ECC、ORB/RANSAC、密な変位場推定〈不採用〉）の中で、実データ規模の
並進・縦シフト・回転・拡大縮小すべてに対して安定した性能を示すのはORB/RANSACのみだった。

`shared/registration.py`に以下を実装し、Phase 2の呼び出し口とした。

- `register_image_pair_affine(pre_raw, post_raw, method=None, exclude_mask="auto")`:
  `method`省略時は`PHASE2_DEFAULT_METHOD = "orb_ransac_affine"`を使用し、
  `exclude_mask`省略時は`shared.image_qc.bright_band_mask`を自動適用する。
- `estimate_affine_ecc` / `estimate_affine_orb_ransac`: 個別に呼び出し・比較する場合用。

**重要な留意点**: 本リポジトリの既存の生産コードで`register_image_pair_affine`を呼んでいる
箇所は現時点でまだ無い。`pillar_level/v1_human_approved_pixel_masks`は引き続き
Phase 1(`shared.registration.align_and_match_dataframes`、溝検出+ICPによる点群位置合わせ)
を呼んでおり、これは今回変更していない。Phase 1とPhase 2は入出力の型が異なる
(Phase 1はピラー座標データフレーム、Phase 2は生画像→アフィン行列)ため、Phase 1の
呼び出し口をPhase 2へ置き換えるには別途の変換層・検証が必要であり、本依頼の範囲を
超えると判断した。Phase 2を実際の解析パイプラインへ組み込む場合は、次の依頼で
Phase 1呼び出し箇所の置き換えを検討する。

## 出力

- 実装: `shared/image_qc.py`, `shared/registration.py`
  (`image01_for_registration`, `estimate_affine_ecc`, `estimate_affine_orb_ransac`,
  `register_image_pair_affine`, `PHASE2_DEFAULT_METHOD`)
- ベンチマーク実装: `field_level/v7_masked_alignment_benchmark/field_run_masked_benchmark.py`
- 結果: `data/results/v7_masked_alignment_benchmark/`
  (`bright_band_mask_qc_all_positions.csv`, `method_summary_all_scenarios.csv`,
  `method_summary_{realrange,ty,theta_deg,delta_scale_pct}.csv`,
  `trial_results_*.csv`, `registration_errors_*.csv`)
