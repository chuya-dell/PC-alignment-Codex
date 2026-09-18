# Phase 2 (QCゲート込み)による本番濃度依存性解析の一気通貫再実行 — 2026-09-18

## 依頼内容と背景

これまでの検証で、本番濃度依存性解析が呼ぶPhase 1(`align_and_match_dataframes`)には
固定ROI内の最小輝度点を検証なしに十字傷として採用する構造的欠陥があり、824〜829の
6日程・382組中68%で画像中心変位10px以上の手法間差分を生んでいたことが確定した
([`PHASE1_PRODUCTION_USAGE_VERIFICATION_20260916.md`](PHASE1_PRODUCTION_USAGE_VERIFICATION_20260916.md))。
標準方式として確定したPhase 2(`register_image_pair_affine`)にはQCゲートを実装済み
([`PHASE2_TRANSFORM_QC_GATE_20260918.md`](PHASE2_TRANSFORM_QC_GATE_20260918.md))だが、
Phase 2を使った本番品質の一気通貫解析(FFT格子生成→pillarごとのコントラスト抽出→
Blank閾値決定→FOV単位exact Mann-Whitney集計)は、これまで位置合わせだけを切り出した
使い捨て比較に留まっていた。

本レポートは、`PC-alignment-anti`(正本・本番パイプライン、参照のみ、コード変更なし)から
有用なロジック(FFT格子点生成、Blank閾値、FOV単位Mann-Whitney)を`PC-alignment-Codex`側に
移植し(出典は各ファイルのdocstringに明記)、Phase 2(QCゲート込み)と組み合わせた新しい
エントリポイント[`field_run_phase2_production_reanalysis.py`](../field_level/v9_phase2_production_reanalysis/field_run_phase2_production_reanalysis.py)
を、824〜829の6日程・382組全件で実行した結果をまとめる。

## 実装の要点

- 移植元・移植内容: [`field_level/v9_phase2_production_reanalysis/NOTES.md`](../field_level/v9_phase2_production_reanalysis/NOTES.md)。
- `AffineTransformQCError`を送出したFOV/サンプル対は`registration_qc_rejected`として記録し、
  Blank閾値・Mann-Whitney計算から除外した。
- 入力ペア一覧(pre/postパス・sample/position/concentration/is_blank)は、2026-09-17の
  Phase 1/Phase 2全数比較で生成済みの`pair_transform_comparison_with_flags_20260917.csv`を
  そのまま使用し、2026-09-17の使い捨て再計算と直接比較可能にした。

## 入力ペア一覧の訂正

依頼は「382組全件」と述べていたが、その382という数はGoogle Drive保存の
`pair_transform_comparison_with_flags_20260917.csv`の行数であり、これは
**2026-09-17比較実行時にPhase 1・Phase 2の両方が成功した行だけを残したファイル**
だった。Phase 1が失敗した(が Phase 2単独なら登録できる可能性がある)ペア
——本タスクがPhase 1依存から切り離したいペアそのもの——がここで暗黙に落ちて
いた(例: 260827_p50_dna sample12のposition 3・5)。

そのため本再実行では、382行版ではなく、フィルタ前の409行版
`pair_transform_comparison_20260917.csv`(dataset×sample×positionの完全な試行
一覧、Phase 1側の成否は問わない)を入力ペア一覧として使った。この409件全てに対して
Phase 2(QCゲート込み)で新規に登録・格子サンプリングをやり直した(Phase 1側の結果は
一切参照していない)。

## 結果

409組中399組が受理され(Phase 2 QC合格)、10組が`registration_qc_rejected`として
除外された(詳細は`data/results/v9_phase2_production_reanalysis/phase2_reanalysis_fov_summary.csv`)。
除外組の変換診断(中心変位・回転・スケール・異方性)は明確に物理的破綻を示している
(例: 260825_p50_dna sample2-position8はdx=-2003px、異方性5634)。

Blank閾値(プールしたBlank格子点コントラスト差分の`mean-3*std`)とFOV単位exact
Mann-Whitney検定(閾値超過率`exceeds_rate`に対して実施、
[`field_level/v9_phase2_production_reanalysis/NOTES.md`](../field_level/v9_phase2_production_reanalysis/NOTES.md)参照)の
結果は`data/results/v9_phase2_production_reanalysis/phase2_reanalysis_mannwhitney_summary.csv`
に保存した。6つの正本日程すべてで、Sample 1(1nM)とBlankの間に統計的に有意な差は
見られなかった(全て両側p>0.05)。

## 2026-09-17使い捨てMann-Whitney再計算との比較

`field_compare_to_20260917_recompute.py`で、2026-09-17の使い捨てPhase 2代入
Mann-Whitney再計算(`phase2_recompute_mannwhitney_summary_20260917.csv`)と突き合わせた。
**6つの正本日程全てで、n_high・n_blank・両側exact p値が完全一致した。**

| dataset | n_high | n_blank | p(two-sided, exact) |
| --- | ---: | ---: | ---: |
| 260824_p50_SHC6OH | 8 | 6 | 0.754579 |
| 260825_p50_dna | 7 | 7 | 0.053030 |
| 260827_p50_dna | 8 | 7 | 0.612587 |
| 260828_p50_dna | 8 | 7 | 0.189277 |
| 260828_p50_sam | 5 | 8 | 0.621601 |
| 260829_p50_sam | 8 | 7 | 0.955089 |

(新規実行値と2026-09-17再計算値は表示桁まで完全一致。詳細は
`data/results/v9_phase2_production_reanalysis/phase2_reanalysis_vs_20260917_recompute.csv`。)

409行版には2026-09-17再計算にない2日程(260826_p50_sam: n_high=7,n_blank=**1**;
260829_p50_dna: 1nMサンプル行が`series=reference_mismatch`でn_high=**0**)も含まれる。
これらが「6日程」の正本集合から外れていた理由(Blankが実質1視野しかない/1nM行が
正規のprimary系列でない)が、今回の再実行で裏付けられた。

## 260825_DNAの再確認

260825_p50_dnaの両側exact p値は**0.053030**で、有意境界(p<0.05)のすぐ外側だった。
片側`alternative="greater"`(Sample1 > Blank)のexact p値は**0.981061**で全く有意でなく、
これは実際の差の向きが逆——**Blank側がSample1より高い**——であることを意味する
(2026-09-17時点の解釈「両側exact 0.053、方向はBlank>Sample1」と完全に一致)。

**したがって既存の慎重な解釈——素朴な濃度依存性(高濃度でシグナル増加)の再現ではなく、
方向が逆であるため要フォローアップ日程——を維持する。** Phase 1依存を排除し
Phase 2(QCゲート込み)で独立に再計算しても、この結論は変わらなかった。

## 保存先

- 本リポジトリ: `data/results/v9_phase2_production_reanalysis/`(Git管理外。
  `phase2_reanalysis_fov_summary.csv`、`phase2_reanalysis_blank_thresholds.csv`、
  `phase2_reanalysis_fov_exceeds_rate.csv`、`phase2_reanalysis_mannwhitney_summary.csv`、
  `phase2_reanalysis_vs_20260917_recompute.csv`、`run_log.txt`)
- Google Drive / Obsidian: `G:\マイドライブ\Obsidian Vault\ラボノート\06_解析\260918_Phase2本番濃度依存性再解析\`
  (同一ファイル群のコピー)
