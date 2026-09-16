# Phase 1(align_and_match_dataframes)の本番パイプラインでの使用有無と影響範囲 — 2026-09-16/17

指示役からの依頼: `align_and_match_dataframes`(Phase 1、溝検出+ICP)がPosition 6で
物理的にあり得ない縦シフト(dy=84.8px)を返すことが判明したが、2026-09-07〜09-10の
パイプライン再構築(canonical_grid方式への移行)により、この経路が既に置き換えられている
可能性がある。実際にコードの呼び出し関係を確認し、実データへの影響を評価する。

**本調査で参照したのはGoogle Drive上のObsidianラボノート、および結果CSV/レポートのみである。
PC-alignment-anti(凍結対象)のGitリポジトリ本体は一度もclone・open・実行していない。**

## 結論

**Phase 1(`align_and_match_dataframes`)は、再構築後の現行パイプライン(canonical_grid方式)
でも引き続き使われている。置き換えられてはいない。** ただし現行パイプラインの正本は
`PC-alignment-Codex`ではなく`PC-alignment-anti`であり、その`registration.py`の
`align_and_match_dataframes`は、本リポジトリの`shared/registration.py`が持つ関数と
（`invalid_mask`引数を除き）同一である。260826 Sample1・Position 6の実データで
`align_and_match_dataframes`を実行したところ、**本リポジトリでの再現実験(dy=84.8px、
ICP反復15回)と、canonical_grid本番結果CSVのicp_iterations(15回)が完全一致**した。
Position 6のexceeds_rateはこの日の同濃度内で最大(1.44%、次点の2倍)だったが、
**この日のSample1 vs Blank統計検定はBlankがn=1で既に検出力がほぼ無く**、Position 6が
結論を反転させた証拠はない。ただし異常な粗シフトを許したまま(ICP収束エラーにならず)
本番データへ混入していたこと自体は事実であり、既存QC(ICP非収束エラー、目視欠陥除外)を
すり抜けていた。

## 1. 現行パイプラインでのPhase 1の使用有無

`G:\マイドライブ\Obsidian Vault\ラボノート\06_解析\260910_p50全濃度系列_rawQCとcanonical_grid直接サンプリング_結果と引き継ぎ.md`
に添付された`PHASE0_repo_verification_report.md`(2026-09-10、計算開始前の正本確認レポート)
を確認した。

- **canonical_grid経路の正本は`PC-alignment-anti`の`main`ブランチ**(`registration.py`
  SHA-256 `bb5e3680…719c`、`PIPELINE.md`記載のハッシュと一致確認済み)。
- 濃度系列サンプリングを行う`run_canonical_theoretical_grid.py`の処理内容(同レポートより):
  1. pre/post画像読み込み
  2. **peak検出CSVを`align_and_match_dataframes(..., local_refinement=False)`に渡し、
     変換の推定だけに使用**(docstring: "Peak detections are used only to fit the
     already hold-out-validated transform; no detection coordinates select, remove,
     or otherwise define sampled grid rows")
  3. pre画像のFFTから理論格子(方位・位相原点)を算出
  4. pre側格子点のコントラストをサンプリング
  5. **手順2で得た変換の逆変換**でpre格子点をpost画像座標へ投影
  6. post画像で検出を経由せず直接コントラストをサンプリング

**すなわちPhase 1(`align_and_match_dataframes`)は、post画像への射影に使う変換の
推定源として、canonical_grid方式の内部で今も使われている。** 「検出を経由しない
直接サンプリング」というのは、位置合わせ自体を検出に頼らないという意味ではなく、
「変換推定後の**格子点選択**に検出結果を使わない(peak検出は変換のフィッティングにのみ
使い、どの格子点をサンプリングするかには使わない)」という意味であり、**変換の推定
そのものはPhase 1のICPに依存したままである。**

なお`local_refinement=True`がデフォルトだが、canonical_grid経路は明示的に
`local_refinement=False`を指定しており、8×8タイル局所補正は使われていない
(溝ランドマーク粗シフト+六方格子回転推定+2次多項式ICPのみ)。前回の報告
([`MASKED_SCORING_AND_PIPELINE_VERIFICATION_20260916.md`](MASKED_SCORING_AND_PIPELINE_VERIFICATION_20260916.md))
で「Phase 1は局所補正を持つ」と述べたが、これは`pillar_level/v1`(Codex側、
`local_refinement`を明示せずデフォルトtrueで呼ぶ)についての記述であり、
**anti側のcanonical_grid経路には局所補正が入っていない**という違いがある。訂正する。

## 2. 実データでの影響範囲確認

### Phase 1(Codex版・anti版とalign_and_match_dataframes本体は同一)をPosition 6の実データで再実行

260826 p50 SAM Sample1のPosition 1・6のpre/postペアに対し、`shared.registration.align_and_match_dataframes`
を実行した(ピラー検出はCodex側の`shared.analyzer`を使用。anti側は別実装のためピクセル単位の
座標は完全一致しないが、`align_and_match_dataframes`本体のロジックはanti/mainとバイト同一)。

| Position | 溝検出dx (px) | 溝検出dy (px) | ICP後並進 | ICP反復回数 | 対応点マッチ率 |
| --- | ---: | ---: | --- | ---: | ---: |
| 1 | 14.38 | −0.47 | [14.34, −0.42] | 12 | 85.14% |
| **6** | 15.09 | **84.78** | [15.04, **84.79**] | 15 | 75.86% |

### canonical_grid本番結果(anti正本の実行結果、2026-09-10)との突き合わせ

`G:\マイドライブ\1.実験データ_gdrive\5.解析結果_remo\260910_canonical_grid_concentration_series\per_fov_canonical_grid_concentration_series.csv`
から260826・Sample1(1nM)の全視野を抽出した。

| Position | icp_iterations(本番) | exceeds_rate_u3 | delta_post_minus_pre_u3 |
| --- | ---: | ---: | ---: |
| 1 | 12 | 0.0012% | +0.000107 |
| 3 | 17 | 1.079% | −0.000518 |
| 4 | 25 | 0.380% | −0.000223 |
| 5 | 17 | 0.026% | −0.000083 |
| **6** | **15** | **1.445%** | −0.000457 |
| 7 | 15 | 0.692% | −0.000138 |
| 8 | 20 | 0.600% | −0.005733 |

**Position 1のicp_iterations(12)、Position 6のicp_iterations(15)は、本レポートの
再現実験と完全一致した。** これは、本番のcanonical_grid実行でも同じ入力・同じ
`align_and_match_dataframes`ロジックが同じ収束経路をたどったことを強く示唆する
(ピラー検出器がanti/Codexで別実装のため断定はできないが、反復回数の完全一致は
偶然とは考えにくい)。**Position 6のexceeds_rate(1.445%)はこの日のSample1で最大**で、
次点(Position 3, 1.079%)の約1.3倍、Position 1(0.0012%)の約1200倍だった。

`icp_converged`列は全てTrueであり、Position 6のdy=84.8pxはICP非収束エラー
(`ValueError("ICP did not converge.")`)を起こさず、**通常のFOVとして本番データに
混入していた。**

### 統計検定への影響評価

260826 SAM Sample1(1nM)vs Blankの検定は、[`260910_p50全濃度系列...md`]の比較表で
p(delta)=0.500, p(exceeds)=0.500だった。この検定のBlankは**Sample9・Position 7のみ
(n=1)**であり(同日の他Blank候補は「100fMは全視野欠陥で0件」等の理由で使えなかった)、
**n=1対n=7のMann-Whitney検定は元々ほぼ検出力を持たない。** Position 6のexceeds_rate
異常だけがこの非有意という結論を左右したとは考えにくいが、根本的に検出力の低い
検定に、最も外れ値的な1視野が寄与していたこと自体は事実であり、望ましい状態ではない。

### 他日程での確認(限定的)

同じくSample1(1nM)についてPosition別exceeds_rateを260828・260829でも確認した。

| 日程 | Position 1(清浄) | Position 5/7(清浄) | Position 6 | 備考 |
| --- | ---: | ---: | ---: | --- |
| 260826 | 0.0012% | 0.026% / 0.692% | **1.445%(最大)** | Position 6が明確に最大 |
| 260828 | 0.049% | (データなし) | 1.810% | Position 3(2.476%)がPosition 6より高い |
| 260829 | 0.036% | 0.042% / 0.030% | 1.589% | Position 2(1.902%)・3(1.665%)も同水準で高く、Position 6は突出しない |

**Position 6は260826では明確な外れ値だったが、260828・260829では
Position 2・3・4等と同程度の「やや高いグループ」の一員であり、Position 6だけが
突出しているわけではない。** これは既存の画像フォレンジクス
([`POSITION6_IMAGE_FORENSICS_20260916.md`](POSITION6_IMAGE_FORENSICS_20260916.md))で
確認した通り、書き込みフィールド境界がPosition 6・7で最も強く、Position 2・3・4・8にも
弱く写り込んでいるという構図と整合する。すなわちPosition 6固有の孤立した問題ではなく、
**書き込みフィールド境界に触れる複数視野に共通する背景ノイズ源**の中でPosition 6が
最も強く影響を受けている、という位置づけが妥当である。

Position 1・5・7に一貫して低いexceeds_rate(0.001〜0.7%)を示す「清浄な」視野群が
存在する一方、Position 2・3・4・6・8は日によって0.4〜2.5%まで上下する「やや高い」
視野群を形成しており、これは[`260910_p50全濃度系列...md`]が既に報告した
「canonical gridの見かけの背景は非ピラー領域(ギャップ)の値とほぼ同スケールで動く
全視野共通ドリフト」という結論とも整合する。**Position 6は既知の一般的な変動要因
(書き込みフィールド境界由来の背景ノイズ)の中で最も強く現れる1例であり、それ単独が
「濃度依存シグナルなし」という全体結論を作り出しているとは考えにくい。**

## 3. Phase 1が置き換えられ使われていない、という前提は成立しない

依頼の前提「もし既に置き換えられ使われていない場合」には該当しなかった。
`align_and_match_dataframes`は2026-09-07〜10の再構築後もcanonical_grid経路の
変換推定として使われ続けている。したがって「実害を伴わないことを確認できる根拠」を
示す代わりに、上記2節の通り**実害の範囲を可能な範囲で直接確認**した。

## 4. 今後の検討事項(本依頼の範囲外、次の依頼向けメモ)

- Position 6のような「ICP収束はするが物理的に誤った粗シフト」を自動検出する
  QC指標(例: 溝検出dx/dyの妥当範囲チェック、対応点マッチ率の下限)が、現行の
  canonical_grid実行には存在しない。ICP非収束エラーと目視欠陥除外だけでは
  この種の異常を捉えられない。
- Position 6・7に加え、Position 2・3・4・8も書き込みフィールド境界の影響を
  一定程度受けていることが今回わかった。マスク処理・QC閾値の対象をPosition 6・7に
  限定せず、これらの視野にも拡張すべきか検討が必要
  (`shared/image_qc.py`の`bright_band_mask`・`saturation_qc`は既に座標非依存で
  全視野に適用可能だが、canonical_grid〈anti側〉には未統合)。
- 今回の確認はPosition 1・6(260826)の1組を中心とした限定的な再現であり、
  全139ペア・全8日程への系統的な影響評価は行っていない。

## 出力

- 本ドキュメントの数値は使い捨てスクリプトの実行結果と、Google Drive上の既存結果CSV
  (`260910_canonical_grid_concentration_series\per_fov_canonical_grid_concentration_series.csv`,
  `excluded_fov_canonical_grid.csv`)の直接参照による。新規のコード変更は無い。
