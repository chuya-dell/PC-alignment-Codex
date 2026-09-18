# 260825_DNA境界近傍p値の人手検証 — 2026-09-18

## 依頼内容

2026-09-18のPhase 2(QCゲート込み)本番品質再解析([`PHASE2_PRODUCTION_REANALYSIS_20260918.md`](PHASE2_PRODUCTION_REANALYSIS_20260918.md))で、
260825_p50_dna(Sample1=1nM, n=7 vs Blank=0M, n=7)の両側exact Mann-Whitney p値が0.053030
(有意境界のすぐ外側)、方向がBlank>Sample1(逆方向)だった件について、技術的アーティファクトか
データ由来かを事実ベースで確認する。生データ・既存の解析結果は改変していない(参照・目視確認・
記録の追加のみ)。

## 1. 目視確認(Sample1・Blank計14組)

対象14組(Sample1: position 1,2,3,4,6,7,8 [position5はQCゲートで既に除外済み]; Blank:
position 1,2,3,4,5,6,8 [position7は元々存在せず])全てのpre/post画像を、1-99.5パーセンタイル
ストレッチした512x512サムネイルで目視確認した([`data/results/v9_phase2_production_reanalysis/260825_dna_visual_audit/`](../data/results/v9_phase2_production_reanalysis/260825_dna_visual_audit/)の`*_pair.png`)。

- **孤立したゴミ・シミ・傷は目立った形では見られなかった。** Blank position 4のpre画像に
  1つの孤立した明るい点(pre画像にのみ存在し、postでは消えている)、Sample1 position 3に
  pre/post両方に同一位置で存在する固定ダスト粒子を確認したが、いずれも1点のみでFOV全体
  (格子点約85,000点)への寄与は無視できる規模。
- **全視野で書き込みフィールド境界線(既知のPosition 6/7問題、[`POSITION6_IMAGE_FORENSICS_20260916.md`](POSITION6_IMAGE_FORENSICS_20260916.md)参照)が
  何らかの形で視野端に写り込んでいる。** どの辺(上/下/左/右)に出るかは位置番号によって
  異なり、角に近い位置(1, 3, 8など)では2辺が交差する形で写り込む。
- **ピント外れ・全体的な輝度異常は見られなかった。** `saturation_qc`による自動フラグは
  Blank position 5・6・8とSample1 position 6・7のpre画像(および複数postでも)で立ったが、
  これはいずれも既知の境界帯の明部画素数によるもので、ピンボケや露光異常ではない
  ([`data/results/v9_phase2_production_reanalysis/260825_dna_visual_audit/image_qc_metrics.csv`](../data/results/v9_phase2_production_reanalysis/260825_dna_visual_audit/image_qc_metrics.csv))。
- **Phase 2の登録(マスク・ORB/RANSAC)は正しく機能しているように見えた。** 14組全てQCゲートを
  通過し(境界帯マスクを除いた領域での特徴点マッチングは妥当)、pre/post間で境界線・基板の
  溝もほぼ同じ位置に重なっている。ただし後述の通り、境界線・溝のごく近傍では数px単位の
  残差ずれが残っている。

## 2. exceeds_rateとの対応関係

`phase2_reanalysis_fov_exceeds_rate.csv`で、Blankのposition 1(0.8031%)・5(0.8066%)・
8(0.7820%)が、他のBlank(0.11-0.34%)およびSample1全体(0.02-0.42%)より明確に高いことを
確認した。格子点ごとのdelta( post_contrast - pre_contrast )の空間分布を元画像に重ねて
可視化した([`s0_pos1_exceeds_map.png`](../data/results/v9_phase2_production_reanalysis/260825_dna_visual_audit/s0_pos1_exceeds_map.png)等)ところ、超過点は目視確認できた欠陥とは対応せず、
以下2種類の要因に対応していた。

1. **マスクの境界フリンジ**: `register_image_pair_affine`が特徴点マッチングから除外する
   境界帯マスク(`bright_band_mask`)の縁のすぐ外側に超過点が薄く並ぶ(position 5・8、
   Sample1 position 2)。**現在のパイプラインは登録時の除外マスクをコントラストサンプリング
   (`sample_contrast`)には渡していない**ため、境界帯の明部/暗部が及ぼす光学的な裾野の
   影響を受けた格子点がそのまま統計に入っている。
2. **`bright_band_mask`が検出しない、より暗い溝**: Blank position 1では、超過点の大半が
   境界帯マスクの縁ではなく、フレーム内側(x≈80px付近)を上下に貫く1本の線に沿って分布して
   いた。この線をpre/postそれぞれフルサイズで切り出すと([`s0_pos1_crop_x80y1600.png`](../data/results/v9_phase2_production_reanalysis/260825_dna_visual_audit/s0_pos1_crop_x80y1600.png))、
   基板の物理的な溝(`detect_grooves`が landmark として使う基準構造)であり、Phase 2の
   グローバルアフィン(局所補正なし)では、この急峻な輝度勾配の直近でわずかな残差ずれが
   増幅されて大きなdelta値になっていることが確認できた。この溝は`bright_band_mask`の
   輝度閾値(55000 DN)より暗く、自動検出に一切かからない。

**すなわちBlank position 1・5・8のexceeds_rate異常は、目視確認できる意味での「欠陥」ではなく、
書き込みフィールド境界・基板溝という既知の物理構造の近傍で、現行の位置合わせ
(グローバルアフィン・局所補正なし)とマスク運用(登録時のみ除外・サンプリング時は除外なし)が
組み合わさって生じる技術的アーティファクトである可能性が高い。**

## 3. 洗浄手順・処理順の確認

`260825_p50_dnaハイブリ.md`(実験ログ)には、260828-p50-SAMの記録
([`260907_p50実験_SAM_DNA濃度依存性_ミスマッチ識別_結果.md`](file://G:/マイドライブ/Obsidian%20Vault/ラボノート/06_解析/260907_p50実験_SAM_DNA濃度依存性_ミスマッチ識別_結果.md)、
「洗浄がSample1→8へ順番・同一ビーカーで行われており、濃度順と処理順が完全に交絡している」)
のような**洗浄順序・処理順に関する記載は一切ない。** 欠陥観察表にもSample0・1(本検定の対象)
についての記載はなく、日程全体の「気づき」として「シミ・ゴミが多い」との記述のみ。

TIFFファイルのファイルシステムmtimeを確認したところ、Sample0・1のpre(`-0`)・post(`-1`)
画像は2026-08-25 20:32〜20:37の約5分間に連続して記録されていたが、**post(`-1`)画像群
(20:32-20:35)がpre(`-0`)画像群(20:35-20:37)より先にタイムスタンプが付いており、
撮影の前後関係と矛盾する。** これはファイルのGoogle Drive同期・転送時刻を反映している
可能性が高く、実際の顕微鏡撮影・処理順序を示す信頼できる根拠にならないと判断した。
TIFF内にはEXIF DateTimeタグも埋め込まれていない。

**結論: 260825_DNAについて、洗浄順序と濃度順の交絡が存在したかどうかは、現在参照可能な
記録からは確認も否定もできない。** 260828-SAMのような明示的な記録は見つからず、
`digital-plasmonic-counting`側の記録(260907ノートが参照している外部ログ)には
今回アクセスしていない。

## 4. 異常視野を除外した場合のMann-Whitney再計算

exceeds_rateが突出していたBlank position 1・5・8(2節で技術的アーティファクトの証拠が
見つかった3視野)を段階的に除外して、FOV単位exact Mann-Whitney(閾値超過率)を再計算した
([`sensitivity_drop_blank_fov_mannwhitney.csv`](../data/results/v9_phase2_production_reanalysis/260825_dna_visual_audit/sensitivity_drop_blank_fov_mannwhitney.csv))。

| シナリオ | n(Sample1) | n(Blank) | 両側exact p | 片側(Sample1>Blank)exact p |
| --- | ---: | ---: | ---: | ---: |
| 元の全件 | 7 | 7 | 0.053030 | 0.981061 |
| Blank position 1除外 | 7 | 6 | 0.101399 | 0.963287 |
| Blank position 8除外 | 7 | 6 | 0.101399 | 0.963287 |
| Blank position 1・8除外 | 7 | 5 | 0.202020 | 0.925505 |
| Blank position 1・5除外 | 7 | 5 | 0.202020 | 0.925505 |
| Blank position 1・5・8除外 | 7 | 4 | 0.412121 | 0.842424 |

境界/溝アーティファクトが疑われる3視野を除くと、p値は0.053→0.412まで単調に上昇し、
「Blank>Sample1」という方向性の根拠(片側p)も0.981→0.842へ大きく弱まった(有意ではない点は
どちらも変わらない)。さらに、境界帯マスクをコントラストサンプリングにも適用する形で
14組全体を再計算したところ(`wide_mask_recompute_fov.csv`、`bright_band_mask`を25px膨張して
`sample_contrast`にも渡す)、両側p=0.053→0.209、片側p=0.981→0.917となり、**マスク運用の
ギャップを埋めるだけでも有意境界から明確に離れる方向に動いた。**

## 5. 最終判断

- **元の境界近傍・逆方向という結果は、技術的アーティファクトに由来する可能性が高いと判断する。**
  根拠: (a) 目視確認で意味のある試料由来の欠陥は見つからなかった、(b) exceeds_rate異常は
  具体的に特定可能な2種類のパイプライン上の原因(登録時マスクがコントラストサンプリングに
  引き継がれていない、輝度閾値が拾わない暗い基板溝の近傍で残差登録誤差が増幅される)と
  空間的に対応した、(c) その原因に該当する視野だけを除く、またはマスク運用を修正するだけで、
  p値・方向性の根拠の両方が有意境界から明確に離れた。
- ただし、**「技術的アーティファクトである」と断定はできない。** 洗浄順・処理順の交絡
  ([`260907_p50実験_SAM_DNA濃度依存性_ミスマッチ識別_結果.md`](file://G:/マイドライブ/Obsidian%20Vault/ラボノート/06_解析/260907_p50実験_SAM_DNA濃度依存性_ミスマッチ識別_結果.md)が
  260828-SAMで報告したような)については確認も否定もできておらず、アーティファクトを
  除去した後もBlank position 1・5・8以外のFOVでは依然としてBlankがやや高めの傾向自体は
  残っている(n=4まで減らした後もp_one_sided_greater=0.842であり、0.5からは離れている)。
- **推奨: 追加実験は必須ではないが、以下2点のパイプライン改善を推奨する。**
  1. `sample_contrast`(コントラストサンプリング)にも、登録で使った`exclude_mask`
     (またはそれと同等の境界帯マスク)を`invalid_mask`として渡し、境界帯フリンジ由来の
     超過点を系統的に除外する。
  2. `bright_band_mask`の輝度閾値だけでは検出できない、より暗い基板溝(`detect_grooves`が
     landmarkとして使う構造)についても、格子点除外の対象にするか、その近傍での局所補正
     (`local_refinement`)を検討する(現行のPhase 2エントリポイントは`local_refinement`を
     使わないグローバルアフィンのみ)。
  この2点を実装した上で260825_DNAを再計算すれば、より確定的な判断ができる。それまでは、
  既存の慎重な解釈(「素朴な濃度依存性の再現ではなく、逆方向で境界近傍、要フォロー
  アップ」)を維持する。

## 保存先

- 本リポジトリ: `data/results/v9_phase2_production_reanalysis/260825_dna_visual_audit/`
  (Git管理外。目視確認用サムネイル14組、超過点の空間マップ4枚、溝のクロップ画像、
  `image_qc_metrics.csv`、`wide_mask_recompute_fov.csv`、`sensitivity_drop_blank_fov_mannwhitney.csv`)
- Google Drive / Obsidian: `G:\マイドライブ\Obsidian Vault\ラボノート\06_解析\260918_260825_DNA境界近傍p値_人手検証\`
  (同一ファイル群のコピー、本文書も含む)
