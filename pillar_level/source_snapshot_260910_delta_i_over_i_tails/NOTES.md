# 260828 p50 SAM/DNA ピラー単位Delta I/I分布比較

## 出所

- 元タスクID: `01a07fd2-2046-7870-beef-09f73fe3177b`
- 元ファイルパス:
  - `G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM`
  - `G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-dna`
  - `C:\Users\chuya\Documents\Codex\2026-09-08\1-8-260830-control-260828-s8\work\analyze_spatial.py`
  - `data/raw/primary_defect_records.csv`

## 配置判断

視野をまたいで非集約の個々のピラーDelta I/Iをプールし、その分布上位裾を主対象にするため、`pillar_level`へ配置した。

## 実装内容

- 中央1000 pxテンプレートの粗平行移動を初期値に、0.25x、0.5x、1.0xのアフィンECCを実行する。
- OpenCV ECCの変換方向に合わせ、post画像へ`WARP_INVERSE_MAP`を指定してpre座標系へ写像する。
- pre画像のFFT一次リングからピラー座標を抽出し、3x3積算値から`100 × (pre-post)/pre`を計算する。
- 一次欠陥記録で欠陥が明示された視野を除外する。未承認の画像由来自動欠陥候補は正式マスクとして使用しない。
- SAM Sample1〜8、DNA Sample1,2,9,4,10,7,8について、中央値、IQR、P99、P99.9、P99.99、Blank平均+3SD超過率をサンプル単位で算出する。
- 非Blank log10濃度、Blankを含む濃度順位、post TIFF実測時刻の処理順位に対するSpearman相関を計算する。推論上のnはサンプル数で、視野・ピラーを反復数として扱わない。

## 結果概要

実行成果はGit管理外の`data/results/260828_p50_delta_i_over_i_tails/`、ユーザー向け成果は元タスクの`outputs/`に保存する。Gitには小さなサンプル別要約CSVだけを`data/results/reference_snapshots/`へ保存する。

## 未解決事項

- 欠陥記録のない視野を完全にcleanと確定する目視完了台帳はない。一次記録に明示された欠陥視野を除外したスナップショットである。
- TIFFのLastWriteTimeを実測時刻の代理にするため、取得ソフトによる時刻書換えの影響を否定できない。
- 濃度と処理順が同一実験内で交絡しており、観察された傾向を濃度効果と時間ドリフトに識別できない。

## 新旧関係

`newness_status=unknown`。作成日時だけでは既存実装との新旧・優劣を判定していない。既存ファイルは上書きしていない。
