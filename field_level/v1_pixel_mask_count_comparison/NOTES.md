# v1_pixel_mask_count_comparison — 視野単位

## 実装内容

- 旧方式で採用された視野数・有効ピラー数と、承認済みピクセルマスク適用後の値を比較する。
- マスクと3×3サンプリング領域が一部でも重なるピラーを無効にする。
- マスクにかかる格子点の割合が30%を超える視野は、部分利用せず視野全体を除外する。
- 全視野レビューと一次記録レビューが完了していない場合は停止する。

## 実行

```powershell
python field_level/v1_pixel_mask_count_comparison/field_pixel_mask_count_comparison.py --manifest data/raw/target_manifest.json --primary-records data/raw/primary_defect_records.csv --review data/results/review_candidates/candidate_review.csv --output data/results/count_comparison --cache data/results/cache
```

## 結果概要

承認プロセス前のため、確定比較表はまだ生成していない。承認後の出力先は `data/results/count_comparison/` である。

## 既知の問題・未解決事項

- 本バージョンは有効数の比較だけを行う。既存のp値を上書き・再確定せず、統計的再検定は別バージョン・別タスクとする。
- 30%は暫定閾値であり、実データの承認マスク面積を確認して将来のバージョンで見直す。
