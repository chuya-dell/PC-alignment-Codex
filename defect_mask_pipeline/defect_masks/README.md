# 人手承認付きピクセル欠陥マスク

座標の基準は常にpre画像のネイティブ画素座標（左上原点、x右向き、y下向き）である。格子ピッチは `7.286 px` を使用するが、マスク座標そのものは格子ピッチから独立している。

## 安全条件

- 一次記録は欠陥の下限であり、未記載は欠陥なしを意味しない。
- 自動処理は周期成分を再構成した残差から、単一ピラーではなく連結領域だけを**候補**として出す。候補は欠陥の判定ではない。
- 全pre/post画像ペアに `full_view_review` 行を作る。一次記録も自動候補もない画像を、自動的にクリーン扱いしない。
- 一次記録がある視野には、候補の有無に関係なく `primary_record` 行を作る。
- 解析に読み込めるのは、pre/post双方がレビュー済みで、少なくとも片方に欠陥を確認し、承認者・承認日・pre座標ポリゴンが埋まった `approved` 行だけである。
- 承認ポリゴンはpost側で独立に指定しない。postへの適用時は確立済みのpre→post逆変換で追随させる。postだけで見つかった候補は逆向き変換でpre座標へ保存する。

## 実行順

```powershell
python calibrate_defect_candidates.py --control-dir 'G:\...\260830_P50_条件a~d' --output <output> 
python prepare_defect_review.py --manifest defect_masks/target_manifest.json --primary-records defect_masks/primary_defect_records.csv --output <review-output> --cache <cache>
```

`candidate_review.csv`をレビューする。`rejected` もpre/post双方の確認欄を埋める。手動で見つけた領域、または一次記録の位置は、pre画像上の頂点を `mask_polygon_pre_json` に `[[x,y], ...]` で記入して `approved` とする。承認済み領域の格子点が30%を超える視野は、部分利用せず視野全体を除外する。

```powershell
python run_pixel_mask_comparison.py --manifest defect_masks/target_manifest.json --primary-records defect_masks/primary_defect_records.csv --review <review-output>/candidate_review.csv --output <comparison-output> --cache <cache>
```

この最終コマンドは、全画像の全視野レビューまたは一次記録行のレビューが未完なら停止する。出力は有効視野数・有効ピラー数の比較だけであり、既存のp値を上書きしない。
