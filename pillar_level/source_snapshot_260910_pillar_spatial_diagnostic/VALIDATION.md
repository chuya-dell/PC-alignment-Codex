# 検証メモ

- Python AST構文検査: `AST_OK`。
- CSV 8ファイル: 全て70行、rank 1〜70、`median_centered_decrease_pct`降順、12列の期待スキーマ: `CSV_ALL_OK`。
- 元タスク側で8枚の空間重ね合わせ・ヒストグラム図を目視確認済み。PNGはGitへ追加していない。
- Git取込時に`git diff --check`を実施し、禁止対象（TIFF/PNG/NPZ/JSON）は今回のスナップショットに含めていない。
- 画像由来欠陥候補は人手承認済みマスクではなく、一次欠陥座標との正式な重ね合わせは未実施である。
