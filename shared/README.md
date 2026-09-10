# shared

`shared/` には、pillar_level と field_level のどちらにも固有ではない処理を置く。

- `registration.py`: pre/post幾何変換
- `lattice_indexing.py`: 六方格子の座標生成
- `theoretical_grid_evaluation.py`: 格子点の画像サンプリング
- `defect_masking.py`: 承認済みpre座標マスクの検証、post追随、3×3重なり判定
- `analyzer.py`: 登録用ピーク座標の抽出
- `qc_filter_v2.py`: 旧処理の安全側互換層。座標なしプレースホルダーや未完了レビューは拒否する

一次記録未記載をクリーン扱いしていた旧処理の監査は `LEGACY_RECORD_ABSENCE_AUDIT.md` を参照する。
