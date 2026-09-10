# Canonical geometry snapshot — 2026-09-10

- 元タスクID: 未取得（この作業環境では元のデスクトップ・タスクIDが公開されていない）。関連する引継ぎタスクID: `01a07e6e-d595-7041-b8ab-27e199a93599`。
- 元ファイルパス: `C:\Users\chuya\Documents\Codex\2026-09-07\pc-alignment-anti-codex-pipeline-rebuild\work\repo\{registration,analyzer,lattice_indexing,theoretical_grid_evaluation}.py`。
- 実装内容: `registration.py` は正本の粗い溝ランドマーク、格子向き、動的距離閾値 ICP、および大域二次多項式変換を含む。`analyzer.py` は幾何推定用の peak 抽出、`lattice_indexing.py` は pre 画像 FFT 由来の理論格子、`theoretical_grid_evaluation.py` は理論格子点の直接強度・コントラストサンプリングを担う。
- 入力: pre/post TIFF と、幾何推定の補助としての pre/post peak 点群。直接サンプリングの評価格子は **pre TIFF のみ** から生成する。peak 検出結果は評価する格子点を選別しない。
- 変換方向: `align_and_match_dataframes(pre_points, post_points, pre_image, post_image)` の最終変換は post/tgt 座標を pre/ref 座標へ写す。したがって pre 基準格子上の点を post で採取するときは `inverse_transform_points(pre_grid, transform)` を使う。
- 検証条件・結果概要: 代表 DNA 1-1 / SAM 1-1 で、local refinement を使わない正本経路の独立 hold-out 幾何残差中央値は報告済みの 0.022–0.035 px（半ピッチ 3.643 px 未満）。同じ経路で、ピークの最近傍一致率は検出再現性に依存し 70% 基準を満たさなかった一方、理論格子への直接サンプリングでは pre/post コントラスト相関が DNA 0.920、SAM 0.825 と報告された。
- 未解決事項: ICP の自己評価は周期格子の 1 ピッチ・エイリアシングを単独では排除しない。格子定数・位相の局所変動、手動欠陥アノテーションの完全性、濃度系列ごとのブランク定義は別途 QC が必要。高密度 peak 点群は変換推定の補助であり、ピラー存在判定の正本ではない。
- 新旧関係: **unknown**。作成日時・ハッシュ差だけでは既存 `shared/` の版より新しい／古い、または置換対象であるとは判定していない。そのため既存 `shared/registration.py` 等は変更していない。

## 内容ハッシュ（SHA-256）

| File | SHA-256 |
| --- | --- |
| registration.py | `61EE088B50EB158C28E7F998F8DA0BE0CD26EE86DC21FC9DE002A8879FED2540` |
| analyzer.py | `B5F28B8E9EC1031CAF0A17396C9BBFB04335BFBE24CAC40EF1CDDE30BC9CBA0B` |
| lattice_indexing.py | `E55191EF8E7218E6AADD0E39E85BE8ECA00EC5121C6ECCBB7DBF71F2E74B6803` |
| theoretical_grid_evaluation.py | `7E17CED46B1D77CEAECF59FB3F47185ACCCE08233CC94728AF18674338D3DEDF` |
