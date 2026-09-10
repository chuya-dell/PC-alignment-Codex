# Cross-day direct-grid sampling snapshot — 2026-09-10

- 元タスクID: 未取得（この作業環境では元のデスクトップ・タスクIDが公開されていない）。関連する引継ぎタスクID: `01a07e6e-d595-7041-b8ab-27e199a93599`。
- 元ファイルパス: `C:\Users\chuya\Documents\Codex\2026-09-07\pc-alignment-anti-codex-pipeline-rebuild\work\run_crossday_direct_sampling.py`、`work\crossday_manifest.json`、`outputs\crossday_p50_260907\{crossday_sample1_blank_summary,crossday_fov_qc}.csv`。
- 実装内容: `field_crossday_direct_sampling.py` は視野単位で、pre 画像 FFT 格子の全有効点を直接サンプリングする。正本幾何を使って post を pre 基準へ逆写像し、`post_contrast - pre_contrast` を評価する。ピークは位置合わせ変換の推定補助のみで、強度評価点の選別には使わない。
- 入力: 日付ごとの pre/post TIFF（Git 管理外）、日別の sample-to-concentration 台帳、手動視野 QC により選んだ position、`field_crossday_manifest.json`。ファイル名は `sample-position-prepost.tif`（例 `1-3-0.tif`, `1-3-1.tif`）を前提とする。
- 変換方向: 共有スナップショットの `registration` 変換は post→pre。pre 格子点を post 座標へ戻して post コントラストを採取する。
- 検証条件: pitch は 7.286 px、格子余白は 30 px、幾何推定には `method=peak, min_dist=3, threshold=0.2` を使用するが、これは全ピラー検出設定として採用したものではない。`local_refinement=False`。主要指標はブランク平均−3σ 未満の割合と、独立視野平均による両側 exact Mann–Whitney 検定である。
- 結果概要: 添付の小型 CSV には 260824 SAM、260825 DNA、260826 SAM、260828 DNA、260829 SAM の集計と FOV QC を保存した。各比較で FOV レベルの Mann–Whitney p 値は 0.05 未満ではなかった。260826 SAM は blank FOV が 1 つのため、p=0.5 は記述的な値に限られる。
- 未解決事項: 260828 SAM Sample8 のブランク／1 fM 解釈に相反する記録があり、本スナップショットの manifest には含めない。260827 DNA と 260829 DNA は最高濃度側が除外されるため Sample1-vs-Blank 比較の対象外。手動欠陥は視野単位の暫定除外であり、ピラー単位アノテーションでの再評価が必要。
- 新旧関係: **unknown**。作成日時から他の解析・既存データの新旧や置換関係を推定していない。既存ファイルは上書きしていない。
