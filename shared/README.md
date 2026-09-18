# shared

`shared/` には、pillar_level と field_level のどちらにも固有ではない処理を置く。

- `registration.py`: pre/post幾何変換
- `lattice_indexing.py`: 六方格子の座標生成
- `theoretical_grid_evaluation.py`: 格子点の画像サンプリング
- `defect_masking.py`: 承認済みpre座標マスクの検証、post追随、3×3重なり判定
- `analyzer.py`: 登録用ピーク座標の抽出
- `qc_filter_v2.py`: 旧処理の安全側互換層。座標なしプレースホルダーや未完了レビューは拒否する
- `image_qc.py`: pillar/field どちらにも依存しない生画像QC。Position 6・7で見つかった
  基板書き込みフィールド境界の明部帯を検出する`bright_band_mask`と、撮影直後の
  輝度統計から同じ問題を自動フラグする`saturation_qc`。詳細は
  `docs/POSITION6_IMAGE_FORENSICS_20260916.md`と`docs/MASKED_ALIGNMENT_FINAL_DECISION_20260916.md`
  を参照。`registration.py`の`register_image_pair_affine`から自動的に呼ばれる。
- `registration.py`: Phase 2の`register_image_pair_affine`は、推定アフィンを下流へ渡す前に
  `assess_affine_transform_qc`で物理的妥当性を検査する。これはピラー・視野のどちらにも
  共通する登録安全境界なので`shared/`に置く。既定では中心変位125 px、回転3度、
  等方スケール0.93--1.05、異方性1.10を超える変換を例外として拒否する。
- `concentration_series_stats.py`: Blank閾値（プールしたBlank格子点コントラスト差分の
  `mean - 3*std`）とFOV単位exact Mann-Whitney検定。`PC-alignment-anti`（参照のみ、
  変更なし）の`investigation/260907_pipeline_rebuild/scripts/run_fov_level_direct_sampling.py`
  を読み取って書き直した。`field_level/v9_phase2_production_reanalysis`から使う。

一次記録未記載をクリーン扱いしていた旧処理の監査は `LEGACY_RECORD_ABSENCE_AUDIT.md` を参照する。
