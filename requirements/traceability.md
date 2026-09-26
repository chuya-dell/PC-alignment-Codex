# 要件・テスト対応表

要件バージョン: v1

| 要件ID | 内容概要 | 対応テスト | 状態 |
|---|---|---|---|
| R-01 | ピラーのグループ化ロジック | `pillar_level/v2_hex_group7/tests/test_hex_group7.py::test_r01_interior_boundary_overlap_rotation` ほか5件 | 実装済み(Round1でREVIEW指摘なし) |
| R-02 | グループ単位の統計値算出 | `test_r02_hand_calculated_field`, `test_missing_and_invalid_metrics_do_not_shrink_group` | 実装済み(Round1でREVIEW指摘なし) |
| R-03 | 既存の個別ピラー解析との共存(回帰) | `test_r03_existing_analysis_exact_before_after_and_output_protection` | 実装済み(Round1でREVIEW指摘なし) |
| R-04 | グループ定義の前提確認(ログ・コメント明記) | `test_r03_existing_analysis_exact_before_after_and_output_protection`, `test_command_line`(標準出力に前提を明記) | 実装済み(Round1でREVIEW指摘なし・本人未確認のまま) |

実装コミット: `641e5f5b38d6f7453a94e85706b304aa8ec3f7df`(`pillar_level/v2_hex_group7/`)
