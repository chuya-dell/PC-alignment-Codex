# v37 未使用確認(Claude Code、2026-10-04 17:51 JST)

版番号 v37 は仮置きで、使う前に次を確認した。

- ローカルブランチ・リモート追跡ブランチ(`git branch -a --list '*v37*'`):該当なし
- タグ(`git tag --list 'v37*'`):該当なし。既存の最大は v36_band_origin_final_20261004
- リモートの heads(`git ls-remote --heads origin`):v37 を含むものなし
- フォルダ(リポジトリ全体の再帰検索 `v37*`):該当なし。`field_level/` の最大は v36_band_origin_final、`data/results/` の最大も v36_band_origin_final
- 結論:v37 は未使用。そのまま使う(番号の繰り上げなし)。

作業ブランチ:`claude/tracking-calibration-20261004`(`claude/band-origin-20261004` の先頭 7e4c181bbf3afee75b626f0f873df2c084317834 から分岐。分岐時点の作業ツリーは変更0件)。
