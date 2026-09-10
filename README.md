# PC-alignment-Codex

このリポジトリはCodex担当分。対応するClaude版は PC-alignment-claude、Antigravity版は PC-alignment-anti。3手法の比較はいずれ実施予定。

プラズモニック結晶を用いるデジタル単分子カウントの位置合わせ処理を、ピラー単位と視野単位で明確に分離して管理する。

## 構成と命名規則

- `pillar_level/`: ピラー1本ごとの座標、コントラスト、欠陥領域の解析。実行ファイルは `pillar_` で始める。
- `field_level/`: 視野を単位とする集計・比較。実行ファイルは `field_` で始める。
- `shared/`: 両粒度で共用する位置合わせ、格子、画像サンプリング、承認済みマスクの安全な読込み。粒度を持たないためここへ置く。
- `data/raw/`: 実験画像へのパスを示すマニフェストと、一次記録から構造化した入力台帳。画像本体は格納しない。
- `data/results/`: 実行で生成するレビュー画像・CSV・比較表の保存先。Gitには入れない。

各バージョンフォルダには必ず `NOTES.md` を置く。既存バージョンのコード・パラメータは上書きせず、変更は新しい `vN_説明/` を追加して行う。

他のPCで作業を再開する場合は、最初に [`AGENTS.md`](AGENTS.md) と [`docs/REPOSITORY_CONVENTIONS.md`](docs/REPOSITORY_CONVENTIONS.md) を読む。新規解析を置く前に、粒度とバージョンをここで決める。

## 現行バージョン

- [`pillar_level/v1_human_approved_pixel_masks/`](pillar_level/v1_human_approved_pixel_masks/): 周期残差を候補化し、目視承認されたpre画像座標ポリゴンだけをマスクにする処理。
- [`field_level/v1_pixel_mask_count_comparison/`](field_level/v1_pixel_mask_count_comparison/): 旧視野全体除外方式と承認済みピクセルマスク方式の有効数比較。p値の再計算はしない。
- [`field_level/source_snapshot_260910_raw_qc_brightness/`](field_level/source_snapshot_260910_raw_qc_brightness/): 他チャットから保存した全視野raw輝度差の補助QC。既存の格子点エンドポイントとは別物で、新旧関係は未判定。

自動候補は欠陥として採用されない。pre/post双方の目視確認、承認者、承認日、pre画像ネイティブ画素座標のポリゴンが揃った承認行だけが解析へ流れる。

## 他チャットからの保存

取り込み済み成果とその出所は [`data/raw/artifact_provenance_registry.csv`](data/raw/artifact_provenance_registry.csv) に記録する。作成日時は出所の手掛かりに留め、内容・パラメータ・検証結果の比較が済むまで、どちらが新しいかを判定しない。
