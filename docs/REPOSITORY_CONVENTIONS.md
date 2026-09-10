# リポジトリ整理規約

## 作業開始時の手順

1. `README.md`、`AGENTS.md`、このファイルを読む。
2. 作業がピラー単位か視野単位かを先に決める。
3. 既存の該当バージョンを確認する。既存処理・パラメータを変えるなら、旧版を変更せず次の `vN_説明` を作る。
4. 入力は `data/raw/`、出力は `data/results/` を使う。コードとデータを同じ場所に置かない。
5. 作業後、該当 `NOTES.md` の「実装内容」「結果概要」「既知の問題」を更新する。
6. 他チャット由来のものは、作成日時だけで既存コードを上書きしない。まず `data/raw/artifact_provenance_registry.csv` に登録し、比較状態を `unknown` として独立保存する。

## 配置の判断表

| 主な解析単位 | 配置先 | 実行ファイル名の例 |
|---|---|---|
| 個々のピラーの座標、強度、コントラスト、マスク重なり | `pillar_level/vN_説明/` | `pillar_extract_contrast.py` |
| 視野ごとの有効数、平均、群比較、除外割合 | `field_level/vN_説明/` | `field_compare_counts.py` |
| 登録変換、格子生成、画像読込、両方で使う検証 | `shared/` | `registration.py` |
| 入力台帳・マニフェスト | `data/raw/` | `primary_defect_records.csv` |
| 実行で生成したCSV、画像、監査ログ、キャッシュ | `data/results/` | `count_comparison/` |

迷う場合は `shared/` に置き、`shared/README.md` へ「両粒度に共通である理由」を1文追記する。

## バージョンフォルダの最小構成

```text
pillar_level/
  v2_短い説明/
    pillar_処理名.py
    NOTES.md
```

`NOTES.md` は必ず次の3項目を含める。

1. 実装内容・旧版からの変更点
2. 結果概要と `data/results/` 内の保存先
3. 既知の問題・未解決事項

## 他実装との関係

- このリポジトリはCodex担当分である。
- Claude担当分は `PC-alignment-claude`、Antigravity担当分は `PC-alignment-anti` に置く。
- 3手法の比較は将来の別タスクとして実施する。現時点で各リポジトリのコード・結果を混在させない。

## 他チャット成果の取り込み

- 取り込み時には、元のCodexタスクID、元パス、元タスクの更新日時、解析粒度、要約、Git上の保存先を `data/raw/artifact_provenance_registry.csv` に残す。
- `newness_status=unknown` は「旧版」でも「最新版」でもなく、比較未実施を意味する。日付だけで変更順序や優劣を決めない。
- 比較を行う場合は、入力データ、パラメータ、依存コード、検証結果を並べてから、明示的に `supersedes_artifact_id` を記録する。比較前の成果は削除・上書きしない。
- 小さな再現用CSVは `data/results/reference_snapshots/` に追跡可能な形で置いてよい。ただし実験画像、生成画像、大容量出力、キャッシュはGitへ追加しない。

## 現行の安全境界

ピクセル欠陥マスクでは、pre画像ネイティブ画素座標を基準にする。自動処理の出力は候補であり、pre/post双方の目視確認、承認者、承認日、pre座標ポリゴンを満たしたものだけを解析に使う。一次記録の未記載はクリーン判定ではない。
