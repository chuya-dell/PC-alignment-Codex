# ai_relay

Claude と Codex の相互交代・相互検証。研究解析ではなく運用ツールのため `tools/` に置く。

- `python tools/ai_relay/relay.py ask "質問" --first codex` : 呼んだ側が使用上限なら、もう片方に回す。
- `python tools/ai_relay/relay.py discuss "質問" --rounds 2` : 回答→相手が検証→反論、最大3往復。
- 既定は読み取りのみ。`--write` で編集を許可。
- 記録は `data/results/ai_relay/board_日付.md`(Git対象外)。
- テスト: `cd tools/ai_relay && python -m unittest test_relay`

## 既知の問題
- Claude 側の使用上限メッセージの実文言は未観測。`LIMIT_RE` は広めの一致。初回の実上限で要確認。
- 認証切れ(`claude auth login` が必要)は上限ではないため交代せずエラーで止まる。
- 再開時刻まで待って元の側へ戻す処理はない(手動で再実行)。
