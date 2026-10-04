# ai_relay

Claude / Codex / Gemini の相互交代・相互検証と、使用上限で止まった側の自動再開。研究解析ではなく運用ツールのため `tools/` に置く。

## relay.py(交代・相互検証)
- `python tools/ai_relay/relay.py ask "質問" --first codex` : 呼んだ側が使用上限または異常終了なら、次の担当(claude → codex → gemini)に回す。
- `python tools/ai_relay/relay.py discuss "質問" --rounds 2` : 回答→相手が検証→反論、最大3往復。
- Gemini は `agy -p`(読み取りのみ。`--write` は拒否)。
- 既定は読み取りのみ。`--write` で編集を許可(claude / codex のみ)。
- 記録は `data/results/ai_relay/board_日付.md`(Git対象外)。
- テスト: `cd tools/ai_relay && python -m unittest test_relay test_codex_resume`

## 自動再開(止まった側をタスクスケジューラが起こす)
どちらの CLI からも独立して動くので、片方が止まっていても、両方止まっていても機能する。無人再開は読み取りのみ。
- `codex_resume.py`: Codex が `usage_limit_exceeded` で止まったセッションを、解除時刻後に `codex exec resume` で再開。Codex 0.160.0 の実ログで文言・形式を確認済み、実機の再開も確認済み。
- `claude_resume.py`: Claude Code 版(`claude -p --resume`)。テストは `claude_resume_tests/`。Claude の実際の制限文言は未観測。
- 登録例(5分毎): `pythonw.exe <path>\codex_resume.py run` をタスクスケジューラに。状態は `~/.codex/codex-resume/`、Claude 版は `~/.claude/claude-resume/`。
- 確認: `python codex_resume.py check`(副作用なし) / `status`。

## 既知の問題
- Claude と Gemini の使用上限メッセージの実文言は未観測。`LIMIT_RE` は広めの一致。初回の実上限で要確認。
- 認証切れ(`claude auth login` が必要)は上限ではないため、交代・再開の対象外。
