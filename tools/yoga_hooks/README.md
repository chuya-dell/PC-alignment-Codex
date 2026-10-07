# yoga_hooks

Yoga（W: ドライブの機械）専用の Claude Code フック。他の機械の設定には使わない。

- `save_session_yoga.py`：会話保存フック。元の `Claude/セッション/save_session.py`（Googleドライブ上の共有ファイル。変更していない）の保存先だけ `W:\GoogleDrive\chuya2816\AI会話ログ` に変えた。W: の同期フォルダがなければ何もせず正常終了する
- 元に戻す：`C:\Users\chuya\.claude\settings.json` の `hooks.Stop` と `hooks.SessionEnd` の `save_session_yoga.py` を、元の `python "G:\マイドライブ\Claude\セッション\save_session.py"` に戻す（G: がある機械ではこちらが正）
