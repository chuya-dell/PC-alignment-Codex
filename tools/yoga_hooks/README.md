# yoga_hooks

Yoga（W: ドライブの機械）専用の Claude Code フック。他の機械の設定には使わない。

- `save_session_yoga.py`：会話保存フック。元の `Claude/セッション/save_session.py`（Googleドライブ上の共有ファイル。変更していない）の保存先だけ `W:\GoogleDrive\chuya2816\AI会話ログ` に変えた。W: の同期フォルダがなければ何もせず正常終了する
- 元に戻す：`C:\Users\chuya\.claude\settings.json` の `hooks.Stop` と `hooks.SessionEnd` の `save_session_yoga.py` を、元の `python "G:\マイドライブ\Claude\セッション\save_session.py"` に戻す（G: がある機械ではこちらが正）

## 置き場所（実際に動いているもの）
**実際に動いているのは `C:\Users\chuya\.claude\hooks\yoga_hooks\` の写し**（`C:\Users\chuya\.claude\settings.json` のフックと、起動器 `Invoke-ClaudeRound.ps1` がここを指す）。このリポジトリの `tools/yoga_hooks/` は、元のコードの保管場所（`main` が正）。作業コピーのブランチを切り替えても、動いている側は壊れない。

**直したら写しも更新する。写しの更新手順:**
1. このリポジトリで直して commit する（`main` に入れる）
2. 写しを上書きする（作業コピーが別のブランチのときは、`git show main:tools/yoga_hooks/save_session_yoga.py` で取り出して置く）:
   `Copy-Item C:\Users\chuya\PC-alignment-Codex\tools\yoga_hooks\save_session_yoga.py C:\Users\chuya\.claude\hooks\yoga_hooks\ -Force`
3. 写しを手で1回動かして、動くことを確かめる

試験用に、環境変数 `SAVE_SESSION_YOGA_DIR` で保存先を変えられる（その親フォルダが無ければ何もせず正常終了する）。
