# round_guard

周回の最後に `ラボノート/03_やること` と `ラボノート/02_進捗` が更新されたかを、Claude Code の Stop フックで確かめる。

- `python round_guard.py start [--theme テーマ名]... [--record 周回記録のパス]`：周回の開始時に印（`~/.claude/round-guard/active-round.json`）を作る。起動器 `Invoke-ClaudeRound.ps1` は自動で `--record` つきで呼ぶ（テーマ名は手で渡す）
- `python round_guard.py end`：周回の終わりに印を消す。印がなければ検査は何もしない
- 検査：開始時刻より後に更新された 03・02 のファイルに、その周回の最終報告書のファイル名かテーマ名が入っていること。02 は、周回記録に「02_進捗は変更なし」があれば可
- 5回止めても直らなければ、周回記録に「未更新のまま終了」と書いて通す。3行報告の(3)にも同じことを書く

## Claude Code の設定を元に戻す手順
Stop フックは `C:\Users\chuya\.claude\settings.json` の `hooks.Stop[0].hooks[0]`（`round_guard.py check`）として入っている。

1. 設定ごと戻す：このフォルダの `claude_settings_before_round_guard_261008.json`（フック追加前の写し）を `C:\Users\chuya\.claude\settings.json` に上書きコピーする
2. フックだけ外す：`settings.json` の `hooks.Stop[0].hooks` から `round_guard.py check` の1件を消す
3. 一時的に止める：`python round_guard.py end` で印を消す（印がなければ検査は働かない）

## 起動器との連携（2026-10-08 確認済み）
`Invoke-ClaudeRound.ps1` は、開始時に `start --record <記録パス> --theme <テーマ名>` を呼び、終了時（finally）に `end` を呼ぶ。テーマ名は `-Theme` で渡せる。省略時は、プロンプトの「引き継ぎを読んで：<テーマ>」から取る（取れなければ最終報告書名だけで内容を確かめる）。

## 置き場所（実際に動いているもの）
**実際に動いているのは `C:\Users\chuya\.claude\hooks\round_guard\` の写し**（`C:\Users\chuya\.claude\settings.json` のフックと、起動器 `Invoke-ClaudeRound.ps1` がここを指す）。このリポジトリの `tools/round_guard/` は、元のコードの保管場所（`main` が正）。作業コピーのブランチを切り替えても、動いている側は壊れない。

**直したら写しも更新する。写しの更新手順:**
1. このリポジトリで直して commit する（`main` に入れる）
2. 写しを上書きする（作業コピーが別のブランチのときは、`git show main:tools/round_guard/round_guard.py` で取り出して置く）:
   `Copy-Item C:\Users\chuya\PC-alignment-Codex\tools\round_guard\round_guard.py C:\Users\chuya\.claude\hooks\round_guard\ -Force`
3. 写しを手で1回動かして、動くことを確かめる

## 履歴
- `v66_round_guard_20261008`：**止まらない版**（終了コード2で止める作りだったが、PowerShell 経由で終了コードが1になり、止めなかった）。使わない
- `v67_round_guard_verified_20261008`：JSON（decision=block）で止める版。内容検査・通すときの記録つき
- `v68_round_guard_launcher_verified_20261008`：起動器との連携（`-Theme`）を確認
- `v69_hooks_copy_20261008`：`C:\Users\chuya\.claude\hooks\` に写しを置き、`main` に取り込んだ版
