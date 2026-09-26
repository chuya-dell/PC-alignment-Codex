# 開発環境の準備記録（2026年9月26日）

## 配置

今回の環境は `C:\Users\labuser` の領域であり、Windowsの利用者一覧に `fdtdremote` は存在しなかった。したがって、これは現在の環境に作成した準備用クローンであり、fdtdremoteでの設定完了を意味しない。

- 準備用リポジトリ: `C:\Users\labuser\Documents\Codex\2026-09-26\gikt\outputs\PC-alignment-Codex`
- 既存の並行作業用クローン: `C:\Users\labuser\dev\PC-alignment-Codex`。未コミットの変更があるため、変更していない。
- 生データ: `F:\4.生データD_remo`。利用者が指定したドライブ上で実在を確認した。
- 生成物: 準備用リポジトリの `data/results/`。
- 仮想環境: 準備用リポジトリの `.venv/`。

Gitは導入済みの `C:\Program Files\Git\cmd\git.exe` を使用した。Pythonは利用可能だった付属のPython 3.12.14を基に独立した仮想環境を作成した。仮想環境の基になるPythonは `C:\Users\labuser\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe` にある。付属環境の削除や変更時は仮想環境を再作成する必要がある。

リモートからクローンし、作業開始前に `git pull --ff-only` で最新であることを確認した。取得時のコミットは `73b7990bda070caf3dfb6e107211acec33c08510`。依存パッケージは元の `requirements.txt` に基づいて導入する。

## 入力の設定

元の `data/raw/target_manifest.json` は外部ストレージ上のパスを含むため、そのまま保持した。ローカル用の `data/raw/target_manifest.local.json` はパスの先頭だけを `F:/4.生データD_remo` に置き換えたもので、6つの参照先フォルダの実在を確認した。実行時にはこの台帳を明示的に選ぶ必要がある。解析プログラム全体の既定値は変更していない。

機器固有の設定は `data/raw/local_environment.json` に保存した。これら2つのローカル設定と `.venv/` は、このクローンの `.git/info/exclude` で追跡対象から除外した。

## 確認手順

準備用リポジトリを作業フォルダにして、PowerShellで実行する。

```powershell
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' shared/environment_check.py
```

確認スクリプトは必要なパッケージを読み込み、既存の画像読込関数で実験画像を開き、結果用の一時フォルダで保存と再読み込みの一致を確認する。生画像は検証前後の内容の指紋を比較する。スペクトルファイルは全内容を読み込めることだけを確認し、その形式の解釈や解析は行わない。報告先は `data/results/environment_check/report.json`。

これは環境の動作確認であり、位置合わせの精度や研究結果の再現性を保証する試験ではない。生データへの書き込み試験は行わず、出力先でのみ書き込みを確認する。

2026年9月26日に実行して成功した。依存関係の整合性確認に問題はなく、必要な8パッケージの読込が成功した。実験画像 `260826-p50-sam/1-1-0.tif` は縦2044画素・横2048画素として読み込めた。出力の保存と再読込が一致し、生画像の内容の指紋も前後で一致した。スペクトルファイル `260625 - spe/0-1-0.spf2` は35,316バイトを読み取れた。実験フォルダ6件の実在も確認できた。

導入済みパッケージの完全な版一覧は `docs/environment-requirements-20260926.txt` に保存した。元の依存関係一覧は変更していない。

## 今後の運用とClaude Codeへの引き継ぎ

- 応答は標準語とし、文章中の英語の頭字語は正式名称で書く。識別に必要なファイル名・コマンドは原表記を保つ。
- 文献引用は指定された著者・誌名・年・巻・ページの順にする。
- 質問前に既存の記録を調べ、同じ質問が続いた場合は別のファイル・フォルダも調べる。
- 作業前に変更状態とブランチを確認し、未保存の作業を上書きせず `git pull --ff-only` を行う。競合時は強制更新しない。並行作業中の変更範囲を確認する。
- 測定用のソフトウェア、ドライバ、設定、生データを変更しない。
- 外部ストレージへの移動・同期は行わない。Obsidianへの記録はlabuser側のClaude Codeが担当する。この記録を引き継ぎに使用できる。
- Antigravityの凍結済みリポジトリのコードは参照・実行しない。既存文書と相違する場合は今回の利用者の指示を優先する。
- fdtdremote側へ移る場合は、その利用者でリポジトリを別途クローンし、Pythonと仮想環境を新規に用意する。今回の仮想環境をコピーして使わない。生データへのアクセスと動作確認もその利用者で再実施する。

研究解析は実行していない。既存の測定環境、既存クローン、凍結済みの別リポジトリ、Obsidianには変更を加えていない。

## Gitの認証設定

利用者の追加指示に基づき、準備用クローン専用のEd25519認証鍵を新規生成した。秘密鍵は `.git/codex-auth/id_ed25519` に置き、アクセス権を作成したWindows利用者とWindowsシステムに限定した。秘密鍵はGitで追跡されない。このクローンのフォルダ全体を第三者へ渡してはならない。

公開鍵は次のとおり。GitHubのchuya-dellアカウントへの登録は本人が実施し、2026年9月26日に専用鍵による同アカウントへの認証成功を確認した。

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFW5S4cHrBdLfRtQp/j3TEZYCZZGi2pInpqqXqdp3FCL fdtdremote-codex-preparation
```

作成者名を `白石忠弥 (Codex)`、メールアドレスを既存クローンに設定されていた `chuya2816@gmail.com` とした。アカウントが指定と異なるため、このクローンの設定だけに適用し、利用者全体の設定は変更していない。メールアドレスのGitHub上での確認済み状態は調べていない。

取得先と送信先は `git@github.com:chuya-dell/PC-alignment-Codex.git`。専用鍵を明示して使用する。接続先の公開鍵はGitHub公式の[接続先公開鍵一覧](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)で確認し、専用の `.git/codex-auth/known_hosts` に保存した。fdtdremote側の鍵と利用者全体の設定は、実際にその利用者で作業できる環境で改めて用意する。

作業開始時と送信直前に `git pull --ff-only` を実行する。履歴が分岐した場合は自動で上書きせず、内容を確認して競合を解消する。作業後は `git status` と差分を確認し、意味のある単位でコミットする。並行作業者には変更予定の範囲を事前に残す。
