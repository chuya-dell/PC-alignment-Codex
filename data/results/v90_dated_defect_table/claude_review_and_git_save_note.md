# Claude Code による検収・訂正・Git 保存の記録（事後の再評価、独立な検証ではない。2026-10-10）
## 実行の経緯
- v90 のコード(field_dated.py)は Codex(main)が書いた。実行中に Codex が利用枠切れ(5時間の枠、回復予定23:01)になったため、**Claude Code が同じコードを実行**した(約25分、python -B field_level/v90_dated_defect_table/field_dated.py)。un_started_*.json は、Codex の途中の試行と最後の実行の開始記録(失敗・中断の履歴として残す)。v91 は Codex(main)が最後まで実行した。
- 事前登録は v1(本文)→v2(Opus の評価後)→v3(Astra の評価後)の追記で、写しを preregistration_v1_v2_v3.md に入れた(SHA-256 512e2e03c76a4ecf4eab061b7d6e1bc29d6fe336b54b54aac56d1905b7435e4c)。Codex 依頼文の写し codex_request_v90_v91.md(SHA-256 d01823bf095b511f254921dfe8ad36f416f28b84f47a94eb8c4282612a01cfe6)。
## 検収で確かめたこと
- v88 の E1 の個数が再計算と完全一致(Codex の確認)。ブランクの E1＝142/56/18(k=6/8/10)は、Opus の独立再計算とも一致。
- Opus が M1_summary から E1・E2・E3u の合計、E3u＝E1 の全日一致、overall_iv を再計算して確認。
## 訂正(Opus の判定による。結果ノートは訂正済み)
1. 初稿の「減少はすべて 260926・27」は誤り。k=6 の減少55のうち45が後半2日、10が早い日(中心8が当日の表に入るため)。
2. 初稿の「差のすべては 260926 の1日」は過大。E4u と E3u の差19のうち17が 260926、2が 260828(中心8の出入り)。繰り越しの遅れは2回観測。
3. 失う有効ピラーの最大は、ブランクで0.56%(M1/M4)、pass_fail の0.69%は非ブランクを含む値。
4. v91 の暫定値との不一致の多くは、私が幅・振幅を視野の中央値で割った比で出していたための単位の違い。正本は v91 の生の値。
## Git に入れなかったファイル(サイズが大きい。SHA-256 は v90_manifest.csv / v91_manifest.csv に記録済み。シード固定でコードを再実行すれば再生成できる)
v90: M1_E5_all_k.csv, M1_E5_pooled_summary.csv, M1_E5_summary.csv, M3_E3h_fields.csv, M3_fields.csv, M4_E5_fields.csv
v91: control_circles.csv, local_fields.csv
閾値は 5 MB。Codex が書いた最終の Codex NOTES.md/report.md はそのまま保存した。