# 依頼：日付別欠陥表(iv)の開発用検証（v90）と、画像レベルの条件調査の再現スクリプト（v91）

最初に読む（UTF-8、日本語。PowerShell なら Get-Content -Encoding UTF8）：
1. 事前登録（正本は v3。v1本文＋v2改訂＋v3改訂の順に読み、後のものを優先する）：`W:\GoogleDrive\chuya2816\Obsidian Vault\_依頼記録\偽陽性ゼロ化_C1D\261010_日付別欠陥表(iv)_設計と合否基準_事前登録.md`
2. 調査ノート：`W:\GoogleDrive\chuya2816\Obsidian Vault\ラボノート\05_解析\261008_【未読】偽陽性ゼロ化_原因確定と標準経路\【未読】9月22〜26日前後の変化_操作履歴と撮影・解析条件の切り分け.md`
3. 既存コード・表：`field_level/v88_defect_exclusion/field_stage1.py`、`field_level/v89_defect_cause/field_cause.py`、`data/results/v88_defect_exclusion/`、`data/results/v89_defect_cause/`、欠陥表 `data/raw/v88_camera_defects_unapproved.csv`。配列は `C:\Users\chuya\PC-alignment-fp\data\results`（v70 fields2、v72 fields、v84 cache、v85 E4_counts_all632.csv）。

## v90：事前登録の欠陥表の作り方（腕 E0〜E8）と指標（M1〜M5）を、事前登録のとおりに実装して全部出す
- 腕（v3により、E7・E4・E4u を主の比較に格上げ。M1 は標準化の再計算版と固定版の2通り＋偽陽性の実数も出す。m(N) は2つの帰無の大きい方）：E0, E1, E2, E3, E3u(=E2∪E3), E4, E4u, E5(n=8,12,16,24,32,48 各200回、種固定), E7(累積), E3h(同日を半分に分けて表を作り、他方でM3), E8(探索的：画像指標の表。閾値は「環状対照との幅の差>0.02 かつ密度比<0.9 の円」を仮置きとして使い、その旨を明記。合否には使わない)。
- m(N)：事前登録の帰無（検出点の位置を同じ視野の有効ピラーの位置から再抽出）で決め、N ごとの値を `m_of_N.csv` に出す。
- 指標：M1(k=6,8,10 の fp91k、ブランク69視野1枚ごと・日付別・全体)、M2、M3(自分の表の内側評価と外側評価を分ける)、M4、M5(中心ごと)。
- 合否欄：事前登録の合否基準（v2）どおりに機械的に判定し、`pass_fail.csv` に出す。基準を満たさないときは満たさないと書く。基準は変更しない（異議は report.md の末尾に書く）。
- 出力：`data/results/v90_dated_defect_table/`（表、`report.md`、`NOTES.md`、`verification.json`、`v90_manifest.csv`）、コード `field_level/v90_dated_defect_table/field_dated.py`。

## v91：調査ノートの節3・4・5の再現スクリプト（探索的。Claude Code が一時スクリプトで出した値の再現と、補正）
- 全547非ブランク視野で、中心10画素以内のピラーと、環状（30〜60画素）の**同数のピラーを含む円**（複数）との差を、pre の pre_ctr と post の post_ctr で、幅(wid)・振幅(amp)・密度で計算。13中心すべて。視野単位のブートストラップ（日付内）で95%区間。
- 日付ごとの表と、境（8：260922|260923、157・309：260924|260926）の前後の表。安定9中心の同じ境での変化も。
- 日ごとのピラー振幅・幅・sigma、post/pre 振幅比（視野単位）、検出点の重心座標（中心ごと・日付ごと）。
- Claude Code の暫定値（調査ノート節4）と照合して、一致/不一致を report に書く（一致させにいかない。違えば違うと書く）。
- 出力：`data/results/v91_image_level_conditions/` と `field_level/v91_image_level_conditions/field_conditions.py`。

## 守ること
- 24視野（識別子が 261008 で始まるもの）の配列・画像と、`v83_new_shots_261008` や生データの `261008_*` フォルダは開かない（Vault の解析フォルダ `261008_【未読】偽陽性ゼロ化_…` は解析ノートの置き場で、読んでよい）。260925 の画像・配列も開かない（開発用の台帳にない）。`array_read_audit.csv` と assert を入れる。
- 評価対象の画像から欠陥表を作らない（ブランクは評価専用。表は非ブランクだけから作る）。
- 新しい2フォルダと新スクリプトだけに書く。既存ファイル・欠陥表(v88)・標準経路は変更しない。Git の commit はしない。
- 乱数の種と実行コマンド・実行時間・出力の SHA-256 を `*_manifest.csv` に出す。
- 事前登録に疑問があれば、基準どおり実行したうえで、report.md の末尾に「基準への異議」として書く（黙って変えない）。
