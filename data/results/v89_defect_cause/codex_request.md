# 依頼：C1+D 不再現4中心（id 8・40・157・309）の原因切り分け（開発用データだけ）

読むもの（UTF-8、日本語）：
- 事前登録（v1 本文 + 末尾 v2 改訂。これが判断基準の正本）：`W:\GoogleDrive\chuya2816\Obsidian Vault\_依頼記録\偽陽性ゼロ化_C1D\261010_不再現4中心_仮説と判断基準_事前登録.md`
- 既存コード・表：`field_level/v88_defect_exclusion/field_stage1.py`、`data/results/v88_defect_exclusion/`（centers_*.csv, center_reproducibility.csv, date_split.csv）、欠陥表 `data/raw/v88_camera_defects_unapproved.csv`
- 配列の場所：`C:\Users\chuya\PC-alignment-fp\data\results`（field_stage1.py の SRC と同じ。ファイル名・読み方も同じ）

やること：事前登録 v2 のとおりに、13中心すべて（不再現4＋再現9の対照）について解析し、`data/results/v89_defect_cause/` に結果表と短い報告書 `report.md` を作る。

必須の出力：
1. `field_table.csv`：547非ブランク視野×13中心の1行ごと。fid, date, board, pos, group, conc、機会(i)(ii)(iii)の近傍ピラー数、検出数、近傍の最大 |z5p|、欠けたピラー数（excl偽・z5p非有限）。
2. `date_center_table.csv`：日付×中心の機会視野数・検出視野数・率・Clopper-Pearson 95%CI（機会(i)主、(ii)(iii)も）。
3. `boundary_scan.csv`：13中心×隣接日の8境：両側の機会視野数・検出視野数・率・率比・両側Fisher p。判定欄に H1／部分的な減少／H2示唆／H3示唆／なし を、事前登録の基準どおりに機械的に付ける。日付を単位にした並べ替え検定の p も併記。
4. `control_check.csv`：再現9中心のうちH1基準を満たす数（2以上なら H1 判定不能と明記）。
5. `board_pos_within_date.csv`：検出のある日付ごとに板・pos と検出の Fisher 検定（p<0.05/13）。
6. `z5p_by_date.csv`：中心10画素以内の |z5p| の日付ごとの中央値・95%点。
7. `unadopted_clusters.csv`：4中心の位置から200画素以内にある、未採用の検出点群（前半・後半それぞれ。点数・視野数・最寄り距離）。
8. 感度（合否に使わない）：最小視野数5で前後半の両方で再現する中心数／8境それぞれで両側再現する中心数。
9. `report.md`：事前の予想（157・309は260922|260923の境、8は第1期間内）と結果の一致・不一致を、判定欄の数値つきで。事実（表から直接読めること）と解釈を分ける。判断基準は変更しない。基準に合わない中心は「原因未確定」と書く。

守ること：
- 24視野（識別子が 261008 で始まるもの）の配列・識別子は開かない。field_stage1.py と同じ ALLOWED の assert を入れ、読み取り履歴 `array_read_audit.csv` を出す。
- 新フォルダ `data/results/v89_defect_cause/` と新スクリプト `field_level/v89_defect_cause/field_cause.py` だけに書く。既存ファイル・欠陥表・標準経路は変更しない。Git の commit はしない。
- 検出条件は field_stage1.py と同じ（abn & finite(z5p) & excl & |z5p|>6、座標 ctr@Llin.T+tL）。機会(ii)は excl を問わず、変換後座標が10画素以内の全ピラー。
- 事前登録の基準に疑問があれば、解析は基準どおり実行したうえで、report.md の末尾に「基準への異議」として書く（黙って変えない）。
- 末尾に、実行コマンド・実行時間・出力ファイルのSHA-256一覧を `v89_manifest.csv` に出す。
