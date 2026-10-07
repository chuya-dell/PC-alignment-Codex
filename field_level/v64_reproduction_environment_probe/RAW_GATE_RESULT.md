# 追加A 原画像再現監査（2026-10-08、未確認）

## 結果
両側289画像対・578画像のSHA-256がすべて一致。578画像の棚卸し・比較を完了した。これは現在の二つの保存先の同一性であり、過去のキャッシュ生成時の画像との同一性を保証しない。

v54と旧標準の283キャッシュ比較では、差分161視野が不一致、識別子283視野は完全一致。格子座標64視野の完全一致判定は偽だが、最大差7.389644451905042e-13画素で固定許容差atol=1e-12以内。これを実質的な格子不一致として説明しない。

差分・座標の4分類から各1視野を選び、両保存先の原画像から再計算（合計8回）。全8回がv54差分に完全一致。旧標準差分は4/8回一致、4/8回不一致。不一致最大差は選択視野で0.14092957973480225。入力の前後指紋は全8回一致。原画像の全283視野再生成は行っていない。診断で旧標準との差が再現したためゲート不通過を確定し、局所補正・位置シャッフル対照・ダミー補正を行わない。

Holm集合は9日程540検定。既存キャッシュ集計の9,027比較一致と原画像ゲートは区別する。v54キャッシュの閾値108・陽性数/率372・濃度別陽性数/未補正p100・相関44の不一致は、Holm集合を揃えても残る。540集合でHolm値の不一致0は、他の差を解消した意味ではない。

## 原因の範囲と限界
現在の画像保存先の違い、監査側とv54側の4依存ソースの違い、選択3視野の1/20スレッド差を除外した。最適化無効でも旧標準を再現しない。差は旧標準キャッシュ生成構成または生成時の入力と、現在の差分生成との間に残るが、根本原因は未特定。準備時環境の版一覧はキャッシュ生成環境の証拠ではない。

旧標準の`01_report.md`第1節は、古い4日程の生成構成を完全証明できないと明記。キャッシュはdelta/xy/idsのみ、生成時の依存バイナリ・変換行列がない。現在の外部マニフェストの作成スクリプト記録だけでは、生成時の依存構成を復元できない。根本原因の達成条件は未達。原因を特定済みとして扱わない。

## 保存と再開
元クローンのindex.lockは存在しない。`.git/codex_permission_probe.tmp`作成はアクセス拒否（残骸なし）。独立チェックアウトはcommit/tag成功。ロック競合ではなくsandbox書込み境界が直接の阻害要因。pushは指示どおりClaude Code担当。

最後に終えた区切り: 578画像の内容比較、4分類8回原画像診断、3視野9回数値環境診断、証拠指紋と出所登録。
再開地点: 旧標準のキャッシュ生成時の依存ソース・パッケージの実体／版・入力指紋・変換行列を回収し、同じ選択視野で再現する。これらの証拠なしに環境差や旧方式混在を原因と断定せず、基準も変更しない。

詳細出力: `data/results/v63_local_correction_reproduction_audit/{input_audit.json,rerun_probe.json,tables/table_input_hash_comparison.csv,tables/table_input_vs_cache_match.csv,tables/table_rerun_probe.csv}`、`data/results/v64_reproduction_environment_probe/{audit_completion.json,environment.json,thread_probe.json,table_thread_probe.csv}`。
実行コマンド: `C:/Users/chuya/PC-alignment-localcorr/.venv/Scripts/python.exe field_level/v63_local_correction_reproduction_audit/field_audit_reproduction.py --rerun 1`、同Pythonでv64の`field_probe_environment.py`、`field_finalize_audit.py`。
代理残差の低下、判定変化、位置合わせ精度向上は別物。局所補正採否は本人待ち、検収はClaude Code待ち。
