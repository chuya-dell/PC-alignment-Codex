# 数値環境診断の観測（未確認）

実行日: 2026-10-08。入力を変更せず、3視野を各3設定で計算。

|視野|最適化有効・1スレッド:旧標準差分最大差|20スレッド|最適化無効|
|---|---:|---:|---:|
|260825_0_1|0.14092957973480225|同じ配列|0.5825446844100952|
|260825_0_2|0（完全一致）|同じ配列|0.5199535489082336|
|260828_1_1|0.0592731237411499|同じ配列|0.05927315354347229|

最適化有効6試行はすべてv54差分配列に完全一致。全9試行で格子座標・識別子は旧標準と一致。1スレッド/20スレッドの違いでは、この3視野の不一致を説明できない。最適化無効を再現方法として採用しない。

Python 3.13.7、OpenCV 4.13.0、NumPy 2.3.3、SciPy 1.16.2で実行。現クローンとv54のregistration.py、lattice_indexing.py、image_qc.py、refinement.pyは内容指紋一致。準備時の版一覧（Python 3.12、OpenCV 5.0.0.93等）はキャッシュ作成環境を証明しない。旧標準報告も4日程キャッシュの生成構成の完全証明がないと明記している。環境差を根本原因として断定できない。

実行: `C:/Users/chuya/PC-alignment-localcorr/.venv/Scripts/python.exe field_level/v64_reproduction_environment_probe/field_probe_environment.py`
詳細: `data/results/v64_reproduction_environment_probe/{thread_probe.json,table_thread_probe.csv,environment.json}`。
原画像の前後SHA-256一致は全3視野で確認。局所補正・対照・ダミーは未実行。代理残差の改善を位置合わせ精度の改善と扱わない。
