# v29 外れ値の帯状分布・周2

## 実装内容

仮説G（ピント・照明・周辺減光）と仮説I（撮影順序・洗浄前後の撮影条件）の比較専用解析。周1の対象・閾値・分類・残差地図を読み取りで利用する。標準経路、マスク、旧版を変更しない。実行ファイル名は `field_` で始める。

## 結果と保存先

`data/results/v29_band_origin_round2/261004_周2_報告.md`。事前予測と内容指紋を保存してから計算した。二定義の632視野、固定29視野をそのまま利用。人工勾配・周期帯の検証と2視野試行を経て全体へ拡張した。

二次面の固定29視野での交差検証説明率中央値は、平均・標準偏差の定義2.882%、中央値・絶対偏差の定義3.763%。周期帯は二次面を引いても残った。画像モデルは元の差を回復した260926の34視野（固定29中5視野）。固定外れ値利用可能部分でGは1.949%／3.084%、Iは7.165%／6.241%、共同6.891%／5.745%。符号付き固有・共通分解、9空間対照、四分位、負値を保存した。Iの局所像の説明率は撮影順序の証拠にはならない。

初回計算後の最終確認中に `_copy_done_2.txt` を確認したため、追加規定・内容指紋を追加画像の計算前に保存し、260828・260922・260924の216組を追加。全体画像指標は4日程286組、画像内記録は578枚（260830の6枚を含む）。初回20検定と初回報告をスナップショットとして保持し、追加12検定を合わせ32比較のボンフェローニ法を適用した。局所変換の回復・仮説B・Fの再解析はしない。

`field_round2_analysis.py` は実在パスと内容指紋の確認、画像指標、空間モデル、初回検定。`field_round2_completion.py` はコピー完了後の入力照合、追加日程、追加検定・数値検証。`field_round2_report.py` は単独像記述、集計、図、入力不変監査、報告を生成する。旧版を実行せず必要な数値規定のみ新しいファイルへコピーした。

## 既知の問題

撮影条件の変更と実際の試料変化は画像だけでは一意に分離できない。更新時刻の撮影時刻としての妥当性は未確認。260830の前後対応候補は確認的検定に使わない。初回規定の開始時には `copy_done_2` がなく、初回結果では別日程の生画像を使用していない。その後の追加完了・結果の先行確認の経緯を最終報告へ記録した。

低次面を移動・回転しても同じ関数空間になるため、Gの移動対照は原因の識別力を持たない。画像Iの全体前後変化の大きさは視野ごとの標準化で消え、局所画像構造との整合性だけを測る。中央の共通領域では全面より説明率が下がる。未回復252組の局所前後比較は欠測。画像統計の補間と鮮明さ代理量は厳密な光学モデルではない。基板間の処理順・濃度交絡は解消していない。

## 再実行

入力が変わらない場合の数値処理は各視野の途中保存を使う。元予測・追加規定の指紋を照合する。既存初回スナップショットを上書きするため `init` や初回 `tests` を無条件に再実行しない。新入力の場合は別版とする。

```text
.venv/Scripts/python.exe -B field_level/v29_band_origin_round2/field_round2_analysis.py validate
.venv/Scripts/python.exe -B field_level/v29_band_origin_round2/field_round2_analysis.py geometry
.venv/Scripts/python.exe -B field_level/v29_band_origin_round2/field_round2_analysis.py raw
.venv/Scripts/python.exe -B field_level/v29_band_origin_round2/field_round2_analysis.py optical
.venv/Scripts/python.exe -B field_level/v29_band_origin_round2/field_round2_completion.py tests
.venv/Scripts/python.exe -B field_level/v29_band_origin_round2/field_round2_completion.py validate
.venv/Scripts/python.exe -B field_level/v29_band_origin_round2/field_round2_report.py
```

コミット・タグ・プッシュ・既存標準経路の変更は行っていない。比較図と数値配列は生成物に限定しGitへ追加しない。指定報告を書いたところで停止する。
