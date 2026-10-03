# v28 外れ値の帯状分布・周1

## 実装内容・旧版からの変更点

新しい比較専用の版。標準経路・既定値・マスク・旧版には変更を加えていない。
`field_round1_preflight.ps1` は前回の入力監査。保持した。

再開後に比較専用の `field_round1_analysis.py`（閾値・密度・共通平均残差・自己相関・スペクトル・分類・図録）、`field_round1_mechanisms.py`（元変換の回復照合、単独画像・十字線・人工データ）、`field_round1_verify.py`（既知周期・数値処理・入力不変の検証）、`field_round1_report.py`（集計・検収報告・出力一覧）を追加した。元入力コードは読込み・必要操作のコピーのみで、凍結中の別リポジトリは参照・実行していない。

元の予測・分類規則と内容指紋は不変。追加実装規定を別文書・別指紋として、再開後の地図・説明率を見る前に保存した。実装初期の方向区分境界の欠落と変換診断定義の誤りを修正し、修正前の試行は別フォルダに保持、最終集計から外した。信号処理の広い依存読込みで動的ライブラリの実行拒否があったため、高速フーリエ変換と数値配列による線形自己相関・山の検出で継続した。描画キャッシュは結果内に限定した。

## 結果概要と保存先

`data/results/v28_band_origin_round1/` に対象一覧、監査表、予測、指紋、`261004_周1_報告.md` を保存。
前回停止報告を保持し、再開後の報告を `261004_周1_報告_完了版.md` として追加する。ファイル名の「完了版」は今回の検収報告を指し、仮説B・Fの科学的検証の完了ではない。

632視野の従来割合を保存差から再現した。3%以上は従来29視野、中央値・絶対偏差104視野。二つの定義、全面・縁200・300画素の3,792条件を分類、全視野の自己相関・スペクトルを保存。全面の従来分類は帯状候補62・うろこ状候補18・その他552視野。固定29視野と定義別対象の比較を保存した。

260926の70視野で元変換の再現を試し、診断と全標本値の両条件で再現0視野。保存差のみの完全一致2視野も元座標の一意回復ではない。仮説B・Fの予定8検定、実データ説明率、共通部分、空間移動対照は未実施・欠測として保存した。人工像の比較を実データの原因説明率の代用にしていない。

既知448画素水平帯、自己相関の重なり、方向区分境界、全632保存差と140生画像の計算前後指紋不変を確認した。生成物は全て `data/results/v28_band_origin_round1/`。個別パスと内容は完了版報告と `round1_output_inventory.csv`。

## 既知の問題・未解決事項

保存データに洗浄後標本化座標・最終変換行列・局所残差がなく、当時の追加位置合わせコードもコピー内にない。元の変換候補が再現しないため、独立検証済み局所残差の測定へ進めない。これらをそろえてB・Fの実データ検定・説明率を完結する必要がある。未検定を0%で補わない。

260830の6画像は読めるが撮影条件・時点の対応表がなく、番号から洗浄前後を推測しない。その他8日程の生画像は未コピー。人工像は全視野固定周期の小さい成分のみで、全条件で実データ閾値による参考外れ値0。これは仮説の否定ではない。

分類は升目・方向の分解能や端の影響を受け、目視印象と一致しない。候補を承認済み欠陥として使わず、標準解析のマスク・除外に入れない。十字線の近似は一部の線で誤差が大きく、参考の位置・方向情報として扱う。

## 再実行

作業フォルダで `.venv/Scripts/python.exe` を使い、以下の順で各スクリプトを実行する。入力の実在名を起点一覧から確認し、元予測の指紋を毎回照合する。分類・図録・変換照合は視野別途中保存があれば再利用する。試行用の別フォルダは再利用しない。

```text
field_round1_analysis.py maps
field_round1_analysis.py classify --limit 2
field_round1_analysis.py classify
field_round1_mechanisms.py recovery --limit 2
field_round1_mechanisms.py recovery
field_round1_mechanisms.py raw
field_round1_mechanisms.py synthetic
field_round1_mechanisms.py markers
field_round1_verify.py
field_round1_analysis.py atlas
field_round1_report.py
```

入力と予測が変わる場合は、既存途中保存を無条件に再利用せず、新しい版として扱う。コミット・タグ・プッシュはClaude Codeの検収後に担当する。

## 2026年10月4日・追加指示による仮説B・F検定編

### 実装内容

最新の継続指示に従って同じ版へ比較専用の追加コードを置いた。既存五スクリプト、元予測・分類規則・追加実装規定・指紋、過去三報告は変更していない。

- `field_round1_bf_recovery.py`：読取り専用の `data/inputs_local/code_v24/` の元実装で位置合わせ・格子・整数標本化を一度だけ再計算し、全ピラーの差と識別・洗浄前座標を照合。視野別に途中保存。
- `field_round1_bf_models.py`：丸め誤差と周期像の予測変数、独立局所追跡、空間交差検証、移動・回転対照、八検定、説明率分解、方式別ブランク再校正と補助相関。
- `field_round1_bf_validate.py`：既知の説明変数と無関係対照、学習余白、非循環移動、ゼロ変換を検証。実際に開いて確認した非周期十字線の独立位相監査。
- `field_round1_bf_report.py`：新しい検収報告、入力の指紋不変監査、指定タグとの改行正規化比較、全出力・入力パス一覧。

回復は最新指示の最大絶対差0.000001以内を適用し、診断一致や微小位置合わせ不採用を追加の回復条件にしない。診断は独立した列で保存する。元の八検定を保持し、最新指示による利用可能部分での検定であることを明示する。未指定の数値実装を `261004_B_F検定編_実行整理.md` に説明率の計算前に保存し、その指紋を記録した。初期の構文誤りを特徴変数の計算前に修正し、分類規則・予測を結果に合わせて変更していない。

### 結果と保存先

`data/results/v28_band_origin_round1/261004_周1_報告_B_F検定編.md` と `round1_bf_output_inventory.csv`。
260926の70視野中34視野を全ピラー最大差0で回復し、外れ値11視野中5視野を含む。36視野は不一致のまま保存。洗浄前格子と識別は全70視野で再現する。

仮説Bは34視野と外れ値5視野の利用可能部分で確認的検定を実施したが、補正後に支持されない。仮説Fは支持・独立検証条件と非周期目印を満たす2視野だけで実施し、外れ値は基板7視野5の1視野だけ。判定不能。説明率・共通部分・固有分・相互作用・負値・九対照を全て保存した。検定群八比較の補正後有意確率は全て1。

使用した生画像140ファイルと保存差70ファイルの実行前後指紋不変を確認。元コード四ファイルは指定タグと改行正規化後に一致する。全出力は生成物の指定先、比較コードはこのフォルダだけ。コミット・タグ・プッシュは未実施。

### 既知の問題

回復可否による選択の偏り、当時の実行コミット未確定、丸め後標本値一致による丸め前行列の非一意性が残る。第一殻だけの周期像では局所像・高次成分を表さない。局所残差候補は30視野で支持条件を通過したが、非周期目印の確認がない28視野は補助のみ。外れ値の基板01視野8と基板7視野8は支持区画が27・31で、必要な32区画を満たさずFを採用していない。

方式別ブランク再校正は回復7／元8視野で不完全なため補助扱い。260830の前後対応は未確定。他日程のコピー完了印を報告時に確認できず、260922・260924・260828の別日程再現は未実施。標準経路やゴミ・シミのマスク既定値を変更していない。

### 追加分の再実行順

```text
.venv/Scripts/python.exe -B field_level/v28_band_origin_round1/field_round1_bf_recovery.py --limit 2
.venv/Scripts/python.exe -B field_level/v28_band_origin_round1/field_round1_bf_recovery.py
.venv/Scripts/python.exe -B field_level/v28_band_origin_round1/field_round1_bf_models.py features --limit 2
.venv/Scripts/python.exe -B field_level/v28_band_origin_round1/field_round1_bf_models.py features
.venv/Scripts/python.exe -B field_level/v28_band_origin_round1/field_round1_bf_models.py markers
.venv/Scripts/python.exe -B field_level/v28_band_origin_round1/field_round1_bf_validate.py
.venv/Scripts/python.exe -B field_level/v28_band_origin_round1/field_round1_bf_models.py evaluate --limit 2
.venv/Scripts/python.exe -B field_level/v28_band_origin_round1/field_round1_bf_models.py evaluate
.venv/Scripts/python.exe -B field_level/v28_band_origin_round1/field_round1_bf_models.py comparison
.venv/Scripts/python.exe -B field_level/v28_band_origin_round1/field_round1_bf_models.py plots
.venv/Scripts/python.exe -B field_level/v28_band_origin_round1/field_round1_bf_report.py
```

目印の目視確認は本実行で開いた四図に限定する。異なる入力で同じ承認を流用しない。入力・コード・予測が変わる場合や独立検証の採用状態を更新する場合は、新しい版か別名の途中保存を使い、現在の結果を上書きしない。
