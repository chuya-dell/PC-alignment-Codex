# PC-alignment-Codex

このリポジトリはCodex担当分。対応するClaude版は PC-alignment-claude、Antigravity版は PC-alignment-anti。3手法の比較はいずれ実施予定。

プラズモニック結晶を用いるデジタル単分子カウントの位置合わせ処理を、ピラー単位と視野単位で明確に分離して管理する。

## 構成と命名規則

- `pillar_level/`: ピラー1本ごとの座標、コントラスト、欠陥領域の解析。実行ファイルは `pillar_` で始める。
- `field_level/`: 視野を単位とする集計・比較。実行ファイルは `field_` で始める。
- `shared/`: 両粒度で共用する位置合わせ、格子、画像サンプリング、承認済みマスクの安全な読込み。粒度を持たないためここへ置く。
- `data/raw/`: 実験画像へのパスを示すマニフェストと、一次記録から構造化した入力台帳。画像本体は格納しない。
- `data/results/`: 実行で生成するレビュー画像・CSV・比較表の保存先。Gitには入れない。

各バージョンフォルダには必ず `NOTES.md` を置く。既存バージョンのコード・パラメータは上書きせず、変更は新しい `vN_説明/` を追加して行う。

他のPCで作業を再開する場合は、最初に [`AGENTS.md`](AGENTS.md) と [`docs/REPOSITORY_CONVENTIONS.md`](docs/REPOSITORY_CONVENTIONS.md) を読む。新規解析を置く前に、粒度とバージョンをここで決める。

## 現行バージョン

- [`pillar_level/v1_human_approved_pixel_masks/`](pillar_level/v1_human_approved_pixel_masks/): 周期残差を候補化し、目視承認されたpre画像座標ポリゴンだけをマスクにする処理。
- [`field_level/v1_pixel_mask_count_comparison/`](field_level/v1_pixel_mask_count_comparison/): 旧視野全体除外方式と承認済みピクセルマスク方式の有効数比較。p値の再計算はしない。
- [`field_level/source_snapshot_260910_raw_qc_brightness/`](field_level/source_snapshot_260910_raw_qc_brightness/): 他チャットから保存した全視野raw輝度差の補助QC。既存の格子点エンドポイントとは別物で、新旧関係は未判定。

## p50 視野内ピッチ均一性（2026-09-15）

高濃度SAM/DNAのpre画像11視野（5日程）で、既存のサブピクセルピーク検出、半径別最近接距離、
5×5局所FFT、Brown放射歪みfitを検証した。全FOVに共通する単調な放射歪み・再現可能な
`k1, k2` は確認されず、確定済みのグローバル `a=7.286 px` を変更しない。
260826 FOV 1-6と260827位置3・5の既知の非収束も、同一の放射歪みパターンでは説明できなかった。
詳細、候補、数値、再実行方法は
[`docs/P50_RADIAL_PITCH_DISTORTION_REPORT_20260915.md`](docs/P50_RADIAL_PITCH_DISTORTION_REPORT_20260915.md) を参照。

## FOV 1-6 ECC原因診断（Phase 2, 2026-09-16）

260826 SAM Sample 1 Position 6を、0.125→0.25→0.5→1.0の深いピラミッドと
25初期並進で独立ECC診断した。全25経路が同じ `rho=0.98660` 近傍の解へ収束し、
標準3段・ゼロ開始も同値だったため、深い初期化が別の大域解を見つけた証拠はない。
ピーク品質・焦点・FFTの粗い安定性でもPosition 6固有の決定的画質劣化は得られなかった。
元の2026-09-01 ECC実装は本リポジトリに無いため、デフォルトは変更しない。
詳細は [`docs/FOV16_ECC_ROOT_CAUSE_PHASE2_REPORT_20260916.md`](docs/FOV16_ECC_ROOT_CAUSE_PHASE2_REPORT_20260916.md) を参照。

## 2026-09-01 ECC実装の捜索（2026-09-16）

当時のアフィン・多段ピラミッドECCを再現するため、PC内、Google Drive同期フォルダ、
`PC-alignment-claude` と `PC-alignment-anti` のGit履歴を捜索した。同名の
`batch_register_images.py` は発見したが、2026-07月版の回転+並進のみの単段ECCであり、
記録された拡大縮小差や3段ピラミッドを出せない別実装だった。したがって再現テストや
デフォルト変更は行っていない。詳細は
[`docs/ECC_IMPLEMENTATION_FORENSICS_20260916.md`](docs/ECC_IMPLEMENTATION_FORENSICS_20260916.md) を参照。

## 2026-09-01 ECC記録に対する再構成検証（2026-09-16）

Phase 2のアフィンECCを0.25→0.5→1.0の3段ピラミッドとして再利用し、Position 1〜5、7、8を
当時の8視野台帳と比較した。全項目一致は0/7で、並進と重なりの良さを同時に再現できなかったため、
この再構成は当時の実装の代理として使わない。従ってPosition 6の0.125開始・多開始点探索は実行せず、
デフォルトも変更しない。詳細は
[`docs/ECC_20260901_REFERENCE_VALIDATION_20260916.md`](docs/ECC_20260901_REFERENCE_VALIDATION_20260916.md) を参照。

## 半合成・既知正解ベンチマークによる方式選定（2026-09-16、要更新 — 下記精査を参照）

Position 1, 2, 5, 6, 8に同一の既知アフィン変換と20/50/100本の人工スパイクを与え、
ECC、ORB/RANSAC、Dense Farnebackを比較した。ECC affine pyramidを標準候補とした。
ECCは平均回収率99.80%、偽陽性率0.00%、Position 6でも99.00% / 0.00%だった。
ORB/RANSACは独立監査・フォールバック候補、Dense Farnebackは不採用とする。詳細は
[`docs/SEMISYNTHETIC_ALIGNMENT_BENCHMARK_20260916.md`](docs/SEMISYNTHETIC_ALIGNMENT_BENCHMARK_20260916.md) を参照。

**「Position 6に特異な悪化はない」という結論は、下記の精査により取り下げ。**
埋め込んだ変換量が実データの並進変動幅（横−26.8〜+9.5px, 縦−8.1〜+20.2px）の
一部しかカバーしていなかった（横は最大変位の約28%のみ、かつ常に負符号）。実データ規模の
変換で再実行すると、ECCはPosition 6相当の大きな並進（−26.8px級）で局所解に陥り、
回収率99.00%→34.67%、偽陽性率0.00%→99.94%まで悪化した。ORB/RANSACはほぼ影響を
受けなかった（99.67%→99.33%）。また回収率以外の基準（計算時間・決定性）を実測した
ところ、ORB/RANSACの方が約2.8倍高速で、決定性も今回の環境では差が出なかった。
標準候補の最終判断には実データ規模の変換をカバーする再評価が必要。詳細・再実行方法・
生データは
[`docs/SEMISYNTHETIC_BENCHMARK_VERIFICATION_20260916.md`](docs/SEMISYNTHETIC_BENCHMARK_VERIFICATION_20260916.md) を参照。

自動候補は欠陥として採用されない。pre/post双方の目視確認、承認者、承認日、pre画像ネイティブ画素座標のポリゴンが揃った承認行だけが解析へ流れる。

## 他チャットからの保存

取り込み済み成果とその出所は [`data/raw/artifact_provenance_registry.csv`](data/raw/artifact_provenance_registry.csv) に記録する。作成日時は出所の手掛かりに留め、内容・パラメータ・検証結果の比較が済むまで、どちらが新しいかを判定しない。
