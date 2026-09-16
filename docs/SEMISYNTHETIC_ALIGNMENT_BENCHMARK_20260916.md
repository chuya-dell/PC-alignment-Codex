# 半合成・既知正解ベンチマークによる位置合わせ方式比較 — 2026-09-16

## 結論

今後の標準候補は **アフィンECC + 0.25→0.5→1.0ピラミッド** とする。
同一の既知アフィン正解、スパイク位置、評価閾値を用いた5視野の半合成ベンチマークで、
ECCは平均回収率99.80%、偽陽性率0.00%、回収本数の線形性（平均R²）0.999984を示した。

ORB+RANSACも同等以上の平均回収率99.93%、偽陽性率0.00%、平均R² 0.999994だった。
ただし特徴点の数・RANSACの対応集合に依存し、反復ごとの決定性と実画像での失敗モードを
より強く管理する必要がある。現行のECCは既存Phase 2コードを直接再利用でき、既知正解への
変換誤差も各軸約0.003 px以下で安定していたため、標準として採用する。

密なFarnebackオプティカルフローからのアフィンfitは、平均回収率26.87%、偽陽性率99.94%で
不採用とする。6〜10 px程度の並進誤差と0.125〜0.300度の回転誤差が残り、周期格子の差分を
大量に偽陽性として検出した。

## ベンチマークの現状

既存の半合成ベンチマーク本体は、PC-alignment-Codex、許可範囲のPC内フォルダ、Google Drive同期
フォルダから確認できなかった。凍結対象のAnti側実装は参照・実行していない。

そのため本比較では、既存のPhase 2の画像読込・生16-bit→0..1正規化・アフィンECCを再利用して、
正解変換とスパイク位置を出力CSVで明示保存する `v5_semisynthetic_alignment_benchmark` を追加した。
これが2026-09-16時点で動作確認済みのversionedベンチマークである。

## 条件

- 画像: 260826 p50 SAM Sample 1のPosition 1, 2, 5, 6, 8のpre画像。
- 合成: 各pre画像から既知の小アフィン変換でmoving画像を作成し、moving座標系へ正のガウス型
  スパイクを20、50、100本ずつ注入した。
- 正解: template→movingの既知アフィン行列と、template座標系の全スパイク位置。
- 回収: 推定行列でmovingをtemplateへ戻し、差分局所極大が正解から3 px以内なら回収。
- 偽陽性率: 正解から3 pxより遠い検出数 / 全検出数。
- 公平性: 同一FOVでは3方式が同一moving画像、同一スパイク位置、同一閾値を使う。

## 手法

| 手法 | 内容 |
| --- | --- |
| `ecc_affine_pyramid` | 既存Phase 2のアフィンECC。0.25→0.5→1.0、各段最大50反復、EPS=1e-6。 |
| `orb_ransac_affine` | ORB特徴点、Lowe比0.72、RANSACによる完全アフィン推定。 |
| `dense_farneback_affine` | 0.25倍Farneback flowを格子サンプルし、RANSAC最小二乗でアフィンfit。 |

## 全視野比較

| 手法 | 平均回収率 | 平均偽陽性率 | 回収本数の平均傾き | 平均R² |
| --- | ---: | ---: | ---: | ---: |
| ECC affine pyramid | 99.80% | 0.00% | 0.9978 | 0.999984 |
| ORB RANSAC affine | 99.93% | 0.00% | 0.9973 | 0.999994 |
| Dense Farneback affine | 26.87% | 99.94% | 0.3288 | 0.944628 |

## Position 6

| 手法 | 回収率 | 偽陽性率 | 回収本数の傾き | R² |
| --- | ---: | ---: | ---: | ---: |
| ECC affine pyramid | 99.00% | 0.00% | 0.9888 | 0.999920 |
| ORB RANSAC affine | 99.67% | 0.00% | 0.9867 | 0.999971 |
| Dense Farneback affine | 28.67% | 99.93% | 0.3163 | 0.965170 |

Position 6でECCまたはORBに極端な悪化はない。少なくともこの既知変換・人工スパイク条件では、
Position 6は局所解トラップを特異的に起こす視野とは示されなかった。

## 採用と運用

1. 標準はECC affine pyramidとする。実データでは変換行列、ECC相関、段別履歴を保存する。
2. ORB/RANSACはECCの独立監査・フォールバック候補として残すが、標準を置き換えない。
3. Dense Farneback affineは本条件では採用しない。
4. このベンチマークは既知の小アフィン変換に限る。実写の傷、局所欠陥、照明変動、非線形歪みを
   模した追加ケースを将来の別バージョンで加える必要がある。

## 出力

- 実行: `field_level/v5_semisynthetic_alignment_benchmark/field_run_semisynthetic_alignment_benchmark.py`
- 集約: `field_level/v5_semisynthetic_alignment_benchmark/field_aggregate_semisynthetic_benchmark.py`
- 試行別結果: `data/results/v5_semisynthetic_alignment_benchmark_v2/semisynthetic_trial_results.csv`
- 既知変換との差: `data/results/v5_semisynthetic_alignment_benchmark_v2/registration_known_truth_errors.csv`
- 方式集約: `data/results/v5_semisynthetic_alignment_benchmark_v2/method_summary.csv`
