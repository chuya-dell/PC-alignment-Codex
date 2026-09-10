# v1_human_approved_pixel_masks — ピラー単位

## 実装内容

- pre画像のネイティブ画素座標を唯一のマスク座標系にした。
- 周期成分をFFTで再構成し、残差の連結領域だけを自動**候補**にする。単一ピラーの輝度変化は候補にしない。
- 一次記録は欠陥の下限として構造化した。未記載は欠陥なしを意味しない。
- すべてのpre/post画像ペアに視野全体レビュー行を作り、一次記録がある視野には独立したレビュー行を必ず作る。
- pre/post双方の確認、承認者、承認日、pre座標ポリゴンを満たす承認行だけを読み込む。

## 実行

リポジトリ直下から実行する。

```powershell
python pillar_level/v1_human_approved_pixel_masks/pillar_calibrate_defect_candidates.py --control-dir <260830-control-dir> --output data/results/control_candidate_sensitivity --max-images 16
python pillar_level/v1_human_approved_pixel_masks/pillar_prepare_defect_review.py --manifest data/raw/target_manifest.json --primary-records data/raw/primary_defect_records.csv --output data/results/review_candidates --cache data/results/cache
```

## 結果概要

260830コントロール代表16画像で、残差z閾値5/6/7の候補中央値は9/9/8領域/画像だった。初期設定は6。数値CSVは `data/results/control_candidate_sensitivity/` に保存する。これは候補感度の確認であり、欠陥の判定・承認ではない。

## 既知の問題・未解決事項

- 候補の妥当性は自動判定しない。実験者によるpre/post画像の確認が必要である。
- 一次記録に座標がないため、一次記録由来の領域はレビュー時にpre座標ポリゴンを入力する必要がある。
- 未承認候補は統計解析に使用できない。
