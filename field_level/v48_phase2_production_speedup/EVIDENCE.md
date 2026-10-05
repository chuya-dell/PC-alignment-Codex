# v48 根拠の記録(EVIDENCE)

規約:AGENTS.md「根拠の記録」(origin/claude/friendly-bardeen-f0ybtq の版)に従う。

## E1 1視野対の時間の内訳
1. 確かめたこと:どの処理が実行時間の大半を占めるか。
2. 方法:同じ処理を段階ごとに `time.perf_counter()` で測るスクリプト(`profile_pair.py`、作業用の一時フォルダ)を、260824_p50_SHC6OH の 1-1、1-2、1-6 の3対で実行。さらに `cProfile` で `lattice_from_fft` と `sample_grid_features` の内部を測定。コードの版:`origin/main` の `ceb764d`(`shared/` と v9)。
3. 数値(1-1の例):読み込み0.277秒、マスク0.118秒、位置合わせ0.711秒、`lattice_from_fft` 1.270秒、`sample_grid_features` 0.805秒、投影後サンプリング0.117秒、合計3.298秒。1-2:3.301秒、1-6:3.356秒。`cProfile`:`sample_grid_features` は `ndarray.mean` を85,793回呼ぶ Python ループが主。`lattice_from_fft` は `estimate_phase_origin_fft` が0.502秒、`estimate_hex_orientation_fft` が0.723秒(うち `fft2` 0.307秒、`maximum_filter` 0.162秒)。
4. 区分:事実(実測)。
5. 別の説明:なし(時間の内訳の測定)。
6. 未確認:他の機械・他の日程の画像での内訳。

## E2 高速版・並列版が結果を変えないこと
1. 確かめたこと:v48 の4通りの実行が、v9 の逐次実行と同じ出力を出すか。
2. 方法:`field_verify_speedup.py`(`--limit 40 --workers 8`)。基準=v9 の `process_pair_cached` を、元の関数・OpenCV 既定のスレッドで逐次実行したキャッシュ。比べたもの:各視野対の要約(状態、個数、全ての小数)と `delta` 配列(完全一致と最大差)。入力:`W:\GoogleDrive\remotefdtd\4.生データD_remo\260824_p50_SHC6OH\Raw_Images_生データのみ` の先頭40対(ファイル名の並び順)。
3. 数値:`sequential_fast_kernels`、`parallel_original`、`parallel_fast` のいずれも、`n_fields`=40、`summaries_identical`=40、`summary_float_max_abs_diff`=0.0、`delta_arrays_bitwise_equal`=40/40、`delta_max_abs_diff_where_different`=0.0。8対の予備実行でも同じ。
4. 区分:事実。
5. 別の説明と採らなかった理由:許容差つきの一致(例:1e-9以内)で十分という考えもあるが、ビット単位で一致したため、許容差は不要だった。
6. 未確認:位置合わせが棄却される対を多く含む視野、全409対。

## E3 出力表の一致(最後まで通した実行)
1. 確かめたこと:v9 の `main()` を逐次で走らせた出力表と、v48 の並列実行の出力表が同一か。
2. 方法:260824_p50_SHC6OH の12対(最後の2対を `is_blank=True` に指定して、v9 の閾値・検定の部分も実行させた)。v9 の `field_run_phase2_production_reanalysis.py` と v48 の `field_run_phase2_production_parallel.py --workers 8` を実行し、出力フォルダの5ファイルの SHA256 を比べた。
3. 数値:`phase2_reanalysis_blank_thresholds.csv`、`phase2_reanalysis_fov_exceeds_rate.csv`、`phase2_reanalysis_fov_summary.csv`、`phase2_reanalysis_mannwhitney_summary.csv`、`phase2_reanalysis_run_summary.json` の全部が IDENTICAL。壁時計の時間:v9 逐次 41.4秒、v48 並列 14.0秒(12対、起動時間を含む)。
4. 区分:事実。
5. 別の説明:なし。
6. 未確認:ブランクの指定は、速度の検証用に私が付けたもの。実際の濃度系列の一覧では未実施。

## E5 現在の標準構成の土台での再検証
1. 確かめたこと:E1・E2・E4 が、現在の標準構成の位置合わせ(v15 以降)を含むコードでも成り立つか。E1〜E4 は `origin/main`(`ceb764d`)の `shared/` で測っており、`origin/main` は `origin/refinement/outlier-artifact-mask-20260927`(`c6be781`)を含まない(`git merge-base --is-ancestor c6be781 ceb764d` が「含まない」)ため。
2. 方法:`c6be781` の作業ツリーに v48 のフォルダを置き、`field_verify_speedup.py --limit 40 --workers 8` と1視野対の内訳測定を、E2 と同じ入力で実行。
3. 数値:逐次・元 115.19秒(2.88秒/対)、逐次・高速版 73.8秒、並列8・元 31.16秒、並列8・高速版 24.04秒(0.601秒/対)=4.8倍。3つの変種とも `summaries_identical`=40、`delta_arrays_bitwise_equal`=40/40、最大差0.0。内訳(1-1):合計2.792秒、`lattice_from_fft` 1.142秒、`sample_grid_features` 0.805秒、`register_image_pair_affine` 0.652秒。
4. 区分:事実。
5. 別の説明:なし。
6. 未確認:E3(出力表の最終比較)は `origin/main` の土台でのみ実施した。ただし、`field_level/v9_phase2_production_reanalysis/` と `shared/concentration_series_stats.py` は、`origin/main` と `c6be781` で差分なし(`git diff --stat` が空)。統計表を作る部分のコードは同一である。

## E4 速度
1. 確かめたこと:速度比。
2. 方法:E2 と同じ実行の壁時計の時間。別に、高速版を4・12・16並列でも実行(40対)。
3. 数値:本文の `NOTES.md` の表のとおり。逐次・元 128.7秒(3.22秒/対)→ 並列8・高速版 28.5秒(0.71秒/対)=4.5倍。
4. 区分:事実(この機械での測定)。
5. 採らなかった案と理由:(a) 画像処理装置(GPU)の利用:助言ではメモリの面で不利になる可能性があった。この機械では内訳の上位が Python ループと小さな配列演算であり、先に結果を変えない範囲の対策で4.5倍が出たため、検討しない(未検証)。(b) 位置合わせの高速化(解像度を下げる・反復回数を減らす等):結果が変わりうる。「結果を変えない範囲」の条件に反するため採らない。(c) `fft2` を `scipy.fft` に替える・並列化する:`estimate_hex_orientation_fft` の約0.3秒だが、スペクトルの極大点の同値判定(`maximum_filter(...) == spectrum`)が丸めの違いで変わるおそれがあり、一致を保証しにくい。今回は採らなかった。
6. 未確認:共用のデルデスクトップでの速度、全409対・1600枚規模での時間。
