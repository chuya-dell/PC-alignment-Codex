# Validation record

## Static checks

`python -m py_compile` を、このフォルダの実行スクリプトと依存する共有スナップショット 4 ファイルに対して実行する。

## Reproducible reference results

`data/results/reference_snapshots/crossday_sample1_blank_summary_260907.csv` は、画像本体やキャッシュを含まない集計済みの再現参照値である。FOV を独立単位にした値は以下。

| Dataset | High / blank samples | FOV n | High threshold-exceedance | Blank threshold-exceedance | FOV exact MW p |
| --- | --- | --- | ---: | ---: | ---: |
| 260824 SAM | 1 / 14 | 8 / 5 | 0.905% | 1.115% | 0.524 |
| 260825 DNA | 1 / 0 | 8 / 7 | 1.143% | 1.050% | 0.613 |
| 260826 SAM | 1 / 9 | 7 / 1 | 0.593% | 1.085% | 0.500* |
| 260828 DNA | 1 / 8 | 8 / 7 | 0.307% | 0.960% | 0.867 |
| 260829 SAM | 1 / 8 | 8 / 7 | 0.856% | 0.807% | 0.694 |

\* blank FOV が 1 のため推論には使わない。

画像・中間 NPZ・ピーク CSV・生成プロットは意図的に含めない。
