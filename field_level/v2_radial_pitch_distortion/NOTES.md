# v2 radial pitch distortion

## 実装内容

`field_assess_radial_pitch_distortion.py` は既存 `shared.analyzer.analyze_image` の
`method="peak"` をそのまま利用する。これはトップハット補正、局所極大、3×3重み付き
重心によるサブピクセル座標である。検出後だけ、最近接距離 5.5–9.0 px の保守的な
QCを掛け、幾何中心 (1023.5, 1021.5) px からの半径別ピッチと 5×5 ローカルFFTを比較する。

Brown fit は、確定済み a=7.286 px のFFT格子に整数インデックスを割り当ててから、
`r_d = r_i (1 + k1 r_i^2 + k2 r_i^4)` をロバスト最小二乗する。グローバル格子定数を
再推定する処理ではない。半径NNとローカルFFTの両方が事前の効果量ゲートを越えた時だけ
実行する。

## 結果概要

11視野（260824–260829、SAM/DNA、260827 Sample12位置3・5を含む）を実行した。
局所FFTの角−中心は−0.70%〜+0.84%で符号が一定せず、共通のBrown係数は得られなかった。
現行のグローバル `a=7.286 px` は変更しない。詳細は
`docs/P50_RADIAL_PITCH_DISTORTION_REPORT_20260915.md` を参照する。

生成物は `data/results/v2_radial_pitch_distortion/` にあり、候補選定表、各FOVの検出点CSV、
半径ビンCSV、5×5 FFT CSV、散布図、Brown fit JSON、横断要約CSVを含む。

## 既知の問題

最近接距離は検出漏れ・誤検出の影響を受けるため、必ずローカルFFTと同じ方向の傾向かを
確認する。検出器は既存ロジックを再利用しており、光軸中心の独立キャリブレーションは
このリポジトリには未登録のため、幾何中心を用いる。
