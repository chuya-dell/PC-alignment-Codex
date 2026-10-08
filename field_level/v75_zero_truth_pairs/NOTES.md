# v75 手順B:正解ゼロの対で読み取りの欠陥を測る(Claude Code 単独。【独立批評待ち(Codex の枠切れ)】)
- `v75_pair_lib.py`:任意の画像対を S0〜S5(+A写像版)で解析(v70 の格子・実中心、v72 の読み取りを再利用)。
- `v75_run_pairs.py real|synth`:連写1組+小刻み6組、合成18組(連写2枚目を4倍sinc→微細格子で1/4画素ずらし→面積積分、回転は Lanczos4)。
- 閾値:各組 mean+3SD、小刻み6組のプール、局所(v72_analyze.local_resid_ids:格子整数座標の実距離5周期の円盤中央値)、頑健SDの k倍。
- 結果:`data/results/v75_zero_truth_pairs/`(Git対象外)。詳細は Vault の【未読】手順B_*.md。
- 注意:v71_zero_truth_pairs は Codex が枠切れで途中停止した未コミットのスクリプト群(結果なし)。本版では使っていない。
- v72_analyze.py は局所閾値の不具合(区画が小さく無効)を修正し、v2 出力(`*_v2.csv`)を追加。初版の Lc 列は無効。
