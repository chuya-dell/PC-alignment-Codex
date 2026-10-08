# v86 部分C:対応率の引き上げ(Claude Code 単独。【独立批評待ち】)
- `v86_mf_centers.py`(整合フィルタの当てはめ)、`v86_variants.py`(要素ごとの比較。整合フィルタは対応率を上げない)、`v86_noise_limited.py`(2枚平均で置換率が下がらない=雑音ではない)、`v86_rates_final.py`(固定29+ブランクの前後比較、傷の外側、局所予測、届かない視野)。
- 結果:`data/results/v86_correspondence_mf/`。置換の閾値は前回のまま。「傷の外側を分母」は定義の変更(本人の確認が要る)。
