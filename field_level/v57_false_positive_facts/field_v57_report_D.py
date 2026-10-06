import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np, json, cv2, hashlib
import field_v57_common as C
from field_v57_md import md
S = C.OUT/'stepD'
o = pd.read_csv(S/'detection_results.csv'); ds = pd.read_csv(S/'pillar_distance_summary.csv')
n = len(o); ok = int((o.status == 'ok').sum()); fail = n-ok
L = []
for r in o.itertuples():
    for l in json.loads(r.lines_json): L.append(dict(fid=r.fid, **l))
L = pd.DataFrame(L)
ct = pd.crosstab(o.n_lines_v, o.n_lines_h); ct.index = [f'縦{i}本' for i in ct.index]; ct.columns = [f'横{c}本' for c in ct.columns]
ctt = ct.reset_index().rename(columns={'index': ''})
st = L.groupby('family').agg(線の数=('z', 'size'), z中央値=('z', 'median'), z最小=('z', 'min'), 見える区間の割合の中央値=('visible_frac', 'median'), 見える区間の割合の最小=('visible_frac', 'min'),
                             太さ中央値px=('thickness_px', 'median'), 太さ最大px=('thickness_px', 'max')).reset_index().rename(columns={'family': '種類'})
st['種類'] = st['種類'].map({'vertical': '縦', 'horizontal': '横'})
weak = L[(L.z < 20) | (L.visible_frac < 0.5)][['fid', 'family', 'z', 'visible_frac', 'thickness_px']].copy()
weak['family'] = weak.family.map({'vertical': '縦', 'horizontal': '横'}); weak.columns = ['視野', '種類', 'z', '見える区間の割合', '太さpx']
odd = o[(o.n_lines_v != 1) | (o.n_lines_h != 1)]
lo = o[o.leftover_scar_like_components > 0][['fid', 'leftover_scar_like_components', 'leftover_scar_like_area_px', 'leftover_scar_like_bboxes']].copy()
lo.columns = ['視野', '成分数', '面積px', '位置(x,y,幅,高さ,面積)']
dd = o[['dark_px_in_mask', 'dark_absorbed_px_before_margin', 'dark_absorbed_px_after_margin', 'dark_leftover_touching_mask_px', 'dark_isolated_components', 'mask_fraction']]
nm = ds.n_in_mask.sum(); npil = ds.n.sum()
# montage of 12 random overlays
rng = np.random.default_rng(C.SEED)
ids = list(rng.choice(o.fid.to_numpy(), 12, replace=False))
ims = [cv2.resize(cv2.imread(str(S/'overlays'/f'{i}.jpg')), (512, 512)) for i in ids]
mont = np.vstack([np.hstack(ims[k:k+4]) for k in range(0, 12, 4)])
cv2.imwrite(str(S/'fig_D_overlay_random12.jpg'), mont, [cv2.IMWRITE_JPEG_QUALITY, 85])
freeze = (S/'DETECTOR_FREEZE.txt').read_text()
txt = f"""---
確認: 未確認
状態: 未読
date: 2026-10-06
---

【未確認】人工知能(Claude Code)が自動で回して出した結果。本人は未読。採否は本人が決める。傷の検出は洗浄前の生画像だけから行った(保存値・超過率は参照していない)。目視承認は待たず、仮置きで進めた。

# v57 手順D:十字傷の検出・完全マスク・距離

目的:擬陽性が出る理由の解明と対策の比較。この手順が効く点:傷の位置を全視野で座標として持ち、ピラーごとに傷からの距離を出す(これまで記録がなかった。v40の停止記録)。

## 先に報告:検出に成功した視野数と失敗した視野数(D1)

- **検出に成功:{ok}視野/{n}視野。失敗:{fail}視野。** 失敗した視野はない。過半数の視野での失敗はなく、D2以降を止める条件には当たらない。
- 成功の定義:縦または横の傷の線が1本以上、直線として当てはまった(ピーク z≥10 かつ線に沿った区間の35%以上で見える。z≥40 なら20%以上)。**全632視野に傷が写っていた**。標準の撮影位置が十字の近くにあるためと読める(解釈、確度:中)。
- 本数:
{md(ctt)}
  1視野に縦1本・横1本の十字が写るものが 603 視野(95%)。{len(odd)}視野は本数が違う(縦2本・横2本、あるいは片方が画像の外)。
- 線ごとの要約:
{md(st, floatfmt='{:.3g}')}
- 弱い検出(z<20 または見える区間<50%)の線:{len(weak)}本。一覧は `stepD/detection_results.csv`(lines_json)。薄い傷・途切れる傷は、見えている部分から当てはめ、途切れた部分は延長で補った(補った区間の数は lines_json の bridged)。
- 検出器は画像だけから作った。調整の経過(周回記録にも記載):初版(傾き±0.10)は、斜めの割れ目と、見える区間の少ない強い傷を取りこぼしたため、傾きの範囲を±0.30に広げ、強いピークの受理条件を追加し、画像の端で切れた傷の成分を取り込む処理と、マスク外の傷らしい成分の数値検査を加えて再実行した。距離別の超過率は、検出器の完成まで一度も見ていない。完成したコードと結果の指紋:
```
{freeze}```

## D1 検出の方法(再現用)

洗浄前画像を2倍に縮小して平滑化し、「背景(σ=20)に対する相対的な明暗の偏り」を傷のエネルギーとする。縦に近い線族と横に近い線族(傾き±0.30以内)に分け、傾きを変えて画像を傾け投影した1次元プロファイルの最大点を線の候補とする。32ピクセルごとの区間で線の位置を探し、見える区間から頑健に直線を当てはめる。線ごとにエネルギーのヒステリシス(種=ロバスト標準偏差の20倍、広がり=10倍)で実際の傷の領域を取り、見えない区間は見える区間の太さの中央値の帯で補う。コード:`field_level/v57_false_positive_facts/field_v57_detect_scars.py`。

## D2 完全マスク(傷の線の太さの実測+両側1周期)

- マスク=傷の領域(ヒステリシスの実測。明るい縁・暗い芯を含む)+端で切れた傷の成分+見えない区間の補い帯。暗部に接する成分を取り込み、両側に1周期(7.286ピクセル)膨張し、もう一度暗部に接する成分を取り込んだ。
- **傷の暗部の画素の取り残しの数値確認:**暗い画素(平滑化後、背景比−8%以下)のうち、マスクに接する(3ピクセル以内)ものの数は、**全632視野で0ピクセル**。マスク内の暗い画素の数の中央値 {int(dd.dark_px_in_mask.median()):,} ピクセル、マスクに取り込んだ数(膨張前)の中央値 {int(dd.dark_absorbed_px_before_margin.median()):,}、(膨張後)の中央値 {int(dd.dark_absorbed_px_after_margin.median())}。マスクと無関係な孤立した暗い成分(ごみ・しみ)は、1視野あたり中央値 {int(dd.dark_isolated_components.median())} 個(マスクしていない)。
- マスクが画像に占める割合:中央値 {dd.mask_fraction.median()*100:.1f}%(範囲 {dd.mask_fraction.min()*100:.1f}〜{dd.mask_fraction.max()*100:.1f}%)。ピラーのうちマスク内に入るのは {nm/npil*100:.1f}%({int(nm):,}/{int(npil):,})。
- **マスク外に残る、傷らしい長い強い成分(傷の線ではないが、しみ・ごみのまとまり)がある視野:9視野。**傷の検出が不完全な視野の疑いとして、感度解析で除く(失敗ではなく、この視野の遠方に傷の影響が残るかもしれない)。
{md(lo)}
- 端で切れた傷の成分(線として当てはまらない)を取り込んだ視野:{int((o.border_clipped_components_absorbed > 0).sum())}視野。
- 重ね合わせ画像(視野ごと。緑=マスクの縁、赤=縦の線、青=横の線):`data/results/v57_false_positive_facts_20261006/stepD/overlays/<視野>.jpg`(632枚。Gitには入れない)。ランダム12視野の一覧:`stepD/fig_D_overlay_random12.jpg`。マスク(視野ごと、ビット詰め):`stepD/<視野>_mask.npz`。

## D3 距離

- ピラーごとに、傷のマスクの縁までの距離(最短、周期=7.286ピクセル単位)と、画像の端(上下左右の最短)までの距離(ピクセル)を求めた。座標は洗浄前画像の座標(手順0-2)で、変換は要らない。
- マスク内のピラーは距離0で、解析から除く(別に記述だけ)。全632視野で、ピラー {int(npil):,} 本のうちマスク内 {int(nm):,} 本({nm/npil*100:.1f}%)。距離が5周期以内のピラー(マスク外)の視野あたり中央値 {int(ds.n_dist_le5.median()):,} 本、20周期超 {int(ds.n_dist_gt20.median()):,} 本。
- 表:`stepD/pillar_distance_summary.csv`、ピラーごとの距離:`stepD/<視野>_pillar_dist.npz`(dist_per=周期単位、edge_px=画像の端から、in_mask)。

## 出せなかったもの・限界

- 傷の検出の目視承認は取っていない(仮置き)。弱い検出(z<20 または見える区間<50%)と、マスク外に傷らしい成分が残る9視野は、感度解析で除く候補。
- 傷が写っていない視野はなかったので、「傷が写っていない」の理由の一覧は空。
"""
open(S/'【未読】v57_手順D_十字傷の検出と完全マスクと距離.md', 'w', encoding='utf-8').write(txt)
print(len(txt), 'weak lines', len(weak))
