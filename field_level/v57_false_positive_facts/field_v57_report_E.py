import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np, json
import field_v57_common as C
from field_v57_md import md
E = C.OUT/'stepE'
meta = pd.read_csv(C.OUT/'stepB'/'B3_field_flags.csv', dtype={'date': str})[['fid', 'date', 'group', 'outlier29']]
bc = pd.read_csv(E/'bin_counts_by_field.csv').merge(meta, on='fid')
bc['set29'] = np.where(bc.outlier29, 'only29', 'non29')
def pooled(g, s, lo, hi, edge=False):
    x = bc[(bc.group == g) & (bc.set29 == s) & (bc.b >= lo) & (bc.b <= hi)]
    n, k = ('n_e300', 'kmean_e300') if edge else ('n_all', 'kmean_all')
    return x[k].sum()/x[n].sum(), int(x[n].sum())
bins = [('マスク内', -1, -1), ('0〜1周期', 0, 0), ('1〜2', 1, 1), ('2〜3', 2, 2), ('3〜5', 3, 4), ('5〜10', 5, 9), ('10〜20', 10, 19), ('20〜30', 20, 29), ('30〜50', 30, 49), ('50以遠', 50, 50)]
rows = []
for lab, lo, hi in bins:
    r = {'距離(傷のマスクの縁から)': lab}
    for g in ('blank', 'analyte', 'mismatch'):
        for s in ('non29', 'only29'):
            if g in ('blank', 'mismatch') and s == 'only29': continue
            rate, n = pooled(g, s, lo, hi)
            r[g + ('(29視野)' if s == 'only29' else '')] = rate
    rows.append(r)
prof = pd.DataFrame(rows)
cols = [c for c in prof.columns if c != '距離(傷のマスクの縁から)']
pr_md = md(prof, pct_cols=tuple(cols), pct_fmt='{:.2f}%')
inm = pd.read_csv(E/'E5_inside_outside_summary.csv')
ren = {'group': '群', 'fieldset': '視野の集合', 'rate_inside': '超過率:マスク内', 'rate_outside_0_5': '超過率:マスク外0〜5周期', 'rate_outside_ge20': '超過率:20周期以遠',
       'rate_outside_ge50': '超過率:50周期以遠', 'share_of_pillars_inside': 'ピラーのうちマスク内の割合', 'share_of_exceeding_pillars_inside': '超過ピラーのうちマスク内の割合'}
inm_md = md(inm.rename(columns=ren), pct_cols=tuple(v for k, v in ren.items() if k not in ('group', 'fieldset')))
e6 = pd.read_csv(E/'E6_inmask_vs_outside_by_edge_class.csv')
reg_map = {'in_mask': 'マスク内', 'out_0_5': 'マスク外0〜5周期', 'out_ge20': 'マスク外20周期以遠'}
e6t = e6.pivot_table(index=['group', 'region'], columns='edge_lo', values='rate').reset_index()
e6t.columns = ['群', '領域', '端から0〜100px', '100〜200px', '200〜300px', '300px以上']
e6n = e6.pivot_table(index=['group', 'region'], columns='edge_lo', values='n').reset_index(); e6n.columns = ['群', '領域', 'n0', 'n1', 'n2', 'n3']
e6t = e6t.merge(e6n, on=['群', '領域'])
e6t['300px以上のピラー数'] = e6t.n3.astype(int); e6t = e6t.drop(columns=['n0', 'n1', 'n2', 'n3'])
e6t['領域'] = e6t['領域'].map(reg_map)
e6_md = md(e6t, pct_cols=('端から0〜100px', '100〜200px', '200〜300px', '300px以上'))
att = pd.read_csv(E/'E1_attenuation_distance.csv')
a = att[(att.defn == 'mean') & (att.fieldset == 'non29')].copy()
a['edge300'] = a.edge300.map({False: '制限なし', True: '端から300px以上'})
a = a[['edge300', 'group', 'n_fields', 'line_i_baseline_upper', 'attenuation_i', 'attenuation_ii', 'rate_bin0_4_mean', 'rate_bin20_30_mean']]
a.columns = ['端の制限', '群', '視野数', 'ブランク基準線の上限(i)', '減衰距離(i)', '減衰距離(ii)', '0〜5周期の平均超過率', '20〜30周期の平均超過率']
att_md = md(a, pct_cols=('ブランク基準線の上限(i)', '0〜5周期の平均超過率', '20〜30周期の平均超過率'))
t = pd.read_csv(E/'E4_signflip_tests_all.csv', dtype={'stratum': str})
prim = t[(t.defn == 'mean') & (~t.near_edge300) & (t.fieldset == 'non29') & (t.near == 5) & (t.far == 20) & (t.stratum == 'ALL') & t.group.isin(['blank', 'analyte', 'mismatch'])].copy()
pren = {'group': '群', 'n_fields_used': '視野数', 'mean_near': '近傍(0〜5周期)の超過率', 'mean_far': '遠方(20周期超・端300px以上)の超過率', 'mean_diff': '差の平均(近傍−遠方)',
        'median_diff': '差の中央値', 'frac_pos': '差が正の視野の割合', 'p_signflip': '符号反転検定p(両側)', 'p_holm': 'ホルム補正p'}
prim_md = md(prim[list(pren)].rename(columns=pren), pct_cols=('近傍(0〜5周期)の超過率', '遠方(20周期超・端300px以上)の超過率', '差の平均(近傍−遠方)', '差の中央値', '差が正の視野の割合'), floatfmt='{:.2g}')
pd_ = t[(t.defn == 'mean') & (~t.near_edge300) & (t.fieldset == 'non29') & (t.near == 5) & (t.far == 20) & (t.stratum != 'ALL') & t.group.isin(['blank', 'analyte'])].copy()
dren = {'stratum': '日程', 'group': '群', 'n_fields_used': '視野数', 'mean_near': '近傍', 'mean_far': '遠方', 'mean_diff': '差の平均', 'p_signflip': 'p(未補正)', 'p_holm': 'ホルム補正p'}
pd_md = md(pd_[list(dren)].rename(columns=dren), pct_cols=('近傍', '遠方', '差の平均'), floatfmt='{:.3g}')
g = t[(t.defn == 'mean') & (t.stratum == 'ALL') & t.group.isin(['blank', 'analyte'])]
sens_rows = []
for (e3, fs, grp), x in g.groupby(['near_edge300', 'fieldset', 'group']):
    x = x[x.p_signflip.notna()]
    sens_rows.append({'近傍にも端300px以上を課す': 'はい' if e3 else 'いいえ', '外れ値29視野': '含む' if fs == 'with29' else '除く(主)', '群': grp,
                      '検定数': len(x), '差が負の検定数': int((x.mean_diff < 0).sum()), '差が正の検定数': int((x.mean_diff > 0).sum()),
                      'ホルム補正後p<0.05の検定数(差は負)': int(((x.p_holm < 0.05) & (x.mean_diff < 0)).sum()),
                      '差の平均の範囲(ポイント)': '%.2f〜%.2f' % (100*x.mean_diff.min(), 100*x.mean_diff.max()), '使えた視野数': '%d〜%d' % (x.n_fields_used.min(), x.n_fields_used.max())})
sens_md = md(pd.DataFrame(sens_rows))
glm = pd.read_csv(E/'E2_glm_near_vs_far_with_edge_adjustment.csv')
glm = glm[glm.fieldset == 'non29'][['group', 'model', 'n_fields', 'or_near', 'or_lo', 'or_hi', 'p']].rename(columns={'group': '群', 'model': 'モデル', 'n_fields': '視野数', 'or_near': '近傍のオッズ比', 'or_lo': '95%下限', 'or_hi': '95%上限'})
glm['モデル'] = glm['モデル'].map({'near_only': '近傍/遠方のみ', 'near+edge': '+端からの距離の区分', 'near+edge+date': '+日程'})
glm_md = md(glm, floatfmt='{:.3g}')
e2 = json.load(open(E/'E2_summary.json'))
e3 = pd.read_csv(E/'E3_distance_specific_blank_thresholds.csv', dtype={'date': str})
e3s = e3.groupby(['dist_lo', 'dist_hi']).agg(ブランクピラー数=('n_blank_pillars', 'sum'), 標準偏差の比の中央値=('sd_ratio_bin_over_global', 'median'), 比の最小=('sd_ratio_bin_over_global', 'min'),
                                           比の最大=('sd_ratio_bin_over_global', 'max'), 閾値の差の中央値=('diff_bin_minus_global', 'median'), 全域の閾値での超過率=('rate_above_global_thr', 'median'),
                                           遠方の閾値での超過率=('rate_above_far_thr', 'median')).reset_index()
e3s['距離(周期)'] = e3s.dist_lo.astype(int).astype(str) + '〜' + e3s.dist_hi.map(lambda v: '' if v > 1e8 else str(int(v)))
e3_md = md(e3s[['距離(周期)', 'ブランクピラー数', '標準偏差の比の中央値', '比の最小', '比の最大', '閾値の差の中央値', '全域の閾値での超過率', '遠方の閾値での超過率']], floatfmt='{:.3g}', pct_cols=('全域の閾値での超過率', '遠方の閾値での超過率'))
e3f = e3[e3.dist_lo == 30][['date', 'thr_global_all_blank', 'thr_far_only']].copy(); e3f['遠方/全域'] = e3f.thr_far_only/e3f.thr_global_all_blank
e3f.columns = ['日程', '全域(全ブランクピラー)の閾値', '遠方のみの閾値', '遠方/全域']
e3f_md = md(e3f, floatfmt='{:.3g}')
fn = e2['fields_with_a_scar_line_within_300px_of_edge']; nd = e2['n_detected']; sh = e2['near_zone_pillars_share_edge_lt300_pooled']*100; nz = e2['fields_with_near_zone_ge_300_pillars_edge_ge_300']
txt = f"""---
確認: 未確認
状態: 未読
date: 2026-10-06
---

【未確認】人工知能(Claude Code)が自動で回して出した結果。本人は未読。採否は本人が決める。この結果は、【未確認】の手順0・D(保存値の読み方、傷の検出)に基づく。既存の保存値と洗浄前の生画像だけを使い、再実行はしていない。外れ値の定義・検定の族・近傍/遠方の定義は、距離別の結果を見る前に固定したもの(`step0/PROTOCOL_AND_PREDICTIONS.md`、指紋 1602f3c8…a6cf2)。

# v57 手順E:距離別プロファイルと主検定

目的:擬陽性が出る理由の解明と対策の比較。この手順が効く点:「傷の付近に陽性が集中する」を、距離の関数として数値で確かめ、集中が及ぶ範囲(減衰距離)と、画像の端との交絡を切り分ける。

## 先に結論(事実)

1. **傷のマスクの内側に、擬陽性が集中する。マスクの外側では、傷に近いほど低い。** マスク内(ピラーの約10%)の超過率は、端から300px未満のクラスで2.0〜3.4%。同じ端クラスのマスク外は0.03〜0.41%(下の表E6。比は約5〜100倍)。超過ピラーのうちマスク内にあるのは、ブランク50%・分子あり65%(外れ値29視野を除く)。マスクの縁を越えると、ブランクの超過率は1〜2周期で0.16%→0.05%へ落ちる。
2. **本人の観察(傷付近に陽性が集中)は、マスクの内側については数値で確認できた。** ただし事前予測1(近傍=マスク縁から5周期以内の超過率が遠方より高い)は**外れた。逆で、近傍のほうが低い**(ブランク 0.06% 対 0.43%、符号反転検定 p<10⁻⁵。分子あり 0.12% 対 0.24%、p=6×10⁻⁵。視野単位)。
3. **減衰距離は、マスクの縁(0周期)。** マスクの縁の外側では、超過率はブランクの基準線の上限(0.5〜0.9%)以下で、距離とともに下がらず、むしろ中央に向かってわずかに上がる。マスクの縁から数周期の追加の除外を支持する数値はない(第3周で改めて比べる)。
4. **外れ値29視野の高い超過率は、傷の近傍ではなく、傷から遠い領域にある**(50周期以遠で5.4〜6.1%。超過ピラーのうちマスク内は7〜23%のみ)。帯状・うろこ状の外れ値は、傷とは別の現象と読める。
5. 画像の端との交絡:傷の線が画像の端から300ピクセル以内にある視野は{fn}/{nd}。近傍のピラーの{sh:.1f}%が端から300ピクセル以内。端のクラスを揃えても(マスク内と外を同じ端クラスで比較)、マスク内の高い超過率は残る。

## 事前予測との照合(書き換えない)

| 事前予測 | 結果 |
|---|---|
| 1. 近傍(5周期以内)の超過率は遠方より高い(確度:中) | **外れた(逆)。** 近傍0.06%/0.12%(ブランク/分子あり)<遠方0.43%/0.24%。 |
| 2. 近傍の集中は、ブランクにも出る(確度:中) | **集中が出なかった**ため、「ブランクにも出る」は成り立たない。ブランクでも近傍<遠方で、逆転の差が最大(ブランク −0.36 ポイント)。マスク内の高い超過率(2.9%)は、ブランクにも分子ありにも同程度に出た(予測の趣旨とは別の所に出た)。 |
| 3. 近傍の集中は、画像の端からの距離を調整しても残る(確度:低) | **前提(集中)がなく、検証できない。** 端の調整後も近傍<遠方は残る(ブランクのオッズ比0.26、分子ありは0.71で有意でない)。マスク内の高い超過率は、端クラスを揃えても残る。 |

## 1 距離別プロファイル(E1)

超過=日程別ブランクの平均+3標準偏差を厳密に超える(主定義)。外れ値29視野を除く(主)と、29視野のみ(記述)を別に出した。区間内の全ピラーでプールした超過率。距離=傷のマスク(傷の領域+両側1周期)の縁から。基準線の上限(ブランク、日程別、ブロック再標本化99%)は 0.5〜0.9%。

{pr_md}

図:`stepE/fig_E1_profile_pooled.png`、日程別 `fig_E1_profile_by_date.png`、符号付き距離(マスクの内側の深さを含む) `fig_E5_signed_distance_profile.png`。表:`E1_profile_by_distance.csv`(視野平均、端の制限あり/なし、中央基準の感度つき)、`E5_signed_distance_profile.csv`。

マスクの内側と外側(外れ値29視野を除く群と、29視野のみ):

{inm_md}

**減衰距離**(超過率が、その距離以遠のすべての区間で基準線以下になる最小の距離。30周期まで。(i)=ブランク基準線の上限=日程別99%点の視野平均、(ii)=ブランクの区間別超過率の視野間の95%上側信頼限界。空欄=30周期までに戻らない):

{att_md}

(i)ではどの群も0周期(マスクの縁)。(ii)で分子ありが戻らない(または5周期)のは、遠方で分子ありの超過率がブランクの上側信頼限界をわずかに上回る区間があるため(差は0.1ポイント未満)。

## 2 画像の端との交絡(E2)

- 傷の線(の位置)が画像の端から300ピクセル以内にある視野:{fn}/{nd}。近傍(マスク縁から0〜5周期)のピラーのうち、画像の端から300ピクセル以内にあるもの:{sh:.1f}%(視野別の中央値 100%)。端から300ピクセル以上の近傍ピラーが300本以上ある視野は {nz} 視野のみ。傷の近傍と画像の端の近傍は、ほとんど区別できない。
- 同じ画像の端クラスの中で、マスク内・マスク外(0〜5周期)・マスク外(20周期以遠)の超過率を比べた(外れ値29視野を除く):

{e6_md}

  マスク内は、画像の端から300px未満のどのクラスでも1.2〜3.4%(端から300px以上は、ピラー数が少なくばらつく)で、同じ端クラスのマスク外より約5〜100倍高い。**マスク内の高い超過率は、画像の端の効果では説明できない**(確度:高)。マスク外では、超過率は端に近いほど低く、中央(端から500px以上)ほど高い(ブランク0.22→0.56%、分子あり0.17→0.28%。`E2_rate_by_distclass_and_edgeclass.csv`)。
- 近傍/遠方を、端の区分で調整した二項回帰(視野でクラスター化した頑健標準誤差。外れ値29視野を除く):

{glm_md}

## 3 距離別のブランク閾値(E3)

ブランクのピラーだけで、距離の区間ごとに閾値(平均+3標準偏差)を作った。マスク内のピラーは除く。標準偏差の比=その区間÷全域(全ブランクピラー)。日程別の9値の中央値と範囲:

{e3_md}

傷に近い区間(0〜30周期)の標準偏差は、全域の0.6〜0.8倍で、30周期以遠(0.94倍)より小さい。したがって、傷から離れたピラーだけを解析するときは、閾値と基準線も同じ領域で作り直す必要がある(第3周)。30周期以遠のブランクだけで作った閾値(遠方のみ)と、全域の閾値(日程別):

{e3f_md}

(全域の閾値はマスク内のピラーを含み、広がりが大きいため、遠方のみの閾値より高い。)

## 4 主検定(E4):各視野の「近傍の超過率−遠方の超過率」の符号反転検定

視野単位。近傍=マスク縁から0〜5周期、遠方=20周期超かつ画像の端から300ピクセル以上。各領域のピラーが300本未満の視野は除く(主設定では除外0)。外れ値29視野は除いた(主)。符号反転は10万〜20万回(主設定)、両側。ホルム補正は、ブランクと分子ありの2検定を族とした。

{prim_md}

結論:**近傍の超過率は遠方より高くない。逆に低い(視野単位の符号反転検定、ホルム補正後 p<0.05)。「近傍が高い」は支持されない。** (ミスマッチは族に入れていない。)

日程別(ブランク・分子あり×9日程、18検定を別の族としてホルム補正):

{pd_md}

感度(近傍3・5・8周期×遠方15・20・30周期の9通り。ブランクと分子ありの各9検定を族としてホルム補正):

{sens_md}

近傍にも「端から300px以上」を課すと、使える視野が減る(ブランク4〜21視野)が、差の向きは同じ。外れ値29視野を含めても向きは同じ。**どの設定でも、近傍が遠方より有意に高い検定は1つもない。**

## 出せなかったもの・限界

- マスク(傷の領域)の広さは、画像のエネルギーの閾値で決めた(仮置き)。「マスク内で超過が集中する」という結論は、マスクの縁の定義に依存する。マスクを狭めた版は、本周では作っていない。
- 傷の検出は目視承認を取っていない(仮置き)。弱い検出の線と、マスク外に傷らしい成分が残る9視野の扱いは、手順Dに記載。
- 「照明の不均一」(本人の言葉の読み替え)は検定していない。ただし、マスク外では超過率と標準偏差が、傷から遠い(画像の中央側)ほど高いという構造が出た(第2周の候補)。
- 距離別プロファイルの区間ごとの標準誤差は、視野の相関を考慮していない記述値。検定は視野単位(符号反転)で行った。
"""
open(E/'【未読】v57_手順E_距離別プロファイルと主検定.md', 'w', encoding='utf-8').write(txt)
print(len(txt))
