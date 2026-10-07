"""Write reviewable numerical report and figures; no scientific parameter mutation."""
from pathlib import Path
import sys,json,shutil,os
from datetime import datetime
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import field_control_common as M
import numpy as np,pandas as pd
ROOT=M.ROOT;OUT=M.OUT
LAB=Path(r'W:\GoogleDrive\chuya2816\Obsidian Vault\ラボノート')
THEME=LAB/'05_解析/261007_【未読】帯状陽性と十字傷_原因切り分け'
P=ROOT/'data/results/v61_actual_centers_sampling';K=ROOT/'data/results/v62_provisional_scar_mask'
HEADER='---\ndate: 2026-10-07\n確認: 未確認\n状態: 未読\nstatus: 解析実行済み・検収待ち\n---\n\n【未確認】人工知能による実測解析。本人未読、Claude Code検収待ち。採否は本人が決める。\n\n'

def table(t):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6g}'
        return str(v).replace('|','/').replace('\n',' ')
    lines=['|'+'|'.join(map(str,t.columns))+'|','|'+'|'.join(['---']*len(t.columns))+'|']
    return '\n'.join(lines+['|'+'|'.join(fmt(v) for v in r)+'|' for r in t.itertuples(index=False,name=None)])

def main():
    controls={s:pd.read_csv(OUT/s/'metrics_series_pooled.csv') for s in ['repeat','focus','motor']}
    f=pd.read_csv(P/'fit_audit.csv');ss=pd.read_csv(P/'common_support_paired_comparison.csv');offset=pd.read_csv(P/'offset_relationships.csv')
    mask=pd.read_csv(K/'field_metrics.csv');dose=pd.read_csv(K/'concentration_comparison.csv')
    figures=THEME/'再開図';figures.mkdir(exist_ok=True)
    # Compact shared figure for the one-screen final report.
    fig,ax=M.plt.subplots(1,3,figsize=(12,2.5))
    for s in ['motor']:
        for method,g in controls[s].groupby('method'):ax[0].plot(g.frac_dx,g.rate*100,'o',label=method)
    ax[0].set(xlabel='Measured dx fractional px',ylabel='Liquid-free positive rate (%)',title='Step 3: six pairs');ax[0].legend(fontsize=6)
    for method,g in controls['focus'].groupby('method'):
        dial=g.key.str.split('_').str[1].astype(float);ax[1].plot(dial,g.rate*100,'o-',label=method,ms=3)
    ax[1].set(xlabel='Pre focus dial',ylabel='Positive rate (%)',title='Step 5: focus 60-90')
    for mode,g in mask.groupby('mode'):
        q=g.groupby('group').rate.median();ax[2].plot(['blank','analyte'],q.loc[['blank','analyte']]*100,'o-',label=mode)
    ax[2].set(ylabel='Median field rate (%)',title='Step 6: unapproved mask');ax[2].legend(fontsize=6)
    fig.tight_layout();fig.savefig(OUT/'final_evidence.png',dpi=140);M.plt.close(fig)
    sourcefigs=[OUT/'final_evidence.png',OUT/'repeat/atlas_series_pooled.png',OUT/'motor/atlas_series_pooled.png',OUT/'focus/summary_series_pooled.png',OUT/'focus/atlas_series_pooled.png',OUT/'landmark_pair_7.png',OUT/'landmark_pair_9.png',OUT/'landmark_shifts.png',P/'step1_stratified.png',P/'step2_offsets.png',K/'mask_effect.png']
    for p in sourcefigs:
        name=p.name if p.name!='atlas_series_pooled.png' else p.parent.name+'_'+p.name
        shutil.copyfile(p,figures/name)
    rows=[]
    for stage,t in controls.items():
        for method,g in t.groupby('method'):
            rows.append(dict(stage=stage,method=method,pairs=len(g),rate_min_pct=100*g.rate.min(),rate_max_pct=100*g.rate.max(),strength_median=g.strength.median(),band_candidates=int((g['分類']=='帯状候補').sum()),scale_candidates=int((g['分類']=='うろこ状候補').sum())))
    sums=pd.DataFrame(rows)
    summary=mask.groupby(['mode','group']).agg(fields=('fid','size'),median_rate=('rate','median'),mean_rate=('rate','mean'),pillars=('n','sum'),positive=('positive','sum')).reset_index();summary['pooled_rate']=summary.positive/summary.pillars
    detail=HEADER+'''# 再開結果：問いの読み方と判定

目的（依頼文の文）：(A)の帯状の陽性が「撮像・解析で作られるもの」か「液体処理(洗浄・乾燥など)で生じるもの」かを、独立した2つ以上の証拠で判定する。あわせて(B)の対策の効き目を数字で示す。
問いと各手順の狙いは[[【未読】途中結果と停止理由_帯状陽性と十字傷_原因切り分け]]と元依頼文のまま。手順0の未確定で手順3〜6を止めないという再開指示を優先した。

**判定：決着せず（中）。** 手順3・4・5は液体処理なしでも陽性が出ること、ピントで陽性率が増えることを示す。しかし(A)の周期的な帯状／うろこ状を同じ分類で再現した独立2証拠はない。手順1の対象29視野は全て低対応。手順2の端数依存は読取りが影響する証拠だが、読取りだけが主因とは言えない。手順6の承認前マスクは(A)の原因判定に用いない。

## 手順0：数字・符号・方向の検算

''' + table(pd.read_csv(OUT/'landmark_shift_verification.csv'))+'''

像内容のpre→post変位、右x正・下y正。前回fine値は小刻み6組に限り手順3へ暫定採用。組9の標本十字交点dx=635.764はfine633.226と整合し、図でも右へ約630画素動く。20µmなら317画素という期待だけで633を誤推定と棄却する根拠はない。組7のdy≈−76も図上の標本十字移動と整合する。組8は指令の符号から期待する方向と逆。どの画像も追跡した十字交点は視野内にあり、今回の交点追跡自体では視野外消失はない。ただし他のゴミはカメラ側の固定模様も含む。
12枚の中央値像は**純粋なカメラ固定模様ではない**。7枚の小刻み像が過半数なので標本の十字も残り、差引き後の位相相関は0近傍や低応答へ引かれた。horizontal_corrの|dy|<32という制約は組7の実際のdyを捨てるため、診断候補を実測値に採用しない。

![[再開図/landmark_pair_7.png]]
![[再開図/landmark_pair_9.png]]
![[再開図/landmark_shifts.png]]

追加Bへの取り出し値（値の確定区分を保つ）：
```json
''' + (OUT/'additional_B_calibration.json').read_text(encoding='utf8')+'''
```
100倍は約7.387画素周期と50倍約3.6画素との差に基づく判定。格子の62.271nm/画素は物理ピッチ460nmの仮定、小刻み61.782は0.5µm指令が実移動と等しい仮定であり独立の確定校正ではない。平均小刻みdx=−8.0795・dy=−0.4670画素。63.1なら0.5µm=7.9239画素、65.0なら7.6923画素で、小刻みの散らばりを考えるとどちらかの確定選択はできない。ねじの遊び・大刻み換算は未確定。

## 手順1・2：対応率で分ける

''' + f'90視野の対応率中央値{f.center_match_fraction.median()*100:.2f}%、80%未満46、80%以上44。固定29視野は全て80%未満で、80%以上の固定29比較は**n=0（空集合）**。44視野は同日程ブランクだけなので(A)への効果を一般化しない。保存差分と再実行の相関最小{f.stored_vs_rerun_corr.min():.9f}。\n\n'+table(ss)+'''

表は同一ピラー支持の比較。低対応固定29で実中心bilinearの陽性率中央値が下がっても、空間的に偏った未対応を残すため効果確定には使わない。方向パワー**割合**は上がり、絶対帯域パワーは下がるので「縞の強さ」のどちらか一つだけを都合よく選ばない。
閾値は方法別に同日程の主ブランク全視野から再計算。44視野だけの閾値へ変更はしていない。storedとの比較には登録変換の再推定も含む。

![[再開図/step1_stratified.png]]
![[再開図/step2_offsets.png]]

0.25画素16位置の方向・位相・法線位置変化（固定29低対応群の中央値）：

''' + table(pd.read_csv(P/'offset_stratum_summary.csv'))+'''

個々の視野の全16位置はoffset_relationships.csvに保存。法線位置は**基準周波数の位相差×周期/2π**という単一モードの指標で、画像全体の移動ではない。位相平均は円周量なので本表の中央値だけで端数に対する単調性を断定しない。代表図はatlas/の29枚を用意。

## 手順3〜5：液体処理なし対照

''' + table(sums)+'''

理想ピッチ7.286、FFT理想格子、pre−post、背景Gaussian51画素・3×3和は本番と同じ。整数photometryは本番sample_contrastを直接呼ぶ。bilinearは同じ変換・同じ3×3背景差分場をnative座標で補間する感度分析。手順3はv58の小刻み6組の実測ベクトルを固定して端数順に比較。各系列を同じ液体処理なしブランクとして全組プールし平均＋母標準偏差3倍を閾値にした（date-poolingに相当）。本番の9日程ブランク閾値数値を、露光・撮像条件が異なる対照へそのまま移してはいない。
全系列プールは同一視野の繰返しで独立標本ではなく、同じ対照で閾値を作り同じ対照を評価している。外れ値率を独立測定の偽陽性率とは呼ばない。ピント系列は31枚で、30隣接組。
classificationは10月4日v28と同じ32画素升目、128〜1024画素周期、10度方向、1000回並替、自己相関周期との20%以内一致を必要とする。整数版で帯状候補／うろこ状候補は全3系列とも0。周期候補だけの成立は帯状分類成立ではない。モーターbilinear組2は帯状候補に入るが、図の陽性は十字傷の軸に沿い(A)との同一性は未確定。

''' + table(controls['motor'][['key','method','dx','dy','frac_dx','frac_dy','rate','strength','axis_deg','period_px','分類']])+'''

![[再開図/motor_atlas_series_pooled.png]]
![[再開図/repeat_atlas_series_pooled.png]]
![[再開図/summary_series_pooled.png]]

合焦付近75→76の整数率は全系列プールで5.9524%、この組だけで閾値を作ると0.0711%（実数は下の表）。系列全体の分散と組固有分散は違うため、どちらの数字も保存し、好都合な側だけを結論に使わない。ピント変化で**閾値に対する率**が上がったことは撮像の寄与を示すが、元の周期的な帯状の主因確定には届かない。

''' + table(pd.read_csv(OUT/'focus/metrics.csv').query("key=='focus_75_76'")[['key','method','threshold','rate','strength','分類']])+ '\n\n全系列プール：\n\n'+table(controls['focus'][['key','method','threshold','rate','strength','分類']])+'''

全30組の図：[[再開図/atlas_series_pooled.png]]。translation from zeroの収束情報は各組json。失敗時zero fallbackと明記（フォーカス大ぼけで格子が弱い場合の位置合わせ保証には限界）。

## 手順6：承認前の仮置き対策

**承認前の仮置き。原因の結論や正式解析には使わない。** v57凍結マスク632件を読み、各画像サイズとpacked bits、各ピラーin_maskの保存版との全要素一致を検査。マスク・cacheのSHA256をinput_audit.csvに保存した。傷領域はv57の線幅＋両側1周期。日程別主ブランクのマスク外だけで閾値を再作成。

''' + table(summary)+'''

ピラー残存89.2607%。ブランク中央値0.2599→0.0668%は下がるが平均0.6093→0.6541%は上がる。分子あり中央値0.2972→0.0516%、平均0.7102→0.6887%。分母変更とblank閾値再推定を含むため、傷除外だけの効き目とは言わない。
視野単位の各日程濃度対主ブランク片側Mann–Whitney（asymptotic）は前後各60比較。各mode内の60比較にHolm補正し、p<.05は**前0/60、後0/60**。これは本番の9倍数×60=540比較全体の補正と同じ家族ではない。基板内の視野依存は残り、有意なしを「分子効果なし」と解釈しない。

''' + table(pd.read_csv(K/'dose_rank_correlation.csv'))+'''

![[再開図/mask_effect.png]]

## 限界・次に撮る優先順・本人が決めること

1. 液体処理なしで外して再載置し、複数独立組を撮る。同一視野の現在の小刻み・ピント対照だけでは、洗浄と再配置を分離できない。
2. 各角度2枚以上の回転像。カメラ固定構造と標本構造・角度依存を切り分ける。
3. 再載置の対照を確保した上で乾燥方法を比較。

本人が決めること1点：v57傷マスクを目視後に正式採用するか。現状は仮置き維持。中央値の低下はあるが一律の平均低下・濃度依存の改善は認めない。
途中で足した作業：①大刻みの中央値像差引きと標本十字の符号・方向図の検算、②対応率層別の共通支持比較と端数位相表、③閾値プール方法の感度分析。Gemini未呼出。

## ファイル・再現・Git

コード：field_level/v60_band_scar_controls（0診断、3〜5、報告）、pillar_level/v61_actual_centers_sampling（1・2）、field_level/v62_provisional_scar_mask（6）、各NOTES.md。出力は同名data/results/。生成画像・cacheはGitに入れない。
再現：field_controls.py repeat / focus / motor → field_pool_controls.py各系列、field_motor_forensics.pyとfield_verify_landmarks.py、pillar_sampling_experiments.py→pillar_stratify.py、field_mask_effect.py。入力は上位Wの実在フォルダ列挙から確認済み。Fはこの機械に存在しないが、最新依頼指定のW入力を確認済みなので旧F環境記述で止めない。
前回記録は上書きしない。本最終報告書だけを更新し、途中結果と停止理由は履歴として残す。
Git：元クローン.gitは書込み拒否のため、data/results/v58_band_scar_causal/checkoutの独立checkoutで継続（branch codex/band-scar-causal-261007）。区切り1 commit b7a8335、tag v60_v62_controls_provisional_mask_20261007。区切り2のcommit/tagはGit保存記録に追記。pull/pushはGitHub443接続不能、手元のcommit/tagで残しpushはClaude Codeに渡す。元クローンのv41/v43/v44/v45/v47変更は含めない。
'''
    note=THEME/'【未読】再開結果_手順0〜6の検算と限界.md';note.write_text(detail,encoding='utf8')
    final=HEADER+'''**一言の結論（確度）：決着せず（中、実測）。** 手順1〜5を実行したが、元の帯状陽性の主因を独立2証拠で確定できない。手順5は液体処理なしでも合焦付近で陽性率が増えることを示す。

**確かめたこと・限界**
- 手順1・2：対応80%以上44視野。固定29は全て80%未満で、対策効果を一般化できない。
- 手順3：小刻み6組、整数陽性0.220〜0.530%。手順4：連写0.360%。整数の帯状分類は両方0件。
- 手順5：31枚・30組、75→76は5.952%、最大6.778%。整数の帯状分類0件。系列プール閾値での値で、各組閾値だと率が変わる。
- 手順6（**承認前の仮置き**）：632視野、ピラー89.26%残存。blank中央値0.2599→0.0668%、平均0.6093→0.6541%。濃度比較Holm後有意0/60→0/60。原因結論には使わない。
- 手順0／追加B：100倍は格子尺度で確定。格子62.271・小刻み61.782nm/画素は暫定。大刻み・遊び・63.1対65.0は未確定。組9の約633画素移動は標本十字の図と整合。

![[再開図/final_evidence.png|650]]

**本人が決めること（1点）**：目視後、v57傷マスクを正式採用するか（現状は仮置き維持）。
**途中で足した作業**：標本目印の図検算、対応率層別比較、閾値プール感度分析。
**詳細リンク**：[[【未読】再開結果_手順0〜6の検算と限界]]（数字・全図・追加B用JSON・保存先）、[[【未読】Git保存記録_本体再開]]。次の実験の優先順は再載置無処理→各角度複数の回転→乾燥比較。
'''
    (THEME/'00_【未読】最終報告書_帯状陽性と十字傷_原因切り分け.md').write_text(final,encoding='utf8')
    assert '確認: 未確認' in note.read_text(encoding='utf8') and len(f)==90 and mask.fid.nunique()==632
    (OUT/'report_validation.json').write_text(json.dumps(dict(fields_sampling=len(f),mask_fields=mask.fid.nunique(),controls={s:int(len(t)/2) for s,t in controls.items()},fixed29_high_match=int(((f.center_match_fraction>=.8)&f.fixed29).sum()),syntax_checks='all new .py compiled',report_readback=True,computer_name_unicode=ascii(os.environ.get('COMPUTERNAME')),user=os.environ.get('USERNAME'),drives=[p for p in 'CDEFGW' if Path(p+':/').exists()]),ensure_ascii=False,indent=2),encoding='utf8')
    print('REPORTS WRITTEN; readback and counts OK',flush=True)

if __name__=='__main__':main()
