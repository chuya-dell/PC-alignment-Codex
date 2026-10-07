"""Freeze incomplete attempts, record rejected calibration and exact resume point."""
from pathlib import Path
import ast,json,csv,io,datetime,shutil,sys
import field_motor_calibration as M
import pandas as pd,numpy as np
ROOT=M.ROOT;OUT=M.OUT
theme='帯状陽性と十字傷_原因切り分け'
vault=Path(r'W:\GoogleDrive\chuya2816\Obsidian Vault\ラボノート')
note=vault/'05_解析'/('261007_【未読】'+theme);note.mkdir(exist_ok=True)
logbytes=(ROOT/'data/results/v58_motor_marker_run.log').read_bytes()
lines=logbytes.decode('utf-16' if logbytes[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig',errors='replace').splitlines()
rows=[ast.literal_eval(x[5:]) for x in lines if x.startswith('PAIR ')]
marks=json.loads((OUT/'marker_landmarks.json').read_text(encoding='utf8'))
rows.append(dict(pair=11,pre=marks[-2]['name'],post=marks[-1]['name'],command_um=(marks[-1]['position_mm']-marks[-2]['position_mm'])*1000,
    dx_px=np.nan,dy_px=np.nan,distance_px=np.nan,marker_dx=marks[-1]['x']-marks[-2]['x'],marker_dy=marks[-1]['y']-marks[-2]['y'],fine_minus_marker_px=11.677098274230957,ecc=np.nan))
t=pd.DataFrame(rows);t['status']=['provisional']*(len(rows)-1)+['fine_registration_rejected'];t.to_csv(OUT/'marker_pairs_checkpoint.csv',index=False)
# Every successful-looking earlier calibration is explicitly superseded by a failure record.
shutil.copyfile(OUT/'calibration.json',OUT/'calibration_second_attempt_failed.json')
lat=json.loads((OUT/'calibration_initial_failed.json').read_text(encoding='utf-8-sig'))
stop=dict(status='incomplete_stopped_after_three_unsuccessful_calibration_attempts',magnification_guess='100x',pitch_px_provisional=lat['pitch_px'],
    nm_per_px_lattice_provisional=lat['nm_per_px_lattice'],nm_per_px_stage=None,comparison_63_1_vs_65_0='not_determined',
    backlash_um=None,stage_axis_deg=None,final_pair_fine_vs_marker_px=11.677098274230957,
    resume='Use sample cross-marker support and a constrained sub-period fine search; verify all 11 pairs before step3. Do not repeat whole-image phase/SIFT alignment dominated by camera-fixed patterns.')
(OUT/'calibration.json').write_text(json.dumps(stop,ensure_ascii=False,indent=2),encoding='utf8')
(ROOT/'data/results/v58_band_scar_causal/STATUS.json').write_text(json.dumps(stop,ensure_ascii=False,indent=2),encoding='utf8')
audit=pd.read_csv(ROOT/'data/results/v59_actual_centers_sampling/fit_audit.csv')
f=pd.read_csv(ROOT/'data/results/v59_actual_centers_sampling/selected_fields.csv')
med=float(audit.center_match_fraction.median());lo=float(audit.center_match_fraction.min());nlow=int((audit.center_match_fraction<.8).sum())
mincorr=float(audit.stored_vs_rerun_corr.min())
fig,ax=M.plt.subplots(1,3,figsize=(13,4));ax[0].plot(t.pair,t.dx_px,'o-',label='dx');ax[0].plot(t.pair,t.dy_px,'o-',label='dy');ax[0].legend();ax[0].set(xlabel='Pair (pair11 rejected)',ylabel='Provisional displacement (px)')
ax[1].plot(t.pair,t.fine_minus_marker_px,'o-');ax[1].axhline(7.3/2,c='r',ls='--');ax[1].set(xlabel='Acquisition pair',ylabel='Fine vs marker difference (px)')
ax[2].hist(audit.center_match_fraction,bins=15);ax[2].set(xlabel='Pre/post center correspondence fraction',ylabel='Fields')
fig.suptitle('INCOMPLETE / NOT VALIDATED / NO CAUSAL CONCLUSION');fig.tight_layout();fig.savefig(OUT/'stop_diagnostics.png',dpi=160);M.plt.close(fig)
header='---\ndate: 2026-10-07\n確認: 未確認\n状態: 未読\nstatus: 停止・未完了\n---\n\n【未確認】人工知能が出した途中結果。本人は未読。採否は本人が決める。\n\n'
final=header+f'''一言の結論（確度）：**決着せず（低）。手順0が未完了で、手順1・2は予備計算、手順3〜6は未実行。原因判定の条件は未達成。**

確かめたこと・限界：12枚の更新時刻順を確認。暫定格子周期 {lat['pitch_px']:.3f}画素から100倍相当、460 nm/周期なら {lat['nm_per_px_lattice']:.2f} nm/画素。ステージ由来nm/画素・63.1対65.0の判定・ねじの遊びは**未確定**。3回目も最後の組で粗合わせと細合わせが11.68画素不一致となり停止。手順1・2の90視野（固定29、同日程ブランク62、重複1）では、実中心対応率80%未満が{nlow}視野あり、対策効果を結論に使えない。

本人が決めること（1点）：v57の632視野の自動傷マスクは目視承認済みか。未承認の仮置きで、手順6は適用していない。採否は本人が決める。

途中で足した作業：カメラ上で固定された模様と標本の傷の移動を切り分ける診断、保存差分と再実行の照合。手順0完了前に手順1・2の予備計算を先行したため、順序どおりの達成として扱わない。

詳細リンク：[[【未読】途中結果と停止理由_{theme}]]。次の実験の暫定優先順：①液体処理なしで外して再載置し複数組、②各角度2枚以上の回転、③乾燥方法比較。まず既存像の移動量推定を検証する。
'''
(note/f'00_【未読】最終報告書_{theme}.md').write_text(final,encoding='utf8')
detail=header+f'''# 途中結果と停止理由

目的：(A)の帯状の陽性が「撮像・解析で作られるもの」か「液体処理(洗浄・乾燥など)で生じるもの」かを、独立した2つ以上の証拠で判定する。あわせて(B)の対策の効き目を数字で示す。

## 達成状態
|手順|狙い（依頼文の文）|状態|
|---|---|---|
|0|指令どおり動いていなければ、以降は実測のずれを使う。|12枚の撮影順確認、10組の暫定値。最後の細合わせを棄却。校正未確定。|
|1|縞が消えれば、原因は値の読み方で、同時に対策になる。|90視野の予備計算。中心対応不足により効果判定は不採用。|
|2|縞が読む位置に従って動く・消えるなら、読み方が縞を作っている証拠になる。|同じ90視野の16端数位置と補間版を計算。手順0完了前の予備計算であり、独立証拠として未検収。|
|3|液体処理なしで縞が出れば、撮像・解析が原因。端数で形が変われば、読み方が原因。|未実行。最後の組の不確かな実測ずれを使わない。|
|4|縞が出ない基準のはず。出れば撮像そのものが原因。|未実行。1.tif、2.tif、Mosaic.tifの実在確認済み。|
|5|ピントの小さな違いで縞が出るかを見る。|未実行。60〜90の31枚を確認（依頼は30枚と記載）。|
|6|十字傷の対策の効き目を数字で示す。|未実行。本人の目視承認状況を照会中。|

## 停止の根拠
初回：全画像の位相相関・特徴点の中央値が固定模様に引かれ、大刻みで約0画素の誤答を生んだ。
2回目：非格子像の通常相関と特徴点の合意法へ変更したが、固定模様の合意が標本の傷の移動を上書きした。
3回目：v57の傷直線の交点を粗い目印とし、その近傍で細合わせ。最後の組の細合わせは交点移動と11.677画素不一致。周期の半分3.65画素を超えたので棄却した。
**同じ移動量推定が3回続けて全組の検証に失敗したため停止。指令・実測の食い違いが停止理由ではない。4回目の再試行はしていない。**

## 手順0の暫定値（採用不可）
座標は像の内容がpreからpostへ動く向き、x右・y下。最後の組の細合わせは欠測にした。

|組|pre→post|指令 µm|dx画素|dy画素|細合わせ−傷目印 画素|
|---|---|---:|---:|---:|---:|
'''
for r in t.itertuples():detail+=f'|{r.pair}|{r.pre}→{r.post}|{r.command_um:.4f}|{r.dx_px:.4f}|{r.dy_px:.4f}|{r.fine_minus_marker_px:.4f}|\n'
detail+=f'''
追加Bへの取り出し用（**暫定・校正採用不可**）：
```json
{json.dumps(stop,ensure_ascii=False,indent=2)}
```

## 手順1・2の予備計算と弱点
90視野、固定29と同日程の主ブランク62（うち1視野が固定29と重複）。各方法で同日程ブランクから平均＋標準偏差3倍を再計算。差の向きは保存版と同じpre−post。
保存差分と再実行整数版の相関の最小値 {mincorr:.6f}。保存変換行列がなく再実行を用いたので、保存版との差を読取方法だけに帰属しない。
実中心の対応率は中央値 {med:.1%}、最小 {lo:.1%}、80%未満 {nlow}/90視野。欠測が局所的に偏るため、陽性率が下がっても読取対策の効果と結論できない。FFTの周波数を連続値で精密化する修正版は準備したが、依存パッケージ不足で実行に入らず停止。依存を任意扱いに直した後の再試行はしていない。
縞の指標は10月4日と同じ32画素升目、128〜1024画素周期、10度方向の最大パワー割合。比率だけでは振幅を表さないので密度標準偏差・帯域パワーも保存。未検証の図を原因の証拠には使わない。

![[stop_diagnostics.png]]
![[step1_comparison.png]]
![[step2_offsets.png]]

## 保存先と再開地点
コード：`{ROOT/'field_level/v58_motor_calibration'}`、`{ROOT/'pillar_level/v59_actual_centers_sampling'}`（各NOTES.md）。
手順0：`{OUT}` の入力一覧・指紋、marker_landmarks.json、marker_pairs_checkpoint.csv、calibration.json（停止状態、stageのnm/画素はnull）。初回・2回目校正値は名前にfailedを付けた記録であり不採用。
手順1・2：`{ROOT/'data/results/v59_actual_centers_sampling'}` のsamples/、audits/、fit_audit.csv、sampling_metrics.csv、thresholds_by_date_method.csv、3図と29枚の比較図。samples_fourier/・audits_fourier/は実行前で空。
実行ログ：`{ROOT/'data/results'}` のv58_motor_calibration_run.log、v58_motor_calibration_run2.log、v58_motor_marker_run.log、v59_sampling_run.log、v59_sampling_fourier_pilot.log。
再開：固定カメラ模様を位置合わせの支持にせず、傷の交点の小数位置と周期半分以内の探索を検証する。全11組の移動と残差図を確定してから、中心fitの欠測を解消、手順1→2→3→4→5→6の順で進む。生データを変更しない。

## Gitと検収
元クローンのgit pullは.git/FETCH_HEAD Permission denied、指定代替.git-tmp作成もPermission denied。生成物内の独立チェックアウトにv57タグから当作業専用ブランチを作った。元の未コミット変更は未操作。ネットワークはGitHubの443番へ接続できず、同期が未確認。コミット・タグ・pushの最終状態は周回記録を参照。
この結果は【未確認】のv57と10月4日の記録を参照した。Claude Code検収待ち。本人の結論採否待ち。Geminiは呼んでいない。
'''
(note/f'【未読】途中結果と停止理由_{theme}.md').write_text(detail,encoding='utf8')
for src in [OUT/'stop_diagnostics.png',ROOT/'data/results/v59_actual_centers_sampling/step1_comparison.png',ROOT/'data/results/v59_actual_centers_sampling/step2_offsets.png']:
    shutil.copyfile(src,note/src.name)
# Append only new provenance rows; leave all pre-existing dirty rows intact.
registry=ROOT/'data/raw/artifact_provenance_registry.csv';alt=ROOT/'data/results/v58_band_scar_causal/checkout/data/raw/artifact_provenance_registry.csv'
prov=[['codex-v58-v57-read-snapshot-261007','unknown','v57 false positive facts','2026-10-06','git tag v57_false_positive_facts_20261006','field_level','read_snapshot','unknown','','read_only',str(ROOT/'data/results/v58_band_scar_causal/checkout'),'Retrieved 2026-10-07; used stored inventory and fixed29; version superiority not compared.'],
['codex-v58-v28-metric-read-snapshot-261007','unknown','v28 band metric code','2026-10-04','git tag v28_band_origin_round1_20261004','field_level','read_snapshot','unknown','','read_only',str(ROOT/'data/results/v58_band_scar_causal/source_snapshots'),'Retrieved 2026-10-07; verified original metric definition; no version replacement.']]
for p in [registry,alt]:
    prior=p.read_text(encoding='utf-8-sig');buf=io.StringIO();writer=csv.writer(buf,lineterminator='\n')
    for row in prov:
        if row[0] not in prior:writer.writerow(row)
    if buf.getvalue():
        with p.open('ab') as out:out.write((('' if prior.endswith('\n') else '\n')+buf.getvalue()).encode('utf8'))
now=datetime.datetime.now().astimezone().isoformat()
rounds=list((vault/'10_引き継ぎ/周回記録').glob('*_Codex周回記録.md'));roundfile=max(rounds,key=lambda p:p.stat().st_mtime)
with roundfile.open('a',encoding='utf8') as out:out.write(f'\n## 停止区切り {now}\n手順0の全組検証が3試行連続で失敗。最終組の粗細不一致11.677画素。指令値との不一致ではなく実測の保証不足で停止。最後に終えた区切り：入力所在と撮影順・版番号確認。手順1/2の予備90視野は中心対応不足で効果判定不採用。手順3〜6未実行。傷マスクの目視承認状況の返答待ち。\n最終報告：{note}\n再開地点：全11組を標本の傷＋周期半分以内の細探索で検証してから手順1へ戻る。4回目試行なし。元クローン.git権限エラー、代替.git-tmp権限エラー、GitHubネットワーク接続不可。独立チェックアウトで自分の2版と自分の台帳2行のみコミット・タグを残す予定。\n記録の自己点検：達成条件未達、未読印・frontmatter・再開地点・途中追加作業・失敗ログ・欠測を記載。記録不足：最後の棄却行の細合わせdx/dyは例外前に保存されなかったため不一致距離のみ。以後のコードには失敗行の保存を追加（未再実行）。\n')
assert '11.677' in roundfile.read_text(encoding='utf-8-sig')
print('NOTE',note,'ROUND',roundfile,'median match',med,'below80',nlow,'mincorr',mincorr,flush=True)
for name in ['v58_motor_calibration','v59_actual_centers_sampling']:
    level='field_level' if name.startswith('v58') else 'pillar_level';p=ROOT/level/name/'NOTES.md'
    with p.open('a',encoding='utf8') as out:out.write(f'\n## 2026-10-07 停止状態\n依頼の3回連続失敗停止規則により未完了。全11組の手順0検証が通らず、最後の粗細不一致11.677画素。手順1/2予備90視野は中心対応80%未満が{nlow}視野で、効果の結論に不採用。最終報告と詳細ノート：{note}。保存校正calibration.jsonは停止状態へ変更し、ステージnm/画素・backlashは未確定。生画像・既存版は未変更。\n')
if __name__=='__main__':pass
