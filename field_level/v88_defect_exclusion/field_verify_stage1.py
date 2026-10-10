"""Codexが保存表を検収する：事後の再評価（独立な検証ではない）。"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/results/v88_defect_exclusion'
SRC = Path(r'C:\Users\chuya\PC-alignment-fp\data\results')
LABEL = '事後の再評価（独立な検証ではない）'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (OUT/'verification.json').exists(), 'Codexは既存検収記録を上書きしない。'
    t = pd.read_csv(OUT/'blank_counts_by_k.csv')
    c = pd.read_csv(OUT/'calibration_k.csv')
    ref = pd.read_csv(SRC/'v78_candidate_path/blank_fp_by_k.csv')
    q = t[(t.radius==0)&(t.side=='positive')].merge(ref[ref.readout=='S5'],on=['fid','k'],suffixes=('_new','_old'))
    assert len(q)==1725 and np.array_equal(q.fp91k_new.to_numpy(),q.fp91k_old.to_numpy())
    assert len(t)==69*6*3*25
    assert (t.groupby(['radius','side','k']).size()==69).all()
    for r in c.itertuples():
        p=t[(t.radius==r.radius)&(t.side==r.side)].groupby('k').fp91k.quantile(.95)
        passing=p[p<=2]
        assert r.reached == bool(len(passing))
        assert r.k == (float(passing.index[0]) if len(passing) else 15.)
    audit=pd.read_csv(OUT/'array_read_audit.csv')
    assert len(audit)==547*2+69*3
    assert not audit.path.str.contains('261008|v83_new_shots').any()
    for p in OUT.glob('*.csv'):
        assert pd.read_csv(p).evaluation.eq(LABEL).all(), p
    result=dict(evaluation=LABEL, baseline_compared_rows=len(q),baseline_max_difference=0., count_rows=len(t),array_reads=len(audit),calibrations_checked=len(c),prohibited_array_reads=0,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__)
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    notes=ROOT/'field_level/v88_defect_exclusion/NOTES.md'
    with notes.open('a',encoding='utf-8') as f:
        f.write('''
## 2026-10-10 第一段階の結果：事後の再評価（独立な検証ではない）
Codexは開発用非ブランク547視野とブランク69視野を計算した。Codexは全日程から13中心を再推定し、固定表との差を最大0.000000000001画素未満と確認した。
Codexは前半5日から10中心、後半4日から12中心を得た。Codexは両群で9/13中心を5画素以内に再現し、固定基準10/13を満たさなかった（反証側の確度の上限は中）。Codexは中心8・40・157・309の片群で再現条件を満たさなかった。
Codexは除外なしの倍率を正側10.5・負側11.5・両側13.0、半径20の倍率を正側9.0・負側9.5・両側10.0と校正した。Codexは全18条件で上限15以内の到達を確認した。
Codexは半径20の実個数の95%点を正側1.0・負側1.0・両側1.6、9.1万ピラー換算の95%点を順に1.332036・1.384590・1.937400と算出した。Codexは各側の実個数の最大を10・4・9と算出したため、95%点の合格を全視野の偽陽性ゼロとは扱わない。
Codexは半径10・15・20・25・30の全側で除外なしの共通倍率における95%点の悪化を認めなかった。Codexは半径20の有効ピラーの損失を最大0.671830%と算出した。
Codexは過去の3集計表を発見し、既存数値だけを履歴表に保存した。Codexは過去の24視野の両側・倍率10.5の平均を前像半径40と後ろ像半径20の双方で1.041667→0.208333と読んだ。Codexは過去の異常ピラー数を104、同じ後ろ像の比較で重複した数を98と読んだ。
Codexは `data/results/v88_defect_exclusion/` に日程分割、中心再現、半径感度、閾値校正、履歴比較、読み取り監査、検収記録を保存した。Codexは除外なしの正側1,725行が旧版と数値差0で一致することを検収した。
Codexは13中心の日程分割の不合格と目視承認前の欠陥表を理由に、標準経路への採用を行わない。Codexは有効側の主張の確度の上限を低とする。
Codexは24視野の画像・差分配列、無作為除外、座標移動、信号注入、日程を抜く交差検証を今回実行しなかった。Codexは保存表から開発用と24視野の識別子の共通部分0を確認した。
Codexは現在の書き込み許可がこのクローン内に限られるため、Vaultの結果・進捗・引き継ぎへ書き込めなかった。Codexは本記録と結果報告書に再開地点を残した。
''')
    # Codexは自己参照を避け、指紋追記前のNOTES全文を固定する。
    snapshot=OUT/'NOTES_before_hash_append.md'
    snapshot.write_bytes(notes.read_bytes())
    digest_rows=[]
    for p in [ROOT/'data/raw/v88_camera_defects_unapproved.csv',snapshot,OUT/'frozen_k.json']:
        digest_rows.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
    with notes.open('a',encoding='utf-8') as f:
        f.write('\n## 2026-10-10 指紋の固定：事後の再評価（独立な検証ではない）\n')
        f.write('Codexは自分の指紋を自身に埋め込む自己参照を避け、指紋追記直前のNOTES全文を `NOTES_before_hash_append.md` に固定した。Codexは追記後のNOTES全体の指紋を `sha256_manifest.csv` に保存する。\n')
        for r in digest_rows:
            f.write(f"Codexが固定した `{r['path']}` のSHA-256値は `{r['sha256']}` である。\n")
        f.write('CodexはGitへの保存手順を `git_save_procedure.md` に用意したが、履歴への保存・送信・枝切り替えを行っていない。\n')
    report='''# 【未読】第一段階の結果：事後の再評価（独立な検証ではない）

Codexは日程分割で9/13中心を両群で再現し、基準10以上に届かなかった。Codexは有効側の確度の上限を低、反証側の上限を中とする。白石さんは本結果をまだ確認していない。

Codexが半径20画素・69ブランク視野で得た校正結果は次の表である。

| 事後の再評価（独立な検証ではない） | 正側（後ろ像の減光） | 負側（後ろ像の増光） | 両側 |
|---|---:|---:|---:|
| 除外なしの倍率 | 10.5 | 11.5 | 13.0 |
| 半径20の倍率 | 9.0 | 9.5 | 10.0 |
| 実個数の95%点 | 1.0 | 1.0 | 1.6 |
| 9.1万ピラー換算の95%点 | 1.332036 | 1.384590 | 1.937400 |
| 実個数の平均 | 0.231884 | 0.173913 | 0.260870 |
| 実個数の最大 | 10 | 4 | 9 |

Codexは半径10・15・20・25・30の全側で、除外なしで固定した倍率における95%点の悪化を認めなかった。Codexは半径20の有効ピラー損失を最大0.671830%と算出した。
Codexは前半5日から10中心、後半4日から12中心を再推定した。Codexは中心8・40・157・309について片群の5画素以内の再現がないことを確認した。Codexは全547視野から元の13中心を数値誤差内で再現した。
Codexは過去の3集計表を発見した。Codexは載せ直し24視野の両側・倍率10.5の平均について、前像半径40と後ろ像半径20の双方で除外前1.041667・除外後0.208333と読んだ。Codexは再現性の過去表から異常ピラー104個と同じ後ろ像での重複98個を読んだ。Codexは過去の配列を再計算していない。

Codexが書いたファイルと内容は次の一覧である（各パスはこのクローンからの相対パスである）。

- Codexが `field_level/v88_defect_exclusion/field_stage1.py` に開発用限定の処理を実装した。
- Codexが `field_level/v88_defect_exclusion/field_verify_stage1.py` に保存表の検収と報告保存を実装した。
- Codexが `field_level/v88_defect_exclusion/NOTES.md` に基準・確度の訂正・結果・指紋を追記した。
- 記録係が `docs/モデル選択の記録.md` に作業種類と格上げなしを追記した。
- Codexが `data/results/v88_defect_exclusion/date_split.csv` と `input_boundary.csv` に日程分割と識別子の共通部分0を保存した。
- Codexが同じ結果フォルダの `centers_early.csv`、`centers_late.csv`、`centers_full.csv`、`center_reproducibility.csv` に推定中心と再現距離を保存した。
- Codexが同じ結果フォルダの `blank_counts_by_k.csv`、`calibration_k.csv`、`radius_sensitivity.csv`、`frozen_k.json` に実個数・換算個数・固定倍率を保存した。
- Codexが同じ結果フォルダの `history_E3_new_hotspot_effect.csv`、`history_E3_new_hotspot_post_effect.csv`、`history_E3_repro_new.csv`、`historical_design_summary.csv` に既存3表の数値と設計経緯を保存した。
- Codexが同じ結果フォルダの `array_read_audit.csv`、`stage1_summary.json`、`verification.json` に読み取り履歴・主要結果・検収結果を保存した。
- Codexが同じ結果フォルダの `NOTES_before_hash_append.md`、`sha256_manifest.csv`、`git_save_procedure.md` に固定記録・指紋・未実行の保存手順を保存した。

Codexが解析と検収に実行したコマンドは次の2つである。

```powershell
python field_level/v88_defect_exclusion/field_stage1.py
python field_level/v88_defect_exclusion/field_verify_stage1.py
```

Codexは保存表を読み戻し、全18校正条件の最小倍率、全31,050行、1,301回の開発用配列読み取りを検収した。Codexは除外なし正側の1,725行を旧版と比較し、数値差0を確認した。
Codexは日程分割の合格に失敗した。Codexは24視野の画像・差分配列、無作為除外・座標移動・信号注入・日程を抜く交差検証を実施していない。CodexはGitへの履歴保存・送信・枝切り替えを実施していない。
Codexは欠陥表の目視承認がまだないため、この計算を感度計算として扱い、標準経路を変更していない。CodexはVaultへの書き込み権限がないため、Vaultの結果・進捗・次の一手・引き継ぎを更新できなかった。
Claude Codeが次に、本報告と指紋を検収し、日程分割の不合格を扱ってから、24視野を開く次段階の可否を判断する。
'''
    (OUT/'【未読】stage1_report.md').write_text(report,encoding='utf-8')
    procedure='''# Git保存手順：事後の再評価（独立な検証ではない）
Codexはこの手順を実行していない。Claude Codeが白石さんの確認を取った後、次の順番で対象だけを保存する。Claude Codeは既存の未保存変更を含めず、画像・キャッシュを追加しない。
```powershell
git diff -- field_level/v88_defect_exclusion/NOTES.md
git status --short
git add -- data/raw/v88_camera_defects_unapproved.csv field_level/v88_defect_exclusion/NOTES.md field_level/v88_defect_exclusion/field_stage1.py field_level/v88_defect_exclusion/field_verify_stage1.py
git add -f -- data/results/v88_defect_exclusion/frozen_k.json data/results/v88_defect_exclusion/NOTES_before_hash_append.md data/results/v88_defect_exclusion/sha256_manifest.csv
git diff --cached --stat
git diff --cached --check
git commit -m "保存: C1+D第一段階（事後の再評価、独立な検証ではない）"
```
Claude Codeは欠陥表の出所が既存の `data/raw/artifact_provenance_registry.csv` のどの行かを検収し、並行変更を混ぜずに保存する。Claude Codeは送信の指示を別に受けるまで送信しない。
'''
    (OUT/'git_save_procedure.md').write_text(procedure,encoding='utf-8')
    sources=[ROOT/'data/raw/v88_camera_defects_unapproved.csv',notes,ROOT/'field_level/v88_defect_exclusion/field_stage1.py',ROOT/'field_level/v88_defect_exclusion/field_verify_stage1.py']
    sources+=sorted(p for p in OUT.iterdir() if p.is_file())
    for f in ['E4_counts_all632.csv','E3_new_hotspot_effect.csv','E3_new_hotspot_post_effect.csv','E3_repro_new.csv']:
        p=SRC/'v85_abnormal_pillars'/f
        if p.exists(): sources.append(p)
    pd.DataFrame([dict(evaluation=LABEL,path=str(p),sha256=sha(p)) for p in sources]).to_csv(OUT/'sha256_manifest.csv',index=False,encoding='utf-8-sig')
    manifest=pd.read_csv(OUT/'sha256_manifest.csv')
    assert all(sha(Path(r.path))==r.sha256 for r in manifest.itertuples())
    assert notes.read_text(encoding='utf-8').endswith('枝切り替えを行っていない。\n')
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':
    main()
