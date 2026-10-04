"""前提の不一致による停止記録。解析・旧版の変更は行わない。"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import csv
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]

def child(parent, name):
    return {p.name: p for p in parent.iterdir()}[name]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    code = Path(__file__).resolve().parent
    data = child(ROOT, 'data')
    results = child(data, 'results')
    out = child(results, 'v37_tracking_calibration')
    prediction = child(code, 'prediction.md')
    first = out / 'prediction_sha256.json'
    if first.exists():
        raise RuntimeError('既存の停止記録を上書きしない')
    save_json(first, {'実在パス': str(prediction), 'SHA256': digest(prediction),
        '予測ファイル保存時刻': datetime.fromtimestamp(prediction.stat().st_mtime, timezone.utc).isoformat(),
        '指紋記録時刻': datetime.now(timezone.utc).isoformat()})
    r8 = child(results, 'v35_band_origin_round8')
    final = child(r8, 'resume_20261004_final')
    checkpoint = child(child(final, 'experiment_checkpoints'), '260926_7_5.json')
    recovery = child(child(child(results, 'v28_band_origin_round1'), 'bf_v24_recovery'), '260926_7_5.json')
    a = json.loads(checkpoint.read_text(encoding='utf-8'))
    b = json.loads(recovery.read_text(encoding='utf-8'))
    assert a['ブランク'] is False and b['ブランク'] is False
    inputs = [prediction, checkpoint, recovery, child(final, 'fixed_field_support.csv'),
              child(out, 'v37_unused_check.md'), child(ROOT, 'AGENTS.md')]
    docs = child(ROOT, 'docs')
    inputs += [child(docs, n) for n in ['LOCAL_ENVIRONMENT_20260926.md', 'REPOSITORY_CONVENTIONS.md']]
    field = child(ROOT, 'field_level')
    for folder, names in {
        'v32_band_origin_round5': ['field_round5_methods.py', 'field_round5_common.py', 'field_round5_local.py', 'field_round5_resume.py'],
        'v35_band_origin_round8': ['field_round8_common.py', 'field_round8_experiment.py', 'field_round8_resume.py', 'field_round8_scores.py']
    }.items():
        inputs += [child(child(field, folder), n) for n in names]
    rows = [{'実在パス': str(p), '開始内容指紋': digest(p)} for p in inputs]
    with (out / 'input_fingerprints.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        row['終了内容指紋'] = digest(Path(row['実在パス']))
        row['不変'] = row['開始内容指紋'] == row['終了内容指紋']
    with (out / 'input_integrity_final.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    state = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip()
    text = f'''# 周9 追跡場の振幅較正：停止報告

## 事実（確度：高）

較正を開始する前に停止した。主384条件・補助384条件の完了数は各0、未実行は各384。実行後の欠損数は未評価であり、未実行を欠損0とは扱わない。周期212画素の利得と手順5の補正後分散比は未算出。

依頼書と prediction.md は260926をブランクと記載している。一方、次の二つの既存記録はいずれも対象260926_7_5を濃度「1フェムトモル毎リットル」、ブランク=falseと記録している。

- `{checkpoint.relative_to(ROOT).as_posix()}`
- `{recovery.relative_to(ROOT).as_posix()}`

周8記録の参考ブランクもfalse。固定視野表は指定された4識別子と一致するが、実験条件を記載しておらず、この不一致を解消できない。

## 解釈と停止理由（確度：中）

依頼の実験条件と保存記録の分類に不一致がある。どちらが実際の実験条件を正しく表すかは未確認。既存記録を根拠に利用者の分類を変更することも、利用者の分類に合わせて既存記録を変更することも行っていない。「できないこと・指示と食い違うことがあれば、代わりのことをせず、status を stopped にして理由を書いて終える」という指定に従って停止した。

## 確認範囲と限界（確度：高）

周8の spectrum 関数は周波数ベクトルの角度に90度を加えて帯方向を出す。21.2505度は帯の線の方向で、波の進む方向へ換算すると111.2505度となる。追跡関数は窓21・31画素を別々に計算し、一致条件を課している。kernel_field は256画素を尺度とする正規分布型の重みで推定する。

停止により、保存場の再現確認、生画像の読み込み、全入力の内容指紋作成、96条件の確定表、測定方法の固定、較正、補正模擬は未実施。input_fingerprints.csv は今回確認した文書・コードだけの部分記録であり、要求された全入力の台帳ではない。結果の表を空値や0で代用していない。減衰、失われた高周波、光学条件の変化、物質変化のいずれも今回の結果で評価できない。

今回確認した入力の終了内容指紋は input_integrity_final.csv に記録し、全件一致。旧版全体、標準経路、マスク、data/inputs_local/ 全体の開始・終了指紋照合は未実施であり、全体の内容不変を指紋で保証したとは述べない。書き込みは指定された二つのフォルダに限定した。事前予測と未使用確認書は変更していない。

## 作業状態（確度：高）

作業ブランチ：`{branch}`。コミット・タグ付け・送信・削除は未実施。取得は許可された二つのフォルダ外に書き込むため実施していない。停止報告保存前の未追跡状況：

```text
{state}
```

結果フォルダは追跡除外されるため、未追跡表示だけで成果物の有無を判断しない。
'''
    (out / '261004_周9_Codex報告.md').write_text(text, encoding='utf-8')
    (code / 'NOTES.md').write_text('''# 実装変更

前提の不一致を記録する field_stopped_audit.py のみ追加。旧版関数の再実装、条件変更、解析経路変更はなし。

# 結果と保存先

data/results/v37_tracking_calibration/ に予測指紋、確認済み入力の部分指紋と終了照合、停止報告を保存。較正完了数は主・補助とも0。

# 既知の問題

260926_7_5のブランク分類が依頼と既存記録で食い違う。実験条件の正否は確認できず、指定に従い停止。全入力台帳・保存場再現・較正・補正模擬は未実施。再実行時は既存の停止記録を上書きせず、この記録を保全する必要がある。
''', encoding='utf-8')
    print(json.dumps({'status': 'stopped', '確認した入力数': len(rows), '内容指紋一致': all(r['不変'] for r in rows)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
