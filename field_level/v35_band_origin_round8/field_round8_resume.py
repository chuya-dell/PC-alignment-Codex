"""Resume in a separate result directory, preserving all preceding artifacts."""
import os
import sys
sys.dont_write_bytecode = True
os.environ['PYTHONUTF8'] = '1'
from pathlib import Path
import shutil
import json
import argparse
import hashlib
import csv

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / 'data/results/v35_band_origin_round8'
RESUME = PRIOR / 'resume_20261004_final'
CODE = Path(__file__).resolve().parent

def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def configure():
    import field_round8_common as common
    common.OUT = RESUME
    os.environ['MPLCONFIGDIR'] = str(RESUME / 'plot_settings')
    common.verify()
    return common

def initialize():
    if (RESUME / 'resume_initial_state.json').exists():
        return
    RESUME.mkdir(exist_ok=True)
    inventory = []
    for path in sorted(PRIOR.rglob('*')):
        if not path.is_file() or RESUME in path.parents:
            continue
        relative = path.relative_to(PRIOR)
        inventory.append(dict(path=str(path),sha256=fingerprint(path),bytes=path.stat().st_size))
        target = RESUME / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in sorted(CODE.iterdir()):
        if path.is_file() and path.name != Path(__file__).name:
            inventory.append(dict(path=str(path),sha256=fingerprint(path),bytes=path.stat().st_size))
    with (RESUME / 'preserved_artifacts_start.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','sha256','bytes']);writer.writeheader();writer.writerows(inventory)
    c = configure(); s=c.setup()
    experiments={p.stem:json.loads(p.read_text(encoding='utf-8')) for p in (RESUME/'experiment_checkpoints').glob('*.json')}
    evaluations={p.stem for p in (RESUME/'evaluation_checkpoints').glob('*.json')}
    pending_rotation=[key for key,r in experiments.items() if not r['条件']['N2'].get('回転整合確認済み',False)]
    state=dict(開始実験完了数=len(experiments),開始評価完了数=len(evaluations),適格数=len(s['eligible']),
               未計算=sorted(s['eligible']-set(experiments)),未評価=sorted(set(experiments)-evaluations),
               回転未修正=pending_rotation,保存済み修正前対照数=len(list((RESUME/'rotation_control_before_correction').glob('*.json'))),
               予測指紋=c.PRED_HASH,保全ファイル数=len(inventory),旧報告実施数=77)
    c.dump(state,RESUME/'resume_initial_state.json')
    source=(CODE/'field_round8_report.py').read_text(encoding='utf-8')
    source=source.replace("report=OUT/'261004_周8_報告.md'", "report=OUT/'261004_周8_最終集計原稿.md'")
    source=source.replace("notes=ROOT/'field_level/v35_band_origin_round8/NOTES.md'", "notes=ROOT/'field_level/v35_band_origin_round8/NOTES_resume.md'")
    source=source.replace('報告は data/results/v35_band_origin_round8/261004_周8_報告.md', '報告は data/results/v35_band_origin_round8/261004_周8_報告_最終版.md')
    (CODE/'field_round8_resume_report.py').write_text(source,encoding='utf-8')
    progress(state,'未計算9視野の計算、修正前後比較、最終集計・報告を再開する。')
    print(json.dumps(state,ensure_ascii=False,indent=2),flush=True)

def progress(state,remaining):
    text='# 周8再開時の確認と残作業\n\n'
    text+='旧途中報告は77視野と記載するが、再開時の保存配列は170視野で、評価も170視野、回転対照の修正は170視野すべて完了。旧成果は上書きせず、再開先へ複製して再利用する。予測と実装修正の記録、途中報告、旧コードを保全する。\n\n'
    text+='```json\n'+json.dumps(state,ensure_ascii=False,indent=2)+'\n```\n\n残作業：'+remaining+'\n'
    (RESUME/'261004_周8_再開進捗.md').write_text(text,encoding='utf-8')

def verify_preserved():
    rows=[]
    with (RESUME/'preserved_artifacts_start.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            current=fingerprint(Path(row['path']));row.update(end_sha256=current,unchanged=str(current==row['sha256']));rows.append(row)
    with (RESUME/'preserved_artifacts_final.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert all(row['unchanged']=='True' for row in rows)
    print('preserved',len(rows),flush=True)

def resume_initialize():
    configure()
    from field_round8_experiment import initialize_worker
    initialize_worker()

def resume_worker(key):
    from field_round8_experiment import worker
    return worker(key)

def resume_run(workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    c=configure();s=c.setup()
    remaining=sorted(s['eligible']-{p.stem for p in (RESUME/'experiment_checkpoints').glob('*.json')})
    print('remaining',len(remaining),flush=True)
    with ProcessPoolExecutor(max_workers=workers,initializer=resume_initialize) as pool:
        futures=[pool.submit(resume_worker,key) for key in remaining]
        for future in as_completed(futures):
            print('saved',*future.result(),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['init','run','evaluate','report','preserve']);parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    if args.action=='init':initialize()
    elif args.action=='preserve':verify_preserved()
    else:
        c=configure()
        if args.action=='run':
            resume_run(args.workers)
        elif args.action=='evaluate':
            from field_round8_evaluate import evaluate
            evaluate()
        elif args.action=='report':
            from field_round8_resume_report import build
            build()
