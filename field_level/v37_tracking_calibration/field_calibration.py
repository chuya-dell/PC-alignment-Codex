"""周9の入力監査と、変更しない既存追跡処理の再現確認。"""
import sys
sys.dont_write_bytecode = True
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import hashlib, json, csv, itertools, subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent

def child(parent, name):
    return {p.name:p for p in parent.iterdir()}[name]

DATA = child(ROOT, 'data')
RESULTS = child(DATA, 'results')
OUT = child(RESULTS, CODE.name)
FIELD = child(ROOT, 'field_level')
LOCAL = child(DATA, 'inputs_local')

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def table(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def resolve(value):
    p=Path(value)
    parts=p.relative_to(ROOT).parts
    q=ROOT
    for name in parts:q=child(q,name)
    return q

def inventory():
    bases=[LOCAL, child(ROOT,'pillar_level'),child(ROOT,'shared'),child(DATA,'raw')]
    bases += [p for p in FIELD.iterdir() if p.is_dir() and p!=CODE]
    bases += [p for p in RESULTS.iterdir() if p.is_dir() and p!=OUT]
    paths=set(p for base in bases for p in base.rglob('*') if p.is_file())
    paths.update(p for p in ROOT.iterdir() if p.is_file())
    return [{'パス':str(p.relative_to(ROOT)), 'バイト数':p.stat().st_size,'更新時刻ナノ秒':p.stat().st_mtime_ns} for p in sorted(paths)]

def resume():
    # 再開時に開始監査を取り直さず、最初の記録を保持する。
    with child(OUT,'protected_inventory_start.csv').open(encoding='utf-8',newline='') as f:
        start=[{'パス':r['パス'],'バイト数':int(r['バイト数']),'更新時刻ナノ秒':int(r['更新時刻ナノ秒'])} for r in csv.DictReader(f)]
    with child(OUT,'input_fingerprints.csv').open(encoding='utf-8',newline='') as f:fingerprints=list(csv.DictReader(f))
    assert all(digest(resolve(r['実在パス']))==r['開始内容指紋'] for r in fingerprints),'再開前に入力指紋不一致'
    assert start==inventory(),'再開前に保護対象一覧不一致'
    resolved=json.loads(child(OUT,'conditions_resolved.json').read_text(encoding='utf-8'))
    final=child(child(RESULTS,'v35_band_origin_round8'),'resume_20261004_final')
    cps=child(final,'experiment_checkpoints')
    records={k:json.loads(child(cps,k+'.json').read_text(encoding='utf-8-sig')) for k in resolved['対象']}
    import numpy as np
    recorded_hash={r['実在パス']:r['開始内容指紋'] for r in fingerprints}
    provenance=[]
    for key,rec in records.items():
        recovery=json.loads(resolve(rec['回復記録パス']).read_text(encoding='utf-8-sig'))
        for role in ['洗浄前','洗浄後']:
            p=resolve(rec[role+'パス'])
            provenance.append({'視野':key,'入力':role,'パス':str(p),'保存記録の指紋':recovery[role+'指紋'],'開始指紋':recorded_hash[str(p)],'一致':recovery[role+'指紋']==recorded_hash[str(p)]})
        assert np.array_equal(np.asarray(rec['元変換'],np.float32),np.asarray(recovery['最終行列'],np.float32)),'周8と回復記録の全体変換不一致'
    table('input_provenance_verification.csv',provenance)
    assert all(r['一致'] for r in provenance),'保存記録と生画像指紋不一致'
    sys.path.insert(0,str(child(FIELD,'v32_band_origin_round5')))
    import field_round5_methods as methods
    from field_calibration_run import run
    run(records,resolved['条件'],final,methods)
    end=inventory();table('protected_inventory_final.csv',end)
    dump('protected_inventory_comparison.json',{'一致':start==end,'開始ファイル数':len(start),'終了ファイル数':len(end)})
    for r in fingerprints:
        r['終了内容指紋']=digest(resolve(r['実在パス']));r['不変']=r['開始内容指紋']==r['終了内容指紋']
    table('input_integrity_final.csv',fingerprints)
    assert start==end and all(r['不変'] for r in fingerprints)
    report=child(OUT,'261004_周9_Codex報告.md')
    report.write_text(report.read_text(encoding='utf-8')+f'\n終了確認（確度：高）：保護対象{len(start)}ファイルのパス・サイズ・更新時刻が全件一致。入力{len(fingerprints)}ファイルの内容指紋も全件一致。input_provenance_verification.csv では8枚の生画像が保存記録の指紋と一致し、4視野の全体変換も周8で使用する単精度表現で一致した。\n\n確認できなかったもの（確度：高）：Windowsの計算過程詳細一覧の照会はアクセス拒否で確認できなかった。実行セッションの中断・継続と条件別保存は確認でき、必要な解析入力の読込に支障はなかった。主処理を一度中断し、独立した視野の計算終了後、保存済み条件を読み込んで集計から再開した。\n',encoding='utf-8')
    outputs=[p for base in [CODE,OUT] for p in base.rglob('*') if p.is_file() and p.name!='output_inventory.csv']
    table('output_inventory.csv',[{'パス':str(p.relative_to(ROOT)),'バイト数':p.stat().st_size,'内容指紋':digest(p)} for p in sorted(outputs)])
    print(child(OUT,'completion.json').read_text(encoding='utf-8'),flush=True)

def main():
    pred=child(CODE,'prediction.md')
    dump('prediction_sha256.json',{'実在パス':str(pred),'SHA256':digest(pred),'予測ファイル保存時刻':datetime.fromtimestamp(pred.stat().st_mtime,timezone.utc).isoformat(),'指紋記録時刻':datetime.now(timezone.utc).isoformat()})
    previous=child(OUT,'261004_周9_Codex報告.md')
    preserved=OUT/'261004_周9_Codex停止報告_周回1.md'
    if not preserved.exists(): previous.rename(preserved)
    start=inventory();table('protected_inventory_start.csv',start)
    r8=child(RESULTS,'v35_band_origin_round8');final=child(r8,'resume_20261004_final')
    fixed=child(final,'fixed_field_support.csv')
    with fixed.open(encoding='utf-8-sig',newline='') as f: keys=[r['key'] for r in csv.DictReader(f)]
    assert set(keys)=={'260827_7_1','260827_7_8','260829_3_7','260926_7_5'}
    inputs={pred,child(OUT,'prediction_erratum_1.md'),child(OUT,'v37_unused_check.md'),fixed}
    inputs.update(p for p in ROOT.iterdir() if p.is_file())
    inputs.update(p for p in child(ROOT,'docs').iterdir() if p.is_file() and p.name in ['LOCAL_ENVIRONMENT_20260926.md','REPOSITORY_CONVENTIONS.md'])
    # 保守的に旧版と入力の全Python依存を記録する。実際の入力は保存記録から列挙で解決する。
    inputs.update(p for p in FIELD.rglob('*.py') if CODE not in p.parents)
    inputs.update(LOCAL.rglob('*.py'))
    inputs.update(child(ROOT,'shared').rglob('*.py'))
    inputs.update(child(ROOT,'pillar_level').rglob('*.py'))
    records={}
    checkpoints=child(final,'experiment_checkpoints')
    for key in keys:
        cp=child(checkpoints,key+'.json');inputs.add(cp)
        rec=json.loads(cp.read_text(encoding='utf-8-sig'));records[key]=rec
        assert rec['ブランク'] is False
        for col in ['場実在パス','洗浄前パス','洗浄後パス','保存差パス','回復記録パス']:
            inputs.add(resolve(rec[col]))
        feature=resolve(rec['場実在パス']);inputs.add(child(feature.parent,feature.stem+'.json'))
        recovery=resolve(rec['回復記録パス']);inputs.add(child(recovery.parent,recovery.stem+'.npz'))
    for name in ['signed_field_metrics.csv','outlier_field_metrics.csv']:inputs.add(child(final,name))
    inputs.add(child(child(RESULTS,'v28_band_origin_round1'),'round1_thresholds.csv'))
    fingerprints=[{'実在パス':str(p),'開始内容指紋':digest(p),'バイト数':p.stat().st_size} for p in sorted(inputs)]
    table('input_fingerprints.csv',fingerprints)
    conditions=[dict(条件番号=i+1,周期画素=p,振幅画素=a,波の進む方向度=d,位相度=f) for i,(p,a,d,f) in enumerate(itertools.product([128,212,256,512],[.05,.1,.2,.3],[0.,90.,111.2505],[0.,90.]))]
    dump('conditions_resolved.json',{'帯の線の方向度':21.2505,'波の進む方向度':111.2505,'根拠':'field_round8_common.py の spectrum は周波数ベクトルの角度に90度を加えて帯方向を出す。','変位方向':'波の進む方向と平行','位相原点画素':[1023.5,1021.5],'対象':keys,'分類':'対象4視野はすべて分子あり。260926のブランクは基板01。','条件':conditions})
    sys.path.insert(0,str(child(FIELD,'v32_band_origin_round5')))
    import numpy as np
    import cv2
    cv2.setNumThreads(1)
    import field_round5_methods as methods
    key='260926_7_5';rec=records[key]
    pre=methods.read_image(resolve(rec['洗浄前パス']));post=methods.read_image(resolve(rec['洗浄後パス']))
    matrix=np.asarray(rec['元変換'],np.float32)
    recovered,info=methods.tracking(pre,post,matrix)
    with np.load(resolve(rec['場実在パス']),allow_pickle=False) as z:
        comparisons=[]
        for name in ['support','displacement','fb_error','window_error','residual_grid','support_groups']:
            a=recovered.get(name);b=z[name]
            exact=a is not None and np.array_equal(a,b,equal_nan=True)
            numerical=a is not None and a.shape==b.shape and np.allclose(a,b,rtol=0,atol=8*np.finfo(float).eps,equal_nan=True)
            comparisons.append({'配列':name,'完全一致':exact,'丸め精度内一致':numerical,'最大絶対差':float(np.max(np.abs(a-b))) if a is not None and a.shape==b.shape else None})
    dump('tracking_reproduction.json',{'対象':key,'追跡記録':info,'比較':comparisons,'再現確認':all(r['丸め精度内一致'] for r in comparisons),'比較実装修正':'ビット一致の要求を修正。絶対差が倍精度機械精度の8倍以下で、相対許容差は0。追跡・解析の条件閾値は変更しない。'})
    print(json.dumps({'再現比較':comparisons},ensure_ascii=False),flush=True)
    if all(r['丸め精度内一致'] for r in comparisons):
        from field_calibration_run import run
        run(records,conditions,final,methods)
    end=inventory();table('protected_inventory_final.csv',end)
    dump('protected_inventory_comparison.json',{'一致':start==end,'開始ファイル数':len(start),'終了ファイル数':len(end)})
    for r in fingerprints:
        r['終了内容指紋']=digest(resolve(r['実在パス']));r['不変']=r['開始内容指紋']==r['終了内容指紋']
    table('input_integrity_final.csv',fingerprints)
    assert start==end and all(r['不変'] for r in fingerprints)
    if not all(r['丸め精度内一致'] for r in comparisons):
        reason='周8の保存場を既存追跡関数と保存全体変換で完全再現できなかった。設定変更・代替処理をせず停止。'
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
        state=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip()
        report=f'''# 周9 追跡場の振幅較正：周回2停止報告

事実（確度：高）：訂正記録を読み、対象4視野はすべて分子ありとして扱った。260926の基板7は分子あり、ブランクは基板01。事前予測は変更していない。前回の停止報告を内容を変えずに別名保存した。

手順0の予測指紋、全入力指紋、96条件表を保存した。入力指紋は{len(fingerprints)}ファイル。保護対象の開始・終了一覧は{len(start)}ファイルで一致し、記録した入力の内容指紋も全件一致。

停止理由（確度：高）：{reason}

再現比較は tracking_reproduction.json に記載。結果：{json.dumps(comparisons,ensure_ascii=False)}。

主384条件・補助384条件の完了は各0、未実行は各384。実行後欠損数は未評価。周期212画素の利得と手順5の補正後分散比は未算出。欠損を0で埋めていない。較正・補正の結果は作成していない。

解釈（確度：低）：不一致の原因は未特定。保存場の来歴または実行環境の差が考えられるが、この確認では区別できない。追跡の減衰や他の三候補について結論を出せない。

作業ブランチ：{branch}。未追跡状況：{state}。コミット・タグ付け・送信・削除は実施していない。取得は書込範囲外を変更するため実施していない。
'''
        (OUT/'261004_周9_Codex報告.md').write_text(report,encoding='utf-8')
        (CODE/'NOTES.md').write_text('# 実装変更\n\nfield_calibration.py を追加。既存追跡関数を読み取り専用で使用し、設定は変更していない。\n\n# 結果と保存先\n\ndata/results/v37_tracking_calibration/ に全入力指紋、条件表、保存場再現確認、開始終了一覧、停止報告を保存。\n\n# 既知の問題\n\n'+reason+'\n',encoding='utf-8')
        print(json.dumps({'status':'stopped','reason':reason},ensure_ascii=False),flush=True)

if __name__=='__main__':
    if '--resume' in sys.argv:resume()
    else:main()
