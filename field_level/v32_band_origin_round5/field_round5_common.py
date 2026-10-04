"""Round-five-only paths and read-only source discovery."""
import sys, os
sys.dont_write_bytecode=True
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json, hashlib, subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
LOCAL=ROOT/'data/inputs_local'
OLD=ROOT/'data/results/v28_band_origin_round1'
PREV=ROOT/'data/results/v31_band_origin_round4'
OUT=ROOT/'data/results/v32_band_origin_round5'
METHODS=['平均標準偏差','中央値絶対偏差']
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def table(p):return pd.read_csv(p,dtype={'日程':str,'基板':str,'date':str,'board':str})
def csv(rows,name):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def dump(obj,p):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=True,default=str),encoding='utf-8')
def verify():
    r=table(OUT/'prediction_sha256.csv').iloc[0]
    assert digest(Path(r.Path)).upper()==r.Hash
def fields():
    f=table(OLD/'all_fields_existing_definition.csv')
    f['key']=f['日程']+'_'+f['基板']+'_'+f['視野番号'].astype(str)
    return f
def discover():
    children={p.name:p for p in LOCAL.iterdir()}
    sources=[p for n,p in children.items() if p.is_dir() and n.startswith('2026') and 'digital_judgment' in n]
    assert len(sources)==1
    folders={p.name:p for p in sources[0].iterdir()}
    return sources[0]
def load(r):
    caches={p.stem:p for p in (discover()/'tables/cached_field_differences').iterdir() if p.suffix=='.npz'}
    with np.load(caches[r.key]) as z:return z['delta'].copy(),z['xy'].copy(),z['ids'].copy()
def init():
    prediction=next(p for p in OUT.iterdir() if '予測と検定規定' in p.name)
    csv([dict(Path=str(prediction.resolve()),Hash=digest(prediction).upper())],'prediction_sha256.csv')
    source=discover();children=list((LOCAL/'raw_readonly').iterdir())
    dates=['260922','260924','260828']
    third=any(p.is_file() and 'copy_done_3' in p.name for p in children)
    if third:dates+=['260923','260927','260829','260825','260827']
    present={n:[str(p) for p in children if p.is_dir() and p.name.startswith(n)] for n in dates}
    dump(dict(入力起点=str(LOCAL),解析入力=str(source),生画像直下=[p.name for p in children],第三コピー完了=third,追加回復対象=dates,実在対象=present,作業ブランチ=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),開始コミット=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),開始状態=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),版確認='開始時点でv32は版フォルダ・解析入力直下・ブランチ・タグに未使用。',環境=dict(Python=sys.version,数値配列=np.__version__,表処理=pd.__version__)),OUT/'round5_preflight.json')
    used=set()
    for folder in ['v28_band_origin_round1','v29_band_origin_round2','v30_band_origin_round3','v31_band_origin_round4']:
        for parent in ['field_level','data/results']:
            used.update(p for p in (ROOT/parent/folder).rglob('*') if p.is_file())
    # Audit copied inputs completely, including images and original-code snapshot.
    used.update(p for p in LOCAL.rglob('*') if p.is_file())
    csv([dict(実在パス=str(p.resolve()),バイト数=p.stat().st_size,開始内容指紋=digest(p)) for p in sorted(used)],'round5_input_integrity.csv')
    print('preflight and prediction hash saved',len(used),flush=True)
def integrity():
    a=table(OUT/'round5_input_integrity.csv');a['終了内容指紋']=[digest(Path(p)) for p in a['実在パス']]
    a['不変']=a['開始内容指紋']==a['終了内容指紋'];csv(a,'round5_input_integrity_final.csv');assert a['不変'].all()
    print('input and prior versions unchanged',len(a),flush=True)
if __name__=='__main__':
    {'init':init,'integrity':integrity}[sys.argv[1]]()
