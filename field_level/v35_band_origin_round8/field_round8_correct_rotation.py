"""Correct an implementation error in the preregistered N2 control only."""
from field_round8_common import *
from field_round8_experiment import prepare,generate,estimate,amplitude,rotate_field
from field_round8_evaluate import load_common,evaluate_key
from concurrent.futures import ProcessPoolExecutor,as_completed
import traceback,argparse

STATE=None;CODE=None;COMMONS=None
def initialize():
    global STATE,CODE,COMMONS
    verify();STATE=setup();CODE=original_code(STATE);COMMONS=load_common(STATE)
    unit=np.zeros((64,64,2));unit[:,:,0]=1
    assert np.array_equal(rotate_field(unit),np.broadcast_to([0.,-1.],unit.shape))
    marker=np.zeros((64,64,2));marker[10,20]=[1.,2.];rot=rotate_field(marker)
    assert np.array_equal(rot[43,10],[2.,-1.])

def correction(key):
    try:
        dest=OUT/'experiment_checkpoints';p=dest/(key+'.json');r=json.loads(p.read_text(encoding='utf-8'))
        if r['条件']['N2'].get('回転整合確認済み',False):return key,True
        with np.load(dest/(key+'.npz')) as z:data={k:z[k].copy() for k in z.files}
        retained={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in data.items() if name!='N2_delta'}
        legacy=OUT/'rotation_control_before_correction';legacy.mkdir(exist_ok=True)
        if not (legacy/(key+'.json')).exists():
            np.savez_compressed(legacy/(key+'.npz'),N2_delta=data['N2_delta'])
            dump(dict(key=key,旧N2条件=r['条件']['N2'],旧実験記録指紋=digest(p)),legacy/(key+'.json'))
        pre=read_image(STATE['native'][r['洗浄前パス']]);m=np.asarray(r['元変換'],np.float32);u=rotate_field(data['local_field']);xy=data['xy']
        expanded=prepare(pre,m);raw,info=generate(pre,m,u,expanded);del expanded
        try:
            mat,est=estimate(pre,raw,CODE[0],CODE[1],m,xy);data['N2_delta']=data['pre_contrast']-sampler(raw,transform(xy,mat));info.update(状態='成功',位置合わせ=est)
        except Exception as exc:
            data['N2_delta']=np.full(len(xy),np.nan);info.update(状態='位置合わせ失敗',理由=repr(exc))
        info.update(場振幅=amplitude(u),回転整合確認済み=True,修正理由='空間配置とベクトルの90度回転を一致。結果に合わせた調整ではなく事前規定の実装修正')
        r['条件']['N2']=info
        assert retained=={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in data.items() if name!='N2_delta'}
        info['他条件配列不変']=True
        temporary=dest/(key+f'.rotation.{os.getpid()}.npz');np.savez_compressed(temporary,**data);os.replace(temporary,dest/(key+'.npz'));dump(r,p)
        evaluate_key(key,STATE,COMMONS)
        return key,True
    except Exception as exc:
        dump(dict(key=key,理由=repr(exc),詳細=traceback.format_exc()),OUT/'rotation_correction_errors'/(key+'.json'));return key,False

def run(workers=3):
    verify();s=setup();expected=s['eligible'];done=set();pending={}
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        while True:
            available=set(p.stem for p in (OUT/'evaluation_checkpoints').glob('*.json'))
            for key in sorted(available-done-set(pending)):
                pending[key]=pool.submit(correction,key)
            for key,future in list(pending.items()):
                if future.done():
                    print('rotation corrected',*future.result(),flush=True);done.add(key);del pending[key]
            if expected<=done:break
            time.sleep(3)
    dump(dict(回転検証済み視野数=len(done),単位横ベクトル回転='(1,0) -> (0,-1)',点の配置='(20,10) -> (10,43)',原予測不変=True,修正前対照保存先=str(OUT/'rotation_control_before_correction')),OUT/'rotation_correction_verification.json')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=3);args=a.parse_args();run(args.workers)
