"""Scientific sanity checks and read-only input verification for this round."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import json,sys,hashlib
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
from field_round1_analysis import OUT, analyze_map, linear_autocorrelation, geometry, verify_prediction

def main():
    checks=[]
    y,x=np.indices((64,64))
    stripe=.05+.04*np.cos(2*np.pi*y*32/448)
    stats,ac,overlap,_=analyze_map(stripe,np.random.default_rng(20261004))
    assert stats['分類']=='帯状候補',stats
    assert abs(stats['第一自己相関周期画素']-448)<=32,stats
    assert stats['第一帯軸度'] in (0,170,10),stats
    checks.append(dict(検証='既知448画素の水平帯',結果=stats))
    a=np.array([[1.,2.,np.nan],[4.,5.,6.]])
    c,o=linear_autocorrelation(a)
    center=(a.shape[0]-1,a.shape[1]-1)
    assert abs(c[center]-1)<1e-6
    assert abs(o[center]-np.isfinite(a).sum())<1e-6
    checks.append(dict(検証='欠測付き自己相関のゼロラグと有効重なり',正常=True))
    for shape in ((64,64),(52,52),(46,46)):
        band,_,_,sector,_,_,_=geometry(shape)
        assert np.array_equal(sector.sum(axis=0),band.astype(int))
    checks.append(dict(検証='方向区分の境界に欠落・重複なし',正常=True))
    usage=pd.read_csv(OUT/'round1_cache_usage_sha256.csv')
    mismatches=[]
    for _,r in usage.iterrows():
        with open(r['実在パス'],'rb') as stream:
            value=hashlib.file_digest(stream,'sha256').hexdigest()
        if value!=r['内容指紋']:
            mismatches.append(r.key)
    assert not mismatches,mismatches
    checks.append(dict(検証='全632差保存ファイルの計算前後の内容指紋',ファイル数=len(usage),不一致数=len(mismatches)))
    recovery=pd.read_csv(OUT/'round1_transform_recovery.csv')
    mismatch=0
    for _,r in recovery.iterrows():
        for kind in ('洗浄前','洗浄後'):
            with open(r[f'実在{kind}パス'],'rb') as stream:
                value=hashlib.file_digest(stream,'sha256').hexdigest()
            mismatch+=value!=r[f'{kind}内容指紋']
    assert mismatch==0
    checks.append(dict(検証='260926全70視野140画像の計算前後の内容指紋',画像数=140,不一致数=mismatch))
    checks.append(dict(検証='元予測の内容指紋',指紋=verify_prediction()))
    (OUT/'round1_verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
    print('verified scientific checks and unchanged inputs',flush=True)

if __name__=='__main__':
    main()
