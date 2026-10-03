"""Numerical checks and independent nonperiodic marker phase audit."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from field_round1_analysis import OUT
from field_round1_bf_recovery import DEST,verify,csv,digest,CODE,ROOT
from field_round1_mechanisms import read_image
from field_round1_bf_models import folds,cv_score,shifted,template,sign_test

def marker_audit():
    # These four images were opened and visually inspected in this continuation.
    inspected={'260926_01_6','260926_01_8','260926_7_5','260926_7_8'}
    geometry=pd.read_csv(OUT/'round1_visually_confirmed_marker_geometry.csv')
    review={};rows=[]
    for _,r in geometry[geometry.key.isin(inspected)].iterrows():
        row=json.loads((DEST/(r.key+'.json')).read_text(encoding='utf-8'))
        pre=read_image(Path(row['洗浄前パス']));post=read_image(Path(row['洗浄後パス']))
        with np.load(DEST/(r.key+'.npz')) as z:m=z['matrix']
        aligned=cv2.warpAffine(post,m,(2048,2044),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        mat=np.array([[1,-r['垂直線横傾き']],[-r['水平線傾き'],1]])
        center=np.linalg.solve(mat,[r['垂直線横切片画素'],r['水平線切片画素']])
        x,y=np.clip(center,[128,128],[1920,1916]).astype(int)
        a=cv2.GaussianBlur(pre.astype(np.float32),(0,0),4)[y-110:y+110,x-110:x+110]
        b=cv2.GaussianBlur(aligned.astype(np.float32),(0,0),4)[y-122:y+122,x-122:x+122]
        result=cv2.matchTemplate(b,a,cv2.TM_CCOEFF_NORMED)
        iy,ix=np.unravel_index(np.argmax(result),result.shape)
        delta=np.array([ix-12,iy-12],float)
        distance=float(np.linalg.norm(delta))
        approved=bool(row['回復'] and distance<7.286/2 and np.isfinite(result.max()))
        review[r.key]=dict(approved=approved,visual_inspected=True,marker='洗浄前後の非周期的十字線と交点を重ねた図を確認',
                           smoothed_marker_offset_px=delta.tolist(),phase_validation='周期像を四画素のガウス平滑化で弱めた独立十字線の相関。変換の更新に使わない。',
                           correlation=float(result.max()),image=str(OUT/'bf_marker_review'/(r.key+'.png')),
                           limitation='非周期目印は縁の一点の参照。全面の局所追跡の独立支持検証を置き換えない。')
        rows.append(dict(key=r.key,目視確認=True,回復=row['回復'],十字線相関最大値=float(result.max()),横差画素=delta[0],縦差画素=delta[1],差距離画素=distance,F位相検証採用=approved))
    (OUT/'bf_marker_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    csv(rows,'round1_bf_marker_phase_audit.csv')
    return rows

def checks():
    verify();yy,xx=np.indices((64,64));valid=np.ones((64,64),bool)
    # A held-out known linear response must be recovered; shuffled maps must fail.
    rng=np.random.default_rng(41);x=rng.normal(size=(64,64,2));y=3*x[:,:,0]-2*x[:,:,1]+rng.normal(size=(64,64))*.03
    good,_=cv_score(y,x,valid);bad,_=cv_score(y,rng.permutation(x.reshape(4096,2)).reshape(x.shape),valid)
    assert good>.999 and bad<.05,(good,bad)
    split,labels=folds(valid)
    for k,(tr,te) in enumerate(split):
        embargo=cv2.dilate((labels==k).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        assert not embargo.ravel()[tr].any()
        assert np.all(labels.ravel()[te]==k)
    test=np.arange(4096,dtype=float).reshape(64,64,1)
    shift=shifted(test,0,16)
    assert np.isnan(shift[:,:16]).all() and np.array_equal(shift[:,16:],test[:,:48])
    # A zero identity transform must yield exactly zero template sampling difference.
    pre=np.uint16(10000+2000*np.cos(xx/7)+1000*np.sin(yy/6))
    xy=np.column_stack([xx.ravel(),yy.ravel()]).astype(float)+.23
    matrix=np.array([[1,0,0],[0,1,0]],np.float32)
    basis=np.array([[7.286,3.643],[0,7.286*np.sqrt(3)/2]])
    td,_,_=template(pre,xy,xy,matrix,basis,np.array([.1,.2]))
    assert np.max(abs(td))==0
    # Verify marker review does not depend on y or model results.
    phase=marker_audit()
    result=dict(known_response_cv=good,shuffled_predictor_cv=bad,embargo_verified=True,noncircular_shift_verified=True,
                zero_transform_template_max=float(np.max(abs(td))),prediction_hash=verify(),marker_audit=phase)
    (OUT/'round1_bf_numerical_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':checks()
