"""Small-scope numerical verification before whole-dataset execution."""
import sys
sys.dont_write_bytecode=True
import json
import numpy as np
import cv2
from field_round3_analysis import OUT,OLD,EDGES,CELLXY,discover,fields,table,folds,cv_score,sample,digest,verify,candidates,dump,registration

verify()
im=table(OUT/'round3_image_features.csv')
assert len(im)==2 and len(table(OUT/'round3_control_features.csv'))==1
assert im['保存差再現群'].sum()==1
f=fields();_,_,_,caches,_=discover()
with np.load(OLD/'density_maps.npz') as z:maps=z['density'];keys=list(z['keys'])
checked=[]
for key in im.key:
    with np.load(caches[key]) as z:xy=z['xy'];delta=z['delta']
    cell=(xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int);n=np.bincount(cell,minlength=4096)
    for mi,(_,r) in enumerate(f[f.key==key].iterrows()):
        positive=np.bincount(cell,weights=(delta>r['閾値']),minlength=4096)
        density=np.divide(positive,n,out=np.full(4096,np.nan),where=n>0).reshape(64,64)
        error=float(np.nanmax(abs(density-maps[keys.index(key),mi])));assert error<1e-7
        checked.append(dict(key=key,定義=r['定義'],密度最大誤差=error))
for tr,te in folds(np.ones((64,64),bool)):
    assert not set(tr)&set(te)
    grid=np.zeros((64,64),np.uint8);grid.ravel()[te]=1
    embargo=cv2.dilate(grid,np.ones((5,5),np.uint8));assert not embargo.ravel()[tr].any()
# Sample maps in native pixel coordinates, including vector lengths beyond OpenCV limits.
image=np.arange(2044,dtype=np.float32)[:,None]+np.zeros((2044,2048),np.float32)
points=np.tile(np.array([[60.,80.]]),(85791,1));assert np.allclose(sample(image,points),80)
scaled=np.arange(511,dtype=np.float32)[:,None]+np.zeros((511,512),np.float32)
assert np.allclose(sample(scaled,np.array([[60.,80.]])/4),20)
# Known smooth affine displacement exercises the actual registration and map direction.
rng=np.random.default_rng(20261004)
template=cv2.resize(rng.random((64,64)).astype(np.float32),(2048,2044))
template=cv2.GaussianBlur(template,(0,0),6)
known=np.array([[1,0,12],[0,1,-8]],np.float32)
moving=cv2.warpAffine(template,known,(2048,2044),borderMode=cv2.BORDER_REFLECT)
matrix,info=registration(template,moving)
error=float(np.max(abs(matrix-known)));assert info['対応採用'] and error<.3,(matrix,info)
# Flat image yields no automatic anomaly, without labeling it defect-free.
flat=candidates(np.full((2044,2048),.5,np.float32));assert flat['info']['候補数']==0
e=table(OUT/'round3_explained_fraction.csv');assert np.isfinite(e['全面決定係数']).all()
assert len(list((OUT/'evaluation').glob('*.json')))==2
summary=dict(小範囲画像組数=2,単独コントロール数=1,初回追加対応失敗='260828_10_1は相関0.895で規定0.90に届かず。閾値は変更しない',保存密度再確認=checked,学習保留重複なし=True,座標縮小倍率確認=True,大きいピラー座標配列読取確認=True,既知移動最大誤差画素=error,一様像の候補数=0,予測指紋不変=True,全体展開可能=True)
dump(summary,OUT/'round3_pilot_verification.json')
print('pilot passed',error,flush=True)
