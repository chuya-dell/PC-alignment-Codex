"""Analytic camera-aperture check independent of measured differences."""
from field_round7_common import *

def main():
    verify();y,x=np.indices((2044,2048),dtype=float)
    frequency=np.array([.137,.079]);amplitude=1000.;background=25000.
    pre=np.rint(background+amplitude*np.sinc(frequency[0])*np.sinc(frequency[1])*np.cos(2*np.pi*(frequency[0]*x+frequency[1]*y))).astype(np.uint16)
    theta=np.deg2rad(.45)
    a=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])@np.array([[1.003,.001],[0,.998]])
    center=np.array([1023.5,1021.5]);m=np.column_stack([a,center+np.array([3.25,-4.375])-a@center]).astype(np.float32)
    simulated,info=synthesize(pre,m)
    inv=np.linalg.inv(m[:,:2].astype(float));freq_after=frequency@inv
    original=(np.column_stack([x.ravel(),y.ravel()])-m[:,2])@inv.T
    expected=(background+amplitude*np.sinc(freq_after[0])*np.sinc(freq_after[1])*np.cos(2*np.pi*(original@frequency))).reshape(pre.shape)
    interior=(x>=100)&(x<1948)&(y>=100)&(y<1944)
    error=simulated[interior].astype(float)-expected[interior]
    result=dict(目的='回転・倍率・せん断・端数並進の向きと正方形画素開口を解析的な余弦波で確認',周波数=frequency.tolist(),振幅=amplitude,平均誤差=float(error.mean()),二乗平均平方根誤差=float(np.sqrt(np.mean(error**2))),最大絶対誤差=float(abs(error).max()),振幅相対二乗平均平方根誤差=float(np.sqrt(np.mean(error**2))/amplitude),生成=info)
    assert result['振幅相対二乗平均平方根誤差']<.015
    dump(result,OUT/'analytic_camera_verification.json');print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
