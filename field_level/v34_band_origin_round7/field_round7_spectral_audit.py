"""Absolute spectral axes and peak-strength audit, descriptive only."""
from field_round7_common import *

def main():
    verify();dest=OUT/'spectral_checkpoints';dest.mkdir(exist_ok=True)
    for p in sorted((OUT/'experiment_checkpoints').glob('*.npz')):
        if (dest/(p.stem+'.json')).exists():continue
        with np.load(p) as z:data={k:z[k] for k in z.files}
        actual=data['actual_delta'];xy=data['xy'];rows=[]
        for variant in ['主帰無','既知変換','N0','N1','N2','N3']:
            sim=data[variant+'_delta'];valid=np.isfinite(actual)&np.isfinite(sim)
            if valid.sum()<100:continue
            a,_=binned(xy,np.where(valid,actual,np.nan));b,_=binned(xy,np.where(valid,sim,np.nan))
            sa,sb=spectrum(a),spectrum(b)
            row=dict(key=p.stem,条件=variant,実測第一軸帯方向=sa['第一軸帯方向'] if sa else np.nan,模擬第一軸帯方向=sb['第一軸帯方向'] if sb else np.nan)
            if sa and sb:
                i=sa['ピーク位置'];j=sb['ピーク位置']
                row.update(第一軸方向差=angle(sa['第一軸帯方向'],sb['第一軸帯方向']),模擬の実測ピーク対最大パワー比=float(abs(sb['変換'][i])**2/abs(sb['変換'][j])**2),模擬ピーク探索下限近接=bool(sb['周期']<=64*1.1),実測ピーク探索上限近接=bool(sa['周期']>=1024/1.1))
            rows.append(row)
        dump(rows,dest/(p.stem+'.json'))
    rows=[r for p in sorted(dest.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))]
    pd.DataFrame(rows).to_csv(OUT/'spectral_first_axes.csv',index=False,encoding='utf-8-sig')
    print('spectral axes',len(rows),flush=True)

if __name__=='__main__':main()
