"""Render sample cross intersections and command-independent rough shifts for all 11 pairs."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).parent))
import field_control_common as M
import numpy as np,pandas as pd

def main():
    marks=json.loads((M.ROOT/'data/results/v58_motor_calibration/marker_landmarks.json').read_text(encoding='utf8'))
    folder=M.folder('261006-p50-ステッピングモーター_test');paths={p.name:p for p in folder.iterdir() if p.suffix=='.tif'}
    old=pd.read_csv(M.ROOT/'data/results/v58_motor_calibration/marker_pairs_checkpoint.csv');rows=[]
    for i,(a,b) in enumerate(zip(marks[:-1],marks[1:]),1):
        dx,dy=b['x']-a['x'],b['y']-a['y'];command=(b['position_mm']-a['position_mm'])*1000
        r=old.iloc[i-1];rows.append(dict(pair=i,pre=a['name'],post=b['name'],command_um=command,marker_dx=dx,marker_dy=dy,fine_dx=r.dx_px,fine_dy=r.dy_px,marker_status='coarse sample-cross intersection, provisional',fine_status='provisional' if i<=10 else 'rejected'))
        if i<7:continue
        fig,ax=M.plt.subplots(1,2,figsize=(12,6))
        for z,m in zip(ax,[a,b]):
            im=M.read(paths[m['name']]);z.imshow(im,cmap='gray',vmin=np.percentile(im,1),vmax=np.percentile(im,99));z.plot(m['x'],m['y'],'r+',ms=15);z.set_title(f"{m['name']} cross=({m['x']:.1f},{m['y']:.1f})")
            z.set(xlabel='native x px',ylabel='native y px')
        fig.suptitle(f'Pair {i}: cross delta=({dx:.1f},{dy:.1f}) px; command={command:.1f} um');fig.tight_layout();fig.savefig(M.OUT/f'landmark_pair_{i}.png',dpi=140);M.plt.close(fig)
    t=pd.DataFrame(rows);t.to_csv(M.OUT/'landmark_shift_verification.csv',index=False)
    fig,ax=M.plt.subplots(1,2,figsize=(12,4));ax[0].plot(t.pair,t.marker_dx,'o-',label='cross dx');ax[0].plot(t.pair,t.marker_dy,'o-',label='cross dy');ax[0].legend();ax[0].set(xlabel='Acquisition pair',ylabel='Sample cross shift (px)')
    ax[1].scatter(t.command_um,t.marker_dx);ax[1].set(xlabel='Command (um)',ylabel='Cross dx (px)');fig.tight_layout();fig.savefig(M.OUT/'landmark_shifts.png',dpi=150);M.plt.close(fig)
    print(t.to_string(index=False))

if __name__=='__main__':main()
