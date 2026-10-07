"""Independent camera-template subtraction diagnostic. Never silently replaces v58 estimates."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).parent))
import field_control_common as M
import numpy as np,pandas as pd,cv2
from scipy import signal

def main():
    inv=pd.read_csv(M.ROOT/'data/results/v58_motor_calibration/input_inventory.csv')
    p=M.folder('261006-p50-ステッピングモーター_test')
    paths={f.name:f for f in p.iterdir() if f.suffix=='.tif'}
    ims=[M.read(paths[Path(x).name]) for x in inv.path]
    template=np.median(np.stack(ims),axis=0);res=[M.nonperiodic(a-template) for a in ims]
    old=pd.read_csv(M.ROOT/'data/results/v58_motor_calibration/marker_pairs_checkpoint.csv')
    rows=[]
    for i in range(6,11):
        a,b=res[i],res[i+1]
        win=cv2.createHanningWindow((a.shape[1],a.shape[0]),cv2.CV_32F)
        phase,response=cv2.phaseCorrelate(a.copy(),b.copy(),win)
        aa=cv2.resize(a,(512,511));bb=cv2.resize(b,(512,511));corr=signal.fftconvolve(bb,aa[::-1,::-1],mode='full')
        yy,xx=np.indices(corr.shape);dx=(xx-511)*4;dy=(yy-510)*4
        # Horizontal stage instruction supports this direction check, not a displacement calibration.
        search=(np.abs(dx)<1100)&(np.abs(dy)<32)&(np.abs(dx)>24)
        best=np.unravel_index(np.argmax(np.where(search,corr,-np.inf)),corr.shape);coarse=np.array([dx[best],dy[best]],float)
        row=dict(pair=i+1,command_um=old.iloc[i].command_um,phase_dx=phase[0],phase_dy=phase[1],response=response,horizontal_corr_dx=coarse[0],horizontal_corr_dy=coarse[1],old_dx=old.iloc[i].dx_px,old_dy=old.iloc[i].dy_px,status='diagnostic candidates; large-step calibration unresolved')
        rows.append(row)
        fig,axs=M.plt.subplots(2,3,figsize=(12,8))
        for ax,im,title in zip(axs[0],[ims[i],ims[i+1],template],['pre','post','pixelwise median; camera + sample mixture']):ax.imshow(cv2.resize(im,(512,511)),cmap='gray');ax.set_title(title);ax.axis('off')
        axs[1,0].imshow(aa,cmap='coolwarm',vmin=-2,vmax=2);axs[1,0].set_title('pre minus template, nonlattice')
        axs[1,1].imshow(bb,cmap='coolwarm',vmin=-2,vmax=2);axs[1,1].set_title('post minus template, nonlattice')
        for ax,method,shift in [(axs[1,2],'horizontal correlation',coarse)]:
            warped=cv2.warpAffine(b,np.float32([[1,0,-shift[0]],[0,1,-shift[1]]]),(b.shape[1],b.shape[0]));ax.imshow(cv2.resize(a-warped,(512,511)),cmap='coolwarm',vmin=-2,vmax=2);ax.set_title(f'{method}: shift={shift}')
        fig.suptitle(f'Pair {i+1}: phase={np.round(phase,2)}, v58={np.round([row["old_dx"],row["old_dy"]],2)}; candidates unapproved');fig.tight_layout();fig.savefig(M.OUT/f'large_pair_{i+1}.png',dpi=130);M.plt.close(fig)
    pd.DataFrame(rows).to_csv(M.OUT/'large_step_candidates.csv',index=False)
    small=old.iloc[:6];v=small[['dx_px','dy_px']].mean().to_numpy()/.5
    result=dict(magnification={'value':'100x','status':'confirmed by lattice scale'},nm_per_px_small={'value':1000/np.linalg.norm(v),'status':'provisional; commanded 0.5 um assumed'},nm_per_px_lattice={'value':62.271231966538465,'status':'provisional; physical pitch 460 nm assumed'},nm_per_px_large={'value':None,'status':'undetermined'},backlash_um={'value':None,'status':'undetermined'},stage_axis_deg_small={'value':float(np.degrees(np.arctan2(-v[1],-v[0]))),'status':'provisional'},comparison_63_1_65_0='undetermined; small-command scale only',small_pair_dx_mean=float(small.dx_px.mean()),step3='use six v58 measured vectors, provisional')
    (M.OUT/'additional_B_calibration.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
