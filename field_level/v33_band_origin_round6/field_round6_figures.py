"""Five field contact views and matched display crops, with recovery state visible."""
from field_round6_common import *
from field_round6_clusters import raw_index,recovery_index,recover
os.environ['MPLCONFIGDIR']=str(OUT/'plot_settings')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Yu Gothic'
plt.rcParams['font.size']=9
FIVE=['260926_3_3','260926_5_3','260926_7_5','260926_7_8','260926_01_8']

def cell_map(xy,value):
    cell=(xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int)
    n=np.bincount(cell,minlength=4096);s=np.bincount(cell,weights=value,minlength=4096)
    return np.divide(s,n,out=np.full(4096,np.nan),where=n>0).reshape(64,64)
def lines(ax,angle,period,center,color,style):
    if not np.isfinite(period) or period<=0:return
    vec=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))]);normal=np.array([-vec[1],vec[0]])
    for step in range(-int(3000/period)-1,int(3000/period)+2):
        c=center+normal*step*period;pts=c+np.array([-3000,3000])[:,None]*vec
        ax.plot(pts[:,0],pts[:,1],color=color,linestyle=style,linewidth=.9,alpha=.8)
def crop_centers(xy,delta,threshold,angle,period):
    normal=np.array([-np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle))]);dm=cell_map(xy,delta>threshold)
    coords=np.column_stack([((xx+.5)*32).ravel(),((yy+.5)*32).ravel()]);valid=np.isfinite(dm.ravel())&np.all((coords>[384,384])&(coords<[1664,1660]),axis=1)
    # Display-only selection of a locally positive-rich cell near the central area.
    density=cv2.GaussianBlur(np.nan_to_num(dm).astype('float32'),(0,0),1.5).ravel()
    ix=int(np.argmax(np.where(valid,density,-1)));on=coords[ix]
    shift=min(float(period)/2 if np.isfinite(period) else 200.,512.)
    choices=[on+normal*shift,on-normal*shift]
    between=min(choices,key=lambda c:np.linalg.norm(np.clip(c,[128,128],[1920,1916])-c))
    return np.clip(on,[128,128],[1920,1916]).astype(int),np.clip(between,[128,128],[1920,1916]).astype(int)

def run():
    verify();f=fields().set_index('key',drop=False);manifest=raw_index();rec=recovery_index();_,_,caches,_=discover()
    cl=table(OLD/'round1_spatial_classification.csv').query('縁除外画素 == 0').set_index(['key','定義'])
    thr=table(OLD/'round1_thresholds.csv').set_index('日程');dest=OUT/'figures';dest.mkdir(exist_ok=True);rows=[]
    for key in FIVE:
        cache=arrays(caches[key]);xy=cache['xy'];d=cache['delta'];pre=read_image(manifest[key]['洗浄前']);postnative=read_image(manifest[key]['洗浄後']);a,state=recover(key,cache,rec)
        post=None
        if a is not None:
            post=cv2.warpAffine(postnative,a['matrix'],(2048,2044),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        c=cl.loc[(key,METHODS[0])];threshold=float(thr.loc['260926','平均標準偏差']);robust=float(thr.loc['260926','中央値絶対偏差閾値'])
        on,between=crop_centers(xy,d,threshold,float(c['第一帯軸度']),float(c['第一スペクトル周期画素']))
        delta=cell_map(xy,d);scale=float(np.nanpercentile(abs(delta),99));scale=max(scale,.001)
        lo,hi=np.percentile(pre,[1,99.5]);fig,axes=plt.subplots(2,3,figsize=(15,10))
        extent=[0,2048,2044,0]
        axes[0,0].imshow(pre,cmap='gray',vmin=lo,vmax=hi,extent=extent);axes[0,0].set_title('(1) 洗浄前の生像')
        axes[0,1].imshow(post if post is not None else postnative,cmap='gray',vmin=lo,vmax=hi,extent=extent)
        axes[0,1].set_title('(2) 洗浄後の生像'+('：前像座標へ再標本化' if post is not None else '：位置合わせ回復不能・固有座標'))
        im=axes[0,2].imshow(delta,cmap='coolwarm',vmin=-scale,vmax=scale,extent=extent);axes[0,2].set_title('(3) 保存差：洗浄前から洗浄後を引く\n32ピクセル升目のピラー差平均');fig.colorbar(im,ax=axes[0,2],fraction=.045)
        axes[1,0].scatter(xy[d>robust,0],xy[d>robust,1],s=.3,c='#da9900',label='中央値と絶対偏差')
        axes[1,0].scatter(xy[d>threshold,0],xy[d>threshold,1],s=.3,c='#b52245',label='平均と標準偏差');axes[1,0].set_title('(4) 外れ値の位置：二定義');axes[1,0].legend(markerscale=8,loc='upper right')
        dm=cell_map(xy,d>threshold);axes[1,1].imshow(dm,cmap='magma',vmin=0,vmax=max(float(np.nanpercentile(dm,99)),.05),extent=extent)
        lines(axes[1,1],float(c['第一帯軸度']),float(c['第一スペクトル周期画素']),on,'cyan','-')
        if c['分類']=='うろこ状候補':lines(axes[1,1],float(c['第二帯軸度']),float(c['第二スペクトル周期画素']),on,'lime','--')
        axes[1,1].set_title('(5) 周1の候補軸とスペクトル周期\n'+str(c['分類']))
        for ax in (axes[0,0],axes[0,2],axes[1,0],axes[1,1]):
            for center,color in ((on,'cyan'),(between,'lime')):ax.add_patch(plt.Rectangle(center-128,256,256,fill=False,edgecolor=color,linewidth=1))
        text=f"{key}\n日程260926・基板{f.loc[key,'基板']}・視野{f.loc[key,'視野番号']}\n濃度：{f.loc[key,'濃度']}\n差は保存された背景除去後9画素和の差。\n画像の前後表示は共通の輝度範囲。\n平均と標準偏差の閾値 {threshold:.6g}\n中央値と絶対偏差の閾値 {robust:.6g}\n第一軸 {c['第一帯軸度']:.1f}度\n第一スペクトル周期 {c['第一スペクトル周期画素']:.1f}ピクセル\n候補線の位相は表示用。物理的な帯境界ではない。\n高速フーリエ変換の整数調波は細帯の証拠にしない。\n位置合わせ：{'数値回復条件一致' if state['回復'] else '回復不能。後像を共通座標へ重ねていない。'}\n水色：帯候補上の切り出し\n緑：半周期ずらした表示用切り出し\n両切り出しは検定へ使わない。"
        axes[1,2].axis('off');axes[1,2].text(0,1,text,va='top',transform=axes[1,2].transAxes,fontsize=10,linespacing=1.6)
        for ax in axes.ravel()[:5]:ax.set_xlim(0,2048);ax.set_ylim(2044,0);ax.set_xlabel('横位置（ピクセル）');ax.set_ylabel('縦位置（ピクセル）')
        fig.suptitle(key+'：前像を基準とした五種類の表示',fontsize=14);fig.tight_layout();fig.savefig(dest/(key+'_並べた図.png'),dpi=160);plt.close(fig)
        fig,ax=plt.subplots(2,3,figsize=(12,8));points=[]
        for row,center in enumerate((on,between)):
            cx,cy=center;x0,x1=cx-128,cx+128;y0,y1=cy-128,cy+128
            ax[row,0].imshow(pre[y0:y1,x0:x1],cmap='gray',vmin=lo,vmax=hi,extent=[x0,x1,y1,y0]);ax[row,0].set_title(('帯候補上' if row==0 else '帯候補間')+'：洗浄前')
            if post is None:ax[row,1].axis('off');ax[row,1].text(.1,.5,'後像の位置合わせ回復不能\n同座標の切り出しは表示しない',transform=ax[row,1].transAxes)
            else:ax[row,1].imshow(post[y0:y1,x0:x1],cmap='gray',vmin=lo,vmax=hi,extent=[x0,x1,y1,y0]);ax[row,1].set_title('洗浄後：前像座標')
            select=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
            lim=float(np.percentile(abs(d[select]),95)) if select.any() else scale
            ax[row,2].scatter(xy[select,0],xy[select,1],c=d[select],cmap='coolwarm',vmin=-lim,vmax=lim,s=8)
            positive=select&(d>threshold);ax[row,2].scatter(xy[positive,0],xy[positive,1],s=18,facecolors='none',edgecolors='black',linewidths=.4)
            ax[row,2].set_title('保存ピラー差：輪郭は平均と標準偏差の外れ値')
            for col in (0,1,2):
                if col!=1 or post is not None:
                    ax[row,col].set_xlim(x0,x1);ax[row,col].set_ylim(y1,y0)
                    ax[row,col].plot(cx,cy,marker='+',markersize=13,markeredgewidth=1.2,color='cyan' if row==0 else 'lime')
            points.append(dict(中心横=int(cx),中心縦=int(cy)))
        fig.suptitle(key+'：256ピクセル四方、十字が選択中心。短周期では帯と間が同じ切り出しに含まれる。');fig.tight_layout();fig.savefig(dest/(key+'_拡大切り出し.png'),dpi=160);plt.close(fig)
        rows.append(dict(key=key,**state,洗浄前パス=str(manifest[key]['洗浄前']),洗浄後パス=str(manifest[key]['洗浄後']),保存差パス=str(caches[key]),分類=c['分類'],第一軸度=float(c['第一帯軸度']),第一周期画素=float(c['第一スペクトル周期画素']),切り出し=points,全景=str(dest/(key+'_並べた図.png')),拡大=str(dest/(key+'_拡大切り出し.png'))))
        print('figure',key,state['回復'],flush=True)
    dump(rows,OUT/'figure_manifest.json')
if __name__=='__main__':run()
