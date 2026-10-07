"""Approval-pending sensitivity only. Stored v57 masks, same-date blank thresholds."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'field_level/v60_band_scar_controls'))
import field_control_common as M
import numpy as np,pandas as pd
from scipy.stats import mannwhitneyu,spearmanr
OUT=ROOT/'data/results/v62_provisional_scar_mask';OUT.mkdir(exist_ok=True)
SRC=ROOT/'data/results/v58_band_scar_causal/checkout/data/results/v57_false_positive_facts_20261006'
MASK=Path(r'C:\Users\chuya\PC-alignment-localcorr\data\results\v57_false_positive_facts_20261006\stepD')
CACHE=Path(r'W:\GoogleDrive\chuya2816\5.解析結果_chu\20260928_digital_judgment_current_alignment\tables\cached_field_differences')

def main():
    inv=pd.read_csv(SRC/'step0/fields_inventory.csv',dtype={'date':str,'board':str})
    info={};pools={};audit=[]
    for r in inv.itertuples():
        p=MASK/(r.fid+'_mask.npz');dp=MASK/(r.fid+'_pillar_dist.npz');cp=CACHE/(r.fid+'.npz')
        with np.load(cp) as z:d=z['delta'];xy=z['xy']
        with np.load(p) as z:
            shape=tuple(z['shape']);mask=np.unpackbits(z['mask'])[:np.prod(shape)].reshape(shape).astype(bool)
        xi=np.clip(np.rint(xy[:,0]).astype(int),0,shape[1]-1);yi=np.clip(np.rint(xy[:,1]).astype(int),0,shape[0]-1)
        inside=mask[yi,xi]
        if dp.exists():
            with np.load(dp) as z:assert np.array_equal(inside,z['in_mask'])
        info[r.fid]=(d,xy,inside)
        if r.group=='blank':
            for mode,good in [('before',np.ones(len(d),bool)),('after',~inside)]:pools.setdefault((r.date,mode),[]).append(d[good])
        audit.append(dict(fid=r.fid,mask_path=str(p),mask_sha256=M.fingerprint(p),cache_sha256=M.fingerprint(cp),source='v57 frozen mask',status='approval pending; sensitivity only'))
    thresholds={};tr=[]
    for (date,mode),vals in pools.items():
        d=np.concatenate(vals);th=float(d.mean()+3*d.std());thresholds[date,mode]=th
        tr.append(dict(date=date,mode=mode,threshold=th,mean=d.mean(),sd=d.std(),n=len(d),blank_fields=len(vals)))
    rows=[];maps={}
    fixed=set(pd.read_csv(SRC/'stepC/C1_outlier29_fields.csv').fid)
    for r in inv.itertuples():
        d,xy,inside=info[r.fid]
        for mode,good in [('before',np.ones(len(d),bool)),('after',~inside)]:
            th=thresholds[r.date,mode];met,dm=M.band_metrics(xy[good],d[good],th,True)
            rows.append(dict(fid=r.fid,date=r.date,board=r.board,field=r.field,condition=r.condition,group=r.group,mode=mode,mask_fraction=float(inside.mean()),threshold=th,**met))
            if r.fid in fixed:maps[r.fid+'_'+mode]=dm
    t=pd.DataFrame(rows);t.to_csv(OUT/'field_metrics.csv',index=False);pd.DataFrame(tr).to_csv(OUT/'blank_thresholds.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'input_audit.csv',index=False)
    dose=[];corr=[]
    for (date,mode),g in t.groupby(['date','mode']):
        blank=g[g.group=='blank'];an=g[g.group=='analyte'].copy();an['dose']=pd.to_numeric(an.condition)
        for concentration,c in an.groupby('dose'):
            test=mannwhitneyu(c.rate,blank.rate,alternative='greater',method='asymptotic')
            dose.append(dict(date=date,mode=mode,dose=concentration,fields=len(c),blank_fields=len(blank),mean=c.rate.mean(),median=c.rate.median(),blank_mean=blank.rate.mean(),p_one_sided=test.pvalue))
        means=an.groupby('dose').rate.mean();rho,p=spearmanr(means.index,means.values)
        corr.append(dict(date=date,mode=mode,rho=rho,p=p,unit='concentration means; descriptive'))
    dt=pd.DataFrame(dose)
    # Match production: Holm across all represented dates/concentrations, separately per mode.
    for mode,g in dt.groupby('mode'):
        vals=g.p_one_sided.to_numpy();order=np.argsort(vals);adj=np.empty(len(vals));adj[order]=np.minimum(1,np.maximum.accumulate((len(vals)-np.arange(len(vals)))*vals[order]));dt.loc[g.index,'p_holm']=adj
    dt.to_csv(OUT/'concentration_comparison.csv',index=False);pd.DataFrame(corr).to_csv(OUT/'dose_rank_correlation.csv',index=False)
    before=t[t['mode']=='before'].set_index('fid');after=t[t['mode']=='after'].set_index('fid').loc[before.index]
    fig,ax=M.plt.subplots(1,3,figsize=(14,4))
    for group,g in before.groupby('group'):
        ax[0].scatter(g.rate*100,after.loc[g.index].rate*100,s=10,label=group)
    ax[0].set(xlabel='Before positive rate (%)',ylabel='Approval-pending mask rate (%)');ax[0].legend(fontsize=7)
    cc=pd.DataFrame(corr)
    for mode,g in cc.groupby('mode'):ax[1].plot(g.date,g.rho,'o-',label=mode)
    ax[1].set(ylabel='Dose-mean Spearman rho');ax[1].tick_params(axis='x',rotation=60);ax[1].legend()
    for mode,g in dt.groupby('mode'):ax[2].scatter(np.log10(g.dose),g['mean']*100,s=15,label=mode)
    ax[2].set(xlabel='log10 concentration (M)',ylabel='Mean field rate (%)');ax[2].legend()
    fig.suptitle('Approval pending: sensitivity only, excluded from causal conclusions');fig.tight_layout();fig.savefig(OUT/'mask_effect.png',dpi=150);M.plt.close(fig)
    for fid in sorted(fixed):
        fig,ax=M.plt.subplots(1,2,figsize=(8,4));vmax=np.nanpercentile(maps[fid+'_before'],99)
        for a,mode in zip(ax,['before','after']):a.imshow(maps[fid+'_'+mode],vmin=0,vmax=vmax,cmap='magma');a.set_title(mode);a.axis('off')
        fig.suptitle(fid+'; provisional unapproved mask');fig.tight_layout();fig.savefig(OUT/(fid+'.png'),dpi=110);M.plt.close(fig)
    print('COMPLETE',len(inv),'fields',len(dt),'dose comparisons',flush=True)

if __name__=='__main__':main()
