from __future__ import annotations
import csv, gzip, json, re, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, norm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

ROOT=Path('C:\\Users\\chuya\\PC-alignment-localcorr')
sys.path.insert(0,str(ROOT))
from shared import registration as reg
from shared.lattice_indexing import lattice_from_fft, grid_coordinates
from shared.v2_registration_precision.refinement import register_refined

OUT=Path('C:\\Users\\chuya\\PC-alignment-localcorr\\data\\results\\v54_local_correction_decision_impact_20261006\\standard')
RAW=Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu')
REMOTE=Path('W:\\GoogleDrive\\chuya2816\\5.生データD_chu')
DATES={
 '260825':('260825_p50_dna',{0:0,1:1e-9,2:1e-10,3:1e-11,4:1e-12,5:1e-13,6:1e-14,7:1e-15,8:1e-12,9:1e-13}),
 '260827':('260827_pp50_dna',{1:'excluded',2:1e-10,3:'excluded',4:1e-12,5:1e-13,6:'excluded',7:1e-15,8:'blank_reference',9:'excluded',10:'excluded',11:0,12:1e-9}),
 '260828':('260828-p50-dna',{1:1e-9,2:1e-10,3:'excluded',4:1e-12,5:1e-13,6:'excluded',7:1e-15,8:0,9:1e-11,10:1e-14,11:'mismatch',12:'mismatch'}),
 '260829':('260829_p50_DNA',{1:1e-9,2:1e-10,3:1e-11,4:1e-12,5:1e-13,6:1e-14,7:1e-15,8:0,9:'mismatch',10:1e-12}),
 '260922':('260922-p50-dna',{1:1e-9,2:1e-10,3:1e-11,4:1e-12,5:1e-13,6:1e-14,7:1e-15,8:0,'01':'mismatch'}),
 '260923':('260923-p50-dna',{1:1e-9,2:1e-10,3:1e-11,4:1e-12,5:1e-13,6:1e-14,7:1e-15,8:0,9:'mismatch',10:'100uL_10fM','01':'mismatch'}),
 '260924':('260924-p50-dna',{1:1e-9,2:1e-10,3:1e-11,4:1e-12,5:1e-13,6:1e-14,7:1e-15,8:0,9:'mismatch'}),
 '260926':('260926-p50-dna',{1:1e-9,2:1e-10,3:1e-11,4:1e-12,5:1e-13,6:1e-14,7:1e-15,'01':0,'02':'mismatch'}),
 '260927':('260927-p50-dna',{4:1e-9,5:1e-10,6:1e-11,7:1e-12,8:1e-13,9:1e-14,10:1e-15,'02':0,'03':'mismatch'})}
NS=[1,1.5,2,2.5,3,3.5,4,5,6]
PITCH=7.286
SEED=20260928
PAT=re.compile(r'^(?:(\d+)-)?(\d+)-(\d+)-([01])(?: \(\d+\))?\.tiff?$',re.I)
EXCLUDED_927={('02',6),('03',2),('03',4)}
EXCLUDED_828={(11,f) for f in range(5,9)}

def board_value(s):
    return (int(s) if not (len(s)>1 and s.startswith('0')) else s) if s.isdigit() else s
def conc_label(v):
    if v=='mismatch': return 'ミスマッチDNA'
    if v==0:return 'ブランク'
    if v=='uncertain':return '未確定'
    if v=='100uL_10fM':return '10fM_100uL別扱い'
    names={1e-9:'1nM',1e-10:'100pM',1e-11:'10pM',1e-12:'1pM',1e-13:'100fM',1e-14:'10fM',1e-15:'1fM',1e-16:'100aM'}
    return names.get(v,str(v))
def read_tif(path):
    return reg.load_image_unicode(str(path))
def build_inventory():
    pairs=[]; inventory=[]; exceptions=[]
    excluded_root=RAW/'260925_p50_dna'
    excluded_files=list(excluded_root.glob('*.tif'))+list(excluded_root.glob('*.tiff')) if excluded_root.exists() else []
    inventory.append(dict(date='260925',folder=str(excluded_root),exists=excluded_root.exists(),image_files=len(excluded_files),boards='',fields='',paired=0,exclusion_reason='依頼者決定: 基板番号の記録失敗（4番欠落、7番相当が二基板）'))
    for date,(dirname,mapping) in DATES.items():
        root=(REMOTE if date in ('260825','260827','260828','260829') else RAW)/dirname
        files=list(root.glob('*.tif'))+list(root.glob('*.tiff'))
        inventory.append(dict(date=date,folder=str(root),exists=root.exists(),image_files=len(files),boards=0,fields=0,paired=0))
        found={}
        for p in files:
            m=PAT.match(p.name)
            if not m:
                exceptions.append(dict(date=date,path=str(p),reason='ファイル名が規則に一致しない'));continue
            mag,board,fov,phase=m.groups()
            if mag=='50':continue
            if mag not in (None,'100'):
                exceptions.append(dict(date=date,path=str(p),reason='対象外倍率'));continue
            key=(board_value(board),int(fov));found.setdefault(key,{})[int(phase)]=p
        inventory[-1]['boards']=len(set(k[0] for k in found)); inventory[-1]['fields']=len(found)
        for (board,fov),phases in found.items():
            if mapping.get(board)=='excluded' or (date=='260827' and board==11 and fov==7) or (date=='260828' and (board,fov) in EXCLUDED_828) or (date=='260927' and (str(board),fov) in EXCLUDED_927):
                exceptions.append(dict(date=date,path=str(root/f'{board}-{fov}'),reason='実験ログに基づく明示除外'))
                continue
            if mapping.get(board) is None:
                exceptions.append(dict(date=date,path=str(root/f'{board}-{fov}'),reason='基板番号と条件の対応が未登録'));continue
            if date=='260827' and board==8: # non-primary blank retained as a separate reference only
                mapping[board]='blank_reference'
            if date=='260927' and str(board) in ('02','03'):
                pass
            rec=dict(date=date,board=board,fov=fov,pre=phases.get(0),post=phases.get(1),root=str(root),condition=mapping.get(board,'unmapped'))
            if set(phases)!={0,1}:
                exceptions.append(dict(date=date,path=str(root/f'{board}-{fov}'),reason='洗浄前後画像の片方が欠落'))
                continue
            pairs.append(rec)
        inventory[-1]['paired']=sum(1 for x in pairs if x['date']==date)
    return pairs,inventory,exceptions

def compute(pairs):
    table=OUT/'tables'; cache=table/'cached_field_differences'; cache.mkdir(parents=True,exist_ok=True)
    rows=[]; all_delta=[]
    for i,p in enumerate(pairs,1):
        ident=f"{p['date']}_{p['board']}_{p['fov']}"; save=cache/(ident+'.npz')
        try:
            if save.exists():
                with np.load(save,allow_pickle=False) as z: delta,xy,ids=z['delta'],z['xy'],z['ids']
                qc={};ref={}
            else:
                pre,post=read_tif(p['pre']),read_tif(p['post'])
                coarse,qc=reg.register_image_pair_affine(pre,post,mask_stains=False,return_qc=True)
                matrix,ref=register_refined(pre,post,stage='subpixel',initial=coarse,mask_stains=False)
                lattice=lattice_from_fft(pre,PITCH); ids,xy=grid_coordinates(lattice,pre.shape[1],pre.shape[0],margin=30)
                postxy=xy@matrix[:,:2].T+matrix[:,2]
                aa=reg.sample_contrast(pre,xy); bb=reg.sample_contrast(post,postxy)
                valid=aa.valid_sampling.to_numpy()&bb.valid_sampling.to_numpy()
                delta=aa.contrast.to_numpy()[valid]-bb.contrast.to_numpy()[valid]
                xy=xy[valid]; ids=ids[valid]
                if len(delta)==0:raise ValueError('有効な格子標本がない')
                np.savez_compressed(save,delta=delta,xy=xy,ids=ids)
            p['condition']=p['condition']; p['concentration']=conc_label(p['condition']);p['path_pre']=str(p['pre']);p['path_post']=str(p['post'])
            p['n_pillars']=len(delta);p['delta_mean']=float(np.mean(delta));p['qc_accepted']=True;p['qc_reasons']='';p['qc_json']=json.dumps(qc,ensure_ascii=False,default=str);p['refinement_json']=json.dumps(ref,ensure_ascii=False,default=str)
            rows.append(p);all_delta.append((p,delta,xy,ids))
            print(f"{i}/{len(pairs)} {ident} {len(delta)}",flush=True)
        except Exception as e:
            p.update(concentration=conc_label(p['condition']),n_pillars=0,qc_accepted=False,qc_reasons=str(e),delta_mean=np.nan)
            rows.append(p)
            if i<=5 or i%25==0: print(f"{i}/{len(pairs)} {ident} FAIL {e}",flush=True)
    return rows,all_delta

def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True);(OUT/'tables').mkdir(parents=True,exist_ok=True)
    pairs,invrows,exceptions=build_inventory(); print('pairs',len(pairs),'inventory',invrows,flush=True)
    rows,data=compute(pairs)
    pd.DataFrame(invrows).to_csv(OUT/'tables/table_dataset_inventory.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(exceptions).to_csv(OUT/'tables/table_filename_exceptions.csv',index=False,encoding='utf-8-sig')
    df=pd.DataFrame([{k:v for k,v in r.items() if k not in ('pre','post')} for r in rows]);df.to_csv(OUT/'tables/table_registration_field_qc.csv',index=False,encoding='utf-8-sig')
    # Persist every sampled difference once; all multiplier analyses consume this table.
    with gzip.open(OUT/'tables/table_per_pillar_differences.csv.gz','wt',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['date','board','field','condition','concentration','pillar_index','lattice_m','lattice_n','x_px','y_px','delta_pre_minus_post'])
        for p,d,xy,ids in data:
            for j,(dv,pt,idx) in enumerate(zip(d,xy,ids)):w.writerow([p['date'],p['board'],p['fov'],p['condition'],p['concentration'],j,int(idx[0]),int(idx[1]),f'{pt[0]:.4f}',f'{pt[1]:.4f}',f'{dv:.10g}'])
    # assign same-date blank thresholds and summarize field rates for all n
    date_blanks={}
    for p,d,xy,ids in data:
        if p['condition']==0:date_blanks.setdefault(p['date'],[]).append(d)
    threshold_rows=[]; pct_rows=[]; fov_rows=[]; dist=[]
    for date,blanks in date_blanks.items():
        b=np.concatenate(blanks); mean=float(b.mean());sd=float(b.std(ddof=0));
        for n in NS:
            th=mean+n*sd;fp=float(np.mean(b>th));threshold_rows.append(dict(date=date,n=n,blank_mean=mean,blank_sd=sd,threshold=th,blank_n=len(b),blank_false_positive_rate=fp,normal_one_sided_rate=float(norm.sf(n))))
        for pct in [90,95,97.5,99,99.9]:pct_rows.append(dict(date=date,percentile=pct,threshold=float(np.percentile(b,pct)),blank_n=len(b),unstable=(pct==99.9 and len(b)<10000)))
        dist.append((date,b,mean,sd))
    for p,d,xy,ids in data:
        if p['date'] not in date_blanks:continue
        b=np.concatenate(date_blanks[p['date']]);mu=float(b.mean());sd=float(b.std(ddof=0));
        for n in NS:
            rate=float(np.mean(d>mu+n*sd));fov_rows.append(dict(date=p['date'],board=p['board'],field=p['fov'],condition=p['condition'],concentration=p['concentration'],n=n,pillar_total=len(d),positive=int((d>mu+n*sd).sum()),excess_rate=rate))
    ff=pd.DataFrame(fov_rows); thresholds=pd.DataFrame(threshold_rows)
    ff.to_csv(OUT/'tables/table_threshold_sweep_excess_rate_by_field.csv',index=False,encoding='utf-8-sig')
    thresholds.to_csv(OUT/'tables/table_threshold_sweep_thresholds_by_date.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pct_rows).to_csv(OUT/'tables/table_threshold_sweep_empirical_percentile_thresholds.csv',index=False,encoding='utf-8-sig')
    # Rates by dose and tests at field level. Concentration increases in reverse numerical order.
    stat=[]; dose_rows=[]
    for date in sorted(ff.date.unique()):
        for n in NS:
            sub=ff[(ff.date==date)&(ff.n==n)]
            conc=sub[sub.condition.apply(lambda x:isinstance(x,float) or isinstance(x,int) and x>0)]
            blank=sub[sub.condition==0]
            if len(conc) and len(blank):
                doses=np.asarray(conc.condition,float);rates=conc.excess_rate.to_numpy(); rho=float(spearmanr(doses,rates).statistic) if len(set(doses))>1 else np.nan
                tests=[]
                for dose in sorted(set(doses)):
                    x=conc.loc[conc.condition==dose,'excess_rate'].to_numpy();y=blank.excess_rate.to_numpy()
                    pv=float(mannwhitneyu(x,y,alternative='greater',method='asymptotic').pvalue) if len(x) and len(y) else np.nan
                    tests.append(pv)
                    dose_rows.append(dict(date=date,n=n,concentration=conc_label(dose),dose_molar=dose,field_n=len(x),pillar_total=int(conc.loc[conc.condition==dose,'pillar_total'].sum()),positive=int(conc.loc[conc.condition==dose,'positive'].sum()),mean=float(np.mean(x)),median=float(np.median(x)),min=float(np.min(x)),max=float(np.max(x)),blank_n=len(y),p_one_sided=pv))
                stat.append(dict(date=date,n=n,spearman_rho=rho,field_n=len(conc),blank_field_n=len(blank),minimum_p=float(np.nanmin(tests)),p_values=';'.join(map(str,tests))))
    dose_df=pd.DataFrame(dose_rows)
    if len(dose_df): dose_df['p_holm_global']=multipletests(dose_df.p_one_sided.to_numpy(),method='holm')[1]
    dose_df.to_csv(OUT/'tables/table_threshold_sweep_concentration_summary.csv',index=False,encoding='utf-8-sig')
    stats=pd.DataFrame(stat); stats.to_csv(OUT/'tables/table_threshold_sweep_statistics_by_date.csv',index=False,encoding='utf-8-sig')
    (OUT/'tables/table_threshold_sweep_multiple_testing.txt').write_text(f"総検定数={len(dose_df)}（画像が読めた解析日程={len(set(dose_df.date))}、倍率数={len(NS)}、各倍率の日程・濃度組み合わせの一日程片側マン・ホイットニーU検定）。報告表のp_one_sidedは未補正、p_holm_globalは全検定に対するHolm補正。260925は全解析から除外。",encoding='utf-8')
    # Plots: all days x multiplier panels; correlations and per-date minimum tests; blank rate and distributions.
    fig,axes=plt.subplots(len(NS),len(DATES),figsize=(22,23),sharex=True,sharey=True)
    for ri,n in enumerate(NS):
        for ci,date in enumerate(sorted(DATES)):
            ax=axes[ri,ci]; sub=ff[(ff.date==date)&(ff.n==n)]
            for cond,g in sub.groupby('condition'):
                if not isinstance(cond,(float,int)) or cond<=0:continue
                dose=-np.log10(cond);ax.scatter(np.full(len(g),dose),g.excess_rate,s=9,alpha=.65)
            ax.set_title(f'{date}, n={n}');ax.grid(alpha=.2)
    fig.supxlabel('濃度（対数尺度、M）');fig.supylabel('超過率');fig.tight_layout();fig.savefig(OUT/'figures/fig_excess_rate_vs_concentration_threshold_sweep.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    for date,g in stats.groupby('date'):ax.plot(g.n,g.spearman_rho,marker='o',label=date)
    ax.set_xlabel('閾値倍率 n');ax.set_ylabel('スピアマン順位相関係数');ax.legend(ncol=2);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUT/'figures/fig_spearman_vs_threshold_multiplier_all_dates.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    for date,g in thresholds.groupby('date'):ax.plot(g.n,g.blank_false_positive_rate,marker='o',label=f'{date} 実測')
    ax.plot(NS,[norm.sf(n) for n in NS],color='black',ls='--',label='正規分布の理論値');ax.set_yscale('log');ax.set_xlabel('閾値倍率 n');ax.set_ylabel('ブランク偽陽性率');ax.legend(ncol=2);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUT/'figures/fig_blank_false_positive_vs_threshold_multiplier.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,(date,b,mu,sd) in zip(axes.flat,dist):
        ax.hist(b,bins=100,log=False,color='#4677a8',alpha=.8)
        for n in NS:ax.axvline(mu+n*sd,lw=.7,alpha=.55,label=f'n={n:g}')
        ax.set_title(date);ax.set_yscale('log');ax.set_xlabel('洗浄前 − 洗浄後 差分');ax.set_ylabel('ブランクのピラー数（対数）')
    fig.tight_layout();fig.savefig(OUT/'figures/fig_blank_difference_histograms_log_all_dates.png',dpi=160);plt.close(fig)
    # Persist config and compact run manifest.
    (OUT/'scripts_and_config'/'threshold_sweep_config.json').write_text(json.dumps({'dates':DATES,'n_values':NS,'seed':SEED,'pitch_px':PITCH,'difference':'pre minus post','blank_std_ddof':0,'random_direction_sensitivity_seed':SEED},indent=2,ensure_ascii=False),encoding='utf-8')
    print('DONE',flush=True)
if __name__=='__main__':main()
