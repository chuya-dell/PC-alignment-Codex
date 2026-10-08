"""Read-only independent checks of v70-v80; writes only v81 audit artifacts.

Original routines are imported only for small controlled generator/readout checks.
No original main/one writer, provisional-mask loader, or localcorr worktree is used.
Run: python -X utf8 field_level/v81_independent_audit/field_audit_ah.py tables|geometry|synthetic|dark
"""
from __future__ import annotations
import sys, json, hashlib, os, types
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT = ROOT / 'data/results/v81_independent_audit'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig')
# v60's helper has import-time writes to its own output directory.  Replace
# only that helper import with read-only inventory/read operations for this audit.
helper=types.ModuleType('field_control_common')
helper.RAW=Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu')
helper.folder=lambda name: next(p for p in helper.RAW.iterdir() if p.is_dir() and p.name==name)
helper.read=lambda p: __import__('tifffile').imread(p).astype(np.float32)
sys.modules['field_control_common']=helper
for part in ['v70_ledger_center_fit', 'v72_real_field_readout', 'v75_zero_truth_pairs', 'v60_band_scar_controls']:
    sys.path.insert(0, str(ROOT / 'field_level' / part))
import numpy as np
import pandas as pd
import cv2, tifffile
from scipy import ndimage
from scipy.stats import spearmanr, mannwhitneyu, chi2
from scipy.spatial import cKDTree
cv2.setNumThreads(1)
RES = ROOT / 'data/results'
V70 = RES / 'v70_ledger_center_fit'
V72 = RES / 'v72_real_field_readout'
V78 = RES / 'v78_candidate_path'
LED = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}).set_index('fid')

def save(name, value):
    (OUT / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf-8')
    print(name, json.dumps(value, ensure_ascii=False, default=str), flush=True)

def robust(v):
    med = np.median(v)
    return float(med), float(1.4826 * np.median(abs(v-med)))

def sub(z, side):
    ctr = z[f'{side}_ctr'].astype(float)
    pred, amp, aloc, wid = [z[f'{side}_{name}'] for name in ['pred', 'amp', 'aloc', 'wid']]
    bad = (np.linalg.norm(ctr-pred, axis=1) > .25*7.286) | (amp < .4*aloc) | (wid < .4)
    for i, j in cKDTree(ctr).query_pairs(2.):
        bad[i if amp[i] < amp[j] else j] = True
    return bad

def table_checks():
    t = pd.read_csv(V70 / 'correspondence_by_field_v2.csv')
    fixed = t[t.fixed29]
    p = pd.read_csv(V70 / 'period_fix.csv').merge(LED[['fixed29','group']], left_on='fid',right_index=True)
    shifted = (p.shift_a != 0) | (p.shift_b != 0)
    corr = {'fields': len(t), 'rate_median':t.rate_main.median(), 'rate_min':t.rate_main.min(), 'fixed29_median':fixed.rate_main.median(), 'fixed29_min':fixed.rate_main.min(),
        'corner_median':t.corner_disagree_px.median(), 'corner_p90':t.corner_disagree_px.quantile(.9), 'U_det_counts':{},
        'period_corrected':int(shifted.sum()), 'fixed29_corrected':int((shifted&p.fixed29).sum()), 'blank_corrected':int((shifted&(p.group=='blank')).sum()), 'unresolved':int((~p.resolved).sum())}
    save('table_A',corr)
    fp = pd.read_csv(V78 / 'blank_fp_by_k.csv',dtype={'date':str})
    rows=[]; loo=[]
    for readout,g in fp.groupby('readout'):
        def choose(q):
            for k,h in q.groupby('k',sort=True):
                if np.quantile(h.fp91k,.95) <= 2: return float(k), True
            return 15.,False
        k,reached = choose(g); q=g[g.k==k]
        rows.append(dict(readout=readout,k=k,reached=reached,n=len(q),mean=q.fp91k.mean(),p95=q.fp91k.quantile(.95),max=q.fp91k.max(),gt2=int((q.fp91k>2).sum())))
        for date in sorted(g.date.unique()):
            kd,ok=choose(g[g.date!=date]); h=g[(g.date==date)&(g.k==kd)]
            loo.append(dict(readout=readout,date=date,k=kd,train_reached=ok,test_p95=h.fp91k.quantile(.95),test_max=h.fp91k.max(),n=len(h)))
    pd.DataFrame(loo).to_csv(OUT/'calibration_loo.csv',index=False)
    save('table_H',rows)
    # Percentile of realized counts is not an upper confidence limit for a future field/mean.
    q=fp[(fp.readout=='S5')&(fp.k==10.5)]
    vals=q.fp91k.to_numpy(); rng=np.random.default_rng(81)
    qb=np.quantile(vals[rng.integers(0,len(vals),(10000,len(vals)))],.95,axis=1)
    save('H_percentile_uncertainty',dict(p95=np.quantile(vals,.95),bootstrap_p95_CI=np.quantile(qb,[.025,.975]).tolist(),prob_p95_above2=float((qb>2).mean()),max=float(vals.max()),above2=int((vals>2).sum())))
    # Recompute all DNA tests from field-level rates using independent Holm expression.
    dna=[]
    saved=pd.read_csv(V78/'dna_dose_tests.csv',dtype={'date':str})
    for path in ['old_standard_3SDpool','new_S5_k10.5_scarexcl','new_S5_k4_scarexcl','new_S5_k6_scarexcl']:
        rt=pd.read_csv(V78/f'dna_rates_{path}.csv'); d=LED.reset_index().merge(rt[['fid','rate']],on='fid'); rr=[]
        for date,h in d.groupby('date'):
            bl=h[h.group=='blank'].rate.dropna()
            for conc,a in h[h.group=='analyte'].groupby('conc_M'):
                x=a.rate.dropna()
                if len(bl)>=3 and len(x)>=3:
                    rr.append((date,conc,float(mannwhitneyu(x,bl,alternative='greater').pvalue)))
        pp=np.array([x[2] for x in rr]); order=np.argsort(pp); adj=np.zeros(len(pp)); adj[order]=np.minimum(1,np.maximum.accumulate(pp[order]*np.arange(len(pp),0,-1)))
        ref=saved[saved.path==path].sort_values(['date','conc_M'])
        dna.append(dict(path=path,tests=len(pp),raw_lt05=int((pp<.05).sum()),holm_lt05=int((adj<.05).sum()),max_raw_error=float(np.max(abs(pp-ref.p.to_numpy()))),max_holm_error=float(np.max(abs(adj-ref.p_holm.to_numpy())))))
    save('DNA_recomputed',dna)
    focus=pd.read_csv(V78/'focus_proxy_by_field.csv')
    save('D_associations',dict(all_rho=float(spearmanr(focus.la_iqr,focus.mad).statistic),blank_rho=float(spearmanr(focus[focus.group=='blank'].la_iqr,focus[focus.group=='blank'].mad).statistic),positive_density_rho_median=float(focus.rho_pos_vs_abs_la.median()),mean_delta_rho_median=float(focus.rho_dm_vs_la.median())))
    # Injection recovery table independently recalculated, without trusting authors' summaries.
    rows=[]
    cal=pd.read_csv(V78/'calibration_k.csv')
    for suffix in ['P2','P3_high']:
        inj=pd.read_csv(RES/f'v77_signal_injection_{suffix}'/'injection_rows.csv')
        for rd in ['S0','S5','S5P']:
            k=float(cal[(cal.readout==rd)&(cal.scope=='all_blanks')&(cal.limit=='main2')].k.iloc[0])
            for (amp,direction),g in inj[inj.readout==rd].groupby(['A','dir']):
                detected=g.valid & (g.snr.abs()>k if direction=='bright' else g.snr>k)
                rows.append(dict(run=suffix,readout=rd,A=amp,dir=direction,k=k,n=len(g),valid=float(g.valid.mean()),recovery=float(detected.mean()),median_ratio=float(g.ratio.median())))
        if suffix=='P2':
            labs=inj.label.unique().tolist(); infos=pd.read_csv(RES/'v77_signal_injection_P2/injection_info.csv')
            periods=p.set_index('fid').reindex([s for s in labs if s!='burst'])
            save('G_labels_and_period_shifts',dict(labels=labs,period_shifts=periods[['shift_a','shift_b','shift_px']].to_dict('index'),infos=infos.to_dict('records')))
    pd.DataFrame(rows).to_csv(OUT/'G_recovery_recomputed.csv',index=False)
    inj=pd.read_csv(RES/'v77_signal_injection_P2/injection_rows.csv');labels=sorted(inj.label.unique());bias=[]
    for rd in ['S0','S5','S5P']:
        h=inj[(inj.readout==rd)&inj.valid];gain=float(h[h.label.isin(labels[::2])].ratio.median());test=h[h.label.isin(labels[1::2])].copy();test['corrected']=test.ratio/gain
        conditions=test.groupby(['A','dir','zone','pattern']).corrected.median()
        bias.append(dict(readout=rd,gain=gain,calibrated_median=float(test.corrected.median()),condition_min=float(conditions.min()),condition_max=float(conditions.max()),conditions=len(conditions),within10pct=int(((conditions>.9)&(conditions<1.1)).sum()),cases_calibration=labels[::2],cases_test=labels[1::2]))
    save('G_bias_recomputed',bias)
    # Recalculate one 91k-normalized count by direct array arithmetic on each blank.
    direct=[]
    for fid in LED[LED.group=='blank'].index:
        with np.load(V72/'fields'/f'{fid}.npz') as a,np.load(V78/'fields'/f'{fid}.npz') as b:
            d=a['S5'].astype(float); v=d[np.isfinite(d)&(b['sd_periods']>2)]
            med,mad=robust(v); n=int((v>med+10.5*mad).sum()); ref=q[q.fid==fid].iloc[0]
            direct.append(dict(fid=fid,n_valid=len(v),raw_count=n,count91k=n/len(v)*91000,saved_count91k=ref.fp91k,error=n/len(v)*91000-ref.fp91k,mad=mad))
    dr=pd.DataFrame(direct);dr.to_csv(OUT/'H_blank_direct.csv',index=False)
    save('H_direct_summary',dict(fields=len(dr),max_count_error=dr.error.abs().max(),raw_count_max=int(dr.raw_count.max()),raw_fields_gt2=int((dr.raw_count>2).sum()),total_count=int(dr.raw_count.sum()),total_valid=int(dr.n_valid.sum())))
    factors=pd.read_csv(V78/'factor_table_counts.csv'); rr=[]
    for sel,name in [(factors.group=='blank','blank69'),(factors.fixed29,'fixed29')]:
        for rd,g in factors[sel&(factors.k==4)].groupby('readout'):
            rr.append(dict(selection=name,readout=rd,mad_median=g.mad.median(),median_absolute_count=g.frac_gt_0p2.median()))
    save('C_factor_table',rr)

def geometry_checks():
    # Ordinary blank, fixed29 large affine error, and worst correspondence: distinct selection rules.
    tab=pd.read_csv(V70/'correspondence_by_field_v2.csv').set_index('fid')
    per=pd.read_csv(V70/'period_fix.csv').set_index('fid')
    fids=['260825_0_1',tab[tab.fixed29].corner_disagree_px.idxmax(),tab.rate_main.idxmin(),per.shift_px.idxmax()]
    rows=[]
    c0=np.array([1024.,1022.]); corner=np.array([[0.,0.],[2048.,0.],[0.,2044.],[2048.,2044.]])-c0
    for fid in dict.fromkeys(fids):
        with np.load(V70/'fields2'/f'{fid}.npz') as z:
            Bp=z['pre_model'][1:].T; Bo=z['post_model'][1:].T; A=z['A']; U=np.rint(np.linalg.solve(Bo,A[:,:2]@Bp)).astype(int)
            L=Bo@U@np.linalg.inv(Bp); diff=np.linalg.norm(corner@(A[:,:2]-L).T,axis=1).max()
            exp=z['pre_pred_lin']@L.T+z['tL']; dist,j=cKDTree(z['post_pred_lin']).query(exp)
            idx_delta=z['post_ids'][j]-z['pre_ids']@U.T; choices,n=np.unique(idx_delta[dist<1.8],axis=0,return_counts=True); mode=choices[np.argmax(n)]
            cons=np.all(idx_delta==mode,axis=1)&(dist<2.)
            ov=(exp[:,0]>=8)&(exp[:,0]<2040)&(exp[:,1]>=8)&(exp[:,1]<2036)
            good=ov&~sub(z,'pre')&~sub(z,'post')[j]&cons
            rate=good.sum()/ov.sum()
            rows.append(dict(fid=fid,U=U.tolist(),U_det=float(np.linalg.det(U)),U_matches_saved=bool(np.array_equal(U,z['U'])),L_max_error=float(abs(L-z['Llin']).max()),corner_px=float(diff),corner_saved=float(tab.loc[fid].corner_disagree_px),index_shift_mode=mode.tolist(),partner_disagreements=int((j!=z['jL']).sum()),numerator=int(good.sum()),denominator=int(ov.sum()),rate=float(rate),saved_rate=float(tab.loc[fid].rate_main),period_shift=per.loc[fid,['shift_a','shift_b','shift_px']].to_dict()))
    save('A_geometry_independent',rows)
    # Increment errors recomputed from saved transforms, no author evaluator invoked.
    S=RES/'v75_zero_truth_pairs/synth'; b=np.load(S/'shift_0.00_0.00.npz'); xy=b['ctr_xy'].astype(float)
    errors={'A':[],'L':[]}; rr=[]
    for path in sorted(S.glob('*.npz')):
        lab=path.stem; z=np.load(path)
        if lab.startswith('shift'):
            dx,dy=map(float,lab.split('_')[1:]); tr=np.tile([dx,dy],(len(xy),1))
        else:
            deg=float(lab.split('_')[1]); ang=-np.deg2rad(deg); rotation=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
            tr=(xy-c0)@rotation.T+c0-xy
        for name,mat,trans in [('A','A',None),('L','Llin','tL')]:
            if trans is None: inc=xy@(z[mat][:,:2]-b[mat][:,:2]).T+z[mat][:,2]-b[mat][:,2]
            else: inc=xy@(z[mat]-b[mat]).T+z[trans]-b[trans]
            e=np.linalg.norm(inc-tr,axis=1); errors[name].append(e); rr.append(dict(label=lab,map=name,p95=float(np.quantile(e,.95)),median=float(np.median(e))))
    pd.DataFrame(rr).to_csv(OUT/'B_increment_errors.csv',index=False)
    save('B_increment_p95',{name:float(np.quantile(np.concatenate(x),.95)) for name,x in errors.items()})

def synthetic_checks():
    import v75_run_pairs as generator
    n=128; yy,xx=np.mgrid[:n,:n]; sigma=4.; center=np.array([64.2,63.7])
    image=np.exp(-((xx-center[0])**2+(yy-center[1])**2)/(2*sigma**2)).astype(np.float32)
    fine=generator.upsample4(image); rr=[]
    for dx,dy in [(0,0),(.25,.5),(.75,.75)]:
        out=generator.integrate4(generator.synth_shift(fine,dx,dy)).astype(float)
        cm=np.array([(out*xx).sum(),(out*yy).sum()])/out.sum()
        rr.append(dict(dx=dx,dy=dy,measured_shift=(cm-center).tolist(),error_vs_named_shift=(cm-center-[dx,dy]).tolist(),error_norm=float(np.linalg.norm(cm-center-[dx,dy]))))
    # Alternating-row/column Nyquist energy is not split into conjugate endpoints in upsample4.
    nyquist=(-1.)**xx
    up=generator.upsample4(nyquist.astype(np.float32)); reconstruct=up[::4,::4]
    import v72_readout as R
    save('B_synthetic_generator',dict(gaussian_centroid=rr,nyquist_integer_reconstruction_max_error=float(abs(reconstruct-nyquist).max()),aperture_area=float(len(R.AP_OFF)*R.AP_W),aperture_centroid=R.AP_OFF.mean(0).tolist(),statement='integrate4 samples pixel n at fine locations n+[0,.25,.5,.75]; its centroid is n+.375, so absolute content displacement is named shift minus .375 on both axes.'))

def dark_checks():
    import v72_readout as R
    parent=Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu')
    found={p.name:p for p in parent.iterdir() if p.is_dir()}
    darkfolder=found['260904_p50_暗時']; burstfolder=found['260904_p50_repeat']
    paths=sorted(darkfolder.glob('*.tif')); ims=[tifffile.imread(p).astype(np.float32) for p in paths]
    a,b=ims[:2]; light=[tifffile.imread(burstfolder/f'{i}.tif').astype(np.float32) for i in [1,2]]
    bd=R.box_contrast(a)-R.box_contrast(b); bl=R.box_contrast(light[0])-R.box_contrast(light[1])
    rr=[]
    # Same actual pre/post aperture, positions and validity as the existing burst trial.
    with np.load(RES/'v75_zero_truth_pairs/real/burst_1_2.npz') as z:
        xy=z['ctr_xy'].astype(float)
        # Stored v75 burst precedes the P implementation; use its lattice map at
        # the actual pre centres, and label this check explicitly as a mapped aperture.
        pos=xy@z['Llin'].T+z['tL']; valid=z['ok']
        dp=R.aper(R.plain_contrast(a),xy)-R.aper(R.plain_contrast(b),pos)
        lp=R.aper(R.plain_contrast(light[0]),xy)-R.aper(R.plain_contrast(light[1]),pos)
        for name,da,la in [('same_box_grid',bd.ravel(),bl.ravel()),('L_mapped_actual_aperture',dp[valid],lp[valid])]:
            da=da[np.isfinite(da)];la=la[np.isfinite(la)]
            rr.append(dict(operator=name,dark_SD=float(da.std()),light_SD=float(la.std()),dark_robust_SD=robust(da)[1],light_robust_SD=robust(la)[1],ratio_variances=float(da.var()/la.var())))
    meta=[]
    for p in paths+[burstfolder/'1.tif',burstfolder/'2.tif']:
        with tifffile.TiffFile(p) as f:
            relevant={t.name:str(t.value)[:200] for t in f.pages[0].tags.values() if any(s in t.name.lower() for s in ['date','description','exposure','gain','software'])}
            meta.append(dict(file=str(p),shape=f.pages[0].shape,tags=relevant,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    save('F_dark_operator_check',dict(per_frame_temporal_pixel_SD=float((a-b).std()/np.sqrt(2)),operators=rr,metadata=meta))

def native_injection_check():
    import v75_pair_lib as P
    import v77_inject as G
    fid='260927_02_1'; r=LED.loc[fid]
    a=tifffile.imread(r.pre_path).astype(np.float32); b=tifffile.imread(r.post_path).astype(np.float32)
    z=np.load(V70/'fields2'/f'{fid}.npz'); ctr=z['pre_ctr'].astype(float); postctr=z['post_ctr'].astype(float)[z['jL']]
    native=P.analyse_pair(a,b)
    pts=np.array([[512.,512.],[1536.,512.],[512.,1536.],[1536.,1536.],[1024.,1022.]])
    def map_disagreement(res):
        return np.linalg.norm((pts@res['Llin'].T+res['tL'])-(pts@z['Llin'].T+z['tL']),axis=1).tolist()
    print('native baseline finished',fid,map_disagreement(native),flush=True)
    # Controlled isolated +/-10sigma changes at corrected physical partners.
    ok=np.load(V72/'fields'/f'{fid}.npz')['ok']; v=np.load(V72/'fields'/f'{fid}.npz')['S3']; sigma=robust(v[np.isfinite(v)])[1]
    candidates=np.flatnonzero(ok&(ctr[:,0]>300)&(ctr[:,0]<1748)&(ctr[:,1]>300)&(ctr[:,1]<1744)); rng=np.random.default_rng(81);rng.shuffle(candidates)
    ids=[]
    for i in candidates:
        if all(np.linalg.norm(ctr[i]-ctr[j])>100 for j in ids):ids.append(i)
        if len(ids)==48:break
    f=b.astype(float)/65535; ids=np.array(ids); clipping=[]
    for number,i in enumerate(ids):
        sign=1 if number<24 else -1
        G.blob(f,*postctr[i],sign*10*sigma/G.S0_BOX)
    clipping_pixels=int(((f<0)|(f>1)).sum());bi=np.clip(f*65535,0,65535).astype(np.float32)
    injected=P.analyse_pair(a,bi)
    rows=[]
    for rd in ['S0','S5','S5P']:
        xy=injected['xy0'] if rd=='S0' else injected['ctr_xy']; d=injected[rd].astype(float);med,mad=robust(d[np.isfinite(d)])
        dist,j=cKDTree(xy).query(ctr[ids]); vals=d[j]; target=np.where(np.arange(len(ids))<24,-10*sigma,10*sigma)
        for direction,sel in [('bright',np.arange(len(ids))<24),('dim',np.arange(len(ids))>=24)]:
            vv=vals[sel]; rows.append(dict(readout=rd,dir=direction,n=int(sel.sum()),valid=float(np.isfinite(vv).mean()),median_ratio=float(np.nanmedian((vv-med)/target[sel])) if np.isfinite(vv).any() else None))
    save('G_native_pipeline_period',dict(fid=fid,baseline_map_error_vs_period_corrected=map_disagreement(native),injected_map_error_vs_period_corrected=map_disagreement(injected),baseline_tL=native['tL'].tolist(),corrected_tL=z['tL'].tolist(),injected_tL=injected['tL'].tolist(),clipped_pixels=clipping_pixels,recovery_rows=rows,note='audit intervention is 24 bright +24 dim isolated 10sigma sites; original 36-condition job not reproduced'))

def band(xy,d,threshold):
    ok=np.isfinite(d)&np.isfinite(xy).all(1);xy=xy[ok];d=d[ok]
    cell=np.clip((xy[:,1]//32).astype(int),0,63)*64+np.clip((xy[:,0]//32).astype(int),0,63)
    n=np.bincount(cell,minlength=4096);positive=np.bincount(cell,weights=d>threshold,minlength=4096)
    density=np.divide(positive,n,out=np.full(4096,np.nan),where=n>0).reshape(64,64)
    image=np.where(np.isfinite(density),density-np.nanmean(density),0)*np.outer(np.hanning(64),np.hanning(64))
    f=np.fft.rfft2(image);fy=np.fft.fftfreq(64,d=32)[:,None];fx=np.fft.rfftfreq(64,d=32)[None,:];r=np.hypot(fy,fx)
    axis=(np.degrees(np.arctan2(np.broadcast_to(fy,r.shape),np.broadcast_to(fx,r.shape)))+90)%180
    mult=np.full(r.shape,2.);mult[:,0]=1;mult[:,-1]=1;power=abs(f)**2*mult
    inband=(r>=1/1024)&(r<=1/128);directions=np.floor(((axis+5)%180)/10).astype(int)
    sums=np.array([power[inband&(directions==i)].sum() for i in range(18)])
    direction=float(sums.argmax()*10);keep=inband&(abs((axis-direction+90)%180-90)<=15)
    peak=np.unravel_index(np.argmax(np.where(keep,power,-1)),r.shape)
    return dict(band_power=float(power[inband].sum()),axis_deg=direction,period_px=float(1/r[peak]),rate=float((d>threshold).mean()))

def structure_checks():
    # Independently reproduce stored zero-pair spectral peaks and candidate H3 percentages.
    rows=[];syn=RES/'v75_zero_truth_pairs/synth'
    for path in sorted(syn.glob('*.npz')):
        with np.load(path) as z:
            d=z['S0'].astype(float);v=d[np.isfinite(d)];result=band(z['xy0'].astype(float),d,v.mean()+3*v.std())
            rows.append(dict(label=path.stem,**result))
    pd.DataFrame(rows).to_csv(OUT/'B_spectral_peaks.csv',index=False)
    f=pd.read_csv(V78/'band_ratio_fields.csv');p=pd.read_csv(V78/'band_ratio_pairs.csv');tol=float(p.ratio.max())
    save('H3_structure_counts',dict(tolerance=tol,all_S5=int((f.ratio_S5<=tol).sum()),all_S0=int((f.ratio_S0<=tol).sum()),fields=len(f),blank_S5=int((f[f.group=='blank'].ratio_S5<=tol).sum()),blank_n=int((f.group=='blank').sum()),fixed_S5=int((f[f.fixed29].ratio_S5<=tol).sum()),fixed_n=int(f.fixed29.sum()),nan_ratios=int(f.ratio_S5.isna().sum())))
    # Reciprocal vectors from an independent zero-padded FFT peak estimator.
    parent=Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu');folder=next(x for x in parent.iterdir() if x.name=='260904_p50_repeat')
    raw=tifffile.imread(folder/'1.tif').astype(np.float32)
    from shared.lattice_indexing import estimate_hex_orientation_fft,hex_basis
    idealK=np.linalg.inv(hex_basis(7.286,estimate_hex_orientation_fft(raw)))
    from scipy import fft as sf
    contrast=raw-cv2.GaussianBlur(raw,(51,51),0);win=np.outer(np.hanning(raw.shape[0]),np.hanning(raw.shape[1])).astype(np.float32)
    fft=abs(sf.fftshift(sf.fft2(contrast*win,s=(4096,4096),workers=1)))
    freq=sf.fftshift(sf.fftfreq(4096)).astype(np.float32);radius=np.hypot(freq[:,None],freq[None,:]);target=2/(np.sqrt(3)*7.35)
    cand=(ndimage.maximum_filter(fft,9)==fft)&(radius>target*.94)&(radius<target*1.06)
    yy,xx=np.where(cand);order=np.argsort(fft[yy,xx])[-12:];vectors=[]
    for i in order:
        y,x=yy[i],xx[i]
        dx=.5*(fft[y,x-1]-fft[y,x+1])/(fft[y,x-1]-2*fft[y,x]+fft[y,x+1])
        dy=.5*(fft[y-1,x]-fft[y+1,x])/(fft[y-1,x]-2*fft[y,x]+fft[y+1,x])
        vectors.append([freq[x]+dx/4096,freq[y]+dy/4096])
    vectors=np.array(vectors);realK=np.array([vectors[np.linalg.norm(vectors-k,axis=1).argmin()] for k in idealK]);delta=realK-idealK
    preds=[]
    for i in range(-2,3):
        for j in range(-2,3):
            if i<0 or (i==0 and j<=0):continue
            q=np.array([i,j])@delta;period=1/np.linalg.norm(q);axis=(np.degrees(np.arctan2(q[1],q[0]))+90)%180
            if 128<=period<=1024:preds.append(dict(h=[i,j],period=period,axis=axis))
    for result in rows:
        result['match']=any(abs(result['period_px']/q['period']-1)<=.1 and abs((result['axis_deg']-q['axis']+90)%180-90)<=10 for q in preds)
    save('B_prediction_independent',dict(ideal_reciprocal=idealK.tolist(),real_reciprocal=realK.tolist(),predictions=preds,matches=int(sum(x['match'] for x in rows)),trials=len(rows),unique_peaks=sorted(set((x['axis_deg'],round(x['period_px'],3)) for x in rows))))
    # Real-image perturbation re-read: selected normal blank, four perturbation types, fixed thresholds.
    th=pd.read_csv(V72/'thresholds_v2.csv',dtype={'date':str});ref=pd.read_csv(V72/'perturbation_by_field.csv');rr=[]
    fid='260825_0_7';r=LED.loc[fid]
    import v72_readout as R
    a=R.box_contrast(tifffile.imread(r.pre_path));b=R.box_contrast(tifffile.imread(r.post_path))
    z1=np.load(V70/'fields'/f'{fid}.npz');z2=np.load(V70/'fields2'/f'{fid}.npz');xy=z1['std_xy'].astype(float);aa=R.rnd(a,xy)
    threshold=float(th[(th.date==r.date)&(th.readout=='S1')&(th.thr=='G')].threshold.iloc[0]);center=np.array([1024.,1022.]);corners=np.array([[0.,0.],[2048.,0.],[0.,2044.],[2048.,2044.]])-center
    for name,E in [('scale',np.eye(2)),('rotation',np.array([[0.,-1],[1.,0.]])),('shear_x',np.array([[0.,1],[0.,0.]])),('shear_y',np.array([[0.,0.],[1.,0.]]))]:
        for m in [0.,4.,8.]:
            if m==0 and name!='scale':continue
            error=E*m/np.linalg.norm(corners@E.T,axis=1).max();post=xy@z2['Llin'].T+z2['tL']+(xy-center)@error.T
            d=(aa-R.rnd(b,post)).astype(float);met=band(xy,d,threshold);saved=ref[(ref.fid==fid)&(ref.type==name)&(ref.m_px==m)].iloc[0]
            rr.append(dict(fid=fid,type=name,m=m,**met,power_error=met['band_power']-saved.band_power,rate_error=met['rate']-saved.rate))
    pd.DataFrame(rr).to_csv(OUT/'C_perturbation_direct.csv',index=False)
    save('C_perturbation_verified',dict(cases=len(rr),max_power_error=max(abs(x['power_error']) for x in rr),max_rate_error=max(abs(x['rate_error']) for x in rr)))
    # Re-pool scar-distance counts; class selection logic is tested without accessing masks.
    t=pd.read_csv(RES/'v74_scar_distance/scar_distance_by_field.csv',dtype={'date':str,'board':str});t=t[t.group=='blank'];names=['in_core(>2per)','in_depth1-2','in_depth0-1','out_0-1','out_1-2','out_2-4','out_4-8','out_>8'];rng=np.random.default_rng(81);rr=[]
    for rd in ['S1','S3']:
        h=t[t.readout==rd].groupby(['substrate','cls'])[['n','pos']].sum().unstack('cls');nn=h['n'][names].to_numpy();pp=h['pos'][names].to_numpy();rate=pp.sum(0)/nn.sum(0)
        idx=rng.integers(0,len(nn),(10000,len(nn)));boot=pp[idx].sum(1)/nn[idx].sum(1);lo,hi=np.quantile(boot,[.025,.975],axis=0)
        first=next((names[i] for i in range(3,8) if np.all(rate[i:]<=hi[-1])),None)
        rr.append(dict(readout=rd,substrates=len(nn),classes=names,rate=rate.tolist(),lo=lo.tolist(),hi=hi.tolist(),first_class=first))
    save('E_distance_repooled',rr)

def focus_checks():
    t=pd.read_csv(V78/'focus_proxy_by_field.csv').set_index('fid')
    fids=['260825_0_1',t.la_iqr.idxmax(),t.mad.idxmax()];rows=[]
    def blocks(xy,v,mean=False):
        cell=np.clip((xy[:,1]//64).astype(int),0,31)*32+np.clip((xy[:,0]//64).astype(int),0,31)
        result=np.full(1024,np.nan)
        for c in np.unique(cell[np.isfinite(v)]):
            values=v[(cell==c)&np.isfinite(v)]
            if len(values)>=20:result[c]=values.mean() if mean else np.median(values)
        return result
    for fid in dict.fromkeys(fids):
        a=np.load(V78/'fields'/f'{fid}.npz');b=np.load(V72/'fields'/f'{fid}.npz');xy=a['ctr'].astype(float);d=b['S5'].astype(float);valid=np.isfinite(d)&(a['sd_periods']>2)
        med,mad=robust(d[valid]);la=blocks(xy,a['amp_ratio'].astype(float));pos=blocks(xy,np.where(valid,(d>med+4*mad).astype(float),np.nan),True);dm=blocks(xy,np.where(valid,d,np.nan))
        good=np.isfinite(la)&np.isfinite(pos);good2=good&np.isfinite(dm)
        rho=float(spearmanr(pos[good],abs(la[good]-np.nanmedian(la))).statistic);rhod=float(spearmanr(dm[good2],la[good2]).statistic);iqr=float(np.nanquantile(la,.75)-np.nanquantile(la,.25))
        rows.append(dict(fid=fid,la_iqr=iqr,saved_la_iqr=float(t.loc[fid].la_iqr),mad=mad,saved_mad=float(t.loc[fid].mad),rho_pos=rho,saved_rho_pos=float(t.loc[fid].rho_pos_vs_abs_la),rho_delta=rhod,saved_rho_delta=float(t.loc[fid].rho_dm_vs_la)))
    save('D_block_direct',rows)
    # P offset ignores its ctr_pre_nodes argument: its return is based entirely
    # on expL (in the saved-field caller, LINEAR predictions rather than centres).
    import v75_pair_lib as P
    z=np.load(V70/'fields2/260825_0_1.npz');ctr=z['pre_ctr'].astype(float);mapped=z['pre_pred_lin']@z['Llin'].T+z['tL'];partner=z['post_ctr'][z['jL']].astype(float)
    ok=~sub(z,'pre')&~sub(z,'post')[z['jL']]&z['consL']
    out=P.local_offset_positions(ctr,mapped,partner,ok);out2=P.local_offset_positions(ctr+np.array([.5,0]),mapped,partner,ok)
    shifts=np.linalg.norm(ctr-z['pre_pred_lin'],axis=1)
    save('G_P_offset_argument_check',dict(fid='260825_0_1',pre_centres_changed_by_px=.5,post_read_position_change_max=float(abs(out-out2).max()),actual_vs_linear_pre_centre_median=float(np.median(shifts[ok])),actual_vs_linear_pre_centre_p95=float(np.quantile(shifts[ok],.95)),conclusion='ctr_pre_nodes is unused; supplied expL is mapped pre_pred_lin. This is a smooth predicted-position readout, not per-pillar L-mapped actual-pre-centre readout.'))

if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'field_level/v77_signal_injection'))
    {'tables':table_checks,'geometry':geometry_checks,'synthetic':synthetic_checks,'dark':dark_checks,'native_injection':native_injection_check,'structure':structure_checks,'focus':focus_checks}[sys.argv[1]]()
