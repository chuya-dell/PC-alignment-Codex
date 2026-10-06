"""v57 step E (statistics): distance profiles (E1), edge confounding (E2), distance-specific blank thresholds (E3), main sign-flip tests (E4).
Everything is computed from stored differences + scar masks; no registration or photometry is re-run. Settings follow data/results/v57.../step0/PROTOCOL_AND_PREDICTIONS.md."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import field_v57_common as C

OUTE = C.OUT/'stepE'; OUTB = C.OUT/'stepB'; OUTD = C.OUT/'stepD'
rng = np.random.default_rng(C.SEED)
for fam in ('Yu Gothic', 'Meiryo', 'MS Gothic'):
    if any(f.name == fam for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = fam; break
plt.rcParams['axes.unicode_minus'] = False
NB = 50; N_MIN = 300

# ---------- metadata ----------
fl = C.field_list()[['fid', 'date', 'board', 'field', 'condition', 'group']]
flags = pd.read_csv(OUTB/'B3_field_flags.csv', dtype={'date': str, 'board': str})[['fid', 'rate_mean', 'rate_med', 'outlier29', 'eligible']]
det = pd.read_csv(OUTD/'detection_results.csv')
meta = fl.merge(flags, on='fid').merge(det[['fid', 'status', 'n_lines_v', 'n_lines_h', 'lines_json']], on='fid', how='left')
meta['detected'] = meta.status == 'ok'
meta['set29'] = np.where(meta.outlier29, 'only29', 'non29')
bc = pd.read_csv(OUTE/'bin_counts_by_field.csv')
base = pd.read_csv(OUTB/'B2_blank_baseline_upper_by_date.csv', dtype={'date': str}).set_index('date')
DEF = {'mean': ('kmean_all', 'kmean_e300'), 'med': ('kmed_all', 'kmed_e300')}

_PIV = {}
def _piv(defn, e300):
    key = (defn, e300)
    if key not in _PIV:
        kc = DEF[defn][1 if e300 else 0]; nc = 'n_e300' if e300 else 'n_all'
        sub = bc[bc.b >= 0]
        _PIV[key] = (sub.pivot(index='fid', columns='b', values=nc), sub.pivot(index='fid', columns='b', values=kc))
    return _PIV[key]

def bins_matrix(fids, defn='mean', e300=False, field_min=0):
    """return (n[f,b], k[f,b]) for b = 0..NB (rows ordered as fids)"""
    n, k = _piv(defn, e300)
    return n.loc[fids].to_numpy(float), k.loc[fids].to_numpy(float)

def profile(fids, defn='mean', e300=False, minn=30):
    n, k = bins_matrix(fids, defn, e300)
    with np.errstate(invalid='ignore', divide='ignore'):
        rate = np.where(n >= minn, k/n, np.nan)
    return dict(n_fields=np.sum(n >= minn, axis=0), mean_rate=np.nanmean(rate, axis=0) if len(fids) else None,
                pooled=np.nansum(np.where(n >= minn, k, 0), axis=0)/np.maximum(np.nansum(np.where(n >= minn, n, 0), axis=0), 1),
                n_pillars=np.nansum(np.where(n >= minn, n, 0), axis=0), rate=rate)

groups = {'blank': lambda m: m.group == 'blank', 'analyte': lambda m: m.group == 'analyte', 'mismatch': lambda m: m.group == 'mismatch',
          'other_100uL': lambda m: m.group == 'other', 'blank_reference': lambda m: m.group == 'blank_reference'}
det_ok = meta[meta.detected]

# ---------- E1 profiles ----------
rows = []
for gname, gf in groups.items():
    for sname in ('non29', 'only29'):
        for stratum in ['ALL'] + sorted(meta.date.unique()):
            sel = det_ok[gf(det_ok) & (det_ok.set29 == sname) & ((det_ok.date == stratum) if stratum != 'ALL' else True)]
            if len(sel) == 0: continue
            for defn in ('mean', 'med'):
                for e300 in (False, True):
                    pr = profile(sel.fid.tolist(), defn, e300)
                    for b in range(NB+1):
                        rows.append(dict(group=gname, fieldset=sname, stratum=stratum, defn=defn, edge300=e300, bin=b, n_fields=int(pr['n_fields'][b]),
                                         mean_rate=pr['mean_rate'][b], pooled_rate=pr['pooled'][b], n_pillars=int(pr['n_pillars'][b])))
prof = pd.DataFrame(rows); prof.to_csv(OUTE/'E1_profile_by_distance.csv', index=False)

# in-mask pillars (descriptive)
inm = bc[bc.b == -1].merge(meta[['fid', 'group', 'set29', 'date']], on='fid')
inm_sum = inm.groupby(['group', 'set29']).agg(fields=('fid', 'size'), n=('n_all', 'sum'), k=('kmean_all', 'sum')).reset_index(); inm_sum['rate'] = inm_sum.k/inm_sum.n
inm_sum.to_csv(OUTE/'E1_in_mask_pillars.csv', index=False)

# ---------- attenuation distance ----------
def blank_line_bootstrap(defn='mean', e300=False, B=2000, minn=30):
    sel = det_ok[(det_ok.group == 'blank')]   # all blank fields (no outlier exclusion: circularity)
    pr = profile(sel.fid.tolist(), defn, e300, minn); R = pr['rate']
    out = np.full(NB+1, np.nan)
    ups = np.zeros((B, NB+1))
    for i in range(B):
        idx = rng.integers(0, R.shape[0], R.shape[0]); ups[i] = np.nanmean(R[idx], axis=0)
    return np.nanquantile(ups, 0.95, axis=0), pr['mean_rate']
att_rows = []
for defn in ('mean', 'med'):
    for e300 in (False, True):
        ci95, blank_mean = blank_line_bootstrap(defn, e300)
        for gname in ('blank', 'analyte', 'mismatch'):
            for sname in ('non29', 'only29'):
                sel = det_ok[groups[gname](det_ok) & (det_ok.set29 == sname)]
                if len(sel) == 0: continue
                pr = profile(sel.fid.tolist(), defn, e300)
                line_i = float(np.mean([base.loc[d, f'block_upper_{defn}_99'] for d in sel.date]))   # B2 baseline upper (field-mean of date lines)
                m = pr['mean_rate']
                def att(line):
                    ok = np.isfinite(m[:31]);
                    below = (m[:31] <= (line if np.isscalar(line) else line[:31])) | ~ok
                    for d in range(0, 31):
                        if below[d:].all(): return d
                    return np.nan   # does not return within 30 periods
                att_rows.append(dict(defn=defn, edge300=e300, group=gname, fieldset=sname, n_fields=len(sel), line_i_baseline_upper=line_i,
                                     attenuation_i=att(line_i), attenuation_ii=att(ci95), rate_bin0_4_mean=float(np.nanmean(m[0:5])), rate_bin20_30_mean=float(np.nanmean(m[20:31])),
                                     rate_far_bins_ge30=float(np.nanmean(m[30:51]))))
pd.DataFrame(att_rows).to_csv(OUTE/'E1_attenuation_distance.csv', index=False)

# ---------- E2: edge confounding ----------
eb = pd.read_csv(OUTE/'edge_by_dist_counts_by_field.csv').merge(meta[['fid', 'group', 'set29', 'date', 'detected']], on='fid')
eb = eb[eb.detected]
rows = []
for (g, s), x in eb.groupby(['group', 'set29']):
    t = x.groupby(['dist_class', 'edge_lo']).agg(n=('n', 'sum'), k=('kmean', 'sum')).reset_index(); t['rate'] = t.k/t.n; t['group'] = g; t['fieldset'] = s
    rows.append(t)
pd.concat(rows).to_csv(OUTE/'E2_rate_by_distclass_and_edgeclass.csv', index=False)
# how many scars are near the image edge
def line_edge_dist(lj):
    if not isinstance(lj, str): return np.nan, np.nan
    L = json.loads(lj); d = []
    for l in L:
        X0 = l['X0_full']; span = 2048 if l['family'] == 'vertical' else 2044
        d.append(min(X0, span-X0))
    return (min(d) if d else np.nan), len(d)
tmp = det_ok.copy(); tmp[['line_edge_min_px', 'n_lines']] = tmp.lines_json.apply(lambda s: pd.Series(line_edge_dist(s)))
tmp[['fid', 'group', 'set29', 'line_edge_min_px', 'n_lines']].to_csv(OUTE/'E2_scar_line_distance_to_image_edge.csv', index=False)
near_edge_share = float((tmp.line_edge_min_px < 300).mean())
# share of near-zone pillars lying within 300 px of an image edge
nz = eb[eb.dist_class == '0-5'].groupby('fid').apply(lambda g: pd.Series(dict(n=g.n.sum(), n_lt300=g[g.edge_lo < 300].n.sum())), include_groups=False)
nz['share_lt300'] = nz.n_lt300/nz.n
e2_summary = dict(n_detected=int(len(tmp)), fields_with_a_scar_line_within_300px_of_edge=int((tmp.line_edge_min_px < 300).sum()),
                  share_fields=near_edge_share, near_zone_pillars_share_edge_lt300_pooled=float(nz.n_lt300.sum()/nz.n.sum()),
                  near_zone_share_edge_lt300_field_median=float(nz.share_lt300.median()),
                  fields_with_near_zone_ge_300_pillars_edge_ge_300=int(((nz.n-nz.n_lt300) >= N_MIN).sum()))
# GLM with cluster-robust SE: exceed ~ near + edge class (+ date), far = dist>20
glm_rows = []
for gname in ('blank', 'analyte'):
    for sname in ('non29', 'with29'):
        x = eb[(eb.group == gname) & eb.dist_class.isin(['0-5', '20+'])]
        if sname == 'non29': x = x[x.set29 == 'non29']
        x = x[x.n > 0].copy(); x['near'] = (x.dist_class == '0-5').astype(float)
        x['edge_cls'] = pd.cut(x.edge_lo, [-1, 99, 199, 299, 499, 1e10], labels=['0-100', '100-200', '200-300', '300-500', '500+'])
        for model in ('near_only', 'near+edge', 'near+edge+date'):
            X = pd.DataFrame({'const': 1.0, 'near': x.near})
            if 'edge' in model: X = X.join(pd.get_dummies(x.edge_cls, drop_first=False).drop(columns='500+').astype(float))
            if 'date' in model: X = X.join(pd.get_dummies(x.date, drop_first=True).astype(float))
            y = np.c_[x.kmean, x.n-x.kmean]
            try:
                fit = sm.GLM(y, X.to_numpy(), family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.fid)[0]})
                j = list(X.columns).index('near'); b = fit.params[j]; se = fit.bse[j]
                glm_rows.append(dict(group=gname, fieldset=sname, model=model, n_fields=int(x.fid.nunique()), n_rows=len(x), log_or_near=b, or_near=np.exp(b),
                                     or_lo=np.exp(b-1.96*se), or_hi=np.exp(b+1.96*se), p=fit.pvalues[j]))
            except Exception as e:
                glm_rows.append(dict(group=gname, fieldset=sname, model=model, error=repr(e)))
glm = pd.DataFrame(glm_rows); glm.to_csv(OUTE/'E2_glm_near_vs_far_with_edge_adjustment.csv', index=False)
json.dump(e2_summary, open(OUTE/'E2_summary.json', 'w'), indent=1)

# ---------- E3: distance-specific blank thresholds ----------
thr = pd.read_csv(OUTB/'B2_blank_threshold_by_date.csv', dtype={'date': str}).set_index('date')
BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 30), (30, 1e9)]
rows = []
blank_ok = det_ok[det_ok.group == 'blank']
for date, g in blank_ok.groupby('date'):
    ds = []; dist_all = []; edge_all = []
    for fid in g.fid:
        d, xy, ids = C.load_cache(fid); z = np.load(OUTD/f'{fid}_pillar_dist.npz')
        keep = ~z['in_mask']; ds.append(d[keep]); dist_all.append(z['dist_per'][keep]); edge_all.append(z['edge_px'][keep])
    d = np.concatenate(ds); dist = np.concatenate(dist_all); edge = np.concatenate(edge_all)
    far = (dist >= 20) & (edge >= 300); far_mu, far_sd = d[far].mean(), d[far].std()
    for lo, hi in BINS:
        s = (dist >= lo) & (dist < hi)
        if s.sum() < 500: continue
        mu, sd = d[s].mean(), d[s].std()
        rows.append(dict(date=date, dist_lo=lo, dist_hi=hi, n_blank_pillars=int(s.sum()), mean=mu, sd=sd, thr_bin=mu+3*sd, thr_global_all_blank=thr.loc[date, 'thr_mean'],
                         thr_far_only=far_mu+3*far_sd, diff_bin_minus_global=mu+3*sd-thr.loc[date, 'thr_mean'], sd_ratio_bin_over_global=sd/thr.loc[date, 'sd'],
                         rate_above_global_thr=float((d[s] > thr.loc[date, 'thr_mean']).mean()), rate_above_far_thr=float((d[s] > far_mu+3*far_sd).mean())))
pd.DataFrame(rows).to_csv(OUTE/'E3_distance_specific_blank_thresholds.csv', index=False)

# ---------- E4: sign-flip tests ----------
def zone_counts(sel, near_lim, far_lim, near_e300, defn):
    nA, kA = bins_matrix(sel.fid.tolist(), defn, False); nE, kE = bins_matrix(sel.fid.tolist(), defn, True)
    nn, kn = (nE, kE) if near_e300 else (nA, kA)
    return nn[:, :near_lim].sum(1), kn[:, :near_lim].sum(1), nE[:, far_lim:].sum(1), kE[:, far_lim:].sum(1)

def signflip_p(x, nsim=20000):
    x = np.asarray(x, float); n = len(x); obs = x.mean()
    if n <= 18:
        signs = np.array(np.meshgrid(*[[-1, 1]]*n)).reshape(n, -1).T
        stats = (signs*x).mean(1)
    else:
        signs = rng.choice([-1.0, 1.0], size=(nsim, n)); stats = (signs*x).mean(1)
    return float((np.abs(stats) >= abs(obs)-1e-15).mean())

NEARS, FARS = (3, 5, 8), (15, 20, 30)
trows = []
for defn in ('mean', 'med'):
    for near_e300 in (False, True):
        for fieldset in ('non29', 'with29'):
            for gname in ('blank', 'analyte', 'mismatch'):
                for stratum in ['ALL'] + sorted(meta.date.unique()):
                    sel = det_ok[groups[gname](det_ok)]
                    if fieldset == 'non29': sel = sel[sel.set29 == 'non29']
                    if stratum != 'ALL': sel = sel[sel.date == stratum]
                    if len(sel) < 3: continue
                    for nl in NEARS:
                        for fl_ in FARS:
                            nn, kn, nf, kf = zone_counts(sel, nl, fl_, near_e300, defn)
                            ok = (nn >= N_MIN) & (nf >= N_MIN)
                            if ok.sum() < 3:
                                trows.append(dict(defn=defn, near_edge300=near_e300, fieldset=fieldset, group=gname, stratum=stratum, near=nl, far=fl_, n_fields_used=int(ok.sum()), n_fields_excluded=int((~ok).sum()), note='fields with >=300 pillars in both zones < 3')); continue
                            rn = kn[ok]/nn[ok]; rf = kf[ok]/nf[ok]; dif = rn-rf
                            trows.append(dict(defn=defn, near_edge300=near_e300, fieldset=fieldset, group=gname, stratum=stratum, near=nl, far=fl_,
                                              n_fields_used=int(ok.sum()), n_fields_excluded=int((~ok).sum()), mean_near=float(rn.mean()), mean_far=float(rf.mean()),
                                              mean_diff=float(dif.mean()), median_diff=float(np.median(dif)), frac_pos=float((dif > 0).mean()), frac_neg=float((dif < 0).mean()),
                                              p_signflip=signflip_p(dif, 200000 if (defn == 'mean' and nl == 5 and fl_ == 20 and not near_e300 and stratum == 'ALL') else 20000)))
tt = pd.DataFrame(trows)
# Holm: primary family = pooled (ALL) tests of blank & analyte for the primary setting; others by family (defn, near_edge300, fieldset, near, far, ALL vs per-date)
tt['family'] = np.where(tt.stratum == 'ALL', 'pooled', 'per_date')
tt['p_holm'] = np.nan
for key, g in tt[tt.p_signflip.notna() & tt.group.isin(['blank', 'analyte'])].groupby(['defn', 'near_edge300', 'fieldset', 'near', 'far', 'family']):
    tt.loc[g.index, 'p_holm'] = multipletests(g.p_signflip, method='holm')[1]
tt.to_csv(OUTE/'E4_signflip_tests_all.csv', index=False)
prim = tt[(tt.defn == 'mean') & (~tt.near_edge300) & (tt.fieldset == 'non29') & (tt.near == 5) & (tt.far == 20)]
prim.to_csv(OUTE/'E4_primary_setting.csv', index=False)
print(prim[prim.stratum == 'ALL'].round(5).to_string())
print(prim[(prim.stratum != 'ALL') & prim.group.isin(['blank', 'analyte'])][['stratum', 'group', 'n_fields_used', 'mean_near', 'mean_far', 'mean_diff', 'p_signflip', 'p_holm']].round(4).to_string())
print(json.dumps(e2_summary, indent=1)); print(glm.round(4).to_string()); print(pd.DataFrame(att_rows).round(4).to_string())

# ---------- figures ----------
def plot_profile(ax, e300, gset, title):
    for gname, col in (('blank', '#1f77b4'), ('analyte', '#d62728'), ('mismatch', '#2ca02c')):
        x = prof[(prof.group == gname) & (prof.fieldset == gset) & (prof.stratum == 'ALL') & (prof.defn == 'mean') & (prof.edge300 == e300) & (prof.bin < 50)]
        ax.plot(x.bin+0.5, 100*x.mean_rate, color=col, label=f'{gname} (n={int(x.n_fields.max())})')
    ax.axhline(100*np.mean([base.loc[d, 'block_upper_mean_99'] for d in det_ok.date]), color='k', ls='--', lw=0.8, label='ブランク基準線の上限(99%)')
    ax.set_xlabel('傷のマスク縁からの距離(周期)'); ax.set_ylabel('超過率(視野平均, %)'); ax.set_title(title); ax.grid(alpha=.3); ax.set_xlim(0, 50)
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
plot_profile(axs[0], False, 'non29', '外れ値29視野を除く(端の制限なし)'); plot_profile(axs[1], True, 'non29', '外れ値29視野を除く(端から300px以上)')
plot_profile(axs[2], False, 'only29', '外れ値29視野のみ(記述)'); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUTE/'fig_E1_profile_pooled.png', dpi=130); plt.close(fig)
dates = sorted(meta.date.unique())
fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharey=True)
for ax, date in zip(axs.flat, dates):
    for gname, col in (('blank', '#1f77b4'), ('analyte', '#d62728')):
        x = prof[(prof.group == gname) & (prof.fieldset == 'non29') & (prof.stratum == date) & (prof.defn == 'mean') & (~prof.edge300) & (prof.bin < 40)]
        if len(x): ax.plot(x.bin+0.5, 100*x.mean_rate, color=col, label=f'{gname} (n={int(x.n_fields.max())})')
    ax.axhline(100*base.loc[date, 'block_upper_mean_99'], color='k', ls='--', lw=.8); ax.set_title(date); ax.grid(alpha=.3); ax.legend(fontsize=7)
fig.supxlabel('傷のマスク縁からの距離(周期)'); fig.supylabel('超過率(視野平均, %)'); fig.tight_layout(); fig.savefig(OUTE/'fig_E1_profile_by_date.png', dpi=110); plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 4.4))
for (g, s), x in eb.groupby(['group', 'set29']):
    if g not in ('blank', 'analyte') or s != 'non29': continue
    for dc, ls in (('0-5', '-'), ('20+', '--')):
        t = x[x.dist_class == dc].groupby('edge_lo').agg(n=('n', 'sum'), k=('kmean', 'sum')); ax.plot(range(len(t)), 100*t.k/t.n, ls=ls, marker='o', label=f'{g} {dc}周期')
ax.set_xticks(range(5)); ax.set_xticklabels(['0-100', '100-200', '200-300', '300-500', '500+']); ax.set_xlabel('画像の端からの距離(px)'); ax.set_ylabel('超過率(%)'); ax.grid(alpha=.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUTE/'fig_E2_edge_confounding.png', dpi=130); plt.close(fig)
print('figures saved')
