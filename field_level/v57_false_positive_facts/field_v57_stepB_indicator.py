"""v57 step B: common indicator (theoretical upper bound u(c) on the field excess rate) from stored differences only.
B1 c0 recomputed from geometry vs stated values; B2 blank baseline upper limits per date (binomial and 32-px block bootstrap);
B3 per-field flags (exceeds u(c) / not eligible) and the alternative view for non-eligible conditions."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, binom, kruskal
import field_v57_common as C

OUT = C.OUT / 'stepB'
NA = 6.02214076e23
# ---------------- B1 geometry ----------------
PITCH_NM = 460.0; D_NM = 230.0; H_NM = 200.0; V_UL = 200.0
a_cell = np.sqrt(3)/2*(PITCH_NM*1e-3)**2           # um^2, hexagonal
a_cap = np.pi*(D_NM*1e-3/2)**2
a_side = np.pi*(D_NM*1e-3)*(H_NM*1e-3)
def n_pillars(side_mm): return (side_mm*1e3)**2/a_cell
def c0_pM(n, v_ul=V_UL): return n/(v_ul*1e-6*NA)*1e12   # mol/L -> pM
rows = []
for side in (5.0, 4.0):
    n = n_pillars(side)
    base = c0_pM(n)
    f_side = (a_cell+a_side)/(a_cap+a_side)          # gold = cap + base film + side wall; one molecule per pillar on cap+side on average
    f_cap = a_cell/a_cap                              # gold = cap + base film; one molecule per pillar cap on average
    stated = {5.0: dict(a=1.13, b_side=2.00, b_cap=5.00), 4.0: dict(a=0.72)}[side]
    rows.append(dict(substrate_mm=side, n_pillars=n, c0_a_pillar_only_pM=base, stated_a=stated['a'],
                     c0_b_side_pM=base*f_side, stated_b_side=stated.get('b_side', np.nan),
                     c0_b_cap_pM=base*f_cap, stated_b_cap=stated.get('b_cap', np.nan)))
geo = pd.DataFrame(rows)
geo['a_cell_um2'] = a_cell; geo['a_cap_um2'] = a_cap; geo['a_side_um2'] = a_side
geo['match_a'] = (geo.c0_a_pillar_only_pM/geo.stated_a-1).abs() < 0.02
geo['match_b_side'] = (geo.c0_b_side_pM/geo.stated_b_side-1).abs() < 0.02
geo['match_b_cap'] = (geo.c0_b_cap_pM/geo.stated_b_cap-1).abs() < 0.02
geo.to_csv(OUT/'B1_c0_geometry_check.csv', index=False, encoding='utf-8-sig')
print(geo.round(4).T)
C0_MAIN_PM = 0.72           # stated 4 mm value = smaller than recomputed (conservative); recomputed shown alongside
C0_RECALC_4MM = float(geo.loc[geo.substrate_mm == 4.0, 'c0_a_pillar_only_pM'].iloc[0])
C0_USED = min(C0_MAIN_PM, C0_RECALC_4MM)

# concentrations present
conds = {'1e-09': 1e-9, '1e-10': 1e-10, '1e-11': 1e-11, '1e-12': 1e-12, '1e-13': 1e-13, '1e-14': 1e-14, '1e-15': 1e-15, '0': 0.0,
         '100uL_10fM': 1e-14, 'mismatch': 1e-9}   # mismatch: nominal 1 nM (assumption); 100uL_10fM: treated as 10 fM (conservative)
occ_rows = []
for k, c in conds.items():
    c_pM = c*1e12
    r = dict(condition=k, c_M=c, note='名目1nM(仮定)' if k == 'mismatch' else ('10fM・100µL(200µL扱いで保守的)' if k == '100uL_10fM' else ''))
    for tag, c0 in (('a4mm_used', C0_USED), ('a4mm_recalc', C0_RECALC_4MM), ('a5mm_recalc', float(geo.c0_a_pillar_only_pM.iloc[0])),
                    ('b_side_5mm', float(geo.c0_b_side_pM.iloc[0])), ('b_cap_5mm', float(geo.c0_b_cap_pM.iloc[0])), ('V10x_a4mm', C0_USED/10)):
        r[f'min1_c_over_c0_{tag}'] = min(1.0, c_pM/c0)
        r[f'expect_1_minus_exp_{tag}'] = 1-np.exp(-c_pM/c0)
    occ_rows.append(r)
occ = pd.DataFrame(occ_rows)
occ.to_csv(OUT/'B1_occupancy_upper_bounds.csv', index=False, encoding='utf-8-sig')
print(occ[['condition', 'min1_c_over_c0_a4mm_used', 'expect_1_minus_exp_a4mm_used', 'min1_c_over_c0_b_side_5mm', 'min1_c_over_c0_b_cap_5mm', 'min1_c_over_c0_V10x_a4mm']].round(4).to_string())

# ---------------- stored values ----------------
f = C.field_list()
data = {fid: C.load_cache(fid) for fid in f.fid}
BLK = 32
rng = np.random.default_rng(C.SEED)
rows_f = []; rows_b = []
block_pool = {}   # date -> list of (n_pillars, n_pos_mean, n_pos_med) per block of main blank fields
for date, g in f.groupby('date'):
    blank = g[g.group == 'blank']
    pool = np.concatenate([data[x][0] for x in blank.fid])
    mu, sd = pool.mean(), pool.std(ddof=0)
    med = np.median(pool); mad = 1.4826*np.median(np.abs(pool-med))
    th_mean = mu+3*sd; th_med = med+3*mad
    pb_mean = float((pool > th_mean).mean()); pb_med = float((pool > th_med).mean())
    rows_b.append(dict(date=date, blank_fields=len(blank), blank_pillars=len(pool), mean=mu, sd=sd, thr_mean=th_mean, median=med, mad=mad, thr_med=th_med, p_blank_mean=pb_mean, p_blank_med=pb_med))
    for r in g.itertuples():
        d, xy, ids = data[r.fid]
        pm = d > th_mean; pd_ = d > th_med
        rows_f.append(dict(fid=r.fid, date=date, board=r.board, field=r.field, condition=r.condition, group=r.group, n=len(d),
                           pos_mean=int(pm.sum()), pos_med=int(pd_.sum()), rate_mean=float(pm.mean()), rate_med=float(pd_.mean())))
    # block table for blanks
    for defn, th in (('mean', th_mean), ('med', th_med)):
        bl = []
        for x in blank.fid:
            d, xy, ids = data[x]
            bx = (xy[:, 0]//BLK).astype(int); by = (xy[:, 1]//BLK).astype(int)
            key = by*100+bx
            u, inv = np.unique(key, return_inverse=True)
            n_b = np.bincount(inv); p_b = np.bincount(inv, weights=(d > th).astype(float))
            bl.append(np.c_[n_b, p_b])
        block_pool[(date, defn)] = np.vstack(bl)
bl = pd.DataFrame(rows_b); ff = pd.DataFrame(rows_f)
ff.to_csv(OUT/'field_rates_all632.csv', index=False, encoding='utf-8-sig'); bl.to_csv(OUT/'B2_blank_threshold_by_date.csv', index=False, encoding='utf-8-sig')

# ---------------- B2 baseline upper limits ----------------
NDRAW = 20000; NBLK = 64*64
res = []
for r in bl.itertuples():
    date = r.date
    blank_fields = ff[(ff.date == date) & (ff.group == 'blank')]
    n_typ = int(blank_fields.n.median())
    row = dict(date=date, blank_fields=r.blank_fields, n_typical=n_typ)
    for defn, pb in (('mean', r.p_blank_mean), ('med', r.p_blank_med)):
        row[f'p_blank_{defn}'] = pb
        for q in (0.99, 0.95):
            row[f'binom_upper_{defn}_{int(q*100)}'] = binom.ppf(q, n_typ, pb)/n_typ
        pool = block_pool[(date, defn)]
        idx = rng.integers(0, len(pool), size=(NDRAW, NBLK), dtype=np.int32)
        npil = pool[idx, 0].sum(1); npos = pool[idx, 1].sum(1)
        rate = npos/npil
        for q in (0.99, 0.95):
            row[f'block_upper_{defn}_{int(q*100)}'] = float(np.quantile(rate, q))
        row[f'block_mean_{defn}'] = float(rate.mean()); row[f'block_sd_{defn}'] = float(rate.std())
        # observed blank field rates for reference
        row[f'blank_obs_max_{defn}'] = float(blank_fields[f'rate_{defn}'].max())
        row[f'blank_obs_mean_{defn}'] = float(blank_fields[f'rate_{defn}'].mean())
    res.append(row)
base = pd.DataFrame(res)
base['diff_block_minus_binom_mean_99'] = base.block_upper_mean_99-base.binom_upper_mean_99
base.to_csv(OUT/'B2_blank_baseline_upper_by_date.csv', index=False, encoding='utf-8-sig')
print(base[['date', 'blank_fields', 'p_blank_mean', 'binom_upper_mean_99', 'block_upper_mean_99', 'diff_block_minus_binom_mean_99', 'blank_obs_max_mean', 'block_upper_mean_95']].round(4).to_string())

# ---------------- B3 u(c) and flags ----------------
bu = base.set_index('date')
def u_of(row, defn='mean', basekind='block', q=99, occ_col='min1_c_over_c0_a4mm_used'):
    key = f'{basekind}_upper_{defn}_{q}'
    o = occ.set_index('condition').loc[str(row.condition) if str(row.condition) in conds else None, occ_col] if str(row.condition) in conds else np.nan
    return bu.loc[row.date, key] + o
ff['cond_key'] = ff.condition.astype(str)
mask_ok = ff.cond_key.isin(conds.keys())
ff['u_block99_mean'] = np.nan; ff['u_binom99_mean'] = np.nan; ff['u_block99_med'] = np.nan; ff['u_block95_mean'] = np.nan
ff['u_block99_mean_V10x'] = np.nan; ff['u_block99_mean_bside'] = np.nan; ff['expect_occ'] = np.nan
O = occ.set_index('condition')
for i, r in ff[mask_ok].iterrows():
    o = O.loc[r.cond_key]
    ff.at[i, 'u_block99_mean'] = bu.loc[r.date, 'block_upper_mean_99']+o['min1_c_over_c0_a4mm_used']
    ff.at[i, 'u_binom99_mean'] = bu.loc[r.date, 'binom_upper_mean_99']+o['min1_c_over_c0_a4mm_used']
    ff.at[i, 'u_block99_med'] = bu.loc[r.date, 'block_upper_med_99']+o['min1_c_over_c0_a4mm_used']
    ff.at[i, 'u_block95_mean'] = bu.loc[r.date, 'block_upper_mean_95']+o['min1_c_over_c0_a4mm_used']
    ff.at[i, 'u_block99_mean_V10x'] = bu.loc[r.date, 'block_upper_mean_99']+o['min1_c_over_c0_V10x_a4mm']
    ff.at[i, 'u_block99_mean_bside'] = bu.loc[r.date, 'block_upper_mean_99']+o['min1_c_over_c0_b_side_5mm']
    ff.at[i, 'expect_occ'] = o['expect_1_minus_exp_a4mm_used']
ff['eligible'] = ff.u_block99_mean < 0.03
ff['eligible_binom'] = ff.u_binom99_mean < 0.03
ff['eligible_V10x'] = ff.u_block99_mean_V10x < 0.03
ff['exceeds_u'] = ff.eligible & (ff.rate_mean > ff.u_block99_mean)          # blank/low conc only
ff['exceeds_u_binom'] = ff.eligible_binom & (ff.rate_mean > ff.u_binom99_mean)
ff['exceeds_u_med'] = (ff.u_block99_med < 0.03) & (ff.rate_med > ff.u_block99_med)
ff['outlier29'] = ff.rate_mean >= 0.03
ff.drop(columns=['cond_key']).to_csv(OUT/'B3_field_flags.csv', index=False, encoding='utf-8-sig')

# eligibility by (date, condition)
el = (ff.groupby(['date', 'condition', 'group'])
        .agg(fields=('fid', 'size'), u_block99=('u_block99_mean', 'first'), u_binom99=('u_binom99_mean', 'first'), u_V10x=('u_block99_mean_V10x', 'first'),
             eligible=('eligible', 'first'), n_exceed=('exceeds_u', 'sum'), n_exceed_binom=('exceeds_u_binom', 'sum'), n_exceed_med=('exceeds_u_med', 'sum'),
             mean_rate=('rate_mean', 'mean'), max_rate=('rate_mean', 'max')).reset_index())
el['share_exceed'] = el.n_exceed/el.fields
el.to_csv(OUT/'B3_eligibility_and_exceedance_by_date_condition.csv', index=False, encoding='utf-8-sig')
print('\nelig. conditions (u<3%), by condition across dates:')
print(el.groupby('condition').agg(dates=('date', 'size'), eligible_dates=('eligible', 'sum'), fields=('fields', 'sum'), n_exceed=('n_exceed', 'sum'), u_min=('u_block99', 'min'), u_max=('u_block99', 'max')).to_string())
tot = ff[ff.eligible]
print('\neligible fields', len(tot), ' exceeding u(c):', int(tot.exceeds_u.sum()), ' by group:', tot.groupby('group').exceeds_u.agg(['sum', 'size']).to_dict())

# non-eligible: concentration trend within date (analyte only)
ne = ff[(~ff.eligible) & (ff.group == 'analyte')].copy()
ne['logc'] = np.log10(ff.loc[ne.index, 'condition'].astype(float))
rows = []
for date, g in ne.groupby('date'):
    if g.logc.nunique() < 3: continue
    rho, p = spearmanr(g.logc, g.rate_mean)
    # permutation p (field level)
    obs = abs(rho); cnt = 0
    for _ in range(5000):
        if abs(spearmanr(g.logc, rng.permutation(g.rate_mean.to_numpy())).statistic) >= obs: cnt += 1
    kw = kruskal(*[x.rate_mean.to_numpy() for _, x in g.groupby('logc')])
    med_by_c = g.groupby('logc').rate_mean.median()
    rows.append(dict(date=date, n_fields=len(g), n_conc=g.logc.nunique(), spearman_rho=rho, p_spearman=p, p_perm=(cnt+1)/5001, kruskal_p=kw.pvalue,
                     median_rate_low_conc=float(med_by_c.iloc[0]), median_rate_high_conc=float(med_by_c.iloc[-1]), ratio_high_over_low=float(med_by_c.iloc[-1]/med_by_c.iloc[0]),
                     frac_fields_rate_ge3pct=float((g.rate_mean >= .03).mean())))
trend = pd.DataFrame(rows); trend.to_csv(OUT/'B3_noneligible_concentration_trend_by_date.csv', index=False, encoding='utf-8-sig')
print(trend.round(4).to_string())
json.dump(dict(C0_USED_pM=C0_USED, C0_RECALC_4MM_pM=C0_RECALC_4MM, NDRAW=NDRAW, NBLK=NBLK, BLK=BLK, SEED=C.SEED), open(OUT/'B_params.json', 'w'))
