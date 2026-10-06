"""v57 step C: current false-positive picture without any exclusion (reference values for the before/after comparison of round 3). Stored values only."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import field_v57_common as C

OUT = C.OUT/'stepC'
f = pd.read_csv(C.OUT/'stepB'/'B3_field_flags.csv', dtype={'date': str, 'board': str})
f['conc_label'] = f.condition.map({'0': 'ブランク', '1e-09': '1nM', '1e-10': '100pM', '1e-11': '10pM', '1e-12': '1pM', '1e-13': '100fM', '1e-14': '10fM', '1e-15': '1fM',
                                    'mismatch': 'ミスマッチ', 'blank_reference': '参考ブランク', '100uL_10fM': '10fM・100µL'})
f['ge3_mean'] = f.rate_mean >= 0.03; f['ge3_med'] = f.rate_med >= 0.03
def summarize(g):
    return pd.Series(dict(fields=len(g), eligible=int(g.eligible.sum()), exceed_u=int(g.exceeds_u.sum()), exceed_u_share_of_eligible=(g.exceeds_u.sum()/g.eligible.sum()) if g.eligible.sum() else np.nan,
                          ge3pct_mean=int(g.ge3_mean.sum()), ge3pct_med=int(g.ge3_med.sum()), share_ge3pct_mean=g.ge3_mean.mean(),
                          mean_rate=g.rate_mean.mean(), median_rate=g.rate_mean.median(), max_rate=g.rate_mean.max()))
t_date = f.groupby('date').apply(summarize, include_groups=False).reset_index()
t_conc = f.groupby('conc_label').apply(summarize, include_groups=False).reset_index()
t_dc = f.groupby(['date', 'conc_label']).apply(summarize, include_groups=False).reset_index()
t_board = f.groupby(['date', 'board', 'conc_label']).apply(summarize, include_groups=False).reset_index()
t_group = f.groupby('group').apply(summarize, include_groups=False).reset_index()
for n, t in (('C1_by_date', t_date), ('C1_by_concentration', t_conc), ('C1_by_date_and_concentration', t_dc), ('C1_by_date_board_concentration', t_board), ('C1_by_group', t_group)):
    t.to_csv(OUT/f'{n}.csv', index=False, encoding='utf-8-sig')
# the 29 fields
f[f.outlier29][['fid', 'date', 'board', 'field', 'conc_label', 'group', 'rate_mean', 'rate_med']].sort_values('rate_mean', ascending=False).to_csv(OUT/'C1_outlier29_fields.csv', index=False, encoding='utf-8-sig')
# reference values (C2)
ref = dict(n_fields=int(len(f)), n_blank_main=int((f.group == 'blank').sum()),
           outlier_counts_mean_def={str(k): int((f.rate_mean >= k).sum()) for k in (0.01, 0.02, 0.03, 0.04, 0.05)},
           outlier_counts_median_def={str(k): int((f.rate_med >= k).sum()) for k in (0.01, 0.02, 0.03, 0.04, 0.05)},
           eligible_fields=int(f.eligible.sum()), exceed_u_fields=int(f.exceeds_u.sum()), exceed_u_share=float(f.exceeds_u.sum()/f.eligible.sum()),
           exceed_u_by_group=f[f.eligible].groupby('group').exceeds_u.agg(['sum', 'size']).to_dict('index'),
           exceed_u_binom=int(f.exceeds_u_binom.sum()), exceed_u_median_definition=int(f.exceeds_u_med.sum()),
           blank_rate_mean=float(f[f.group == 'blank'].rate_mean.mean()), blank_rate_max=float(f[f.group == 'blank'].rate_mean.max()),
           note='reference for round-3 before/after comparison; mean-based exceedance with the date blank threshold (v25 stored differences); no exclusions')
json.dump(ref, open(OUT/'C2_reference_values.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(ref, indent=1, ensure_ascii=False)[:1500])
pd.set_option('display.width', 250, 'display.max_columns', 30)
print(t_date.round(4).to_string()); print(t_conc.round(4).to_string()); print(t_group.round(4).to_string())
# concordance of the 3% outlier definition with exceed_u among eligible
print(pd.crosstab(f.outlier29, f.exceeds_u))
# board / date structure of the 29
print(f[f.outlier29].groupby(['date', 'board', 'conc_label']).size().to_string())
