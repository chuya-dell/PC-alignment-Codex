import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
import field_v57_common as C
for fam in ('Yu Gothic', 'Meiryo', 'MS Gothic'):
    if any(f.name == fam for f in font_manager.fontManager.ttflist): plt.rcParams['font.family'] = fam; break
plt.rcParams['axes.unicode_minus'] = False
OUTE = C.OUT/'stepE'
b = pd.read_csv(OUTE/'bin_counts_signed_by_field.csv')
fl = pd.read_csv(C.OUT/'stepB'/'B3_field_flags.csv', dtype={'date': str})[['fid', 'group', 'outlier29', 'date', 'condition']]
b = b.merge(fl, on='fid')
b['set29'] = np.where(b.outlier29, 'only29', 'non29')
def prof(sel, edge='all'):
    x = b[sel & (b.edge == edge)].groupby('lo').agg(n=('n', 'sum'), k=('k', 'sum'), sd=('sd', 'median')).reset_index()
    x['rate'] = x.k/x.n
    return x
res = []
for g in ('blank', 'analyte', 'mismatch'):
    for s in ('non29', 'only29'):
        for e in ('all', 'e300'):
            x = prof((b.group == g) & (b.set29 == s), e); x['group'] = g; x['fieldset'] = s; x['edge'] = e; res.append(x)
R = pd.concat(res); R.to_csv(OUTE/'E5_signed_distance_profile.csv', index=False)
# summary: share of exceeding pillars inside vs outside the mask, rates inside vs outside
rows = []
for g in ('blank', 'analyte', 'mismatch'):
    for s in ('non29', 'only29'):
        x = R[(R.group == g) & (R.fieldset == s) & (R.edge == 'all')]
        ins = x[x.lo < 0]; out = x[x.lo >= 0]; o05 = x[(x.lo >= 0) & (x.lo < 5)]; far = x[(x.lo >= 20) & np.isfinite(x.lo)]; far50 = x[x.lo >= 50]
        rows.append(dict(group=g, fieldset=s, rate_inside=ins.k.sum()/ins.n.sum(), rate_outside_0_5=o05.k.sum()/o05.n.sum(), rate_outside_ge20=far.k.sum()/far.n.sum(),
                         rate_outside_ge50=far50.k.sum()/far50.n.sum(), share_of_pillars_inside=ins.n.sum()/x.n.sum(), share_of_exceeding_pillars_inside=ins.k.sum()/x.k.sum()))
S = pd.DataFrame(rows); S.to_csv(OUTE/'E5_inside_outside_summary.csv', index=False)
pd.set_option('display.width', 250); print(S.round(4).to_string())
x = R[(R.group == 'blank') & (R.fieldset == 'non29') & (R.edge == 'all')]
print((100*x.set_index('lo').rate).round(2).loc[[-np.inf]+list(range(-20, 0, 2))+[0, 1, 2, 3, 5, 10, 20, 30, 50]].to_string())
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), sharey=False)
for ax, e, ttl in ((axs[0], 'all', '全ピラー'), (axs[1], 'e300', '画像の端から300px以上')):
    for g, col in (('blank', '#1f77b4'), ('analyte', '#d62728'), ('mismatch', '#2ca02c')):
        x = R[(R.group == g) & (R.fieldset == 'non29') & (R.edge == e) & np.isfinite(R.lo) & (R.lo >= -30) & (R.lo < 50) & (R.n > 2000)]
        ax.plot(x.lo+0.5, 100*x.rate, color=col, label=g, marker='.')
    ax.axvline(0, color='k', lw=0.8); ax.set_xlabel('傷のマスク縁からの符号付き距離(周期。負=マスクの内側の深さ)'); ax.set_ylabel('超過率(%)'); ax.set_title(ttl+'(外れ値29視野を除く)'); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUTE/'fig_E5_signed_distance_profile.png', dpi=130)
