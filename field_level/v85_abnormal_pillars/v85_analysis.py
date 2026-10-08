"""v85 analysis: (2) abnormal pillars by distance to the cross-scar core / bright band / field edge vs a uniform-area reference (all valid pillars of the same fields),
(4) by date, board, position 1-8 (pooled maps, count table, position-6/7 test with within-board permutation), (5) by DNA concentration (all 632 fields).
Abnormal = post fit failed & P readout valid & |z|>6 (k=4/8 as sensitivity).   usage: python v85_analysis.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
for f in ('Yu Gothic', 'Meiryo', 'MS Gothic'):
    if any(f in x.name for x in font_manager.fontManager.ttflist): plt.rcParams['font.family'] = f; break
plt.rcParams['axes.unicode_minus'] = False
V84 = ROOT / 'data/results/v84_1fM_strategies'; V85 = ROOT / 'data/results/v85_abnormal_pillars'
VAULT = Path(r'W:\GoogleDrive\chuya2816\Obsidian Vault\ラボノート\05_解析\261008_【未読】偽陽性ゼロ化_原因確定と標準経路\図と表')
EDGES = {'core': [0, 2, 4, 8, 16, 32, 1e9], 'band': [-1e9, 0, 2, 4, 8, 16, 1e9], 'edge': [0, 2, 4, 8, 16, 32, 1e9]}
LAB = {'core': ['0-2', '2-4', '4-8', '8-16', '16-32', '>32'], 'band': ['中', '外0-2', '2-4', '4-8', '8-16', '>16'], 'edge': ['0-2', '2-4', '4-8', '8-16', '16-32', '>32']}
rng = np.random.default_rng(5)


def poisson_ci(k, mu):
    lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0; hi = stats.chi2.ppf(0.975, 2 * (k + 1)) / 2
    return lo / mu if mu > 0 else np.nan, hi / mu if mu > 0 else np.nan


def distance_profiles():
    a = pd.read_csv(V85 / 'abn_pillars.csv'); r = pd.read_csv(V85 / 'ref_hist.csv')
    out = []
    for scope, ss in (('傷の外(2周期より外)', a.sd > 2), ('全部(傷の中を含む)', a.sd > -1e9)):
        for kz in (6, 4):
            aa = a[ss & (a.z.abs() > kz)]
            for nm in ('core', 'band', 'edge'):
                hcol = ('ho_' if scope.startswith('傷の外') else 'h_') + nm
                rr = r[r.has_band] if nm == 'band' else r[r.n_lines.notna() if 'n_lines' in r else r.fid.notna()]
                ref = np.sum([json.loads(x) for x in rr[hcol]], axis=0).astype(float)
                fids = set(rr.fid)
                vals = aa[aa.fid.isin(fids)][nm].values
                vals = vals[np.isfinite(vals)]; obs = np.histogram(vals, bins=EDGES[nm])[0]; exp = len(vals) * ref / ref.sum()
                for i, l in enumerate(LAB[nm]):
                    lo, hi = poisson_ci(obs[i], exp[i])
                    out.append(dict(scope=scope, k=kz, metric=nm, bin=l, observed=int(obs[i]), expected=float(exp[i]), ratio=float(obs[i] / exp[i]) if exp[i] > 0 else np.nan, ratio_lo=lo, ratio_hi=hi, n_total=len(vals), n_fields=len(fids)))
    t = pd.DataFrame(out); t.to_csv(V85 / 'E2_distance_ratio.csv', index=False); return t


def plot_distance(t):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for r_, scope in enumerate(('傷の外(2周期より外)', '全部(傷の中を含む)')):
        for c_, (nm, ttl) in enumerate((('core', '十字傷の芯からの距離(周期)'), ('band', '明部帯からの距離(周期)'), ('edge', '視野の端からの距離(周期)'))):
            q = t[(t.scope == scope) & (t.metric == nm) & (t.k == 6)]; ax = axs[r_, c_]
            x = np.arange(len(q)); ax.bar(x - .2, q.observed / q.n_total, width=.4, color='#d62728', label='異常ピラー(6σ超)')
            ax.bar(x + .2, q.expected / q.n_total, width=.4, color='#9e9e9e', label='比較対象(同じ面積の一様な分布)'); ax.set_xticks(x); ax.set_xticklabels(q.bin, fontsize=8)
            for xi, (o, rt) in enumerate(zip(q.observed, q.ratio)):
                ax.text(xi - .2, o / q.n_total.iloc[0], f'{int(o)}\n×{rt:.1f}' if np.isfinite(rt) else f'{int(o)}', ha='center', va='bottom', fontsize=7)
            ax.set_title(f'{scope}:{ttl}(n={int(q.n_total.iloc[0])})', fontsize=9); ax.set_ylabel('割合'); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(VAULT / 'E2_距離別の異常ピラー.png', dpi=120); plt.close()


def position_date():
    led = pd.read_csv(ROOT / 'data/results/v70_ledger_center_fit/ledger.csv', dtype={'date': str, 'board': str})
    meta = pd.read_csv(V84 / 'field_meta.csv', dtype={'date': str, 'board': str}); meta = meta[meta.ok]
    rows = []; pts = []
    for r in meta.itertuples():
        z = np.load(V84 / 'cache' / f'{r.fid}.npz'); okP = np.isfinite(z['z5p']); abn = z['abn'] & okP; zz = np.abs(z['z5p']); ex = z['excl']
        d = dict(fid=r.fid, date=r.date, board=r.board, pos=int(r.field), group=r.group, conc=r.conc_M, fixed29=r.fixed29, sigma=r.sigma, n_valid=r.n_valid,
                 abn6_out=int((abn & (zz > 6) & ex).sum()), abn6_all=int((abn & (zz > 6)).sum()), abn4_out=int((abn & (zz > 4) & ex).sum()), abn8_out=int((abn & (zz > 8) & ex).sum()),
                 abn_cand_out=int((abn & ex).sum()), pos_sign=float(np.mean(z['z5p'][abn & (zz > 6)] > 0)) if (abn & (zz > 6)).any() else np.nan)
        rows.append(d)
        idx = np.where(abn & (zz > 6) & ex)[0]
        for i in idx: pts.append((r.fid, r.date, int(r.field), r.group, float(z['ctr'][i, 0]), float(z['ctr'][i, 1])))
    t = pd.DataFrame(rows); t.to_csv(V85 / 'E4_counts_all632.csv', index=False)
    p = pd.DataFrame(pts, columns=['fid', 'date', 'pos', 'group', 'x', 'y']); p.to_csv(V85 / 'abn_points_all632_k6_out.csv', index=False)
    return t, p


def position_stats(t):
    out = {}
    for nm, q in (('blank69', t[t.group == 'blank']), ('fixed29', t[t.fixed29]), ('all632', t)):
        g = q.groupby('pos').abn6_out.agg(['mean', 'sum', 'count']); out[nm] = g.round(2).to_dict('index')
        kw = stats.kruskal(*[q[q.pos == p].abn6_out.values for p in range(1, 9) if (q.pos == p).any()]); out[nm + '_kruskal'] = dict(H=float(kw.statistic), p=float(kw.pvalue))
        m67 = q.pos.isin([6, 7]); u = stats.mannwhitneyu(q[m67].abn6_out, q[~m67].abn6_out); out[nm + '_mw_6_7_vs_rest'] = dict(mean67=float(q[m67].abn6_out.mean()), mean_rest=float(q[~m67].abn6_out.mean()), p=float(u.pvalue))
        # within-board permutation
        obs = q[m67].abn6_out.mean() - q[~m67].abn6_out.mean(); cnt = 0; N = 5000
        grp = [(b, d.index.values, d.pos.values, d.abn6_out.values) for b, d in q.groupby(['date', 'board'])]
        for _ in range(N):
            s67 = []; sr = []
            for b, idx, pos, v in grp:
                pp = rng.permutation(pos); s67 += list(v[np.isin(pp, [6, 7])]); sr += list(v[~np.isin(pp, [6, 7])])
            cnt += (np.mean(s67) - np.mean(sr)) >= obs
        out[nm + '_perm_6_7'] = dict(obs_diff=float(obs), p_one_sided=float((cnt + 1) / (N + 1)))
        # dispersion of counts (Poisson?)
        out[nm + '_dispersion'] = dict(mean=float(q.abn6_out.mean()), var=float(q.abn6_out.var()), index=float(q.abn6_out.var() / q.abn6_out.mean()))
    # date/board effect (blank)
    q = t[t.group == 'blank']; kw = stats.kruskal(*[d.abn6_out.values for _, d in q.groupby('date')]); out['blank_by_date_kruskal'] = dict(H=float(kw.statistic), p=float(kw.pvalue))
    out['blank_by_date_mean'] = q.groupby('date').abn6_out.mean().round(2).to_dict()
    out['abnormal_sign_positive_fraction_all'] = float(t.pos_sign.mean())
    json.dump(out, open(V85 / 'E4_position_stats.json', 'w'), indent=1, ensure_ascii=False, default=float); return out


def plot_position_date(t, p):
    # (a) table date x position (blank + fixed29 separately would be busy: show blank, all-632 counts as heat tables)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (ttl, q) in zip(axs, (('ブランク69視野:異常ピラー(6σ超、傷の外)の個数', t[t.group == 'blank']), ('全632視野(DNAなど含む)の平均個数', t))):
        pv = q.pivot_table(index=['date', 'board'], columns='pos', values='abn6_out', aggfunc='mean')
        im = ax.imshow(pv.values, cmap='Reds', aspect='auto'); ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns); ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels([f'{d} 基板{b}' for d, b in pv.index], fontsize=8)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if np.isfinite(v): ax.text(j, i, f'{v:.0f}' if v >= 10 else f'{v:.1f}', ha='center', va='center', fontsize=7)
        ax.set_title(ttl, fontsize=10); ax.set_xlabel('位置'); plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(VAULT / 'E4_日程基板位置別の個数.png', dpi=120); plt.close()
    # (b) pooled maps per position (blank + fixed29 fields only, k6 outside scar) and per date
    pb = p[p.group == 'blank']
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for pos in range(1, 9):
        ax = axs[(pos - 1) // 4, (pos - 1) % 4]; q = pb[pb.pos == pos]; nf = (t[(t.group == 'blank') & (t.pos == pos)]).shape[0]
        ax.scatter(q.x, q.y, s=14, c='#d62728', alpha=.7); ax.set_xlim(0, 2048); ax.set_ylim(2044, 0); ax.set_aspect('equal'); ax.set_title(f'位置{pos}:ブランク{nf}視野の合計 {len(q)}個', fontsize=9); ax.grid(alpha=.2)
    plt.suptitle('位置1〜8別:異常ピラー(6σ超、傷の外)の視野内座標(ブランク視野すべて重ね合わせ)'); plt.tight_layout(); plt.savefig(VAULT / 'E4_位置別の地図.png', dpi=110); plt.close()
    dates = sorted(pb.date.unique()); fig, axs = plt.subplots(3, 3, figsize=(14, 14))
    for ax, d in zip(axs.ravel(), dates):
        q = pb[pb.date == d]; nf = t[(t.group == 'blank') & (t.date == d)].shape[0]; ax.scatter(q.x, q.y, s=12, c=q.pos, cmap='tab10', vmin=1, vmax=8, alpha=.8); ax.set_xlim(0, 2048); ax.set_ylim(2044, 0); ax.set_aspect('equal'); ax.set_title(f'{d}:ブランク{nf}視野 {len(q)}個(色=位置)', fontsize=9)
    plt.tight_layout(); plt.savefig(VAULT / 'E4_日程別の地図.png', dpi=100); plt.close()


def concentration(t, p):
    """Each board carries ONE concentration, so the unit is the board (date x board): board mean/median of the field counts."""
    q = t.copy(); out = {}
    bd = q.groupby(['date', 'board']).agg(n=('fid', 'size'), conc=('conc', 'first'), group=('group', 'first'), mean=('abn6_out', 'mean'), median=('abn6_out', 'median'), mean_all=('abn6_all', 'mean')).reset_index()
    bd.to_csv(V85 / 'E5_board_table.csv', index=False)
    an = bd[bd.group.isin(['analyte', 'blank']) & bd.conc.notna()].copy(); an['logc'] = np.log10(an.conc.replace(0, np.nan))
    typ = an[an['median'] < 10]; out['outlier_boards(mean>10)'] = an[an['mean'] > 10][['date', 'board', 'conc', 'mean', 'median']].round(2).to_dict('records')
    rows = []
    for d, g in an.groupby('date'):
        gg = g[g.conc > 0]
        if len(gg) >= 5:
            rho, pv = stats.spearmanr(gg.logc, gg['median']); rows.append(dict(date=d, n_boards=len(gg), rho_median=float(rho), p=float(pv), blank_median=float(g[g.conc == 0]['median'].mean()) if (g.conc == 0).any() else np.nan))
    out['within_date_spearman_board_median'] = rows
    out['sign_consistent'] = dict(pos=int(sum(r['rho_median'] > 0 for r in rows)), neg=int(sum(r['rho_median'] < 0 for r in rows)), n=len(rows))
    a2 = an[an.conc > 0].copy(); a2['adj'] = a2['median'] - a2.groupby('date')['median'].transform('median'); rho, pv = stats.spearmanr(a2.logc, a2.adj); out['pooled_adj_median_spearman'] = dict(rho=float(rho), p=float(pv), n=len(a2))
    bl = an[an.conc == 0]; out['blank_boards_median_mean'] = float(bl['median'].mean()); out['analyte_boards_median_mean'] = float(a2['median'].mean())
    out['mannwhitney_blank_vs_analyte_board_median'] = dict(p=float(stats.mannwhitneyu(bl['median'], a2['median']).pvalue), n_blank=len(bl), n_analyte=len(a2))
    # per concentration class: mean over boards (median of fields), and fraction of outlier boards
    cls = []
    for c, g in an.groupby('conc'):
        cls.append(dict(conc=c, n_boards=len(g), mean_of_board_median=float(g['median'].mean()), n_outlier=int((g['mean'] > 10).sum())))
    out['by_conc'] = cls
    json.dump(out, open(V85 / 'E5_conc_stats.json', 'w'), indent=1, ensure_ascii=False, default=float)
    cm = plt.get_cmap('tab10'); dates = sorted(an.date.unique())
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for i, d in enumerate(dates):
        g = an[an.date == d].sort_values('conc'); x = np.log10(g.conc.replace(0, 1e-16)); axs[0].plot(x, g['median'], 'o-', color=cm(i), label=d, ms=4, alpha=.8); axs[1].plot(x, g['mean'], 'o-', color=cm(i), ms=4, alpha=.8)
    axs[0].set_title('基板ごとの異常ピラー数の中央値(視野8個の中央値)'); axs[1].set_title('基板ごとの平均(外れ値の基板を含む)'); axs[1].set_yscale('symlog', linthresh=10)
    for ax in axs[:2]: ax.set_xlabel('log10(濃度 M)(ブランク=-16)'); ax.set_ylabel('異常ピラー(6σ超、傷の外)/視野')
    axs[0].legend(fontsize=6, ncol=2)
    cls_ = pd.DataFrame(cls); axs[2].bar([('ブランク' if c == 0 else f'{c:.0e}') for c in cls_.conc], cls_.mean_of_board_median, color='#d62728'); axs[2].set_title('濃度ごと(基板の中央値の平均)'); axs[2].tick_params(axis='x', labelsize=7, rotation=45)
    for i, (c, n, no) in enumerate(zip(cls_.conc, cls_.n_boards, cls_.n_outlier)): axs[2].text(i, cls_.mean_of_board_median.iloc[i], f'n={n} 外{no}', ha='center', va='bottom', fontsize=6)
    plt.tight_layout(); plt.savefig(VAULT / 'E5_濃度別の異常ピラー.png', dpi=120); plt.close()
    # pooled positions: concentration classes (ブランク / ≤1pM / 10pM-0.1nM / 1nM) and outlier boards
    key = q.merge(bd[['date', 'board', 'mean']].rename(columns={'mean': 'bmean'}), on=['date', 'board'])
    groups = [('ブランク', key[(key.conc == 0)]), ('≤1pM', key[(key.conc > 0) & (key.conc <= 1e-12)]), ('10pM-0.1nM', key[(key.conc > 1e-12) & (key.conc <= 1e-10)]), ('1nM', key[(key.conc >= 1e-9) & (key.bmean < 50)]), ('外れ値の基板(平均>50)', key[key.bmean > 50])]
    fig, axs = plt.subplots(1, 5, figsize=(20, 4.4))
    for ax, (l, sel) in zip(axs, groups):
        pq = p[p.fid.isin(sel.fid)]; ax.scatter(pq.x, pq.y, s=3, c='#d62728', alpha=.5); ax.set_xlim(0, 2048); ax.set_ylim(2044, 0); ax.set_aspect('equal'); ax.set_title(f'{l}:{len(sel)}視野、{len(pq)}個(視野あたり {len(pq) / max(len(sel), 1):.1f})', fontsize=9)
    plt.tight_layout(); plt.savefig(VAULT / 'E5_濃度別の位置.png', dpi=100); plt.close()
    return out, [(l, len(sel)) for l, sel in groups]


if __name__ == '__main__':
    t_dist = distance_profiles(); plot_distance(t_dist)
    pd.set_option('display.width', 250)
    print(t_dist[(t_dist.k == 6)].round(2).to_string())
    t, p = position_date(); st = position_stats(t); plot_position_date(t, p)
    print(json.dumps({k: v for k, v in st.items() if 'blank69' in k or k.startswith(('blank_by', 'abnormal'))}, indent=0, ensure_ascii=False, default=float))
    out, nc = concentration(t, p); print(json.dumps({k: v for k, v in out.items() if k != 'by_board'}, indent=0, ensure_ascii=False, default=float)); print(nc)
