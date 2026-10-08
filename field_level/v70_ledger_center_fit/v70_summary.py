"""v70 summary: correspondence rate (non-substituted fraction), reasons, residuals, standard re-run table, figures."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import v70_lib as L
OUT = L.ROOT / 'data/results/v70_ledger_center_fit'
FIELDS = OUT / 'fields'


def cf_of(z, side, aref):
    ctr = z[f'{side}_ctr']; pred = z[f'{side}_pred']
    return dict(ctr=ctr.astype(np.float64), pred=pred, amp=z[f'{side}_amp'], wid=z[f'{side}_wid'],
                shift=np.linalg.norm(ctr - pred, axis=1), aref=aref)


def per_field(fid, rec):
    js = json.loads((FIELDS / f'{fid}.json').read_text(encoding='utf8'))
    if not js.get('ok'):
        return None
    z = np.load(FIELDS / f'{fid}.npz')
    A = z['matrix']; h, w = 2044, 2048
    pre = cf_of(z, 'pre', js['pre_aref']); post = cf_of(z, 'post', js['post_aref'])
    exp = z['pre_pred'] @ A[:, :2].T + A[:, 2]
    overlap = (exp[:, 0] >= 8) & (exp[:, 0] < w - 8) & (exp[:, 1] >= 8) & (exp[:, 1] < h - 8)
    j = z['match_j']; cons = z['idx_consistent']
    out = dict(fid=fid, n_pre_nodes=len(pre['ctr']), n_overlap=int(overlap.sum()), pitch_pre=js['pre_info']['pitch_px'],
               pitch_post=js['post_info']['pitch_px'], lat_resid_med_pre=js['pre_info']['residual_median_px'],
               U_deviation=js['U_deviation'], index_consistent_fraction=js['index_consistent_fraction'],
               np_coarse_resid_px=js['np_coarse_resid_px'], np_fine_resid_px=js['np_fine_resid_px'], aff_dx=js['aff_center_shift'][0], aff_dy=js['aff_center_shift'][1],
               stored_corr=js['stored_corr'], stored_rmse=js['stored_rmse'], n_std=js['n_std'], seconds=js['seconds'])
    for crit in L.CRIT:
        Sp = L.substituted(pre, crit); So = L.substituted(post, crit)
        ok = (~Sp) & (~So[j]) & cons & overlap
        out[f'rate_{crit}'] = float(ok.sum() / max(overlap.sum(), 1))
        out[f'sub_pre_{crit}'] = float(Sp[overlap].mean()); out[f'sub_post_{crit}'] = float(So.mean())
        if crit == 'main':
            ok_main = ok; Sp_m = Sp; So_m = So[j]
    # reason classification on the main set for overlap nodes that are not ok
    c = L.CRIT['main']
    bad = overlap & ~ok_main
    r_amp = bad & ((pre['amp'] < c['amp_frac'] * pre['aref']) | (post['amp'][j] < c['amp_frac'] * post['aref']))
    r_shift = bad & ~r_amp & ((pre['shift'] > c['shift_max']) | (post['shift'][j] > c['shift_max']))
    r_wid = bad & ~r_amp & ~r_shift & ((pre['wid'] < c['wid_lo']) | (pre['wid'] > c['wid_hi']) | (post['wid'][j] < c['wid_lo']) | (post['wid'][j] > c['wid_hi']))
    r_idx = bad & ~r_amp & ~r_shift & ~r_wid & ~cons
    r_dup = bad & ~r_amp & ~r_shift & ~r_wid & ~r_idx
    n = max(overlap.sum(), 1)
    out.update(fail_amp=r_amp.sum() / n, fail_shift=r_shift.sum() / n, fail_wid=r_wid.sum() / n, fail_index=r_idx.sum() / n, fail_dup=r_dup.sum() / n)
    # clumping of failures: share of failures inside worst 5% of 64px blocks
    bx = np.clip((pre['pred'][bad, 0] // 64).astype(int), 0, 31); by = np.clip((pre['pred'][bad, 1] // 64).astype(int), 0, 31)
    if bad.sum() > 0:
        cnt = np.bincount(by * 32 + bx, minlength=32 * 32); top = np.sort(cnt)[::-1][:int(0.05 * 1024)]
        out['fail_share_in_worst5pct_blocks'] = float(top.sum() / bad.sum())
    else:
        out['fail_share_in_worst5pct_blocks'] = np.nan
    # residual of mapped pre centers to post centers (non-substituted pairs)
    if ok_main.sum() > 100:
        pc = pre['ctr'][ok_main] @ A[:, :2].T + A[:, 2]; qc = post['ctr'][j][ok_main]
        r = qc - pc; out['res_med_px'] = float(np.median(np.linalg.norm(r, axis=1))); out['res_p95_px'] = float(np.percentile(np.linalg.norm(r, axis=1), 95))
        X = np.column_stack([pre['ctr'][ok_main], np.ones(ok_main.sum())]); coef = np.linalg.lstsq(X, qc, rcond=None)[0]
        r2 = qc - X @ coef; out['res_refit_med_px'] = float(np.median(np.linalg.norm(r2, axis=1))); out['res_refit_p95_px'] = float(np.percentile(np.linalg.norm(r2, axis=1), 95))
        out['affine_vs_refit_dtrans_px'] = float(np.linalg.norm((A[:, :2] @ np.array([w / 2, h / 2]) + A[:, 2]) - (coef.T[:, :2] @ np.array([w / 2, h / 2]) + coef.T[:, 2])))
    return out


def main():
    led = pd.read_csv(OUT / 'ledger.csv', dtype={'date': str, 'board': str})
    rows = []
    for r in led.itertuples():
        if (FIELDS / f'{r.fid}.json').exists():
            try:
                d = per_field(r.fid, r)
            except Exception as e:
                d = dict(fid=r.fid, summary_error=repr(e))
            if d: rows.append(d)
    t = pd.DataFrame(rows).merge(led[['fid', 'date', 'board', 'field', 'group', 'condition', 'fixed29', 'substrate_id_raw']], on='fid', how='left')
    t.to_csv(OUT / 'correspondence_by_field.csv', index=False, encoding='utf-8-sig')
    ok = t.dropna(subset=['rate_main'])
    failed = led[~led.fid.isin(ok.fid)]
    failed.to_csv(OUT / 'fields_failed_or_missing.csv', index=False, encoding='utf-8-sig')
    lines = []
    for crit in L.CRIT:
        r = ok[f'rate_{crit}']
        lines.append(dict(criterion=crit, selection='all', fields=len(r), median=r.median(), min=r.min(), p05=r.quantile(.05), mean=r.mean(),
                          lt80=int((r < .8).sum()), lt85=int((r < .85).sum()), lt95=int((r < .95).sum()),
                          pass_median95=bool(r.median() >= .95), pass_min85=bool(r.min() >= .85)))
        r = ok[ok.fixed29][f'rate_{crit}']
        lines.append(dict(criterion=crit, selection='fixed29', fields=len(r), median=r.median(), min=r.min(), p05=r.quantile(.05), mean=r.mean(),
                          lt80=int((r < .8).sum()), lt85=int((r < .85).sum()), lt95=int((r < .95).sum()),
                          pass_median95=bool(r.median() >= .95), pass_min85=bool(r.min() >= .85)))
    pd.DataFrame(lines).to_csv(OUT / 'correspondence_summary.csv', index=False, encoding='utf-8-sig')
    bydate = ok.groupby('date').rate_main.agg(['count', 'median', 'min']).reset_index(); bydate.to_csv(OUT / 'correspondence_by_date.csv', index=False)
    # reason table for fields below 85% under main criterion
    low = ok[ok.rate_main < .85].sort_values('rate_main')
    low[['fid', 'date', 'fixed29', 'group', 'rate_main', 'fail_amp', 'fail_shift', 'fail_wid', 'fail_index', 'fail_dup', 'fail_share_in_worst5pct_blocks', 'pitch_pre', 'lat_resid_med_pre', 'np_coarse_resid_px']].to_csv(OUT / 'low_correspondence_reasons.csv', index=False, encoding='utf-8-sig')
    # standard re-run vs stored
    thr = {}
    for date, g in ok.merge(led[['fid']], on='fid').groupby('date'):
        vals = []
        for fid in g[g.group == 'blank'].fid:
            vals.append(np.load(FIELDS / f'{fid}.npz')['std_delta'])
        v = np.concatenate(vals); thr[date] = (float(v.mean()), float(v.std()), float(v.mean() + 3 * v.std()))
    srows = []
    for r in ok.itertuples():
        d = np.load(FIELDS / f'{r.fid}.npz')['std_delta']; th = thr[r.date][2]
        srows.append(dict(fid=r.fid, date=r.date, group=r.group, condition=r.condition, fixed29=r.fixed29, n=len(d), positive_rate=float((d > th).mean()), threshold=th, stored_corr=r.stored_corr, stored_rmse=r.stored_rmse))
    st = pd.DataFrame(srows); st.to_csv(OUT / 'current_standard_v70_by_field.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([dict(date=k, blank_mean=v[0], blank_sd=v[1], threshold_3sd=v[2]) for k, v in thr.items()]).to_csv(OUT / 'current_standard_v70_thresholds.csv', index=False)
    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for crit in L.CRIT: ax[0].hist(ok[f'rate_{crit}'], bins=np.linspace(0, 1, 51), alpha=.5, label=crit)
    ax[0].axvline(.95, c='k', ls=':'); ax[0].axvline(.85, c='k', ls='--'); ax[0].set(xlabel='correspondence rate (non-substituted)', ylabel='fields'); ax[0].legend()
    ax[1].scatter(ok.pitch_pre, ok.rate_main, s=10, c=ok.fixed29.map({True: 'r', False: 'b'})); ax[1].set(xlabel='fitted pitch (px)', ylabel='rate (main)')
    ax[2].scatter(ok.lat_resid_med_pre, ok.rate_main, s=10, c=ok.fixed29.map({True: 'r', False: 'b'})); ax[2].set(xlabel='lattice-fit residual median (px)', ylabel='rate (main)')
    fig.tight_layout(); fig.savefig(OUT / 'correspondence_overview.png', dpi=140); plt.close(fig)
    print(pd.DataFrame(lines).to_string()); print(bydate.to_string()); print('failed/missing fields:', len(failed))
    print(ok[['fail_amp', 'fail_shift', 'fail_wid', 'fail_index', 'fail_dup']].describe().loc[['mean', '50%', 'max']])
    print('stored_corr min/median', ok.stored_corr.min(), ok.stored_corr.median(), 'np_coarse_resid >4.5px:', int((ok.np_coarse_resid_px > 4.5).sum()), 'U_dev max', ok.U_deviation.max())


if __name__ == '__main__':
    main()
