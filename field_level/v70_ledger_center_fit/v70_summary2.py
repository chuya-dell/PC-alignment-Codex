"""v70 summary (revised): correspondence rate from fields2 (smooth lattice, lattice-implied physical map)."""
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
import v70_centers2 as C
OUT = L.ROOT / 'data/results/v70_ledger_center_fit'
F1, F2 = OUT / 'fields', OUT / 'fields2'
W, H = 2048, 2044


def analyse(fid):
    j2 = json.loads((F2 / f'{fid}.json').read_text(encoding='utf8'))
    if not j2.get('ok'): return None
    j1 = json.loads((F1 / f'{fid}.json').read_text(encoding='utf8'))
    z = np.load(F2 / f'{fid}.npz')
    pre = dict(ctr=z['pre_ctr'].astype(float), pred=z['pre_pred'], amp=z['pre_amp'], wid=z['pre_wid'], aloc=z['pre_aloc'])
    post = dict(ctr=z['post_ctr'].astype(float), pred=z['post_pred'], amp=z['post_amp'], wid=z['post_wid'], aloc=z['post_aloc'])
    A = z['A']; exp = z['pre_pred_lin'] @ A[:, :2].T + A[:, 2]
    ov = (exp[:, 0] >= 8) & (exp[:, 0] < W - 8) & (exp[:, 1] >= 8) & (exp[:, 1] < H - 8)
    expL = z['pre_pred_lin'] @ z['Llin'].T + z['tL']
    ovL = (expL[:, 0] >= 8) & (expL[:, 0] < W - 8) & (expL[:, 1] >= 8) & (expL[:, 1] < H - 8)
    out = dict(fid=fid, n_nodes=len(pre['ctr']), n_overlap_L=int(ovL.sum()), n_overlap_A=int(ov.sum()),
               pitch_pre=j2['pitch_pre'], pitch_post=j2['pitch_post'], pitch_ratio=j2['pitch_post'] / j2['pitch_pre'],
               corner_disagree_px=j2['corner_disagree_px'], scale_A=j2['scale_A'], scale_L=j2['scale_L'], frac_corr_px=j2['frac_corr_px'],
               consistent_L=j2['consistent_L'], consistent_A=j2['consistent_A'], dL_median=j2['dL_median'], dL_p95=j2['dL_p95'],
               poly_vs_lin_med=j2['pre_info']['poly_vs_linear_median'], poly_vs_lin_p95=j2['pre_info']['poly_vs_linear_p95'], poly_vs_lin_max=j2['pre_info']['poly_vs_linear_max'],
               poly_resid_med=j2['pre_info']['poly_resid_median'], lat_resid_med_pre=j2['pre_info']['residual_median_px'],
               anchor_ecc_fine_resid=j1['np_fine_resid_px'], anchor_confirmed=bool(j1['np_fine_resid_px'] < 3.6),
               stored_corr=j1['stored_corr'], U_deviation=j1['U_deviation'])
    for crit in C.CRIT2:
        Sp = C.sub_flags(pre, crit); So = C.sub_flags(post, crit)
        okL = (~Sp) & (~So[z['jL']]) & z['consL'] & ovL
        okA = (~Sp) & (~So[z['jA']]) & z['consA'] & ov
        out[f'rate_{crit}'] = float(okL.sum() / max(ovL.sum(), 1)); out[f'rateA_{crit}'] = float(okA.sum() / max(ov.sum(), 1))
        out[f'sub_pre_{crit}'] = float(Sp[ovL].mean()); out[f'sub_post_{crit}'] = float(So.mean())
        if crit == 'main': okm = okL; Spm = Sp; Som = So[z['jL']]
    c = C.CRIT2['main']
    bad = ovL & ~okm
    sh_pre = np.linalg.norm(pre['ctr'] - pre['pred'], axis=1); sh_post = np.linalg.norm(post['ctr'] - post['pred'], axis=1)[z['jL']]
    r_amp = bad & ((pre['amp'] < c['amp_frac'] * pre['aloc']) | (post['amp'][z['jL']] < c['amp_frac'] * post['aloc'][z['jL']]))
    r_shift = bad & ~r_amp & ((sh_pre > c['shift_max']) | (sh_post > c['shift_max']))
    r_idx = bad & ~r_amp & ~r_shift & ~z['consL']
    r_oth = bad & ~r_amp & ~r_shift & ~r_idx
    n = max(ovL.sum(), 1)
    out.update(fail_amp=r_amp.sum() / n, fail_shift=r_shift.sum() / n, fail_index=r_idx.sum() / n, fail_other=r_oth.sum() / n)
    if bad.sum() > 0:
        bx = np.clip((pre['pred'][bad, 0] // 64).astype(int), 0, 31); by = np.clip((pre['pred'][bad, 1] // 64).astype(int), 0, 31)
        cnt = np.bincount(by * 32 + bx, minlength=1024); out['fail_share_worst5pct_blocks'] = float(np.sort(cnt)[::-1][:51].sum() / bad.sum())
    if okm.sum() > 100:
        pc = pre['ctr'][okm] @ z['Llin'].T + z['tL']; qc = post['ctr'][z['jL']][okm]
        r = qc - pc; rn = np.linalg.norm(r, axis=1)
        out['res_L_med_px'] = float(np.median(rn)); out['res_L_p95_px'] = float(np.percentile(rn, 95))
        pa = pre['ctr'][okm] @ A[:, :2].T + A[:, 2]
        # A-based map after periodic wrapping onto the nearest post pillar (only meaningful modulo the lattice)
        ra = post['ctr'][z['jA']][okm] - pa; out['res_A_med_px'] = float(np.median(np.linalg.norm(ra, axis=1)))
        X = np.column_stack([pre['ctr'][okm], np.ones(okm.sum())]); coef = np.linalg.lstsq(X, qc, rcond=None)[0]
        rr = np.linalg.norm(qc - X @ coef, axis=1); out['res_refit_med_px'] = float(np.median(rr)); out['res_refit_p95_px'] = float(np.percentile(rr, 95))
    return out


def main():
    led = pd.read_csv(OUT / 'ledger.csv', dtype={'date': str, 'board': str})
    rows = []
    for r in led.itertuples():
        if (F2 / f'{r.fid}.json').exists() and (F1 / f'{r.fid}.json').exists():
            try:
                d = analyse(r.fid)
            except Exception as e:
                d = dict(fid=r.fid, summary_error=repr(e))
            if d: rows.append(d)
    t = pd.DataFrame(rows).merge(led[['fid', 'date', 'board', 'field', 'group', 'condition', 'fixed29', 'substrate_id_raw']], on='fid', how='left')
    t.to_csv(OUT / 'correspondence_by_field_v2.csv', index=False, encoding='utf-8-sig')
    ok = t.dropna(subset=['rate_main']); led[~led.fid.isin(ok.fid)].to_csv(OUT / 'fields_failed_or_missing_v2.csv', index=False, encoding='utf-8-sig')
    lines = []
    for sel, name in [(ok, 'all'), (ok[ok.fixed29], 'fixed29'), (ok[ok.anchor_confirmed], 'anchor_confirmed'), (ok[~ok.anchor_confirmed], 'anchor_unconfirmed')]:
        for crit in C.CRIT2:
            for col, tag in [(f'rate_{crit}', 'L_map'), (f'rateA_{crit}', 'A_map')]:
                r = sel[col]
                lines.append(dict(selection=name, criterion=crit, mapping=tag, fields=len(r), median=r.median(), min=r.min(), p05=r.quantile(.05), mean=r.mean(),
                                  lt80=int((r < .8).sum()), lt85=int((r < .85).sum()), lt95=int((r < .95).sum()),
                                  pass_median95=bool(r.median() >= .95), pass_min85=bool(r.min() >= .85)))
    s = pd.DataFrame(lines); s.to_csv(OUT / 'correspondence_summary_v2.csv', index=False, encoding='utf-8-sig')
    ok.groupby('date').rate_main.agg(['count', 'median', 'min']).reset_index().to_csv(OUT / 'correspondence_by_date_v2.csv', index=False)
    low = ok[ok.rate_main < .85].sort_values('rate_main')
    cols = ['fid', 'date', 'fixed29', 'group', 'rate_main', 'fail_amp', 'fail_shift', 'fail_index', 'fail_other', 'fail_share_worst5pct_blocks', 'corner_disagree_px', 'poly_vs_lin_p95', 'anchor_confirmed']
    low[cols].to_csv(OUT / 'low_correspondence_reasons_v2.csv', index=False, encoding='utf-8-sig')
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    ax[0].hist(ok.rate_main, bins=np.linspace(0, 1, 51), alpha=.6, label='L map'); ax[0].hist(ok.rateA_main, bins=np.linspace(0, 1, 51), alpha=.6, label='A map')
    ax[0].axvline(.95, c='k', ls=':'); ax[0].axvline(.85, c='k', ls='--'); ax[0].set(xlabel='correspondence rate (main)', ylabel='fields'); ax[0].legend()
    ax[1].scatter(ok.corner_disagree_px, ok.rateA_main, s=8, c=ok.fixed29.map({True: 'r', False: 'b'})); ax[1].set(xscale='log', xlabel='|A - lattice linear map| at corners (px)', ylabel='rate with A map')
    ax[2].scatter(ok.poly_vs_lin_p95, ok.rate_main, s=8, c=ok.fixed29.map({True: 'r', False: 'b'})); ax[2].set(xlabel='distortion: cubic vs linear lattice p95 (px)', ylabel='rate (L map)')
    ax[3].scatter(ok.pitch_ratio, ok.corner_disagree_px, s=8); ax[3].set(yscale='log', xlabel='post/pre fitted pitch', ylabel='|A - L| corners (px)')
    fig.tight_layout(); fig.savefig(OUT / 'correspondence_overview_v2.png', dpi=130); plt.close(fig)
    pd.set_option('display.width', 250)
    print(s[s.criterion == 'main'].to_string())
    print(ok[['fail_amp', 'fail_shift', 'fail_index', 'fail_other']].describe().loc[['mean', '50%', 'max']])
    print('anchor_confirmed', int(ok.anchor_confirmed.sum()), '/', len(ok), '; pitch_ratio range', ok.pitch_ratio.min(), ok.pitch_ratio.max())
    print('res_L med/p95 (median over fields)', ok.res_L_med_px.median(), ok.res_L_p95_px.median())
    print('corner_disagree quantiles', ok.corner_disagree_px.quantile([.5, .75, .9, .95]).round(2).to_dict())


if __name__ == '__main__':
    main()
