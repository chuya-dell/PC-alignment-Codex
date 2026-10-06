"""v51: v49 benchmark with explicit arms and one error definition.

Arms (same truth, same noise draw, same error = |remaining displacement| at pillar centres):
  A  no deformation, no correction   : remaining = base
  B  deformation,    no correction   : remaining = true + base        (standard affine, local field unmodelled)
  C  deformation,    correction      : remaining = true + base - fn(xy)  (selected from the v49 family)
  D  no deformation, correction      : (amplitude 0 only) remaining = base - fn(xy)
  G  deformation/none, gated         : PROVISIONAL (not pre-registered in v49). Falls back to
                                       'no correction' unless the selected candidate beats the zero-correction
                                       validation score on the same held-out validation pillars.
Seeds, fields, noise, split, candidates are exactly those of v49 (imported, not copied).
Coordinate-level: no image sampling effects (pixel quantisation, PSF, detector noise on images) are included.
"""
import argparse, json, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'field_level/v49_semisynthetic_local_correction/field_run_semisynthetic_local_correction.py'
spec = importlib.util.spec_from_file_location('v49', P); v49 = importlib.util.module_from_spec(spec); sys.modules['v49'] = v49; spec.loader.exec_module(v49)


def stats(resid):
    e = np.linalg.norm(resid, axis=1)
    return float(np.median(e)), float(np.quantile(e, .95))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--v49-output', type=Path, required=True, help='only to compare numbers, never aggregated'); a = ap.parse_args()
    (a.output / 'tables').mkdir(parents=True, exist_ok=True)
    ledger = json.loads((ROOT / 'data/results/v11_registration_precision_20260926/baseline_inputs.json').read_text(encoding='utf8'))
    centres = v49.load_centres(ledger, a.data_root, a.v49_output / 'centre_cache')
    chosen = []
    for p in sorted({x['path'] for x in ledger}):
        options = [x for x in ledger if x['path'] == p]; chosen.append(next((x for x in options if x['case_id'].endswith('axis_09')), options[0]))
    rows = []; c = 0
    for ci, item in enumerate(chosen):
        xy = centres[str(v49.local_path(item['path'], a.data_root))]; b = v49.baseline_error(xy, 1000 + ci)
        for amp in (0., .1, .3, .415, .6):
            scales = (.125, .25, .5) if amp else (.25,); kinds = ('vertical', 'diagonal', 'irregular') if amp else ('none',)
            for scale in scales:
                for kind in kinds:
                    for noise in (0., .46):
                        c += 1; key = c + ci * 10000
                        true = v49.field(xy, amp, scale, kind, key); rng = np.random.default_rng(v49.SEED + c * 101)
                        observed = true + b + rng.normal(0, noise, (len(xy), 2))
                        name, cv, fn = v49.choose_and_fit(xy, observed, key); corr = fn(xy)
                        valid = ~v49.split(xy, key); zero_score = float(np.median(np.linalg.norm(observed[valid], axis=1)))
                        use_local = cv < zero_score
                        rem_C = true + b - corr; rem_G = rem_C if use_local else true + b
                        base_row = dict(case_id=item['case_id'], amplitude_rms_px=amp, scale_fraction=scale, field_kind=kind, noise_sd_px=noise, n_pillars=len(xy),
                                        selected_candidate=name, cv_score_px=cv, zero_correction_score_px=zero_score, gate_used_local=bool(use_local))
                        arms = {'B_deform_no_correction': true + b, 'C_deform_correction': rem_C, 'G_deform_gated_provisional': rem_G}
                        if amp == 0: arms = {'A_nodeform_no_correction': b, 'D_nodeform_correction': rem_C, 'G_nodeform_gated_provisional': rem_G}
                        for arm, rem in arms.items():
                            m, p95 = stats(rem); rows.append(dict(base_row, arm=arm, median_error_px=m, p95_error_px=p95))
        print(f'source {ci+1}/{len(chosen)}', flush=True)
    d = pd.DataFrame(rows); d.to_csv(a.output / 'tables' / 'per_fov_arms.csv', index=False)

    # --- verification 3: deformation enters the standard (no-correction) error
    B = d[d.arm == 'B_deform_no_correction']
    chk = B.groupby('amplitude_rms_px').median_error_px.median().rename('median_error_B_px').reset_index()
    A = d[d.arm == 'A_nodeform_no_correction'].median_error_px.median()
    chk['nodeform_A_px'] = A; chk['ratio_B_over_amplitude'] = chk.median_error_B_px / chk.amplitude_rms_px
    chk.to_csv(a.output / 'tables' / 'check_deformation_in_standard_error.csv', index=False)
    assert (chk.median_error_B_px > 0.5 * chk.amplitude_rms_px).all(), 'deformation not in standard error'

    # --- verification 4: no mixing with superseded outputs; reproduce v49 numbers from fixed v49 table
    new = pd.read_csv(a.v49_output / 'tables' / 'paired_outcomes.csv')
    mm = d[d.arm.isin(['B_deform_no_correction', 'C_deform_correction'])].pivot_table(index=['case_id', 'amplitude_rms_px', 'scale_fraction', 'field_kind', 'noise_sd_px'], columns='arm', values='median_error_px').reset_index()
    j = mm.merge(new, on=['case_id', 'amplitude_rms_px', 'scale_fraction', 'field_kind', 'noise_sd_px'])
    rep = dict(n_joined=len(j), max_abs_diff_B_vs_v49_standard=float((j.B_deform_no_correction - j.standard_affine).abs().max()),
               max_abs_diff_C_vs_v49_selected=float((j.C_deform_correction - j.selected_local).abs().max()),
               used_superseded_dir=False, aggregated_files=['tables/per_fov_arms.csv (this run only)'])
    (a.output / 'tables' / 'check_reproduces_fixed_v49.json').write_text(json.dumps(rep, indent=2), encoding='utf8')

    # --- summaries
    tol = 0.002
    def outcome(df, arm):
        w = df.pivot_table(index=['case_id', 'amplitude_rms_px', 'scale_fraction', 'field_kind', 'noise_sd_px'], columns='arm', values='median_error_px')
        ref = [x for x in w.columns if x.startswith(('A_', 'B_'))][0]
        out = w[arm] - w[ref]
        return dict(n=len(out), improved=int((out <= -tol).sum()), worsened=int((out >= tol).sum()), unchanged=int((out.abs() < tol).sum()),
                    median_no_correction_px=float(w[ref].median()), median_arm_px=float(w[arm].median()), median_improvement_px=float(-out.median()),
                    mean_improvement_px=float(-out.mean()))
    srows = []
    for (amp, scale, noise), g in d[d.amplitude_rms_px > 0].groupby(['amplitude_rms_px', 'scale_fraction', 'noise_sd_px']):
        for arm in ('C_deform_correction', 'G_deform_gated_provisional'):
            srows.append(dict(amplitude_rms_px=amp, scale_fraction=scale, noise_sd_px=noise, arm=arm, **outcome(g[g.arm.isin([arm, 'B_deform_no_correction'])], arm)))
    for noise, g in d[d.amplitude_rms_px == 0].groupby('noise_sd_px'):
        for arm in ('D_nodeform_correction', 'G_nodeform_gated_provisional'):
            srows.append(dict(amplitude_rms_px=0., scale_fraction=np.nan, noise_sd_px=noise, arm=arm, **outcome(g[g.arm.isin([arm, 'A_nodeform_no_correction'])], arm)))
    pd.DataFrame(srows).to_csv(a.output / 'tables' / 'improvement_vs_no_correction.csv', index=False)
    gate = d[d.arm.str.startswith('G_')].groupby(['amplitude_rms_px', 'noise_sd_px']).gate_used_local.mean().rename('fraction_local_used').reset_index()
    gate.to_csv(a.output / 'tables' / 'gate_usage.csv', index=False)
    print(chk.round(3).to_string()); print(json.dumps(rep, indent=1)); print(pd.DataFrame(srows).round(3).to_string())


if __name__ == '__main__':
    main()
