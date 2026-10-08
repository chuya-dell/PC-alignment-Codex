"""v74 (re-run after independent critique): background estimated WITHOUT scar pixels.  The pre-image scar mask is transformed to the post image
with the lattice-implied pre->post map (Llin, tL) before it is used for the post background.  Earlier version used the pre mask on the post image
untransformed (a 5-30 px error) and over-stated the damage near the scar."""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v74_scar_distance', 'field_level/v72_real_field_readout', 'field_level/v70_ledger_center_fit'): sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd, cv2
import v74_scar_distance as E, v72_readout as R
V70, V72 = E.V70, E.V72
led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}).set_index('fid')
th = pd.read_csv(V72 / 'thresholds_v2.csv', dtype={'date': str}); th = th[(th.thr == 'G') & (th.readout == 'S3')].set_index('date').threshold
sets = json.load(open(V72 / 'sets.json')); blanks = [f for f in sets['all_blank'] if f not in sets['fixed29']]
rng = np.random.default_rng(7); pick = list(rng.choice(blanks, size=16, replace=False))
acc = {k: np.zeros(8) for k in ('n', 'p_std', 'p_mbg_wrong', 'p_mbg_mapped')}
for fid in pick:
    r = led.loc[fid]; mask = R.load_mask(fid); z = np.load(V72 / 'fields' / f'{fid}.npz'); z2 = np.load(V70 / 'fields2' / f'{fid}.npz')
    a = R.L.read(r.pre_path); b = R.L.read(r.post_path)
    M = np.hstack([z2['Llin'], z2['tL'][:, None]]).astype(np.float32)
    mask_post = cv2.warpAffine(mask.astype(np.uint8), M, (R.W, R.H), flags=cv2.INTER_NEAREST) > 0
    ca = E.bg_masked_contrast(a, mask)
    cb_wrong = E.bg_masked_contrast(b, mask); cb_map = E.bg_masked_contrast(b, mask_post)
    ctr = z['ctr_xy'].astype(float); bctr = z2['post_ctr'].astype(float)[z2['jL']]
    S3 = z['S3']; Sw = R.rnd(ca, ctr) - R.rnd(cb_wrong, bctr); Sm = R.rnd(ca, ctr) - R.rnd(cb_map, bctr)
    ok = np.isfinite(S3) & np.isfinite(Sw) & np.isfinite(Sm)
    sd = E.signed_dist_periods(mask, ctr); cls = np.digitize(sd, E.EDGES[1:-1]); t = th[r.date]
    for k in range(8):
        s = ok & (cls == k); acc['n'][k] += s.sum(); acc['p_std'][k] += (S3[s] > t).sum(); acc['p_mbg_wrong'][k] += (Sw[s] > t).sum(); acc['p_mbg_mapped'][k] += (Sm[s] > t).sum()
df = pd.DataFrame(dict(cls=E.NAMES, n=acc['n'], rate_standard_bg=acc['p_std'] / acc['n'], rate_scar_masked_bg_UNMAPPED_mask=acc['p_mbg_wrong'] / acc['n'], rate_scar_masked_bg_MAPPED_mask=acc['p_mbg_mapped'] / acc['n']))
df['ratio_mapped'] = df.rate_scar_masked_bg_MAPPED_mask / df.rate_standard_bg
df.to_csv(ROOT / 'data/results/v74_scar_distance/background_variant_v2.csv', index=False); print(df.round(5).to_string(index=False))
