"""v84 part B-4 (board level): detection of 1 fM-like signals by comparing the per-field excess count of a SAMPLE board (8 fields) with n_b blank boards.
Decision: mean N_exc(k0) over the 8 fields of the sample board > mean_blank_boards + t*SD_blank_boards*sqrt(1+1/n_b) (t=2.0, one-sided, a t-test-like rule).
Null boards = the 9 development blank boards (8 fields each, leave-the-test-board-out: n_b = 8 reference boards); the sample board = one of the 9 blank boards with signals injected (delta domain).
Also: the false-alarm rate on un-injected boards, and the minimal strength with power >= 90%.   usage: python v84_board_power.py"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v84_strategies as V
R = V.R; rng = np.random.default_rng(99)


def main():
    m, F = V.load_blanks(); bd = np.array([f['board'] for f in F]); boards = sorted(set(bd)); rows = []
    for key, lab, thin in (('z5', 'S5', True), ('z5p', 'S5P', False)):
        for N in (1, 8):
            for k0 in (3, 4, 5, 6, 8):
                for side in (1, -1):
                    if side < 0 and lab == 'S5P': continue
                    n0 = V.null_counts(F, key, k0, side, N).astype(float)
                    bm = {b: n0[bd == b].mean() for b in boards}
                    # false alarm
                    fa = []
                    for b in boards:
                        ref = np.array([bm[x] for x in boards if x != b]); thr = ref.mean() + 2.0 * ref.std(ddof=1) * np.sqrt(1 + 1 / len(ref)); fa.append(bm[b] > thr)
                    for A in ((1, 2, 3) if side > 0 else (2, 3, 4, 5, 6, 8)):
                        for M in (18, 46):
                            det = []
                            for b in boards:
                                ref = np.array([bm[x] for x in boards if x != b]); thr = ref.mean() + 2.0 * ref.std(ddof=1) * np.sqrt(1 + 1 / len(ref))
                                idxs = np.where(bd == b)[0]
                                for rep in range(30):
                                    tot = 0.0
                                    for i in idxs:
                                        z = V.avg_transform(F[i][key], N); n = len(z)
                                        mm = M if not (thin and side > 0) else rng.binomial(M, V.vprob(A))
                                        idx = rng.choice(n, mm, replace=False); zs = z[idx]; zn = zs + side * A / (V.rN(N) if N > 1 else 1.0)
                                        tot += n0[i] + ((np.sum(zn > k0) - np.sum(zs > k0)) if side > 0 else (np.sum(zn < -k0) - np.sum(zs < -k0)))
                                    det.append(tot / len(idxs) > thr)
                            rows.append(dict(readout=lab, N=N, k0=k0, side='dark' if side > 0 else 'bright', A=A, M=M, power=float(np.mean(det)), false_alarm=float(np.mean(fa))))
        print('done', lab, flush=True)
    t = pd.DataFrame(rows); t.to_csv(R / 'B4_board_power.csv', index=False)
    pd.set_option('display.width', 250)
    best = t.sort_values('power', ascending=False).groupby(['readout', 'N', 'side', 'A', 'M']).head(1).sort_values(['readout', 'side', 'A', 'M', 'N'])
    print(best[['readout', 'side', 'A', 'M', 'N', 'k0', 'power', 'false_alarm']].round(3).to_string())


if __name__ == '__main__':
    main()
