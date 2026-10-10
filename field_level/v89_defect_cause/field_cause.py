"""Codexが事前登録第2版どおりに開発547視野の13中心を診断する。"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import time
import math
import itertools
from functools import lru_cache
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import beta, fisher_exact
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(r'C:\Users\chuya\PC-alignment-fp\data\results')
OUT = ROOT / 'data/results/v89_defect_cause'
PREREG = Path(r'W:\GoogleDrive\chuya2816\Obsidian Vault\_依頼記録\偽陽性ゼロ化_C1D\261010_不再現4中心_仮説と判断基準_事前登録.md')
TARGETS = {8, 40, 157, 309}
ALLOWED = set()
READ_LOG = []
LABEL = '事後の再評価（独立な検証ではない）'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(name, rows, columns=None):
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    df.insert(0, 'evaluation', LABEL)
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig', mode='x')
    return df


def load_dev(folder, fid):
    assert fid in ALLOWED and not fid.startswith('261008')
    path = SRC / folder / f'{fid}.npz'
    READ_LOG.append(dict(path=str(path), fid=fid, purpose='development_nonblank_only'))
    return np.load(path, allow_pickle=False)


def rate(k, n):
    return k / n if n else np.nan


def ci(k, n):
    if not n:
        return np.nan, np.nan
    return (0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1. if k == n else float(beta.ppf(.975, k+1, n-k)))


def comparison(n1, k1, n2, k2):
    r1, r2 = rate(k1, n1), rate(k2, n2)
    if not n1 or not n2:
        return r1, r2, np.nan, np.nan, 'none', 0, 0
    # Equal rates have no uniquely lower side; use early deterministically.
    low = 'early' if r1 <= r2 else 'late'
    ln, lk, lr, hr = (n1, k1, r1, r2) if low == 'early' else (n2, k2, r2, r1)
    ratio = lr / hr if hr else np.nan
    p = float(fisher_exact([[k1, n1-k1], [k2, n2-k2]], alternative='two-sided').pvalue)
    return r1, r2, ratio, p, low, ln, lk


def classify(ratio, p, ln, lk):
    labels = []
    if ln >= 30 and ratio <= .1 and p < .0005:
        labels.append('H1')
    if .1 <= ratio <= .5:
        labels.append('部分的な減少')
    if ln < 10:
        labels.append('H2示唆')
    if 5 <= lk <= 9 and ratio >= .5:
        labels.append('H3示唆')
    return '／'.join(labels) or 'なし'


def fisher_rx2(ns, ks):
    """Exact conditional probability-ordering Fisher test, no simulation.

    Integer combination products allow exact tie handling. Dynamic extrema
    bound each subtree; Vandermonde's identity sums wholly accepted subtrees.
    """
    ns, ks = tuple(map(int, ns)), tuple(map(int, ks))
    total, successes = sum(ns), sum(ks)
    if len(ns) < 2 or successes in (0, total):
        return 1.
    if successes > total // 2:
        ks = tuple(n-k for n, k in zip(ns, ks))
        successes = total-successes
    weights = [tuple(math.comb(n, k) for k in range(min(n, successes)+1)) for n in ns]
    observed = math.prod(w[k] for w, k in zip(weights, ks))
    tails = [sum(ns[i:]) for i in range(len(ns)+1)]

    @lru_cache(None)
    def bounds(i, s):
        if i == len(ns):
            return (1, 1) if s == 0 else (0, 0)
        lo, hi = max(0, s-tails[i+1]), min(ns[i], s)
        candidates = [(weights[i][k], bounds(i+1, s-k)) for k in range(lo, hi+1)]
        return min(w*b[0] for w, b in candidates), max(w*b[1] for w, b in candidates)

    def visit(i, s, prefix):
        mn, mx = bounds(i, s)
        if prefix*mx <= observed:
            return prefix*math.comb(tails[i], s)
        if prefix*mn > observed:
            return 0
        return sum(visit(i+1, s-k, prefix*weights[i][k])
                   for k in range(max(0, s-tails[i+1]), min(ns[i], s)+1))

    return visit(0, successes, 1) / math.comb(total, successes)


def components(points):
    if points.empty:
        return pd.DataFrame(columns=['component', 'x', 'y', 'n', 'nf']), np.array([], int)
    xy = points[['x', 'y']].to_numpy()
    pairs = cKDTree(xy).query_pairs(10., output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy)))
    _, labels = connected_components(graph, directed=False)
    result = points.assign(component=labels).groupby('component').agg(
        x=('x', 'mean'), y=('y', 'mean'), n=('x', 'size'), nf=('fid', 'nunique')).reset_index()
    return result, labels


def main():
    global ALLOWED
    started = time.perf_counter()
    utc_start = datetime.now(timezone.utc).isoformat()
    assert not OUT.exists(), 'Codexは既存の結果フォルダを上書きしない。'
    OUT.mkdir(parents=True)
    ledger = SRC / 'v85_abnormal_pillars/E4_counts_all632.csv'
    defs_path = ROOT / 'data/raw/v88_camera_defects_unapproved.csv'
    inputs = [ledger, defs_path, PREREG, ROOT/'field_level/v88_defect_exclusion/field_stage1.py']
    inputs += sorted((ROOT/'data/results/v88_defect_exclusion').glob('centers_*.csv'))
    inputs += [ROOT/'data/results/v88_defect_exclusion'/n for n in ['center_reproducibility.csv', 'date_split.csv']]
    input_hashes = {str(p): sha(p) for p in inputs}
    meta = pd.read_csv(ledger, dtype={'fid': str, 'date': str, 'board': str})
    assert len(meta) == 632 and meta.fid.is_unique
    assert not meta.date.eq('261008').any() and not meta.fid.str.startswith('261008').any()
    ALLOWED = set(meta.fid)
    bm = meta.groupby(['date', 'board']).abn6_out.mean().rename('bmean').reset_index()
    nb = meta.merge(bm, on=['date', 'board'])
    nb = nb[(nb.group != 'blank') & (nb.bmean < 50)].copy()
    assert len(nb) == 547 and meta.group.eq('blank').sum() == 69
    defs = pd.read_csv(defs_path)
    assert len(defs) == 13 and defs.defect_id.is_unique
    rp = pd.read_csv(inputs[-2])
    controls = set(rp.loc[rp.both_reproduced, 'defect_id'].astype(int))
    assert len(controls) == 9 and set(defs.defect_id) - controls == TARGETS
    dates = sorted(nb.date.unique())
    assert len(dates) == 9
    split = pd.read_csv(inputs[-1], dtype={'date': str})
    assert set(split[split.group == 'early'].date) == set(dates[:5])
    rows, points, zrows = [], [], []
    values = {(int(d.defect_id), date): [] for d in defs.itertuples() for date in dates}
    for index, r in enumerate(nb.itertuples()):
        with load_dev('v84_1fM_strategies/cache', r.fid) as z, load_dev('v70_ledger_center_fit/fields2', r.fid) as c:
            xy = z['ctr'].astype(float) @ c['Llin'].T + c['tL']
            zz = np.abs(z['z5p'].astype(float))
            finite, excl = np.isfinite(zz), z['excl'].astype(bool)
            valid = finite & excl
            keep = z['abn'].astype(bool) & valid & (zz > 6)
            points.extend((r.fid, r.date, x, y) for x, y in xy[keep])
            tree = cKDTree(xy[np.isfinite(xy).all(axis=1)])
            xy_ids = np.flatnonzero(np.isfinite(xy).all(axis=1))
            for d in defs.itertuples():
                ids20 = xy_ids[tree.query_ball_point([d.x_camera, d.y_camera], 20.)]
                ids10 = ids20[np.linalg.norm(xy[ids20] - [d.x_camera, d.y_camera], axis=1) <= 10.]
                masks = {'i': ids10[valid[ids10]], 'ii': ids10, 'iii': ids20[valid[ids20]]}
                row = dict(fid=r.fid, date=r.date, board=r.board, pos=r.pos, group=r.group, conc=r.conc, defect_id=int(d.defect_id))
                for name, ids in masks.items():
                    row[f'near_pillars_{name}'] = len(ids)
                    row[f'detection_pillars_{name}'] = int(keep[ids].sum())
                    row[f'opportunity_{name}'] = bool(len(ids))
                    row[f'detected_{name}'] = bool(keep[ids].any())
                for radius, ids in [(10, ids10), (20, ids20)]:
                    row[f'missing_excl_false_{radius}'] = int((~excl[ids]).sum())
                    row[f'missing_z5p_nonfinite_{radius}'] = int((~finite[ids]).sum())
                    row[f'missing_union_{radius}'] = int((~valid[ids]).sum())
                    vals = zz[ids[finite[ids]]]
                    row[f'max_abs_z5p_{radius}'] = float(vals.max()) if len(vals) else np.nan
                values[int(d.defect_id), r.date].extend(zz[ids10[finite[ids10]]].tolist())
                row['max_abs_z5p_10_valid'] = float(zz[masks['i']].max()) if len(masks['i']) else np.nan
                rows.append(row)
        if index % 75 == 0:
            print(f'Codex: development fields {index}/547', flush=True)
    fields = pd.DataFrame(rows)
    assert len(fields) == 547*13 and not fields.duplicated(['fid', 'defect_id']).any()
    for mode in ['i', 'ii', 'iii']:
        assert (fields[f'detected_{mode}'] <= fields[f'opportunity_{mode}']).all()
    table('field_table.csv', fields)
    pts = pd.DataFrame(points, columns=['fid', 'date', 'x', 'y'])
    date_rows = []
    for (did, date), g in fields.groupby(['defect_id', 'date']):
        row = dict(defect_id=did, date=date, n_fields=len(g))
        for mode in ['i', 'ii', 'iii']:
            n, k = int(g[f'opportunity_{mode}'].sum()), int(g[f'detected_{mode}'].sum())
            low, high = ci(k, n)
            row.update({f'opportunity_fields_{mode}': n, f'detection_fields_{mode}': k,
                        f'rate_{mode}': rate(k, n), f'ci95_low_{mode}': low, f'ci95_high_{mode}': high})
        date_rows.append(row)
        vals = np.asarray(values[did, date])
        zrows.append(dict(defect_id=did, date=date, n_finite_pillars=len(vals),
                          median_abs_z5p=float(np.median(vals)) if len(vals) else np.nan,
                          p95_abs_z5p=float(np.quantile(vals, .95)) if len(vals) else np.nan,
                          population='all_finite_pillars_within_10px_including_excl_false'))
    dc = pd.DataFrame(date_rows)
    table('date_center_table.csv', dc)
    table('z5p_by_date.csv', zrows)
    boundaries = []
    for did, g in dc.groupby('defect_id'):
        g = g.set_index('date').loc[dates]
        counts = g[['opportunity_fields_i', 'detection_fields_i']].to_numpy(int)
        differences = []
        for cut in range(1, 9):
            n1, k1 = counts[:cut].sum(axis=0); n2, k2 = counts[cut:].sum(axis=0)
            differences.append(abs(rate(k1, n1)-rate(k2, n2)))
        for cut in range(1, 9):
            row = dict(defect_id=int(did), boundary=f'{dates[cut-1]}|{dates[cut]}', n_dates_early=cut, n_dates_late=9-cut)
            for mode in ['i', 'ii', 'iii']:
                ns = g[f'opportunity_fields_{mode}'].to_numpy(int)
                ks = g[f'detection_fields_{mode}'].to_numpy(int)
                n1, n2, k1, k2 = int(ns[:cut].sum()), int(ns[cut:].sum()), int(ks[:cut].sum()), int(ks[cut:].sum())
                r1, r2, ratio, p, low, ln, lk = comparison(n1, k1, n2, k2)
                row.update({f'early_opportunity_{mode}': n1, f'late_opportunity_{mode}': n2,
                            f'early_detection_{mode}': k1, f'late_detection_{mode}': k2,
                            f'early_rate_{mode}': r1, f'late_rate_{mode}': r2, f'low_high_rate_ratio_{mode}': ratio,
                            f'fisher_p_{mode}': p, f'low_side_{mode}': low,
                            f'classification_{mode}': classify(ratio, p, ln, lk)})
            observed = differences[cut-1]
            perm_stats = []
            for combo in itertools.combinations(range(9), cut):
                mask = np.zeros(9, bool); mask[list(combo)] = True
                n1, k1 = counts[mask].sum(axis=0); n2, k2 = counts[~mask].sum(axis=0)
                perm_stats.append(abs(rate(k1, n1)-rate(k2, n2)))
            finite_stats = np.asarray(perm_stats); finite_stats = finite_stats[np.isfinite(finite_stats)]
            row['date_permutation_p'] = float(np.mean(finite_stats >= observed-1e-12)) if np.isfinite(observed) and len(finite_stats) else np.nan
            row['date_permutation_n'] = len(finite_stats)
            row['boundary_shift_p'] = float(np.mean(np.asarray(differences) >= observed-1e-12)) if np.isfinite(observed) else np.nan
            row['classification'] = row['classification_i']
            boundaries.append(row)
    bs = pd.DataFrame(boundaries)
    hits = set(bs.loc[bs.classification_i.str.contains('H1'), 'defect_id'])
    control_hits = hits & controls
    h1_unusable = len(control_hits) >= 2
    bs['h1_control_status'] = '判定不能' if h1_unusable else '対照条件を満たす'
    table('boundary_scan.csv', bs)
    table('control_check.csv', [dict(defect_id=did, h1_raw_pass=did in hits,
         n_controls=9, n_controls_h1=len(control_hits), h1_status='判定不能' if h1_unusable else '対照条件を満たす') for did in sorted(controls)])
    board_rows = []
    for (did, date), g in fields.groupby(['defect_id', 'date']):
        if not g.detected_i.any():
            continue
        q = g[g.opportunity_i]
        for factor in ['board', 'pos']:
            categories = q.groupby(factor).detected_i.agg(['size', 'sum'])
            p = fisher_rx2(categories['size'], categories['sum'])
            for cat, c in categories.iterrows():
                board_rows.append(dict(defect_id=int(did), date=date, factor=factor, category=str(cat),
                     opportunity_fields=int(c['size']), detection_fields=int(c['sum']), rate=rate(c['sum'], c['size']),
                     fisher_p=p, threshold=.05/13, significant=p < .05/13,
                     test='two_sided_Fisher_Freeman_Halton_exact_probability_ordering'))
    bp = table('board_pos_within_date.csv', board_rows)
    print('Codex: exact within-date tests complete', flush=True)
    unadopted, sensitivity = [], []
    cached_components = {}
    for cut in range(1, 9):
        for side, ds in [('early', dates[:cut]), ('late', dates[cut:])]:
            q = pts[pts.date.isin(ds)].reset_index(drop=True)
            comp, labels = components(q)
            cached_components[cut, side] = comp
            if cut == 5:
                old = pd.read_csv(ROOT/f'data/results/v88_defect_exclusion/centers_{side}.csv')
                adopted = comp[(comp.n >= 8) & (comp.nf >= 10)]
                assert len(old) == len(adopted)
                assert np.allclose(old[['x','y','n','nf']], adopted[['x','y','n','nf']], atol=1e-8)
                xy = q[['x', 'y']].to_numpy()
                for d in defs[defs.defect_id.isin(TARGETS)].itertuples():
                    distances = np.linalg.norm(xy - [d.x_camera, d.y_camera], axis=1)
                    for c in comp[(comp.n < 8) | (comp.nf < 10)].itertuples():
                        nearest = float(distances[labels == c.component].min())
                        if nearest <= 200:
                            unadopted.append(dict(defect_id=int(d.defect_id), half=side, component=int(c.component),
                                 x=c.x, y=c.y, n_points=int(c.n), n_fields=int(c.nf), nearest_point_distance_px=nearest,
                                 centroid_distance_px=float(np.hypot(c.x-d.x_camera, c.y-d.y_camera)),
                                 reason='n_points<8' if c.n < 8 else 'n_fields<10'))
        for minimum in [5, 10]:
            both_ids = []
            for d in defs.itertuples():
                ok = True
                for side in ['early', 'late']:
                    comp = cached_components[cut, side]
                    q = comp[(comp.n >= 8) & (comp.nf >= minimum)]
                    distance = cKDTree(q[['x','y']]).query([d.x_camera,d.y_camera])[0] if len(q) else np.inf
                    ok &= distance <= 5
                if ok:
                    both_ids.append(int(d.defect_id))
            sensitivity.append(dict(boundary=f'{dates[cut-1]}|{dates[cut]}', original_half_split=cut == 5,
                                    min_fields=minimum, min_points=8, n_both_reproduced=len(both_ids),
                                    reproduced_ids=json.dumps(both_ids), used_for_decision=False))
        print(f'Codex: clustering boundary {cut}/8', flush=True)
    table('unadopted_clusters.csv', unadopted, columns=['defect_id','half','component','x','y','n_points','n_fields','nearest_point_distance_px','centroid_distance_px','reason'])
    st = table('sensitivity.csv', sensitivity)
    table('array_read_audit.csv', READ_LOG)
    assert len(READ_LOG) == 1094 and all(r['fid'] in ALLOWED and not r['fid'].startswith('261008') for r in READ_LOG)
    assert input_hashes == {str(p): sha(p) for p in inputs}
    table('input_integrity.csv', [dict(path=p, sha256_before=h, sha256_after=h, unchanged=True) for p,h in input_hashes.items()])
    report = ['# Codexによる不再現4中心の原因切り分け', '',
              'Codexは事前登録第2版に従い、開発用547非ブランク視野×13中心を解析した。Codexは261008で始まる識別子を含む資料と配列を開かなかった。',
              'Codexは欠陥候補を診断対象として扱い、欠陥表の承認状態と標準経路を変更しなかった。', '', '## 表から読める事実', '',
              f'Codexは対照9中心のうち{len(control_hits)}中心（{sorted(control_hits)}）で時間依存の数値基準を検出した。対照条件は時間依存の判定を「'+('判定不能' if h1_unusable else '対照条件を満たす')+'」とする。', '',
              '|中心|境|前側 検出/機会|後側 検出/機会|低率/高率|Fisher両側確率|日付並べ替え確率|機械判定|',
              '|---|---|---|---|---|---|---|---|']
    for did in sorted(TARGETS):
        q = bs[bs.defect_id == did]
        # Show every H1 boundary and the strongest boundary otherwise; predictions also shown.
        selected = q[q.classification_i.str.contains('H1')]
        if selected.empty:
            selected = q.loc[[q.fisher_p_i.idxmin()]]
        if did in (157, 309):
            selected = pd.concat([selected, q[q.boundary == '260922|260923']]).drop_duplicates('boundary')
        if did == 8:
            selected = pd.concat([selected, q[q.boundary == '260829|260922']]).drop_duplicates('boundary')
        if did == 40:
            selected = pd.concat([selected, q[q.classification_i.str.contains('H3')]]).drop_duplicates('boundary')
        for r in selected.itertuples():
            boundary_text = r.boundary.replace('|', '\\|')
            report.append(f'|{did}|{boundary_text}|{r.early_detection_i}/{r.early_opportunity_i}|{r.late_detection_i}/{r.late_opportunity_i}|{r.low_high_rate_ratio_i:.6g}|{r.fisher_p_i:.6g}|{r.date_permutation_p:.6g}|{r.classification}|')
    report += ['', '## 予想と結果の照合・解釈', '']
    for did in sorted(TARGETS):
        q = bs[bs.defect_id == did]
        passed = q[q.classification_i.str.contains('H1')]
        if did in (157,309):
            prediction_match = '一致' if (passed.boundary == '260922|260923').any() else '不一致'
            detected_dates = dc[(dc.defect_id == did) & (dc.detection_fields_i > 0)].date.astype(str).tolist()
            report.append(f'Codexは中心{did}の予想境260922|260923で基準が成立するかの照合を「{prediction_match}」と記録した。Codexは実際の検出日を{detected_dates}と記録し、基準の成立だけでは予想境を出現日と特定できないと解釈した。')
        if did == 8:
            # v2 prediction: disappearance during first period, particularly 260829–260922.
            predicted = passed[passed.boundary == '260829|260922']
            report.append('Codexは中心8の予想期間260829〜260922との照合を「'+('一致' if len(predicted) else '不一致')+'」と記録した。')
        if len(passed) and not h1_unusable:
            report.append(f'Codexは中心{did}に時間依存の統計基準の成立を認めるが、カメラ・光学系・試料条件の物理的原因を特定していない。')
        else:
            report.append(f'Codexは中心{did}を「原因未確定」とした。Codexは対照条件による判定不能、部分的な減少、機会不足の示唆、閾値条件の示唆を原因の証明として扱わない。')
    significant = bp[bp.significant].drop_duplicates(['defect_id','date','factor'])
    report.append(f'Codexは同日内の板・位置検定で{len(significant)}組の中心×日付×要因が確率0.05/13未満になったことを記録した。')
    report.append('Codexは板番号を日付ごとの分類として扱い、異なる日付の同じ板番号を同一の物理的板と解釈しない。Codexは日付と板・試料条件の交絡を残した。')
    report += ['', '|中心|日付順の検出視野数（260825・260827・260828・260829・260922・260923・260924・260926・260927）|', '|---|---|']
    for did in sorted(TARGETS):
        daily_counts = dc[dc.defect_id == did].set_index('date').loc[dates, 'detection_fields_i'].astype(int).tolist()
        report.append(f'|{did}|'+ '・'.join(map(str, daily_counts))+'|')
    report += ['', 'Codexは中心157・309の検出が最後の2日に集中する事実を一過性の手がかりとして記録した。Codexはそれ以降の日付を解析していないため、継続的な出現と一過性を区別できないと解釈した。']
    sensitivity_half = st[(st.original_half_split) & (st.min_fields == 5)].iloc[0]
    report.append(f'Codexは最小視野数5の前後半再現数を{sensitivity_half.n_both_reproduced}/13と記録した。Codexは感度表の全8境の再現数を合否に使わなかった。')
    report += ['', '## 計算上の定義（未指定の細部は仮置き）', '',
               'Codexは主機会を除外条件が真かつ有限値の10画素近傍、第二機会を除外条件を問わない全10画素近傍、第三機会を主機会の20画素近傍とした。Codexは全機会で元の検出条件を維持した。',
               'Codexは欠けた数について、除外条件が偽、値が非有限、両者の和集合を別々に数えた。Codexは強度分布を除外条件が偽の有限値も含む全10画素近傍ピラーから計算した。',
               'Codexは複数の判定が成立する境に判定を併記した。Codexは両側ゼロ検出の率比を未定義とした。',
               'Codexは日付並べ替え検定で日付内の全視野を保ち、境の前側日付数を固定した全組合せの絶対検出率差を比較した。Codexは別列boundary_shift_pに全8境の絶対率差が観測境以上となる割合も記録した。',
               'Codexは板・位置検定で主機会がある全カテゴリの多行二列のFisher正確検定を計算し、固定周辺和の下で観測表以下の確率を持つ表を整数演算で総和した。Codexはカテゴリ行に同じ全体検定確率を記載した。',
               'Codexは未採用点群を距離10画素の連結成分のうち点数8未満または視野数10未満と定義し、対象中心への最近接点距離が200画素以内の成分を残した。Codexは重心距離も併記した。',
               'Codexは感度解析でも点数8以上・重心距離5画素以内の条件を維持した。Codexは元の前後半の採用中心座標・点数・視野数の一致を既存表で検証した。',
               'Codexは1094件の配列読込をarray_read_audit.csvに残し、入力文書・表・旧コードの内容が実行前後で不変であることをinput_integrity.csvに残した。',
               'Codexは指定された新しい2フォルダ以外へ書き込まず、共有ノート更新・既存コード更新・Gitの取得とコミットを行わなかった。', '',
               '## 基準への異議', '',
               'Codexは13中心×8境の正確検定確率が同日内の視野間依存を扱わない点を指摘する。Codexは登録された判定基準を変更せず、日付単位の並べ替え結果を補助として併記した。',
               'Codexは境ずらし8通りの確率の最小値が1/8であり、確率0.0005と同じ判定には使えない点を指摘する。Codexは固定日付数の全組合せ検定にも日付の交換可能性の仮定が必要と記録した。',
               'Codexは対照9中心の時間的安定が保証されない点と、低率/高率が0.1と0.5の境で複数の判定に入る点を記録した。',
               'Codexは一過性、強度変化、試料条件、中心移動について事前登録に合否基準がないため、日別表・強度表・未採用点群を証拠資料として提示し、原因の確定判定を追加しなかった。']
    (OUT/'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    notes = ['# Codexによる第89版の実装記録', '', '## 実装内容・旧版からの変更点', '',
             'Codexは第88版の開発入力・変換座標・検出条件を維持し、13中心の機会3定義、日別信頼区間、8境の機械判定、日付並べ替え、同日内の板・位置検定、未採用点群、再現感度を追加した。', '',
             '## 結果概要と保存先', '',
             'Codexはdata/results/v89_defect_cause/に結果を保存した。Codexは中心8・157・309で時間依存の基準成立、中心40で部分的な減少と原因未確定を記録した。Codexは対照中心42でも時間依存の基準成立を検出した。', '',
             '## 既知の問題・未解決事項', '',
             'Codexは物理的原因、日付と試料条件の交絡、同日内の依存、中心移動の有無を確定していない。Codexは予想した境の基準成立と、実際の出現時期の特定を区別した。',
             'Codexは書込範囲の指示に従い、この記録をコードフォルダではなく新しい結果フォルダに置いた。Codexは入力文書と旧結果の内容を変更していない。',
             'Codexはv89_manifest.csvの自己参照ハッシュを除き、全出力ファイルと実行スクリプトのSHA-256を記録した。']
    (OUT/'NOTES.md').write_text('\n'.join(notes)+'\n', encoding='utf-8')
    elapsed = time.perf_counter()-started
    manifest = []
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            manifest.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size,
                                 command='python -B field_level/v89_defect_cause/field_cause.py', elapsed_seconds=elapsed, utc_start=utc_start))
    manifest.append(dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(Path(__file__)), bytes=Path(__file__).stat().st_size,
                         command='python -B field_level/v89_defect_cause/field_cause.py', elapsed_seconds=elapsed, utc_start=utc_start))
    table('v89_manifest.csv', manifest)
    print(json.dumps(dict(n_fields=547, n_centers=13, raw_h1_ids=sorted(hits), control_h1_ids=sorted(control_hits),
                          h1_unusable=h1_unusable, seconds=elapsed, prohibited_arrays_opened=0), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
