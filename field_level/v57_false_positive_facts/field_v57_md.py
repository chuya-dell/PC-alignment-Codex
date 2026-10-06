"""tiny markdown table helper (tabulate is not installed)"""
import numpy as np, pandas as pd
def md(df, floatfmt='{:.4g}', pct_cols=(), pct_fmt='{:.2f}%'):
    cols = list(df.columns)
    def fmt(c, v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return '—'
        if c in pct_cols: return pct_fmt.format(100*float(v))
        if isinstance(v, (float, np.floating)): return floatfmt.format(v)
        if isinstance(v, (bool, np.bool_)): return 'はい' if v else 'いいえ'
        return str(v)
    out = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '|'.join(['---']*len(cols)) + '|']
    for _, r in df.iterrows(): out.append('| ' + ' | '.join(fmt(c, r[c]) for c in cols) + ' |')
    return '\n'.join(out)
