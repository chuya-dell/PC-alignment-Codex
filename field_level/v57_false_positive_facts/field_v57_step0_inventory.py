"""v57 step 0: inventory of stored per-pillar differences (coordinates, magnification, raw image paths). Read-only."""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np, pandas as pd, tifffile
import field_v57_common as C

f = C.field_list()
rows = []
for r in f.itertuples():
    d, xy, ids = C.load_cache(r.fid)
    A = np.c_[np.ones(len(ids)), ids[:, 0], ids[:, 1]].astype(float)
    cx, *_ = np.linalg.lstsq(A, xy[:, 0], rcond=None); cy, *_ = np.linalg.lstsq(A, xy[:, 1], rcond=None)
    res = np.hypot(A @ cx - xy[:, 0], A @ cy - xy[:, 1])
    a1 = np.array([cx[1], cy[1]]); a2 = np.array([cx[2], cy[2]])
    p1, p2 = np.linalg.norm(a1), np.linalg.norm(a2)
    p3 = np.linalg.norm(a1 - a2); p4 = np.linalg.norm(a1 + a2)
    pitch = float(np.median(sorted([p1, p2, p3, p4])[:3]))  # three shortest = hexagonal nearest-neighbour vectors
    pre = C.resolve_raw(r.date, r.path_pre.replace(chr(92), '/').split('/')[-1])
    post = C.resolve_raw(r.date, r.path_post.replace(chr(92), '/').split('/')[-1])
    shape = None
    if pre is not None:
        with tifffile.TiffFile(pre) as t: shape = t.pages[0].shape
    rows.append(dict(fid=r.fid, date=r.date, board=r.board, field=r.field, condition=r.condition, group=r.group,
        n=len(d), nan_delta=int(np.isnan(d).sum()), nan_xy=int(np.isnan(xy).sum()),
        x_min=float(xy[:, 0].min()), x_max=float(xy[:, 0].max()), y_min=float(xy[:, 1].min()), y_max=float(xy[:, 1].max()),
        img_h=None if shape is None else shape[0], img_w=None if shape is None else shape[1],
        lattice_fit_resid_med=float(np.median(res)), lattice_fit_resid_max=float(res.max()),
        pitch_px=pitch, a1x=cx[1], a1y=cy[1], a2x=cx[2], a2y=cy[2],
        pre_path=str(pre) if pre else '', post_path=str(post) if post else '',
        pre_name=pre.name if pre else ''))
t = pd.DataFrame(rows)
t['mag_guess'] = np.where(t.pitch_px > 5.5, '100x(pitch~7.3)', np.where(t.pitch_px > 2.5, '50x(pitch~3.6)', 'unknown'))
t['name_prefix_50'] = t.pre_name.str.startswith('50-')
t.to_csv(C.OUT / 'step0' / 'fields_inventory.csv', index=False, encoding='utf-8-sig')
print(t.groupby(['date']).agg(n_fields=('fid', 'size'), pre_missing=('pre_path', lambda s: (s == '').sum()),
      pitch_min=('pitch_px', 'min'), pitch_max=('pitch_px', 'max'), pitch_med=('pitch_px', 'median')))
print(t.mag_guess.value_counts().to_dict(), 'prefix50', int(t.name_prefix_50.sum()))
print('shapes', t.groupby(['img_h', 'img_w']).size().to_dict())
print('xy bounds', t.x_min.min(), t.x_max.max(), t.y_min.min(), t.y_max.max(), 'lattice resid max', t.lattice_fit_resid_max.max(), 'nan', t.nan_delta.sum(), t.nan_xy.sum())
print(t.n.describe())
print('pre names with non-simple prefix:', sorted(set(t.pre_name.str.extract(r'^(\D*?)\d')[0].fillna(''))))
