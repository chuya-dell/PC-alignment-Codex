"""Read-only transform audit and known-cause semi-synthetic comparisons.

The coarse estimator and sampler below copy only the necessary operations from
shared/registration.py, so no legacy code or unapproved masks are executed.
Original refinement code is absent. Never tune a candidate to cached outcomes.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, json, re, sys, hashlib
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from field_round1_analysis import ROOT, OUT, LOCAL, discover, fields, load, write_csv, verify_prediction, plotting

def raw_paths():
    root = next(p for p in LOCAL.iterdir() if p.name == 'raw_readonly')
    children = list(root.iterdir())
    if not any('copy_done' in p.name for p in children if p.is_file()):
        raise RuntimeError('生画像はコピー未完了')
    return {p.name: p for p in children if p.is_dir()}

def read_image(p):
    a = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None or a.ndim != 2 or a.shape != (2044, 2048):
        raise RuntimeError('画像の形式・寸法不一致 '+str(p))
    return a

def image_pairs():
    roots = raw_paths()
    root = next(p for n, p in roots.items() if n.startswith('260926'))
    table = pd.read_csv(discover()/'tables'/'table_registration_field_qc.csv', dtype={'date':str, 'board':str})
    actual = {p.name: p for p in root.iterdir() if p.is_file()}
    result = {}
    for _, r in table[table.date == '260926'].iterrows():
        before = str(r.path_pre).replace('\\', '/').split('/')[-1]
        after = str(r.path_post).replace('\\', '/').split('/')[-1]
        key = f'{r.date}_{r.board}_{int(r.field)}'
        if before in actual and after in actual:
            result[key] = (actual[before], actual[after])
    return result

def contrast_image(raw):
    a = raw.astype(np.float32)/65535.
    return a-cv2.GaussianBlur(a, (51, 51), 0)

def sample(raw, xy, bilinear=False):
    # Preserve float32 normalization/background and nine separate additions.
    a = raw.astype(np.float32)/65535.
    b = cv2.GaussianBlur(a, (51, 51), 0)
    h, w = a.shape
    valid = np.isfinite(xy).all(axis=1)
    if bilinear:
        x, y = np.floor(xy).astype(int).T
        valid &= (x >= 1) & (x < w-2) & (y >= 1) & (y < h-2)
        tx, ty = (xy-np.floor(xy)).T
    else:
        x, y = np.rint(xy).astype(int).T
        valid &= (x >= 1) & (x < w-1) & (y >= 1) & (y < h-1)
    intensity = np.zeros(len(xy))
    background = np.zeros(len(xy))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not bilinear:
                intensity[valid] += a[y[valid]+dy, x[valid]+dx]
                background[valid] += b[y[valid]+dy, x[valid]+dx]
            else:
                for ox, oy, weight in ((0,0,(1-tx)*(1-ty)),(1,0,tx*(1-ty)),(0,1,(1-tx)*ty),(1,1,tx*ty)):
                    intensity[valid] += a[y[valid]+dy+oy, x[valid]+dx+ox]*weight[valid]
                    background[valid] += b[y[valid]+dy+oy, x[valid]+dx+ox]*weight[valid]
    value = intensity-background
    value[~valid] = np.nan
    return value

def coarse(pre, post):
    a = np.uint8(np.clip(pre.astype(np.float32)/65535.*255, 0, 255))
    b = np.uint8(np.clip(post.astype(np.float32)/65535.*255, 0, 255))
    detector = cv2.ORB_create(nfeatures=12000, fastThreshold=3)
    ka, da = detector.detectAndCompute(a, None)
    kb, db = detector.detectAndCompute(b, None)
    if da is None or db is None:
        raise ValueError('特徴記述子を取得できない')
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance < .72*n.distance]
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    if len(src) < 8:
        raise ValueError('対応点不足')
    matrix, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=4000, confidence=.995)
    if matrix is None:
        raise ValueError('粗い変換推定失敗')
    return matrix.astype(np.float32)

def diagnostics(matrix):
    linear = matrix[:, :2].astype(float)
    center = np.array([(2048-1)/2, (2044-1)/2])
    delta = linear@center+matrix[:, 2]-center
    singular = np.linalg.svd(linear, compute_uv=False)
    # Exact definitions used by the current shared estimator.
    return dict(dx_center_px=float(delta[0]), dy_center_px=float(delta[1]), rotation_deg=float(np.degrees(np.arctan2(linear[1,0], linear[0,0]))),
                scale=float(np.sqrt(abs(np.linalg.det(linear)))), anisotropy=float(singular[0]/singular[-1]), determinant=float(np.linalg.det(linear)))

def expected_diagnostics(s):
    return {k:float(v) for k,v in re.findall(r"'(dx_center_px|dy_center_px|rotation_deg|scale|anisotropy|determinant)': ([\d.eE+\-]+)", str(s))}

def recovery_stage(limit=0):
    pairs = image_pairs()
    quality = pd.read_csv(discover()/'tables'/'table_alignment_quality_features_and_sensitivity_by_field.csv',dtype={'date':str,'board':str}).set_index('field_key')
    dest = OUT/'transform_recovery'
    dest.mkdir(exist_ok=True)
    selected = fields()
    selected = selected[selected['日程'] == '260926']
    if limit:
        selected = selected.iloc[:limit]
    for _, r in selected.iterrows():
        done = dest/(r.key+'.json')
        if done.exists():
            continue
        row = dict(key=r.key, 実在洗浄前パス=str(pairs[r.key][0]), 実在洗浄後パス=str(pairs[r.key][1]),
                   元変換一意確定=False, 元座標未保存=True)
        try:
            pre, post = [read_image(p) for p in pairs[r.key]]
            row.update(洗浄前内容指紋=hashlib.sha256(pairs[r.key][0].read_bytes()).hexdigest(),
                       洗浄後内容指紋=hashlib.sha256(pairs[r.key][1].read_bytes()).hexdigest())
            matrix = coarse(pre, post)
            dg = diagnostics(matrix)
            old = expected_diagnostics(quality.loc[r.key, 'coarse_qc_json'])
            row['元微小位置合わせ採用'] = bool(quality.loc[r.key, 'subpixel_refinement_accepted'])
            row['元マスク割合'] = float(re.search(r"'mask_fraction': ([\d.eE+\-]+)", str(quality.loc[r.key, 'coarse_qc_json'])).group(1))
            error = max(abs(dg[k]-old[k]) for k in dg) if len(old)==6 else np.inf
            d, xy, _ = load(r)
            postxy = xy@matrix[:, :2].T+matrix[:, 2]
            actual = sample(pre, xy)-sample(post, postxy)
            finite = np.isfinite(actual)
            row.update(診断最大絶対差=float(error), 比較可能ピラー数=int(finite.sum()), 保存ピラー数=len(d),
                       差最大絶対差=float(np.max(abs(actual[finite]-d[finite]))) if finite.any() else np.nan,
                       差平均絶対差=float(np.mean(abs(actual[finite]-d[finite]))) if finite.any() else np.nan,
                       差一致ピラー率=float(np.mean(finite & (abs(actual-d) <= 1e-6))), 推定行列=matrix.tolist(), **dg)
            row['診断と全標本値の再現'] = bool(not row['元微小位置合わせ採用'] and error<=1e-6 and finite.all() and row['差最大絶対差']<=1e-6)
            row['状態'] = '再現候補（元座標の一意性未確定）' if row['診断と全標本値の再現'] else '再現不成立'
            if row['診断と全標本値の再現']:
                np.savez_compressed(dest/(r.key+'.npz'), matrix=matrix, postxy=postxy, candidate_delta=actual)
        except Exception as e:
            row.update(状態='回復実行失敗', 理由=str(e), 診断と全標本値の再現=False)
        done.write_text(json.dumps(row,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
        print('recovery', r.key, row['状態'], flush=True)
    write_csv([json.loads(p.read_text(encoding='utf-8')) for p in sorted(dest.glob('*.json'))], 'round1_transform_recovery.csv')

def binned(xy, values):
    cell = (xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int)
    good = np.isfinite(values)
    n = np.bincount(cell[good],minlength=4096)
    s = np.bincount(cell[good],weights=values[good],minlength=4096)
    return np.divide(s,n,out=np.full(4096,np.nan),where=n>0).reshape(64,64)

def single_images_stage():
    plt = plotting()
    plt.rcParams['font.family'] = 'Yu Gothic'
    dest = OUT/'raw_review'
    dest.mkdir(exist_ok=True)
    f = fields()
    pairs = image_pairs()
    requested = ['260926_3_3','260926_5_3','260926_7_5','260926_7_8','260926_01_8']
    # Include the existing 6/7 positional structure, without declaring a defect.
    requested += ['260926_01_6','260926_01_7']
    rows = []
    for key in requested:
        r = f[f.key == key].iloc[0]
        _, xy, _ = load(r)
        images = [read_image(p) for p in pairs[key]]
        fig, axes = plt.subplots(2,3,figsize=(13,8))
        arrays = {}
        for phase, image in enumerate(images):
            values = sample(image, xy)
            m = binned(xy, values)
            arrays[f'phase{phase}_pre_native_sampling_map'] = m
            ax = axes[phase,0]
            lo, hi = np.percentile(image,[1,99.5])
            ax.imshow(image,cmap='gray',vmin=lo,vmax=hi)
            ax.set_title(('洗浄前' if phase==0 else '洗浄後')+' 生画像')
            axes[phase,1].imshow(m,cmap='viridis')
            axes[phase,1].set_title('洗浄前の固定座標での単独輝度地図')
            # Same native-coordinate crop, deliberately not implied to be registered.
            axes[phase,2].imshow(image[800:1000,800:1000],cmap='gray',vmin=lo,vmax=hi)
            axes[phase,2].set_title('座標800～1000画素の拡大（未位置合わせ）')
            rows.append(dict(key=key, 画像時点='洗浄前' if phase==0 else '洗浄後', 実在パス=str(pairs[key][phase]),
                             平均画素値=float(image.mean()), 画素標準偏差=float(image.std()), 飽和画素数=int((image==65535).sum()),
                             地図座標系='洗浄前の固定座標。洗浄後に追従していない', 内容指紋=hashlib.sha256(pairs[key][phase].read_bytes()).hexdigest()))
        fig.suptitle(key)
        fig.tight_layout()
        fig.savefig(dest/(key+'.png'),dpi=150)
        plt.close(fig)
        np.savez_compressed(dest/(key+'.npz'),**arrays)
    roots = raw_paths()
    control = next(p for n,p in roots.items() if n.startswith('260830'))
    names = sorted([p for p in control.iterdir() if p.suffix.lower() in ('.tif','.tiff')])
    fig, axes = plt.subplots(2,3,figsize=(13,8))
    for ax,p in zip(axes.flat,names):
        image = read_image(p)
        lo,hi = np.percentile(image,[1,99.5])
        ax.imshow(image,cmap='gray',vmin=lo,vmax=hi)
        ax.set_title(p.name+'（時点未確定）')
        rows.append(dict(key='260830_'+p.stem,画像時点='未確定',実在パス=str(p),平均画素値=float(image.mean()),
                         画素標準偏差=float(image.std()),飽和画素数=int((image==65535).sum()),内容指紋=hashlib.sha256(p.read_bytes()).hexdigest()))
    fig.tight_layout()
    fig.savefig(dest/'260830_six_single_images.png',dpi=150)
    plt.close(fig)
    write_csv(rows,'round1_raw_single_image_audit.csv')

def synthetic_stage():
    f = fields()
    r = f[(f['日程']=='260926') & f['ブランク']].iloc[0]
    p = image_pairs()[r.key][0]
    raw = read_image(p)
    _, xy, ids = load(r)
    fit = np.linalg.lstsq(np.column_stack([ids,np.ones(len(ids))]),xy,rcond=None)[0]
    basis = fit[:2].T
    origin = fit[2]
    reciprocal = np.linalg.inv(basis)
    vectors = np.array([reciprocal[0],reciprocal[1],reciprocal[0]+reciprocal[1]])
    yy,xx = np.indices(raw.shape)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    # Subsample deterministically to estimate a template using the pre image only.
    points = grid[::17]
    def design(points):
        phases = (points-origin)@vectors.T*2*np.pi
        return np.column_stack([np.ones(len(points)),np.cos(phases),np.sin(phases)])
    coeff = np.linalg.lstsq(design(points),raw.ravel()[::17],rcond=None)[0]
    template = (design(grid)@coeff).reshape(raw.shape).astype(np.float32)
    theta = np.radians(.14)
    linear = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    center = np.array([1024,1022])
    affine = np.column_stack([linear,center-linear@center])
    nominal = (grid-affine[:,2])@np.linalg.inv(linear).T
    local = nominal.copy()
    local[:,0] -= .30*np.sin(2*np.pi*nominal[:,1]/448)
    post_affine = (design(nominal)@coeff).reshape(raw.shape).astype(np.float32)
    post_local = (design(local)@coeff).reshape(raw.shape).astype(np.float32)
    postxy = xy@linear.T+affine[:,2]
    # Known local distortion inverse fixed-point solution at each sample.
    truth = xy.copy()
    truth[:,0] += .30*np.sin(2*np.pi*xy[:,1]/448)
    corrected = truth@linear.T+affine[:,2]
    th = pd.read_csv(OUT/'round1_thresholds.csv',dtype={'日程':str})
    threshold = float(th.loc[th['日程']=='260926','平均標準偏差'].iloc[0])
    rows,arrays = [],{}
    for name,post,target in [('ゼロ変換',template,xy),('回転のみ',post_affine,postxy),('回転と局所変位未補正',post_local,postxy),('回転と局所変位既知補正',post_local,corrected)]:
        for interpolate in (False,True):
            difference = sample(template,xy,interpolate)-sample(post,target,interpolate)
            label = name+('_双一次補間' if interpolate else '_整数丸め')
            good = np.isfinite(difference)
            m = binned(xy,difference)
            arrays[label] = m
            rows.append(dict(条件=label,標本数=int(good.sum()),差標準偏差=float(np.std(difference[good])),
                             差二乗平均平方根=float(np.sqrt(np.mean(difference[good]**2))),差平均地図分散=float(np.nanvar(m)),
                             実データ閾値を借りた参考超過率=float(np.mean(difference[good]>threshold)),
                             閾値=threshold,注意='人工像の参考値。方式別ブランク再校正なし。実データ原因の説明率ではない'))
    write_csv(rows,'round1_semisynthetic_mechanism.csv')
    np.savez_compressed(OUT/'semisynthetic_maps.npz',**arrays,template_coefficients=coeff,reciprocal_vectors=vectors,known_affine=affine)
    plt=plotting()
    plt.rcParams['font.family']='Yu Gothic'
    fig,axes=plt.subplots(4,2,figsize=(11,16))
    scale=max(np.nanpercentile(np.abs(a),99) for a in arrays.values())
    for ax,(label,m) in zip(axes.flat,arrays.items()):
        ax.imshow(m,cmap='coolwarm',vmin=-scale,vmax=scale)
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(OUT/'semisynthetic_comparison.png',dpi=150)
    plt.close(fig)

def marker_stage():
    # These seven raw-image reviews were visually checked before measurement.
    # Ranges are for the clearly visible cross-shaped reference lines only.
    ranges = {
        '260926_3_3': ((1800,2044),(1800,2048)),
        '260926_5_3': ((1770,1950),(1800,2048)),
        '260926_7_5': ((50,220),(20,180)),
        '260926_7_8': ((60,240),(1800,2048)),
        '260926_01_8': ((60,230),(1800,2048)),
        '260926_01_6': ((1750,1980),(10,180)),
        '260926_01_7': ((1820,2044),(1770,2048)),
    }
    pairs = image_pairs()
    f = fields().set_index('key',drop=False)
    threshold = pd.read_csv(OUT/'round1_thresholds.csv',dtype={'日程':str}).set_index('日程').loc['260926']
    rows, overlap_rows = [],[]
    def robust_line(x,y):
        keep = np.ones(len(x),bool)
        for _ in range(6):
            coeff = np.polyfit(x[keep],y[keep],1)
            residual = y-(x*coeff[0]+coeff[1])
            mad = np.median(abs(residual[keep]-np.median(residual[keep])))
            keep = abs(residual) <= max(4,3*1.4826*mad)
        return coeff,int(keep.sum()),float(np.median(abs(residual[keep])))
    for key,(hr,vr) in ranges.items():
        raw = read_image(pairs[key][0])
        # Bright horizontal ridge; strong vertical edge at the visible marker.
        smooth = cv2.GaussianBlur(raw.astype(np.float32),(0,0),1.)
        x = np.arange(250,1750)
        yh = np.argmax(smooth[hr[0]:hr[1],:][:,x],axis=0)+hr[0]
        horizontal,n1,err1 = robust_line(x,yh)
        edge = np.abs(cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3))
        y = np.arange(300,1700)
        xv = np.argmax(edge[y,vr[0]:vr[1]],axis=1)+vr[0]
        vertical,n2,err2 = robust_line(y,xv)
        angles = [float(np.degrees(np.arctan(horizontal[0]))%180),float(np.degrees(np.arctan2(1,vertical[0]))%180)]
        rows.append(dict(key=key,洗浄前パス=str(pairs[key][0]),水平線傾き=horizontal[0],水平線切片画素=horizontal[1],
                         垂直線横傾き=vertical[0],垂直線横切片画素=vertical[1],水平線角度=angles[0],垂直線角度=angles[1],
                         水平支持画素=n1,垂直支持画素=n2,水平近似誤差中央値画素=err1,垂直近似誤差中央値画素=err2,
                         確認='作成した生画像比較図で線状マーカーを確認。欠陥承認ではなく、除外・マスクに使わない'))
        d,xy,_ = load(f.loc[key])
        horizontal_dist = abs(xy[:,1]-horizontal[0]*xy[:,0]-horizontal[1])/np.sqrt(1+horizontal[0]**2)
        vertical_dist = abs(xy[:,0]-vertical[0]*xy[:,1]-vertical[1])/np.sqrt(1+vertical[0]**2)
        near = np.minimum(horizontal_dist,vertical_dist)<=64
        for method,th in [('平均標準偏差',threshold['平均標準偏差']),('中央値絶対偏差',threshold['中央値絶対偏差閾値'])]:
            pos = d>th
            overlap_rows.append(dict(key=key,定義=method,参照線から64画素以内ピラー占有率=float(near.mean()),
                                     外れ値の参照線64画素以内占有率=float(pos[near].sum()/pos.sum()) if pos.any() else np.nan,
                                     近傍外れ値割合=float(pos[near].mean()),遠方外れ値割合=float(pos[~near].mean()),
                                     注意='追加の記述的重なり。解析からの除外や仮説A再検定ではない'))
    write_csv(rows,'round1_visually_confirmed_marker_geometry.csv')
    write_csv(overlap_rows,'round1_marker_proximity.csv')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['recovery','raw','synthetic','markers'])
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    print('prediction unchanged',verify_prediction(),flush=True)
    if args.stage=='recovery':
        recovery_stage(args.limit)
    elif args.stage=='raw':
        single_images_stage()
    elif args.stage=='synthetic':
        synthetic_stage()
    else:
        marker_stage()
