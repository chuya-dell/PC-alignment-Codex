from __future__ import annotations

import numpy as np

import cv2

from scipy import fft

yy,xx=np.indices((64,64))

def correlate_full(a, b):
    shape = tuple(x+y-1 for x, y in zip(a.shape, b.shape))
    return fft.irfftn(fft.rfftn(a, shape)*fft.rfftn(b, shape), shape)

def local_peaks(a):
    return np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]))+1

def half_width(a, peak):
    height = a[peak]/2
    left, right = peak, peak
    while left > 0 and a[left] > height:
        left -= 1
    while right < len(a)-1 and a[right] > height:
        right += 1
    xl = left+(height-a[left])/(a[left+1]-a[left]) if a[left+1] != a[left] else left
    xr = right-1+(height-a[right-1])/(a[right]-a[right-1]) if a[right] != a[right-1] else right
    return xr-xl

def angular_difference(a, b):
    return np.abs((np.asarray(a)-b+90) % 180-90)

def geometry(shape):
    h, w = shape
    fy = fft.fftfreq(h, d=32)[:, None]
    fx = fft.rfftfreq(w, d=32)[None, :]
    radius = np.hypot(fy, fx)
    band = (radius >= 1/1024) & (radius <= 1/128)
    axis = (np.degrees(np.arctan2(np.broadcast_to(fy, radius.shape), np.broadcast_to(fx, radius.shape)))+90) % 180
    angles = np.arange(0, 180, 10)
    # Equal +/-5 degree bins define the direction search; +/-15 determines concentration.
    direction_bin = np.floor(((axis+5) % 180)/10).astype(int)
    sectors = np.array([band & (direction_bin == j) for j in range(len(angles))])
    multiplicity = np.full(radius.shape, 2.)
    multiplicity[:, 0] = 1
    if w % 2 == 0:
        multiplicity[:, -1] = 1
    window = np.outer(np.hanning(h), np.hanning(w))
    return band, axis, angles, sectors, multiplicity, window, radius

def linear_autocorrelation(a):
    mask = np.isfinite(a)
    centered = np.where(mask, a-np.nanmean(a), 0)
    numerator = correlate_full(centered, centered[::-1, ::-1])
    overlap = correlate_full(mask.astype(float), mask[::-1, ::-1].astype(float))
    ac = np.divide(numerator, overlap, out=np.full_like(numerator, np.nan), where=overlap > .5)
    var = np.nanvar(a)
    if var > 0:
        ac /= var
    else:
        ac[:] = np.nan
    return ac, overlap

def projection(a, axis):
    # Average along the band axis, leaving a one-dimensional normal coordinate.
    yy, xx = np.indices(a.shape)
    normal = np.radians(axis-90)
    coord = xx*np.cos(normal)+yy*np.sin(normal)
    bins = np.rint(coord-coord.min()).astype(int)
    mask = np.isfinite(a)
    n = np.bincount(bins[mask], minlength=bins.max()+1)
    val = np.bincount(bins[mask], weights=a[mask], minlength=len(n))
    p = np.divide(val, n, out=np.full(len(n), np.nan), where=n > 0)
    keep = n >= max(3, .2*n.max())
    if not keep.any():
        return np.array([])
    lo, hi = np.flatnonzero(keep)[[0, -1]]
    return p[lo:hi+1]

def projection_metrics(a, axis, spectrum_period):
    p = projection(a, axis)
    if len(p) < 12 or not np.isfinite(p).all() or np.var(p) == 0:
        return np.nan, np.nan, False
    c = p-p.mean()
    ac = correlate_full(c, c[::-1])[len(c)-1:]/np.arange(len(c), 0, -1)
    ac /= ac[0]
    peaks = local_peaks(ac)
    candidates = peaks[(peaks >= 4) & (peaks <= 32) & (peaks <= len(p)//2) & (ac[peaks] > 0)]
    period = float(candidates[0]*32) if len(candidates) else np.nan
    positive = p-np.median(p)
    hills = local_peaks(positive)
    hills = hills[positive[hills] > 0]
    width = float(np.median([half_width(positive, h) for h in hills])*32) if len(hills) else np.nan
    agree = bool(np.isfinite(period) and abs(period-spectrum_period)/spectrum_period <= .2)
    return period, width, agree

def analyze_map(a, rng, repetitions=1000):
    band, axis, angles, sectors, mult, window, radius = geometry(a.shape)
    mask = np.isfinite(a)
    mean = np.nanmean(a)
    c = np.where(mask, a-mean, 0)
    power = abs(fft.rfft2(c*window))**2*mult
    weights = np.einsum('ahw,hw->a', sectors, power)
    rank = np.argsort(weights)[::-1]
    first = float(angles[rank[0]])
    total = power[band].sum()
    strongest = float(weights[rank[0]]/total) if total > 0 else 0.
    concentration = float(power[band & (angular_difference(axis, first) <= 15)].sum()/total) if total > 0 else 0.
    second_candidates = [int(k) for k in rank if 30 <= angular_difference(angles[k], first) <= 150]
    second = float(angles[second_candidates[0]]) if second_candidates else np.nan
    conc2 = float(power[band & (angular_difference(axis, second) <= 15)].sum()/total) if total > 0 else 0.
    def spectral_period(direction):
        select = band & (angular_difference(axis, direction) <= 15)
        if not select.any() or total == 0:
            return np.nan
        best = np.argmax(np.where(select, power, -1))
        return float(1/radius.ravel()[best])
    sp1, sp2 = spectral_period(first), spectral_period(second)
    per1, width, agree1 = projection_metrics(a, first, sp1)
    per2, _, agree2 = projection_metrics(a, second, sp2)
    values = c[mask]
    null = []
    # Vectorized batches: re-use frequency and direction masks for all shuffles.
    for start in range(0, repetitions, 50):
        size = min(50, repetitions-start)
        batch = np.zeros((size, *a.shape), dtype=np.float32)
        batch[:, mask] = rng.permuted(np.broadcast_to(values, (size, len(values))), axis=1)
        bp = abs(fft.rfft2(batch*window, axes=(-2, -1), workers=2))**2*mult
        sums = np.einsum('ahw,bhw->ba', sectors.astype(float), bp, optimize=True)
        totals = bp[:, band].sum(axis=1)
        null.extend(np.divide(sums.max(axis=1), totals, out=np.zeros(size), where=totals>0))
    critical = float(np.percentile(null, 95))
    periodic = strongest > critical
    category = 'その他'
    if periodic and concentration >= .45 and agree1:
        category = '帯状候補'
    elif periodic and concentration >= .20 and conc2 >= .20 and concentration+conc2 >= .60 and agree1 and agree2:
        category = 'うろこ状候補'
    ac, overlap = linear_autocorrelation(a)
    row = dict(分類=category, 周期候補=bool(periodic), 第一帯軸度=first, 第二帯軸度=second,
               第一方向集中度=concentration, 第二方向集中度=conc2, 最大方向パワー割合=strongest,
               並替95百分位=critical, 分類用比較確率=float((1+np.sum(np.array(null)>=strongest))/(repetitions+1)),
               第一スペクトル周期画素=sp1, 第二スペクトル周期画素=sp2, 第一自己相関周期画素=per1,
               第二自己相関周期画素=per2, 帯幅画素=width if agree1 else np.nan,
               第一周期一致=agree1, 第二周期一致=agree2,
               カメラ軸最小角度差=float(min(angular_difference(first, 0), angular_difference(first, 90))))
    return row, ac, overlap, power

def folds(valid):
    labels=((xx//8)+(yy//8))%8;out=[]
    for k in range(8):
        test=(labels==k)&valid
        embargo=cv2.dilate((labels==k).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        out.append((np.flatnonzero(valid&~embargo),np.flatnonzero(test)))
    return out

def cv_score(response,X,valid):
    response=response.ravel().astype(float);X=X.reshape(4096,-1).astype(float)
    sse=0.;base=0.;pred=np.full(4096,np.nan)
    for tr,te in folds(valid):
        if len(te)==0:continue
        if len(tr)<20:return np.nan,pred.reshape(64,64)
        xm=X[tr].mean(axis=0);xs=X[tr].std(axis=0);xs[xs<1e-12]=1
        a=(X[tr]-xm)/xs;b=(X[te]-xm)/xs;ym=response[tr].mean()
        coef=np.linalg.solve(a.T@a+np.eye(a.shape[1]),a.T@(response[tr]-ym))
        pr=ym+b@coef;pred[te]=pr
        sse+=float(np.sum((response[te]-pr)**2));base+=float(np.sum((response[te]-ym)**2))
    return (1-sse/base if base>0 else np.nan),pred.reshape(64,64)

def shifted(a,dy,dx):
    out=np.full_like(a,np.nan)
    out[max(0,dy):min(64,64+dy),max(0,dx):min(64,64+dx)]=a[max(0,-dy):min(64,64-dy),max(0,-dx):min(64,64-dx)]
    return out

def controls(a):return [shifted(a,0,d) for d in (-16,-8,8,16)]+[shifted(a,d,0) for d in (-16,-8,8,16)]+[np.rot90(a)]

def read_image(p):
    a=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
    assert a.ndim==2 and a.shape==(2044,2048)
    return a.astype(np.float32)/65535
