"""Locate high-error synthetic fits and high-residual real FOVs with overlays."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared import registration as reg
from shared.image_qc import bright_band_mask, stain_artifact_mask
from shared.lattice_indexing import lattice_from_fft
from field_level.v10_registration_precision.field_precision_benchmark import errors


def matrix_from_row(row):
    return np.array([[row[f"m{i}{j}"] for j in range(3)] for i in range(2)], dtype=np.float32)


def orb_diagnostics(pre, post):
    """Re-run the production ORB/RANSAC recipe and retain explicit match metadata."""
    cv2.setRNGSeed(20260926)
    a = np.uint8(np.clip(reg.image01_for_registration(pre) * 255, 0, 255))
    b = np.uint8(np.clip(reg.image01_for_registration(post) * 255, 0, 255))
    valid = (~bright_band_mask(pre)).astype(np.uint8) * 255
    detector = cv2.ORB_create(nfeatures=12000, fastThreshold=3)
    ka, da = detector.detectAndCompute(a, valid)
    kb, db = detector.detectAndCompute(b, valid)
    if da is None or db is None:
        return None, None, {"features_pre": len(ka), "features_post": len(kb), "matches": 0,
                            "inliers": 0, "inlier_ratio": 0.0}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < .72*n.distance]
    if len(good) < 8:
        return ka, good, {"features_pre": len(ka), "features_post": len(kb), "matches": len(good),
                          "inliers": 0, "inlier_ratio": 0.0}
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    _, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5,
                                     maxIters=4000, confidence=.995)
    inlier_bits = np.zeros(len(good), dtype=bool) if inliers is None else inliers.ravel().astype(bool)
    return (ka, kb, good, inlier_bits), (src, dst), {
        "features_pre": len(ka), "features_post": len(kb), "matches": len(good),
        "inliers": int(inlier_bits.sum()),
        "inlier_ratio": float(inlier_bits.mean()) if len(good) else 0.0,
    }


def case_error_vector(estimate, truth, shape):
    center = np.array([(shape[1]-1)/2, (shape[0]-1)/2], dtype=float)
    return (estimate[:, :2]-truth[:, :2]) @ center + estimate[:, 2]-truth[:, 2]


def synthetic_diagnostics(precision_root, output, data_root, min_rmse=1.0):
    base = pd.read_csv(precision_root/'baseline.csv')
    sub = pd.read_csv(precision_root/'subpixel.csv').set_index('case_id')
    ledger = {item['case_id']: item for item in json.loads((precision_root/'baseline_inputs.json').read_text(encoding='utf-8'))}
    selected = base[(base.status == 'ok') & (base.grid_rmse_px >= min_rmse)].sort_values('grid_rmse_px', ascending=False)
    summary = []
    for _, row in selected.iterrows():
        item = ledger[row.case_id]
        path = Path(item['path'])
        if not path.exists():
            relative = str(item['path']).split('4.生データD_remo\\', 1)[-1].replace('\\', '/')
            path = data_root / relative
        pre = reg.load_image_unicode(str(path))
        truth = np.asarray(item['truth'], np.float32)
        post = cv2.warpAffine(pre, truth, (pre.shape[1], pre.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
        estimate = matrix_from_row(row)
        match_data, match_points, md = orb_diagnostics(pre, post)
        if match_data is not None:
            inliers = match_data[3]
            inlier_src = match_points[0][inliers]
            span = np.ptp(inlier_src, axis=0) if len(inlier_src) else np.zeros(2)
            md.update({"inlier_span_x_fraction": float(span[0]/pre.shape[1]),
                       "inlier_span_y_fraction": float(span[1]/pre.shape[0]),
                       "inlier_bbox_fraction": float(span[0]*span[1]/(pre.shape[0]*pre.shape[1]))})
        else:
            md.update({"inlier_span_x_fraction": 0.0,"inlier_span_y_fraction": 0.0,
                       "inlier_bbox_fraction": 0.0})
        lattice = lattice_from_fft(reg.image01_for_registration(pre), 7.286)
        delta = case_error_vector(estimate, truth, pre.shape)
        coeff = np.linalg.solve(lattice.basis, delta)
        cell = np.rint(coeff)
        nearest = lattice.basis @ cell
        sr = sub.loc[row.case_id]
        sub_diag = json.loads(sr.diagnostics).get('subpixel', {})
        summary.append({"case_id": row.case_id, "source": row.source, "position": int(row.position),
                        "scenario": row.scenario, "baseline_rmse_px": float(row.grid_rmse_px),
                        "subpixel_rmse_px": float(sr.grid_rmse_px),
                        "baseline_center_error_px": float(row.center_error_px),
                        "subpixel_accepted": bool(sub_diag.get('accepted', False)),
                        "subpixel_reason": sub_diag.get('reason', ''),
                        "rotation_edge_error_px": float(row.rotation_edge_error_px),
                        "scale_edge_error_px": float(row.scale_edge_error_px),
                        **md, "center_delta_x_px": float(delta[0]), "center_delta_y_px": float(delta[1]),
                        "nearest_hex_i": int(cell[0]), "nearest_hex_j": int(cell[1]),
                        "nearest_hex_residual_px": float(np.linalg.norm(delta-nearest)),
                        "nearest_hex_shift_px": float(np.linalg.norm(nearest)),
                        "fft_angle_deg": float(np.rad2deg(lattice.angle_rad))})
        # Full-frame alignment overlay plus verified RANSAC correspondences.
        aligned = cv2.warpAffine(post, estimate, (pre.shape[1], pre.shape[0]),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT_101)
        pre_view = cv2.normalize(pre, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        align_view = cv2.normalize(aligned, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        overlay = cv2.addWeighted(pre_view, .5, align_view, .5, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        for y in np.linspace(80, pre.shape[0]-80, 6).astype(int):
            for x in np.linspace(80, pre.shape[1]-80, 6).astype(int):
                v = (estimate[:, :2]-truth[:, :2]) @ np.array([x, y], float) + estimate[:, 2]-truth[:, 2]
                cv2.arrowedLine(overlay, (int(x), int(y)), (int(x+v[0]*8), int(y+v[1]*8)),
                                (0, 0, 255), 1, tipLength=.25)
        match_img = np.zeros((pre.shape[0], pre.shape[1]*2, 3), np.uint8)
        if match_data is not None:
            ka, kb, good, inlier_bits = match_data
            indexed = list(range(len(good)))
            if len(indexed) > 120:
                rng = np.random.default_rng(260927)
                chosen_in = np.flatnonzero(inlier_bits)
                chosen_out = np.flatnonzero(~inlier_bits)
                chosen = np.r_[chosen_in[:60], rng.choice(chosen_out, min(60, len(chosen_out)), replace=False)]
                indexed = chosen.tolist()
            match_img = cv2.drawMatches(pre_view, ka, cv2.cvtColor(cv2.normalize(post,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),cv2.COLOR_GRAY2BGR), kb,
                                        [good[i] for i in indexed], None,
                                        matchesMask=[int(inlier_bits[i]) for i in indexed],
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            match_img = cv2.resize(match_img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        output.mkdir(parents=True, exist_ok=True)
        safe = row.case_id.replace('/', '_').replace('\\', '_')
        overlay = cv2.resize(overlay, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output/f"synthetic_{safe}_overlay.png"), overlay)
        cv2.imwrite(str(output/f"synthetic_{safe}_matches.jpg"), match_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    pd.DataFrame(summary).to_csv(output/'synthetic_outliers.csv', index=False)


def real_diagnostics(real_root, output, data_root, top_each=8):
    residuals = pd.read_csv(real_root/'real_residuals.csv')
    manifest = pd.concat([pd.read_csv(p) for p in real_root.glob('manifest_*.csv')], ignore_index=True)
    manifest = manifest.drop_duplicates(['dataset','sample','position'])
    lookup = {f"{r.dataset}_{r['sample']}_{r.position}": r for _,r in manifest.iterrows()}
    chosen = pd.concat([residuals.nlargest(top_each,'baseline_phase_median_px'),
                        residuals.nlargest(top_each,'baseline_photometric_residual')]).drop_duplicates('fov_key')
    rows=[]; output.mkdir(parents=True, exist_ok=True)
    for _, result in chosen.iterrows():
        row=lookup[result.fov_key]
        pre=reg.load_image_unicode(str(row.pre_path));post=reg.load_image_unicode(str(row.post_path))
        diag=json.loads((real_root/'diagnostics'/f"{result.fov_key}.json").read_text(encoding='utf-8'))
        matrix=np.asarray(diag['matrix'],np.float32)
        aligned=cv2.warpAffine(post,matrix,(pre.shape[1],pre.shape[0]),
                               flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT_101)
        hp_pre=pre-cv2.GaussianBlur(pre,(0,0),3);hp_post=aligned-cv2.GaussianBlur(aligned,(0,0),3)
        difference=np.abs(hp_pre-hp_post)
        mask=stain_artifact_mask(pre)
        rows.append({"fov_key":result.fov_key,"baseline_phase_median_px":result.baseline_phase_median_px,
                     "baseline_photometric_residual":result.baseline_photometric_residual,
                     "highpass_correlation":1-result.baseline_photometric_residual,
                     "stain_mask_fraction":float(mask.mean()),"pre_path":row.pre_path,"post_path":row.post_path})
        h,w=pre.shape;size=(640,640)
        views=[pre,aligned,difference]
        titles=['pre raw','post warped by saved production matrix','absolute high-pass residual']
        canvas=np.zeros((size[1]+35,size[0]*3,3),np.uint8)
        for i,(im,title) in enumerate(zip(views,titles)):
            scale=min(size[0]/w,size[1]/h);resized=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            view=cv2.normalize(resized,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            if i==0:
                m=cv2.resize(mask.astype(np.uint8),None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)>0
                color=cv2.cvtColor(view,cv2.COLOR_GRAY2BGR);color[m]=(.55*color[m]+.45*np.array([0,0,255])).astype(np.uint8)
            else: color=cv2.cvtColor(view,cv2.COLOR_GRAY2BGR)
            x=i*size[0];canvas[35:35+view.shape[0],x:x+view.shape[1]]=color
            cv2.putText(canvas,title,(x+8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        safe=result.fov_key.replace('/','_').replace('\\','_')
        cv2.imwrite(str(output/f"real_{safe}_residual.jpg"),canvas,[cv2.IMWRITE_JPEG_QUALITY,90])
    pd.DataFrame(rows).to_csv(output/'real_outlier_candidates.csv',index=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--precision-root',type=Path,required=True)
    p.add_argument('--real-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-rmse',type=float,default=1.0)
    p.add_argument('--top-real-each',type=int,default=4)
    a=p.parse_args()
    synthetic_diagnostics(a.precision_root,a.output,a.data_root,a.min_rmse)
    real_diagnostics(a.real_root,a.output,a.data_root,a.top_real_each)

if __name__=='__main__': main()
