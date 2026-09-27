# Initial affine spatial-support diagnosis and benchmark — 2026-09-28

## Finding

The Phase-2 image-pair entry point `register_image_pair_affine` does not use the cross-scratch detector as its initial affine estimator. It uses ORB keypoints, one-way Lowe-ratio descriptor matching, and affine RANSAC. `detect_grooves` is used by the separate detected-pillar coordinate alignment path (`align_and_match_dataframes`). The 452-case semi-synthetic benchmark in this report exercises the former path.

The large initial errors are chiefly a geometric-support problem in the RANSAC consensus. The worst case, `260825_p50_dna_1-6_axis_14`, had 11,183 pre and 11,685 post ORB features, 1,705 ratio-filtered matches, and 1,274 RANSAC inliers (74.7%). Those inliers covered about 1,937 px horizontally but only 192 px vertically in a 2,048 px field. A high inlier count and fraction therefore concealed a nearly one-dimensional fit that extrapolated poorly across the full image. Other large errors show the same pattern: `260829-p50-sam_1-6_axis_15` had 3,096 inliers but only 343 px vertical coverage; `260829-p50-sam_1-6_axis_08` had 3,972 inliers but only 224 px vertical coverage. This points to ambiguous or spatially clustered correspondences on the periodic image texture and an under-supported affine fit, rather than too few detected features.

The `260829-p50-sam_1-6_axis_14` case had broad horizontal but partial vertical support (about 1,947 × 932 px) and a 1.006 px coarse RMSE. Its correction candidate also benefited from the alternative-fit search, but its support pattern is less severely degenerate than the maximum-error case.

## Change evaluated

The normal ORB/RANSAC estimate is preserved when its RANSAC inlier bounding box covers at least 70% of the image area. Below that support level, the estimator also tries two deterministic alternatives: Lowe ratio 0.80 with a 2.5 px RANSAC threshold, and the configured Lowe ratio with a 4.0 px threshold. It selects the candidate with the largest inlier bounding-box area multiplied by the inlier fraction, and only adopts an alternative when it scores above the primary fit. ORB feature extraction and descriptor matching are shared across candidates.

This keeps the existing transform on 425/452 conditions. In the 27 low-support conditions that entered candidate search, 25 improved and 2 worsened slightly (by 0.034 and 0.049 px; `260825_p50_dna_1-1_axis_15` and `260825_p50_dna_1-7_axis_05`); the other 425 were unchanged. The two-axis coverage threshold and candidate scoring are empirical guardrails calibrated on this benchmark and should be monitored on new image populations.

## 452-condition comparison

All stages used the same 452 known transforms, source-image hashes, image renderer, and view-grid RMSE definition as v11. The current baseline was rerun through the current standard entry point before comparison. The candidate implementation's default hypothesis reproduced that rerun within 0.0001 px for all 452 cases.

| Stage | RMSE median | RMSE 95th percentile | RMSE max | RMSE ≥0.5 px | RMSE ≥1 px |
|---|---:|---:|---:|---:|---:|
| Existing coarse ORB/RANSAC | 0.08935 | 0.36425 | 6.82909 | 13 | 5 |
| Spatial-support coarse ORB/RANSAC | 0.08805 | 0.30424 | 0.54116 | 2 | 0 |
| Existing coarse + subpixel refinement | 0.004013 | 0.015886 | 6.82909 | 5 | 4 |
| Spatial-support coarse + same subpixel refinement | 0.003970 | 0.014505 | 0.06411 | 0 | 0 |

The 2 residual coarse cases above 0.5 px are `260825_p50_dna_1-6_axis_15` (0.53579 px) and `260829-p50-sam_1-6_axis_14` (0.54116 px). After subpixel refinement, neither remains an outlier. The new end-to-end subpixel path improved 9 cases, worsened 4 by at most 0.00128 px, and left 439 unchanged relative to the previously recorded subpixel stage (using a 0.0001 px tie tolerance).

Three representative subpixel RMSEs changed as follows:

| Case | Existing subpixel | Spatial-support coarse + subpixel |
|---|---:|---:|
| 260825 DNA pos6 axis14 | 6.82909 px | 0.02584 px |
| 260829 SAM pos6 axis14 | 0.03035 px | 0.01117 px |
| 260829 SAM pos6 axis15 | 2.84353 px | 0.01109 px |

## Cell guard

The subpixel LK tracking and cumulative-update guard remain at `pitch/4 = 1.8215 px`. With the better initial affine, the existing guard allowed all 452 cases to complete, with a maximum final RMSE of 0.06411 px and no case at or above 0.5 px. The prior `3.227 px` rejected correction was an artifact of its inaccurate coarse estimate; it no longer requires relaxing the same-cell safeguard. No guard threshold was changed.

## Validation and limits

- The current standard coarse estimator completed 452/452 cases with zero failures.
- The spatial-search-plus-subpixel benchmark completed 452/452 cases with zero failures.
- `python -m unittest discover -s shared/tests -v`: 9 tests passed, including new tests for row-concentrated support fallback and preserving well-supported primary fits.
- `py_compile` and `git diff --check` passed.
- `PC-alignment-anti` was not modified. Stain-mask defaults, lattice refinement, iterative refinement, and photometric sampling are outside this change.
- The benchmark's support score uses an axis-aligned inlier bounding box. It is useful for identifying this failure mode, but it is not a universal quality measure; new real-image cohorts should retain residual QC and review rejected/low-support cases.

Reproducibility files are under `data/results/v15_initial_affine_spatial_support_20260928/`; source diagnostic and benchmark scripts are under `field_level/v15_initial_affine_diagnostics/`.
