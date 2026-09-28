# Real-data evaluation of v15 spatial-support registration — 2026-09-28

## Scope and method

Applied the v15 adaptive spatial-support ORB/RANSAC estimator followed by the existing subpixel refinement to the same 409-pair production manifest used in the v11 real-data evaluation. The quarter-pitch guard remains unchanged at 1.8215 px. The concentration pipeline, FFT grid construction, sampling, blank threshold, and exact Mann–Whitney test were rerun through the established Phase-2 production reanalysis path. The comparisons below use the saved v11 real baseline as the reference.

High-pass photometric residual and local phase displacement are image-consistency surrogates, not physical ground-truth registration errors. They were compared only for the 399 FOVs accepted by both runs, with unchanged residual definitions and tile sampling.

## Registration and residual results

| Measure | Previous baseline | v15 spatial-support + subpixel |
|---|---:|---:|
| Attempted pairs | 409 | 409 |
| QC accepted | 399 | 401 |
| QC rejected | 10 | 8 |
| Same QC status | — | 407/409 |

The two newly accepted FOVs are `260824_p50_SHC6OH`, sample 8 position 4, and `260825_p50_dna`, sample 4 position 5. No previously accepted FOV became rejected.

On the 399 shared accepted FOVs, final photometric residual decreased in 255, increased in 76, and was unchanged in 68; median paired change was **−0.01342** and mean change **−0.05918**. Local phase median decreased in 191, increased in 140, and was unchanged in 68; its median paired change was **0.000 px**, while the mean change was **−0.31215 px**. Thus the photometric residual shows a clear cohort-level improvement, while phase displacement is mixed with no median shift and a lower mean.

Looking at the improved coarse transform before subpixel refinement, photometric residual decreased in 177, increased in 78, and was unchanged in 144; median change was 0. Local phase median decreased in 180, increased in 123, and was unchanged in 96; median change was also 0. This indicates the v15 coarse change is beneficial for a subset of real FOVs, but does not shift the median real-data surrogate by itself.

## Concentration-dependence result

All seven datasets with a computable two-sided exact Mann–Whitney p-value remain above 0.05. One dataset (`260829_p50_dna`) still has no usable high-concentration FOVs and therefore no computable p-value. The conclusion of no detected concentration dependence is unchanged. The p-values are **not numerically identical** to the previous run, because the new transform changes sampled contrasts; they should not be described as unchanged.

| Dataset | n high / blank | Previous p | v15 p |
|---|---:|---:|---:|
| 260824_p50_SHC6OH | 8 / 6 | 0.6620 | 0.9497 |
| 260825_p50_dna | 7 / 7 | 0.1282 | 0.6200 |
| 260826_p50_sam | 7 / 1 | 1.0000 | 1.0000 |
| 260827_p50_dna | 8 / 7 | 0.5358 | 0.3357 |
| 260828_p50_dna | 8 / 7 | 0.3969 | 0.6126 |
| 260828_p50_sam | 5 / 8 | 0.7242 | 0.7242 |
| 260829_p50_dna | 0 / 5 | not estimable | not estimable |
| 260829_p50_sam | 8 / 7 | 0.5358 | 0.8665 |

## Reproducibility and limits

- Registration implementation: v15 commit `735271e`, tag `v15_registration_initial_affine_spatial_support_20260928`.
- The 409-row source manifest is the attached production manifest under `data/results/v10_phase2_bright_band_mask_20260926/masked_50862c8/source_manifest_attached.csv`.
- Full machine-readable outputs are under `data/results/v16_real_spatial_subpixel_20260928/`.
- The run completed with 409/409 attempted pairs represented and no processing errors. It resulted in 401 QC-accepted and 8 QC-rejected pairs.
- Residual metrics are not independent physical truth. The phase metric's unchanged median and mixed per-FOV changes mean the evidence supports a photometric residual improvement but not a uniform geometric improvement across the full real cohort.
- No code in `PC-alignment-anti` was modified. No changes were made to the stain-mask default, lattice refinement, iterative refinement, or quarter-pitch guard.
