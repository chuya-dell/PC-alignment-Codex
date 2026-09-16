# v7 masked alignment benchmark

Re-runs the v5 real-range/axis-sweep scenarios (translation-dominant realrange, and isolated
ty/theta_deg/delta_scale_pct sweeps) through the promoted Phase 2 functions in
`shared.registration` (`estimate_affine_ecc`, `estimate_affine_orb_ransac`), each with and
without the Position 6/7 write-field-boundary mask from `shared.image_qc.bright_band_mask`.

Key result: masking fixes ORB+RANSAC's Position 6 rotation weakness (FP 98.89% -> 0.00%) and a
minor scale-related FP issue, without breaking anything ORB already did well. Masking does NOT
fix ECC's Position 6 large-translation local-minimum trap (recovery stays ~26-34%) -- that is a
capture-range/periodicity problem, not a masking problem. See
`docs/MASKED_ALIGNMENT_FINAL_DECISION_20260916.md` for the full comparison and the resulting
decision (ORB+RANSAC affine, masked, is now `shared.registration.PHASE2_DEFAULT_METHOD`).

Pitfall hit and fixed during this work: the first version of `estimate_affine_ecc` masked by
replacing excluded pixels with a constant (median) value in both images before correlating.
That introduced a large, sharp, artificial rectangular edge at the mask boundary that ECC would
latch onto -- far worse than no masking at all (ty-axis error at Position 6 went from 0.003 px
unmasked to 467 px with value-flattening). The fix was to pass the mask through
`cv2.findTransformECC`'s own `inputMask` parameter instead, with no pixel-value change.
