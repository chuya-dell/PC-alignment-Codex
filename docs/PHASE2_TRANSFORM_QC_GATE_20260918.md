# Phase 2 affine transform QC gate (2026-09-18)

## Decision

`register_image_pair_affine` now validates its estimated affine by default and raises
`AffineTransformQCError` on rejection. A bad image registration must not be silently
converted to a Phase 1 registration: Phase 1 has an independently established landmark
misidentification failure mode. Batch callers should catch this exception, mark the FOV
as `registration_qc_rejected`, and exclude it before summary statistics.

## Thresholds

The gate measures displacement at the image centre and requires all of:

| Metric | Accepted range | Rationale |
|---|---:|---|
| Centre dx/dy | abs(value) <= 125 px | Stricter than former 150 px gross-failure screen; wider than the largest non-degenerate 382-pair displacement (106.159 px). |
| Rotation | abs(value) <= 3 deg | Wider than the non-degenerate maximum 1.907 deg and well below 9--176 deg failures. |
| Isotropic scale | 0.93--1.05 | Covers observed non-degenerate 0.939--1.027; excludes 0.027--0.064 collapses. |
| Linear anisotropy | <= 1.10 | Rejects highly sheared/near-singular affine matrices even when determinant alone is insufficient. |

For the 376 non-degenerate rows in `pair_transform_comparison_with_flags_20260917.csv`,
95th/99th percentiles were abs(dx)=33.803/47.023 px, abs(dy)=28.088/47.376 px,
and abs(rotation)=0.380/0.855 deg. The bounds preserve substantial operational margin.

## Mask coverage diagnostic

The function also emits a `RuntimeWarning` when the supplied/automatic bright-band mask
covers more than 50% of an image. This does not independently reject a transform because
a broad mask can still yield a valid registration, but it records the ORB feature-starvation
risk that caused the observed degeneracies. The transformed-matrix gate remains the safety
boundary. Mask morphology was not changed by this task.

## Validation

Use `field_level/v8_phase2_transform_qc/field_validate_phase2_transform_qc.py` with the
saved comparison CSV. It is a scalar-metric regression test, not a rerun of the 382 image
registrations. The 382-row CSV contains six rows satisfying its stated old numerical screen;
the often-cited eight refers to the separate 394-row downstream recomputation population.

## Real-image entry-point smoke test

`field_smoke_test_phase2_qc.py` calls `register_image_pair_affine` itself and catches
`AffineTransformQCError` as `registration_qc_rejected`, which is the required contract for a
future batch caller.  On 2026-09-18, the two rows present only in the 394-row recomputation
were rerun from their TIFFs: 260824 SHC6OH Sample12-P4 was rejected (centre dx=520.372 px,
dy=-899.182 px, scale=0.036591, anisotropy=1450.882); 260826 SAM Sample10-P4 was rejected
(dx=-356.248 px, dy=-748.382 px, rotation=-25.070 deg, scale=0.050000, anisotropy=3.851).
Two normal controls (260824 SHC6OH Sample1-P1 and 260826 SAM Sample1-P1) returned accepted
matrices without exceptions.  Together with the earlier six-row scalar regression, coverage
is 8/8 known degenerate cases rejected; the two newly located cases have additionally passed
the real entry-point integration test.
