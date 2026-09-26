# Opt-in subpixel registration

## Implementation
Production affine initialization is unchanged. Refine detected corners on float intensities,
track local displacements with bidirectional Lucas–Kanade correspondence, and robustly refit
the full affine on training correspondences. A held-out median correspondence error must
improve. Require spatial support and cap the change everywhere at one quarter pitch.
Both native-coordinate bright-band masks are dilated to protect patch footprints.
The coarse and final transforms must pass the existing physical QC gate.

## Results
Staged results are stored under data/results/v11_registration_precision_20260926.

## Known issues
Local tracking cannot repair a wrong coarse lattice cell. Coarse failures remain failures.
The existing groove detector already has parabolic subpixel interpolation; it is not the
production initializer and is intentionally unchanged. Low-texture/no-support cases keep
the coarse estimate with a diagnostic reason. Real residuals have no known geometric truth.
