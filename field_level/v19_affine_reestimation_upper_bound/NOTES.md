# v19 affine re-estimation diagnostic

This version adds a diagnostic program only. It does not alter the standard registration path, defaults, or the integer-coordinate sampler.

`affine_reestimation_diagnostic.py` reuses the independent center detector from v17. For each of the 452 semi-synthetic cases, it makes unique nearest-neighbor correspondences under the current transform, keeps matches within half the nominal 7.286 px pitch, and refits an affine transform with Huber iteratively reweighted least squares. It rematches and refits for at most 10 iterations. Convergence requires unchanged ordered one-to-one assignments and a maximum transformed-frame change below 1e-4 px. The robust fit is capped at 30 iterations and uses a 1.345 Huber cutoff with a 0.12 px scale floor.

For spatial validation, each image is divided into a 4 by 4 grid. A fit is estimated from correspondences outside each block and evaluated on the held-out block. The benchmark result is diagnostic: the same-cell associations are seeded by the current transform, and the held-out residual is not a known-truth error. The decisive comparison is the absolute error against the known semi-synthetic transform.

The 452-condition semi-synthetic comparison was completed. Its result does not pass the prerequisite for treating a real-data refit as an upper bound: the center-refit has a higher median absolute error than the current standard. Real-data refitting was therefore not run in this version.
