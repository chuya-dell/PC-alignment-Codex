# Registration precision, staged experiment

## Implementation
Base: fadcfda. The legacy registration and production implementations remain unchanged.
This version evaluates the production entry point with known pre-to-post transforms.
The old entry-point docstring says post-to-pre, but its feature coordinates and production
forward sampling implement pre-to-post. Synthetic rendering uses the forward transform.
Stage order: baseline, subpixel, lattice, iterative. Each implementation and measurement
is committed separately. Anti code and its frozen failure investigation are out of scope.

## Results
Results go to data/results/v11_registration_precision_20260926, with identical cases per stage.
The 452-case known-truth baseline has 0.0893 px median and 0.3642 px 95th percentile
spatial RMSE (maximum 6.8291 px). Subpixel refinement reduced these to 0.0040 and
0.0159 px, respectively; 420/452 cases improved and none worsened. Large coarse errors
persist unchanged because the local refinement rejects moves exceeding a quarter pitch.
The 409-pair real-data baseline rerun gave the same 399 accepted / 10 QC-rejected pairs and
the same eight dataset-level Mann–Whitney outcomes as the preceding masked reanalysis.
The fixed-pitch FFT lattice stage improved 180 cases, worsened 204, and left 68 unchanged
against subpixel refinement. Its median and 95th-percentile spatial errors were higher.
Do not promote the grid stage solely because its held-out photometric objective decreases.
Inputs are read only from the mounted data root (currently W:/4.生データD_remo).

## Known limitations
Semi-synthetic truth measures resampling accuracy, not physical truth in real pre/post pairs.
Real-image photometric residual is a surrogate and must be reported separately.
Failure rates and rejected cases are retained. No significance result proves absence of an effect.
