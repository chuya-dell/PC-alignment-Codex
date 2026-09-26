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
Inputs are read only from the mounted data root (currently W:/4.生データD_remo).

## Known limitations
Semi-synthetic truth measures resampling accuracy, not physical truth in real pre/post pairs.
Real-image photometric residual is a surrogate and must be reported separately.
Failure rates and rejected cases are retained. No significance result proves absence of an effect.
