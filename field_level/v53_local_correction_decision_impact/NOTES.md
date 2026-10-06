# v53: local-correction decision-impact (stopped before analysis)

## Implementation change

No decision-impact implementation was added.  This version records the required
standard-judgment reproduction gate and its failure before any local correction,
dummy correction, or decision comparison was run.

## Result and storage

The gate failure is documented in
`data/results/v53_local_correction_decision_impact_20261006/`.  No measurement
tables or figures were generated.

## Known issue

The local worktree has no `.venv`, and its available Python 3.13 environment
does not contain `statsmodels`.  The supplied standard script calls
`multipletests` without importing it; its permitted import-only derived copy
would require `from statsmodels.stats.multitest import multipletests`, which
cannot be imported in this environment.  Therefore the standard output cannot
be reproduced here and the requested comparison must not proceed.
