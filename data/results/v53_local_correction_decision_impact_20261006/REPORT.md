# v53: Local correction inserted into the decision process

確認: 未確認

## Outcome

The requested comparison was not run.  The mandatory reproduction of the
standard digital-judgment pipeline could not be performed in this isolated
worktree, so no claim about a local correction's effect on positive-pillar
counts or concentration-dependence tests is made.

## Fact

- The required handover notes, repository conventions, and local-environment
  note were read after enumerating the specified handover directory.
- Before choosing this version, the Google Drive results root was enumerated.
  It already contained v48 through v52; v53 is the smallest unused version.
- The specified source script exists at
  `W:/GoogleDrive/chuya2816/5.解析結果_chu/20260928_digital_judgment_current_alignment/scripts_and_config/field_run_digital_judgment.py`.
- That script calls `multipletests(...)` but does not import it.  The allowed
  derived repair is therefore an import of
  `statsmodels.stats.multitest.multipletests` only.
- The worktree does not contain `.venv/Scripts/python.exe`.  The available
  Python is 3.13, and attempting that required import raised
  `ModuleNotFoundError: No module named 'statsmodels'`.
- The four requested DNA raw-data directories were enumerated under
  `W:/GoogleDrive/chuya2816/5.生データD_chu/`; missing raw data is not the
  present blocker.
- The original script also embeds a different machine's repository path and a
  non-existent remote-data path.  These could be redirected only in a derived
  copy, but doing so cannot overcome the missing required package.

## Interpretation

Because the standard run has not been reproduced, a correction-versus-standard
comparison would not have a verified baseline.  Proceeding would violate the
fixed prerequisite in the request.  No inference is made about registration
accuracy, measurement precision, or assay performance.

The proposed position-only correction remains technically distinct from a
pillar-correspondence holdout: it can be fitted from v24 independent-centre
residual vectors and evaluated at FFT-grid coordinates without a pillar ID
join.  If subsequently run, its report must explicitly retain the limitation
that its split is not mediated by a pillar correspondence.

## Confidence

High for the environment and import failure: both the absence of `.venv` and
the Python import failure were directly checked.  High for the source-script
missing import: the inspected source contains the call and no corresponding
import.  No confidence statement is possible for correction effects because
they were not computed.

## Standard configuration scope retained for a future run

The planned comparison is restricted to 260825, 260827, 260828, and 260829
DNA fields.  The standard configuration's default-disabled stain mask and
bright-band mask would remain disabled.  It would retain the same threshold
and statistical procedure (field-level Spearman and one-sided
Mann--Whitney-U with global Holm correction), and it would pass sampling
through the existing integer-rounded photometry behavior.  These are planned
conditions only, not completed results.

## Band arrangement

Unmeasured.  Existing v26/v27 band-related artifacts were found, but no
standard run was reproduced and no new band metric was introduced.

## Needed input for the next run

Provide the project Python environment with `statsmodels` installed (or an
approved, immutable environment path containing it).  Then an import-only,
path-localized derived copy of the standard script can be run and compared
byte-for-byte/numerically with the accepted output before any correction work.

## Files written

| Path | Content |
|---|---|
| `field_level/v53_local_correction_decision_impact/NOTES.md` | Version notes and stopping condition |
| `data/results/v53_local_correction_decision_impact_20261006/STATUS.md` | Concise stopped status |
| `data/results/v53_local_correction_decision_impact_20261006/REPORT.md` | Fact / interpretation / confidence report |
| `data/results/v53_local_correction_decision_impact_20261006/tables/table_standard_reproduction_gate.csv` | Reproduction gate evidence |
