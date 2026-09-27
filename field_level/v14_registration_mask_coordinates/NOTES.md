# Pre/post stain-mask coordinate alignment

The opt-in registration path detects stain candidates in each native image frame,
maps post candidates into pre coordinates using the initial pre-to-post affine,
unions there, maps the union back to post coordinates, and refits with separate
per-image exclusion masks. Up to three masked refits are allowed. The default
path with mask_stains=False remains unchanged.

Validation:

- field_verify_mask_coordinates.py checks newly attached post-only stains and
  pre-only stains removed by washing under known translation.
- field_level/v11_registration_artifact_mask/field_verify_artifact_mask.py checks
  scratch protection and unchanged default behavior.
- field_visualize_pair_mask_alignment.py creates pair overlays for all four
  logged 260922 defect cases, at 50x and 100x.
- field_level/v10_registration_precision/field_run_precision_parallel.py ran
  452 paired known-transform cases for artifact_mask and artifact_subpixel.

The saved 452-row outputs are under data/results/v14_registration_mask_coordinates_20260927.
The quantitative report is docs/REGISTRATION_MASK_COORDINATES_RESULT_20260927.md.
Photometric sampling exclusions are deliberately not implemented in this stage.
