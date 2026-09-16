# v6 Position 6 image forensics

Direct image-content investigation of 260826 p50 SAM Sample 1 Position 6 vs. Positions 1, 2, 5, 8
(no registration is performed here). Reuses `shared.analyzer` for sub-pixel peak detection,
`shared.lattice_indexing` for the FFT-fit theoretical lattice and per-tile detection/missing rate,
and `shared.defect_masking.residual_candidate_regions` for automatic (unapproved) defect
candidates. Adds native-16-bit brightness/saturation statistics and Laplacian/Tenengrad focus
metrics local to this script (small enough not to warrant a shared module).

Found that Position 6 (and the adjacent Position 7) contains a bright, partially saturated
crossing-line structure near the frame edges, reproduced at nearly the same pixel location on a
different chip imaged two days later (260828-p50-SAM). This is not present at Positions 1, 2, 5, 8
(or is present only as a much fainter, non-saturating version). See
`docs/POSITION6_IMAGE_FORENSICS_20260916.md` for the full write-up and interpretation.

Known limitation: the automatic defect-candidate detector's "whole-frame" region (a low-frequency
shading residual that always exceeds the z-score threshold once morphologically closed) and its
thin edge-hugging strips are not real defects; they were filtered out by hand when comparing
candidates across positions in the doc, not in this script's output. A human should still review
`defect_candidates_*.csv` before treating any bounding box as an approved defect, per
`shared/defect_masking.py`'s own review-only contract.
