# Validation record: 2026-09-11

- Syntax: `ast.parse` completed successfully for `pillar_compare_delta_i_over_i_tails.py`.
- Runtime environment: system Python 3.13 with OpenCV 4.13.0. The bundled workspace Python did not provide `cv2`, so it was not used for image processing.
- ECC safety check: the script sets `cv2.setNumThreads(1)`. This retains the requested coarse-template initialized 0.25x→0.5x→1.0x affine ECC and prevents the host's intermittent OpenCV process termination during full-resolution calls.
- Partial input check before the source drive disconnected: all 54 SAM clean FOVs were cached; DNA cache creation had started. The generated cache, TIFFs, and any eventual figures are intentionally outside Git.
- Blocking condition: `G:` was no longer mounted (`net use G:` returned no network connection), so no final distribution CSV, figures, or report is committed from incomplete input coverage.
- Reproduction: reconnect the two source roots listed in `NOTES.md`, then run the script with the arguments documented there. Existing cache files are reused and non-image final outputs are regenerated only after all selected FOVs complete.

Newness status is `unknown`; no relationship to prior analyses has been inferred from timestamps.
