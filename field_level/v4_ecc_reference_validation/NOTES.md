# v4 ECC reference validation

`field_validate_ecc_against_20260901.py` reuses the versioned Phase-2 affine
ECC code path and evaluates it against the seven non-Position-6 values in the
2026-09-01 ledger. It must pass the recorded transform comparison before a
Position-6 deep-pyramid or multi-start rescue is attempted.

2026-09-16実行では全項目一致0/7となり、Position 6の救済探索は実行しなかった。
詳細は `docs/ECC_20260901_REFERENCE_VALIDATION_20260916.md` を参照。
