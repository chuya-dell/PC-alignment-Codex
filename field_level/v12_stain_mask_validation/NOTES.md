# 記録済み欠陥マスク検証

260922の一次記録で欠陥が明記された視野を、現行のオプトイン`stain_artifact_mask`で可視化する。
50倍・100倍のpre/postを含む。ログは画素範囲を示さないため、画素レベルの性能スコアは出さない。
加えて260829 SAM pos6 axis14で候補とFFT格子中心の重なりを数え、マスク悪化原因の候補を調べる。

実行例:

```powershell
& .\.venv\Scripts\python.exe field_level\v12_stain_mask_validation\field_validate_recorded_defects.py `
  --data-root 'W:\4.生データD_remo' `
  --output data\results\v13_stain_mask_validation_20260927
```

結果と制約は`docs/STAIN_MASK_RECORDED_DEFECT_VALIDATION_20260927.md`に記録。候補は診断専用であり、
マスク既定動作も研究解析も変更しない。格子参照型の本番実装は未着手。
