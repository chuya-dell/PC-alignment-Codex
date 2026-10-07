from pathlib import Path
import numpy as np, subprocess
p=Path(r'W:\GoogleDrive\chuya2816\5.解析結果_chu\20260928_digital_judgment_current_alignment')
z=np.load(p/'tables/cached_field_differences/260926_7_5.npz')
print('CACHE',[(k,z[k].shape) for k in z.files])
for f in (p/'scripts_and_config').iterdir(): print('SCRIPT',f.name)
repo=Path(__file__).resolve().parents[2]
out=repo/'data/results/v58_band_scar_causal/source_snapshots'; out.mkdir(exist_ok=True)
for name in ['field_round1_analysis.py']:
    content=subprocess.check_output(['git','show','v28_band_origin_round1_20261004:field_level/v28_band_origin_round1/'+name],cwd=repo)
    (out/name).write_bytes(content)
print('SNAPSHOT',out)
