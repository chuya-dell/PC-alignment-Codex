"""Summarize completed read-only audits and register their source provenance."""
import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/results/v64_reproduction_environment_probe'
V63 = ROOT/'data/results/v63_local_correction_reproduction_audit'

def main():
    images = pd.read_csv(V63/'tables/table_input_hash_comparison.csv')
    probes = pd.read_csv(V63/'tables/table_rerun_probe.csv')
    cache = pd.read_csv(V63/'tables/table_input_vs_cache_match.csv')
    threads = pd.read_csv(OUT/'table_thread_probe.csv')
    assert len(images) == 578 and images.bytes_equal.all()
    assert len(probes) == 8 and probes.source_unchanged.all() and probes.v54_delta_identical.all()
    assert probes.ids_identical.all()
    assert len(threads) == 9
    evidence = list(V63.glob('*.json')) + list((V63/'tables').glob('table_input*.csv')) + [V63/'tables/table_rerun_probe.csv', OUT/'thread_probe.json', OUT/'table_thread_probe.csv', OUT/'environment.json']
    summary = dict(gate_passed=False, root_cause_identified=False,
        images=len(images), identical_images=int(images.bytes_equal.sum()),
        diagnostic_fields=4, raw_reruns=len(probes), raw_reruns_match_v54=int(probes.v54_delta_identical.sum()),
        raw_reruns_match_reference_delta=int(probes.delta_identical.sum()),
        raw_reruns_source_unchanged=int(probes.source_unchanged.sum()),
        cache_fields=len(cache), cache_delta_mismatch_fields=int((~cache.delta_identical).sum()),
        cache_xy_exact_mismatch_fields=int((~cache.xy_identical).sum()),
        cache_xy_max_abs=float(cache.xy_max_abs_difference.max()),
        cache_xy_within_fixed_atol=bool((cache.xy_max_abs_difference<=1e-12).all()),
        cache_ids_match_fields=int(cache.ids_identical.sum()),
        thread_probes=len(threads), fixed_holm_family=540,
        local_correction_controls_dummy='not executed: required raw reproduction gate failed',
        limitation='Current copies match each other; equality with images at historical cache-generation time remains unproven. Historical dependency binaries/transforms are not archived in the cache.',
        evidence=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in evidence])
    (OUT/'audit_completion.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    # Append only this audit's rows. Never replace or stage another task's rows.
    input_meta = json.loads((V63/'input_audit.json').read_text(encoding='utf-8'))
    new_rows = []
    for i, s in enumerate(input_meta['sources']):
        if 'snapshot' not in s:
            continue
        new_rows.append(dict(artifact_id=f'v63-reference-audit-20261008-{i}', source_thread_id='unknown',
            source_title='Historical standard reproduction reference', source_updated_at_jst='unknown',
            source_path=s['source'], analysis_level='field_level', artifact_kind='reference_snapshot',
            newness_status='unknown', supersedes_artifact_id='', import_state='snapshot_compared_read_only',
            repository_destination=str(Path(s['snapshot']).relative_to(ROOT)).replace('\\','/'),
            summary=f"Acquired {datetime.now().astimezone().isoformat()}; sha256={s['sha256']}; reference relation only; not evidence of newer version or original cache-generation environment."))
    for repo in (ROOT, V63/'checkout'):
        registry = repo/'data/raw/artifact_provenance_registry.csv'
        original = registry.read_bytes()
        existing = list(csv.DictReader(io.StringIO(original.decode('utf-8-sig'))))
        ids = {r['artifact_id'] for r in existing}
        fields = list(existing[0])
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writerows(r for r in new_rows if r['artifact_id'] not in ids)
        addition = stream.getvalue().encode('utf-8')
        if addition:
            with registry.open('ab') as f:
                if original and not original.endswith(b'\n'): f.write(b'\n')
                f.write(addition)
        assert registry.read_bytes().startswith(original)
    print(json.dumps({k:v for k,v in summary.items() if k!='evidence'},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
