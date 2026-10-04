param([ValidateSet('start','finish')][string]$Phase = 'start')
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$out = Join-Path $repo 'data/results/v36_band_origin_final'
if (-not (Test-Path -LiteralPath $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
$sources = @()
foreach ($folder in @('data/inputs_local/lab_notes_final','data/inputs_local/lab_notes')) {
  $sources += Get-ChildItem -LiteralPath (Join-Path $repo $folder) -File
}
$roundFolders = Get-ChildItem -LiteralPath (Join-Path $repo 'data/results') -Directory | Where-Object Name -Match '^v(28|29|30|31|32|33|34|35)_band_origin_round[1-8]$'
foreach ($folder in $roundFolders) {
  $sources += Get-ChildItem -LiteralPath $folder.FullName -File | Where-Object Extension -In '.md','.csv','.json'
  foreach ($resume in (Get-ChildItem -LiteralPath $folder.FullName -Directory | Where-Object Name -In 'resume_20261004','resume_20261004_final')) {
    $sources += Get-ChildItem -LiteralPath $resume.FullName -File | Where-Object Extension -In '.md','.csv','.json'
  }
}
$sources = @($sources | Sort-Object FullName -Unique)
$oldTables = Join-Path $repo 'data/inputs_local/20260928_digital_judgment_current_alignment/20261002_外れ値視野_追加検証_v27/tables'
$sources += Get-ChildItem -LiteralPath $oldTables -File | Where-Object Name -In '相関と並替検定要約.csv','縁除外幅別_相関.csv'
$sources += Get-Item -LiteralPath (Join-Path $repo 'data/results/v33_band_origin_round6/figures/260926_7_5_並べた図.png')
$sources += Get-Item -LiteralPath (Join-Path $repo 'data/results/v35_band_origin_round8/figures/260926_7_5_null_comparison.png')
$sources = @($sources | Sort-Object FullName -Unique)
$records = foreach ($source in $sources) {
  # Read existing text and fingerprint it; no image analysis, model fitting or tests.
  if ($source.Extension -ne '.png') { $null = [System.IO.File]::ReadAllText($source.FullName, [System.Text.Encoding]::UTF8) }
  [pscustomobject]@{ path=$source.FullName; bytes=$source.Length; sha256=(Get-FileHash -LiteralPath $source.FullName -Algorithm SHA256).Hash }
}
if ($Phase -eq 'start') {
  $records | Export-Csv -LiteralPath (Join-Path $out 'source_read_registry.csv') -NoTypeInformation -Encoding UTF8
  Write-Output ('Read and fingerprinted sources: ' + $records.Count)
} else {
  $initial = Import-Csv -LiteralPath (Join-Path $out 'source_read_registry.csv') -Encoding UTF8
  $lookup = @{}
  foreach ($row in $records) { $lookup[$row.path] = $row }
  $checked = foreach ($row in $initial) {
    [pscustomobject]@{ path=$row.path; start_sha256=$row.sha256; end_sha256=$lookup[$row.path].sha256; unchanged=($row.sha256 -eq $lookup[$row.path].sha256) }
  }
  $checked | Export-Csv -LiteralPath (Join-Path $out 'source_integrity_final.csv') -NoTypeInformation -Encoding UTF8
  if (@($checked | Where-Object { -not $_.unchanged }).Count) { throw 'A referenced source changed.' }
  $report = Join-Path $out '261004_最終報告書.md'
  $body = [System.IO.File]::ReadAllText($report, [System.Text.Encoding]::UTF8)
  $links = [regex]::Matches($body, '\]\(([^)]+)\)')
  foreach ($link in $links) {
    $target = $link.Groups[1].Value.Split('#')[0]
    if ($target -and $target -notmatch '^https?://') {
      if ($target -ne 'final_verification.json' -and -not (Test-Path -LiteralPath (Join-Path $out $target))) { throw ('Missing report link: ' + $target) }
    }
  }
  $numericChecks = @(
    @{file='v35_band_origin_round8/resume_20261004_final/signed_field_metrics.csv'; selector=@{key='260926_7_5';条件='主帰無'};column='相関';printed='0.49711';digits=5;scale=1},
    @{file='v35_band_origin_round8/resume_20261004_final/signed_field_metrics.csv'; selector=@{key='260926_7_5';条件='主帰無'};column='分散比';printed='0.25416';digits=5;scale=1},
    @{file='v35_band_origin_round8/resume_20261004_final/signed_field_metrics.csv'; selector=@{key='260926_7_5';条件='主帰無'};column='無調整分散再現';printed='0.24707';digits=5;scale=1},
    @{file='v35_band_origin_round8/resume_20261004_final/signed_field_metrics.csv'; selector=@{key='260926_7_5';条件='主帰無'};column='実測周期';printed='212.08';digits=2;scale=1},
    @{file='v35_band_origin_round8/resume_20261004_final/signed_field_metrics.csv'; selector=@{key='260926_7_5';条件='主帰無'};column='模擬周期';printed='234.80';digits=2;scale=1},
    @{file='v35_band_origin_round8/resume_20261004_final/outlier_field_metrics.csv'; selector=@{key='260926_7_5';条件='主帰無';定義='平均標準偏差'};column='模擬外れ値割合';printed='0.4662';digits=4;scale=100},
    @{file='v35_band_origin_round8/resume_20261004_final/outlier_field_metrics.csv'; selector=@{key='260926_7_5';条件='主帰無';定義='平均標準偏差'};column='実測外れ値割合';printed='5.0471';digits=4;scale=100},
    @{file='v32_band_origin_round5/resume_20261004/round5_J_reference_excluded_tests.csv'; selector=@{指標='外れ値割合';定義='平均標準偏差'};column='加算効果80パーセント検出近似';printed='0.4339';digits=4;scale=100},
    @{file='v32_band_origin_round5/resume_20261004/round5_J_reference_excluded_tests.csv'; selector=@{指標='外れ値割合';定義='中央値絶対偏差'};column='加算効果80パーセント検出近似';printed='0.8573';digits=4;scale=100},
    @{file='v34_band_origin_round7/metric_summary.csv'; selector=@{集合='固定29回復';定義='符号付き差';指標='分散比'};column='中央値';printed='0.00704';digits=5;scale=1},
    @{file='v35_band_origin_round8/resume_20261004_final/metric_summary.csv'; selector=@{集合='固定29該当';条件='主帰無';定義='平均標準偏差';指標='全面決定係数'};column='中央値';printed='2.581';digits=3;scale=100},
    @{file='v35_band_origin_round8/resume_20261004_final/metric_summary.csv'; selector=@{集合='固定29該当';条件='主帰無';定義='中央値絶対偏差';指標='全面決定係数'};column='中央値';printed='3.483';digits=3;scale=100}
  )
  $numericRows = foreach ($check in $numericChecks) {
    $sourcePath = Join-Path (Join-Path $repo 'data/results') $check.file
    $matched = @(Import-Csv -LiteralPath $sourcePath -Encoding UTF8 | Where-Object {
      $matchesSelector = $true
      foreach ($column in $check.selector.Keys) { if ($_.$column -ne $check.selector[$column]) { $matchesSelector = $false } }
      $matchesSelector
    })
    if ($matched.Count -ne 1) { throw ('Nonunique numeric source: ' + $check.file) }
    $value = [double]$matched[0].($check.column)
    $rounded = [math]::Round($value * $check.scale, $check.digits).ToString(('F' + $check.digits), [System.Globalization.CultureInfo]::InvariantCulture)
    if ($rounded -ne $check.printed -or -not $body.Contains($check.printed)) { throw ('Numeric transcription mismatch: ' + $check.printed) }
    [pscustomobject]@{path=$sourcePath; selector=($check.selector | ConvertTo-Json -Compress);column=$check.column;source_value=$value;scale=$check.scale;printed=$check.printed;verified=$true}
  }
  $numericRows | Export-Csv -LiteralPath (Join-Path $out 'numeric_transcription_checks.csv') -NoTypeInformation -Encoding UTF8
  [pscustomobject]@{ source_count=$checked.Count; source_change_count=0; report_links=$links.Count; numeric_transcription_checks=$numericRows.Count; report_sha256=(Get-FileHash -LiteralPath $report -Algorithm SHA256).Hash; new_analysis_performed=$false } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'final_verification.json') -Encoding UTF8
  Write-Output ('Unchanged sources: ' + $checked.Count + '; verified links: ' + $links.Count)
}
