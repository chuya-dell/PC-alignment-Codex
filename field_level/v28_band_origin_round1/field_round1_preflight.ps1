param([string]$Repository = (Get-Location).Path)
$ErrorActionPreference = 'Stop'
$culture = [Globalization.CultureInfo]::InvariantCulture
$inputRoot = Get-Item -LiteralPath (Join-Path $Repository 'data/inputs_local')
$children = @(Get-ChildItem -LiteralPath $inputRoot.FullName)
$sources = @($children | Where-Object { $_.PSIsContainer -and $_.Name -like '2026*digital_judgment*' })
if ($sources.Count -ne 1) { throw '解析入力フォルダを一意に特定できません。' }
$source = $sources[0]
$sourceChildren = @(Get-ChildItem -LiteralPath $source.FullName)
$tables = $sourceChildren | Where-Object Name -eq 'tables'
$cache = Get-ChildItem -LiteralPath $tables.FullName | Where-Object Name -eq 'cached_field_differences'
$v27 = $sourceChildren | Where-Object Name -like '*v27'
$v27tables = Get-ChildItem -LiteralPath $v27.FullName | Where-Object Name -eq 'tables'
$fieldTable = Get-ChildItem -LiteralPath $v27tables.FullName | Where-Object Name -eq '全視野と外れ値判定.csv'
$manifest = Get-ChildItem -LiteralPath $tables.FullName | Where-Object Name -eq 'table_difference_cache_manifest_by_field.csv'
$raw = $children | Where-Object Name -eq 'raw_readonly'
$rawChildren = @(Get-ChildItem -LiteralPath $raw.FullName)
$output = Join-Path $Repository 'data/results/v28_band_origin_round1'
New-Item -ItemType Directory -Path $output -Force | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem
$expected = @{}
Import-Csv -LiteralPath $manifest.FullName -Encoding UTF8 | ForEach-Object {
    $name = ($_.cache_file -split '[\\/]')[-1]
    $expected[$name] = [long]$_.bytes
}
$cacheLookup = @{}
$inventory = @(Get-ChildItem -LiteralPath $cache.FullName -File | Where-Object Extension -eq '.npz' | ForEach-Object {
    $file = $_
    $cacheLookup[$file.BaseName] = $file
    $archive = [IO.Compression.ZipFile]::OpenRead($file.FullName)
    try { $entries = @($archive.Entries | ForEach-Object FullName) } finally { $archive.Dispose() }
    [pscustomobject]@{path=$file.FullName; bytes=$file.Length; expected_bytes=$expected[$file.Name]; size_matches=($file.Length -eq $expected[$file.Name]); archive_entries=($entries -join ';'); values_read=$false}
})
$inventory | Export-Csv -LiteralPath (Join-Path $output 'cache_inventory.csv') -NoTypeInformation -Encoding UTF8
$fields = @(Import-Csv -LiteralPath $fieldTable.FullName -Encoding UTF8 | ForEach-Object {
    $row = $_
    $key = '{0}_{1}_{2}' -f $row.date,$row.board,$row.field
    if (-not $cacheLookup.ContainsKey($key)) { throw "保存データがありません: $key" }
    $rate = [double]::Parse($row.positive_fraction,$culture)
    $condition = $row.condition
    $concentration = switch ($condition) {
        '0' {'ブランク'} '1e-09' {'1ナノモル毎リットル'} '1e-9' {'1ナノモル毎リットル'}
        '1e-10' {'100ピコモル毎リットル'} '1e-11' {'10ピコモル毎リットル'}
        '1e-12' {'1ピコモル毎リットル'} '1e-13' {'100フェムトモル毎リットル'}
        '1e-14' {'10フェムトモル毎リットル'} '1e-15' {'1フェムトモル毎リットル'}
        'mismatch' {'ミスマッチ配列'} '100uL_10fM' {'100マイクロリットル・10フェムトモル毎リットル'}
        default { $condition }
    }
    [pscustomobject]@{日程=$row.date; 基板=$row.board; 視野番号=$row.field; 位置番号=$row.field; 濃度=$concentration; 入力条件原表記=$condition; ブランク=($condition -eq '0'); 外れ値ピラー割合=$rate; 外れ値ピラー割合百分率=(100*$rate); 閾値=$row.threshold; ピラー数=$row.pillars; 外れ値ピラー数=$row.positive; 外れ値視野3percent以上=($rate -ge 0.03); 保存データ実在パス=$cacheLookup[$key].FullName; 出典=$fieldTable.FullName; 今回差から再計算=$false}
})
$fields | Export-Csv -LiteralPath (Join-Path $output 'all_fields_existing_definition.csv') -NoTypeInformation -Encoding UTF8
$outliers = @($fields | Where-Object 外れ値視野3percent以上)
$outliers | Export-Csv -LiteralPath (Join-Path $output 'outlier_fields_ge3percent.csv') -NoTypeInformation -Encoding UTF8
$cutoffs = @(0.01,0.02,0.03,0.04,0.05 | ForEach-Object {
    $cutoff = $_
    [pscustomobject]@{外れ値視野基準割合=$cutoff; 以上視野数=@($fields | Where-Object { $_.外れ値ピラー割合 -ge $cutoff }).Count; 超視野数=@($fields | Where-Object { $_.外れ値ピラー割合 -gt $cutoff }).Count; 出典=$fieldTable.FullName; 中央値絶対偏差定義='未計算'}
})
$cutoffs | Export-Csv -LiteralPath (Join-Path $output 'field_cutoff_counts.csv') -NoTypeInformation -Encoding UTF8
$dates = @($fields | Group-Object 日程 | ForEach-Object {
    [pscustomobject]@{日程=$_.Name; 全視野数=$_.Count; 外れ値視野数=@($_.Group | Where-Object 外れ値視野3percent以上).Count; ブランク視野数=@($_.Group | Where-Object ブランク).Count; 位置6または7の外れ値視野数=@($_.Group | Where-Object { $_.外れ値視野3percent以上 -and $_.位置番号 -in @('6','7') }).Count}
})
$dates | Export-Csv -LiteralPath (Join-Path $output 'date_counts.csv') -NoTypeInformation -Encoding UTF8
$keys = @('260926_01_8','260926_3_3','260926_5_3','260926_7_5','260926_7_8')
$five = @($fields | Where-Object { ('{0}_{1}_{2}' -f $_.日程,$_.基板,$_.視野番号) -in $keys })
$five | Export-Csv -LiteralPath (Join-Path $output 'requested_five_fields.csv') -NoTypeInformation -Encoding UTF8
$five | ForEach-Object { Get-FileHash -LiteralPath $_.保存データ実在パス -Algorithm SHA256 } | Export-Csv -LiteralPath (Join-Path $output 'five_cache_sha256.csv') -NoTypeInformation -Encoding UTF8
$used = @($fieldTable.FullName,$manifest.FullName)
$used | ForEach-Object { Get-FileHash -LiteralPath $_ -Algorithm SHA256 } | Export-Csv -LiteralPath (Join-Path $output 'source_table_sha256.csv') -NoTypeInformation -Encoding UTF8
$status = [ordered]@{
    observed_at=(Get-Date).ToString('o'); computer=$env:COMPUTERNAME; user=$env:USERNAME
    recognized_drives=@(Get-PSDrive -PSProvider FileSystem | Select-Object Name,Root)
    input_root=$inputRoot.FullName; input_children=@($children | Select-Object Name,FullName)
    source=$source.FullName; source_children=@($sourceChildren | Select-Object Name,FullName)
    raw_children=@($rawChildren | Select-Object Name,FullName)
    copy_done_files=@($rawChildren | Where-Object { -not $_.PSIsContainer -and $_.Name -like '*copy_done*' } | Select-Object FullName)
    raw_copy_complete=(@($rawChildren | Where-Object { -not $_.PSIsContainer -and $_.Name -like '*copy_done*' }).Count -gt 0)
    cache_count=$inventory.Count; size_mismatches=@($inventory | Where-Object { -not $_.size_matches }).Count
    archive_schema_counts=@($inventory | Group-Object archive_entries | Select-Object Name,Count)
    field_count=$fields.Count; unique_field_count=@($fields | ForEach-Object { '{0}_{1}_{2}' -f $_.日程,$_.基板,$_.視野番号 } | Sort-Object -Unique).Count
    outlier_count=$outliers.Count; cutoff_counts=$cutoffs; date_counts=$dates
    all_fields_share_field_position= $true
    branch=(& git branch --show-current); commit=(& git rev-parse HEAD)
    checked_branches=@(& git branch --all); checked_tags=@(& git tag --list)
    cache_values_read=$false; scientific_calculation_executed=$false
}
$status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'preflight.json') -Encoding UTF8
$status | ConvertTo-Json -Depth 3
