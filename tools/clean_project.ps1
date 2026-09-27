param([switch]$BuildArtifacts)
$ErrorActionPreference = 'Stop'
$ttsRoot = [IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent)).TrimEnd('\')
$ttsRemoved = @()
$ttsPreviousBytes = 0
$ttsReportPath = Join-Path $ttsRoot 'logs\cleanup-result.json'
if (Test-Path -LiteralPath $ttsReportPath) {
    $ttsPrevious = Get-Content -LiteralPath $ttsReportPath -Raw | ConvertFrom-Json
    $ttsRemoved = @($ttsPrevious.removed)
    $ttsPreviousBytes = [long]$ttsPrevious.total_bytes
}
function Remove-TtsGenerated([string]$Path) {
    $ttsTarget = [IO.Path]::GetFullPath($Path)
    if (-not $ttsTarget.StartsWith($ttsRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe cleanup path: $ttsTarget" }
    if (-not (Test-Path -LiteralPath $ttsTarget)) { return }
    $ttsItem = Get-Item -LiteralPath $ttsTarget -Force
    if ($ttsItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing reparse point: $ttsTarget" }
    $ttsBytes = if ($ttsItem.PSIsContainer) { (Get-ChildItem -LiteralPath $ttsTarget -File -Recurse -Force | Measure-Object Length -Sum).Sum } else { $ttsItem.Length }
    Remove-Item -LiteralPath $ttsTarget -Recurse -Force
    $script:ttsRemoved += [pscustomobject]@{ path=$ttsTarget; bytes=[long]$ttsBytes }
}
# Only known release archives and generated logs; never recurse into models,
# user output, job caches, databases, source, or Python environments.
foreach ($ttsFolder in @('release-assets','release-assets-250','release-assets-250b')) {
    $ttsPath = Join-Path $ttsRoot $ttsFolder
    if (Test-Path -LiteralPath $ttsPath) {
        foreach ($ttsZip in Get-ChildItem -LiteralPath $ttsPath -Filter '*.zip' -File) { Remove-TtsGenerated $ttsZip.FullName }
        if (-not (Get-ChildItem -LiteralPath $ttsPath -Force)) { Remove-TtsGenerated $ttsPath }
    }
}
$ttsCutoff = (Get-Date).Date
foreach ($ttsLog in Get-ChildItem -LiteralPath (Join-Path $ttsRoot 'logs') -Force) {
    $ttsDatedRun = $false
    if ($ttsLog.PSIsContainer -and $ttsLog.Name -match '^(benchmark|selftest|ui-e2e)-(\d{8})-') {
        $ttsDatedRun = $Matches[2] -lt $ttsCutoff.ToString('yyyyMMdd')
    }
    if ($ttsLog.LastWriteTime -lt $ttsCutoff -or $ttsDatedRun) {
        Remove-TtsGenerated $ttsLog.FullName
    }
}
foreach ($ttsCache in @('__pycache__','.pytest_cache','app\__pycache__','tests\__pycache__','tools\__pycache__')) {
    Remove-TtsGenerated (Join-Path $ttsRoot $ttsCache)
}
if ($BuildArtifacts) {
    foreach ($ttsCache in @('build','package')) { Remove-TtsGenerated (Join-Path $ttsRoot $ttsCache) }
    foreach ($ttsRuntime in @('runtime_cpu','runtime_gpu')) {
        $ttsPackages = Join-Path $ttsRoot "portable\VieNeuTTSStudio\$ttsRuntime\Lib\site-packages"
        if (Test-Path -LiteralPath $ttsPackages) {
            Get-ChildItem -LiteralPath $ttsPackages -File -Recurse | Where-Object { $_.Extension -in @('.lib','.pdb') } | ForEach-Object { Remove-TtsGenerated $_.FullName }
        }
    }
}
$ttsReport = [pscustomobject]@{ created=(Get-Date).ToString('o'); total_bytes=($ttsRemoved | Measure-Object bytes -Sum).Sum; removed=$ttsRemoved }
$ttsReport | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ttsReportPath -Encoding utf8
Write-Host ('Removed {0:N2} GiB this run, {1:N2} GiB total; details in logs\cleanup-result.json' -f (($ttsReport.total_bytes-$ttsPreviousBytes) / 1GB), ($ttsReport.total_bytes / 1GB))
