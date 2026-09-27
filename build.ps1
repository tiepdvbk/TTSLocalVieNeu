param([switch]$Portable, [switch]$SkipTests, [switch]$ReuseBuild)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$ttsRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')

function Remove-BuildDirectory([string]$RelativePath) {
    $ttsTarget = [IO.Path]::GetFullPath((Join-Path $ttsRoot $RelativePath))
    if (-not $ttsTarget.StartsWith($ttsRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe build path: $ttsTarget"
    }
    if (Test-Path -LiteralPath $ttsTarget) { Remove-Item -LiteralPath $ttsTarget -Recurse -Force }
}

$ttsRunning = Get-Process -Name VieNeuTTSStudio -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq (Join-Path $ttsRoot 'VieNeuTTSStudio.exe') }
if ($ttsRunning) { throw 'Close TTS Studio in this folder before building. Your saved jobs are preserved.' }
$ttsPython = Join-Path $ttsRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ttsPython)) { throw 'Missing .venv\Scripts\python.exe; restore the project build environment first.' }
$ttsBasePython = (& $ttsPython -c 'import sys; print(sys.base_prefix)').Trim()
if ($LASTEXITCODE -ne 0) { throw 'Could not locate base Python.' }

# Install the pinned checkout without changing CUDA dependencies.
if (-not $ReuseBuild) {
foreach ($ttsRuntime in @('.venv','.gpu')) {
    $ttsInterpreter = Join-Path $ttsRoot "$ttsRuntime\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $ttsInterpreter)) { throw "Missing runtime: $ttsInterpreter" }
    & $ttsInterpreter -m pip install --no-deps --no-build-isolation .\vendor\VieNeu-TTS
    if ($LASTEXITCODE -ne 0) { throw "SDK installation failed: $ttsRuntime" }
}
if (-not $SkipTests) {
    & $ttsPython -m pytest tests -q
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed; existing EXE was not replaced.' }
}
& $ttsPython -m PyInstaller --noconfirm --clean --distpath package --workpath build VieNeuTTSStudio.spec
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed; existing EXE was not replaced.' }
}
if (-not (Test-Path -LiteralPath 'package\VieNeuTTSStudio\VieNeuTTSStudio.exe')) { throw 'No completed build to package.' }
foreach ($ttsIcuName in @('icuuc.dll', 'icudt78.dll')) {
    $ttsPackIcu = Join-Path $ttsRoot "package\VieNeuTTSStudio\_internal\$ttsIcuName"
    if (Test-Path -LiteralPath $ttsPackIcu) { Remove-Item -LiteralPath $ttsPackIcu }
}
# Replace the whole runtime so stale DLLs cannot remain from older builds.
Remove-BuildDirectory '_internal'
Copy-Item -LiteralPath 'package\VieNeuTTSStudio\_internal' -Destination '.' -Recurse -Force
Copy-Item -LiteralPath 'package\VieNeuTTSStudio\VieNeuTTSStudio.exe' -Destination '.\VieNeuTTSStudio.exe' -Force
Write-Host "EXE: $ttsRoot\VieNeuTTSStudio.exe"

if ($Portable) {
    $ttsPortable = Join-Path $ttsRoot 'portable\VieNeuTTSStudio'
    Remove-BuildDirectory 'portable\VieNeuTTSStudio'
    New-Item -ItemType Directory -Path $ttsPortable -Force | Out-Null
    Copy-Item -Path 'package\VieNeuTTSStudio\*' -Destination $ttsPortable -Recurse -Force
    foreach ($ttsDir in @('models','licenses','voice_samples')) {
        if (Test-Path -LiteralPath $ttsDir) { Copy-Item -LiteralPath $ttsDir -Destination $ttsPortable -Recurse -Force }
    }
    New-Item -ItemType Directory -Path (Join-Path $ttsPortable 'tools') -Force | Out-Null
    Copy-Item -LiteralPath 'tools\engine_worker.py','tools\runtime_support.py','tools\ffmpeg.exe' -Destination (Join-Path $ttsPortable 'tools') -Force
    foreach ($ttsFile in @('HUONG_DAN.html','HUONG_DAN_V2.html','README.md','THIRD_PARTY_NOTICES.txt','vendor-revision.txt','model-revisions.json','voice-calibration.json','UPGRADE_NOTES.md')) {
        if (Test-Path -LiteralPath $ttsFile) { Copy-Item -LiteralPath $ttsFile -Destination $ttsPortable -Force }
    }
    foreach ($ttsName in @('data','cache','logs','output')) {
        New-Item -ItemType Directory -Path (Join-Path $ttsPortable $ttsName) -Force | Out-Null
    }
    foreach ($ttsSpec in @(@('runtime_cpu','.venv'), @('runtime_gpu','.gpu'))) {
        $ttsDestination = Join-Path $ttsPortable $ttsSpec[0]
        New-Item -ItemType Directory -Path (Join-Path $ttsDestination 'Lib') -Force | Out-Null
        Get-ChildItem -LiteralPath $ttsBasePython -File | Copy-Item -Destination $ttsDestination -Force
        Copy-Item -LiteralPath (Join-Path $ttsBasePython 'DLLs') -Destination $ttsDestination -Recurse -Force
        # Copy the standard library only, never unrelated global packages.
        # This avoids first copying then deleting gigabytes of global tools.
        Get-ChildItem -LiteralPath (Join-Path $ttsBasePython 'Lib') | Where-Object { $_.Name -notin @('site-packages','__pycache__') } | Copy-Item -Destination (Join-Path $ttsDestination 'Lib') -Recurse -Force
        $ttsSitePackages = Join-Path $ttsDestination 'Lib\site-packages'
        New-Item -ItemType Directory -Path $ttsSitePackages -Force | Out-Null
        # Linker/debug libraries are build-time files, not needed by the
        # portable inference runtime. Copy DLL/PYD/Python/data files in parallel.
        & "$env:SystemRoot\System32\robocopy.exe" (Join-Path $ttsRoot ($ttsSpec[1] + '\Lib\site-packages')) $ttsSitePackages /E /MT:8 /R:1 /W:1 /XF *.lib *.pdb /NFL /NDL /NJH /NJS /NP
        if ($LASTEXITCODE -ge 8) { throw "Runtime copy failed: $($ttsSpec[0]) (robocopy $LASTEXITCODE)" }
        $global:LASTEXITCODE = 0
        Write-Host "Runtime ready: $($ttsSpec[0])"
    }
    Write-Host "Portable: $ttsPortable"
}
exit 0
