param([switch]$Portable, [switch]$SkipTests, [switch]$SkipSamples, [switch]$SetupOnly)
$ErrorActionPreference = 'Stop'
$ttsRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $ttsRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ttsRoot '.tools\python'
$env:UV_PYTHON_BIN_DIR = Join-Path $ttsRoot '.tools\bin'
$env:UV_CACHE_DIR = Join-Path $ttsRoot '.tools\uv-cache'
# Cache and environments share this project volume; hardlinks avoid a second
# multi-GB CUDA copy. uv never requires cached files to remain after install.
$env:UV_LINK_MODE = 'hardlink'
$env:UV_HTTP_TIMEOUT = '300'
$env:UV_INDEX_URL = 'https://pypi.org/simple'
function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program $($Arguments -join ' ')" }
}
try {
    $ttsRunning = Get-Process -Name VieNeuTTSStudio -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq (Join-Path $ttsRoot 'VieNeuTTSStudio.exe') }
    if ($ttsRunning) { throw 'Close TTS Studio in this folder, then run BUILD.bat again.' }
    Write-Host '[2/7] Preparing local Python and package installer...'
    $ttsUv = Join-Path $ttsRoot '.tools\uv\uv.exe'
    if (-not (Test-Path -LiteralPath $ttsUv)) {
        $ttsZip = Join-Path $ttsRoot '.tools\uv.zip'
        Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/download/0.12.19/uv-x86_64-pc-windows-msvc.zip' -OutFile $ttsZip
        if ((Get-FileHash -LiteralPath $ttsZip).Hash -ne '6dbb02d79e419522f1c500f0adb1cddcff0cda7d59b0d66ea7f5e3b4a1b2f5f0') { throw 'uv checksum mismatch.' }
        Expand-Archive -LiteralPath $ttsZip -DestinationPath (Join-Path $ttsRoot '.tools\uv') -Force
        Remove-Item -LiteralPath $ttsZip
    }
    foreach ($ttsEnv in @('.venv','.gpu')) {
        $ttsPython = Join-Path $ttsRoot "$ttsEnv\Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $ttsPython)) {
            Invoke-Checked $ttsUv @('venv', '--managed-python', '--python', '3.12.10', '--seed', $ttsEnv)
        }
    }
    Write-Host '[3/7] Installing pinned CPU / CUDA 12.6 packages...'
    Invoke-Checked $ttsUv @('pip','install','--python','.venv\Scripts\python.exe','--no-deps','-r','requirements-lock.txt')
    # Only PyTorch comes from the PyTorch index. All other pinned wheels use PyPI.
    Invoke-Checked $ttsUv @('pip','install','--python','.gpu\Scripts\python.exe','--no-deps','torch==2.8.0+cu126','--index-url','https://download.pytorch.org/whl/cu126')
    $ttsGpuReq = Join-Path $ttsRoot '.tools\gpu-requirements.txt'
    Get-Content requirements-gpu-lock.txt | Where-Object { $_ -notmatch '^torch==' } | Set-Content -LiteralPath $ttsGpuReq -Encoding utf8
    Invoke-Checked $ttsUv @('pip','install','--python','.gpu\Scripts\python.exe','--no-deps','-r',$ttsGpuReq)
    Write-Host '[4/7] Preparing pinned VieNeu SDK and FFmpeg...'
    Invoke-Checked '.venv\Scripts\python.exe' @('tools\setup_assets.py')
    foreach ($ttsEnv in @('.venv','.gpu')) {
        Invoke-Checked "$ttsEnv\Scripts\python.exe" @('-m','pip','install','--no-deps','--no-build-isolation','.\vendor\VieNeu-TTS')
    }
    Write-Host '[5/7] Downloading models (per-file progress; rerun to resume)...'
    Invoke-Checked '.venv\Scripts\python.exe' @('tools\download_models.py')
    if (-not $SkipSamples) {
        Write-Host '[6/7] Preparing 25 short-story voice samples (cached for next build)...'
        Invoke-Checked '.venv\Scripts\python.exe' @('tools\prepare_samples.py')
    }
    if (-not $SetupOnly) {
        Write-Host '[7/7] Testing and building EXE...'
        & (Join-Path $ttsRoot 'build.ps1') -Portable:$Portable -SkipTests:$SkipTests
        exit $LASTEXITCODE
    }
    Write-Host 'SETUP OK. Models are ready for offline use.'
    exit 0
} catch {
    Write-Host ('FAILED: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Fix the reported issue and run BUILD.bat again. Completed model downloads are reused.'
    exit 1
}
