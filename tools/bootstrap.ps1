param([switch]$Portable, [switch]$SkipTests, [switch]$SkipSamples, [switch]$SetupOnly, [switch]$NoPause)
# This entry point deliberately supports Windows PowerShell 5.1, already on Windows 10/11.
$ErrorActionPreference = 'Stop'
$ttsRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $ttsRoot
function Get-TtsHash([string]$Path) {
    $ttsStream = [IO.File]::OpenRead($Path)
    $ttsSha = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($ttsSha.ComputeHash($ttsStream)).Replace('-', '').ToLowerInvariant() }
    finally { $ttsStream.Dispose(); $ttsSha.Dispose() }
}
try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows x64 is required.' }
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $ttsTools = Join-Path $ttsRoot '.tools'
    New-Item -ItemType Directory -Path $ttsTools -Force | Out-Null
    $ttsPwsh = Join-Path $ttsTools 'powershell\pwsh.exe'
    if (-not (Test-Path -LiteralPath $ttsPwsh)) {
        Write-Host '[1/7] Downloading local PowerShell 7 (no admin, no PATH changes)...'
        $ttsZip = Join-Path $ttsTools 'powershell.zip'
        $ttsHash = '02fe458be20493fbdf43f61ea20610b811ee6c738ab1676c61b9cfcd1a33c860'
        if (-not (Test-Path -LiteralPath $ttsZip) -or (Get-TtsHash $ttsZip) -ne $ttsHash) {
            $ttsUrl = 'https://github.com/PowerShell/PowerShell/releases/download/v7.6.6/PowerShell-7.6.6-win-x64.zip'
            if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
                & curl.exe --fail --location --retry 3 --progress-bar $ttsUrl --output $ttsZip
                if ($LASTEXITCODE -ne 0) { throw 'PowerShell download failed.' }
            } else {
                Invoke-WebRequest -UseBasicParsing $ttsUrl -OutFile $ttsZip
            }
        }
        if ((Get-TtsHash $ttsZip) -ne $ttsHash) { throw 'PowerShell download checksum mismatch; run BUILD.bat again.' }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $ttsExtractRoot = [IO.Path]::GetFullPath((Join-Path $ttsTools 'powershell')) + '\'
        $ttsArchive = [IO.Compression.ZipFile]::OpenRead($ttsZip)
        try {
            foreach ($ttsEntry in $ttsArchive.Entries) {
                $ttsEntryPath = [IO.Path]::GetFullPath((Join-Path $ttsExtractRoot $ttsEntry.FullName))
                if (-not $ttsEntryPath.StartsWith($ttsExtractRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe archive path.' }
                if ($ttsEntry.Name) {
                    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($ttsEntryPath)) | Out-Null
                    [IO.Compression.ZipFileExtensions]::ExtractToFile($ttsEntry, $ttsEntryPath, $true)
                }
            }
        } finally { $ttsArchive.Dispose() }
        Remove-Item -LiteralPath $ttsZip
    }
    & $ttsPwsh -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'setup.ps1') -Portable:$Portable -SkipTests:$SkipTests -SkipSamples:$SkipSamples -SetupOnly:$SetupOnly
    exit $LASTEXITCODE
} catch {
    Write-Host ('SETUP FAILED: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
