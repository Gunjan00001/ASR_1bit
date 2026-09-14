# Phase 0 (Windows) — install FFmpeg shared DLLs for torchcodec.
#
# Hugging Face `datasets` audio decoding requires torchcodec, which in turn
# needs FFmpeg *shared* libraries (avcodec/avformat/avutil/...). The static
# ffmpeg.exe alone is NOT enough. This script:
#   1. downloads a BtbN win64-gpl-shared build,
#   2. copies its bin/*.dll into the project venv's Scripts dir,
#   3. registers that dir via a .pth file (Python 3.14 no longer searches the
#      app dir for DLLs, so torchcodec cannot find them otherwise).
#
# Usage (PowerShell, from repo root):
#   .\scripts\setup_windows_ffmpeg.ps1
# Re-run after recreating .venv.

$ErrorActionPreference = 'Stop'

$venvScripts = Join-Path $PSScriptRoot '..\.venv\Scripts' | Resolve-Path
$venvSite = Join-Path $PSScriptRoot '..\.venv\Lib\site-packages' | Resolve-Path
$zip = Join-Path $env:TEMP 'ffmpeg-shared.zip'
$extractDir = Join-Path $env:TEMP 'ffmpeg-shared-setup'

if (-not (Test-Path $zip)) {
    Invoke-WebRequest `
        -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip' `
        -OutFile $zip -MaximumRedirection 5
}
if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
Expand-Archive -Path $zip -DestinationPath $extractDir -Force

$binDir = Get-ChildItem $extractDir -Recurse -Filter 'avcodec-*.dll' |
    Select-Object -First 1 -ExpandProperty DirectoryName
Copy-Item (Join-Path $binDir '*.dll') -Destination $venvScripts -Force

$pth = Join-Path $venvSite 'zz_ffmpeg_dll_dir.pth'
"import os; os.add_dll_directory(r'$venvScripts')" | Set-Content $pth -NoNewline

& (Join-Path $venvScripts 'python.exe') -c "from torchcodec.decoders import AudioDecoder; print('AudioDecoder import: OK')"
