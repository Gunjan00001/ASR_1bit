# Download the outputs of the attention-only QAT Kaggle kernel.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\kaggle_output.ps1
[CmdletBinding()]
param(
    [string]$KernelId = "gunjanpal/asr-1bit-qat-attn",
    [string]$OutDir = "kaggle\output"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Push-Location $RepoRoot
try {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    kaggle kernels output $KernelId -p $OutDir
    if ($LASTEXITCODE -ne 0) { throw "kaggle kernels output failed" }

    Write-Host ""
    Write-Host "Downloaded to $OutDir :"
    Get-ChildItem -Recurse -File $OutDir | Select-Object FullName, Length | Format-Table -AutoSize
}
finally {
    Pop-Location
}
