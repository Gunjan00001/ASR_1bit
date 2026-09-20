# Check the status of the attention-only QAT Kaggle kernel.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\kaggle_status.ps1
[CmdletBinding()]
param(
    [string]$KernelId = "gunjanpal/asr-1bit-qat-attn"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
# Refresh the OAuth access token non-interactively if it has expired.
kaggle auth print-access-token > $null 2>&1
kaggle kernels status $KernelId
if ($LASTEXITCODE -ne 0) { throw "kaggle kernels status failed" }
