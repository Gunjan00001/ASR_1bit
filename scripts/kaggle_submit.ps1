# Submit the attention-only QAT kernel to Kaggle.
#
# GitHub is the source of truth: this refuses to submit unless the working tree
# is clean and HEAD == origin/main, then pins the exact commit SHA into the
# pushed kernel (replacing the __REPO_REVISION__ placeholder).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\kaggle_submit.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\kaggle_submit.ps1 -Accelerator NvidiaL4
[CmdletBinding()]
param(
    [string]$KernelId = "gunjanpal/asr-1bit-qat-attn",
    [string]$Accelerator = "NvidiaTeslaT4"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Repository: $RepoRoot"
Push-Location $RepoRoot
try {
    $dirty = git status --porcelain
    if ($dirty) {
        throw "Working tree is not clean. Commit before submitting:`n$dirty"
    }

    git fetch origin --quiet
    if ($LASTEXITCODE -ne 0) { throw "git fetch failed" }

    $head = (git rev-parse HEAD).Trim()
    $origin = (git rev-parse origin/main).Trim()
    if ($head -ne $origin) {
        throw "HEAD ($head) != origin/main ($origin). Commit and push before submitting."
    }
    $rev = $head

    $build = Join-Path $RepoRoot "kaggle\build"
    if (Test-Path $build) { Remove-Item -Recurse -Force $build }
    New-Item -ItemType Directory -Path $build | Out-Null

    $utf8 = New-Object System.Text.UTF8Encoding($false)

    $kernel = Get-Content (Join-Path $RepoRoot "kaggle\kernel.py") -Raw
    if ($kernel -notmatch "__REPO_REVISION__") {
        throw "kernel.py does not contain the __REPO_REVISION__ placeholder"
    }
    $kernel = $kernel.Replace("__REPO_REVISION__", $rev)
    [System.IO.File]::WriteAllText((Join-Path $build "kernel.py"), $kernel, $utf8)

    $meta = Get-Content (Join-Path $RepoRoot "kaggle\kernel-metadata.template.json") -Raw
    $meta = $meta.Replace("__KERNEL_ID__", $KernelId).Replace("__MACHINE_SHAPE__", $Accelerator)
    [System.IO.File]::WriteAllText((Join-Path $build "kernel-metadata.json"), $meta, $utf8)

    Write-Host "Submitting $KernelId at revision $rev on $Accelerator ..."
    kaggle kernels push -p $build
    if ($LASTEXITCODE -ne 0) { throw "kaggle kernels push failed" }

    Write-Host ""
    Write-Host "Submitted. Next:"
    Write-Host "  status : powershell -ExecutionPolicy Bypass -File .\scripts\kaggle_status.ps1"
    Write-Host "  output : powershell -ExecutionPolicy Bypass -File .\scripts\kaggle_output.ps1"
}
finally {
    Pop-Location
}
