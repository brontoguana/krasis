param(
    [Parameter(Mandatory = $true)]
    [string]$InstallRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Runtime-Manifest.ps1")

$CurrentPath = Join-Path $InstallRoot "runtime\current.txt"
if (-not (Test-Path $CurrentPath -PathType Leaf)) {
    throw "Installed runtime activation pointer is missing: $CurrentPath"
}
$CurrentName = (Get-Content -Raw $CurrentPath).Trim()
if ($CurrentName -notmatch "^[A-Za-z0-9._-]+$") {
    throw "Installed runtime activation pointer is invalid: $CurrentName"
}
$RuntimeRoot = Join-Path $InstallRoot "runtime\releases\$CurrentName"
$Manifest = Read-KrasisRuntimeManifest -RuntimeRoot $RuntimeRoot
$Probe = Assert-KrasisPrivateRuntime `
    -RuntimeRoot $RuntimeRoot `
    -Manifest $Manifest `
    -IncludeTorch

Write-Host "Installed private runtime validated:" -ForegroundColor Green
Write-Host "  Python: $($Probe.python_version) ($($Probe.python_cache_tag))"
Write-Host "  Krasis: $($Probe.krasis_version)"
Write-Host "  PyTorch: $($Probe.torch_version), CUDA $($Probe.torch_cuda)"
Write-Host "  Runtime: $RuntimeRoot"
