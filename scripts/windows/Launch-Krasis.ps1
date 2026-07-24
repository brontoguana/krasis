param(
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallRoot = Split-Path -Parent $ScriptDir
. (Join-Path $ScriptDir "Runtime-Manifest.ps1")

try {
    $CurrentPath = Join-Path $InstallRoot "runtime\current.txt"
    if (-not (Test-Path $CurrentPath -PathType Leaf)) {
        throw "Krasis private-runtime activation pointer is missing."
    }
    $CurrentName = (Get-Content -Raw $CurrentPath).Trim()
    if ($CurrentName -notmatch "^[A-Za-z0-9._-]+$") {
        throw "Krasis private-runtime activation pointer is invalid."
    }

    $RuntimeRoot = Join-Path $InstallRoot "runtime\releases\$CurrentName"
    $Manifest = Read-KrasisRuntimeManifest -RuntimeRoot $RuntimeRoot
    [void](Assert-KrasisPrivateRuntime `
        -RuntimeRoot $RuntimeRoot `
        -Manifest $Manifest `
        -IncludeTorch)
    $Python = Join-Path $RuntimeRoot "python.exe"

    Remove-Item Env:PYTHONHOME,Env:PYTHONPATH,Env:PYTHONUSERBASE -ErrorAction SilentlyContinue
    $env:PYTHONNOUSERSITE = "1"
    $env:KRASIS_WINDOWS_NATIVE = "1"
    $env:PYTHONUTF8 = "1"

    & $Python -I -m krasis.launcher
    $status = $LASTEXITCODE
} catch {
    Write-Host "Krasis private-runtime validation failed:" -ForegroundColor Red
    Write-Host "  $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Run Krasis Update or reinstall Krasis to repair the private runtime."
    $status = 1
}

if ($status -ne 0 -and -not $NoPause) {
    Write-Host ""
    Write-Host "Krasis exited with status $status." -ForegroundColor Yellow
    Read-Host "Press Enter to close"
}

exit $status
