param(
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallRoot = Split-Path -Parent $ScriptDir
$Python = Join-Path $InstallRoot "venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Krasis private Python environment is missing." -ForegroundColor Red
    Write-Host "Run the Krasis installer repair action or execute:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File `"$ScriptDir\Install-Krasis.ps1`""
    if (-not $NoPause) {
        Read-Host "Press Enter to close"
    }
    exit 1
}

$env:KRASIS_WINDOWS_NATIVE = "1"
$env:PYTHONUTF8 = "1"

try {
    & $Python -m krasis.launcher
    $status = $LASTEXITCODE
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    $status = 1
}

if ($status -ne 0 -and -not $NoPause) {
    Write-Host ""
    Write-Host "Krasis exited with status $status." -ForegroundColor Yellow
    Read-Host "Press Enter to close"
}

exit $status
