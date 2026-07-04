param(
    [string]$InstallRoot,
    [string]$Wheelhouse,
    [switch]$NoShortcut,
    [switch]$DesktopShortcut,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $InstallRoot = Split-Path -Parent $ScriptDir
}
if ([string]::IsNullOrWhiteSpace($Wheelhouse)) {
    $Wheelhouse = Join-Path $ScriptDir "wheelhouse"
}

function Find-KrasisPython {
    $rootBundled = Join-Path $InstallRoot "python\python.exe"
    if (Test-Path $rootBundled) {
        return $rootBundled
    }

    $bundled = Join-Path $ScriptDir "python\python.exe"
    if (Test-Path $bundled) {
        return $bundled
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $candidate = & $py.Source -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $candidate -and (Test-Path $candidate.Trim())) {
            return $candidate.Trim()
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        $candidate = & $python.Source -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $candidate -and (Test-Path $candidate.Trim())) {
            return $candidate.Trim()
        }
    }

    throw "No suitable Python 3.10+ interpreter found. Install Python 3.12 or use a Krasis installer bundle that includes Python."
}

New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
$VenvDir = Join-Path $InstallRoot "venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if ($Force -and (Test-Path $VenvDir)) {
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvPython)) {
    $SourcePython = Find-KrasisPython
    Write-Host "Creating Krasis private Python environment..."
    & $SourcePython -m venv $VenvDir
}

if (-not (Test-Path $Wheelhouse)) {
    throw "Wheelhouse not found: $Wheelhouse"
}

$Wheel = Get-ChildItem -Path $Wheelhouse -Filter "krasis-*.whl" |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
if (-not $Wheel) {
    throw "No Krasis wheel found in wheelhouse: $Wheelhouse"
}

Write-Host "Installing Krasis from $($Wheel.Name)..."
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install --no-index --find-links $Wheelhouse $Wheel.FullName

if (-not $NoShortcut) {
    $Programs = [Environment]::GetFolderPath("Programs")
    $KrasisMenu = Join-Path $Programs "Krasis"
    New-Item -ItemType Directory -Force -Path $KrasisMenu | Out-Null

    $Launcher = Join-Path $ScriptDir "Launch-Krasis.ps1"
    $ShortcutPath = Join-Path $KrasisMenu "Krasis.lnk"
    $Shell = New-Object -ComObject WScript.Shell
    $Shortcut = $Shell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $Shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -NoExit -File `"$Launcher`""
    $Shortcut.WorkingDirectory = [Environment]::GetFolderPath("UserProfile")
    $Shortcut.WindowStyle = 3
    $Shortcut.Description = "Open the Krasis interactive launcher"
    $Shortcut.Save()

    if ($DesktopShortcut) {
        $Desktop = [Environment]::GetFolderPath("Desktop")
        $DesktopPath = Join-Path $Desktop "Krasis.lnk"
        Copy-Item -Force $ShortcutPath $DesktopPath
    }
}

Write-Host "Krasis Windows install complete." -ForegroundColor Green
