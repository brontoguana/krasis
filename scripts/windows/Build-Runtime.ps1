param(
    [Parameter(Mandatory = $true)]
    [string]$Wheelhouse,
    [Parameter(Mandatory = $true)]
    [string]$PythonInstaller,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeRequirements,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [Parameter(Mandatory = $true)]
    [string]$PythonVersion,
    [Parameter(Mandatory = $true)]
    [string]$TorchVersion,
    [Parameter(Mandatory = $true)]
    [string]$TorchCuda,
    [Parameter(Mandatory = $true)]
    [string]$TorchUrl
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Runtime-Manifest.ps1")

$WheelhousePath = (Resolve-Path $Wheelhouse).Path
$PythonInstallerPath = (Resolve-Path $PythonInstaller).Path
$RuntimeRequirementsPath = (Resolve-Path $RuntimeRequirements).Path
$OutputPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDir))
$RelocationProbe = "$OutputPath-relocation-probe"

Remove-Item -Recurse -Force $OutputPath -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $RelocationProbe -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

$InstallerArgs = @(
    "/quiet",
    "InstallAllUsers=0",
    "TargetDir=`"$OutputPath`"",
    "Include_pip=1",
    "Include_launcher=0",
    "Include_doc=0",
    "Include_test=0",
    "AssociateFiles=0",
    "Shortcuts=0",
    "PrependPath=0"
)
Write-Host "Building private CPython $PythonVersion runtime..."
& $PythonInstallerPath $InstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "CPython staging installer failed with status $LASTEXITCODE."
}

$Python = Join-Path $OutputPath "python.exe"
if (-not (Test-Path $Python -PathType Leaf)) {
    throw "Staged private Python executable is missing: $Python"
}

$VersionParts = $PythonVersion.Split(".")
if ($VersionParts.Count -lt 2) {
    throw "Invalid Python version: $PythonVersion"
}
$AbiDigits = "$($VersionParts[0])$($VersionParts[1])"
$PthPath = Join-Path $OutputPath "python${AbiDigits}._pth"
$PthContent = @(
    "python${AbiDigits}.zip",
    ".",
    "DLLs",
    "Lib",
    "Lib\site-packages",
    "import site"
)
Set-Content -Path $PthPath -Value $PthContent -Encoding ASCII

$Wheel = @(
    Get-ChildItem -Path $WheelhousePath -Filter "krasis-*.whl"
)
if ($Wheel.Count -ne 1) {
    throw "Expected exactly one Krasis wheel in $WheelhousePath, found $($Wheel.Count)."
}

$OldBytecode = [Environment]::GetEnvironmentVariable("PYTHONDONTWRITEBYTECODE", "Process")
try {
    $env:PYTHONDONTWRITEBYTECODE = "1"
    Write-Host "Installing Krasis and pinned core dependencies into the private runtime..."
    & $Python -I -B -m pip install `
        --no-index `
        --find-links $WheelhousePath `
        --no-compile `
        --no-warn-script-location `
        --disable-pip-version-check `
        --requirement $RuntimeRequirementsPath `
        $Wheel[0].FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Private-runtime package installation failed with status $LASTEXITCODE."
    }

    # Console-script wrappers contain build-machine paths and are never used.
    # Krasis and pip are always invoked as modules by the absolute interpreter.
    Remove-Item -Recurse -Force (Join-Path $OutputPath "Scripts") -ErrorAction SilentlyContinue

    $Probe = Get-KrasisRuntimeProbe -Python $Python
    if ($Probe.python_version -ne $PythonVersion) {
        throw "Staged Python version mismatch: expected $PythonVersion, got $($Probe.python_version)."
    }
    if ($Probe.python_cache_tag -ne "cpython-$AbiDigits") {
        throw "Staged Python ABI mismatch: expected cpython-$AbiDigits, got $($Probe.python_cache_tag)."
    }
    if ($Probe.architecture -ne "AMD64" -or [int]$Probe.pointer_bits -ne 64) {
        throw "Staged Python architecture mismatch: expected AMD64/64-bit."
    }

    Get-ChildItem -Path $OutputPath -Recurse -Directory -Filter "__pycache__" |
        Remove-Item -Recurse -Force
    Get-ChildItem -Path $OutputPath -Recurse -File -Include "*.pyc", "*.pyo" |
        Remove-Item -Force

    $PayloadHash = Get-KrasisRuntimePayloadHash -RuntimeRoot $OutputPath
    $Manifest = [ordered]@{
        schema_version = 1
        bundle_id = "krasis-$Version-cp$AbiDigits-win_amd64"
        release_version = $Version
        python_version = $PythonVersion
        python_cache_tag = "cpython-$AbiDigits"
        architecture = "AMD64"
        krasis_version = $Probe.krasis_version
        torch_version = $TorchVersion
        torch_cuda = $TorchCuda
        torch_url = $TorchUrl
        payload_sha256 = $PayloadHash
    }
    $Manifest |
        ConvertTo-Json -Depth 4 |
        Set-Content -Path (Join-Path $OutputPath "runtime-manifest.json") -Encoding UTF8

    $ReadManifest = Read-KrasisRuntimeManifest -RuntimeRoot $OutputPath
    $VerifiedHash = Get-KrasisRuntimePayloadHash -RuntimeRoot $OutputPath
    if ($VerifiedHash -ne $ReadManifest.payload_sha256) {
        throw "Private-runtime payload hash is not reproducible after manifest creation."
    }
    [void](Assert-KrasisPrivateRuntime -RuntimeRoot $OutputPath -Manifest $ReadManifest)

    Write-Host "Validating that the private runtime is relocatable..."
    Copy-Item -Recurse -Force $OutputPath $RelocationProbe
    $RelocatedManifest = Read-KrasisRuntimeManifest -RuntimeRoot $RelocationProbe
    [void](Assert-KrasisPrivateRuntime -RuntimeRoot $RelocationProbe -Manifest $RelocatedManifest)
} finally {
    if ($null -eq $OldBytecode) {
        Remove-Item Env:PYTHONDONTWRITEBYTECODE -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONDONTWRITEBYTECODE = $OldBytecode
    }
    Remove-Item -Recurse -Force $RelocationProbe -ErrorAction SilentlyContinue
}

Write-Host "Private Krasis runtime built and validated: $OutputPath" -ForegroundColor Green
