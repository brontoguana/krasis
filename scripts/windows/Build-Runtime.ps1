param(
    [Parameter(Mandatory = $true)]
    [string]$Wheelhouse,
    [Parameter(Mandatory = $true)]
    [string]$PythonRuntimeArchive,
    [Parameter(Mandatory = $true)]
    [string]$BuildPython,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeRequirements,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [Parameter(Mandatory = $true)]
    [string]$PythonVersion,
    [Parameter(Mandatory = $true)]
    [string]$PythonRuntimeArchiveSha256,
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
$PythonRuntimeArchivePath = (Resolve-Path $PythonRuntimeArchive).Path
$BuildPythonPath = (Resolve-Path $BuildPython).Path
$RuntimeRequirementsPath = (Resolve-Path $RuntimeRequirements).Path
$OutputPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDir))
$RelocationProbe = "$OutputPath-relocation-probe"

Remove-Item -Recurse -Force $OutputPath -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $RelocationProbe -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

$ActualPythonRuntimeArchiveHash = (
    Get-FileHash -Algorithm SHA256 -Path $PythonRuntimeArchivePath
).Hash.ToLowerInvariant()
if ($ActualPythonRuntimeArchiveHash -ne $PythonRuntimeArchiveSha256.ToLowerInvariant()) {
    throw "CPython runtime archive SHA-256 mismatch: expected $PythonRuntimeArchiveSha256, got $ActualPythonRuntimeArchiveHash."
}

Write-Host "Extracting private CPython $PythonVersion embeddable runtime..."
Expand-Archive -Path $PythonRuntimeArchivePath -DestinationPath $OutputPath -Force

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
$PrivateSitePackages = Join-Path $OutputPath "Lib\site-packages"
New-Item -ItemType Directory -Force -Path $PrivateSitePackages | Out-Null

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
    & $BuildPythonPath -m pip install `
        --target $PrivateSitePackages `
        --no-index `
        --find-links $WheelhousePath `
        --only-binary ":all:" `
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
    Remove-Item -Recurse -Force (Join-Path $OutputPath "bin") -ErrorAction SilentlyContinue

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
        python_runtime_archive_sha256 = $ActualPythonRuntimeArchiveHash
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
