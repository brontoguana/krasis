param(
    [Parameter(Mandatory = $true)]
    [string]$InstallRoot,
    [Parameter(Mandatory = $true)]
    [string]$RuntimePackage
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "Runtime-Manifest.ps1")

$RuntimePackagePath = (Resolve-Path $RuntimePackage).Path
$PackageManifest = Read-KrasisRuntimeManifest -RuntimeRoot $RuntimePackagePath
$VersionPath = Join-Path $InstallRoot "VERSION.txt"
if (-not (Test-Path $VersionPath -PathType Leaf)) {
    throw "Krasis installer version marker is missing: $VersionPath"
}
$InstallerVersion = (Get-Content -Raw $VersionPath).Trim()
if ($InstallerVersion -ne $PackageManifest.release_version) {
    throw "Installer/runtime version mismatch: installer=$InstallerVersion, runtime=$($PackageManifest.release_version)."
}
$PackageHash = Get-KrasisRuntimePayloadHash -RuntimeRoot $RuntimePackagePath
if ($PackageHash -ne $PackageManifest.payload_sha256) {
    throw "Packaged private-runtime hash mismatch: expected $($PackageManifest.payload_sha256), got $PackageHash."
}

New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
$RuntimeRoot = Join-Path $InstallRoot "runtime"
$ReleasesRoot = Join-Path $RuntimeRoot "releases"
New-Item -ItemType Directory -Force -Path $ReleasesRoot | Out-Null

$SafeBundleId = "$($PackageManifest.bundle_id)"
if ($SafeBundleId -notmatch "^[A-Za-z0-9._-]+$") {
    throw "Private-runtime bundle ID contains unsafe characters: $SafeBundleId"
}
$ActivationName = "$SafeBundleId-$([Guid]::NewGuid().ToString('N'))"
$NewRuntime = Join-Path $ReleasesRoot $ActivationName
$CurrentPath = Join-Path $RuntimeRoot "current.txt"
$CurrentBackupPath = Join-Path $RuntimeRoot "current.txt.rollback"
$CurrentTempPath = Join-Path $RuntimeRoot "current.txt.new"
$OldCurrent = $null
$PointerSwitched = $false

function Test-KrasisPythonTreeInUse {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TreeRoot
    )

    try {
        $PythonProcesses = @(
            Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction Stop
        )
    } catch {
        Write-Warning "Could not inspect running Python processes; retaining $TreeRoot."
        return $true
    }
    $ResolvedTree = [System.IO.Path]::GetFullPath($TreeRoot).TrimEnd("\", "/")
    foreach ($Process in $PythonProcesses) {
        $Executable = "$($Process.ExecutablePath)"
        $CommandLine = "$($Process.CommandLine)"
        if ([string]::IsNullOrWhiteSpace($Executable) -and
            [string]::IsNullOrWhiteSpace($CommandLine)) {
            Write-Warning "A Python process could not be inspected; retaining $TreeRoot."
            return $true
        }
        if ($Executable.StartsWith(
            $ResolvedTree + [System.IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase
        ) -or $CommandLine.IndexOf(
            $ResolvedTree,
            [StringComparison]::OrdinalIgnoreCase
        ) -ge 0) {
            return $true
        }
    }
    return $false
}

if (Test-Path $CurrentPath -PathType Leaf) {
    $OldCurrent = (Get-Content -Raw $CurrentPath).Trim()
}

$SavedEnvironment = @{}
foreach ($Name in @(
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONUSERBASE",
    "PIP_CONFIG_FILE",
    "PIP_TARGET",
    "PIP_PREFIX",
    "PIP_USER",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "PYTHONDONTWRITEBYTECODE"
)) {
    $SavedEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, "Process")
}

try {
    Remove-Item Env:PYTHONHOME,Env:PYTHONPATH,Env:PYTHONUSERBASE -ErrorAction SilentlyContinue
    Remove-Item Env:PIP_TARGET,Env:PIP_PREFIX,Env:PIP_USER,Env:PIP_INDEX_URL,Env:PIP_EXTRA_INDEX_URL -ErrorAction SilentlyContinue
    $env:PIP_CONFIG_FILE = "NUL"
    $env:PYTHONDONTWRITEBYTECODE = "1"

    Write-Host "Staging Krasis private Python $($PackageManifest.python_version)..."
    New-Item -ItemType Directory -Force -Path $NewRuntime | Out-Null
    Copy-Item -Recurse -Force (Join-Path $RuntimePackagePath "*") $NewRuntime

    $StagedManifest = Read-KrasisRuntimeManifest -RuntimeRoot $NewRuntime
    $StagedHash = Get-KrasisRuntimePayloadHash -RuntimeRoot $NewRuntime
    if ($StagedHash -ne $StagedManifest.payload_sha256) {
        throw "Staged private-runtime hash mismatch: expected $($StagedManifest.payload_sha256), got $StagedHash."
    }
    [void](Assert-KrasisPrivateRuntime -RuntimeRoot $NewRuntime -Manifest $StagedManifest)

    $PrivatePython = Join-Path $NewRuntime "python.exe"
    Write-Host "Installing pinned CUDA PyTorch $($StagedManifest.torch_version)..."
    Write-Host "This is a large first-install download and can take several minutes."
    & $PrivatePython -I -B -m pip install `
        --no-cache-dir `
        --disable-pip-version-check `
        --no-deps `
        --only-binary ":all:" `
        "$($StagedManifest.torch_url)"
    if ($LASTEXITCODE -ne 0) {
        throw "Pinned CUDA PyTorch installation failed with status $LASTEXITCODE."
    }

    $Probe = Assert-KrasisPrivateRuntime `
        -RuntimeRoot $NewRuntime `
        -Manifest $StagedManifest `
        -IncludeTorch
    $Activation = [ordered]@{
        schema_version = 1
        activated_utc = [DateTime]::UtcNow.ToString("o")
        release_version = $StagedManifest.release_version
        bundle_id = $StagedManifest.bundle_id
        payload_sha256 = $StagedManifest.payload_sha256
        python_version = $Probe.python_version
        python_cache_tag = $Probe.python_cache_tag
        krasis_version = $Probe.krasis_version
        torch_version = $Probe.torch_version
        torch_cuda = $Probe.torch_cuda
    }
    $Activation |
        ConvertTo-Json -Depth 4 |
        Set-Content -Path (Join-Path $NewRuntime "activation.json") -Encoding UTF8

    Set-Content -Path $CurrentTempPath -Value $ActivationName -Encoding ASCII -NoNewline
    if (Test-Path $CurrentPath -PathType Leaf) {
        Remove-Item -Force $CurrentBackupPath -ErrorAction SilentlyContinue
        [System.IO.File]::Replace($CurrentTempPath, $CurrentPath, $CurrentBackupPath, $true)
    } else {
        [System.IO.File]::Move($CurrentTempPath, $CurrentPath)
    }
    $PointerSwitched = $true

    $ActivatedName = (Get-Content -Raw $CurrentPath).Trim()
    if ($ActivatedName -ne $ActivationName) {
        throw "Private-runtime activation pointer did not select the staged release."
    }
    [void](Assert-KrasisPrivateRuntime `
        -RuntimeRoot (Join-Path $ReleasesRoot $ActivatedName) `
        -Manifest $StagedManifest `
        -IncludeTorch)

    Remove-Item -Force $CurrentBackupPath -ErrorAction SilentlyContinue

    foreach ($LegacyPath in @(
        (Join-Path $InstallRoot "python"),
        (Join-Path $InstallRoot "venv")
    )) {
        if (Test-Path $LegacyPath) {
            if (Test-KrasisPythonTreeInUse -TreeRoot $LegacyPath) {
                Write-Warning "Legacy runtime is still in use and was retained: $LegacyPath"
            } else {
                try {
                    Remove-Item -Recurse -Force $LegacyPath
                    Write-Host "Removed legacy runtime: $LegacyPath"
                } catch {
                    Write-Warning "Could not remove inactive legacy runtime ${LegacyPath}: $($_.Exception.Message)"
                }
            }
        }
    }

    Write-Host ""
    Write-Host "Krasis private runtime is ready." -ForegroundColor Green
    Write-Host "  Python: $($Probe.python_version) ($($Probe.python_cache_tag))"
    Write-Host "  Krasis: $($Probe.krasis_version)"
    Write-Host "  PyTorch: $($Probe.torch_version), CUDA $($Probe.torch_cuda)"
} catch {
    $RollbackSucceeded = -not $PointerSwitched
    if ($PointerSwitched) {
        try {
            if (Test-Path $CurrentBackupPath -PathType Leaf) {
                [System.IO.File]::Replace($CurrentBackupPath, $CurrentPath, $null, $true)
            } elseif ($null -ne $OldCurrent) {
                Set-Content -Path $CurrentPath -Value $OldCurrent -Encoding ASCII -NoNewline
            } else {
                Remove-Item -Force $CurrentPath -ErrorAction SilentlyContinue
            }
            $RollbackValue = if (Test-Path $CurrentPath -PathType Leaf) {
                (Get-Content -Raw $CurrentPath).Trim()
            } else {
                $null
            }
            $RollbackSucceeded = $RollbackValue -ne $ActivationName
        } catch {
            Write-Warning "Private-runtime activation rollback failed: $($_.Exception.Message)"
        }
    }
    if ($RollbackSucceeded -and (Test-Path $NewRuntime)) {
        Remove-Item -Recurse -Force $NewRuntime -ErrorAction SilentlyContinue
    } elseif (Test-Path $NewRuntime) {
        Write-Warning "The newly activated runtime was retained because rollback could not be confirmed."
    }
    throw
} finally {
    Remove-Item -Force $CurrentTempPath -ErrorAction SilentlyContinue
    foreach ($Name in $SavedEnvironment.Keys) {
        $Value = $SavedEnvironment[$Name]
        if ($null -eq $Value) {
            Remove-Item "Env:$Name" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
        }
    }
}
