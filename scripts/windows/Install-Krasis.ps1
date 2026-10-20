param(
    [Parameter(Mandatory = $true)]
    [string]$InstallRoot,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeArchive,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeArchiveSha256
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "Runtime-Manifest.ps1")

$RuntimeArchivePath = (Resolve-Path $RuntimeArchive).Path
if ($RuntimeArchiveSha256 -notmatch "^[0-9a-fA-F]{64}$") {
    throw "Private-runtime archive SHA-256 is invalid: $RuntimeArchiveSha256"
}
$ActualArchiveSha256 = Get-KrasisFileSha256 -Path $RuntimeArchivePath
if ($ActualArchiveSha256 -ne $RuntimeArchiveSha256.ToLowerInvariant()) {
    throw "Private-runtime archive SHA-256 mismatch: expected $RuntimeArchiveSha256, got $ActualArchiveSha256."
}

$VersionPath = Join-Path $InstallRoot "VERSION.txt"
if (-not (Test-Path $VersionPath -PathType Leaf)) {
    throw "Krasis installer version marker is missing: $VersionPath"
}
$InstallerVersion = (Get-Content -Raw $VersionPath).Trim()

New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
$RuntimeRoot = Join-Path $InstallRoot "runtime"
$ReleasesRoot = Join-Path $RuntimeRoot "releases"
$ArtifactsRoot = Join-Path $RuntimeRoot "artifacts"
New-Item -ItemType Directory -Force -Path $ReleasesRoot | Out-Null

$StagingName = ".staging-$([Guid]::NewGuid().ToString('N'))"
$NewRuntime = Join-Path $ReleasesRoot $StagingName
$ActivationName = $null
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

function Remove-KrasisLegacyRuntimeTrees {
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
}

function Test-KrasisRuntimeManifestIdentity {
    param(
        [Parameter(Mandatory = $true)]
        $InstalledManifest,
        [Parameter(Mandatory = $true)]
        $ExpectedManifest
    )

    foreach ($PropertyName in @(
        "schema_version",
        "bundle_id",
        "release_version",
        "python_version",
        "python_runtime_archive_sha256",
        "python_cache_tag",
        "architecture",
        "krasis_version",
        "torch_version",
        "torch_cuda",
        "torch_url",
        "payload_sha256"
    )) {
        $InstalledProperty = $InstalledManifest.PSObject.Properties[$PropertyName]
        $ExpectedProperty = $ExpectedManifest.PSObject.Properties[$PropertyName]
        if ($null -eq $InstalledProperty -or $null -eq $ExpectedProperty) {
            return $false
        }
        if (-not [string]::Equals(
            "$($InstalledProperty.Value)",
            "$($ExpectedProperty.Value)",
            [StringComparison]::Ordinal
        )) {
            return $false
        }
    }
    return $true
}

function Get-KrasisReusableRuntime {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CurrentPath,
        [Parameter(Mandatory = $true)]
        [string]$ReleasesRoot,
        [Parameter(Mandatory = $true)]
        $ExpectedManifest
    )

    if (-not (Test-Path $CurrentPath -PathType Leaf)) {
        return $null
    }

    try {
        $CurrentName = (Get-Content -Raw $CurrentPath).Trim()
        if ($CurrentName -notmatch "^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$") {
            throw "activation pointer is invalid"
        }
        $InstalledRoot = Join-Path $ReleasesRoot $CurrentName
        $InstalledManifest = Read-KrasisRuntimeManifest -RuntimeRoot $InstalledRoot
        if (-not (Test-KrasisRuntimeManifestIdentity `
            -InstalledManifest $InstalledManifest `
            -ExpectedManifest $ExpectedManifest)) {
            return $null
        }

        $ActivationPath = Join-Path $InstalledRoot "activation.json"
        if (-not (Test-Path $ActivationPath -PathType Leaf)) {
            throw "activation record is missing"
        }
        $Activation = Get-Content -Raw $ActivationPath | ConvertFrom-Json
        if (
            [int]$Activation.schema_version -ne 1 -or
            "$($Activation.release_version)" -ne "$($ExpectedManifest.release_version)" -or
            "$($Activation.bundle_id)" -ne "$($ExpectedManifest.bundle_id)" -or
            "$($Activation.payload_sha256)" -ne "$($ExpectedManifest.payload_sha256)" -or
            "$($Activation.python_version)" -ne "$($ExpectedManifest.python_version)" -or
            "$($Activation.python_cache_tag)" -ne "$($ExpectedManifest.python_cache_tag)" -or
            "$($Activation.krasis_version)" -ne "$($ExpectedManifest.krasis_version)" -or
            "$($Activation.torch_version)" -ne "$($ExpectedManifest.torch_version)" -or
            "$($Activation.torch_cuda)" -ne "$($ExpectedManifest.torch_cuda)"
        ) {
            throw "activation record does not match the runtime manifest"
        }

        $Probe = Assert-KrasisPrivateRuntime `
            -RuntimeRoot $InstalledRoot `
            -Manifest $InstalledManifest `
            -IncludeTorch
        return [PSCustomObject]@{
            name = $CurrentName
            root = $InstalledRoot
            probe = $Probe
        }
    } catch {
        Write-Host (
            "Existing Krasis runtime cannot be reused and will be repaired: " +
            $_.Exception.Message
        ) -ForegroundColor Yellow
        return $null
    }
}

function Get-KrasisTorchArtifact {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PrivatePython,
        [Parameter(Mandatory = $true)]
        [string]$ArtifactsRoot,
        [Parameter(Mandatory = $true)]
        $Manifest
    )

    $TorchUri = [Uri]"$($Manifest.torch_url)"
    if ($TorchUri.Fragment -notmatch "^#sha256=([0-9a-f]{64})$") {
        throw "Pinned CUDA PyTorch URL does not contain a valid SHA-256 fragment."
    }
    $TorchSha256 = $Matches[1]
    $TorchFileName = [Uri]::UnescapeDataString(
        [IO.Path]::GetFileName($TorchUri.AbsolutePath)
    )
    if ($TorchFileName -notmatch "^[A-Za-z0-9._+-]+\.whl$") {
        throw "Pinned CUDA PyTorch wheel filename is invalid: $TorchFileName"
    }

    $TorchArtifactsRoot = Join-Path $ArtifactsRoot "torch"
    New-Item -ItemType Directory -Force -Path $TorchArtifactsRoot | Out-Null
    $TorchArtifactRoot = Join-Path $TorchArtifactsRoot $TorchSha256
    New-Item -ItemType Directory -Force -Path $TorchArtifactRoot | Out-Null
    $TorchArtifactPath = Join-Path $TorchArtifactRoot $TorchFileName
    if (Test-Path $TorchArtifactPath -PathType Leaf) {
        $CachedSha256 = Get-KrasisFileSha256 -Path $TorchArtifactPath
        if ($CachedSha256 -ne $TorchSha256) {
            throw (
                "Cached CUDA PyTorch wheel SHA-256 mismatch: expected " +
                "$TorchSha256, got $CachedSha256. Remove the invalid artifact " +
                "and rerun the installer: $TorchArtifactPath"
            )
        }
        Write-Host "Using verified cached CUDA PyTorch wheel: $TorchArtifactPath"
        return $TorchArtifactPath
    }

    $DownloadRoot = Join-Path $TorchArtifactsRoot ".download-$([Guid]::NewGuid().ToString('N'))"
    try {
        New-Item -ItemType Directory -Force -Path $DownloadRoot | Out-Null
        Write-Host "Downloading pinned CUDA PyTorch $($Manifest.torch_version)..."
        Write-Host "This is a large one-time download and can take several minutes."
        & $PrivatePython -I -B -m pip download `
            --no-cache-dir `
            --disable-pip-version-check `
            --no-deps `
            --only-binary ":all:" `
            --dest $DownloadRoot `
            "$($Manifest.torch_url)" | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Pinned CUDA PyTorch download failed with status $LASTEXITCODE."
        }

        $DownloadedWheels = @(Get-ChildItem -Path $DownloadRoot -File -Filter "*.whl")
        if ($DownloadedWheels.Count -ne 1) {
            throw "Pinned CUDA PyTorch download produced $($DownloadedWheels.Count) wheel files instead of one."
        }
        $DownloadedSha256 = Get-KrasisFileSha256 -Path $DownloadedWheels[0].FullName
        if ($DownloadedSha256 -ne $TorchSha256) {
            throw "Downloaded CUDA PyTorch wheel SHA-256 mismatch: expected $TorchSha256, got $DownloadedSha256."
        }
        [IO.File]::Move($DownloadedWheels[0].FullName, $TorchArtifactPath)
        return $TorchArtifactPath
    } finally {
        Remove-Item -Recurse -Force $DownloadRoot -ErrorAction SilentlyContinue
    }
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

    Write-Host "Extracting verified Krasis private runtime..."
    New-Item -ItemType Directory -Force -Path $NewRuntime | Out-Null
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory(
        $RuntimeArchivePath,
        $NewRuntime
    )

    $StagedManifest = Read-KrasisRuntimeManifest -RuntimeRoot $NewRuntime
    if ($InstallerVersion -ne $StagedManifest.release_version) {
        throw "Installer/runtime version mismatch: installer=$InstallerVersion, runtime=$($StagedManifest.release_version)."
    }
    $SafeBundleId = "$($StagedManifest.bundle_id)"
    if ($SafeBundleId -notmatch "^[A-Za-z0-9._-]+$") {
        throw "Private-runtime bundle ID contains unsafe characters: $SafeBundleId"
    }
    [void](Assert-KrasisPrivateRuntime -RuntimeRoot $NewRuntime -Manifest $StagedManifest)

    $ReusableRuntime = Get-KrasisReusableRuntime `
        -CurrentPath $CurrentPath `
        -ReleasesRoot $ReleasesRoot `
        -ExpectedManifest $StagedManifest
    if ($null -ne $ReusableRuntime) {
        Remove-Item -Recurse -Force $NewRuntime
        Remove-KrasisLegacyRuntimeTrees
        Write-Host ""
        Write-Host (
            "Krasis private runtime already matches this installer; " +
            "reusing the validated installation."
        ) -ForegroundColor Green
        Write-Host "  Python: $($ReusableRuntime.probe.python_version) ($($ReusableRuntime.probe.python_cache_tag))"
        Write-Host "  Krasis: $($ReusableRuntime.probe.krasis_version)"
        Write-Host "  PyTorch: $($ReusableRuntime.probe.torch_version), CUDA $($ReusableRuntime.probe.torch_cuda)"
        Write-Host "  Runtime: $($ReusableRuntime.root)"
        return
    }

    $ActivationName = "$SafeBundleId-$([Guid]::NewGuid().ToString('N'))"
    $ActivatedRuntime = Join-Path $ReleasesRoot $ActivationName
    [System.IO.Directory]::Move($NewRuntime, $ActivatedRuntime)
    $NewRuntime = $ActivatedRuntime

    $PrivatePython = Join-Path $NewRuntime "python.exe"
    $TorchArtifactPath = Get-KrasisTorchArtifact `
        -PrivatePython $PrivatePython `
        -ArtifactsRoot $ArtifactsRoot `
        -Manifest $StagedManifest
    Write-Host (
        "Installing pinned CUDA PyTorch $($StagedManifest.torch_version) " +
        "into the new Krasis runtime..."
    )
    & $PrivatePython -I -B -m pip install `
        --no-cache-dir `
        --disable-pip-version-check `
        --no-deps `
        --no-index `
        --only-binary ":all:" `
        $TorchArtifactPath
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

    Remove-KrasisLegacyRuntimeTrees

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
            $RollbackSucceeded = (
                $null -eq $ActivationName -or
                $RollbackValue -ne $ActivationName
            )
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
