param(
    [ValidateSet("stable", "prerelease")]
    [string]$Channel = "stable",
    [switch]$PauseOnFailure
)

$ErrorActionPreference = "Stop"
$Repository = "brontoguana/krasis"
$ApiRoot = "https://api.github.com/repos/$Repository"
$Headers = @{
    "Accept" = "application/vnd.github+json"
    "User-Agent" = "Krasis-Windows-Updater"
    "X-GitHub-Api-Version" = "2022-11-28"
}

function Get-KrasisRelease {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RequestedChannel
    )

    if ($RequestedChannel -eq "stable") {
        return Invoke-RestMethod `
            -Uri "$ApiRoot/releases/latest" `
            -Headers $Headers `
            -Method Get
    }

    $Releases = @(
        Invoke-RestMethod `
            -Uri "$ApiRoot/releases?per_page=30" `
            -Headers $Headers `
            -Method Get
    )
    $Release = $Releases |
        Where-Object { $_.prerelease -and -not $_.draft } |
        Select-Object -First 1
    if (-not $Release) {
        throw "No published Krasis prerelease was found."
    }
    return $Release
}

function Get-WindowsInstallerAsset {
    param(
        [Parameter(Mandatory = $true)]
        $Release
    )

    $Assets = @(
        $Release.assets |
            Where-Object { $_.name -match "^KrasisSetup-.+-win64\.exe$" }
    )
    if ($Assets.Count -eq 0) {
        throw "Release $($Release.tag_name) does not contain a Krasis Windows installer."
    }
    if ($Assets.Count -gt 1) {
        throw "Release $($Release.tag_name) contains multiple matching Windows installers."
    }
    return $Assets[0]
}

$ChannelLabel = if ($Channel -eq "prerelease") {
    "prerelease"
} else {
    "stable release"
}
$TempDir = Join-Path `
    ([System.IO.Path]::GetTempPath()) `
    ("krasis-update-" + [Guid]::NewGuid().ToString("N"))

try {
    [Net.ServicePointManager]::SecurityProtocol = `
        [Net.ServicePointManager]::SecurityProtocol -bor `
        [Net.SecurityProtocolType]::Tls12

    Write-Host "Checking for the latest Krasis $ChannelLabel..." -ForegroundColor Cyan
    $Release = Get-KrasisRelease -RequestedChannel $Channel
    $Asset = Get-WindowsInstallerAsset -Release $Release

    New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
    $InstallerPath = Join-Path $TempDir $Asset.name
    Write-Host "Downloading $($Release.tag_name): $($Asset.name)"

    $SavedProgressPreference = $ProgressPreference
    try {
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest `
            -Uri $Asset.browser_download_url `
            -Headers $Headers `
            -OutFile $InstallerPath `
            -UseBasicParsing
    } finally {
        $ProgressPreference = $SavedProgressPreference
    }

    if (-not (Test-Path $InstallerPath)) {
        throw "The Windows installer download did not create $InstallerPath."
    }
    $DownloadedSize = (Get-Item $InstallerPath).Length
    if ($DownloadedSize -ne [Int64]$Asset.size) {
        throw "The Windows installer download was incomplete: expected $($Asset.size) bytes, received $DownloadedSize."
    }

    Write-Host "Starting the Krasis $($Release.tag_name) installer..." -ForegroundColor Cyan
    $Installer = Start-Process `
        -FilePath $InstallerPath `
        -ArgumentList "/SP-" `
        -PassThru `
        -Wait
    if ($Installer.ExitCode -ne 0) {
        throw "The Krasis installer exited with status $($Installer.ExitCode)."
    }

    Write-Host "Krasis $($Release.tag_name) is installed." -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "Krasis update failed: $($_.Exception.Message)" -ForegroundColor Red
    if ($PauseOnFailure) {
        try {
            [void](Read-Host "Press Enter to close")
        } catch {
            # If input is unavailable, preserve the original update failure.
        }
    }
    exit 1
} finally {
    if (Test-Path $TempDir) {
        Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
    }
}
