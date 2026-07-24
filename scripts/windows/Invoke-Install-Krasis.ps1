param(
    [Parameter(Mandatory = $true)]
    [string]$InstallScript,
    [Parameter(Mandatory = $true)]
    [string]$InstallRoot,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeArchive,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeArchiveSha256,
    [Parameter(Mandatory = $true)]
    [string]$LogPath
)

$ErrorActionPreference = "Stop"
$ExitCode = 0
$TranscriptStarted = $false

try {
    $LogDirectory = Split-Path -Parent $LogPath
    if (-not [string]::IsNullOrWhiteSpace($LogDirectory)) {
        New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
    }
    Start-Transcript -Path $LogPath -Force | Out-Null
    $TranscriptStarted = $true
    & $InstallScript `
        -InstallRoot $InstallRoot `
        -RuntimeArchive $RuntimeArchive `
        -RuntimeArchiveSha256 $RuntimeArchiveSha256
} catch {
    $ExitCode = 1
    Write-Host "Krasis private-runtime installation failed:" -ForegroundColor Red
    Write-Host ($_ | Out-String) -ForegroundColor Red
} finally {
    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
}

exit $ExitCode
