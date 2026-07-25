param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimePackage,
    [Parameter(Mandatory = $true)]
    [string]$LauncherExe,
    [string]$OutputDir = "dist",
    [string]$Version = "0.0.0"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. (Join-Path $PSScriptRoot "Runtime-Manifest.ps1")
$RuntimePackagePath = (Resolve-Path $RuntimePackage).Path
$LauncherExePath = (Resolve-Path $LauncherExe).Path
$LauncherIconPath = (Resolve-Path (Join-Path $RepoRoot "assets\windows\krasis.ico")).Path
$OutputPath = Join-Path $RepoRoot $OutputDir
$BuildRoot = Join-Path $RepoRoot "target\windows-installer"
$Stage = Join-Path $BuildRoot "staging"

Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Join-Path $Stage "bin") | Out-Null
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

Copy-Item -Force $LauncherExePath (Join-Path $Stage "bin\Krasis.exe")
Copy-Item -Force $LauncherIconPath (Join-Path $Stage "bin\Krasis.ico")
Copy-Item -Force (Join-Path $PSScriptRoot "Install-Krasis.ps1") (Join-Path $Stage "bin\Install-Krasis.ps1")
Copy-Item -Force (Join-Path $PSScriptRoot "Invoke-Install-Krasis.ps1") (Join-Path $Stage "bin\Invoke-Install-Krasis.ps1")
Copy-Item -Force (Join-Path $PSScriptRoot "Update-Krasis.ps1") (Join-Path $Stage "bin\Update-Krasis.ps1")
Copy-Item -Force (Join-Path $PSScriptRoot "Runtime-Manifest.ps1") (Join-Path $Stage "bin\Runtime-Manifest.ps1")
Copy-Item -Force (Join-Path $PSScriptRoot "Remove-KrasisRuntime.ps1") (Join-Path $Stage "bin\Remove-KrasisRuntime.ps1")
Set-Content -Path (Join-Path $Stage "VERSION.txt") -Value $Version -Encoding ASCII

$ManifestPath = Join-Path $RuntimePackagePath "runtime-manifest.json"
if (-not (Test-Path $ManifestPath -PathType Leaf)) {
    throw "Private-runtime package manifest is missing: $ManifestPath"
}
$Manifest = Read-KrasisRuntimeManifest -RuntimeRoot $RuntimePackagePath
if ($Manifest.release_version -ne $Version) {
    throw "Private-runtime release version $($Manifest.release_version) does not match installer version $Version."
}
$PayloadHash = Get-KrasisRuntimePayloadHash -RuntimeRoot $RuntimePackagePath
if ($PayloadHash -ne $Manifest.payload_sha256) {
    throw "Private-runtime payload changed before installer packaging."
}
[void](Assert-KrasisPrivateRuntime -RuntimeRoot $RuntimePackagePath -Manifest $Manifest)

$RuntimeArchive = Join-Path $Stage "runtime-package.zip"
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $RuntimePackagePath,
    $RuntimeArchive,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false
)
$RuntimeArchiveSha256 = Get-KrasisFileSha256 -Path $RuntimeArchive
if ($RuntimeArchiveSha256 -notmatch "^[0-9a-f]{64}$") {
    throw "Private-runtime archive SHA-256 is invalid: $RuntimeArchiveSha256"
}
Write-Host "Private-runtime archive SHA-256: $RuntimeArchiveSha256"

$IsccPath = $null
$Iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($Iscc) {
    $IsccPath = $Iscc.Source
}
if (-not $IsccPath) {
    $Candidate = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (Test-Path $Candidate) {
        $IsccPath = $Candidate
    }
}

if ($IsccPath) {
    $env:KRASIS_WINDOWS_STAGING = $Stage
    $env:KRASIS_WINDOWS_OUTPUT = $OutputPath
    $env:KRASIS_WINDOWS_VERSION = $Version
    $env:KRASIS_RUNTIME_ARCHIVE_SHA256 = $RuntimeArchiveSha256
    & $IsccPath (Join-Path $PSScriptRoot "KrasisInstaller.iss")
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup failed with exit code $LASTEXITCODE"
    }
} else {
    $ZipPath = Join-Path $OutputPath "Krasis-Windows-Package-$Version.zip"
    Remove-Item -Force $ZipPath -ErrorAction SilentlyContinue
    Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $ZipPath
    Write-Host "ISCC.exe not found; wrote fallback package $ZipPath"
}
