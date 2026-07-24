param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimePackage,
    [string]$OutputDir = "dist",
    [string]$Version = "0.0.0"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RuntimePackagePath = (Resolve-Path $RuntimePackage).Path
$OutputPath = Join-Path $RepoRoot $OutputDir
$BuildRoot = Join-Path $RepoRoot "target\windows-installer"
$Stage = Join-Path $BuildRoot "staging"

Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Join-Path $Stage "bin") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Stage "runtime-package") | Out-Null
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

Copy-Item -Force (Join-Path $PSScriptRoot "Launch-Krasis.ps1") (Join-Path $Stage "bin\Launch-Krasis.ps1")
Copy-Item -Force (Join-Path $PSScriptRoot "Install-Krasis.ps1") (Join-Path $Stage "bin\Install-Krasis.ps1")
Copy-Item -Force (Join-Path $PSScriptRoot "Invoke-Install-Krasis.ps1") (Join-Path $Stage "bin\Invoke-Install-Krasis.ps1")
Copy-Item -Force (Join-Path $PSScriptRoot "Update-Krasis.ps1") (Join-Path $Stage "bin\Update-Krasis.ps1")
Copy-Item -Force (Join-Path $PSScriptRoot "Runtime-Manifest.ps1") (Join-Path $Stage "bin\Runtime-Manifest.ps1")
Copy-Item -Recurse -Force (Join-Path $RuntimePackagePath "*") (Join-Path $Stage "runtime-package")
Set-Content -Path (Join-Path $Stage "VERSION.txt") -Value $Version -Encoding ASCII

$ManifestPath = Join-Path $Stage "runtime-package\runtime-manifest.json"
if (-not (Test-Path $ManifestPath -PathType Leaf)) {
    throw "Private-runtime package manifest was not staged: $ManifestPath"
}
$Manifest = Get-Content -Raw $ManifestPath | ConvertFrom-Json
if ($Manifest.release_version -ne $Version) {
    throw "Private-runtime release version $($Manifest.release_version) does not match installer version $Version."
}

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
