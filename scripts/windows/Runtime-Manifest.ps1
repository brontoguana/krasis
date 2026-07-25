function Get-KrasisFileSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $ResolvedPath = (Resolve-Path $Path).Path
    $Sha256 = [System.Security.Cryptography.SHA256]::Create()
    $Stream = $null
    try {
        $Stream = [System.IO.File]::OpenRead($ResolvedPath)
        $Hash = $Sha256.ComputeHash($Stream)
        return ([BitConverter]::ToString($Hash)).Replace("-", "").ToLowerInvariant()
    } finally {
        if ($null -ne $Stream) {
            $Stream.Dispose()
        }
        $Sha256.Dispose()
    }
}

function Get-KrasisRuntimePayloadHash {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RuntimeRoot
    )

    $ResolvedRoot = (Resolve-Path $RuntimeRoot).Path
    $Utf8 = New-Object System.Text.UTF8Encoding($false)
    $Aggregate = [System.Security.Cryptography.SHA256]::Create()
    try {
        $Files = @(
            Get-ChildItem -Path $ResolvedRoot -Recurse -File |
                ForEach-Object {
                    $Relative = $_.FullName.Substring($ResolvedRoot.Length).TrimStart("\", "/")
                    $Relative = $Relative.Replace("\", "/")
                    [PSCustomObject]@{
                        Relative = $Relative
                        FullName = $_.FullName
                        Length = $_.Length
                    }
                } |
                Where-Object {
                    $_.Relative -notin @("runtime-manifest.json", "activation.json")
                }
        )

        [string[]]$Records = @(
            foreach ($File in $Files) {
                $FileHash = Get-KrasisFileSha256 -Path $File.FullName
                "$($File.Relative)`t$($File.Length)`t$FileHash`n"
            }
        )
        [Array]::Sort($Records, [StringComparer]::Ordinal)
        foreach ($Record in $Records) {
            $Bytes = $Utf8.GetBytes($Record)
            [void]$Aggregate.TransformBlock($Bytes, 0, $Bytes.Length, $Bytes, 0)
        }
        [void]$Aggregate.TransformFinalBlock([byte[]]@(), 0, 0)
        return ([BitConverter]::ToString($Aggregate.Hash)).Replace("-", "").ToLowerInvariant()
    } finally {
        $Aggregate.Dispose()
    }
}

function Read-KrasisRuntimeManifest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RuntimeRoot
    )

    $ManifestPath = Join-Path $RuntimeRoot "runtime-manifest.json"
    if (-not (Test-Path $ManifestPath -PathType Leaf)) {
        throw "Krasis private-runtime manifest is missing: $ManifestPath"
    }
    $Manifest = Get-Content -Raw $ManifestPath | ConvertFrom-Json
    if ([int]$Manifest.schema_version -ne 1) {
        throw "Unsupported Krasis private-runtime manifest schema: $($Manifest.schema_version)"
    }
    if ("$($Manifest.payload_sha256)" -notmatch "^[0-9a-f]{64}$") {
        throw "Krasis private-runtime manifest contains an invalid payload SHA-256."
    }
    if ("$($Manifest.python_runtime_archive_sha256)" -notmatch "^[0-9a-f]{64}$") {
        throw "Krasis private-runtime manifest contains an invalid CPython runtime archive SHA-256."
    }
    if ("$($Manifest.torch_url)" -notmatch "^https://.+#sha256=[0-9a-f]{64}$") {
        throw "Krasis private-runtime manifest does not contain a hash-bound HTTPS PyTorch wheel URL."
    }
    return $Manifest
}

function Get-KrasisRuntimeProbe {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Python,
        [switch]$IncludeTorch
    )

    if (-not (Test-Path $Python -PathType Leaf)) {
        throw "Krasis private Python executable is missing: $Python"
    }

    $ProbeCode = @'
import importlib
import importlib.metadata
import json
import os
import platform
import re
import site
import ssl
import struct
import sys
import tempfile

native = importlib.import_module("krasis.krasis")
launcher = importlib.import_module("krasis.launcher")
chat = importlib.import_module("krasis.chat")
console_input = importlib.import_module("krasis.console_input")

def decode_windows_key(chars):
    chars = iter(chars)
    return console_input.read_windows_key(lambda: next(chars))

def probe_console_mode():
    modes = [0x0010 | 0x0040 | 0x0020]
    writes = []
    def set_mode(mode):
        writes.append(mode)
        modes[0] = mode
    with console_input.windows_console_key_mode(
        get_mode=lambda: modes[0],
        set_mode=set_mode,
    ):
        active = modes[0]
    return {
        "active": active,
        "restored": modes[0],
        "writes": writes,
    }

def probe_launch_config():
    fd, path = tempfile.mkstemp(prefix="krasis-runtime-probe-", suffix=".conf")
    try:
        launcher._write_launch_config(fd, {"MODEL_PATH": r"C:\Krasis\模型"})
        raw = open(path, "rb").read()
        text = raw.decode("utf-8")
        return {
            "strict_utf8": True,
            "has_utf8_dash": b"\xe2\x80\x94" in raw,
            "has_non_ascii_value": r'MODEL_PATH="C:\Krasis\模型"' in text,
        }
    finally:
        os.unlink(path)

data = {
    "python_version": platform.python_version(),
    "python_cache_tag": sys.implementation.cache_tag,
    "architecture": platform.machine(),
    "pointer_bits": struct.calcsize("P") * 8,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "isolated": sys.flags.isolated,
    "ignore_environment": sys.flags.ignore_environment,
    "user_site_enabled": bool(site.ENABLE_USER_SITE),
    "sys_path": sys.path,
    "krasis_version": importlib.metadata.version("krasis"),
    "native_module": native.__name__,
    "launcher_windows_console": bool(launcher._HAS_WINDOWS_CONSOLE),
    "chat_windows_console": bool(chat._HAS_WINDOWS_CONSOLE),
    "windows_console_key_probe": [
        decode_windows_key(("\xe0", "H")),
        decode_windows_key(("\xe0", "P")),
        decode_windows_key(("\x00", "K")),
        decode_windows_key(("\x00", "M")),
        decode_windows_key(("\r",)),
        decode_windows_key(("\x1b",)),
        decode_windows_key(("\x08",)),
    ],
    "windows_console_mode_probe": probe_console_mode(),
    "launch_config_probe": probe_launch_config(),
    "ssl_version": ssl.OPENSSL_VERSION,
    "sre_magic": getattr(importlib.import_module("_sre"), "MAGIC", None),
    "regex_probe": bool(re.fullmatch(r"Krasis-[0-9]+", "Krasis-312")),
}
if os.environ.get("KRASIS_RUNTIME_PROBE_TORCH") == "1":
    torch = importlib.import_module("torch")
    data["torch_version"] = torch.__version__
    data["torch_cuda"] = torch.version.cuda
print("KRASIS_RUNTIME_PROBE=" + json.dumps(data, sort_keys=True))
'@

    $OldProbeTorch = [Environment]::GetEnvironmentVariable("KRASIS_RUNTIME_PROBE_TORCH", "Process")
    $Process = $null
    try {
        $env:KRASIS_RUNTIME_PROBE_TORCH = if ($IncludeTorch) { "1" } else { "0" }
        # Windows PowerShell 5.1 applies legacy native-command quoting to -c
        # arguments and can corrupt Python source containing quotes. Send the
        # probe through stdin using .NET so neither PowerShell nor cmd.exe
        # parses the source text.
        $StartInfo = New-Object System.Diagnostics.ProcessStartInfo
        $StartInfo.FileName = (Resolve-Path $Python).Path
        $StartInfo.Arguments = "-I -B -"
        $StartInfo.UseShellExecute = $false
        $StartInfo.CreateNoWindow = $true
        $StartInfo.RedirectStandardInput = $true
        $StartInfo.RedirectStandardOutput = $true
        $StartInfo.RedirectStandardError = $true

        $Process = New-Object System.Diagnostics.Process
        $Process.StartInfo = $StartInfo
        if (-not $Process.Start()) {
            throw "Could not start Krasis private Python runtime probe."
        }
        $StandardOutputTask = $Process.StandardOutput.ReadToEndAsync()
        $StandardErrorTask = $Process.StandardError.ReadToEndAsync()
        $Process.StandardInput.Write($ProbeCode)
        $Process.StandardInput.Close()
        $Process.WaitForExit()
        $StandardOutput = $StandardOutputTask.Result
        $StandardError = $StandardErrorTask.Result
        $ExitCode = $Process.ExitCode
        $Output = @()
        if (-not [string]::IsNullOrWhiteSpace($StandardOutput)) {
            $Output += @($StandardOutput -split "\r?\n")
        }
        if (-not [string]::IsNullOrWhiteSpace($StandardError)) {
            $Output += @($StandardError -split "\r?\n")
        }
    } finally {
        if ($null -ne $Process) {
            $Process.Dispose()
        }
        if ($null -eq $OldProbeTorch) {
            Remove-Item Env:KRASIS_RUNTIME_PROBE_TORCH -ErrorAction SilentlyContinue
        } else {
            $env:KRASIS_RUNTIME_PROBE_TORCH = $OldProbeTorch
        }
    }

    if ($ExitCode -ne 0) {
        throw "Krasis private-runtime probe failed with status ${ExitCode}: $($Output -join [Environment]::NewLine)"
    }
    $ProbeLine = $Output |
        Where-Object { "$_".StartsWith("KRASIS_RUNTIME_PROBE=") } |
        Select-Object -Last 1
    if (-not $ProbeLine) {
        throw "Krasis private-runtime probe did not return structured output: $($Output -join [Environment]::NewLine)"
    }
    return "$ProbeLine".Substring("KRASIS_RUNTIME_PROBE=".Length) | ConvertFrom-Json
}

function Assert-KrasisPrivateRuntime {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RuntimeRoot,
        [Parameter(Mandatory = $true)]
        $Manifest,
        [switch]$IncludeTorch
    )

    $ResolvedRoot = (Resolve-Path $RuntimeRoot).Path.TrimEnd("\", "/")
    $Python = Join-Path $ResolvedRoot "python.exe"
    $Probe = Get-KrasisRuntimeProbe -Python $Python -IncludeTorch:$IncludeTorch

    if ($Probe.python_version -ne $Manifest.python_version) {
        throw "Private Python version mismatch: expected $($Manifest.python_version), got $($Probe.python_version)."
    }
    if ($Probe.python_cache_tag -ne $Manifest.python_cache_tag) {
        throw "Private Python ABI mismatch: expected $($Manifest.python_cache_tag), got $($Probe.python_cache_tag)."
    }
    if ($Probe.architecture -ne $Manifest.architecture -or [int]$Probe.pointer_bits -ne 64) {
        throw "Private Python architecture mismatch: expected $($Manifest.architecture)/64-bit, got $($Probe.architecture)/$($Probe.pointer_bits)-bit."
    }
    if ($Probe.krasis_version -ne $Manifest.krasis_version) {
        throw "Private Krasis version mismatch: expected $($Manifest.krasis_version), got $($Probe.krasis_version)."
    }
    if ($Probe.native_module -ne "krasis.krasis") {
        throw "Krasis native extension did not import from the private runtime."
    }
    if (-not [bool]$Probe.launcher_windows_console -or -not [bool]$Probe.chat_windows_console) {
        throw "Krasis launcher/chat did not enable native Windows console input."
    }
    $ExpectedKeys = @("UP", "DOWN", "LEFT", "RIGHT", "ENTER", "ESC", "BACKSPACE")
    $ActualKeyProbe = [string]::Join(",", @($Probe.windows_console_key_probe))
    $ExpectedKeyProbe = [string]::Join(",", $ExpectedKeys)
    if ($ActualKeyProbe -ne $ExpectedKeyProbe) {
        throw "Krasis native Windows console key decoder failed its runtime probe."
    }
    if (
        ([int]$Probe.windows_console_mode_probe.active -band 0x0010) -ne 0 -or
        ([int]$Probe.windows_console_mode_probe.active -band 0x0040) -ne 0 -or
        ([int]$Probe.windows_console_mode_probe.active -band 0x0080) -eq 0 -or
        [int]$Probe.windows_console_mode_probe.restored -ne (0x0010 -bor 0x0040 -bor 0x0020)
    ) {
        throw "Krasis native Windows console mode isolation/restoration probe failed."
    }
    if (
        -not [bool]$Probe.launch_config_probe.strict_utf8 -or
        -not [bool]$Probe.launch_config_probe.has_utf8_dash -or
        -not [bool]$Probe.launch_config_probe.has_non_ascii_value
    ) {
        throw "Krasis launcher generated-config UTF-8 probe failed."
    }
    if (-not $Probe.regex_probe) {
        throw "Private Python regex/stdlib validation failed."
    }
    if ([int]$Probe.isolated -ne 1 -or [int]$Probe.ignore_environment -ne 1) {
        throw "Private Python is not running in isolated, environment-ignoring mode."
    }
    if ([bool]$Probe.user_site_enabled) {
        throw "Private Python unexpectedly enabled user site-packages."
    }

    $ProbePrefix = [System.IO.Path]::GetFullPath("$($Probe.prefix)").TrimEnd("\", "/")
    $ProbeBasePrefix = [System.IO.Path]::GetFullPath("$($Probe.base_prefix)").TrimEnd("\", "/")
    if (-not $ProbePrefix.Equals($ResolvedRoot, [StringComparison]::OrdinalIgnoreCase) -or
        -not $ProbeBasePrefix.Equals($ResolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Private Python resolved outside its runtime root: prefix=$ProbePrefix, base_prefix=$ProbeBasePrefix."
    }
    foreach ($Entry in @($Probe.sys_path)) {
        if ([string]::IsNullOrWhiteSpace("$Entry")) {
            continue
        }
        $ResolvedEntry = [System.IO.Path]::GetFullPath("$Entry")
        if (-not $ResolvedEntry.StartsWith(
            $ResolvedRoot + [System.IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase
        ) -and -not $ResolvedEntry.Equals($ResolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Private Python imported a path outside its runtime root: $ResolvedEntry"
        }
    }

    if ($IncludeTorch) {
        if ($Probe.torch_version -ne $Manifest.torch_version) {
            throw "Private PyTorch version mismatch: expected $($Manifest.torch_version), got $($Probe.torch_version)."
        }
        if ($Probe.torch_cuda -ne $Manifest.torch_cuda) {
            throw "Private PyTorch CUDA build mismatch: expected $($Manifest.torch_cuda), got $($Probe.torch_cuda)."
        }
    }
    return $Probe
}
