param(
    [Parameter(Mandatory = $true)]
    [string]$InstallRoot
)

$ErrorActionPreference = "Stop"

function Test-KrasisRuntimeInUse {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RuntimeRoot
    )

    try {
        $PythonProcesses = @(
            Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction Stop
        )
    } catch {
        throw "Could not inspect running Python processes; refusing to remove the Krasis runtime."
    }

    $ResolvedRuntime = [System.IO.Path]::GetFullPath($RuntimeRoot).TrimEnd("\", "/")
    foreach ($Process in $PythonProcesses) {
        $Executable = "$($Process.ExecutablePath)"
        $CommandLine = "$($Process.CommandLine)"
        if ([string]::IsNullOrWhiteSpace($Executable) -and
            [string]::IsNullOrWhiteSpace($CommandLine)) {
            throw "A running Python process could not be inspected; refusing to remove the Krasis runtime."
        }
        if ($Executable.StartsWith(
            $ResolvedRuntime + [System.IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase
        ) -or $CommandLine.IndexOf(
            $ResolvedRuntime,
            [StringComparison]::OrdinalIgnoreCase
        ) -ge 0) {
            return $true
        }
    }
    return $false
}

$LongPathDeleteSource = @'
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

public static class KrasisLongPathDelete
{
    private const int ERROR_FILE_NOT_FOUND = 2;
    private const int ERROR_PATH_NOT_FOUND = 3;
    private const int ERROR_ACCESS_DENIED = 5;
    private const int ERROR_NO_MORE_FILES = 18;
    private const int ERROR_SHARING_VIOLATION = 32;
    private const int ERROR_DIR_NOT_EMPTY = 145;
    private const uint FILE_ATTRIBUTE_DIRECTORY = 0x10;
    private const uint FILE_ATTRIBUTE_REPARSE_POINT = 0x400;
    private const uint FILE_ATTRIBUTE_NORMAL = 0x80;
    private const uint INVALID_FILE_ATTRIBUTES = 0xffffffff;
    private static readonly IntPtr InvalidHandleValue = new IntPtr(-1);

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct WIN32_FIND_DATA
    {
        public uint dwFileAttributes;
        public System.Runtime.InteropServices.ComTypes.FILETIME ftCreationTime;
        public System.Runtime.InteropServices.ComTypes.FILETIME ftLastAccessTime;
        public System.Runtime.InteropServices.ComTypes.FILETIME ftLastWriteTime;
        public uint nFileSizeHigh;
        public uint nFileSizeLow;
        public uint dwReserved0;
        public uint dwReserved1;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
        public string cFileName;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 14)]
        public string cAlternateFileName;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr FindFirstFileW(
        string lpFileName,
        out WIN32_FIND_DATA lpFindFileData
    );

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool FindNextFileW(
        IntPtr hFindFile,
        out WIN32_FIND_DATA lpFindFileData
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool FindClose(IntPtr hFindFile);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool DeleteFileW(string lpFileName);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool RemoveDirectoryW(string lpPathName);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetFileAttributesW(
        string lpFileName,
        uint dwFileAttributes
    );

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern uint GetFileAttributesW(string lpFileName);

    public static void DeleteTree(string path)
    {
        string fullPath = Path.GetFullPath(path).TrimEnd('\\', '/');
        string extendedPath;
        if (fullPath.StartsWith(@"\\", StringComparison.Ordinal))
        {
            extendedPath = @"\\?\UNC\" + fullPath.Substring(2);
        }
        else
        {
            extendedPath = @"\\?\" + fullPath;
        }
        uint attributes = GetFileAttributesW(extendedPath);
        if (attributes == INVALID_FILE_ATTRIBUTES)
        {
            int error = Marshal.GetLastWin32Error();
            if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND)
            {
                return;
            }
            throw Error("inspect", extendedPath, error);
        }
        if ((attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0)
        {
            if ((attributes & FILE_ATTRIBUTE_DIRECTORY) != 0)
            {
                RemoveDirectory(extendedPath);
            }
            else
            {
                DeleteFile(extendedPath);
            }
            return;
        }
        if ((attributes & FILE_ATTRIBUTE_DIRECTORY) == 0)
        {
            throw new IOException(
                "Krasis runtime cleanup target is not a directory: '" +
                extendedPath + "'."
            );
        }
        DeleteDirectory(extendedPath);
    }

    private static void DeleteDirectory(string path)
    {
        WIN32_FIND_DATA data;
        IntPtr handle = FindFirstFileW(path + @"\*", out data);
        if (handle == InvalidHandleValue)
        {
            int error = Marshal.GetLastWin32Error();
            if (error == ERROR_FILE_NOT_FOUND)
            {
                RemoveDirectory(path);
                return;
            }
            if (error == ERROR_PATH_NOT_FOUND)
            {
                return;
            }
            throw Error("enumerate", path, error);
        }

        try
        {
            while (true)
            {
                string name = data.cFileName;
                if (name != "." && name != "..")
                {
                    string child = path + @"\" + name;
                    bool isDirectory =
                        (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
                    bool isReparsePoint =
                        (data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
                    if (isDirectory && !isReparsePoint)
                    {
                        DeleteDirectory(child);
                    }
                    else if (isDirectory)
                    {
                        RemoveDirectory(child);
                    }
                    else
                    {
                        DeleteFile(child);
                    }
                }

                if (!FindNextFileW(handle, out data))
                {
                    int error = Marshal.GetLastWin32Error();
                    if (error != ERROR_NO_MORE_FILES)
                    {
                        throw Error("enumerate", path, error);
                    }
                    break;
                }
            }
        }
        finally
        {
            FindClose(handle);
        }

        RemoveDirectory(path);
    }

    private static void DeleteFile(string path)
    {
        SetFileAttributesW(path, FILE_ATTRIBUTE_NORMAL);
        for (int attempt = 0; attempt < 5; attempt++)
        {
            if (DeleteFileW(path))
            {
                return;
            }
            int error = Marshal.GetLastWin32Error();
            if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND)
            {
                return;
            }
            if (!IsTransient(error) || attempt == 4)
            {
                throw Error("delete file", path, error);
            }
            Thread.Sleep(100);
        }
    }

    private static void RemoveDirectory(string path)
    {
        for (int attempt = 0; attempt < 5; attempt++)
        {
            if (RemoveDirectoryW(path))
            {
                return;
            }
            int error = Marshal.GetLastWin32Error();
            if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND)
            {
                return;
            }
            if (!IsTransient(error) || attempt == 4)
            {
                throw Error("remove directory", path, error);
            }
            Thread.Sleep(100);
        }
    }

    private static bool IsTransient(int error)
    {
        return error == ERROR_ACCESS_DENIED ||
               error == ERROR_SHARING_VIOLATION ||
               error == ERROR_DIR_NOT_EMPTY;
    }

    private static Exception Error(string action, string path, int error)
    {
        return new IOException(
            "Could not " + action + " '" + path + "': " +
            new Win32Exception(error).Message + " (Win32 " + error + ")."
        );
    }
}
'@

try {
    $ResolvedInstallRoot = [System.IO.Path]::GetFullPath($InstallRoot).TrimEnd("\", "/")
    if ([string]::IsNullOrWhiteSpace($ResolvedInstallRoot)) {
        throw "Krasis install root is empty."
    }
    $ExpectedPrefix = $ResolvedInstallRoot + [System.IO.Path]::DirectorySeparatorChar
    $CleanupRoots = @(
        foreach ($Name in @("runtime", "python", "venv")) {
            $Candidate = [System.IO.Path]::GetFullPath(
                (Join-Path $ResolvedInstallRoot $Name)
            ).TrimEnd("\", "/")
            if (-not $Candidate.StartsWith(
                $ExpectedPrefix,
                [StringComparison]::OrdinalIgnoreCase
            )) {
                throw "Krasis runtime cleanup target escaped the install root: $Candidate"
            }
            if (Test-Path $Candidate) {
                $Candidate
            }
        }
    )
    if ($CleanupRoots.Count -eq 0) {
        exit 0
    }
    foreach ($CleanupRoot in $CleanupRoots) {
        if (Test-KrasisRuntimeInUse -RuntimeRoot $CleanupRoot) {
            throw "Krasis is still running from $CleanupRoot. Close Krasis and retry uninstall."
        }
    }

    Add-Type -TypeDefinition $LongPathDeleteSource -Language CSharp
    foreach ($CleanupRoot in $CleanupRoots) {
        [KrasisLongPathDelete]::DeleteTree($CleanupRoot)
        if (Test-Path $CleanupRoot) {
            throw "Krasis runtime cleanup returned without removing $CleanupRoot."
        }
        Write-Host "Removed Krasis runtime tree: $CleanupRoot"
    }
    exit 0
} catch {
    Write-Error $_
    exit 1
}
