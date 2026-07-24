#define AppVersion GetEnv("KRASIS_WINDOWS_VERSION")
#define SourceDir GetEnv("KRASIS_WINDOWS_STAGING")
#define OutputDir GetEnv("KRASIS_WINDOWS_OUTPUT")

[Setup]
AppId={{D8C9713A-4E29-4B86-93E7-D2D6E68C6E22}
AppName=Krasis
AppVersion={#AppVersion}
AppPublisher=Krasis
DefaultDirName={localappdata}\Programs\Krasis
DefaultGroupName=Krasis
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=KrasisSetup-{#AppVersion}-win64
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\bin\Launch-Krasis.ps1

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Krasis\Krasis"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -NoExit -WindowStyle Maximized -File ""{app}\bin\Launch-Krasis.ps1"""; WorkingDir: "{app}"
Name: "{autoprograms}\Krasis\Krasis Update"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -NoExit -WindowStyle Maximized -File ""{app}\bin\Update-Krasis.ps1"" -Channel stable"; WorkingDir: "{app}"
Name: "{autoprograms}\Krasis\Krasis Prerelease"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -NoExit -WindowStyle Maximized -File ""{app}\bin\Update-Krasis.ps1"" -Channel prerelease"; WorkingDir: "{app}"

[Run]
Filename: "{app}\bin\python-installer.exe"; Parameters: "/quiet InstallAllUsers=0 TargetDir=""{app}\python"" Include_pip=1 Include_launcher=0 PrependPath=0 Include_test=0 Shortcuts=0"; Flags: runhidden waituntilterminated; Check: FileExists(ExpandConstant('{app}\bin\python-installer.exe'))
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\bin\Install-Krasis.ps1"" -InstallRoot ""{app}"" -Wheelhouse ""{app}\bin\wheelhouse"" -NoShortcut"; Flags: runhidden waituntilterminated
