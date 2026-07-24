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
Source: "{#SourceDir}\bin\*"; DestDir: "{app}\bin"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\VERSION.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\runtime-package\*"; DestDir: "{tmp}\KrasisRuntime-{#AppVersion}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Krasis\Krasis"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -NoExit -WindowStyle Maximized -File ""{app}\bin\Launch-Krasis.ps1"""; WorkingDir: "{app}"
Name: "{autoprograms}\Krasis\Krasis Update"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -NoExit -WindowStyle Maximized -File ""{app}\bin\Update-Krasis.ps1"" -Channel stable"; WorkingDir: "{app}"
Name: "{autoprograms}\Krasis\Krasis Prerelease"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -NoExit -WindowStyle Maximized -File ""{app}\bin\Update-Krasis.ps1"" -Channel prerelease"; WorkingDir: "{app}"

[InstallDelete]
Type: files; Name: "{app}\bin\python-installer.exe"
Type: filesandordirs; Name: "{app}\bin\wheelhouse"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\venv"
Type: files; Name: "{app}\runtime-install.log"

[Code]
var
  RuntimeInstallExitCode: Integer;

function GetCustomSetupExitCode: Integer;
begin
  Result := RuntimeInstallExitCode;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  PowerShell: String;
  InstallScript: String;
  InvokeScript: String;
  RuntimePackage: String;
  Parameters: String;
begin
  if CurStep <> ssPostInstall then
    exit;

  PowerShell := ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe');
  InstallScript := ExpandConstant('{app}\bin\Install-Krasis.ps1');
  InvokeScript := ExpandConstant('{app}\bin\Invoke-Install-Krasis.ps1');
  RuntimePackage := ExpandConstant('{tmp}\KrasisRuntime-{#AppVersion}');
  Parameters :=
    '-NoProfile -ExecutionPolicy Bypass -File ' + AddQuotes(InvokeScript) +
    ' -InstallScript ' + AddQuotes(InstallScript) +
    ' -InstallRoot ' + AddQuotes(ExpandConstant('{app}')) +
    ' -RuntimePackage ' + AddQuotes(RuntimePackage) +
    ' -LogPath ' + AddQuotes(ExpandConstant('{app}\runtime-install.log'));

  if not Exec(PowerShell, Parameters, '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then begin
    RuntimeInstallExitCode := 1;
    RaiseException('Unable to start Krasis private-runtime installation.');
  end;
  if ResultCode <> 0 then begin
    RuntimeInstallExitCode := ResultCode;
    RaiseException(
      Format('Krasis private-runtime installation failed with status %d.', [ResultCode])
    );
  end;
end;
