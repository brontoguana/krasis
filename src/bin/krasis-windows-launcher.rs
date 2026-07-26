use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

const ACTIVATION_FILE: &str = "runtime/current.txt";
const UPDATE_DIR: &str = "bin";
const UPDATE_SCRIPT_NAME: &str = "Update-Krasis.ps1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UpdateChannel {
    Stable,
    Prerelease,
}

impl UpdateChannel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Prerelease => "prerelease",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LauncherMode {
    Interactive,
    Update(UpdateChannel),
}

#[derive(Debug)]
struct LaunchPaths {
    install_root: PathBuf,
    runtime_root: PathBuf,
    python: PathBuf,
}

fn launcher_mode(executable: &Path) -> Result<LauncherMode, String> {
    let stem = executable
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "Krasis launcher executable name is invalid.".to_string())?;
    if stem.eq_ignore_ascii_case("Krasis") || stem.eq_ignore_ascii_case("krasis-windows-launcher") {
        return Ok(LauncherMode::Interactive);
    }
    if stem.eq_ignore_ascii_case("Krasis Update") {
        return Ok(LauncherMode::Update(UpdateChannel::Stable));
    }
    if stem.eq_ignore_ascii_case("Krasis Prerelease") {
        return Ok(LauncherMode::Update(UpdateChannel::Prerelease));
    }
    Err(format!(
        "Unrecognized Krasis launcher executable name: {stem}"
    ))
}

fn resolve_install_root(executable: &Path) -> Result<PathBuf, String> {
    let bin_dir = executable
        .parent()
        .ok_or_else(|| "Krasis launcher executable has no parent directory.".to_string())?;
    bin_dir
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Krasis launcher is not inside the installation directory.".to_string())
}

fn valid_release_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    !bytes.is_empty()
        && bytes.first().is_some_and(u8::is_ascii_alphanumeric)
        && bytes.last().is_some_and(u8::is_ascii_alphanumeric)
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn resolve_launch_paths(executable: &Path) -> Result<LaunchPaths, String> {
    let install_root = resolve_install_root(executable)?;
    let activation_path = install_root.join(ACTIVATION_FILE);
    let release_name = fs::read_to_string(&activation_path)
        .map_err(|error| {
            format!(
                "Krasis private-runtime activation pointer is unavailable at {}: {error}",
                activation_path.display()
            )
        })?
        .trim()
        .to_string();
    if !valid_release_name(&release_name) {
        return Err("Krasis private-runtime activation pointer is invalid.".to_string());
    }

    let runtime_root = install_root
        .join("runtime")
        .join("releases")
        .join(release_name);
    let python = runtime_root.join("python.exe");
    if !python.is_file() {
        return Err(format!(
            "Krasis private Python executable is missing: {}",
            python.display()
        ));
    }

    Ok(LaunchPaths {
        install_root,
        runtime_root,
        python,
    })
}

fn resolve_update_script(executable: &Path) -> Result<(PathBuf, PathBuf), String> {
    let install_root = resolve_install_root(executable)?;
    let script = install_root.join(UPDATE_DIR).join(UPDATE_SCRIPT_NAME);
    if !script.is_file() {
        return Err(format!(
            "Krasis updater script is missing: {}",
            script.display()
        ));
    }
    Ok((install_root, script))
}

fn updater_arguments(script: &Path, channel: UpdateChannel) -> Vec<OsString> {
    [
        OsString::from("-NoLogo"),
        OsString::from("-NoProfile"),
        OsString::from("-ExecutionPolicy"),
        OsString::from("Bypass"),
        OsString::from("-File"),
        script.as_os_str().to_owned(),
        OsString::from("-Channel"),
        OsString::from(channel.as_str()),
        OsString::from("-PauseOnFailure"),
    ]
    .into_iter()
    .collect()
}

#[cfg(windows)]
fn system_powershell() -> Result<PathBuf, String> {
    use std::os::windows::ffi::OsStringExt;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetSystemDirectoryW(buffer: *mut u16, size: u32) -> u32;
    }

    let mut buffer = vec![0_u16; 260];
    loop {
        let length = unsafe { GetSystemDirectoryW(buffer.as_mut_ptr(), buffer.len() as u32) };
        if length == 0 {
            return Err(format!(
                "Cannot resolve the Windows system directory: {}",
                io::Error::last_os_error()
            ));
        }
        if (length as usize) < buffer.len() {
            let system_dir = PathBuf::from(OsString::from_wide(&buffer[..length as usize]));
            let powershell = system_dir.join("WindowsPowerShell/v1.0/powershell.exe");
            if !powershell.is_file() {
                return Err(format!(
                    "Windows PowerShell is missing: {}",
                    powershell.display()
                ));
            }
            return Ok(powershell);
        }
        buffer.resize(length as usize + 1, 0);
    }
}

#[cfg(windows)]
fn launch_updater(
    install_root: &Path,
    script: &Path,
    channel: UpdateChannel,
) -> Result<(), String> {
    let powershell = system_powershell()?;
    Command::new(&powershell)
        .args(updater_arguments(script, channel))
        .current_dir(install_root)
        .spawn()
        .map(|_| ())
        .map_err(|error| {
            format!(
                "Unable to start the Krasis {} updater with {}: {error}",
                channel.as_str(),
                powershell.display()
            )
        })
}

#[cfg(not(windows))]
fn launch_updater(
    _install_root: &Path,
    _script: &Path,
    _channel: UpdateChannel,
) -> Result<(), String> {
    Err("The native Krasis updater is available only on Windows.".to_string())
}

fn launch_krasis(paths: &LaunchPaths) -> Result<ExitStatus, String> {
    Command::new(&paths.python)
        .args(["-I", "-m", "krasis.launcher"])
        .current_dir(&paths.install_root)
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONUSERBASE")
        .env("PYTHONNOUSERSITE", "1")
        .env("KRASIS_WINDOWS_NATIVE", "1")
        .env("PYTHONUTF8", "1")
        .status()
        .map_err(|error| {
            format!(
                "Unable to start Krasis with its private runtime at {}: {error}",
                paths.python.display()
            )
        })
}

fn pause_after_failure() {
    if io::stdin().is_terminal() {
        eprint!("\nPress Enter to close.");
        let _ = io::stderr().flush();
        let mut line = String::new();
        let _ = io::stdin().read_line(&mut line);
    }
}

fn run() -> Result<i32, String> {
    let executable =
        env::current_exe().map_err(|error| format!("Cannot resolve Krasis.exe: {error}"))?;
    let mode = launcher_mode(&executable)?;

    let mut arguments = env::args_os();
    let _program = arguments.next();
    match mode {
        LauncherMode::Interactive => {
            let paths = resolve_launch_paths(&executable)?;
            match arguments.next() {
                None => {
                    let status = launch_krasis(&paths)?;
                    Ok(status.code().unwrap_or(1))
                }
                Some(argument) if argument == "--probe" && arguments.next().is_none() => {
                    println!(
                        "KRASIS_WINDOWS_LAUNCHER_PROBE={}",
                        paths.runtime_root.display()
                    );
                    Ok(0)
                }
                Some(_) => Err("Usage: Krasis.exe [--probe]".to_string()),
            }
        }
        LauncherMode::Update(channel) => {
            let (install_root, script) = resolve_update_script(&executable)?;
            match arguments.next() {
                None => {
                    launch_updater(&install_root, &script, channel)?;
                    Ok(0)
                }
                Some(argument) if argument == "--probe" && arguments.next().is_none() => {
                    println!(
                        "KRASIS_WINDOWS_UPDATER_PROBE={}:{}",
                        channel.as_str(),
                        script.display()
                    );
                    Ok(0)
                }
                Some(_) => Err(format!(
                    "Usage: {} [--probe]",
                    executable
                        .file_name()
                        .and_then(|value| value.to_str())
                        .unwrap_or("Krasis updater")
                )),
            }
        }
    }
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(message) => {
            eprintln!("Krasis could not start:\n  {message}");
            eprintln!("\nRun Krasis Update or reinstall Krasis to repair the private runtime.");
            pause_after_failure();
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        launcher_mode, resolve_launch_paths, updater_arguments, valid_release_name, LauncherMode,
        UpdateChannel,
    };
    use std::fs;
    use std::path::Path;

    #[test]
    fn installed_executable_names_select_fixed_modes() {
        assert_eq!(
            launcher_mode(Path::new("Krasis.exe")).unwrap(),
            LauncherMode::Interactive
        );
        assert_eq!(
            launcher_mode(Path::new("Krasis Update.exe")).unwrap(),
            LauncherMode::Update(UpdateChannel::Stable)
        );
        assert_eq!(
            launcher_mode(Path::new("Krasis Prerelease.exe")).unwrap(),
            LauncherMode::Update(UpdateChannel::Prerelease)
        );
        assert!(launcher_mode(Path::new("renamed.exe")).is_err());
    }

    #[test]
    fn updater_arguments_are_isolated_and_channel_specific() {
        let script = Path::new(r"C:\Program Files\Krasis\bin\Update-Krasis.ps1");
        let stable = updater_arguments(script, UpdateChannel::Stable);
        let prerelease = updater_arguments(script, UpdateChannel::Prerelease);
        assert_eq!(stable[7], "stable");
        assert_eq!(prerelease[7], "prerelease");
        assert_eq!(stable[4], "-File");
        assert_eq!(stable[5], script.as_os_str());
        assert_eq!(stable[6], "-Channel");
        assert_eq!(stable[8], "-PauseOnFailure");
    }

    #[test]
    fn release_name_is_strictly_local() {
        for valid in ["1.0.16-rc.6", "cp312_win-amd64", "release.1"] {
            assert!(valid_release_name(valid), "{valid}");
        }
        for invalid in [
            "",
            ".",
            "..",
            ".hidden",
            "trailing.",
            "../escape",
            "nested/release",
            r"nested\release",
            "release name",
        ] {
            assert!(!valid_release_name(invalid), "{invalid}");
        }
    }

    #[test]
    fn resolves_only_the_activated_private_python() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "krasis-launcher-test-{}-{unique}",
            std::process::id(),
        ));
        let bin = root.join("bin");
        let release = root.join("runtime/releases/krasis-test-runtime");
        fs::create_dir_all(&bin).unwrap();
        fs::create_dir_all(&release).unwrap();
        fs::write(root.join("runtime/current.txt"), "krasis-test-runtime\n").unwrap();
        fs::write(release.join("python.exe"), b"test").unwrap();

        let paths = resolve_launch_paths(&bin.join("Krasis.exe")).unwrap();
        assert_eq!(paths.install_root, root);
        assert_eq!(paths.runtime_root, release);
        assert_eq!(paths.python, release.join("python.exe"));

        fs::remove_dir_all(paths.install_root).unwrap();
    }
}
