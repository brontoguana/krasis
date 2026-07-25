use std::env;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

const ACTIVATION_FILE: &str = "runtime/current.txt";

#[derive(Debug)]
struct LaunchPaths {
    install_root: PathBuf,
    runtime_root: PathBuf,
    python: PathBuf,
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
    let bin_dir = executable
        .parent()
        .ok_or_else(|| "Krasis launcher executable has no parent directory.".to_string())?;
    let install_root = bin_dir
        .parent()
        .ok_or_else(|| "Krasis launcher is not inside the installation directory.".to_string())?;
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
        install_root: install_root.to_path_buf(),
        runtime_root,
        python,
    })
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
    let paths = resolve_launch_paths(&executable)?;

    let mut arguments = env::args_os();
    let _program = arguments.next();
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
    use super::{resolve_launch_paths, valid_release_name};
    use std::fs;

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
