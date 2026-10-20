use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const REMOTE_WRITER: &str = r#"set -eu; umask 027
dir="$HOME/krasis-capture-phases/$(cat /proc/sys/kernel/random/boot_id)"
mkdir -p "$dir"
file="$dir/rust-$$.jsonl"
(set -C; : > "$file")
sync -f "$dir"
while IFS= read -r line; do
    printf '%s\n' "$line" >> "$file"
    sync -d "$file"
    printf '%s\n' "$line"
done"#;

pub(crate) struct CapturePhase {
    file: File,
    child: Child,
    input: ChildStdin,
    output: ChildStdout,
    boot: String,
    sequence: u64,
    boundaries_only: bool,
    first_layer_only: bool,
    first_layer: Option<usize>,
}

impl Drop for CapturePhase {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl CapturePhase {
    pub(crate) fn from_env() -> Result<Option<Self>, String> {
        let (boundaries_only, first_layer_only) = match std::env::var("KRASIS_CAPTURE_PHASE_SCOPE")
        {
            Err(std::env::VarError::NotPresent) => (false, false),
            Ok(value) if value == "all" => (false, false),
            Ok(value) if value == "boundaries" => (true, false),
            Ok(value) if value == "first_layer" => (false, true),
            _ => return Err("capture phase scope must be all, boundaries or first_layer".into()),
        };
        if std::env::var_os("KRASIS_CAPTURE_PHASE_RECEIVER").is_some() {
            return Err(
                "UDP phase capture is unsupported; use KRASIS_CAPTURE_PHASE_SSH_TARGET".into(),
            );
        }
        let Some(path) = std::env::var_os("KRASIS_CAPTURE_PHASE_FILE") else {
            if std::env::var_os("KRASIS_CAPTURE_PHASE_SSH_TARGET").is_some()
                || std::env::var_os("KRASIS_CAPTURE_PHASE_SCOPE").is_some()
            {
                return Err("capture phase target requires a phase file".into());
            }
            return Ok(None);
        };
        if !cfg!(target_os = "linux") {
            return Err("durable capture phase diagnostics require Linux".into());
        }
        let target = std::env::var("KRASIS_CAPTURE_PHASE_SSH_TARGET")
            .map_err(|_| "capture phase file requires an SSH target".to_string())?;
        if target.is_empty()
            || target.starts_with('-')
            || !target
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '@' | '.' | '-' | '_' | ':'))
        {
            return Err("invalid capture phase SSH target".into());
        }
        let mut command = Command::new("ssh");
        command.args([
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            "--",
            &target,
            REMOTE_WRITER,
        ]);
        let mut sink = Self::open(Path::new(&path), command)?;
        sink.boundaries_only = boundaries_only;
        sink.first_layer_only = first_layer_only;
        Ok(Some(sink))
    }

    fn open(path: &Path, mut command: Command) -> Result<Self, String> {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("create capture phase file: {e}"))?;
        File::open(path.parent().ok_or("capture phase file needs parent")?)
            .and_then(|dir| dir.sync_all())
            .map_err(|e| format!("sync capture phase directory: {e}"))?;
        let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .map_err(|e| format!("read capture phase boot identity: {e}"))?;
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("start acknowledged phase transport: {e}"))?;
        let input = child.stdin.take().expect("piped child stdin");
        let output = child.stdout.take().expect("piped child stdout");
        Ok(Self {
            file,
            child,
            input,
            output,
            boot: boot.trim().into(),
            sequence: 0,
            boundaries_only: false,
            first_layer_only: false,
            first_layer: None,
        })
    }

    #[cfg(unix)]
    fn acknowledge(&mut self, record: &[u8]) -> Result<(), String> {
        use std::os::fd::AsRawFd;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut received = vec![0u8; record.len()];
        let mut filled = 0;
        while filled < received.len() {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Err("capture phase acknowledgement timed out".into());
            }
            let mut fd = libc::pollfd {
                fd: self.output.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            let ready = unsafe { libc::poll(&mut fd, 1, remaining.as_millis().max(1) as i32) };
            if ready < 0 {
                let error = std::io::Error::last_os_error();
                if error.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(format!("poll capture phase acknowledgement: {error}"));
            }
            if ready == 0 {
                return Err("capture phase acknowledgement timed out".into());
            }
            let count = self
                .output
                .read(&mut received[filled..])
                .map_err(|e| format!("read capture phase acknowledgement: {e}"))?;
            if count == 0 {
                return Err("capture phase receiver closed before acknowledgement".into());
            }
            filled += count;
            if received[..filled] != record[..filled] {
                return Err("capture phase receiver returned a mismatched acknowledgement".into());
            }
        }
        Ok(())
    }

    #[cfg(not(unix))]
    fn acknowledge(&mut self, _record: &[u8]) -> Result<(), String> {
        Err("durable capture phase diagnostics require Linux".into())
    }

    pub(crate) fn mark(
        &mut self,
        phase: &str,
        layer: Option<usize>,
        rows: usize,
    ) -> Result<(), String> {
        // Layer-scoped copies can be excluded explicitly for timing-sensitive
        // investigations. Request/init/commit boundaries retain durable ACKs.
        if self.boundaries_only && layer.is_some() {
            return Ok(());
        }
        // Select the first actually captured routed layer, independent of the
        // model's dense prefix or layer numbering. Keep it across requests.
        if self.first_layer_only {
            if let Some(layer) = layer {
                if *self.first_layer.get_or_insert(layer) != layer {
                    return Ok(());
                }
            }
        }
        self.sequence += 1;
        let record = serde_json::json!({
            "tag": "KRASIS_CAPTURE_PHASE", "boot": self.boot,
            "pid": std::process::id(), "seq": self.sequence,
            "unix_ns": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| format!("capture phase clock: {e}"))?.as_nanos().to_string(),
            "phase": phase, "layer": layer, "rows": rows,
            "scope": if self.boundaries_only { "boundaries" }
                else if self.first_layer_only { "first_layer" } else { "all" },
        })
        .to_string()
            + "\n";
        self.file
            .write_all(record.as_bytes())
            .and_then(|_| self.file.sync_all())
            .map_err(|e| format!("persist capture phase: {e}"))?;
        self.input
            .write_all(record.as_bytes())
            .and_then(|_| self.input.flush())
            .map_err(|e| format!("send capture phase: {e}"))?;
        self.acknowledge(record.as_bytes())
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    fn test_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "capture-phase-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        dir
    }

    #[test]
    fn acknowledged_records_match_synced_remote_file_and_no_overwrite() {
        let dir = test_dir();
        let path = dir.join("phase.jsonl");
        let remote = dir.join("remote.jsonl");
        let mut command = Command::new("sh");
        command.args(["-c", "set -eu; while IFS= read -r line; do printf '%s\\n' \"$line\" >> \"$1\"; sync -d \"$1\"; printf '%s\\n' \"$line\"; done", "--"]);
        command.arg(&remote);
        let mut sink = CapturePhase::open(&path, command).unwrap();
        for _ in 0..50 {
            sink.mark("copy_done", Some(3), 8192).unwrap();
        }
        assert_eq!(
            std::fs::read(&path).unwrap(),
            std::fs::read(&remote).unwrap()
        );
        assert!(CapturePhase::open(&path, Command::new("cat")).is_err());
        drop(sink);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn incorrect_acknowledgement_stops_capture() {
        let dir = test_dir();
        let mut command = Command::new("sh");
        command.args(["-c", "read -r line; echo WRONG"]);
        let mut sink = CapturePhase::open(&dir.join("phase.jsonl"), command).unwrap();
        assert!(sink
            .mark("copy_begin", Some(3), 8192)
            .unwrap_err()
            .contains("mismatched"));
        drop(sink);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn boundary_scope_skips_layer_records_and_keeps_durable_boundaries() {
        let dir = test_dir();
        let path = dir.join("phase.jsonl");
        let mut command = Command::new("cat");
        command.arg("-");
        let mut sink = CapturePhase::open(&path, command).unwrap();
        sink.boundaries_only = true;
        sink.mark("inputs_copy_begin", Some(3), 8192).unwrap();
        assert_eq!(sink.sequence, 0);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
        sink.mark("armed_request_begin", None, 41754).unwrap();
        sink.mark("commit_done", None, 0).unwrap();
        assert_eq!(sink.sequence, 2);
        let records = std::fs::read_to_string(&path).unwrap();
        assert_eq!(records.lines().count(), 2);
        for record in records.lines() {
            let value: serde_json::Value = serde_json::from_str(record).unwrap();
            assert_eq!(value["scope"], "boundaries");
            assert!(value["layer"].is_null());
        }
        drop(sink);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn first_layer_scope_keeps_runtime_selected_layer_and_all_boundaries() {
        let dir = test_dir();
        let path = dir.join("phase.jsonl");
        let mut command = Command::new("cat");
        command.arg("-");
        let mut sink = CapturePhase::open(&path, command).unwrap();
        sink.first_layer_only = true;
        sink.mark("armed_request_begin", None, 100).unwrap();
        assert_eq!(sink.first_layer, None);
        for _ in 0..2 {
            sink.mark("stream_sync_begin", Some(19), 10).unwrap();
            sink.mark("append_done", Some(19), 10).unwrap();
            let sequence = sink.sequence;
            sink.mark("stream_sync_begin", Some(20), 10).unwrap();
            sink.mark("append_done", Some(20), 10).unwrap();
            assert_eq!(sink.sequence, sequence);
            sink.mark("commit_done", None, 0).unwrap();
        }
        assert_eq!(sink.first_layer, Some(19));
        assert_eq!(sink.sequence, 7);
        for record in std::fs::read_to_string(&path).unwrap().lines() {
            let value: serde_json::Value = serde_json::from_str(record).unwrap();
            assert_eq!(value["scope"], "first_layer");
            assert!(value["layer"].is_null() || value["layer"] == 19);
        }
        drop(sink);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
