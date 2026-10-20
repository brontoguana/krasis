//! Durable HTTP evidence for the single-threaded conversation-cache client.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub fn save(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| format!("persist {}: {e}", path.display()))?;
    #[cfg(unix)]
    File::open(path.parent().ok_or("evidence has no parent")?)
        .and_then(|parent| parent.sync_all())
        .map_err(|e| format!("sync evidence directory: {e}"))?;
    Ok(())
}

pub struct Client {
    pub root: PathBuf,
    pub elapsed_ms: f64,
    pub context: Value,
    port: u16,
    sequence: usize,
}

impl Client {
    pub fn new(port: u16, root: PathBuf) -> Result<Self, String> {
        // Caller chooses a fresh directory. Reusing evidence must fail visibly.
        fs::create_dir(&root).map_err(|e| format!("create {}: {e}", root.display()))?;
        let executable = std::env::current_exe().map_err(|e| e.to_string())?;
        let bytes = fs::read(&executable).map_err(|e| e.to_string())?;
        save(
            &root.join("client-identity.json"),
            &serde_json::to_vec_pretty(&json!({
                "executable": executable,
                "sha256": format!("{:x}", Sha256::digest(&bytes)),
                "argv": std::env::args().collect::<Vec<_>>(),
                "port": port,
                "debug_distributions": std::env::var_os("KRASIS_CACHE_BENCH_DEBUG").is_some(),
            }))
            .map_err(|e| e.to_string())?,
        )?;
        if let Some(identity) = std::env::var_os("KRASIS_CACHE_BENCH_RUNTIME_IDENTITY") {
            save(
                &root.join("runtime-identity.txt"),
                &fs::read(identity).map_err(|e| e.to_string())?,
            )?;
        }
        Ok(Self {
            root,
            elapsed_ms: 0.0,
            context: Value::Null,
            port,
            sequence: 0,
        })
    }

    pub fn json(
        &mut self,
        method: &str,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value, String> {
        let prefix = self.root.join(format!("{:04}", self.sequence));
        self.sequence += 1;
        let artifact = |suffix: &str| prefix.with_extension(suffix);
        let body_text = body
            .map(serde_json::to_string)
            .transpose()
            .map_err(|e| e.to_string())?
            .unwrap_or_default();
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.port, body_text.len(), body_text,
        );
        save(
            &artifact("context.json"),
            &serde_json::to_vec_pretty(&json!({
                "method": method, "path": path, "context": self.context,
            }))
            .map_err(|e| e.to_string())?,
        )?;
        save(&artifact("request.http"), request.as_bytes())?;
        let started = Instant::now();
        let mut response = Vec::new();
        let transport = (|| -> Result<(), String> {
            let address = ("127.0.0.1", self.port)
                .to_socket_addrs()
                .map_err(|e| e.to_string())?
                .next()
                .ok_or("server address did not resolve")?;
            let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(10))
                .map_err(|e| e.to_string())?;
            let timeout = Some(Duration::from_secs(1_800));
            stream
                .set_read_timeout(timeout)
                .map_err(|e| e.to_string())?;
            stream
                .set_write_timeout(timeout)
                .map_err(|e| e.to_string())?;
            stream
                .write_all(request.as_bytes())
                .map_err(|e| e.to_string())?;
            stream
                .read_to_end(&mut response)
                .map_err(|e| e.to_string())?;
            Ok(())
        })();
        self.elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        // Save even malformed, non-200 and partial responses before parsing/oracles.
        save(&artifact("response.http"), &response)?;
        save(
            &artifact("transport.json"),
            &serde_json::to_vec_pretty(&json!({
                "elapsed_ms": self.elapsed_ms, "error": transport.as_ref().err(),
                "request_sha256": format!("{:x}", Sha256::digest(request.as_bytes())),
                "response_sha256": format!("{:x}", Sha256::digest(&response)),
            }))
            .map_err(|e| e.to_string())?,
        )?;
        transport?;
        parse_response(&response)
    }
}

fn parse_response(response: &[u8]) -> Result<Value, String> {
    let separator = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or("HTTP response is missing its header separator")?;
    let header = std::str::from_utf8(&response[..separator]).map_err(|e| e.to_string())?;
    let status = header
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or("HTTP response is missing a status code")?;
    let body = &response[separator + 4..];
    if status != 200 {
        return Err(format!(
            "server returned HTTP {status}: {}",
            String::from_utf8_lossy(body)
        ));
    }
    serde_json::from_slice(body).map_err(|e| format!("server returned invalid JSON: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    fn root() -> PathBuf {
        std::env::temp_dir().join(format!(
            "cache-evidence-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }
    #[test]
    fn failed_response_is_durable_before_validation_and_cannot_be_overwritten() {
        for reply in [
            "HTTP/1.1 500 Error\r\nContent-Length: 3\r\n\r\nbad",
            "HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nbad",
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let worker = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 4096];
                stream.read(&mut request).unwrap();
                stream.write_all(reply.as_bytes()).unwrap();
            });
            let root = root();
            let mut client = Client::new(port, root.clone()).unwrap();
            assert!(client
                .json("POST", "/probe", Some(&json!({"input": "λ"})))
                .is_err());
            worker.join().unwrap();
            assert_eq!(
                fs::read(root.join("0000.response.http")).unwrap(),
                reply.as_bytes()
            );
            assert!(fs::read_to_string(root.join("0000.request.http"))
                .unwrap()
                .contains("λ"));
            assert!(save(&root.join("0000.response.http"), b"replacement").is_err());
            assert!(Client::new(port, root.clone()).is_err());
            fs::remove_dir_all(root).unwrap();
        }
    }
    #[test]
    fn equality_failure_keeps_the_exact_exchange() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let worker = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 4096];
            stream.read(&mut request).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\n{\"content\":\"actual\"}")
                .unwrap();
        });
        let root = root();
        let mut client = Client::new(port, root.clone()).unwrap();
        let value = client
            .json("POST", "/probe", Some(&json!({"seed": 0})))
            .unwrap();
        assert_ne!(value["content"], "expected");
        assert!(fs::read_to_string(root.join("0000.response.http"))
            .unwrap()
            .contains("actual"));
        worker.join().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}
