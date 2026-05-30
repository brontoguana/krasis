"""Managed reverse SSH tunnel support for exposing local Krasis remotely."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Optional


@dataclass(frozen=True)
class SshTunnelTarget:
    destination: str
    ssh_port: Optional[int] = None


def parse_ssh_tunnel_target(raw: str) -> SshTunnelTarget:
    """Parse user@host or user@host:port into an SSH destination and port."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("SSH tunnel target is empty")
    if text.startswith("-") or any(ch.isspace() for ch in text):
        raise ValueError("SSH tunnel target must be user@host or user@host:port")
    if "@" not in text:
        raise ValueError("SSH tunnel target must include a user, e.g. user@host")

    destination = text
    ssh_port = None
    if text.count(":") == 1:
        host_part, port_part = text.rsplit(":", 1)
        if port_part:
            if not port_part.isdigit():
                raise ValueError("SSH tunnel port must be numeric")
            port = int(port_part)
            if port < 1 or port > 65535:
                raise ValueError("SSH tunnel port must be in 1..65535")
            destination = host_part
            ssh_port = port
    if not destination or destination.startswith("-") or destination.endswith("@"):
        raise ValueError("SSH tunnel target must be user@host or user@host:port")
    return SshTunnelTarget(destination=destination, ssh_port=ssh_port)


def build_ssh_tunnel_command(
    target: str,
    *,
    local_port: int,
    remote_port: Optional[int] = None,
    remote_bind: str = "127.0.0.1",
    local_bind: str = "127.0.0.1",
    key_path: Optional[str] = None,
) -> list[str]:
    """Build the ssh command for a reverse tunnel without invoking a shell."""
    if local_port < 1 or local_port > 65535:
        raise ValueError("local_port must be in 1..65535")
    if remote_port is None:
        remote_port = local_port
    if remote_port < 1 or remote_port > 65535:
        raise ValueError("remote_port must be in 1..65535")

    parsed = parse_ssh_tunnel_target(target)
    cmd = [
        "ssh",
        "-N",
        "-T",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "ConnectTimeout=10",
        "-R", f"{remote_bind}:{remote_port}:{local_bind}:{local_port}",
    ]
    key_path = (key_path or "").strip()
    if key_path:
        if any(ch.isspace() for ch in key_path):
            raise ValueError("SSH key path must not contain whitespace")
        key_path = os.path.expanduser(key_path)
        cmd.extend(["-i", key_path, "-o", "IdentitiesOnly=yes"])
    if parsed.ssh_port is not None:
        cmd.extend(["-p", str(parsed.ssh_port)])
    cmd.append(parsed.destination)
    return cmd


class SshReverseTunnel:
    """Lifecycle wrapper for a reverse SSH tunnel process."""

    def __init__(
        self,
        target: str,
        *,
        local_port: int,
        remote_port: Optional[int] = None,
        key_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.target = target
        self.local_port = int(local_port)
        self.remote_port = int(remote_port) if remote_port is not None else int(local_port)
        self.key_path = (key_path or "").strip()
        self.logger = logger or logging.getLogger(__name__)
        self.command = build_ssh_tunnel_command(
            target,
            local_port=self.local_port,
            remote_port=self.remote_port,
            key_path=self.key_path,
        )
        self.process: Optional[subprocess.Popen[str]] = None
        self._log_thread: Optional[threading.Thread] = None
        self._stderr_lines: list[str] = []

    def start(self, startup_timeout: float = 12.0) -> None:
        if not shutil.which("ssh"):
            raise RuntimeError("ssh executable not found in PATH")
        if self.process is not None:
            return
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._log_thread = threading.Thread(target=self._drain_output, daemon=True)
        self._log_thread.start()
        atexit.register(self.stop)

        deadline = time.time() + max(0.1, startup_timeout)
        while time.time() < deadline:
            rc = self.process.poll()
            if rc is not None:
                detail = self.last_output()
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(f"SSH tunnel failed to start (exit {rc}){suffix}")
            time.sleep(0.05)
        self.logger.info(
            "SSH tunnel active: remote 127.0.0.1:%d -> local 127.0.0.1:%d via %s",
            self.remote_port,
            self.local_port,
            self.target,
        )

    def _drain_output(self) -> None:
        proc = self.process
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            text = line.rstrip()
            if not text:
                continue
            self._stderr_lines.append(text)
            del self._stderr_lines[:-8]
            self.logger.warning("ssh tunnel: %s", text)

    def last_output(self) -> str:
        return " | ".join(self._stderr_lines[-4:])

    def stop(self) -> None:
        proc = self.process
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3.0)
        self.process = None
