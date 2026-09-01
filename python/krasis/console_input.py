"""Native Windows console-key handling shared by the launcher and chat client."""

import time
from typing import Callable, Optional

try:
    import msvcrt as _msvcrt
except ImportError:
    _msvcrt = None


HAS_WINDOWS_CONSOLE = _msvcrt is not None

_WINDOWS_EXTENDED_KEYS = {
    "H": "UP",
    "P": "DOWN",
    "K": "LEFT",
    "M": "RIGHT",
}


def _read_windows_key_unmanaged(read_char: Callable[[], str]) -> str:
    char = read_char()
    if char in ("\x00", "\xe0"):
        return _WINDOWS_EXTENDED_KEYS.get(read_char(), "")
    if char in ("\r", "\n"):
        return "ENTER"
    if char == "\x1b" or char == "\x03":
        return "ESC"
    if char == "\x08":
        return "BACKSPACE"
    return char


def read_windows_key(
    read_char: Optional[Callable[[], str]] = None,
) -> str:
    """Read and normalize one key from the native Windows console."""
    if read_char is None:
        if _msvcrt is None:
            raise RuntimeError("native Windows console input is unavailable")
        read_char = _msvcrt.getwch

    return _read_windows_key_unmanaged(read_char)


def read_windows_key_timeout(
    timeout: float,
    *,
    key_available: Optional[Callable[[], bool]] = None,
    read_char: Optional[Callable[[], str]] = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Optional[str]:
    """Read one Windows console key before a bounded timeout."""
    if key_available is None:
        if _msvcrt is None:
            raise RuntimeError("native Windows console input is unavailable")
        key_available = _msvcrt.kbhit

    deadline = monotonic() + max(0.0, timeout)
    while True:
        if key_available():
            if read_char is None:
                read_char = _msvcrt.getwch
            return _read_windows_key_unmanaged(read_char)
        remaining = deadline - monotonic()
        if remaining <= 0:
            return None
        sleep(min(0.01, remaining))


def discard_windows_keys(
    *,
    key_available: Optional[Callable[[], bool]] = None,
    read_char: Optional[Callable[[], str]] = None,
) -> int:
    """Discard complete key events already queued in the Windows console."""
    if key_available is None:
        if _msvcrt is None:
            raise RuntimeError("native Windows console input is unavailable")
        key_available = _msvcrt.kbhit
    if read_char is None:
        if _msvcrt is None:
            raise RuntimeError("native Windows console input is unavailable")
        read_char = _msvcrt.getwch

    discarded = 0
    while key_available():
        _read_windows_key_unmanaged(read_char)
        discarded += 1
    return discarded
