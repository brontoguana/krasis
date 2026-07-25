"""Native Windows console-key handling shared by the launcher and chat client."""

from contextlib import contextmanager, nullcontext
import time
from typing import Callable, Iterator, Optional

try:
    import msvcrt as _msvcrt
except ImportError:
    _msvcrt = None


HAS_WINDOWS_CONSOLE = _msvcrt is not None

_STD_INPUT_HANDLE = -10
_ENABLE_MOUSE_INPUT = 0x0010
_ENABLE_QUICK_EDIT_MODE = 0x0040
_ENABLE_EXTENDED_FLAGS = 0x0080

_WINDOWS_EXTENDED_KEYS = {
    "H": "UP",
    "P": "DOWN",
    "K": "LEFT",
    "M": "RIGHT",
}


def _native_console_mode_functions() -> tuple[Callable[[], int], Callable[[int], None]]:
    """Return strict GetConsoleMode/SetConsoleMode wrappers for standard input."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_std_handle = kernel32.GetStdHandle
    get_std_handle.argtypes = [wintypes.DWORD]
    get_std_handle.restype = wintypes.HANDLE
    get_console_mode = kernel32.GetConsoleMode
    get_console_mode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    get_console_mode.restype = wintypes.BOOL
    set_console_mode = kernel32.SetConsoleMode
    set_console_mode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    set_console_mode.restype = wintypes.BOOL

    handle = get_std_handle(_STD_INPUT_HANDLE)

    def read_mode() -> int:
        mode = wintypes.DWORD()
        if not get_console_mode(handle, ctypes.byref(mode)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(mode.value)

    def write_mode(mode: int) -> None:
        if not set_console_mode(handle, mode):
            raise ctypes.WinError(ctypes.get_last_error())

    return read_mode, write_mode


@contextmanager
def windows_console_key_mode(
    *,
    get_mode: Optional[Callable[[], int]] = None,
    set_mode: Optional[Callable[[int], None]] = None,
) -> Iterator[None]:
    """Prevent Windows mouse/QuickEdit input from suspending a key read."""
    if get_mode is None or set_mode is None:
        if _msvcrt is None:
            yield
            return
        get_mode, set_mode = _native_console_mode_functions()

    original_mode = get_mode()
    key_mode = (
        (original_mode | _ENABLE_EXTENDED_FLAGS)
        & ~_ENABLE_QUICK_EDIT_MODE
        & ~_ENABLE_MOUSE_INPUT
    )
    if key_mode != original_mode:
        set_mode(key_mode)
    try:
        yield
    finally:
        if key_mode != original_mode:
            set_mode(original_mode)


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
    injected_reader = read_char is not None
    if read_char is None:
        if _msvcrt is None:
            raise RuntimeError("native Windows console input is unavailable")
        read_char = _msvcrt.getwch

    if injected_reader or _msvcrt is None:
        return _read_windows_key_unmanaged(read_char)
    with windows_console_key_mode():
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
    injected_io = key_available is not None or read_char is not None
    if key_available is None:
        if _msvcrt is None:
            raise RuntimeError("native Windows console input is unavailable")
        key_available = _msvcrt.kbhit

    deadline = monotonic() + max(0.0, timeout)
    mode_scope = (
        windows_console_key_mode()
        if _msvcrt is not None and not injected_io
        else nullcontext()
    )
    with mode_scope:
        while True:
            if key_available():
                if read_char is None:
                    read_char = _msvcrt.getwch
                return _read_windows_key_unmanaged(read_char)
            remaining = deadline - monotonic()
            if remaining <= 0:
                return None
            sleep(min(0.01, remaining))
