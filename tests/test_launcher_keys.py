import os
import sys
import threading
import time
import unittest

from krasis import launcher


class LauncherKeyTests(unittest.TestCase):
    def _read_key_from_pty(self, chunks):
        master_fd, slave_fd = os.openpty()
        original_stdin = sys.stdin
        try:
            with os.fdopen(slave_fd, "r", buffering=1) as slave:
                sys.stdin = slave

                def writer():
                    for delay, data in chunks:
                        time.sleep(delay)
                        os.write(master_fd, data)

                thread = threading.Thread(target=writer)
                thread.start()
                try:
                    return launcher._read_key()
                finally:
                    thread.join(timeout=1.0)
        finally:
            sys.stdin = original_stdin
            os.close(master_fd)

    def test_burst_text_is_preserved_between_raw_key_reads(self):
        master_fd, slave_fd = os.openpty()
        original_stdin = sys.stdin
        payload = b'profile.conf'
        try:
            with os.fdopen(slave_fd, 'r', buffering=1) as slave:
                sys.stdin = slave
                def writer():
                    time.sleep(0.02)
                    os.write(master_fd, payload)
                thread = threading.Thread(target=writer)
                thread.start()
                actual = [launcher._read_key_timeout(0.2) for _ in payload]
                thread.join(timeout=1)
                self.assertEqual(actual, list(payload.decode()))
        finally:
            sys.stdin = original_stdin
            os.close(master_fd)

    def test_blocking_reader_preserves_fast_text_and_enter(self):
        master_fd, slave_fd = os.openpty()
        original_stdin = sys.stdin
        try:
            with os.fdopen(slave_fd, 'r', buffering=1) as slave:
                sys.stdin = slave
                def writer():
                    time.sleep(0.02)
                    os.write(master_fd, b'abc\r')
                    # Prevent a failing old implementation from hanging.
                    time.sleep(0.3)
                    os.write(master_fd, b'\x1b')
                thread = threading.Thread(target=writer)
                thread.start()
                actual = []
                for _ in range(4):
                    key = launcher._read_key()
                    actual.append(key)
                    if key == launcher.KEY_ESCAPE:
                        break
                thread.join(timeout=1)
                self.assertEqual(actual, ['a', 'b', 'c', launcher.KEY_ENTER])
        finally:
            sys.stdin = original_stdin
            os.close(master_fd)

    def test_down_arrow_tolerates_delayed_escape_sequence(self):
        key = self._read_key_from_pty([(0.01, b"\x1b"), (0.05, b"[B")])
        self.assertEqual(key, launcher.KEY_DOWN)

    def test_plain_escape_does_not_need_second_escape(self):
        start = time.time()
        key = self._read_key_from_pty([(0.01, b"\x1b")])
        elapsed = time.time() - start
        self.assertEqual(key, launcher.KEY_ESCAPE)
        self.assertLess(elapsed, 0.5)


if __name__ == "__main__":
    unittest.main()
