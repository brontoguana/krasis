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
