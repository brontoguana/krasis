"""Regression for the real terminal listener exiting before Rust run starts."""
import os
import threading
import unittest
from krasis.server import _listen_for_shutdown

class TerminalShutdownTests(unittest.TestCase):
    def test_q_listener_survives_pre_run_state_and_stops_running_server(self):
        read_fd, write_fd = os.pipe()
        stop = threading.Event()
        polled_before_run = threading.Event()
        stopped = threading.Event()
        running = threading.Event()
        class Server:
            def is_running(self):
                if not running.is_set():
                    polled_before_run.set()
                return running.is_set()
        def on_exit(sig, frame):
            stopped.set()
        with os.fdopen(read_fd) as stream:
            thread = threading.Thread(target=_listen_for_shutdown,
                args=(Server(), stop, on_exit, stream))
            thread.start()
            try:
                self.assertTrue(polled_before_run.wait(2), 'listener must observe the startup gap')
                self.assertTrue(thread.is_alive(), 'a not-yet-started Rust server must not end the listener')
                running.set()
                os.write(write_fd, b'Q\n')
                self.assertTrue(stopped.wait(2), 'actual terminal Q must stop the server')
            finally:
                stop.set()
                os.close(write_fd)
                thread.join(2)
            self.assertFalse(thread.is_alive())

    def test_listener_can_be_cancelled_when_run_never_starts(self):
        stop = threading.Event()
        polled = threading.Event()
        class Server:
            def is_running(self):
                polled.set()
                return False
        thread = threading.Thread(target=_listen_for_shutdown,
            args=(Server(), stop, lambda *_: self.fail('unexpected stop callback'), None))
        thread.start()
        self.assertTrue(polled.wait(2))
        stop.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())

if __name__ == '__main__':
    unittest.main()
