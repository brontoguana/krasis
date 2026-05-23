import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_RS = ROOT / "src" / "server.rs"
GPU_DECODE_RS = ROOT / "src" / "gpu_decode.rs"
VRAM_MONITOR_RS = ROOT / "src" / "vram_monitor.rs"


class RequestProgressLoggingTests(unittest.TestCase):
    def test_chat_requests_print_prefill_and_decode_progress(self):
        server = SERVER_RS.read_text()

        self.assertIn("\\x1b[32mprefill:", server)
        self.assertIn("min free during prefill", server)
        self.assertIn("current_request_lows()", server)
        self.assertIn("reset_request_lows();", server)

        gpu_decode = GPU_DECODE_RS.read_text()
        self.assertIn("\\x1b[32mdecode:", gpu_decode)
        self.assertIn("min free during decode", gpu_decode)

    def test_vram_monitor_exposes_request_low_water_snapshot(self):
        monitor = VRAM_MONITOR_RS.read_text()

        self.assertIn("pub fn reset_request_lows()", monitor)
        self.assertIn("pub fn current_request_lows() -> Vec<(i32, u64)>", monitor)


if __name__ == "__main__":
    unittest.main()
