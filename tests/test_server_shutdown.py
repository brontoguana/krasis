import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from krasis import server


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "torch._inductor.async_compile"


class ServerShutdownTests(unittest.TestCase):
    def test_absent_inductor_module_does_not_import_or_create_pool(self):
        with mock.patch.dict(sys.modules, {MODULE_NAME: None}):
            server._shutdown_inductor_compile_workers()

    def test_loaded_inductor_pool_uses_its_canonical_shutdown(self):
        shutdown = mock.Mock()
        module = types.SimpleNamespace(shutdown_compile_workers=shutdown)
        with mock.patch.dict(sys.modules, {MODULE_NAME: module}):
            server._shutdown_inductor_compile_workers()
        shutdown.assert_called_once_with()

    def test_loaded_module_without_shutdown_contract_fails_closed(self):
        module = types.SimpleNamespace()
        with mock.patch.dict(sys.modules, {MODULE_NAME: module}):
            with self.assertRaisesRegex(RuntimeError, "shutdown_compile_workers"):
                server._shutdown_inductor_compile_workers()

    def test_compile_workers_stop_before_cuda_cleanup_and_main_frame_release(self):
        source = (ROOT / "python/krasis/server.py").read_text()
        shutdown_tail = source[source.index("# ── Clean exit before Python teardown"):]
        ind = shutdown_tail.index("_shutdown_inductor_compile_workers()")
        resources = shutdown_tail.index("rust_server.release_runtime_resources()")
        cuda = shutdown_tail.index("_cleanup_cuda()")
        frame_release = shutdown_tail.index("return None")
        self.assertLess(ind, resources)
        self.assertLess(resources, cuda)
        self.assertLess(ind, cuda)
        self.assertLess(cuda, frame_release)

    def test_normal_exit_performs_strict_cleanup_after_server_main_frame_returns(self):
        source = (ROOT / "python/krasis/server.py").read_text()
        entrypoint = source[source.index('if __name__ == "__main__":'):]
        self.assertIn("    main()", entrypoint)
        self.assertIn("_cleanup_cuda(fail_closed=True)", entrypoint)
        self.assertLess(
            entrypoint.index("main()"),
            entrypoint.index("_cleanup_cuda(fail_closed=True)"),
        )
        self.assertNotIn("os._exit", entrypoint)

    def test_host_registrations_release_before_store_and_model_backing_clear(self):
        source = (ROOT / "python/krasis/server.py").read_text()
        shutdown_tail = source[source.index("# ── Clean exit before Python teardown"):]
        release = shutdown_tail.index("rust_server.release_runtime_resources()")
        accounting = shutdown_tail.index("phase=expert_host_unregistered")
        self.assertGreater(accounting, release)
        for fragment in (
            '("_gpu_decode_store", "_aux_gpu_decode_store")',
            '_model._aux_gpu_decode_stores = []',
            '_model.gpu_prefill_managers.clear()',
        ):
            self.assertGreater(shutdown_tail.index(fragment), release)
        for fragment in ("gpu_store = None", "store = None", "aux_store = None"):
            self.assertGreater(shutdown_tail.index(fragment), release)

    def test_runtime_signal_handler_preserves_shutdown_diagnostics(self):
        source = (ROOT / "python/krasis/server.py").read_text()
        handler_start = source.index("    def _handle_exit(sig, frame):")
        handler_end = source.index("\n    signal.signal(signal.SIGINT", handler_start)
        handler = source[handler_start:handler_end]
        self.assertIn("rust_server.stop()", handler)
        self.assertNotIn("sys.stdout", handler)
        self.assertNotIn("sys.stderr", handler)
        self.assertNotIn("logging.disable", handler)
        self.assertNotIn("os._exit", handler)

    def test_shutdown_observations_cover_every_resource_boundary(self):
        source = (ROOT / "python/krasis/server.py").read_text()
        for phase in (
            "post_server_report",
            "vram_monitor_stopped",
            "inductor_stopped",
            "rust_runtime_released",
            "python_model_collected",
            "first_cuda_cleanup_complete",
            "main_return",
            "main_frame_released",
            "final_cuda_cleanup_complete",
        ):
            self.assertIn(f'_shutdown_observation("{phase}")', source)

    def test_largest_mapping_inventory_is_ordered_and_bounded(self):
        maps = """1000-3000 rw-p 00000000 00:00 0
4000-5000 r-xp 00000000 00:00 0 /tmp/small mapping
8000-c000 rw-s 00000000 00:00 0 /tmp/largest
"""
        with mock.patch("builtins.open", mock.mock_open(read_data=maps)):
            summary = server._largest_memory_mappings(limit=2)
        self.assertEqual(
            summary,
            "16k:rw-s:/tmp/largest,8k:rw-p:[anonymous]",
        )


if __name__ == "__main__":
    unittest.main()
