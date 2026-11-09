"""No-GPU negative controls for experimental GLM heatmap treatment identity."""
import os
import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from krasis import server

class Glm53HeatmapControlsTest(unittest.TestCase):
    def capture(self, glm=True, prefill=None):
        model = SimpleNamespace(cfg=SimpleNamespace(is_glm5_next=glm),
            _gpu_decode_store=SimpleNamespace(native_scalar_runtime_identity=lambda: None,
                native_prefill_mla_identity=lambda: json.dumps(prefill)))
        with patch.object(server, '_runtime_heatmap_capture_config', return_value={'base': 1}):
            return server._actual_runtime_heatmap_capture_config(model, SimpleNamespace())

    def test_indexer_treatment_identity_is_bound_and_cannot_be_ignored(self):
        with patch.dict(os.environ, {}, clear=True):
            baseline = self.capture(False)
        with patch.dict(os.environ, {"KRASIS_INDEXER_QUERY_TILE": "1"}, clear=True):
            candidate = self.capture(False)
        self.assertEqual(candidate["indexer_query_tile_control"], "1")
        self.assertNotEqual(candidate, baseline)
        self.assertFalse(server._runtime_matches_approved_heatmap_policy(
            candidate, [baseline], {"ignored_runtime_fields": ["indexer_query_tile_control"]})[0])

    def test_prefill_treatment_binds_non_glm_fallback_and_finalized_plans(self):
        identity = {"format": "prefill_mla_f32_v1", "sealed": True, "plans": []}
        with patch.dict(os.environ, {"KRASIS_PREFILL_MLA_F32": "1"}, clear=True):
            actual = self.capture(False, identity)
            self.assertEqual(actual["prefill_mla_f32_control"], "1")
            self.assertEqual(actual["prefill_mla_f32_identity"], identity)
            with self.assertRaisesRegex(RuntimeError, "not finalized"):
                self.capture(False, {**identity, "sealed": False})
            with self.assertRaisesRegex(RuntimeError, "not finalized"):
                self.capture(False, {**identity, "format": "unknown"})
        with patch.dict(os.environ, {"KRASIS_PREFILL_MLA_F32": "0"}, clear=True):
            self.assertEqual(self.capture(False), {"base": 1, "prefill_mla_f32_control": "0"})

    def test_prefill_boolean_aliases_require_finalized_identity(self):
        identity = {"format": "prefill_mla_f32_v1", "sealed": True, "plans": []}
        for raw in ["1", "true", "yes", "on", " TRUE ", " On "]:
            with self.subTest(raw=raw), patch.dict(os.environ, {"KRASIS_PREFILL_MLA_F32": raw}, clear=True):
                self.assertTrue(server._prefill_mla_f32_enabled())
                self.assertEqual(self.capture(False, identity)["prefill_mla_f32_identity"], identity)
                with self.assertRaisesRegex(RuntimeError, "not finalized"):
                    self.capture(False, {**identity, "sealed": False})
        for raw in ["0", "false", "no", "off", " FALSE ", " Off "]:
            with self.subTest(raw=raw), patch.dict(os.environ, {"KRASIS_PREFILL_MLA_F32": raw}, clear=True):
                self.assertFalse(server._prefill_mla_f32_enabled())
                self.assertEqual(self.capture(False), {"base": 1, "prefill_mla_f32_control": raw})

    def test_invalid_prefill_boolean_fails_before_loading(self):
        for raw in ["", "2", "enabled", "invalid"]:
            with self.subTest(raw=raw), patch.dict(os.environ, {"KRASIS_PREFILL_MLA_F32": raw}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "Invalid KRASIS_PREFILL_MLA_F32"):
                    server._prefill_mla_f32_enabled()

    def test_default_and_other_architectures_keep_compatibility_shape(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.capture()['prefill_transfer_policy'], {
                'version': 4, 'phase_arena': '1', 'shared_source_reclaim': '1', 'hqq_packed_components': '1', 'joint_planner': '1', 'compact_pure_kda': '1', 'shared_overlap_eligible': '1'})
        with patch.dict(os.environ, {'KRASIS_GLM53_PRESERVE_NUMERIC_F32': '1'}, clear=True):
            self.assertEqual(self.capture(False), {'base': 1})

    def test_control_switch_rejects_other_treatment_even_under_ignore_policy(self):
        for key, value in [('KRASIS_PREFILL_PHASE_ARENA', '0'),
                           ('KRASIS_RECLAIM_SHARED_SOURCES', '0'),
                           ('KRASIS_HQQ_PACK_COMPONENTS', '0'),
                           ('KRASIS_PREFILL_TRANSFER_PLAN', '0'),
                           ('KRASIS_GLM53_PRESERVE_NUMERIC_F32', '1'),
                           ('KRASIS_DEEPSEEK_V4_PREFILL_HC_PROJECT_MODE', 'fp32_pedantic_gemm'),
                           ('KRASIS_GLM53_KDA_SPLIT_CANDIDATE', '1'),
                           ('KRASIS_PREFILL_COMPACT_KDA_SCRATCH', '0'),
                           ('KRASIS_PREFILL_SHARED_OVERLAP_DMA', '0'),
                           ('KRASIS_GLM53_KDA_SPLIT_PREPARATION', 'bf16_qk_beta'),
                           ('KRASIS_MLA_HOST_BACKED_BACKEND', 'vmm'),
                           ('KRASIS_MLA_HOST_BACKED_WINDOW_TOKENS', '32768'),
                           ('KRASIS_GLM53_FUSED_MLA_REUSE_CANDIDATE', '1'),
                           ('KRASIS_MLA_STAGE_SELECTED_KV', '1'),
                           ('KRASIS_CPU_TAIL_HOST_TIMING', '1'),
                           ('KRASIS_CPU_TAIL_RACE', '1'),
                           ('KRASIS_CPU_TAIL_TRANSPOSED', '1'),
                           ('KRASIS_CPU_TAIL_WORKERS', '2'),
                           ('KRASIS_SPLIT_EXPERT_LAUNCH', '1'),
                           ('KRASIS_GPU_ROUTE_SYNC', '0'),
                           ('KRASIS_PREFETCH', '0')]:
            with self.subTest(key=key), patch.dict(os.environ, {}, clear=True):
                baseline=self.capture()
                os.environ[key]=value; current=self.capture()
                self.assertNotEqual(current, baseline)
                self.assertTrue(server._runtime_matches_approved_heatmap_policy(current,[current],None)[0])
                self.assertFalse(server._runtime_matches_approved_heatmap_policy(current,[baseline],None)[0])
                self.assertFalse(server._runtime_matches_approved_heatmap_policy(current,[baseline],
                    {'ignored_runtime_fields':['experimental_glm53_controls']})[0])

    def test_transfer_policy_cannot_be_ignored(self):
        with patch.dict(os.environ, {}, clear=True):
            current = self.capture()
            old = {'base': 1}
            self.assertFalse(server._runtime_matches_approved_heatmap_policy(
                current, [old], {'ignored_runtime_fields': ['prefill_transfer_policy']})[0])

    def test_cpu_calibration_content_changes_invalidate_same_path(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            artifact = Path(directory) / 'calibration.json'
            artifact.write_text('{"team_threads":[8,8]}')
            os.environ['KRASIS_CPU_TAIL_CALIBRATION_JSON'] = str(artifact)
            original = self.capture()
            artifact.write_text('{"team_threads":[16,16]}')
            current = self.capture()
            self.assertNotEqual(current, original)
            self.assertFalse(server._runtime_matches_approved_heatmap_policy(
                current, [original], {'ignored_runtime_fields':['experimental_glm53_controls']})[0])

    def test_runtime_code_fingerprint_includes_parameter_loader(self):
        files=server._krasis_runtime_code_hash()['files']
        self.assertIn('python/krasis/weight_loader.py',files)
        self.assertIn('src/prefill_transfer.rs',files)

if __name__=='__main__':unittest.main()
