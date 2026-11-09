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
        if prefill is None:
            prefill = {"format": "prefill_mla_f32_v1", "sealed": True, "plans": []}
        model = SimpleNamespace(cfg=SimpleNamespace(is_glm5_next=glm),
            _gpu_decode_store=SimpleNamespace(native_scalar_runtime_identity=lambda: None,
                native_prefill_mla_identity=lambda: json.dumps(prefill)))
        with patch.object(server, '_runtime_heatmap_capture_config', return_value={'base': 1}):
            return server._actual_runtime_heatmap_capture_config(model, SimpleNamespace())

    def setUp(self):
        self.compiled = dict(COLD_EXPERT_PRUNING=True, CPU_MARLIN_READER=False,
            CPU_COLD_LANE=False, EXACT_ATTENTION=True, ORDERED_ABSORPTION=True,
            PREFILL_F32=True, INDEXER_QUERY_TILE=True, INDEX_SCORE_ROW_BANDS=True,
            INDEX_RADIX_TOPK=True, KDA_RELAY=True,
            VALUE_REASSOCIATION=False, TENSOR_DECODE_ATTENTION=False, TENSOR_ABSORPTION=False)
        self.patcher = patch.object(server, '_compiled_optimisations', side_effect=lambda: dict(self.compiled))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_compiled_set_cannot_be_ignored(self):
        baseline = self.capture(False)
        for name in self.compiled:
            original = self.compiled[name]
            self.compiled[name] = not original
            current = self.capture(False)
            self.assertNotEqual(current, baseline)
            self.assertFalse(server._runtime_matches_approved_heatmap_policy(
                current, [baseline], {'ignored_runtime_fields': ['compiled_optimisations']})[0])
            self.compiled[name] = original

    def test_retired_runtime_controls_cannot_change_build(self):
        baseline = self.capture(False)
        for key in ['KRASIS_PREFILL_MLA_F32', 'KRASIS_INDEXER_QUERY_TILE',
                    'KRASIS_PREFILL_INDEX_SCORE_ROW_BANDS', 'KRASIS_PREFILL_MLA_VALUE_REASSOC',
                    'KRASIS_MLA_K6_SPLIT', 'KRASIS_PREFILL_MLA_ABSORB']:
            with patch.dict(os.environ, {key: 'invalid'}):
                self.assertEqual(self.capture(False), baseline)

    def test_prepared_plans_must_be_sealed_and_identity_bound(self):
        identity = {'format': 'prefill_mla_f32_v1', 'sealed': True, 'plans': []}
        actual = self.capture(False, identity)
        changed = {**actual, 'prefill_mla_f32_identity': {**identity, 'plans': [{'plan': 'different'}]}}
        self.assertFalse(server._runtime_matches_approved_heatmap_policy(
            changed, [actual], {'ignored_runtime_fields': ['prefill_mla_f32_identity']})[0])
        with self.assertRaisesRegex(RuntimeError, 'not finalized'):
            self.capture(False, {**identity, 'sealed': False})
        self.compiled['PREFILL_F32'] = False
        self.assertNotIn('prefill_mla_f32_identity', self.capture(False))

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

    def test_retired_cpu_calibration_cannot_select_or_spoof_native_build(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            baseline = self.capture()
            artifact = Path(directory) / 'calibration.json'
            artifact.write_text('{"team_threads":[8,8]}')
            os.environ['KRASIS_CPU_TAIL_CALIBRATION_JSON'] = str(artifact)
            os.environ['KRASIS_CPU_TAIL_RACE'] = '1'
            os.environ['KRASIS_CPU_TAIL_TRANSPOSED'] = '1'
            os.environ['KRASIS_CPU_TAIL_WORKERS'] = '2'
            self.assertEqual(self.capture(), baseline)
            artifact.unlink()  # Production must not read a retired calibration path.
            self.assertEqual(self.capture(), baseline)
            self.compiled['CPU_COLD_LANE'] = True
            current = self.capture()
            self.assertNotEqual(current, baseline)
            self.assertFalse(server._runtime_matches_approved_heatmap_policy(
                current, [baseline], {'ignored_runtime_fields':['compiled_optimisations']})[0])

    def test_runtime_code_fingerprint_includes_parameter_loader(self):
        files=server._krasis_runtime_code_hash()['files']
        self.assertIn('python/krasis/weight_loader.py',files)
        self.assertIn('src/prefill_transfer.rs',files)
        for path in ['src/optimisations.rs','src/cpu_tail.rs','src/cpu_cold_lane.rs',
                     'src/kernel/avx2.rs','src/weights/marlin.rs']:
            self.assertIn(path, files)

if __name__=='__main__':unittest.main()
