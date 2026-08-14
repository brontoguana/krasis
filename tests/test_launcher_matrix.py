import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from krasis.attention_backend import quantize_hqq4_tensor_rust
from krasis import chat as chat_mod
from krasis import console_input as console_input_mod
from krasis.config import configure_adaptive_cold_mass_pruning
from krasis import launcher as launcher_mod
from krasis import nvidia_smi as nvidia_smi_mod
from krasis import vram_budget as vram_budget_mod
from krasis.launcher import Launcher, LauncherConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
NONEXISTENT_MODEL = "/tmp/nonexistent-krasis-launcher-matrix-model"


def _parse_key_value_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@contextlib.contextmanager
def _patched_launcher_env():
    old_detect = launcher_mod.detect_hardware
    old_execvp = os.execvp
    old_krasis_home = os.environ.get("KRASIS_HOME")
    with tempfile.TemporaryDirectory(prefix="krasis-launcher-home-") as home:
        launcher_mod.detect_hardware = lambda: {
            "gpu_count": 2,
            "gpus": [
                {"index": 0, "name": "Test GPU 0", "memory_total_mb": 24_000},
                {"index": 1, "name": "Test GPU 1", "memory_total_mb": 12_000},
            ],
        }
        os.environ["KRASIS_HOME"] = home
        try:
            yield
        finally:
            launcher_mod.detect_hardware = old_detect
            os.execvp = old_execvp
            if old_krasis_home is None:
                os.environ.pop("KRASIS_HOME", None)
            else:
                os.environ["KRASIS_HOME"] = old_krasis_home


class _ExecIntercept(Exception):
    def __init__(self, path: str, args: list[str]):
        super().__init__(path, args)
        self.path = path
        self.args = args


def _capture_launch_config(cfg: LauncherConfig, *, benchmark: bool = True) -> tuple[Path, list[str]]:
    args = argparse.Namespace()
    with _patched_launcher_env():
        launcher = Launcher(args)
        launcher.cfg = cfg
        launcher.selected_gpus = [
            {"index": idx, "name": f"Test GPU {idx}", "memory_total_mb": 24_000}
            for idx in cfg.selected_gpu_indices
        ]
        if not launcher.selected_gpus:
            launcher.selected_gpus = [{"index": 0, "name": "Test GPU 0", "memory_total_mb": 24_000}]

        def fake_execvp(path: str, argv: list[str]) -> None:
            raise _ExecIntercept(path, list(argv))

        os.execvp = fake_execvp
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                launcher.launch_server(benchmark=benchmark)
        except _ExecIntercept as exc:
            config_path = Path(exc.args[exc.args.index("--config") + 1])
            return config_path, exc.args
    raise AssertionError("launcher.launch_server() did not exec")


def _base_config() -> LauncherConfig:
    cfg = LauncherConfig()
    cfg.model_path = NONEXISTENT_MODEL
    cfg.selected_gpu_indices = [0]
    cfg.host = "127.0.0.1"
    cfg.port = 65_501
    cfg.krasis_threads = 8
    return cfg


def _run_server_start_smoke(config_path: Path, scenario: str, expected_fragments: list[str]) -> str:
    with tempfile.TemporaryDirectory(prefix=f"krasis-{scenario}-run-") as run_dir:
        env = os.environ.copy()
        env["KRASIS_RUN_DIR"] = run_dir
        env["KRASIS_RUN_TYPE"] = f"launcher-matrix-{scenario}"
        python_path = str(REPO_ROOT / "python")
        env["PYTHONPATH"] = f"{python_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else python_path
        proc = subprocess.run(
            [sys.executable, "-m", "krasis.server", "--config", str(config_path), "--benchmark"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
            check=False,
        )
        log_path = Path(run_dir) / "krasis.log"
        log_output = log_path.read_text() if log_path.exists() else ""
    output = proc.stdout + "\n" + log_output
    if proc.returncode == 0:
        raise AssertionError(f"{scenario}: expected nonexistent model failure, got success\n{output}")
    forbidden = [
        "server.py: error: argument",
        "invalid float value",
        "invalid int value",
        "invalid choice",
        "usage: server.py",
    ]
    for text in forbidden:
        if text in output:
            raise AssertionError(f"{scenario}: server failed before launch on {text!r}\n{output}")
    required = [
        "=== Resolved arguments ===",
        f"model_path = '{NONEXISTENT_MODEL}'",
        "config.json",
    ] + expected_fragments
    for text in required:
        if text not in output:
            raise AssertionError(f"{scenario}: missing expected output {text!r}\n{output}")
    return output


class LauncherMatrixTest(unittest.TestCase):
    def test_layer_group_one_survives_saved_and_cli_launcher_paths(self) -> None:
        cfg = LauncherConfig()
        cfg.apply_saved({"CFG_LAYER_GROUP_SIZE": "1"})
        self.assertEqual(cfg.layer_group_size, 1)

        with mock.patch.object(
            sys,
            "argv",
            ["krasis", "--layer-group-size", "1"],
        ):
            args = launcher_mod.parse_args()
        cli_cfg = LauncherConfig()
        launcher_mod._apply_cli_overrides(cli_cfg, args)
        self.assertEqual(cli_cfg.layer_group_size, 1)

        layer_group_option = next(
            option for option in launcher_mod.OPTIONS
            if option.key == "layer_group_size"
        )
        self.assertIn(1, layer_group_option.choices)
        self.assertEqual(
            launcher_mod._format_value(layer_group_option, 1),
            "1 layer (double-buffered)",
        )

        for value in ("0", "-1", "not-an-integer"):
            with self.subTest(saved_value=value):
                invalid_cfg = LauncherConfig()
                with self.assertRaisesRegex(
                    ValueError,
                    "CFG_LAYER_GROUP_SIZE",
                ):
                    invalid_cfg.apply_saved({"CFG_LAYER_GROUP_SIZE": value})

        with mock.patch.object(
            sys,
            "argv",
            ["krasis", "--layer-group-size", "0"],
        ):
            invalid_args = launcher_mod.parse_args()
        with self.assertRaisesRegex(ValueError, "--layer-group-size"):
            launcher_mod._apply_cli_overrides(LauncherConfig(), invalid_args)

    def test_launcher_kv_capacity_uses_configured_allocation(self) -> None:
        rank = {"kv_tokens": 1_048_576, "kv_alloc_tokens": 294_432}
        self.assertEqual(launcher_mod._allocated_kv_tokens(rank), 294_432)
        self.assertEqual(
            launcher_mod._allocated_kv_tokens({"kv_tokens": 149_808}),
            149_808,
        )

    def test_native_windows_ram_detection_uses_global_memory_status(self) -> None:
        import ctypes

        class Kernel32:
            @staticmethod
            def GlobalMemoryStatusEx(status_ptr) -> int:
                status_ptr._obj.ullTotalPhys = 192 * 1024**3
                return 1

        fake_windll = type("FakeWindll", (), {"kernel32": Kernel32()})()
        with mock.patch.object(ctypes, "windll", fake_windll, create=True):
            self.assertEqual(
                vram_budget_mod._detect_windows_total_ram_gb(),
                192,
            )

    def test_deepseek_v4_launcher_capabilities_are_model_specific(self) -> None:
        launcher = Launcher.__new__(Launcher)
        launcher.cfg = LauncherConfig()
        launcher.model_info = {"name": "DeepSeek-V4", "arch": "deepseek_v4"}

        launcher._apply_model_recommended_defaults()
        self.assertEqual(launcher.cfg.attention_quant, "hqq6")
        self.assertEqual(launcher.cfg.kv_dtype, "native")
        self.assertEqual(
            launcher._attention_choices(),
            ["hqq4", "hqq46_auto", "hqq6", "hqq68_auto", "hqq8", "bf16"],
        )
        self.assertEqual(launcher._kv_choices(), ["native", "bf16"])
        self.assertEqual(launcher._multi_gpu_choices(), ["auto"])
        kv_option = next(
            option for option in launcher_mod.OPTIONS if option.key == "kv_dtype"
        )
        self.assertEqual(launcher_mod._format_value(kv_option, "native"), "Native")
        launcher._validate_model_capabilities()

        for supported_attention in ("hqq4", "hqq46_auto", "hqq6", "hqq68_auto", "hqq8"):
            launcher.cfg.attention_quant = supported_attention
            launcher._validate_model_capabilities()
        launcher.cfg.attention_quant = "hqq8"
        launcher.cfg.kv_dtype = "k6v6"
        with self.assertRaisesRegex(ValueError, "does not support cache mode"):
            launcher._validate_model_capabilities()

        launcher.cfg.kv_dtype = "native"
        launcher.cfg.multi_gpu_mode = "peer"
        with self.assertRaisesRegex(ValueError, "does not support topology mode"):
            launcher._validate_model_capabilities()
        launcher.cfg.multi_gpu_mode = "auto"
        launcher.cfg.selected_gpu_indices = [0, 1]
        self.assertFalse(launcher._ensure_interactive_attention_ready())
        with self.assertRaisesRegex(ValueError, "multi-GPU execution is not yet launcher-qualified"):
            launcher._validate_model_topology()
        launcher.cfg.multi_gpu_mode = "peer"
        with self.assertRaisesRegex(ValueError, "peer expert serving"):
            launcher._validate_model_topology()

        launcher.cfg.selected_gpu_indices = [0]
        launcher.cfg.multi_gpu_mode = "auto"
        launcher.cfg.attention_quant = "bf16"
        launcher.cfg._attention_quant_explicit = True
        launcher.cfg.kv_dtype = "bf16"
        launcher.cfg._kv_dtype_explicit = True
        launcher._apply_model_recommended_defaults()
        self.assertEqual(launcher.cfg.attention_quant, "bf16")
        self.assertEqual(launcher.cfg.kv_dtype, "bf16")
        self.assertTrue(launcher._ensure_interactive_attention_ready())

    def test_gemma4_launcher_exposes_validated_mixed_hqq_presets(self) -> None:
        launcher = Launcher.__new__(Launcher)
        launcher.cfg = LauncherConfig()
        launcher.model_info = {"name": "Gemma4", "arch": "gemma4_text"}

        self.assertEqual(
            launcher._attention_choices(),
            ["hqq4", "hqq46_auto", "hqq6", "hqq68_auto"],
        )
        for supported_attention in ("hqq46_auto", "hqq68_auto"):
            launcher.cfg.attention_quant = supported_attention
            launcher._validate_model_capabilities()

    def test_every_download_catalog_entry_drives_launcher_capabilities(self) -> None:
        from krasis.hf_downloader import supported_models

        measured_topologies = {
            "qcn": ("auto", "peer"),
            "qwen35-35b": ("auto", "layer-split"),
            "qwen35-122b": ("auto", "peer"),
            "glm52": ("auto", "peer"),
        }
        expert_option = next(
            option for option in launcher_mod.OPTIONS
            if option.key == "gpu_expert_bits"
        )
        self.assertEqual(expert_option.choices, [4])

        for spec in supported_models():
            with self.subTest(model=spec.key):
                launcher = Launcher.__new__(Launcher)
                launcher.cfg = LauncherConfig()
                launcher.model_info = {
                    "name": spec.display_name,
                    "path": f"/models/{spec.local_dir_name}",
                    "arch": "catalog-test",
                    "support_key": spec.key,
                }
                launcher._apply_model_recommended_defaults()
                self.assertEqual(launcher.cfg.attention_quant, spec.default_attention)
                self.assertEqual(launcher.cfg.kv_dtype, spec.default_kv)
                self.assertEqual(launcher._attention_choices(), list(spec.attention_modes))
                self.assertEqual(launcher._kv_choices(), list(spec.kv_modes))
                self.assertEqual(launcher._multi_gpu_choices(), list(spec.multi_gpu_modes))
                self.assertEqual(
                    spec.multi_gpu_modes,
                    measured_topologies.get(spec.key, ("auto",)),
                )
                self.assertEqual(
                    spec.multi_gpu_qualified,
                    spec.key in measured_topologies,
                )
                if spec.max_context_tokens:
                    self.assertEqual(
                        launcher.cfg.max_context_tokens,
                        spec.max_context_tokens,
                    )
                launcher._validate_model_capabilities()
                for attention, kv in spec.runtime_profiles:
                    launcher.cfg.attention_quant = attention
                    launcher.cfg.kv_dtype = kv
                    launcher._validate_model_capabilities()

                unmeasured = next(
                    (
                        (attention, kv)
                        for attention in spec.attention_modes
                        for kv in spec.kv_modes
                        if (attention, kv) not in spec.runtime_profiles
                    ),
                    None,
                )
                if unmeasured is not None:
                    launcher.cfg.attention_quant, launcher.cfg.kv_dtype = unmeasured
                    with self.assertRaisesRegex(ValueError, "has not been launcher-qualified"):
                        launcher._validate_model_capabilities()

                launcher.cfg.attention_quant = spec.default_attention
                launcher.cfg.kv_dtype = spec.default_kv
                launcher.cfg.selected_gpu_indices = [0, 1]
                if spec.multi_gpu_qualified:
                    launcher._validate_model_topology()
                else:
                    with self.assertRaisesRegex(ValueError, "not yet launcher-qualified"):
                        launcher._validate_model_topology()
                launcher.cfg.selected_gpu_indices = []

                launcher.cfg.gpu_expert_bits = 8
                launcher.cfg.cpu_expert_bits = 8
                with self.assertRaisesRegex(ValueError, "INT4 experts only"):
                    launcher._validate_model_capabilities()

    def test_qcn_release_matrix_uses_launcher_qualified_profiles(self) -> None:
        from krasis.hf_downloader import supported_model_spec
        from tests.release_test import CONFIG_VARIANTS

        spec = supported_model_spec("qcn")
        for variant in CONFIG_VARIANTS:
            with self.subTest(variant=variant["name"]):
                self.assertEqual(variant["bits"], 4)
                self.assertIn(
                    (variant["attention"], variant["kv"]),
                    spec.runtime_profiles,
                )
                if variant.get("multi_gpu"):
                    self.assertIn("peer", spec.multi_gpu_modes)

    def test_profile_cycles_keep_catalog_pairs_qualified(self) -> None:
        from krasis.hf_downloader import supported_model_spec

        launcher = Launcher.__new__(Launcher)
        launcher.cfg = LauncherConfig()
        spec = supported_model_spec("qcn")
        launcher.model_info = {
            "name": spec.display_name,
            "path": f"/models/{spec.local_dir_name}",
            "arch": "qwen3_next",
            "support_key": spec.key,
        }
        launcher._apply_model_recommended_defaults()
        attention_option = next(
            option for option in launcher_mod.OPTIONS
            if option.key == "attention_quant"
        )
        launcher._cycle_value(attention_option, 1)
        self.assertEqual(
            (launcher.cfg.attention_quant, launcher.cfg.kv_dtype),
            ("hqq6", "k6v6"),
        )
        launcher._validate_model_capabilities()

    def test_every_catalog_entry_is_selectable_in_download_screen(self) -> None:
        from krasis.hf_downloader import supported_models

        launcher = Launcher.__new__(Launcher)
        candidates = []
        for spec in supported_models():
            candidate = mock.Mock()
            candidate.display_name = spec.display_name
            candidate.repo_id = spec.repo_id
            candidate.support_notes = spec.notes
            candidate.runtime_ram_bytes = 1024**3
            candidate.local_dir_name = spec.local_dir_name
            candidate.metadata_error = ""
            candidate.recommended_config = spec.recommended_config
            candidate.revision = spec.revision
            candidates.append(candidate)

        for target_idx, candidate in enumerate(candidates):
            with self.subTest(model=candidate.repo_id):
                keys = [launcher_mod.KEY_DOWN] * target_idx + [launcher_mod.KEY_ENTER]
                with mock.patch.object(launcher_mod, "_read_key", side_effect=keys):
                    with contextlib.redirect_stdout(io.StringIO()):
                        selected = launcher._supported_hf_models_screen(candidates)
                self.assertIs(selected, candidate)

    def test_glm52_launcher_rejects_unqualified_context_before_load(self) -> None:
        launcher = Launcher.__new__(Launcher)
        launcher.cfg = LauncherConfig()
        launcher.cfg.max_context_tokens = 8192
        launcher.model_info = {
            "name": "GLM-5.2",
            "path": "/models/GLM-5.2",
            "arch": "glm_moe_dsa",
            "support_key": "glm52",
        }
        launcher._apply_model_recommended_defaults()
        with self.assertRaisesRegex(ValueError, "qualified only through 4,096"):
            launcher._validate_model_capabilities()

    def test_launcher_metadata_and_budget_use_normalized_step_and_nemotron_layers(self) -> None:
        step_cfg = {
            "model_type": "step3p5",
            "hidden_size": 128,
            "intermediate_size": 256,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_attention_groups": 2,
            "head_dim": 32,
            "vocab_size": 1024,
            "moe_num_experts": 8,
            "moe_top_k": 2,
            "moe_layers_enum": "1,2,3",
            "share_expert_dim": 64,
            "layer_types": [
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            "attention_other_setting": {
                "attention_type": "sliding_attention",
                "num_attention_heads": 6,
                "num_attention_groups": 2,
                "head_dim": 32,
            },
            "max_position_embeddings": 4096,
            "tie_word_embeddings": True,
        }
        nemotron_cfg = {
            "model_type": "nemotron_h",
            "hidden_size": 128,
            "intermediate_size": 256,
            "moe_intermediate_size": 64,
            "mlp_hidden_act": "relu2",
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 1024,
            "num_local_experts": 8,
            "experts_per_token": 2,
            "hybrid_override_pattern": "MEM*",
            "mamba_num_heads": 4,
            "mamba_head_dim": 32,
            "ssm_state_size": 16,
            "expand": 2,
            "conv_kernel": 4,
            "n_groups": 1,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": True,
        }
        with tempfile.TemporaryDirectory(prefix="krasis-launcher-normalized-") as root:
            for name, config in (("Step-3.7-Flash", step_cfg), ("Nemotron", nemotron_cfg)):
                model_dir = Path(root) / name
                model_dir.mkdir()
                (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

            step_path = str(Path(root) / "Step-3.7-Flash")
            step_info = launcher_mod._model_info_from_path(step_path)
            self.assertEqual(step_info["experts"], 8)
            self.assertEqual(step_info["moe_layers"], 3)
            self.assertEqual(step_info["dense_layers"], 1)
            self.assertGreater(step_info["ram_gb"], 0)
            step_budget = vram_budget_mod.compute_launcher_budget(
                step_path,
                [4],
                attention_quant="hqq4",
                kv_dtype="k4v4",
                gpu_vram_mb=24_000,
                total_ram_gb=128,
                kv_cache_mb=200,
            )
            self.assertEqual(step_budget["n_experts"], 8)
            self.assertEqual(step_budget["num_moe_layers"], 3)
            self.assertEqual(step_budget["num_full_attention_layers"], 4)
            self.assertGreater(step_budget["ram_total_mb"], 0)

            nemotron_path = str(Path(root) / "Nemotron")
            nemotron_info = launcher_mod._model_info_from_path(nemotron_path)
            self.assertEqual(nemotron_info["experts"], 8)
            self.assertEqual(nemotron_info["moe_layers"], 1)
            self.assertEqual(nemotron_info["num_kv_layers"], 1)
            nemotron_budget = vram_budget_mod.compute_launcher_budget(
                nemotron_path,
                [4],
                attention_quant="hqq4",
                kv_dtype="k4v4",
                gpu_vram_mb=24_000,
                total_ram_gb=128,
                kv_cache_mb=200,
            )
            self.assertEqual(nemotron_budget["num_moe_layers"], 1)
            self.assertEqual(nemotron_budget["num_full_attention_layers"], 1)
            nemotron_model_cfg = vram_budget_mod.ModelConfig.from_model_path(
                nemotron_path
            )
            params_per_expert = 2 * 128 * 64
            bytes_per_expert = params_per_expert // 2 + (params_per_expert // 128) * 2
            self.assertEqual(
                vram_budget_mod.estimate_int4_expert_cache_bytes(
                    nemotron_model_cfg
                ),
                bytes_per_expert * 8,
            )
            self.assertGreater(
                nemotron_budget["ranks"][0]["recurrent_state_mb"],
                0,
            )

    def test_linear_attention_recurrent_budget_matches_runtime_state_shapes(self) -> None:
        cfg = mock.Mock()
        cfg.is_linear_attention_layer.side_effect = lambda layer_idx: layer_idx == 0
        cfg.is_mamba2_layer.return_value = False
        cfg.linear_num_key_heads = 2
        cfg.linear_num_value_heads = 4
        cfg.linear_key_head_dim = 8
        cfg.linear_value_head_dim = 16
        cfg.linear_conv_kernel_dim = 3
        expected_conv_bytes = (2 * 2 * 8 + 4 * 16) * 3 * 4
        expected_recurrent_bytes = 4 * 8 * 16 * 4
        self.assertEqual(
            vram_budget_mod._persistent_recurrent_state_bytes(cfg, 0, 1),
            expected_conv_bytes + expected_recurrent_bytes,
        )

    def test_deepseek_v4_native_budget_uses_exact_nonlinear_layout(self) -> None:
        cfg = {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 3,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "index_head_dim": 128,
            "sliding_window": 128,
            # Checkpoint configs can include auxiliary entries after the
            # runtime transformer layers; the estimator must mirror the
            # loader and ignore only that validated suffix.
            "compress_ratios": [0, 4, 128, 0, 0],
        }
        tokens = 4096
        native = vram_budget_mod._deepseek_v4_rank_cache_bytes(
            cfg, 0, 3, tokens, "native"
        )
        bf16 = vram_budget_mod._deepseek_v4_rank_cache_bytes(
            cfg, 0, 3, tokens, "bf16"
        )
        self.assertLess(native, bf16)
        self.assertEqual(
            vram_budget_mod._deepseek_v4_tokens_for_rank_budget(
                cfg, 0, 3, native, tokens, "native"
            ),
            tokens,
        )
        self.assertLess(
            vram_budget_mod._deepseek_v4_tokens_for_rank_budget(
                cfg, 0, 3, native - 1, tokens, "native"
            ),
            tokens,
        )

    def test_deepseek_v4_hqq_fallback_uses_real_phase_one_shapes(self) -> None:
        cfg = {
            "model_type": "deepseek_v4",
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "head_dim": 512,
            "q_lora_rank": 1024,
            "o_lora_rank": 1024,
        }
        self.assertEqual(
            vram_budget_mod._hqq_attention_tensor_shapes_for_layer(
                cfg, "full_attention", 0
            ),
            [
                ("wq_a", 1024, 4096),
                ("wq_b", 32768, 1024),
                ("wkv", 512, 4096),
                ("wo_b", 4096, 1024),
            ],
        )

    def test_dev_benchmark_cleanup_uses_selected_gpu_override(self):
        dev_source = (REPO_ROOT / "dev").read_text(encoding="utf-8")
        benchmark_start = dev_source.index("do_benchmark() {")
        benchmark_end = dev_source.index("\nsummarize_dynamic_hcs_from_log() {", benchmark_start)
        benchmark_source = dev_source[benchmark_start:benchmark_end]

        self.assertIn(
            'selected_gpus_override=$(extract_selected_gpus_arg "$@")',
            benchmark_source,
        )
        self.assertIn(
            'cleanup_gpu "$conf" "$selected_gpus_override"',
            benchmark_source,
        )
        self.assertNotIn('cleanup_gpu "$conf"\n', benchmark_source)

    def test_dev_reference_cleanup_uses_selected_gpu_override(self):
        dev_source = (REPO_ROOT / "dev").read_text(encoding="utf-8")
        reference_start = dev_source.index("do_reference_test() {")
        reference_end = dev_source.index("\ndo_reference_inventory() {", reference_start)
        reference_source = dev_source[reference_start:reference_end]

        self.assertIn(
            'selected_gpus_override=$(extract_selected_gpus_arg "$@")',
            reference_source,
        )
        self.assertIn(
            'cleanup_gpu "$conf" "$selected_gpus_override"',
            reference_source,
        )
        self.assertIn(
            'KRASIS_REFERENCE_SELECTED_GPUS="$selected_gpus_override"',
            reference_source,
        )
        self.assertIn('reference_args+=("$1")', reference_source)
        self.assertNotIn('cleanup_gpu "$conf"\n', reference_source)

    maxDiff = None

    def test_native_windows_console_key_decoding(self) -> None:
        for encoded, expected in (
            (["\xe0", "H"], launcher_mod.KEY_UP),
            (["\xe0", "P"], launcher_mod.KEY_DOWN),
            (["\x00", "K"], launcher_mod.KEY_LEFT),
            (["\x00", "M"], launcher_mod.KEY_RIGHT),
            (["\r"], launcher_mod.KEY_ENTER),
            (["\x1b"], launcher_mod.KEY_ESCAPE),
            (["\x03"], launcher_mod.KEY_ESCAPE),
            (["\x08"], launcher_mod.KEY_BACKSPACE),
            (["q"], launcher_mod.KEY_QUIT),
        ):
            chars = iter(encoded)
            self.assertEqual(
                console_input_mod.read_windows_key(lambda: next(chars)),
                expected,
            )

    def test_native_windows_console_timeout_and_launcher_chat_wiring(self) -> None:
        self.assertIsNone(
            console_input_mod.read_windows_key_timeout(
                0.0,
                key_available=lambda: False,
            )
        )

        old_launcher_flag = launcher_mod._HAS_WINDOWS_CONSOLE
        old_launcher_reader = launcher_mod._read_windows_key
        old_chat_flag = chat_mod._HAS_WINDOWS_CONSOLE
        old_chat_reader = chat_mod._read_windows_key_native
        try:
            launcher_mod._HAS_WINDOWS_CONSOLE = True
            launcher_mod._read_windows_key = lambda: launcher_mod.KEY_RIGHT
            chat_mod._HAS_WINDOWS_CONSOLE = True
            chat_mod._read_windows_key_native = lambda: chat_mod.KEY_DOWN
            self.assertEqual(launcher_mod._read_key(), launcher_mod.KEY_RIGHT)
            self.assertEqual(chat_mod._read_key(), chat_mod.KEY_DOWN)
        finally:
            launcher_mod._HAS_WINDOWS_CONSOLE = old_launcher_flag
            launcher_mod._read_windows_key = old_launcher_reader
            chat_mod._HAS_WINDOWS_CONSOLE = old_chat_flag
            chat_mod._read_windows_key_native = old_chat_reader

    def test_native_windows_console_input_never_mutates_mouse_modes(self) -> None:
        self.assertFalse(
            hasattr(console_input_mod, "windows_console_key_mode"),
            "native character input must not call SetConsoleMode",
        )
        source = (REPO_ROOT / "python" / "krasis" / "console_input.py").read_text()
        for forbidden in (
            "SetConsoleMode",
            "GetConsoleMode",
            "ENABLE_MOUSE_INPUT",
            "ENABLE_QUICK_EDIT_MODE",
        ):
            self.assertNotIn(forbidden, source)

    def test_generated_launch_config_is_strict_utf8(self) -> None:
        cfg = _base_config()
        cfg.model_path = f"{NONEXISTENT_MODEL}-\u6a21\u578b"
        config_path, _args = _capture_launch_config(cfg)
        try:
            raw = config_path.read_bytes()
            decoded = raw.decode("utf-8")
            self.assertIn("Krasis launch config \u2014", decoded)
            self.assertIn(f'MODEL_PATH="{cfg.model_path}"', decoded)
        finally:
            config_path.unlink(missing_ok=True)

    def test_cuda_warmup_uses_packaged_marlin_without_triton(self) -> None:
        model_source = (REPO_ROOT / "python" / "krasis" / "model.py").read_text()
        warmup_start = model_source.index("    def warmup_cuda_runtime(")
        warmup_end = model_source.index("    @staticmethod", warmup_start)
        warmup_source = model_source[warmup_start:warmup_end]
        self.assertIn("from krasis.marlin_utils import", warmup_source)
        self.assertIn("for device in devices:", warmup_source)
        self.assertIn("self.quant_cfg.expert_group_size", warmup_source)
        self.assertIn("gptq_marlin_gemm(", warmup_source)
        self.assertNotIn("gs = 128", warmup_source)
        self.assertNotIn("from krasis.triton_moe", warmup_source)
        self.assertNotIn("import triton", warmup_source)
        self.assertFalse(
            (REPO_ROOT / "python" / "krasis" / "triton_moe.py").exists()
        )

    def test_launcher_header_fills_terminal_width_and_shows_version(self) -> None:
        lines = launcher_mod._launcher_header_lines("9.8.7-test", width=96)
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertEqual(launcher_mod._visible_len(line), 96)
        self.assertIn("Krasis MoE Server v9.8.7-test", launcher_mod._ANSI_RE.sub("", lines[1]))

    def test_launcher_on_off_labels_are_consistent(self) -> None:
        bool_opt = launcher_mod.ConfigOption("Test bool", "test_bool")
        ssh_opt = launcher_mod.ConfigOption("SSH Tunnel", "ssh_tunnel", opt_type="text")

        enabled = launcher_mod._format_value(bool_opt, True)
        disabled = launcher_mod._format_value(bool_opt, False)
        ssh_disabled = launcher_mod._format_value(ssh_opt, "")

        self.assertEqual(launcher_mod._ANSI_RE.sub("", enabled), "On")
        self.assertEqual(launcher_mod._ANSI_RE.sub("", disabled), "Off")
        self.assertEqual(launcher_mod._ANSI_RE.sub("", ssh_disabled), "Off")
        self.assertIn(launcher_mod.GREEN, enabled)
        self.assertIn(launcher_mod.DIM, disabled)
        self.assertIn(launcher_mod.DIM, ssh_disabled)

    def test_adaptive_cold_mass_pruning_launcher_sequence_and_environment(self) -> None:
        cfg = LauncherConfig()
        opt = next(
            item for item in launcher_mod.OPTIONS
            if item.key == "adaptive_cold_mass_pruning"
        )
        self.assertEqual(cfg.adaptive_cold_mass_pruning, "off")
        self.assertEqual(cfg.to_save_dict()["CFG_ADAPTIVE_COLD_MASS_PRUNING"], "off")
        self.assertEqual(opt.choices, ["off", "75/3", "75/5", "75/8", "75/10"])

        launcher = Launcher.__new__(Launcher)
        launcher.cfg = cfg
        for expected in ("75/3", "75/5", "75/8", "75/10", "off"):
            launcher._cycle_value(opt, 1)
            self.assertEqual(cfg.adaptive_cold_mass_pruning, expected)
        launcher._cycle_value(opt, -1)
        self.assertEqual(cfg.adaptive_cold_mass_pruning, "75/10")

        off_display = launcher_mod._ANSI_RE.sub("", launcher_mod._format_value(opt, "off"))
        policy_display = launcher_mod._ANSI_RE.sub("", launcher_mod._format_value(opt, "75/8"))
        self.assertEqual(off_display, "Off")
        self.assertEqual(policy_display, "75/8 (approximate)")

        env = {
            "KRASIS_ADAPTIVE_COLD_DROP": "1",
            "KRASIS_ADAPTIVE_COLD_DROP_PROTECT_PCT": "50",
            "KRASIS_ADAPTIVE_COLD_DROP_MASS_PCT": "50",
            "KRASIS_ADAPTIVE_COLD_DROP_SHADOW_MASS_PCTS": "3,5,8,10",
        }
        original_env = dict(env)
        self.assertIsNone(configure_adaptive_cold_mass_pruning(None, env))
        self.assertEqual(env, original_env)
        self.assertEqual(configure_adaptive_cold_mass_pruning("75/8", env), "75/8")
        self.assertEqual(env["KRASIS_ADAPTIVE_COLD_DROP"], "1")
        self.assertEqual(env["KRASIS_ADAPTIVE_COLD_DROP_PROTECT_PCT"], "75")
        self.assertEqual(env["KRASIS_ADAPTIVE_COLD_DROP_MASS_PCT"], "8")
        self.assertIn("KRASIS_ADAPTIVE_COLD_DROP_SHADOW_MASS_PCTS", env)

        self.assertEqual(configure_adaptive_cold_mass_pruning("off", env), "off")
        self.assertNotIn("KRASIS_ADAPTIVE_COLD_DROP", env)
        self.assertNotIn("KRASIS_ADAPTIVE_COLD_DROP_PROTECT_PCT", env)
        self.assertNotIn("KRASIS_ADAPTIVE_COLD_DROP_MASS_PCT", env)
        self.assertIn("KRASIS_ADAPTIVE_COLD_DROP_SHADOW_MASS_PCTS", env)
        with self.assertRaisesRegex(ValueError, "Unsupported adaptive cold-mass pruning policy"):
            configure_adaptive_cold_mass_pruning("75/12", env)

    def test_self_update_channels_use_installer_contract(self) -> None:
        self.assertEqual(launcher_mod._self_update_bash_args("stable"), ["bash", "-s", "--"])
        self.assertEqual(
            launcher_mod._self_update_bash_args("prerelease"),
            ["bash", "-s", "--", "prerelease"],
        )

        old_fetch = launcher_mod._fetch_installer_script
        old_run = launcher_mod.subprocess.run
        calls: list[tuple[list[str], bytes, bool]] = []

        def fake_run(args, *, input=None, check=False, **_kwargs):
            calls.append((list(args), input, check))
            return subprocess.CompletedProcess(args, 0)

        try:
            launcher_mod._fetch_installer_script = lambda: b"#!/bin/bash\necho installer\n"
            launcher_mod.subprocess.run = fake_run

            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stable_exit:
                    launcher_mod._do_self_update("stable")
            self.assertEqual(stable_exit.exception.code, 0)

            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as prerelease_exit:
                    launcher_mod._do_self_update("prerelease")
            self.assertEqual(prerelease_exit.exception.code, 0)
        finally:
            launcher_mod._fetch_installer_script = old_fetch
            launcher_mod.subprocess.run = old_run

        self.assertEqual(calls[0], (["bash", "-s", "--"], b"#!/bin/bash\necho installer\n", False))
        self.assertEqual(
            calls[1],
            (["bash", "-s", "--", "prerelease"], b"#!/bin/bash\necho installer\n", False),
        )

    def test_nvidia_smi_discovery_checks_native_windows_install_path(self) -> None:
        old_os_name = nvidia_smi_mod.os.name
        old_which = nvidia_smi_mod.shutil.which
        old_program_files = os.environ.get("ProgramFiles")
        old_program_w6432 = os.environ.get("ProgramW6432")
        with tempfile.TemporaryDirectory(prefix="krasis-nvsmi-win-") as root:
            nvsmi = Path(root) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
            nvsmi.parent.mkdir(parents=True)
            nvsmi.write_text("")
            try:
                nvidia_smi_mod.os.name = "nt"
                nvidia_smi_mod.shutil.which = lambda _name: None
                os.environ["ProgramFiles"] = root
                os.environ["ProgramW6432"] = ""

                self.assertEqual(nvidia_smi_mod.find_nvidia_smi(), str(nvsmi))
                self.assertEqual(launcher_mod._find_nvidia_smi(), str(nvsmi))
            finally:
                nvidia_smi_mod.os.name = old_os_name
                nvidia_smi_mod.shutil.which = old_which
                if old_program_files is None:
                    os.environ.pop("ProgramFiles", None)
                else:
                    os.environ["ProgramFiles"] = old_program_files
                if old_program_w6432 is None:
                    os.environ.pop("ProgramW6432", None)
                else:
                    os.environ["ProgramW6432"] = old_program_w6432

    def test_windows_installer_owns_private_runtime_and_native_launcher(self) -> None:
        windows_dir = REPO_ROOT / "scripts" / "windows"
        installer_source = (windows_dir / "KrasisInstaller.iss").read_text()
        build_source = (windows_dir / "Build-Installer.ps1").read_text()
        runtime_build_source = (windows_dir / "Build-Runtime.ps1").read_text()
        runtime_manifest_source = (windows_dir / "Runtime-Manifest.ps1").read_text()
        install_source = (windows_dir / "Install-Krasis.ps1").read_text()
        invoke_install_source = (windows_dir / "Invoke-Install-Krasis.ps1").read_text()
        remove_runtime_source = (
            windows_dir / "Remove-KrasisRuntime.ps1"
        ).read_text()
        native_launcher_source = (
            REPO_ROOT / "src" / "bin" / "krasis-windows-launcher.rs"
        ).read_text()
        workflow_source = (
            REPO_ROOT / ".github" / "workflows" / "windows-installer.yml"
        ).read_text()
        fla_compiler_source = (
            REPO_ROOT / "src" / "cuda" / "fla" / "compile_kernels.py"
        ).read_text()
        sidecar_builder_source = (
            REPO_ROOT / "scripts" / "build_sidecars.py"
        ).read_text()
        fla_sidecar_contract = json.loads(
            (
                REPO_ROOT / "python" / "krasis" / "fla_sidecar_contract.json"
            ).read_text(encoding="utf-8")
        )

        self.assertNotIn(
            r'Name: "{autoprograms}\Krasis\Krasis Update"; Filename: "{app}\bin\Krasis Update.exe"',
            installer_source,
        )
        self.assertNotIn(
            r'Name: "{autoprograms}\Krasis\Krasis Prerelease"; Filename: "{app}\bin\Krasis Prerelease.exe"',
            installer_source,
        )
        self.assertIn(
            r'Name: "{autoprograms}\Krasis\Krasis"; Filename: "{app}\bin\Krasis.exe"',
            installer_source,
        )
        self.assertEqual(installer_source.count("Flags: runmaximized"), 1)
        self.assertNotIn(
            r'Filename: "{sys}\WindowsPowerShell',
            installer_source,
        )
        self.assertNotIn(
            '$LauncherExePath (Join-Path $Stage "bin\\Krasis Update.exe")',
            build_source,
        )
        self.assertNotIn(
            '$LauncherExePath (Join-Path $Stage "bin\\Krasis Prerelease.exe")',
            build_source,
        )
        self.assertNotIn(
            '"Update-Krasis.ps1") (Join-Path $Stage "bin\\Update-Krasis.ps1")',
            build_source,
        )
        self.assertFalse((windows_dir / "Update-Krasis.ps1").exists())
        for retired_path in (
            r'Type: files; Name: "{app}\bin\Krasis Update.exe"',
            r'Type: files; Name: "{app}\bin\Krasis Prerelease.exe"',
            r'Type: files; Name: "{app}\bin\Update-Krasis.ps1"',
            r'Type: files; Name: "{autoprograms}\Krasis\Krasis Update.lnk"',
            r'Type: files; Name: "{autoprograms}\Krasis\Krasis Prerelease.lnk"',
        ):
            self.assertIn(retired_path, installer_source)
        self.assertIn("[string]$LauncherExe", build_source)
        self.assertIn(
            '$LauncherExePath (Join-Path $Stage "bin\\Krasis.exe")',
            build_source,
        )
        self.assertIn(
            '"assets\\windows\\krasis.ico"',
            build_source,
        )
        self.assertIn(
            '$LauncherIconPath (Join-Path $Stage "bin\\Krasis.ico")',
            build_source,
        )
        self.assertNotIn("Launch-Krasis.ps1", build_source)
        self.assertIn(
            r'Type: files; Name: "{app}\bin\Launch-Krasis.ps1"',
            installer_source,
        )
        self.assertIn(
            '"Runtime-Manifest.ps1") (Join-Path $Stage "bin\\Runtime-Manifest.ps1")',
            build_source,
        )
        self.assertIn(
            '"Invoke-Install-Krasis.ps1") '
            '(Join-Path $Stage "bin\\Invoke-Install-Krasis.ps1")',
            build_source,
        )
        self.assertIn(
            '"Remove-KrasisRuntime.ps1") '
            '(Join-Path $Stage "bin\\Remove-KrasisRuntime.ps1")',
            build_source,
        )
        self.assertIn(
            "[System.IO.Compression.ZipFile]::CreateFromDirectory",
            build_source,
        )
        self.assertIn(
            "$env:KRASIS_RUNTIME_ARCHIVE_SHA256 = $RuntimeArchiveSha256",
            build_source,
        )
        self.assertIn(
            r'Source: "{#SourceDir}\runtime-package.zip"',
            installer_source,
        )
        self.assertIn("RuntimeArchiveSha256", installer_source)
        self.assertIn("-RuntimeArchiveSha256", installer_source)
        self.assertIn("CurStepChanged(CurStep: TSetupStep)", installer_source)
        self.assertIn("ResultCode <> 0", installer_source)
        self.assertIn("GetCustomSetupExitCode", installer_source)
        self.assertIn("RuntimeInstallExitCode := ResultCode", installer_source)
        self.assertIn("Invoke-Install-Krasis.ps1", installer_source)
        self.assertIn(
            "CurUninstallStepChanged(CurUninstallStep: TUninstallStep)",
            installer_source,
        )
        self.assertIn("CurUninstallStep <> usUninstall", installer_source)
        self.assertIn("Remove-KrasisRuntime.ps1", installer_source)
        self.assertIn("Krasis private-runtime cleanup failed", installer_source)
        self.assertNotIn(
            "createallsubdirs deleteafterinstall",
            installer_source,
        )
        self.assertNotIn(
            r'Filename: "{app}\bin\python-installer.exe"',
            installer_source,
        )
        self.assertNotIn("TargetDir=", installer_source)
        self.assertNotIn(r"-Wheelhouse ""{app}", installer_source)

        self.assertIn("Get-KrasisRuntimePayloadHash", runtime_manifest_source)
        self.assertIn("Get-KrasisFileSha256", runtime_manifest_source)
        self.assertNotIn("Get-FileHash", runtime_manifest_source)
        self.assertIn("[System.IO.File]::OpenRead", runtime_manifest_source)
        self.assertIn("[System.Security.Cryptography.SHA256]::Create()", runtime_manifest_source)
        self.assertIn("Assert-KrasisPrivateRuntime", runtime_manifest_source)
        self.assertEqual(
            fla_sidecar_contract,
            {
                "schema_version": 1,
                "architectures": [80, 89, 90, 120],
                "h_values": [32, 64],
            },
        )
        self.assertIn('f"krasis_fla_sm{arch}.dll"', runtime_manifest_source)
        self.assertIn('"fla_architectures": fla_architectures', runtime_manifest_source)
        self.assertIn(
            "[IO.File]::WriteAllText($ProbePath, $ProbeCode, $Utf8NoBom)",
            runtime_manifest_source,
        )
        self.assertIn(
            "New-Object System.Text.UTF8Encoding($false)",
            runtime_manifest_source,
        )
        self.assertIn(
            "$StartInfo.Arguments = '-I -B \"' + $ProbePath + '\"'",
            runtime_manifest_source,
        )
        self.assertNotIn("RedirectStandardInput", runtime_manifest_source)
        self.assertNotIn("$Process.StandardInput", runtime_manifest_source)
        self.assertIn(
            "$Process.StandardOutput.ReadToEndAsync()",
            runtime_manifest_source,
        )
        self.assertIn(
            "$Process.StandardError.ReadToEndAsync()",
            runtime_manifest_source,
        )
        self.assertNotIn(
            "& $Python -I -B -c $ProbeCode",
            runtime_manifest_source,
        )
        self.assertIn('"isolated": sys.flags.isolated', runtime_manifest_source)
        self.assertIn('"ignore_environment": sys.flags.ignore_environment', runtime_manifest_source)
        self.assertIn("user site-packages", runtime_manifest_source)
        self.assertIn("Get-KrasisRuntimePayloadHash", runtime_build_source)
        self.assertIn("$RelocationProbe", runtime_build_source)
        self.assertIn("runtime-manifest.json", runtime_build_source)
        self.assertIn("Expand-Archive -Path $PythonRuntimeArchivePath", runtime_build_source)
        self.assertIn("--target $PrivateSitePackages", runtime_build_source)
        self.assertIn('--only-binary ":all:"', runtime_build_source)
        self.assertIn("& $BuildPythonPath -m pip install", runtime_build_source)
        self.assertNotIn("InstallerArgs", runtime_build_source)
        self.assertNotIn("& $PythonInstallerPath", runtime_build_source)

        self.assertIn('$RuntimeRoot = Join-Path $InstallRoot "runtime"', install_source)
        self.assertIn('$CurrentPath = Join-Path $RuntimeRoot "current.txt"', install_source)
        self.assertIn("[System.IO.File]::Replace", install_source)
        self.assertIn("Get-KrasisFileSha256 -Path $RuntimeArchivePath", install_source)
        self.assertIn(
            "[System.IO.Compression.ZipFile]::ExtractToDirectory",
            install_source,
        )
        self.assertNotIn("Copy-Item -Recurse -Force", install_source)
        self.assertIn("--no-deps", install_source)
        self.assertIn('"$($StagedManifest.torch_url)"', install_source)
        self.assertIn("(Join-Path $InstallRoot \"python\")", install_source)
        self.assertIn("(Join-Path $InstallRoot \"venv\")", install_source)
        self.assertNotIn("Get-Command py", install_source)
        self.assertNotIn("Get-Command python", install_source)
        self.assertNotIn("-m venv", install_source)

        self.assertIn("Start-Transcript -Path $LogPath -Force", invoke_install_source)
        self.assertIn("& $InstallScript", invoke_install_source)
        self.assertIn(
            "-RuntimeArchiveSha256 $RuntimeArchiveSha256",
            invoke_install_source,
        )
        self.assertIn("$ExitCode = 1", invoke_install_source)
        self.assertIn("exit $ExitCode", invoke_install_source)

        self.assertIn("KrasisLongPathDelete", remove_runtime_source)
        self.assertIn("FindFirstFileW", remove_runtime_source)
        self.assertIn("GetFileAttributesW", remove_runtime_source)
        self.assertIn("DeleteFileW", remove_runtime_source)
        self.assertIn("RemoveDirectoryW", remove_runtime_source)
        self.assertIn("FILE_ATTRIBUTE_REPARSE_POINT", remove_runtime_source)
        self.assertIn("Test-KrasisRuntimeInUse", remove_runtime_source)
        self.assertIn('@("runtime", "python", "venv")', remove_runtime_source)
        self.assertIn(
            "refusing to remove the Krasis runtime",
            remove_runtime_source,
        )
        self.assertIn("empty-cleanup-probe", workflow_source)
        self.assertIn("reparse-cleanup-probe", workflow_source)
        self.assertIn("must-survive.txt", workflow_source)
        self.assertIn("Compile installer script syntax", workflow_source)
        self.assertIn("krasis-inno-syntax", workflow_source)
        self.assertIn(
            "Uninstall traversed a runtime reparse point outside the install root.",
            workflow_source,
        )

        self.assertIn('const ACTIVATION_FILE: &str = "runtime/current.txt"', native_launcher_source)
        self.assertIn('join("runtime")', native_launcher_source)
        self.assertIn('join("releases")', native_launcher_source)
        self.assertIn('join("python.exe")', native_launcher_source)
        self.assertIn('.args(["-I", "-m", "krasis.launcher"])', native_launcher_source)
        self.assertIn('.env_remove("PYTHONHOME")', native_launcher_source)
        self.assertIn('.env_remove("PYTHONPATH")', native_launcher_source)
        self.assertNotIn("UpdateChannel", native_launcher_source)
        self.assertNotIn("system_powershell", native_launcher_source)
        self.assertNotIn("GetSystemDirectoryW", native_launcher_source)
        self.assertIn("Command::new(&paths.python)", native_launcher_source)
        self.assertNotIn("Update-Krasis.ps1", native_launcher_source)
        self.assertNotIn("Krasis Update.exe", native_launcher_source)
        self.assertNotIn("Krasis Prerelease.exe", native_launcher_source)
        self.assertNotIn("SetConsoleMode", native_launcher_source)
        self.assertNotIn("GetConsoleWindow", native_launcher_source)
        self.assertNotIn(r"venv\Scripts\python.exe", native_launcher_source)
        self.assertIn("Build native Windows launcher", workflow_source)
        self.assertIn("cargo test --release --bin krasis-windows-launcher", workflow_source)
        self.assertIn("-LauncherExe target/release/krasis-windows-launcher.exe", workflow_source)
        self.assertIn('$nativeLauncher = Join-Path $testRoot "bin\\Krasis.exe"', workflow_source)
        self.assertIn("& $nativeLauncher --probe", workflow_source)
        self.assertIn(
            r"SetupIconFile={#SourceDir}\bin\Krasis.ico",
            installer_source,
        )
        self.assertIn(
            "cargo:rustc-link-arg-bin=krasis-windows-launcher=",
            (REPO_ROOT / "build.rs").read_text(),
        )

        self.assertIn('KRASIS_WINDOWS_PYTHON_VERSION: "3.12.10"', workflow_source)
        self.assertIn(
            'KRASIS_WINDOWS_PYTHON_ARCHIVE_SHA256: '
            '"4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3"',
            workflow_source,
        )
        self.assertIn("python-$pyver-embed-amd64.zip", workflow_source)
        self.assertIn("-PythonRuntimeArchive python-runtime.zip", workflow_source)
        self.assertNotIn("-PythonInstaller python-installer.exe", workflow_source)
        self.assertIn('KRASIS_WINDOWS_TORCH_VERSION: "2.9.1+cu128"', workflow_source)
        self.assertIn("Test clean install, isolation, legacy repair, and uninstall", workflow_source)
        self.assertIn(
            "Compare payload digest under Windows PowerShell 5.1",
            workflow_source,
        )
        self.assertIn("Test-InstalledRuntime.ps1", workflow_source)
        self.assertIn('Get-Content $installLog', workflow_source)
        self.assertIn("Requested install-root contents:", workflow_source)
        self.assertIn("Krasis private-runtime install transcript:", workflow_source)
        self.assertIn("Krasis private-runtime uninstall log:", workflow_source)
        self.assertIn("Retained private-runtime entries:", workflow_source)
        self.assertIn("Processes executing from the retained runtime:", workflow_source)
        self.assertIn("PendingFileRenameOperations", workflow_source)
        self.assertIn("generate-windows-fla-sources:", workflow_source)
        self.assertIn("needs: generate-windows-fla-sources", workflow_source)
        self.assertGreaterEqual(
            workflow_source.count("if: github.event_name == 'workflow_dispatch'"),
            2,
        )
        self.assertIn("promote-tested-windows-release:", workflow_source)
        self.assertIn("if: github.event_name == 'release'", workflow_source)
        self.assertIn("--target-platform windows", workflow_source)
        self.assertIn("KRASIS_FLA_REQUIRE_ALL_ARCHS", workflow_source)
        self.assertIn("windows-fla-manifest.json", workflow_source)
        self.assertIn("'triton==3.5.1'", workflow_source)
        self.assertIn("'torch==2.13.0+cpu'", workflow_source)
        self.assertIn("Restore portable Windows FLA source cache", workflow_source)
        self.assertIn("Restore Windows FLA DLL cache", workflow_source)
        self.assertIn(
            r"path: ${{ runner.temp }}\krasis-windows-fla-dll-cache",
            workflow_source,
        )
        self.assertIn(
            '$flaDllCacheRoot = Join-Path $env:RUNNER_TEMP '
            '"krasis-windows-fla-dll-cache"',
            workflow_source,
        )
        self.assertNotIn("path: target/windows-fla-dll-cache", workflow_source)
        self.assertIn(
            "Expected Windows FLA DLL cache files:",
            workflow_source,
        )
        self.assertIn(
            "Actual Windows FLA DLL cache files:",
            workflow_source,
        )
        self.assertIn(
            "${{ runner.temp }}/krasis-windows-fla-dll-cache/**",
            workflow_source,
        )
        self.assertIn("krasis-windows-fla-dlls-v2-", workflow_source)
        self.assertNotIn("krasis-windows-fla-dlls-v1-", workflow_source)
        self.assertIn("Unexpected Windows FLA linker byproducts:", workflow_source)
        self.assertIn(
            "Built Windows FLA DLL cache file inventory is invalid:",
            workflow_source,
        )
        self.assertGreaterEqual(
            workflow_source.count(
                'Join-Path $env:RUNNER_TEMP "krasis-windows-sidecar-release-cache"'
            ),
            2,
        )
        self.assertNotIn(
            'Join-Path $PWD "target\\windows-release-cache"',
            workflow_source,
        )
        self.assertNotIn("target/windows-release-cache", workflow_source)
        self.assertIn("cache_key_sha256", workflow_source)
        self.assertIn("Get-FileHash -Algorithm SHA256", workflow_source)
        self.assertIn("dumpbin /nologo /exports", workflow_source)
        self.assertIn(
            "python scripts/build_sidecars.py restore-bundle --github",
            workflow_source,
        )
        self.assertNotIn(
            "python scripts/build_sidecars.py build --force",
            workflow_source,
        )
        self.assertIn(
            "krasis-windows-release-${{ github.sha }}",
            workflow_source,
        )
        self.assertIn("windows-release-provenance.json", workflow_source)
        self.assertIn('"workflow_dispatch"', workflow_source)
        self.assertIn('installer_lifecycle = "passed"', workflow_source)
        self.assertIn(
            'test "$(jq -r \'.conclusion\' <<<"$run_json")" = "success"',
            workflow_source,
        )
        self.assertIn(
            'test "$(git rev-list -n1 "$RELEASE_TAG")" = "$GITHUB_SHA"',
            workflow_source,
        )
        self.assertNotIn(
            'if: github.event_name == \'release\'\n        env:\n'
            '          GH_TOKEN: ${{ github.token }}\n        run: |\n'
            '          gh release upload "${{ github.event.release.tag_name }}" '
            'dist/KrasisSetup-*-win64.exe',
            workflow_source,
        )
        self.assertIn("FLA_ARCHS = read_fla_architectures()", sidecar_builder_source)
        self.assertIn('f"krasis_fla_sm{arch}.dll"', sidecar_builder_source)
        self.assertIn("fla_sidecar_contract.json", sidecar_builder_source)
        self.assertIn("portable_embedded_cubins", fla_compiler_source)
        self.assertIn('__declspec(dllexport)', fla_compiler_source)
        self.assertIn('"target_platform": "windows"', fla_compiler_source)
        self.assertIn("if: always()", workflow_source)
        self.assertIn("${{ runner.temp }}/krasis-installer-test.log", workflow_source)
        self.assertIn("${{ runner.temp }}/krasis-uninstaller-test.log", workflow_source)
    def test_hf_results_screen_fits_short_terminal_without_wrapping(self) -> None:
        long_summary = " ".join(["very-long-summary"] * 20)
        candidates = [
            argparse.Namespace(
                repo_id=f"Example/Model-{idx}-with-a-very-long-repository-name",
                summary=long_summary,
                gated=False,
                private=False,
                has_safetensors=True,
                int4_payload_bytes=12_345_678_901,
                selected_bytes=98_765_432_109,
                safetensors_total_bytes=98_765_432_109,
                pipeline_tag="text-generation",
                downloads=123_456,
                likes=789,
                last_modified="2026-05-16",
            )
            for idx in range(8)
        ]
        launcher = Launcher.__new__(Launcher)
        old_clear = launcher_mod._clear_screen
        old_read_key = launcher_mod._read_key
        old_get_terminal_size = launcher_mod.shutil.get_terminal_size
        try:
            launcher_mod._clear_screen = lambda: None
            launcher_mod._read_key = lambda: launcher_mod.KEY_ESCAPE
            launcher_mod.shutil.get_terminal_size = lambda fallback: os.terminal_size((60, 14))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                selected = launcher._hf_results_screen(candidates)
        finally:
            launcher_mod._clear_screen = old_clear
            launcher_mod._read_key = old_read_key
            launcher_mod.shutil.get_terminal_size = old_get_terminal_size

        self.assertIsNone(selected)
        rendered = out.getvalue()
        lines = rendered.splitlines()
        self.assertLessEqual(len(lines), 14)
        self.assertEqual(lines[0], "")
        self.assertIn("Hugging Face results", launcher_mod._ANSI_RE.sub("", lines[1]))
        self.assertIn("Example/Model-0", launcher_mod._ANSI_RE.sub("", rendered))
        for line in lines:
            self.assertLessEqual(launcher_mod._visible_len(line), 60)

    def test_hqq4_interactive_preset_quantizer_symbol_available(self) -> None:
        weight = torch.linspace(-1.0, 1.0, steps=32, dtype=torch.float32).reshape(4, 8)
        tensors = quantize_hqq4_tensor_rust(weight, group_size=8, inner_threads=1)
        self.assertEqual(tuple(tensors["packed"].shape), (4, 4))
        self.assertEqual(tuple(tensors["scales"].shape), (4, 1))
        self.assertEqual(tuple(tensors["zeros"].shape), (4, 1))

    def test_save_config_round_trip_keeps_advanced_fields(self) -> None:
        cfg = _base_config()
        cfg.selected_gpu_indices = [0, 1]
        cfg.pp_partition = "20,20"
        cfg.layer_group_size = 6
        cfg.kv_cache_mb = 1800
        cfg.max_context_tokens = 32768
        cfg.kv_dtype = "k4v4"
        cfg.gpu_expert_bits = 8
        cfg.expert_group_size = 64
        cfg.gpu_expert_int4_calib = "search_rmse"
        cfg.cpu_expert_bits = 8
        cfg.attention_quant = "hqq68_auto"
        cfg.hqq_cache_profile = "selfcal_v1"
        cfg.hqq_group_size = 64
        cfg.hqq_auto_budget_pct = 25.0
        cfg.hqq46_auto_budget_mib = 384
        cfg.hqq_sidecar_manifest = ""
        cfg.shared_expert_quant = "bf16"
        cfg.dense_mlp_quant = "bf16"
        cfg.lm_head_quant = "bf16"
        cfg.krasis_threads = 12
        cfg.host = "127.0.0.1"
        cfg.port = 18012
        cfg.ssh_tunnel = "alice@example.com:2222"
        cfg.ssh_key_path = "~/.ssh/id_ed25519"
        cfg.gpu_prefill_threshold = 512
        cfg.gguf_path = "~/models/cpu-experts.gguf"
        cfg.heatmap_path = "~/heatmaps/qcn.json"
        cfg.vram_safety_margin = 900
        cfg.hcs = False
        cfg.multi_gpu_hcs = True
        cfg.multi_gpu_mode = "peer"
        cfg.dynamic_peer = True
        cfg.dynamic_hcs = False
        cfg.dynamic_hcs_tail_blocks = 5
        cfg.adaptive_cold_mass_pruning = "75/8"
        cfg.expert_compression = True
        cfg.expert_compression_sidecar = "~/cache/model.krec"
        cfg.expert_compression_pipeline = "streaming"
        cfg.stream_attention = True
        cfg.draft_model = "~/models/draft"
        cfg.draft_k = 5
        cfg.draft_context = 1024
        cfg.temperature = 0.25
        cfg.force_load = True
        cfg.force_rebuild_cache = True
        cfg.force_rebuild_hqq_cache = True
        cfg.build_cache = True
        cfg.enable_thinking = False
        cfg.prefix_cache = True
        cfg.prefix_cache_ram_fraction = 0.375
        with tempfile.NamedTemporaryFile("w", suffix=".conf", prefix="krasis-save-roundtrip-", delete=False) as f:
            path = Path(f.name)
        try:
            launcher_mod._save_config(str(path), cfg.to_save_dict())
            values = _parse_key_value_config(path)
            self.assertEqual(set(values), set(launcher_mod.CONFIG_KEYS))
            self.assertEqual(values.get("MODEL_PATH"), NONEXISTENT_MODEL)
            self.assertEqual(values.get("CFG_SELECTED_GPUS"), "0,1")
            self.assertEqual(values.get("CFG_PP_PARTITION"), "20,20")
            self.assertEqual(values.get("CFG_LAYER_GROUP_SIZE"), "6")
            self.assertEqual(values.get("CFG_KV_CACHE_MB"), "1800")
            self.assertEqual(values.get("CFG_MAX_CONTEXT_TOKENS"), "32768")
            self.assertEqual(values.get("CFG_KV_DTYPE"), "k4v4")
            self.assertEqual(values.get("CFG_GPU_EXPERT_BITS"), "8")
            self.assertEqual(values.get("CFG_EXPERT_GROUP_SIZE"), "64")
            self.assertEqual(values.get("CFG_GPU_EXPERT_INT4_CALIB"), "search_rmse")
            self.assertEqual(values.get("CFG_CPU_EXPERT_BITS"), "8")
            self.assertEqual(values.get("CFG_ATTENTION_QUANT"), "hqq68_auto")
            self.assertEqual(values.get("CFG_HQQ_CACHE_PROFILE"), "selfcal_v1")
            self.assertEqual(values.get("CFG_HQQ_GROUP_SIZE"), "64")
            self.assertEqual(values.get("CFG_HQQ_AUTO_BUDGET_PCT"), "25.0")
            self.assertEqual(values.get("CFG_HQQ46_AUTO_BUDGET_MB"), "")
            self.assertEqual(values.get("CFG_HQQ_SIDECAR_MANIFEST"), "")
            self.assertEqual(values.get("CFG_SHARED_EXPERT_QUANT"), "bf16")
            self.assertEqual(values.get("CFG_DENSE_MLP_QUANT"), "bf16")
            self.assertEqual(values.get("CFG_LM_HEAD_QUANT"), "bf16")
            self.assertEqual(values.get("CFG_KRASIS_THREADS"), "12")
            self.assertEqual(values.get("CFG_HOST"), "127.0.0.1")
            self.assertEqual(values.get("CFG_PORT"), "18012")
            self.assertEqual(values.get("CFG_SSH_TUNNEL"), "alice@example.com:2222")
            self.assertEqual(values.get("CFG_SSH_KEY_PATH"), "~/.ssh/id_ed25519")
            self.assertEqual(values.get("CFG_GPU_PREFILL_THRESHOLD"), "512")
            self.assertEqual(values.get("CFG_GGUF_PATH"), "~/models/cpu-experts.gguf")
            self.assertEqual(values.get("CFG_HEATMAP_PATH"), "~/heatmaps/qcn.json")
            self.assertEqual(values.get("CFG_VRAM_SAFETY_MARGIN"), "900")
            self.assertEqual(values.get("CFG_HCS"), "0")
            self.assertEqual(values.get("CFG_MULTI_GPU_HCS"), "1")
            self.assertEqual(values.get("CFG_MULTI_GPU_MODE"), "peer")
            self.assertEqual(values.get("CFG_DYNAMIC_PEER"), "1")
            self.assertEqual(values.get("CFG_DYNAMIC_HCS"), "0")
            self.assertEqual(values.get("CFG_DYNAMIC_HCS_TAIL_BLOCKS"), "5")
            self.assertEqual(values.get("CFG_ADAPTIVE_COLD_MASS_PRUNING"), "75/8")
            self.assertEqual(values.get("CFG_EXPERT_COMPRESSION"), "1")
            self.assertEqual(
                values.get("CFG_EXPERT_COMPRESSION_SIDECAR"), "~/cache/model.krec"
            )
            self.assertEqual(
                values.get("CFG_EXPERT_COMPRESSION_PIPELINE"), "streaming"
            )
            self.assertEqual(values.get("CFG_STREAM_ATTENTION"), "1")
            self.assertEqual(values.get("CFG_DRAFT_MODEL"), "~/models/draft")
            self.assertEqual(values.get("CFG_DRAFT_K"), "5")
            self.assertEqual(values.get("CFG_DRAFT_CONTEXT"), "1024")
            self.assertEqual(values.get("CFG_TEMPERATURE"), "0.25")
            self.assertEqual(values.get("CFG_FORCE_LOAD"), "1")
            self.assertEqual(values.get("CFG_FORCE_REBUILD_CACHE"), "1")
            self.assertEqual(values.get("CFG_FORCE_REBUILD_HQQ_CACHE"), "1")
            self.assertEqual(values.get("CFG_BUILD_CACHE"), "1")
            self.assertEqual(values.get("CFG_ENABLE_THINKING"), "0")
            self.assertEqual(values.get("CFG_PREFIX_CACHE"), "1")
            self.assertEqual(values.get("CFG_PREFIX_CACHE_RAM_FRACTION"), "0.375")

            loaded = LauncherConfig()
            loaded.apply_saved(launcher_mod._load_config(str(path)))
            self.assertEqual(loaded.model_path, NONEXISTENT_MODEL)
            self.assertEqual(loaded.selected_gpu_indices, [0, 1])
            self.assertEqual(loaded.pp_partition, "20,20")
            self.assertEqual(loaded.layer_group_size, 6)
            self.assertEqual(loaded.kv_cache_mb, 1800)
            self.assertEqual(loaded.max_context_tokens, 32768)
            self.assertEqual(loaded.kv_dtype, "k4v4")
            self.assertEqual(loaded.gpu_expert_bits, 8)
            self.assertEqual(loaded.expert_group_size, 64)
            self.assertEqual(loaded.gpu_expert_int4_calib, "search_rmse")
            self.assertEqual(loaded.cpu_expert_bits, 8)
            self.assertEqual(loaded.attention_quant, "hqq68_auto")
            self.assertEqual(loaded.hqq_cache_profile, "selfcal_v1")
            self.assertEqual(loaded.hqq_group_size, 64)
            self.assertEqual(loaded.hqq_auto_budget_pct, 25.0)
            self.assertEqual(loaded.shared_expert_quant, "bf16")
            self.assertEqual(loaded.dense_mlp_quant, "bf16")
            self.assertEqual(loaded.lm_head_quant, "bf16")
            self.assertEqual(loaded.krasis_threads, 12)
            self.assertEqual(loaded.host, "127.0.0.1")
            self.assertEqual(loaded.port, 18012)
            self.assertEqual(loaded.ssh_tunnel, "alice@example.com:2222")
            self.assertEqual(loaded.ssh_key_path, os.path.expanduser("~/.ssh/id_ed25519"))
            self.assertEqual(loaded.gpu_prefill_threshold, 512)
            self.assertEqual(loaded.gguf_path, "~/models/cpu-experts.gguf")
            self.assertEqual(loaded.heatmap_path, os.path.expanduser("~/heatmaps/qcn.json"))
            self.assertEqual(loaded.vram_safety_margin, 900)
            self.assertFalse(loaded.hcs)
            self.assertTrue(loaded.multi_gpu_hcs)
            self.assertEqual(loaded.multi_gpu_mode, "peer")
            self.assertTrue(loaded.dynamic_peer)
            self.assertFalse(loaded.dynamic_hcs)
            self.assertEqual(loaded.dynamic_hcs_tail_blocks, 5)
            self.assertEqual(loaded.adaptive_cold_mass_pruning, "75/8")
            self.assertTrue(loaded.expert_compression)
            self.assertEqual(
                loaded.expert_compression_sidecar,
                os.path.expanduser("~/cache/model.krec"),
            )
            self.assertEqual(loaded.expert_compression_pipeline, "streaming")
            self.assertTrue(loaded.stream_attention)
            self.assertEqual(loaded.draft_model, os.path.expanduser("~/models/draft"))
            self.assertEqual(loaded.draft_k, 5)
            self.assertEqual(loaded.draft_context, 1024)
            self.assertEqual(loaded.temperature, 0.25)
            self.assertTrue(loaded.force_load)
            self.assertTrue(loaded.force_rebuild_cache)
            self.assertTrue(loaded.force_rebuild_hqq_cache)
            self.assertTrue(loaded.build_cache)
            self.assertFalse(loaded.enable_thinking)
            self.assertTrue(loaded.prefix_cache)
            self.assertEqual(loaded.prefix_cache_ram_fraction, 0.375)
        finally:
            path.unlink(missing_ok=True)

    def test_saved_context_cap_rejects_invalid_values(self) -> None:
        for value, message in (
            ("not-an-integer", "must be an integer"),
            ("-1", "must be non-negative"),
        ):
            with self.subTest(value=value):
                cfg = LauncherConfig()
                with self.assertRaisesRegex(ValueError, message):
                    cfg.apply_saved({"CFG_MAX_CONTEXT_TOKENS": value})

    def test_saved_prefix_cache_fraction_rejects_invalid_values(self) -> None:
        for value in ("not-a-number", "nan", "inf", "0", "-0.1", "1.1"):
            with self.subTest(value=value):
                cfg = LauncherConfig()
                with self.assertRaisesRegex(ValueError, "CFG_PREFIX_CACHE_RAM_FRACTION"):
                    cfg.apply_saved({"CFG_PREFIX_CACHE_RAM_FRACTION": value})

    def test_conversation_cache_defaults_on_and_saved_zero_disables_it(self) -> None:
        cfg = LauncherConfig()
        self.assertTrue(cfg.prefix_cache)
        cfg.apply_saved({"CFG_PREFIX_CACHE": "0"})
        self.assertFalse(cfg.prefix_cache)

    def test_no_prefix_cache_cli_explicitly_disables_default(self) -> None:
        with mock.patch.object(sys, "argv", ["krasis", "--no-prefix-cache"]):
            args = launcher_mod.parse_args()
        cfg = LauncherConfig()
        launcher_mod._apply_cli_overrides(cfg, args)
        self.assertFalse(cfg.prefix_cache)

    def test_interactive_load_config_preserves_saved_kv_attention_safety_and_ssh(self) -> None:
        launcher = Launcher.__new__(Launcher)
        launcher.cfg = LauncherConfig()
        launcher.hw = {
            "gpu_count": 1,
            "gpus": [{"index": 0, "name": "Test GPU 0", "vram_mb": 24_000}],
        }
        launcher.selected_gpus = []
        launcher.model_info = None
        launcher.budget = None
        launcher.budget_error = None
        launcher._compute_budget = lambda: None
        launcher._read_model_info = lambda: None

        old_clear = launcher_mod._clear_screen
        old_read_key = launcher_mod._read_key
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="krasis-load-screen-") as tmp:
            path = Path(tmp) / "saved.conf"
            path.write_text(
                "\n".join([
                    'CFG_SELECTED_GPUS="0"',
                    'CFG_KV_DTYPE="k4v4"',
                    'CFG_ATTENTION_QUANT="hqq4"',
                    'CFG_VRAM_SAFETY_MARGIN="900"',
                    'CFG_ADAPTIVE_COLD_MASS_PRUNING="75/5"',
                    'CFG_SSH_TUNNEL="alice@example.com:2222"',
                    'CFG_SSH_KEY_PATH="~/.ssh/id_ed25519"',
                    'CFG_ENABLE_THINKING="0"',
                    "",
                ])
            )
            os.chdir(tmp)
            try:
                launcher_mod._clear_screen = lambda: None
                launcher_mod._read_key = lambda: launcher_mod.KEY_ENTER
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(launcher._load_config_screen())
            finally:
                launcher_mod._clear_screen = old_clear
                launcher_mod._read_key = old_read_key
                os.chdir(old_cwd)

        self.assertEqual(launcher.cfg.kv_dtype, "k4v4")
        self.assertEqual(launcher.cfg.attention_quant, "hqq4")
        self.assertEqual(launcher.cfg.vram_safety_margin, 900)
        self.assertEqual(launcher.cfg.adaptive_cold_mass_pruning, "75/5")
        self.assertEqual(launcher.cfg.ssh_tunnel, "alice@example.com:2222")
        self.assertEqual(launcher.cfg.ssh_key_path, os.path.expanduser("~/.ssh/id_ed25519"))
        self.assertFalse(launcher.cfg.enable_thinking)

    def test_stable_gpu_selectors_round_trip_and_resolve(self) -> None:
        cfg = LauncherConfig()
        cfg.apply_saved({
            "CFG_SELECTED_GPUS": "GPU-test-big,00000000:81:00.0",
        })
        self.assertEqual(cfg.selected_gpu_specs, ["GPU-test-big", "00000000:81:00.0"])
        self.assertEqual(cfg.selected_gpu_indices, [])
        self.assertEqual(cfg.to_save_dict()["CFG_SELECTED_GPUS"], "GPU-test-big,00000000:81:00.0")

        launcher = Launcher.__new__(Launcher)
        launcher.cfg = cfg
        launcher.hw = {
            "gpu_count": 2,
            "gpus": [
                {
                    "index": 0,
                    "name": "Small GPU",
                    "vram_mb": 24_000,
                    "uuid": "GPU-test-small",
                    "pci_bus_id": "00000000:81:00.0",
                },
                {
                    "index": 1,
                    "name": "Big GPU",
                    "vram_mb": 96_000,
                    "uuid": "GPU-test-big",
                    "pci_bus_id": "00000000:C5:00.0",
                },
            ],
        }
        launcher.selected_gpus = []
        launcher._resolve_selected_gpus()

        self.assertEqual([g["index"] for g in launcher.selected_gpus], [1, 0])
        self.assertEqual(cfg.selected_gpu_indices, [1, 0])
        self.assertEqual(cfg.to_save_dict()["CFG_SELECTED_GPUS"], "GPU-test-big,00000000:81:00.0")

    def test_gpu_alias_selectors_resolve_uniquely(self) -> None:
        cfg = LauncherConfig()
        cfg.apply_saved({
            "CFG_SELECTED_GPUS": "6000,20GB",
        })

        launcher = Launcher.__new__(Launcher)
        launcher.cfg = cfg
        launcher.hw = {
            "gpu_count": 2,
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA RTX A4500",
                    "vram_mb": 20_470,
                    "uuid": "GPU-test-a4500",
                    "pci_bus_id": "00000000:81:00.0",
                },
                {
                    "index": 1,
                    "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                    "vram_mb": 97_887,
                    "uuid": "GPU-test-rtx-pro-6000",
                    "pci_bus_id": "00000000:C5:00.0",
                },
            ],
        }
        launcher.selected_gpus = []
        launcher._resolve_selected_gpus()

        self.assertEqual([g["index"] for g in launcher.selected_gpus], [1, 0])
        self.assertEqual(cfg.selected_gpu_indices, [1, 0])
        self.assertEqual(cfg.to_save_dict()["CFG_SELECTED_GPUS"], "6000,20GB")

        cfg = LauncherConfig()
        cfg.apply_saved({
            "CFG_SELECTED_GPUS": "96GB",
        })
        launcher.cfg = cfg
        launcher.selected_gpus = []
        launcher._resolve_selected_gpus()

        self.assertEqual([g["index"] for g in launcher.selected_gpus], [1])
        self.assertEqual(cfg.selected_gpu_indices, [1])
        self.assertEqual(cfg.to_save_dict()["CFG_SELECTED_GPUS"], "96GB")

    def test_gpu_alias_selector_ambiguity_is_not_silent(self) -> None:
        cfg = LauncherConfig()
        cfg.apply_saved({
            "CFG_SELECTED_GPUS": "6000",
        })

        launcher = Launcher.__new__(Launcher)
        launcher.cfg = cfg
        launcher.hw = {
            "gpu_count": 2,
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA RTX A6000",
                    "vram_mb": 49_140,
                    "uuid": "GPU-test-a6000",
                    "pci_bus_id": "00000000:81:00.0",
                },
                {
                    "index": 1,
                    "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                    "vram_mb": 97_887,
                    "uuid": "GPU-test-rtx-pro-6000",
                    "pci_bus_id": "00000000:C5:00.0",
                },
            ],
        }
        launcher.selected_gpus = []

        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            launcher._resolve_selected_gpus()

        self.assertEqual(launcher.selected_gpus, [])
        self.assertEqual(cfg.selected_gpu_indices, [])
        self.assertIn("ambiguous '6000'", stderr.getvalue())

    def test_gpu_alias_duplicate_resolution_is_not_silent(self) -> None:
        cfg = LauncherConfig()
        cfg.apply_saved({
            "CFG_SELECTED_GPUS": "6000,96GB",
        })

        launcher = Launcher.__new__(Launcher)
        launcher.cfg = cfg
        launcher.hw = {
            "gpu_count": 2,
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA RTX A4500",
                    "vram_mb": 20_470,
                    "uuid": "GPU-test-a4500",
                    "pci_bus_id": "00000000:81:00.0",
                },
                {
                    "index": 1,
                    "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                    "vram_mb": 97_887,
                    "uuid": "GPU-test-rtx-pro-6000",
                    "pci_bus_id": "00000000:C5:00.0",
                },
            ],
        }
        launcher.selected_gpus = []

        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            launcher._resolve_selected_gpus()

        self.assertEqual(launcher.selected_gpus, [])
        self.assertEqual(cfg.selected_gpu_indices, [])
        self.assertIn("duplicate resolved GPU: 96GB", stderr.getvalue())

    def test_gpu_alias_launch_uses_resolved_uuid(self) -> None:
        cfg = _base_config()
        cfg.selected_gpu_specs = ["6000"]
        cfg.selected_gpu_indices = [1]
        cfg.attention_quant = "hqq6"

        launcher = Launcher.__new__(Launcher)
        launcher.cfg = cfg
        launcher.hw = {
            "gpu_count": 2,
            "gpus": [
                {"index": 0, "name": "NVIDIA RTX A4500", "vram_mb": 20_470},
                {"index": 1, "name": "NVIDIA RTX PRO 6000 Blackwell", "vram_mb": 97_887},
            ],
        }
        launcher.selected_gpus = [
            {
                "index": 1,
                "name": "NVIDIA RTX PRO 6000 Blackwell",
                "vram_mb": 97_887,
                "uuid": "GPU-test-rtx-pro-6000",
                "pci_bus_id": "00000000:C5:00.0",
            }
        ]

        old_execvp = os.execvp
        old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")

        def fake_execvp(path: str, argv: list[str]) -> None:
            raise _ExecIntercept(path, list(argv))

        os.execvp = fake_execvp
        try:
            with self.assertRaises(_ExecIntercept) as raised:
                with contextlib.redirect_stdout(io.StringIO()):
                    launcher.launch_server(benchmark=True)
            config_path = Path(raised.exception.args[raised.exception.args.index("--config") + 1])
            try:
                values = _parse_key_value_config(config_path)
                self.assertEqual(values.get("CFG_SELECTED_GPUS"), "6000")
                self.assertEqual(values.get("CFG_NUM_GPUS"), "1")
                self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "GPU-test-rtx-pro-6000")
            finally:
                config_path.unlink(missing_ok=True)
        finally:
            os.execvp = old_execvp
            if old_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cvd

    def test_server_numeric_gpu_selector_resolves_to_uuid(self) -> None:
        from krasis import server as server_mod

        old_inventory = server_mod._nvidia_smi_gpu_inventory
        try:
            server_mod._nvidia_smi_gpu_inventory = lambda source: [
                {
                    "index": 0,
                    "name": "NVIDIA RTX A4500",
                    "vram_mb": 20_470,
                    "uuid": "GPU-test-a4500",
                    "pci_bus_id": "00000000:81:00.0",
                },
                {
                    "index": 1,
                    "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                    "vram_mb": 97_887,
                    "uuid": "GPU-test-rtx-pro-6000",
                    "pci_bus_id": "00000000:C5:00.0",
                },
            ]

            self.assertEqual(
                server_mod._normalize_selected_gpus("1", "test"),
                "GPU-test-rtx-pro-6000",
            )
            self.assertEqual(
                server_mod._normalize_selected_gpus("0,1", "test"),
                "GPU-test-a4500,GPU-test-rtx-pro-6000",
            )
        finally:
            server_mod._nvidia_smi_gpu_inventory = old_inventory

    def test_launcher_generated_configs_start_server_parse_path(self) -> None:
        scenarios = []

        cfg = _base_config()
        cfg.attention_quant = "hqq6"
        cfg.hqq_auto_budget_pct = 50.0
        cfg.hqq46_auto_budget_mib = 256
        cfg.hqq_sidecar_manifest = ""
        scenarios.append((
            "plain_hqq6_stale_budget",
            cfg,
            {
                "CFG_ATTENTION_QUANT": "hqq6",
                "CFG_KV_DTYPE": "k6v6",
                "CFG_GPU_EXPERT_BITS": "4",
                "CFG_CPU_EXPERT_BITS": "4",
                "CFG_SELECTED_GPUS": "0",
                "CFG_NUM_GPUS": "1",
            },
            ["attention_quant = 'hqq6'", "hqq_auto_budget_pct = None"],
            {"CFG_HQQ_AUTO_BUDGET_PCT", "CFG_HQQ46_AUTO_BUDGET_MB", "CFG_HQQ_SIDECAR_MANIFEST"},
        ))

        cfg = _base_config()
        cfg.attention_quant = "hqq4"
        cfg.kv_dtype = "k4v4"
        cfg.max_context_tokens = 2048
        cfg.hqq_auto_budget_pct = 20.0
        cfg.hqq46_auto_budget_mib = 128
        scenarios.append((
            "plain_hqq4_stale_budget",
            cfg,
            {
                "CFG_ATTENTION_QUANT": "hqq4",
                "CFG_KV_DTYPE": "k4v4",
                "CFG_MAX_CONTEXT_TOKENS": "2048",
            },
            [
                "attention_quant = 'hqq4'",
                "hqq_auto_budget_pct = None",
                "max_context_tokens = 2048",
            ],
            {"CFG_HQQ_AUTO_BUDGET_PCT", "CFG_HQQ46_AUTO_BUDGET_MB", "CFG_HQQ_SIDECAR_MANIFEST"},
        ))

        cfg = _base_config()
        cfg.attention_quant = "hqq46_auto"
        cfg.hqq_auto_budget_pct = 10.0
        scenarios.append((
            "hqq46_auto_10pct",
            cfg,
            {"CFG_ATTENTION_QUANT": "hqq46_auto", "CFG_HQQ_AUTO_BUDGET_PCT": "10.0"},
            ["attention_quant = 'hqq46_auto'", "hqq_auto_budget_pct = 10.0"],
            {"CFG_HQQ46_AUTO_BUDGET_MB", "CFG_HQQ_SIDECAR_MANIFEST"},
        ))

        cfg = _base_config()
        cfg.attention_quant = "hqq68_auto"
        cfg.hqq_auto_budget_pct = 10.0
        cfg.gpu_expert_bits = 8
        cfg.cpu_expert_bits = 8
        cfg.kv_dtype = "bf16"
        scenarios.append((
            "hqq68_auto_10pct_int8_bf16kv",
            cfg,
            {
                "CFG_ATTENTION_QUANT": "hqq68_auto",
                "CFG_HQQ_AUTO_BUDGET_PCT": "10.0",
                "CFG_GPU_EXPERT_BITS": "8",
                "CFG_CPU_EXPERT_BITS": "8",
                "CFG_KV_DTYPE": "bf16",
            },
            ["attention_quant = 'hqq68_auto'", "hqq_auto_budget_pct = 10.0"],
            {"CFG_HQQ46_AUTO_BUDGET_MB", "CFG_HQQ_SIDECAR_MANIFEST"},
        ))

        cfg = _base_config()
        cfg.attention_quant = "bf16"
        cfg.kv_dtype = "k4v4"
        cfg.dynamic_hcs = False
        cfg.dynamic_hcs_tail_blocks = 5
        cfg.dynamic_peer = True
        cfg.adaptive_cold_mass_pruning = "75/8"
        cfg.expert_compression_pipeline = "auto"
        cfg.enable_thinking = False
        cfg.force_load = True
        cfg.force_rebuild_cache = True
        cfg.force_rebuild_hqq_cache = True
        cfg.build_cache = True
        scenarios.append((
            "bf16_advanced_toggles",
            cfg,
            {
                "CFG_ATTENTION_QUANT": "bf16",
                "CFG_DYNAMIC_HCS": "0",
                "CFG_DYNAMIC_HCS_TAIL_BLOCKS": "5",
                "CFG_DYNAMIC_PEER": "1",
                "CFG_ADAPTIVE_COLD_MASS_PRUNING": "75/8",
                "CFG_EXPERT_COMPRESSION_PIPELINE": "auto",
                "CFG_ENABLE_THINKING": "0",
                "CFG_FORCE_LOAD": "1",
                "CFG_FORCE_REBUILD_CACHE": "1",
                "CFG_FORCE_REBUILD_HQQ_CACHE": "1",
                "CFG_BUILD_CACHE": "1",
            },
            [
                "attention_quant = 'bf16'",
                "dynamic_hcs = False",
                "dynamic_peer = True",
                "adaptive_cold_mass_pruning = '75/8'",
                "expert_compression_pipeline = 'auto'",
                "enable_thinking = False",
            ],
            {"CFG_HQQ_AUTO_BUDGET_PCT", "CFG_HQQ46_AUTO_BUDGET_MB", "CFG_HQQ_SIDECAR_MANIFEST"},
        ))

        cfg = _base_config()
        cfg.attention_quant = "hqq8"
        cfg.selected_gpu_indices = [0, 1]
        cfg.layer_group_size = 6
        cfg.expert_group_size = 64
        cfg.vram_safety_margin = 900
        cfg.port = 65_502
        cfg.ssh_tunnel = "alice@example.com:2222"
        cfg.ssh_key_path = "~/.ssh/id_ed25519"
        cfg.hcs = False
        cfg.multi_gpu_hcs = True
        cfg.multi_gpu_mode = "layer-split"
        cfg.heatmap_path = "~/heatmaps/qwen36.json"
        cfg.stream_attention = True
        cfg.draft_model = "~/models/draft"
        cfg.draft_k = 5
        cfg.draft_context = 1024
        cfg.temperature = 0.25
        scenarios.append((
            "hqq8_two_gpu_shape",
            cfg,
            {
                "CFG_ATTENTION_QUANT": "hqq8",
                "CFG_SELECTED_GPUS": "0,1",
                "CFG_NUM_GPUS": "2",
                "CFG_LAYER_GROUP_SIZE": "6",
                "CFG_EXPERT_GROUP_SIZE": "64",
                "CFG_VRAM_SAFETY_MARGIN": "900",
                "CFG_PORT": "65502",
                "CFG_SSH_TUNNEL": "alice@example.com:2222",
                "CFG_SSH_KEY_PATH": "~/.ssh/id_ed25519",
                "CFG_HCS": "0",
                "CFG_MULTI_GPU_HCS": "1",
                "CFG_MULTI_GPU_MODE": "layer-split",
                "CFG_HEATMAP_PATH": "~/heatmaps/qwen36.json",
                "CFG_STREAM_ATTENTION": "1",
                "CFG_DRAFT_MODEL": "~/models/draft",
                "CFG_DRAFT_K": "5",
                "CFG_DRAFT_CONTEXT": "1024",
                "CFG_TEMPERATURE": "0.25",
            },
            [
                "attention_quant = 'hqq8'",
                "num_gpus = 2",
                "selected_gpus = 'GPU-",
                "expert_group_size = 64",
                "ssh_tunnel = 'alice@example.com:2222'",
                f"ssh_key_path = '{os.path.expanduser('~/.ssh/id_ed25519')}'",
                "hcs = False",
                "multi_gpu_hcs = True",
                "multi_gpu_mode = 'layer-split'",
                f"heatmap_path = '{os.path.expanduser('~/heatmaps/qwen36.json')}'",
                "stream_attention = True",
                f"draft_model = '{os.path.expanduser('~/models/draft')}'",
                "draft_k = 5",
                "draft_context = 1024",
                "temperature = 0.25",
            ],
            {"CFG_HQQ_AUTO_BUDGET_PCT", "CFG_HQQ46_AUTO_BUDGET_MB", "CFG_HQQ_SIDECAR_MANIFEST"},
        ))

        generated_paths: list[Path] = []
        try:
            for name, cfg, expected, expected_output, absent_keys in scenarios:
                with self.subTest(name=name):
                    config_path, cmd_args = _capture_launch_config(cfg)
                    generated_paths.append(config_path)
                    values = _parse_key_value_config(config_path)
                    for key, value in expected.items():
                        self.assertEqual(values.get(key), value, f"{name}: {key}")
                    for key in absent_keys:
                        self.assertNotIn(key, values, f"{name}: inactive {key} should not be serialized")
                    self.assertIn("--config", cmd_args)
                    self.assertIn("--benchmark", cmd_args)
                    _run_server_start_smoke(config_path, name, expected_output)
        finally:
            for path in generated_paths:
                path.unlink(missing_ok=True)

    def test_legacy_blank_values_are_unset_before_argparse(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".conf", prefix="krasis-legacy-blank-", delete=False) as f:
            f.write(f'MODEL_PATH="{NONEXISTENT_MODEL}"\n')
            f.write('CFG_SELECTED_GPUS="0"\n')
            f.write('CFG_NUM_GPUS=""\n')
            f.write('CFG_KV_CACHE_MB=""\n')
            f.write('CFG_DYNAMIC_HCS_TAIL_BLOCKS=""\n')
            f.write('CFG_HQQ_AUTO_BUDGET_PCT=""\n')
            f.write('CFG_HQQ46_AUTO_BUDGET_MB=""\n')
            f.write('CFG_HQQ_SIDECAR_MANIFEST=""\n')
            f.write('CFG_GGUF_PATH=""\n')
            f.write('CFG_ATTENTION_QUANT="hqq6"\n')
            config_path = Path(f.name)
        try:
            _run_server_start_smoke(
                config_path,
                "legacy_blank_optional_values",
                [
                    "attention_quant = 'hqq6'",
                    "hqq_auto_budget_pct = None",
                    "hqq46_auto_budget_mib = None",
                    "dynamic_hcs_tail_blocks = 2",
                    "kv_cache_mb = 1000",
                    "num_gpus = 1",
                ],
            )
        finally:
            config_path.unlink(missing_ok=True)
