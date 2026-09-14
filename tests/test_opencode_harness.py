import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import importlib.util
import http.client
import http.server
import threading
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGURE = REPO_ROOT / "podman" / "opencode" / "configure.sh"
CONFIGURE_ADQ = REPO_ROOT / "podman" / "opencode" / "configure-adq.sh"
ADQ_RUNNER = REPO_ROOT / "podman" / "opencode" / "run-adq-suite.sh"
ADQ_NETWORK = REPO_ROOT / "podman" / "opencode" / "prepare-adq-network.sh"
ADQ_RELAY = REPO_ROOT / "podman" / "opencode" / "adq-model-relay.js"
ADQ_RENDERED_CAPTURE = REPO_ROOT / "podman" / "opencode" / "capture-adq-rendered-inputs.js"
ADQ_REPLAY_CAPTURE = REPO_ROOT / "podman" / "opencode" / "capture-adq-replay.sh"
ADQ_REPLAY_RELAY = REPO_ROOT / "podman" / "opencode" / "adq-replay-capture-relay.js"
ADQ_REPLAY_EXTRACT = REPO_ROOT / "tests" / "adq_extract_ledger_replay.py"
ADQ_REPLAY = REPO_ROOT / "tests" / "adq_replay_exact_state.py"
ADQ_REQUEST_SEQUENCE = REPO_ROOT / "tests" / "adq_replay_request_sequence.py"
ADQ_WITNESS_REVIEW = REPO_ROOT / "tests" / "adq_review_ledger_witness.py"
ADQ_WITNESS_CASES = REPO_ROOT / "tests" / "adq_build_ledger_witness_cases.py"
LEDGER_TASK = REPO_ROOT / "podman" / "opencode" / "adq-fixtures" / "ledger" / "TASK.md"
LEDGER_CONTRACT = REPO_ROOT / "podman" / "opencode" / "adq-fixtures" / "ledger" / "CONTRACT.json"
LEDGER_PUBLIC_TEST = REPO_ROOT / "podman" / "opencode" / "adq-fixtures" / "ledger" / "test" / "ledger.test.js"
LEDGER_ORACLE = REPO_ROOT / "podman" / "opencode" / "adq-oracles" / "ledger.test.mjs"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpencodeHarnessTests(unittest.TestCase):
    def test_adq_relay_injects_and_captures_explicit_greedy_temperature(self) -> None:
        received: list[dict] = []

        class Upstream(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                received.append(json.loads(self.rfile.read(length)))
                body = b'{"usage":{"prompt_tokens":3}}'
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args) -> None:
                pass

        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        with tempfile.TemporaryDirectory() as temp_dir:
            capture_dir = Path(temp_dir) / "capture"
            relay_port = self._unused_port()
            env = os.environ.copy()
            env.update(
                {
                    "ADq_UPSTREAM_HOST": "127.0.0.1",
                    "ADq_UPSTREAM_PORT": str(upstream.server_port),
                    "ADq_LISTEN_PORT": str(relay_port),
                    "ADq_CAPTURE_DIR": str(capture_dir),
                }
            )
            relay = subprocess.Popen(
                ["node", str(ADQ_RELAY)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                self._wait_for_port(relay_port)
                connection = http.client.HTTPConnection("127.0.0.1", relay_port, timeout=5)
                request = {"model": "test", "messages": [], "stream": False}
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=json.dumps(request),
                    headers={"content-type": "application/json"},
                )
                response = connection.getresponse()
                response.read()
                connection.close()
                self.assertEqual(response.status, 200)
                self.assertEqual(received, [{**request, "temperature": 0}])
                captures = []
                for meta_path in sorted(capture_dir.glob("*.meta.json")):
                    candidate = json.loads(meta_path.read_text())
                    if candidate["path"] == "/v1/chat/completions":
                        captures.append((meta_path, candidate))
                self.assertEqual(len(captures), 1)
                meta_path, meta = captures[0]
                raw_path = meta_path.with_name(
                    meta_path.name.removesuffix(".meta.json") + ".request.raw"
                )
                raw = json.loads(raw_path.read_text())
                self.assertEqual(raw["temperature"], 0)
                self.assertTrue(meta["greedy_temperature_zero"])

                connection = http.client.HTTPConnection("127.0.0.1", relay_port, timeout=5)
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=json.dumps({**request, "temperature": 0.6}),
                    headers={"content-type": "application/json"},
                )
                rejected = connection.getresponse()
                rejection_body = rejected.read().decode()
                connection.close()
                self.assertEqual(rejected.status, 400)
                self.assertIn("requires temperature 0", rejection_body)
                self.assertEqual(len(received), 1)
            finally:
                relay.terminate()
                relay.wait(timeout=5)

    @staticmethod
    def _unused_port() -> int:
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), http.server.BaseHTTPRequestHandler
        )
        port = server.server_port
        server.server_close()
        return port

    @staticmethod
    def _wait_for_port(port: int) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.2)
                connection.request("GET", "/v1/models")
                response = connection.getresponse()
                response.read()
                connection.close()
                return
            except OSError:
                time.sleep(0.05)
        raise AssertionError("ADQ relay did not start")

    def test_configure_separates_container_endpoint_from_host_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            (data_dir / "config").mkdir(parents=True)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            curl_log = root / "curl.log"
            fake_curl = bin_dir / "curl"
            fake_curl.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" > \"$KRASIS_TEST_CURL_LOG\"\n"
                "printf '%s\\n' '{\"data\":[{\"id\":\"test-model\",\"max_context_tokens\":524288}]}'\n",
                encoding="utf-8",
            )
            fake_curl.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "KRASIS_OPENCODE_DATA_DIR": str(data_dir),
                    "KRASIS_TEST_CURL_LOG": str(curl_log),
                    "PATH": f"{bin_dir}:{env['PATH']}",
                }
            )
            subprocess.run(
                [
                    str(CONFIGURE),
                    "http://host.containers.internal:8012/v1",
                    "http://127.0.0.1:8012/v1",
                    "test-model",
                    "8192",
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )

            config = json.loads((data_dir / "config" / "opencode.json").read_text())
            self.assertEqual(
                config["provider"]["krasis"]["options"]["baseURL"],
                "http://host.containers.internal:8012/v1",
            )
            self.assertIn("http://127.0.0.1:8012/v1/models", curl_log.read_text())

    def test_configure_requires_both_explicit_endpoints(self) -> None:
        result = subprocess.run(
            [str(CONFIGURE), "http://127.0.0.1:8012/v1", "test-model", "8192"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("host-probe-http-base-url", result.stderr)

    def test_adq_config_enables_only_local_engineering_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            (data_dir / "config").mkdir(parents=True)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_curl = bin_dir / "curl"
            fake_curl.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' '{\"data\":[{\"id\":\"test-model\",\"meta\":{\"n_ctx\":524288}}]}'\n",
                encoding="utf-8",
            )
            fake_curl.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "KRASIS_OPENCODE_DATA_DIR": str(data_dir),
                    "PATH": f"{bin_dir}:{env['PATH']}",
                }
            )
            subprocess.run(
                [
                    str(CONFIGURE_ADQ),
                    "http://host.containers.internal:8012/v1",
                    "http://127.0.0.1:8012/v1",
                    "test-model",
                    "16384",
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            config = json.loads((data_dir / "config" / "opencode.json").read_text())
            permissions = config["agent"]["krasis-adq"]["permission"]
            self.assertEqual(permissions["bash"], "allow")
            self.assertEqual(permissions["edit"], "allow")
            self.assertEqual(config["permission"]["webfetch"], "deny")
            self.assertEqual(config["share"], "disabled")
            self.assertFalse(config["provider"]["krasis"]["options"]["timeout"])
            self.assertNotIn("chunkTimeout", config["provider"]["krasis"]["options"])

    def test_adq_runner_has_objective_oracles_and_preservation_checks(self) -> None:
        text = ADQ_RUNNER.read_text(encoding="utf-8")
        self.assertIn("ledger config scheduler history", text)
        self.assertIn('KRASIS_ADQ_TASKS:-ledger config scheduler history', text)
        self.assertIn('ledger|config|scheduler|history', text)
        self.assertIn("public-tests.log", text)
        self.assertIn("hidden-tests.log", text)
        self.assertIn("notes_hash", text)
        self.assertIn("protected_hash", text)
        self.assertIn("test_files_changed", text)
        self.assertIn("forbidden_paths_changed", text)
        self.assertIn("worker_internal:true", text)
        self.assertIn("external_probe_blocked:true", text)
        self.assertIn("per_task_mount_only:true", text)
        self.assertIn("/adq-workspace", text)
        self.assertIn("--target-tokens 500000", text)
        self.assertIn('jq --rawfile history "$workspace/ISSUE_HISTORY.md"', text)
        self.assertIn('--title "ADQ $task"', text)
        self.assertIn('any(.[]; . >= 500000)', text)
        self.assertIn("observed_prompt_tokens", text)
        self.assertIn("KRASIS_ADQ_RENDERED_INPUTS", text)
        self.assertIn("rendered-input-preflight.template.json", text)
        self.assertIn("rendered-input-preflight.tokens.json", text)
        self.assertIn('"http://127.0.0.1:$PORT/apply-template"', text)
        self.assertIn('"http://127.0.0.1:$PORT/tokenize"', text)
        self.assertIn("trap cleanup EXIT", text)
        self.assertIn("podman rm --force --time 0", text)
        self.assertIn("fixture-manifest.sha256", text)
        self.assertIn("fixture-identity.json", text)
        self.assertIn("fixture_manifest_sha256", text)
        self.assertIn("oracle_sha256", text)
        self.assertIn("contract_version", text)
        self.assertIn('KRASIS_ADQ_CONTRACT_DIR', text)
        self.assertIn('sha256sum -c MANIFEST.sha256', text)
        self.assertIn('format_version:3', text)
        self.assertIn("jq -e '.temperature == 0'", text)
        self.assertIn("greedy_request_count", text)
        self.assertIn("collect_worktree_evidence", text)
        self.assertIn("ls-files --others --exclude-standard", text)
        self.assertIn("changed-paths.json", text)
        self.assertIn(
            'KRASIS_OPENCODE_DATA_DIR="$runtime" bash "$CONFIGURE_SCRIPT"', text
        )
        self.assertIn('bash "$NETWORK_SCRIPT" host.containers.internal', text)

    def test_adq_worktree_evidence_includes_untracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
            (workspace / "src").mkdir()
            (workspace / "test").mkdir()
            (workspace / "src" / "existing.js").write_text("export const value = 1;\n")
            subprocess.run(["git", "add", "."], cwd=workspace, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=ADQ",
                    "-c",
                    "user.email=adq@invalid",
                    "commit",
                    "-qm",
                    "baseline",
                ],
                cwd=workspace,
                check=True,
            )
            (workspace / "src" / "existing.js").write_text("export const value = 2;\n")
            (workspace / "src" / "new.js").write_text("export const added = true;\n")
            (workspace / "test" / "new.test.js").write_text("// regression\n")
            (workspace / "forbidden.txt").write_text("must be detected\n")
            patch = root / "worktree.patch"
            paths = root / "changed-paths.json"

            subprocess.run(
                [
                    "bash",
                    str(ADQ_RUNNER),
                    "--collect-worktree-evidence",
                    str(workspace),
                    str(patch),
                    str(paths),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                json.loads(paths.read_text()),
                ["forbidden.txt", "src/existing.js", "src/new.js", "test/new.test.js"],
            )
            patch_text = patch.read_text()
            self.assertIn("a/src/existing.js", patch_text)
            self.assertIn("a/src/new.js", patch_text)
            self.assertIn("a/test/new.test.js", patch_text)
            self.assertIn("a/forbidden.txt", patch_text)

    def test_ledger_duplicate_contract_is_explicit_and_consistent(self) -> None:
        task = LEDGER_TASK.read_text(encoding="utf-8")
        contract = json.loads(LEDGER_CONTRACT.read_text(encoding="utf-8"))
        public_test = LEDGER_PUBLIC_TEST.read_text(encoding="utf-8")
        oracle = LEDGER_ORACLE.read_text(encoding="utf-8")
        self.assertEqual(contract["contract"], "krasis-adq-ledger")
        self.assertEqual(contract["contract_version"], 2)
        self.assertEqual(
            contract["duplicate_identity"]["stage"],
            "after parseEvent normalization",
        )
        self.assertEqual(
            contract["duplicate_identity"]["declared_fields"],
            ["account", "amount", "supersedes"],
        )
        self.assertIn("after `parseEvent` normalization", task)
        self.assertIn("undeclared fields do not affect duplicate identity", task)
        self.assertIn("normalized declared fields", public_test)
        self.assertIn("normalized semantic duplicates", oracle)
        self.assertIn('"amount":"2"', public_test)
        self.assertIn('"amount":"2"', oracle)

    def test_adq_worker_network_is_internal_and_relay_is_fixed(self) -> None:
        network = ADQ_NETWORK.read_text(encoding="utf-8")
        relay = ADQ_RELAY.read_text(encoding="utf-8")
        self.assertIn("network create --internal", network)
        self.assertIn('CONTAINER="opencode-adq-test"', network)
        self.assertIn("unexpectedly has external network access", network)
        self.assertIn("unexpectedly has direct host access", network)
        self.assertIn('request.url.startsWith("/v1/")', relay)
        self.assertIn("ADQ_RELAY_TIMING prompt_tokens=", relay)
        self.assertIn("hostname: upstreamHost", relay)
        self.assertIn("ADq_CAPTURE_DIR", relay)
        self.assertIn("request_sha256", relay)
        self.assertIn("response_sha256", relay)
        self.assertIn("payload.temperature = 0", relay)
        self.assertIn("greedy_temperature_zero: prepared.greedy", relay)
        self.assertIn("/adq-relay-capture", network)
        self.assertNotIn("new URL(request.url", relay)

        rendered = ADQ_RENDERED_CAPTURE.read_text(encoding="utf-8")
        self.assertIn('postJson("/apply-template"', rendered)
        self.assertIn('postJson("/tokenize"', rendered)
        self.assertIn("input_token_ids_sha256", rendered)
        self.assertIn("inference_prompt_tokens", rendered)
        self.assertIn("inference_token_count_match", rendered)
        self.assertIn('hasOwnProperty.call(request, "tool_choice")', rendered)
        self.assertIn('hasOwnProperty.call(request, "enable_thinking")', rendered)
        self.assertIn("KRASIS_DEV_SCRIPT", rendered)

    def test_exact_state_capture_is_isolated_and_never_mutates_source(self) -> None:
        runner = ADQ_REPLAY_CAPTURE.read_text(encoding="utf-8")
        relay = ADQ_REPLAY_RELAY.read_text(encoding="utf-8")
        self.assertIn("podman network create --internal", runner)
        self.assertIn('cp -a "$SOURCE_RUNTIME" "$TEMP_ROOT/runtime"', runner)
        self.assertIn('cp -a "$SOURCE_WORKSPACE" "$TEMP_ROOT/workspace"', runner)
        self.assertIn('SOURCE_RUNTIME="$SOURCE_RUN/.runtime-$TASK"', runner)
        self.assertIn('SOURCE_WORKSPACE="$SOURCE_RUN/$TASK"', runner)
        self.assertIn("evidence is immutable", runner)
        self.assertIn("ADQ_EXACT_STATE_CAPTURE_SENTINEL_DO_NOT_USE_AS_MODEL_EVIDENCE", runner)
        self.assertIn('request.url !== "/v1/chat/completions"', relay)
        self.assertNotIn("upstream", relay.lower())

    def test_exact_state_capture_uses_caller_frozen_terminal_identity(self) -> None:
        module = load_script(ADQ_REPLAY_EXTRACT, "adq_extract_ledger_replay_test")
        terminal = "new retained failure ending"
        terminal_sha256 = module.sha256_bytes(terminal.encode("utf-8"))
        self.assertEqual(
            module.validate_terminal_text(terminal, terminal_sha256), terminal_sha256
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            module.validate_terminal_text(terminal + "!", terminal_sha256)
        source = ADQ_REPLAY_EXTRACT.read_text(encoding="utf-8")
        self.assertNotIn("Let me create the new ledger.js.", source)
        self.assertIn("--expected-terminal-text-sha256", source)
        self.assertIn("source campaign manifest is not immutably marked FAILED", source)
        self.assertIn('parser.add_argument("--task", default="ledger")', source)

    def test_exact_state_replay_requires_numerical_not_just_token_determinism(self) -> None:
        replay = ADQ_REPLAY.read_text(encoding="utf-8")
        self.assertIn('"per_token_data": response.get("per_token_data")', replay)
        self.assertIn('"first_token_top_k": response.get("first_token_top_k")', replay)
        self.assertIn('"numerically_deterministic"', replay)
        self.assertIn('"request_execution"', replay)
        self.assertIn('ThreadPoolExecutor', replay)
        self.assertIn('start_barrier.wait()', replay)
        self.assertIn('"--concurrent"', replay)
        self.assertIn('row["numerically_matches_first_replay"]', replay)

    def test_request_sequence_reconstructs_every_assistant_boundary(self) -> None:
        module = load_script(ADQ_REQUEST_SEQUENCE, "adq_replay_request_sequence_test")
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "inspect", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": "result"},
            {"role": "assistant", "content": "inspect more", "tool_calls": [{"id": "b"}]},
            {"role": "tool", "tool_call_id": "b", "content": "result 2"},
        ]
        prefixes = module.request_prefixes(messages)
        self.assertEqual([len(prefix) for prefix in prefixes], [2, 4, 6])
        self.assertEqual([prefix[-1]["role"] for prefix in prefixes], ["user", "tool", "tool"])
        self.assertEqual(messages, prefixes[-1])
        with self.assertRaisesRegex(ValueError, "no preceding assistant"):
            module.request_prefixes(messages[:2])
        with self.assertRaisesRegex(ValueError, "model boundary"):
            module.request_prefixes(messages + [{"role": "assistant", "content": "bad end"}])

    def test_witness_cases_can_freeze_a_distinct_current_diagnostic_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state = root / "state"
            state.mkdir()
            (state / "manifest.json").write_text(
                json.dumps({"source_task": "scheduler"}), encoding="utf-8"
            )
            (state / "preloop-input-token-ids.json").write_text(
                json.dumps({"input_token_ids": [7, 8]}), encoding="utf-8"
            )
            (state / "accepted-output-token-ids.json").write_text(
                json.dumps(
                    {
                        "terminal_stop_token_id": 1,
                        "retokenized_visible_candidate_token_ids": [20, 21],
                        "retokenized_visible_token_count": 2,
                        "retokenized_visible_candidate_sha256": "unused-current-mode",
                        "recorded_generated_token_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            replay = root / "replay.json"
            replay.write_text(
                json.dumps(
                    {
                        "deterministic": True,
                        "reproduces_accepted_observables": False,
                        "reconstructed_token_ids": [30, 31, 1],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "witness-inputs.json"
            env = os.environ.copy()
            env["KRASIS_DEV_SCRIPT"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ADQ_WITNESS_CASES),
                    "--state-dir",
                    str(state),
                    "--replay-report",
                    str(replay),
                    "--trajectory-source",
                    "deterministic-current-replay",
                    "--output",
                    str(output),
                    "--checkpoints",
                    "0,2",
                    "--profile",
                    "diagnostic",
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(artifact["original_output_token_ids_retained"])
            self.assertIn("post-failure diagnostic", artifact["token_identity_basis"])
            self.assertEqual(
                artifact["conversations"][0]["turns"][-1]["prompt"],
                "current-scheduler-output-prefix-2",
            )
            self.assertEqual(
                artifact["conversations"][0]["turns"][-1]["accepted_next_token_id"],
                1,
            )

    def test_ledger_witness_review_checks_exact_inputs_and_terminal_decision(self) -> None:
        module = load_script(ADQ_WITNESS_REVIEW, "adq_review_ledger_witness_test")
        input_ids = [7, 8]
        input_hash = module.canonical_sha256(input_ids)
        witness_input_hash = module.witness_input_sha256(input_ids)
        inputs = {
            "format": "krasis_adq_ledger_teacher_forced_inputs",
            "profile_id": "frozen-ledger",
            "trajectory_source": "accepted-retokenized-visible-text",
            "original_output_token_ids_retained": False,
            "token_identity_basis": "test basis",
            "terminal_stop_token_id": 1,
            "conversations": [
                {
                    "turns": [
                        {
                            "prompt": "accepted-ledger-output-prefix-2",
                            "input_token_ids": input_ids,
                            "input_sha256": input_hash,
                            "checkpoint_output_tokens": 2,
                            "accepted_next_token_id": 1,
                        }
                    ]
                }
            ],
        }
        witness = {
            "runtime": "llama-witness",
            "profile_id": "source",
            "max_new_tokens": 1,
            "conversations": [
                {
                    "turns": [
                        {
                            "prompt": "accepted-ledger-output-prefix-2",
                            "input_token_ids": input_ids,
                            "input_sha256": witness_input_hash,
                            "token_ids": [1],
                            "stopped_eos": True,
                            "per_token_data": [
                                {
                                    "top_k": [
                                        {"token_id": 1, "log_prob": -0.25},
                                        {"token_id": 9, "log_prob": -1.25},
                                    ]
                                }
                            ],
                        }
                    ]
                }
            ],
        }
        review = module.build_review(inputs, witness)
        self.assertTrue(review["source_matches_accepted_terminal_stop"])
        self.assertEqual(review["source_selected_matches_accepted_count"], 1)
        self.assertEqual(review["rows"][0]["accepted_source_top_k_rank"], 1)
        self.assertEqual(review["terminal_accepted_stop_source_rank"], 1)
        self.assertEqual(review["terminal_source_top_k"][1]["token_id"], 9)
        self.assertEqual(
            review["rows"][0]["input_sha256_canonical_json"], input_hash
        )
        self.assertEqual(
            review["rows"][0]["input_sha256_witness_csv"], witness_input_hash
        )
        witness["conversations"][0]["turns"][0]["input_token_ids"] = [7, 9]
        with self.assertRaisesRegex(ValueError, "input token IDs differ"):
            module.build_review(inputs, witness)

    def test_ledger_terminal_continuation_builder_is_final_boundary_only(self) -> None:
        source = ADQ_WITNESS_CASES.read_text(encoding="utf-8")
        self.assertIn('"--terminal-only"', source)
        self.assertIn("checkpoints != [visible_count]", source)
        self.assertIn('"terminal_only": args.terminal_only', source)


if __name__ == "__main__":
    unittest.main()
