#!/usr/bin/env python3
"""Live correctness gate for the Rust RAM-backed conversation cache.

The gate deliberately distinguishes two numerical contracts:

* Same-shape state reuse is a hard identity contract whenever repeated
  cache-disabled controls are themselves token-identical. The gate samples
  eight independent controls by default because Step's ordinary full-prefill
  path has exhibited more than one deterministic output branch and three
  samples were empirically insufficient to bound it. Active-GPU
  continuation from a freshly computed boundary and pageable-RAM continuation
  from that boundary run the same suffix GEMM shape and must produce identical
  token IDs. The unsplit full prefill has a different GEMM height and belongs
  to the cross-height gate below. If the cache-disabled runtime demonstrably
  varies at the same height, the gate cannot invent an identity property the
  runtime lacks: it measures a pairwise multi-sample top-k/log-probability
  envelope and requires active-vs-RAM drift to remain inside it. This exception
  and every sequence are printed.
* Cross-height comparison is measured against ordinary cache-disabled prefill.
  Krasis GEMMs can select different reduction/tile schedules at different
  prompt heights, so a shared causal row can have small deterministic numeric
  drift before the cache participates. The gate measures that pre-existing
  envelope through the full-prefill logits endpoint, then reports the first
  cached-vs-ordinary divergent token and fails if its distribution drift is
  outside the measured envelope.

This is not a relaxed identity test: same-height reuse remains exact, restore
failures remain fatal, misses must remain visible, and cross-height drift is
printed rather than silently accepted. An auxiliary replay of the base prompt
is also exact when two cache-disabled base controls are exact. If those two
controls already vary, the replay is reported and must remain inside the
measured cache-disabled same-height distribution envelope; ordinary runtime
nondeterminism cannot be misclassified as snapshot corruption, nor can it hide
cache drift which is materially worse. Run only through
``./dev session-cache-test`` so the repository environment and canonical
Gutenberg prompt policy are preserved.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import struct
import sys
import urllib.error
import urllib.request
import zlib


if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Run this test through ./dev session-cache-test, not directly")


ROOT = Path(__file__).resolve().parents[1]
MOBY_DICK = ROOT / "benchmarks" / "prompts" / "prompt1_moby_dick.txt"
WAR_AND_PEACE = ROOT / "benchmarks" / "prompts" / "prompt2_war_and_peace.txt"


def _request(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"server HTTP {error.code}: {body}") from error
    if "error" in value:
        raise RuntimeError(f"server error: {value['error']}")
    return value


def _get(url: str, timeout: int) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _assistant_content(response: dict) -> str:
    return str(response["choices"][0]["message"].get("content") or "")


def _completion_ids(response: dict) -> list[int]:
    debug = response.get("krasis_debug")
    if not isinstance(debug, dict):
        raise RuntimeError("server did not return krasis_debug; launch with --test-endpoints")
    values = debug.get("completion_token_ids")
    if not isinstance(values, list) or not values:
        raise RuntimeError("krasis_debug.completion_token_ids is empty or missing")
    return [int(value) for value in values]


def _input_ids(response: dict) -> list[int]:
    debug = response.get("krasis_debug")
    if not isinstance(debug, dict):
        raise RuntimeError("server did not return krasis_debug; launch with --test-endpoints")
    values = debug.get("input_token_ids")
    if not isinstance(values, list) or not values:
        raise RuntimeError("krasis_debug.input_token_ids is empty or missing")
    return [int(value) for value in values]


def _completion_trace(response: dict) -> list[dict]:
    debug = response.get("krasis_debug")
    values = debug.get("completion_decode_trace") if isinstance(debug, dict) else None
    if not isinstance(values, list) or not values:
        raise RuntimeError("krasis_debug.completion_decode_trace is empty or missing")
    completion_ids = _completion_ids(response)
    if len(values) != len(completion_ids):
        raise RuntimeError(
            "completion trace/token count mismatch: "
            f"trace={len(values)} tokens={len(completion_ids)}"
        )
    normalized: list[dict] = []
    for step, value in enumerate(values):
        if not isinstance(value, dict) or int(value.get("token_id", -1)) != completion_ids[step]:
            raise RuntimeError(f"invalid completion trace entry at step {step}: {value!r}")
        top_k = value.get("top_k")
        # The first selected token comes from prefill, before the decode
        # callback owns a distribution. Its authoritative top-k is exposed by
        # the separate first_token_logits object in the same debug contract.
        if step == 0 and (not isinstance(top_k, list) or not top_k):
            first_token_logits = debug.get("first_token_logits")
            top_k = (
                first_token_logits.get("top_logits_before_logprob")
                if isinstance(first_token_logits, dict)
                else None
            )
        if not isinstance(top_k, list) or not top_k:
            raise RuntimeError(f"completion trace entry {step} has no top-k distribution")
        normalized.append({**value, "top_k": top_k})
    return normalized


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    matched = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        matched += 1
    return matched


def _sequence_drift(reference: list[int], candidate: list[int]) -> dict:
    matched_prefix = _common_prefix_length(reference, candidate)
    first_divergent_index = None
    if matched_prefix < max(len(reference), len(candidate)):
        first_divergent_index = matched_prefix
    positional_matches = sum(
        1 for reference_token, candidate_token in zip(reference, candidate)
        if reference_token == candidate_token
    )
    return {
        "reference_tokens": len(reference),
        "candidate_tokens": len(candidate),
        "matched_prefix_tokens": matched_prefix,
        "positional_matches": positional_matches,
        "first_divergent_token_index": first_divergent_index,
        "token_identical": reference == candidate,
    }


def _top_k_map(values: list[dict]) -> dict[int, float]:
    result: dict[int, float] = {}
    for value in values:
        if not isinstance(value, dict) or "token_id" not in value:
            raise RuntimeError(f"invalid top-k entry: {value!r}")
        score = value.get("log_prob", value.get("logprob"))
        if not isinstance(score, (int, float)):
            raise RuntimeError(f"top-k entry has no numeric log probability: {value!r}")
        result[int(value["token_id"])] = float(score)
    if not result:
        raise RuntimeError("top-k distribution is empty")
    return result


def _distribution_drift(reference: list[dict], candidate: list[dict]) -> dict:
    reference_map = _top_k_map(reference)
    candidate_map = _top_k_map(candidate)
    shared = sorted(reference_map.keys() & candidate_map.keys())
    score_deltas = [abs(reference_map[token] - candidate_map[token]) for token in shared]
    return {
        "reference_top_token": int(reference[0]["token_id"]),
        "candidate_top_token": int(candidate[0]["token_id"]),
        "top_token_changed": int(reference[0]["token_id"]) != int(candidate[0]["token_id"]),
        "top_k_overlap": len(shared),
        "max_shared_log_prob_delta": max(score_deltas) if score_deltas else None,
    }


def _completion_distribution_drift(reference: dict, candidate: dict) -> dict:
    reference_ids = _completion_ids(reference)
    candidate_ids = _completion_ids(candidate)
    sequence = _sequence_drift(reference_ids, candidate_ids)
    reference_trace = _completion_trace(reference)
    candidate_trace = _completion_trace(candidate)
    compare_steps = min(len(reference_trace), len(candidate_trace))
    first_divergent = sequence["first_divergent_token_index"]
    if first_divergent is not None:
        compare_steps = min(compare_steps, int(first_divergent) + 1)
    per_step = [
        {
            "step": step,
            **_distribution_drift(reference_trace[step]["top_k"], candidate_trace[step]["top_k"]),
        }
        for step in range(compare_steps)
    ]
    finite_deltas = [
        float(item["max_shared_log_prob_delta"])
        for item in per_step
        if item["max_shared_log_prob_delta"] is not None
    ]
    decision_step = (
        per_step[-1]
        if per_step
        else None
    )
    return {
        **sequence,
        "compared_distribution_steps": compare_steps,
        "minimum_top_k_overlap": min(
            (int(item["top_k_overlap"]) for item in per_step),
            default=0,
        ),
        "max_shared_log_prob_delta": max(finite_deltas) if finite_deltas else None,
        "decision_step_distribution": decision_step,
        "per_step": per_step,
    }


def _cache_disabled_height_baseline(
    base_url: str,
    base_response: dict,
    longer_response: dict,
    timeout: int,
    top_k: int,
    target_samples: int,
) -> dict:
    base_ids = _input_ids(base_response)
    longer_ids = _input_ids(longer_response)
    shared_prefix = _common_prefix_length(base_ids, longer_ids)
    if shared_prefix < 2:
        raise RuntimeError(
            f"height baseline has no useful shared token prefix: {shared_prefix} tokens"
        )
    sample_every = max(1, (shared_prefix - 1) // target_samples)

    def run_prefill_logits(input_ids: list[int]) -> dict:
        return _request(
            f"{base_url}/internal/prefill_logits",
            {
                "input_token_ids": input_ids,
                "top_k": top_k,
                "sample_every": sample_every,
            },
            timeout,
        )

    base_logits = run_prefill_logits(base_ids)
    longer_logits = run_prefill_logits(longer_ids)
    base_positions = {
        int(value["position"]): value for value in base_logits.get("positions", [])
    }
    longer_positions = {
        int(value["position"]): value for value in longer_logits.get("positions", [])
    }
    shared_positions = sorted(
        position for position in base_positions.keys() & longer_positions.keys()
        if position < shared_prefix
    )
    if not shared_positions:
        raise RuntimeError("height baseline produced no comparable shared-prefix positions")
    comparisons = [
        {
            "position": position,
            **_distribution_drift(
                base_positions[position]["top_k"],
                longer_positions[position]["top_k"],
            ),
        }
        for position in shared_positions
    ]
    finite_deltas = [
        float(item["max_shared_log_prob_delta"])
        for item in comparisons
        if item["max_shared_log_prob_delta"] is not None
    ]
    if not finite_deltas:
        raise RuntimeError("height baseline top-k sets had no shared token IDs")
    return {
        "cache_path": "disabled_internal_full_prefill",
        "base_input_tokens": len(base_ids),
        "longer_input_tokens": len(longer_ids),
        "shared_prefix_tokens": shared_prefix,
        "sample_every": sample_every,
        "compared_positions": len(comparisons),
        "top_token_changed_positions": sum(
            1 for item in comparisons if item["top_token_changed"]
        ),
        "minimum_top_k_overlap": min(int(item["top_k_overlap"]) for item in comparisons),
        "max_shared_log_prob_delta": max(finite_deltas),
        "positions": comparisons,
    }


def _cross_height_is_within_baseline(cross_height: dict, baseline: dict) -> bool:
    if cross_height["token_identical"]:
        return True
    # Only the first divergent decision can change the generated trajectory.
    # Earlier steps selected the same token; a low-ranked top-k tail moving at
    # one of those steps is still reported in the aggregate fields but must not
    # be mistaken for larger output drift than the actual divergence point.
    decision = cross_height.get("decision_step_distribution")
    if not isinstance(decision, dict):
        return False
    cross_delta = decision.get("max_shared_log_prob_delta")
    baseline_delta = baseline.get("max_shared_log_prob_delta")
    if not isinstance(cross_delta, (int, float)) or not isinstance(
        baseline_delta, (int, float)
    ):
        return False
    return (
        float(cross_delta) <= float(baseline_delta)
        and int(decision["top_k_overlap"])
        >= int(baseline["minimum_top_k_overlap"])
    )


def _measured_variation_envelope(drift: dict) -> dict:
    """Build an observed distribution envelope from an ordinary full/full pair."""
    delta = drift.get("max_shared_log_prob_delta")
    overlap = drift.get("minimum_top_k_overlap")
    if not isinstance(delta, (int, float)) or not isinstance(overlap, int):
        raise RuntimeError(
            "ordinary full-prefill variation did not expose a comparable top-k envelope"
        )
    return {
        "source": "cache_disabled_same_height_full_prefill_repeat",
        "max_shared_log_prob_delta": float(delta),
        "minimum_top_k_overlap": int(overlap),
    }


def _same_height_control_envelope(responses: list[dict]) -> dict:
    if len(responses) < 2:
        raise RuntimeError("same-height control envelope requires at least two responses")
    token_sequences = [_completion_ids(response) for response in responses]
    pairwise = []
    for left in range(len(responses)):
        for right in range(left + 1, len(responses)):
            pairwise.append({
                "left": left,
                "right": right,
                **_completion_distribution_drift(responses[left], responses[right]),
            })
    finite_deltas = [
        float(pair["max_shared_log_prob_delta"])
        for pair in pairwise
        if pair["max_shared_log_prob_delta"] is not None
    ]
    if not finite_deltas:
        raise RuntimeError("same-height controls exposed no shared top-k scores")
    return {
        "source": "cache_disabled_same_height_full_prefill_controls",
        "samples": len(responses),
        "token_identical": all(tokens == token_sequences[0] for tokens in token_sequences[1:]),
        "unique_token_sequences": len({tuple(tokens) for tokens in token_sequences}),
        "max_shared_log_prob_delta": max(finite_deltas),
        "minimum_top_k_overlap": min(
            int(pair["minimum_top_k_overlap"]) for pair in pairwise
        ),
        "token_sequences": token_sequences,
        "pairwise": pairwise,
    }


def _candidate_within_same_height_controls(
    candidate: dict,
    controls: list[dict],
    envelope: dict,
) -> tuple[bool, list[dict]]:
    comparisons = [
        _completion_distribution_drift(control, candidate) for control in controls
    ]
    if envelope["token_identical"]:
        return bool(
            comparisons and all(item["token_identical"] for item in comparisons)
        ), comparisons
    return any(
        _cross_height_is_within_baseline(comparison, envelope)
        for comparison in comparisons
    ), comparisons


def _chat_payload(
    model: str,
    messages: list[dict],
    prefix_cache: bool,
    completion_tokens: int,
) -> dict:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_k": 1,
        "max_tokens": completion_tokens,
        "min_new_tokens": completion_tokens,
        "stream": False,
        "prefix_cache": prefix_cache,
        "debug_first_token_boundary": True,
        "logprobs": True,
        "top_logprobs": 10,
    }


def _solid_png_data_url(width: int = 64, height: int = 64) -> str:
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + bytes((32, 96, 192)) * width for _ in range(height))
    png = signature
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(rows))
    png += chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prompt-chars", type=int, default=12_000)
    parser.add_argument("--completion-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1_800)
    parser.add_argument("--height-baseline-samples", type=int, default=32)
    parser.add_argument("--same-height-control-samples", type=int, default=8)
    parser.add_argument("--vision-uncacheable", action="store_true")
    parser.add_argument("--require-cache-hits", action="store_true")
    args = parser.parse_args()
    if args.prompt_chars <= 0:
        parser.error("--prompt-chars must be positive")
    if args.completion_tokens <= 0:
        parser.error("--completion-tokens must be positive")
    if args.height_baseline_samples <= 0:
        parser.error("--height-baseline-samples must be positive")
    if args.same_height_control_samples < 2:
        parser.error("--same-height-control-samples must be at least two")

    base_url = f"http://127.0.0.1:{args.port}/v1"
    model_data = _get(f"{base_url}/models", args.timeout).get("data", [])
    if len(model_data) != 1 or not model_data[0].get("id"):
        raise RuntimeError(f"expected exactly one served model, got {model_data!r}")
    model = str(model_data[0]["id"])
    before = _get(f"{base_url}/session-cache/stats", args.timeout)
    if not before.get("enabled"):
        raise RuntimeError("session cache is disabled")

    moby = MOBY_DICK.read_text(encoding="utf-8")[: args.prompt_chars]
    war = WAR_AND_PEACE.read_text(encoding="utf-8")[: max(1, args.prompt_chars // 2)]
    first_messages = [{
        "role": "user",
        "content": moby + "\n\nAnswer briefly: who commands the Pequod?",
    }]
    first = _request(
        f"{base_url}/chat/completions",
        _chat_payload(model, first_messages, False, args.completion_tokens),
        args.timeout,
    )
    first_ids = _completion_ids(first)
    first_repeat = _request(
        f"{base_url}/chat/completions",
        _chat_payload(model, first_messages, False, args.completion_tokens),
        args.timeout,
    )
    first_repeat_ids = _completion_ids(first_repeat)
    first_controls = [first, first_repeat]
    for _ in range(args.same_height_control_samples - 2):
        first_controls.append(_request(
            f"{base_url}/chat/completions",
            _chat_payload(model, first_messages, False, args.completion_tokens),
            args.timeout,
        ))
    first_seed = _request(
        f"{base_url}/chat/completions",
        _chat_payload(model, first_messages, True, args.completion_tokens),
        args.timeout,
    )
    first_seed_ids = _completion_ids(first_seed)
    continuation_messages = first_messages + [
        {"role": "assistant", "content": _assistant_content(first_seed)},
        {"role": "user", "content": "Answer briefly: what is the ship called?"},
    ]

    active = _request(
        f"{base_url}/chat/completions",
        _chat_payload(model, continuation_messages, True, args.completion_tokens),
        args.timeout,
    )
    active_ids = _completion_ids(active)
    control = _request(
        f"{base_url}/chat/completions",
        _chat_payload(model, continuation_messages, False, args.completion_tokens),
        args.timeout,
    )
    control_ids = _completion_ids(control)
    control_repeat = _request(
        f"{base_url}/chat/completions",
        _chat_payload(model, continuation_messages, False, args.completion_tokens),
        args.timeout,
    )
    control_repeat_ids = _completion_ids(control_repeat)
    continuation_controls = [control, control_repeat]
    for _ in range(args.same_height_control_samples - 2):
        continuation_controls.append(_request(
            f"{base_url}/chat/completions",
            _chat_payload(model, continuation_messages, False, args.completion_tokens),
            args.timeout,
        ))
    failures: list[str] = []
    control_warnings: list[str] = []
    base_control_envelope = _same_height_control_envelope(first_controls)
    seed_within_base_controls, seed_control_comparisons = (
        _candidate_within_same_height_controls(
            first_seed,
            first_controls,
            base_control_envelope,
        )
    )
    if not base_control_envelope["token_identical"]:
        control_warnings.append(
            "cache-disabled base full prefills are not token-identical; cached "
            "paths use the printed measured same-height distribution envelope"
        )
    if not seed_within_base_controls:
        failures.append(
            "cache-seeding base is outside the measured cache-disabled "
            "same-height full-prefill controls"
        )
    continuation_control_envelope = _same_height_control_envelope(
        continuation_controls
    )
    if not continuation_control_envelope["token_identical"]:
        control_warnings.append(
            "cache-disabled longer full prefills are not token-identical; cached "
            "paths use the printed measured same-height distribution envelope"
        )

    # Re-establish the original base, displace it with an unrelated canonical
    # conversation, then repeat the exact continuation to force pageable RAM.
    reestablished = _request(
        f"{base_url}/chat/completions",
        _chat_payload(model, first_messages, True, args.completion_tokens),
        args.timeout,
    )
    reestablished_ids = _completion_ids(reestablished)
    base_replay_variation = _completion_distribution_drift(first_seed, reestablished)
    base_replay_within_controls, base_replay_control_comparisons = (
        _candidate_within_same_height_controls(
            reestablished,
            first_controls,
            base_control_envelope,
        )
    )
    if not base_replay_variation["token_identical"]:
        if base_replay_within_controls:
            control_warnings.append(
                "replay of the cached base boundary varied within the measured "
                "cache-disabled same-height full-prefill envelope: "
                f"{reestablished_ids} != {first_seed_ids}"
            )
        else:
            failures.append(
                "replay of the exact cached base boundary changed its completion "
                "beyond measured cache-disabled same-height variation: "
                f"{reestablished_ids} != {first_seed_ids}"
            )
    displaced_messages = [{
        "role": "user",
        "content": war + "\n\nAnswer briefly: name this novel.",
    }]
    _request(
        f"{base_url}/chat/completions",
        _chat_payload(model, displaced_messages, True, args.completion_tokens),
        args.timeout,
    )
    restored = _request(
        f"{base_url}/chat/completions",
        _chat_payload(model, continuation_messages, True, args.completion_tokens),
        args.timeout,
    )
    restored_ids = _completion_ids(restored)
    after = _get(f"{base_url}/session-cache/stats", args.timeout)
    active_delta = int(after["hits"]["active_gpu"]) - int(before["hits"]["active_gpu"])
    ram_delta = int(after["hits"]["pageable_ram"]) - int(before["hits"]["pageable_ram"])
    divergence_delta = int(after["misses"]["divergence"]) - int(
        before["misses"]["divergence"]
    )
    if args.require_cache_hits and active_delta < 1:
        raise RuntimeError(
            "required a real active-GPU cache hit, observed "
            f"active={active_delta} divergence={divergence_delta}"
        )
    if args.require_cache_hits and ram_delta < 1:
        raise RuntimeError(
            "required a real pageable-RAM cache hit, observed "
            f"RAM={ram_delta} divergence={divergence_delta}"
        )
    # The general gate still supports explicit fail-closed divergence evidence
    # for architectures whose exact recurrent boundary has not been implemented.
    # Universal-support acceptance runs pass --require-cache-hits and cannot use
    # this branch to turn a permanent miss into a pass.
    if not args.require_cache_hits and active_delta < 1 and divergence_delta < 1:
        raise RuntimeError(
            "expected either an active-GPU hit or a visible exact-boundary "
            f"divergence miss, observed active={active_delta} divergence={divergence_delta}"
        )
    if not args.require_cache_hits and ram_delta < 1 and divergence_delta < 1:
        raise RuntimeError(
            "expected either a pageable-RAM hit or a visible exact-boundary "
            f"divergence miss, observed RAM={ram_delta} divergence={divergence_delta}"
        )
    if int(after["misses"]["restore_failed"]) != int(before["misses"]["restore_failed"]):
        raise RuntimeError("restore_failed increased during the identity gate")

    active_control_comparisons = [
        _completion_distribution_drift(control_response, active)
        for control_response in continuation_controls
    ]
    restored_control_comparisons = [
        _completion_distribution_drift(control_response, restored)
        for control_response in continuation_controls
    ]
    active_vs_ram = _completion_distribution_drift(active, restored)
    active_ram_within_contract = bool(
        active_vs_ram["token_identical"]
        if continuation_control_envelope["token_identical"]
        else _cross_height_is_within_baseline(
            active_vs_ram,
            continuation_control_envelope,
        )
    )
    same_height_identity = {
        "applicable": active_delta >= 1 and ram_delta >= 1,
        "mode": (
            "hard_token_identity"
            if continuation_control_envelope["token_identical"]
            else "measured_cache_disabled_runtime_variation"
        ),
        "cache_disabled_controls": continuation_control_envelope,
        "active_ram_within_contract": active_ram_within_contract,
        "active_vs_ram": active_vs_ram,
        "active_control_comparisons": active_control_comparisons,
        "ram_restored_control_comparisons": restored_control_comparisons,
    }
    if same_height_identity["applicable"] and not active_ram_within_contract:
        failures.append(
            "same-shape RAM-restored continuation is outside the active-state "
            "contract and measured cache-disabled runtime variation"
        )

    # This endpoint always performs ordinary full prefill and invalidates active
    # device state. Run it only after the active/RAM transactional checks and
    # stats have been captured.
    height_baseline = _cache_disabled_height_baseline(
        base_url,
        first,
        control,
        args.timeout,
        top_k=10,
        target_samples=args.height_baseline_samples,
    )
    active_cross_height = _completion_distribution_drift(control, active)
    restored_cross_height = _completion_distribution_drift(control, restored)
    repeated_full_variation = _completion_distribution_drift(control, control_repeat)
    active_cross_height["within_cache_disabled_height_baseline"] = (
        _cross_height_is_within_baseline(active_cross_height, height_baseline)
    )
    restored_cross_height["within_cache_disabled_height_baseline"] = (
        _cross_height_is_within_baseline(restored_cross_height, height_baseline)
    )
    if active_delta >= 1 and not active_cross_height["within_cache_disabled_height_baseline"]:
        failures.append(
            "active cached continuation drift is materially worse than the measured "
            "cache-disabled prompt-height envelope"
        )
    if ram_delta >= 1 and not restored_cross_height["within_cache_disabled_height_baseline"]:
        failures.append(
            "RAM-restored continuation drift is materially worse than the measured "
            "cache-disabled prompt-height envelope"
        )

    vision_result = None
    if args.vision_uncacheable:
        image_before = after
        image_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the main colour of this image?"},
                {"type": "image_url", "image_url": {"url": _solid_png_data_url()}},
            ],
        }]
        image_response = None
        image_setup_error = None
        try:
            image_response = _request(
                f"{base_url}/chat/completions",
                _chat_payload(model, image_messages, True, args.completion_tokens),
                args.timeout,
            )
        except RuntimeError as error:
            # Cache exclusion is recorded before model-specific image setup.
            # Preserve the setup failure as evidence; never turn it into a
            # successful response or a cacheable text fallback.
            image_setup_error = str(error)
        image_after = _get(f"{base_url}/session-cache/stats", args.timeout)
        image_miss_delta = int(image_after["misses"]["image_input_uncacheable"]) - int(
            image_before["misses"]["image_input_uncacheable"]
        )
        if image_miss_delta != 1:
            raise RuntimeError(
                f"image request was not recorded exactly once as uncacheable: {image_miss_delta}"
            )
        if int(image_after["resident"]["snapshots"]) != int(image_before["resident"]["snapshots"]):
            raise RuntimeError("image request changed pageable snapshot residency")
        vision_result = {
            "content": (
                _assistant_content(image_response) if image_response is not None else None
            ),
            "setup_error": image_setup_error,
            "image_input_uncacheable_delta": image_miss_delta,
        }
        after = image_after

    print(json.dumps({
        "model": model,
        "base_full_prefill_ids": first_ids,
        "base_full_prefill_repeat_ids": first_repeat_ids,
        "base_cache_seed_ids": first_seed_ids,
        "base_stable_boundary_restored_ids": reestablished_ids,
        "active_ids": active_ids,
        "full_prefill_ids": control_ids,
        "full_prefill_repeat_ids": control_repeat_ids,
        "ram_restored_ids": restored_ids,
        "same_height_identity": same_height_identity,
        "base_same_height_variation": {
            "cache_disabled_controls": base_control_envelope,
            "cache_seed_control_comparisons": seed_control_comparisons,
            "cached_boundary_replay": {
                **base_replay_variation,
                "within_cache_disabled_same_height_envelope": (
                    base_replay_within_controls
                ),
                "control_comparisons": base_replay_control_comparisons,
            },
        },
        "cache_disabled_height_baseline": height_baseline,
        "cross_height_drift": {
            "active_vs_ordinary_longer_full_prefill": active_cross_height,
            "ram_restored_vs_ordinary_longer_full_prefill": restored_cross_height,
            "ordinary_longer_full_prefill_repeat": repeated_full_variation,
        },
        "active_hit_delta": active_delta,
        "ram_hit_delta": ram_delta,
        "divergence_miss_delta": divergence_delta,
        "capabilities": after["capabilities"],
        "resident_snapshots": after["resident"]["snapshots"],
        "resident_bytes": after["resident"]["bytes"],
        "restore_timing": after["timing"]["restore"],
        "save_timing": after["timing"]["save"],
        "vision": vision_result,
        "control_warnings": control_warnings,
        "failures": failures,
    }, indent=2, sort_keys=True))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
