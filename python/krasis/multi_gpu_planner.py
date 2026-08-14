"""Runtime-measured contiguous layer planning for serial multi-GPU decode.

The decode pipeline executes GPU segments serially.  VRAM capacity therefore
cannot be the only input to the split: a layer assigned to a slower device is
on the token critical path.  This module measures the real cold-expert copy
topology and a model-derived layer working set on every selected device, then
uses those measurements with the loaded layer/expert geometry to compare every
valid contiguous assignment.

This is startup-only orchestration.  No Python code is introduced into the
per-token path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Callable, Sequence


@dataclass(frozen=True)
class DeviceServiceProfile:
    gpu_index: int
    probe_bytes: int
    d2d_probe_bytes: int
    h2d_seconds_per_byte: float
    d2d_seconds_per_byte: float
    h2d_p50_us: float
    h2d_p95_us: float
    d2d_p50_us: float
    d2d_p95_us: float

    @property
    def relative_uncertainty(self) -> float:
        """One-sided measured tail spread used for split-change admission."""
        h2d = max(0.0, self.h2d_p95_us / self.h2d_p50_us - 1.0)
        d2d = max(0.0, self.d2d_p95_us / self.d2d_p50_us - 1.0)
        return max(h2d, d2d)


@dataclass(frozen=True)
class SplitPlan:
    splits: tuple[int, ...]
    predicted_seconds_per_token: float
    preferred_seconds_per_token: float
    admitted: bool
    uncertainty_seconds: float


@dataclass(frozen=True)
class PeerPlan:
    """Prediction for a primary-plus-peer expert-serving topology."""

    peer_residents: tuple[tuple[int, int], ...]
    primary_residents: tuple[tuple[int, int], ...]
    captured_routes_per_token: float
    cold_routes_before_per_token: float
    cold_routes_after_per_token: float
    captured_cold_fraction: float
    predicted_seconds_per_token: float
    predicted_primary_only_seconds_per_token: float


def peer_plan_is_admissible(
    *,
    peer_plan: PeerPlan,
    layer_split_seconds_per_token: float,
    uncertainty_seconds: float,
    admitted_route_counts: Sequence[bool],
) -> bool:
    """Return whether a measured peer plan can perform useful peer work."""
    return bool(
        peer_plan.peer_residents
        and peer_plan.captured_routes_per_token > 0.0
        and any(admitted_route_counts)
        and peer_plan.predicted_seconds_per_token + uncertainty_seconds
        < layer_split_seconds_per_token
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = quantile * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def measure_device_service_profiles(
    device_indices: Sequence[int],
    component_bytes: Sequence[int],
    d2d_probe_bytes: int,
    *,
    warmup_samples: int = 3,
    measured_samples: int = 17,
) -> list[DeviceServiceProfile]:
    """Measure pinned H2D and model-working-set D2D service using CUDA events.

    H2D reproduces one loaded expert's actual one-to-four component copy
    topology.  D2D uses a size derived by the caller from the loaded model's
    maximum resident layer or routed-expert working set.  Fixed sample counts
    are statistical protocol parameters, not hardware/model performance
    constants.  Any allocation or CUDA failure propagates and fails startup;
    there is deliberately no advertised-bandwidth fallback.
    """
    components = tuple(int(size) for size in component_bytes)
    if not components or any(size <= 0 for size in components):
        raise ValueError("component_bytes must contain only positive sizes")
    probe_bytes = sum(components)
    if d2d_probe_bytes <= 0:
        raise ValueError("d2d_probe_bytes must be positive")
    if measured_samples <= 0 or warmup_samples < 0:
        raise ValueError("invalid service calibration sample counts")

    import torch

    host = torch.empty(probe_bytes, dtype=torch.uint8, pin_memory=True)
    profiles: list[DeviceServiceProfile] = []
    try:
        for gpu_index in device_indices:
            with torch.cuda.device(gpu_index):
                h2d_dst = torch.empty(probe_bytes, dtype=torch.uint8, device=f"cuda:{gpu_index}")
                d_src = torch.empty(d2d_probe_bytes, dtype=torch.uint8, device=f"cuda:{gpu_index}")
                d_dst = torch.empty_like(d_src)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                h2d_us: list[float] = []
                d2d_us: list[float] = []
                total = warmup_samples + measured_samples
                for sample in range(total):
                    start.record()
                    offset = 0
                    for component_size in components:
                        h2d_dst.narrow(0, offset, component_size).copy_(
                            host.narrow(0, offset, component_size),
                            non_blocking=True,
                        )
                        offset += component_size
                    end.record()
                    end.synchronize()
                    h2d_elapsed_us = float(start.elapsed_time(end)) * 1_000.0

                    start.record()
                    d_dst.copy_(d_src, non_blocking=True)
                    end.record()
                    end.synchronize()
                    d2d_elapsed_us = float(start.elapsed_time(end)) * 1_000.0

                    if sample >= warmup_samples:
                        h2d_us.append(h2d_elapsed_us)
                        d2d_us.append(d2d_elapsed_us)

                h2d_p50 = statistics.median(h2d_us)
                d2d_p50 = statistics.median(d2d_us)
                if h2d_p50 <= 0.0 or d2d_p50 <= 0.0:
                    raise RuntimeError(
                        f"cuda:{gpu_index} service calibration returned a non-positive latency"
                    )
                profiles.append(
                    DeviceServiceProfile(
                        gpu_index=int(gpu_index),
                        probe_bytes=int(probe_bytes),
                        d2d_probe_bytes=int(d2d_probe_bytes),
                        h2d_seconds_per_byte=(h2d_p50 / 1_000_000.0) / probe_bytes,
                        d2d_seconds_per_byte=(d2d_p50 / 1_000_000.0) / d2d_probe_bytes,
                        h2d_p50_us=h2d_p50,
                        h2d_p95_us=_percentile(h2d_us, 0.95),
                        d2d_p50_us=d2d_p50,
                        d2d_p95_us=_percentile(d2d_us, 0.95),
                    )
                )
                del h2d_dst, d_dst, d_src, start, end
                torch.cuda.synchronize(gpu_index)
                torch.cuda.empty_cache()
    finally:
        del host
    return profiles


def _boundaries(splits: Sequence[int], num_layers: int) -> tuple[int, ...]:
    return (0, *tuple(int(value) for value in splits), int(num_layers))


def _segment_cost_seconds(
    gpu_ordinal: int,
    layer_start: int,
    layer_end: int,
    layer_resident_bytes: Sequence[int],
    layer_is_moe: Sequence[bool],
    profiles: Sequence[DeviceServiceProfile],
    hcs_budget_bytes: Callable[[int, int, int], int],
    expert_bytes: int,
    experts_per_layer: int,
    experts_per_token: int,
    terminal_bytes: int,
) -> float:
    profile = profiles[gpu_ordinal]
    resident_layer_bytes = sum(layer_resident_bytes[layer_start:layer_end])
    moe_layers = sum(1 for is_moe in layer_is_moe[layer_start:layer_end] if is_moe)
    segment_experts = moe_layers * experts_per_layer
    resident_experts = min(
        segment_experts,
        max(0, hcs_budget_bytes(gpu_ordinal, layer_start, layer_end)) // expert_bytes,
    )
    resident_fraction = resident_experts / segment_experts if segment_experts else 1.0

    routed_expert_bytes = moe_layers * experts_per_token * expert_bytes
    cold_expert_bytes = routed_expert_bytes * (1.0 - resident_fraction)
    device_bytes = resident_layer_bytes + routed_expert_bytes
    if gpu_ordinal == len(profiles) - 1:
        device_bytes += terminal_bytes
    return (
        device_bytes * profile.d2d_seconds_per_byte
        + cold_expert_bytes * profile.h2d_seconds_per_byte
    )


def predicted_split_seconds(
    splits: Sequence[int],
    *,
    layer_resident_bytes: Sequence[int],
    layer_is_moe: Sequence[bool],
    profiles: Sequence[DeviceServiceProfile],
    hcs_budget_bytes: Callable[[int, int, int], int],
    expert_bytes: int,
    experts_per_layer: int,
    experts_per_token: int,
    terminal_bytes: int,
) -> float:
    num_layers = len(layer_resident_bytes)
    boundaries = _boundaries(splits, num_layers)
    if len(boundaries) != len(profiles) + 1:
        raise ValueError("split/profile cardinality mismatch")
    return sum(
        _segment_cost_seconds(
            gpu,
            boundaries[gpu],
            boundaries[gpu + 1],
            layer_resident_bytes,
            layer_is_moe,
            profiles,
            hcs_budget_bytes,
            expert_bytes,
            experts_per_layer,
            experts_per_token,
            terminal_bytes,
        )
        for gpu in range(len(profiles))
    )


def optimize_contiguous_splits(
    *,
    preferred_splits: Sequence[int],
    layer_resident_bytes: Sequence[int],
    layer_is_moe: Sequence[bool],
    profiles: Sequence[DeviceServiceProfile],
    hcs_budget_bytes: Callable[[int, int, int], int],
    expert_bytes: int,
    experts_per_layer: int,
    experts_per_token: int,
    terminal_bytes: int,
    minimum_layers_per_device: int = 2,
) -> SplitPlan:
    """Minimize predicted serial service time with contiguous dynamic programming.

    If the measured improvement does not clear the devices' observed p50→p95
    spread, retain the existing VRAM-derived assignment.  This makes unchanged
    matched-card plans the deterministic tie winner and prevents calibration
    noise from regressing a working configuration.
    """
    num_layers = len(layer_resident_bytes)
    num_devices = len(profiles)
    if len(layer_is_moe) != num_layers:
        raise ValueError("layer byte and MoE masks differ in length")
    if num_devices < 2 or len(preferred_splits) != num_devices - 1:
        raise ValueError("invalid preferred split cardinality")
    if expert_bytes <= 0 or experts_per_layer <= 0 or experts_per_token <= 0:
        raise ValueError("expert geometry must be positive")
    if num_layers < num_devices * minimum_layers_per_device:
        raise ValueError("not enough layers for the minimum per-device segment")

    preferred = tuple(int(value) for value in preferred_splits)
    preferred_seconds = predicted_split_seconds(
        preferred,
        layer_resident_bytes=layer_resident_bytes,
        layer_is_moe=layer_is_moe,
        profiles=profiles,
        hcs_budget_bytes=hcs_budget_bytes,
        expert_bytes=expert_bytes,
        experts_per_layer=experts_per_layer,
        experts_per_token=experts_per_token,
        terminal_bytes=terminal_bytes,
    )

    # DP state after assigning `gpu_count` devices and `layer_end` layers:
    # predicted seconds, distance from the legacy boundary positions, splits.
    states: dict[tuple[int, int], tuple[float, int, tuple[int, ...]]] = {
        (0, 0): (0.0, 0, ())
    }
    for gpu in range(num_devices):
        remaining_devices = num_devices - gpu - 1
        next_states: dict[tuple[int, int], tuple[float, int, tuple[int, ...]]] = {}
        for (assigned_devices, layer_start), (cost, penalty, splits) in states.items():
            if assigned_devices != gpu:
                continue
            minimum_end = layer_start + minimum_layers_per_device
            maximum_end = num_layers - remaining_devices * minimum_layers_per_device
            for layer_end in range(minimum_end, maximum_end + 1):
                if gpu == num_devices - 1 and layer_end != num_layers:
                    continue
                segment_cost = _segment_cost_seconds(
                    gpu,
                    layer_start,
                    layer_end,
                    layer_resident_bytes,
                    layer_is_moe,
                    profiles,
                    hcs_budget_bytes,
                    expert_bytes,
                    experts_per_layer,
                    experts_per_token,
                    terminal_bytes,
                )
                next_splits = splits if gpu == num_devices - 1 else (*splits, layer_end)
                next_penalty = penalty
                if gpu < num_devices - 1:
                    next_penalty += (layer_end - preferred[gpu]) ** 2
                candidate = (cost + segment_cost, next_penalty, next_splits)
                key = (gpu + 1, layer_end)
                current = next_states.get(key)
                if current is None or candidate[:2] < current[:2]:
                    next_states[key] = candidate
        states = next_states

    best = states.get((num_devices, num_layers))
    if best is None:
        raise RuntimeError("no valid contiguous multi-GPU layer assignment")
    best_seconds, _, best_splits = best
    relative_uncertainty = max(profile.relative_uncertainty for profile in profiles)
    uncertainty_seconds = max(preferred_seconds, best_seconds) * relative_uncertainty
    admitted = preferred_seconds - best_seconds > uncertainty_seconds
    if not admitted:
        return SplitPlan(
            splits=preferred,
            predicted_seconds_per_token=preferred_seconds,
            preferred_seconds_per_token=preferred_seconds,
            admitted=False,
            uncertainty_seconds=uncertainty_seconds,
        )
    return SplitPlan(
        splits=best_splits,
        predicted_seconds_per_token=best_seconds,
        preferred_seconds_per_token=preferred_seconds,
        admitted=True,
        uncertainty_seconds=uncertainty_seconds,
    )


def _service_curve_us(expected_routes: float, curve_us: Sequence[float]) -> float:
    """Conservatively interpolate a measured integer route-count curve."""
    if expected_routes <= 0.0:
        return 0.0
    if not curve_us or any(not math.isfinite(value) or value <= 0.0 for value in curve_us):
        raise ValueError("peer service curve must contain finite positive measurements")
    capped = min(expected_routes, float(len(curve_us)))
    lower_routes = max(1, math.floor(capped))
    upper_routes = min(len(curve_us), math.ceil(capped))
    if lower_routes == upper_routes:
        return float(curve_us[lower_routes - 1])
    fraction = capped - lower_routes
    return (
        float(curve_us[lower_routes - 1]) * (1.0 - fraction)
        + float(curve_us[upper_routes - 1]) * fraction
    )


def predict_peer_expert_plan(
    *,
    heatmap_counts: dict[str, int],
    total_decode_tokens: int,
    ranking: Sequence[tuple[int, int]],
    primary_capacity_experts: int,
    peer_capacity_experts: int,
    layer_resident_bytes: Sequence[int],
    layer_is_moe: Sequence[bool],
    primary_profile: DeviceServiceProfile,
    expert_bytes: int,
    service_p95_us_by_routes: Sequence[float],
    rtt_p95_us: float,
    terminal_bytes: int,
    local_cold_seconds_per_expert: float | None = None,
) -> PeerPlan:
    """Predict peer serving from the exact approved-heatmap route counts.

    The peer tier is disjoint from the primary's predicted HCS residents.  For
    every layer, primary work and peer service overlap, so the critical-path
    contribution is their maximum rather than their sum.  P95 transport and
    service measurements deliberately make this a conservative admission
    model; no advertised device specification participates.
    """
    if total_decode_tokens <= 0:
        raise ValueError("heatmap total_decode_tokens must be positive")
    if len(layer_resident_bytes) != len(layer_is_moe):
        raise ValueError("layer byte and MoE masks differ in length")
    if primary_capacity_experts < 0 or peer_capacity_experts < 0:
        raise ValueError("expert capacities must be non-negative")
    if expert_bytes <= 0 or terminal_bytes < 0:
        raise ValueError("expert and terminal byte geometry is invalid")
    if not math.isfinite(rtt_p95_us) or rtt_p95_us <= 0.0:
        raise ValueError("peer RTT must be finite and positive")
    if local_cold_seconds_per_expert is None:
        local_cold_seconds_per_expert = (
            expert_bytes * primary_profile.h2d_seconds_per_byte
        )
    if (
        not math.isfinite(local_cold_seconds_per_expert)
        or local_cold_seconds_per_expert <= 0.0
    ):
        raise ValueError("local cold-expert service time must be finite and positive")

    ordered = tuple(dict.fromkeys((int(layer), int(expert)) for layer, expert in ranking))
    primary_residents = ordered[:primary_capacity_experts]
    primary_set = set(primary_residents)
    peer_residents = tuple(
        pair for pair in ordered if pair not in primary_set
    )[:peer_capacity_experts]
    peer_set = set(peer_residents)

    layer_total = [0.0] * len(layer_resident_bytes)
    layer_primary = [0.0] * len(layer_resident_bytes)
    layer_peer = [0.0] * len(layer_resident_bytes)
    for key, raw_count in heatmap_counts.items():
        if key == "_metadata":
            continue
        try:
            layer_text, expert_text = key.split(",", 1)
            pair = (int(layer_text), int(expert_text))
            count = int(raw_count)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid heatmap entry {key!r}: {raw_count!r}") from exc
        layer = pair[0]
        if layer < 0 or layer >= len(layer_resident_bytes) or count < 0:
            raise ValueError(f"heatmap entry outside loaded geometry: {key!r}={count}")
        per_token = count / total_decode_tokens
        layer_total[layer] += per_token
        if pair in primary_set:
            layer_primary[layer] += per_token
        elif pair in peer_set:
            layer_peer[layer] += per_token

    predicted_peer = 0.0
    predicted_primary = 0.0
    cold_before = 0.0
    cold_after = 0.0
    captured = 0.0
    for layer, resident_bytes in enumerate(layer_resident_bytes):
        routes = layer_total[layer]
        primary_hot = min(routes, layer_primary[layer])
        peer_routes = min(max(0.0, routes - primary_hot), layer_peer[layer])
        before = max(0.0, routes - primary_hot)
        after = max(0.0, before - peer_routes)
        cold_before += before
        cold_after += after
        captured += peer_routes

        primary_only_compute = resident_bytes + routes * expert_bytes
        predicted_primary += (
            primary_only_compute * primary_profile.d2d_seconds_per_byte
            + before * local_cold_seconds_per_expert
        )

        local_routes = max(0.0, routes - peer_routes)
        local_seconds = (
            (resident_bytes + local_routes * expert_bytes)
            * primary_profile.d2d_seconds_per_byte
            + after * local_cold_seconds_per_expert
        )
        peer_seconds = 0.0
        if peer_routes > 0.0:
            peer_seconds = (
                rtt_p95_us
                + _service_curve_us(peer_routes, service_p95_us_by_routes)
            ) / 1_000_000.0
        predicted_peer += max(local_seconds, peer_seconds)

    predicted_peer += terminal_bytes * primary_profile.d2d_seconds_per_byte
    predicted_primary += terminal_bytes * primary_profile.d2d_seconds_per_byte
    return PeerPlan(
        peer_residents=peer_residents,
        primary_residents=primary_residents,
        captured_routes_per_token=captured,
        cold_routes_before_per_token=cold_before,
        cold_routes_after_per_token=cold_after,
        captured_cold_fraction=(captured / cold_before if cold_before > 0.0 else 0.0),
        predicted_seconds_per_token=predicted_peer,
        predicted_primary_only_seconds_per_token=predicted_primary,
    )
