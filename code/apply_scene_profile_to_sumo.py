"""Apply exogenous scene profiles to a running SUMO simulation."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import logging
import math
import os
import re
from typing import Any, Iterable

import sumolib
import traci
import traci.constants as tc

from congestion_propagation import resolve_spillover_runtime_config
from sim_scene_profiles import (
    DEFAULT_CONGESTION_LEVEL_TO_WEIGHT,
    SimSceneProfile,
    build_environment_weight_layers,
    build_sumo_runtime_options,
    get_scene_profile,
    write_scene_route_file,
)

LOGGER = logging.getLogger(__name__)

CANONICAL_CONGESTION_SPEED_FACTORS = {
    "normal": 1.00,
    "relief": 1.08,
    "mild": 0.82,
    "moderate": 0.62,
    "heavy": 0.45,
    "severe": 0.28,
    "blocked": 0.04,
    "lane_reduction": 0.72,
}

CONGESTION_LABEL_ALIASES = {
    "normal": "normal",
    "freeflow": "relief",
    "relief": "relief",
    "clear": "relief",
    "mild": "mild",
    "light": "mild",
    "minor": "mild",
    "moderate": "moderate",
    "medium": "moderate",
    "heavy": "heavy",
    "major": "heavy",
    "severe": "severe",
    "critical": "severe",
    "blocked": "blocked",
    "closure": "blocked",
    "closed": "blocked",
    "lane_reduction": "lane_reduction",
}

ALL_DISALLOWED_VCLASSES = (
    "private",
    "passenger",
    "bus",
    "coach",
    "delivery",
    "truck",
    "trailer",
    "taxi",
    "emergency",
    "authority",
    "army",
    "vip",
    "hov",
    "motorcycle",
    "moped",
    "bicycle",
    "pedestrian",
    "evehicle",
    "custom1",
    "custom2",
)

DEFAULT_MIN_SPEED_MPS = 0.35
DEFAULT_BLOCKED_TRAVELTIME_SECONDS = 36000.0
DEFAULT_LANE_REDUCTION_TRAVELTIME_MULTIPLIER = 1.8


@dataclass(frozen=True)
class TLSProfileSpec:
    profile_name: str
    fixed_program_id: str = "0"
    fixed_phase_index: int | None = None
    fixed_state_mode: str | None = None


TLS_PROFILE_REGISTRY = {
    "static_all_intersections": TLSProfileSpec(
        profile_name="static_all_intersections",
        fixed_program_id="0",
    ),
    "static_program_0": TLSProfileSpec(
        profile_name="static_program_0",
        fixed_program_id="0",
    ),
    "static_program_0_phase_0": TLSProfileSpec(
        profile_name="static_program_0_phase_0",
        fixed_program_id="0",
        fixed_phase_index=0,
    ),
    "static_program_0_phase_2": TLSProfileSpec(
        profile_name="static_program_0_phase_2",
        fixed_program_id="0",
        fixed_phase_index=2,
    ),
    "all_red_lock": TLSProfileSpec(
        profile_name="all_red_lock",
        fixed_program_id="0",
        fixed_state_mode="all_red",
    ),
}


@dataclass
class LaneSnapshot:
    lane_id: str
    max_speed: float
    allowed: tuple[str, ...]
    disallowed: tuple[str, ...]
    length: float


@dataclass
class TLSSnapshot:
    tls_id: str
    program_id: str
    phase_index: int
    state: str


def normalize_congestion_label(level: str | None) -> str:
    key = str(level or "").strip().lower()
    return CONGESTION_LABEL_ALIASES.get(key, "normal")


def resolve_speed_factor_map(profile: SimSceneProfile) -> dict[str, float]:
    factors = dict(CANONICAL_CONGESTION_SPEED_FACTORS)
    for raw_key, value in dict(profile.congestion_level_to_speed_factor or {}).items():
        normalized = normalize_congestion_label(raw_key)
        try:
            factors[normalized] = float(value)
        except Exception:
            continue
    # Preserve explicit aliases for downstream lookups.
    for raw_key, normalized in CONGESTION_LABEL_ALIASES.items():
        factors[raw_key] = factors.get(normalized, CANONICAL_CONGESTION_SPEED_FACTORS["normal"])
    return factors


def scene_runtime_route_path(results_dir: str, scene_profile: SimSceneProfile) -> str:
    runtime_dir = os.path.join(results_dir, "scene_runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    safe_name = re.sub(r"[^a-z0-9._-]+", "_", scene_profile.scene_name.lower()).strip("_")
    if not safe_name:
        safe_name = "scene"
    return os.path.join(runtime_dir, f"{safe_name}__seed{int(scene_profile.simulation_seed)}.rou.xml")


def build_scene_reroute_options(profile: SimSceneProfile) -> list[str]:
    if not profile.reroute_enabled:
        LOGGER.info(
            "Reroute effective value: disabled for scene=%s reason=fixed route protocol",
            profile.scene_name,
        )
        return [
            "--device.rerouting.probability",
            "0",
        ]
    LOGGER.info(
        "Reroute effective value: enabled for scene=%s period=%ss threshold=%.2fx+%.1fs",
        profile.scene_name,
        int(profile.reroute_period),
        float(profile.reroute_threshold_factor),
        float(profile.reroute_threshold_constant),
    )
    return [
        "--device.rerouting.probability",
        "1",
        "--device.rerouting.deterministic",
        "--device.rerouting.period",
        str(int(profile.reroute_period)),
        "--device.rerouting.pre-period",
        str(int(profile.reroute_period)),
        "--device.rerouting.adaptation-weight",
        "1.0",
        "--device.rerouting.adaptation-interval",
        str(int(profile.reroute_period)),
    ]


def build_scene_sumo_command(
    *,
    config: dict[str, Any],
    scene_profile: SimSceneProfile,
    route_file_path: str,
    tripinfo_path: str,
    step_length: float,
) -> list[str]:
    cmd = [
        config.get("SUMO_BINARY", "sumo"),
        "-c",
        config.get("SUMO_CONFIG_PATH"),
    ]
    cmd.extend(
        build_sumo_runtime_options(
            scene_profile,
            route_file_path=route_file_path,
            tripinfo_path=tripinfo_path,
            step_length=step_length,
        )
    )
    cmd.extend(build_scene_reroute_options(scene_profile))
    return cmd


def _lane_ids_for_edge(edge_id: str) -> list[str]:
    lane_count = int(traci.edge.getLaneNumber(edge_id))
    return [f"{edge_id}_{idx}" for idx in range(lane_count)]


def _canonical_weight_speed_points() -> list[tuple[float, float]]:
    return [
        (float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["normal"]), float(CANONICAL_CONGESTION_SPEED_FACTORS["normal"])),
        (float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["light"]), float(CANONICAL_CONGESTION_SPEED_FACTORS["mild"])),
        (float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["moderate"]), float(CANONICAL_CONGESTION_SPEED_FACTORS["moderate"])),
        (float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["heavy"]), float(CANONICAL_CONGESTION_SPEED_FACTORS["heavy"])),
        (float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["severe"]), float(CANONICAL_CONGESTION_SPEED_FACTORS["severe"])),
        (float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["blocked"]), float(CANONICAL_CONGESTION_SPEED_FACTORS["blocked"])),
    ]


def runtime_weight_to_speed_factor(weight: float) -> float:
    value = float(weight)
    points = _canonical_weight_speed_points()
    if value <= points[0][0]:
        return points[0][1]
    for (left_weight, left_factor), (right_weight, right_factor) in zip(points, points[1:]):
        if value <= right_weight:
            span = max(right_weight - left_weight, 1e-9)
            ratio = (value - left_weight) / span
            return left_factor + ((right_factor - left_factor) * ratio)
    return points[-1][1]


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _normal_weight_multiplier(weight: float) -> float:
    base = float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT.get("normal", 2.0))
    return float(weight) / max(base, 1e-9)


def _profile_direct_affected_edges(profile: SimSceneProfile) -> set[str]:
    return (
        set(dict(profile.incident_edges or {}).keys())
        | set(tuple(profile.blocked_edges or ()))
        | set(dict(profile.lane_reduction_edges or {}).keys())
    )


def print_scene_injection_summary(scene_name: str) -> dict[str, Any]:
    """Dry-run the exogenous scene injection and print route overlap diagnostics."""
    from config import ConfigManager
    from demo_scene_selector import select_showcase_scenarios
    from roadnet_meta import live_edge_ids

    config = ConfigManager().config
    requested_name = str(scene_name or "").strip()
    showcase_cases = {}
    try:
        showcase_cases = select_showcase_scenarios(config.get("SUMO_NET_PATH"))
    except Exception as exc:
        LOGGER.warning("Could not load showcase cases for dry-run: %s", exc)

    showcase_case = dict(showcase_cases.get(requested_name) or {})
    profile_name = str(showcase_case.get("scene_profile") or requested_name)
    profile = get_scene_profile(profile_name)
    edge_ids = tuple(live_edge_ids())
    enable_spillover = bool(showcase_case or requested_name.endswith("_showcase") or requested_name.endswith("_conservative"))
    layers = build_environment_weight_layers(edge_ids, profile, enable_spillover=enable_spillover)
    spillover_weights = dict(layers.get("spillover_weights") or {})
    spillover_hops = dict(layers.get("spillover_hop_by_edge") or {})
    affected_direct = _profile_direct_affected_edges(profile)
    affected_all = affected_direct | set(spillover_weights)
    baseline_path = list(showcase_case.get("baseline_path") or [])

    multipliers: dict[str, dict[str, Any]] = {}
    for edge_id, level in dict(profile.incident_edges or {}).items():
        level_key = normalize_congestion_label(level)
        speed_factor = float(resolve_speed_factor_map(profile).get(level_key, 1.0))
        weight = float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT.get(level_key, DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["moderate"]))
        multipliers[edge_id] = {
            "source": "incident",
            "level": level_key,
            "weight_multiplier_vs_normal": _normal_weight_multiplier(weight),
            "travel_time_multiplier_approx": 1.0 / max(speed_factor, 1e-9),
        }
    for edge_id in tuple(profile.blocked_edges or ()):
        weight = float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["blocked"])
        multipliers[edge_id] = {
            "source": "blocked",
            "level": "blocked",
            "weight_multiplier_vs_normal": _normal_weight_multiplier(weight),
            "travel_time_multiplier_approx": DEFAULT_BLOCKED_TRAVELTIME_SECONDS,
        }
    for edge_id, remaining_lanes in dict(profile.lane_reduction_edges or {}).items():
        speed_factor = float(resolve_speed_factor_map(profile).get("lane_reduction", 1.0))
        weight = float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["lane_reduction"])
        if int(remaining_lanes) <= 1:
            weight = max(weight, float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["heavy"]))
        entry = multipliers.setdefault(edge_id, {})
        entry.update(
            {
                "source": "lane_reduction" if not entry else f"{entry.get('source')}+lane_reduction",
                "remaining_lanes": int(remaining_lanes),
                "lane_reduction_weight_multiplier_vs_normal": _normal_weight_multiplier(weight),
                "lane_reduction_travel_time_multiplier_approx": (
                    1.0 / max(speed_factor, 1e-9)
                )
                * DEFAULT_LANE_REDUCTION_TRAVELTIME_MULTIPLIER,
            }
        )
    runtime_cfg = resolve_spillover_runtime_config(profile.scene_name, profile.scene_type)
    hop_tt_multipliers = dict(runtime_cfg.get("hop_travel_time_multiplier") or {})
    for edge_id, weight in spillover_weights.items():
        hop = int(spillover_hops.get(edge_id, 0) or 0)
        multipliers[edge_id] = {
            "source": "spillover",
            "hop": hop,
            "weight_multiplier_vs_normal": _normal_weight_multiplier(float(weight)),
            "runtime_travel_time_multiplier": float(hop_tt_multipliers.get(hop, 1.0)),
        }

    baseline_edge_effects = [
        {
            "edge_id": edge_id,
            "affected": edge_id in affected_all,
            "source": multipliers.get(edge_id, {}).get("source", "none"),
            "multiplier": multipliers.get(edge_id),
        }
        for edge_id in baseline_path
    ]
    summary = {
        "requested_scene": requested_name,
        "profile": profile.scene_name,
        "scene_type": profile.scene_type,
        "incident_edges": dict(profile.incident_edges or {}),
        "blocked_edges": list(profile.blocked_edges or ()),
        "lane_reduction_edges": dict(profile.lane_reduction_edges or {}),
        "spillover_edges": sorted(spillover_weights),
        "edge_multipliers": {edge_id: multipliers[edge_id] for edge_id in sorted(multipliers)},
        "baseline_path_edges": baseline_path,
        "baseline_path_edge_effects": baseline_edge_effects,
        "reroute_period": int(profile.reroute_period),
        "reroute_enabled": bool(profile.reroute_enabled),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


class SUMOSceneProfileApplier:
    """Owns scene-driven SUMO environment injection for one simulation run."""

    def __init__(
        self,
        config: dict[str, Any],
        scene_profile: SimSceneProfile,
        *,
        runtime_spillover_payload: dict[str, Any] | None = None,
        scene_mode: str = "benchmark",
    ):
        self.config = config
        self.scene_profile = scene_profile
        self.scene_mode = str(scene_mode or "benchmark")
        self.speed_factor_map = resolve_speed_factor_map(scene_profile)
        self.runtime_route_path = scene_runtime_route_path(
            config.get("RESULTS_DIR", "/root/autodl-tmp/results"),
            scene_profile,
        )
        self.scene_effects_active = False
        self._last_reroute_time: float | None = None
        self._lane_snapshots: dict[str, LaneSnapshot] = {}
        self._tls_snapshots: dict[str, TLSSnapshot] = {}
        self._planner_vehicle_ids: set[str] = set()
        self._tls_spec = TLS_PROFILE_REGISTRY.get(
            scene_profile.tls_profile_name,
            TLS_PROFILE_REGISTRY["static_all_intersections"],
        )
        self._net = None
        self._net_edges_by_id: dict[str, Any] = {}
        runtime_spillover_payload = dict(runtime_spillover_payload or {})
        self._runtime_raw_weights = dict(runtime_spillover_payload.get("raw_only_weights") or {})
        self._runtime_spillover_weights = dict(runtime_spillover_payload.get("spillover_weights") or {})
        self._runtime_spillover_hops = {
            str(edge_id): int(hop)
            for edge_id, hop in dict(runtime_spillover_payload.get("spillover_hop_by_edge") or {}).items()
        }
        self._runtime_spillover_enabled = bool(
            self.scene_mode == "showcase" and self._runtime_spillover_weights
        )
        self._runtime_spillover_config = resolve_spillover_runtime_config(
            self.scene_profile.scene_name,
            self.scene_profile.scene_type,
        )
        self._affected_edges = sorted(
            set(scene_profile.incident_edges.keys())
            | set(scene_profile.blocked_edges)
            | set(scene_profile.lane_reduction_edges.keys())
            | (set(self._runtime_spillover_weights.keys()) if self._runtime_spillover_enabled else set())
        )
        self._load_network_if_available()

    def runtime_spillover_summary(self) -> dict[str, Any]:
        hop_1_count = sum(1 for hop in self._runtime_spillover_hops.values() if int(hop) == 1)
        hop_2_count = sum(1 for hop in self._runtime_spillover_hops.values() if int(hop) == 2)
        return {
            "scene_mode": self.scene_mode,
            "runtime_spillover_enabled": bool(self._runtime_spillover_enabled),
            "runtime_spillover_edge_count": len(self._runtime_spillover_weights),
            "runtime_spillover_1hop_count": hop_1_count,
            "runtime_spillover_2hop_count": hop_2_count,
            "runtime_spillover_config": dict(self._runtime_spillover_config or {}),
            "runtime_spillover_weight_range": (
                {
                    "min": min(float(weight) for weight in self._runtime_spillover_weights.values()),
                    "max": max(float(weight) for weight in self._runtime_spillover_weights.values()),
                }
                if self._runtime_spillover_weights
                else None
            ),
        }

    def prepare_runtime_route_file(self) -> str:
        return write_scene_route_file(self.runtime_route_path, self.scene_profile)

    def build_sumo_command(self, *, tripinfo_path: str, step_length: float) -> list[str]:
        self.prepare_runtime_route_file()
        cmd = build_scene_sumo_command(
            config=self.config,
            scene_profile=self.scene_profile,
            route_file_path=self.runtime_route_path,
            tripinfo_path=tripinfo_path,
            step_length=step_length,
        )
        LOGGER.info(
            "SUMO command prepared scene=%s reroute_enabled=%s route_file=%s",
            self.scene_profile.scene_name,
            bool(self.scene_profile.reroute_enabled),
            self.runtime_route_path,
        )
        return cmd

    def capture_base_state(self) -> None:
        self._lane_snapshots.clear()
        self._tls_snapshots.clear()
        for edge_id in self._affected_edges:
            try:
                for lane_id in _lane_ids_for_edge(edge_id):
                    self._lane_snapshots[lane_id] = LaneSnapshot(
                        lane_id=lane_id,
                        max_speed=float(traci.lane.getMaxSpeed(lane_id)),
                        allowed=tuple(traci.lane.getAllowed(lane_id)),
                        disallowed=tuple(traci.lane.getDisallowed(lane_id)),
                        length=float(traci.lane.getLength(lane_id)),
                    )
            except Exception as exc:
                LOGGER.warning("Failed to capture lane state for %s: %s", edge_id, exc)
        for tls_id in traci.trafficlight.getIDList():
            try:
                self._tls_snapshots[tls_id] = TLSSnapshot(
                    tls_id=tls_id,
                    program_id=str(traci.trafficlight.getProgram(tls_id)),
                    phase_index=int(traci.trafficlight.getPhase(tls_id)),
                    state=str(traci.trafficlight.getRedYellowGreenState(tls_id)),
                )
            except Exception as exc:
                LOGGER.warning("Failed to capture TLS state for %s: %s", tls_id, exc)

    def bootstrap_after_start(self) -> None:
        self.capture_base_state()
        self.apply_tls_profile(force=True)

    def register_planner_vehicle(self, vehicle_id: str) -> None:
        self._planner_vehicle_ids.add(vehicle_id)

    def apply_tls_profile(self, *, force: bool = False) -> None:
        for tls_id in traci.trafficlight.getIDList():
            try:
                program_id = str(self.scene_profile.tls_fixed_program_id or self._tls_spec.fixed_program_id)
                if force or self._tls_spec.fixed_program_id is not None:
                    traci.trafficlight.setProgram(tls_id, program_id)
                if self._tls_spec.fixed_phase_index is not None:
                    current_phase = int(traci.trafficlight.getPhase(tls_id))
                    if force or current_phase != self._tls_spec.fixed_phase_index:
                        traci.trafficlight.setPhase(tls_id, self._tls_spec.fixed_phase_index)
                if self._tls_spec.fixed_state_mode is not None:
                    base_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                    target_state = self._build_fixed_tls_state(base_state, self._tls_spec.fixed_state_mode)
                    if force or target_state != base_state:
                        traci.trafficlight.setRedYellowGreenState(tls_id, target_state)
            except Exception as exc:
                LOGGER.warning("Failed to apply TLS profile %s to %s: %s", self._tls_spec.profile_name, tls_id, exc)

    def restore_base_state(self) -> None:
        self._restore_lane_state()
        for tls_id, snapshot in self._tls_snapshots.items():
            try:
                traci.trafficlight.setProgram(tls_id, snapshot.program_id)
                traci.trafficlight.setPhase(tls_id, snapshot.phase_index)
                traci.trafficlight.setRedYellowGreenState(tls_id, snapshot.state)
            except Exception:
                pass

    def on_simulation_step(self) -> None:
        current_time = float(traci.simulation.getTime())
        if self.scene_profile.evaluation_start_time <= current_time <= self.scene_profile.evaluation_end_time:
            if not self.scene_effects_active:
                self._apply_window_effects()
                self.scene_effects_active = True
            if self._tls_spec.fixed_phase_index is not None or self._tls_spec.fixed_state_mode is not None:
                self.apply_tls_profile(force=False)
            self._maybe_run_periodic_reroute(current_time)
        elif self.scene_effects_active:
            self._restore_lane_state()
            self.scene_effects_active = False

    def _apply_window_effects(self) -> None:
        for edge_id, severity in dict(self.scene_profile.incident_edges or {}).items():
            self._apply_edge_speed_factor(edge_id, normalize_congestion_label(severity))
        for edge_id, reduction_spec in dict(self.scene_profile.lane_reduction_edges or {}).items():
            self._apply_lane_reduction(edge_id, reduction_spec)
        for edge_id in tuple(self.scene_profile.blocked_edges or ()):
            self._apply_blocked_edge(edge_id)
        if self._runtime_spillover_enabled:
            self._apply_runtime_spillover_effects()

    def _apply_runtime_spillover_effects(self) -> None:
        direct_affected = set(self.scene_profile.incident_edges.keys()) | set(self.scene_profile.blocked_edges) | set(
            self.scene_profile.lane_reduction_edges.keys()
        )
        for edge_id, weight in dict(self._runtime_spillover_weights or {}).items():
            if edge_id in direct_affected:
                continue
            hop = int(self._runtime_spillover_hops.get(edge_id, 2) or 2)
            self._apply_runtime_spillover_edge(edge_id, float(weight), hop)

    def _apply_runtime_spillover_edge(self, edge_id: str, runtime_weight: float, hop: int) -> None:
        min_speed = float(self.scene_profile.speed_profile.get("minimum_edge_speed_mps", DEFAULT_MIN_SPEED_MPS))
        raw_speed_factor = max(
            runtime_weight_to_speed_factor(runtime_weight),
            float(CANONICAL_CONGESTION_SPEED_FACTORS["blocked"]),
        )
        hop_bounds = dict(self._runtime_spillover_config.get("hop_speed_factor_bounds") or {}).get(int(hop))
        if isinstance(hop_bounds, (list, tuple)) and len(hop_bounds) >= 2:
            lower_bound = min(float(hop_bounds[0]), float(hop_bounds[1]))
            upper_bound = max(float(hop_bounds[0]), float(hop_bounds[1]))
            speed_factor = _clamp(raw_speed_factor, lower_bound, upper_bound)
        else:
            speed_factor = raw_speed_factor
        lane_ids = _lane_ids_for_edge(edge_id)
        edge_travel_time = 0.0
        for lane_id in lane_ids:
            snapshot = self._lane_snapshots.get(lane_id)
            if snapshot is None:
                continue
            lane_speed = max(min_speed, snapshot.max_speed * speed_factor)
            traci.lane.setMaxSpeed(lane_id, lane_speed)
            edge_travel_time = max(edge_travel_time, snapshot.length / max(lane_speed, min_speed))
        if edge_travel_time > 0.0:
            multiplier = float(
                dict(self._runtime_spillover_config.get("hop_travel_time_multiplier") or {}).get(int(hop), 1.0)
            )
            adjusted_tt = edge_travel_time * multiplier
            traci.edge.adaptTraveltime(
                edge_id,
                adjusted_tt,
                begin=float(self.scene_profile.evaluation_start_time),
                end=float(self.scene_profile.evaluation_end_time),
            )
            traci.edge.setEffort(
                edge_id,
                adjusted_tt,
                begin=float(self.scene_profile.evaluation_start_time),
                end=float(self.scene_profile.evaluation_end_time),
            )

    def _apply_edge_speed_factor(self, edge_id: str, level: str) -> None:
        factor = float(self.speed_factor_map.get(level, self.speed_factor_map["normal"]))
        min_speed = float(self.scene_profile.speed_profile.get("minimum_edge_speed_mps", DEFAULT_MIN_SPEED_MPS))
        edge_travel_time = 0.0
        lane_ids = _lane_ids_for_edge(edge_id)
        for lane_id in lane_ids:
            snapshot = self._lane_snapshots.get(lane_id)
            if snapshot is None:
                continue
            lane_speed = max(min_speed, snapshot.max_speed * factor)
            traci.lane.setMaxSpeed(lane_id, lane_speed)
            edge_travel_time = max(edge_travel_time, snapshot.length / max(lane_speed, min_speed))
        if edge_travel_time > 0:
            traci.edge.adaptTraveltime(
                edge_id,
                edge_travel_time,
                begin=float(self.scene_profile.evaluation_start_time),
                end=float(self.scene_profile.evaluation_end_time),
            )
            traci.edge.setEffort(
                edge_id,
                edge_travel_time,
                begin=float(self.scene_profile.evaluation_start_time),
                end=float(self.scene_profile.evaluation_end_time),
            )

    def _apply_lane_reduction(self, edge_id: str, reduction_spec: Any) -> None:
        if isinstance(reduction_spec, dict):
            remaining_lanes = int(reduction_spec.get("remaining_lanes", reduction_spec.get("lanes_left", 1)))
        else:
            remaining_lanes = int(reduction_spec)
        lane_ids = _lane_ids_for_edge(edge_id)
        active_count = max(1, min(len(lane_ids), remaining_lanes))
        self._apply_edge_speed_factor(edge_id, "lane_reduction")
        for lane_id in lane_ids[active_count:]:
            self._close_lane(lane_id)
        if lane_ids:
            representative = self._lane_snapshots.get(lane_ids[0])
            if representative is not None:
                reduced_speed = max(
                    float(self.scene_profile.speed_profile.get("minimum_edge_speed_mps", DEFAULT_MIN_SPEED_MPS)),
                    representative.max_speed * float(self.speed_factor_map["lane_reduction"]),
                )
                reduced_tt = (representative.length / max(reduced_speed, DEFAULT_MIN_SPEED_MPS)) * DEFAULT_LANE_REDUCTION_TRAVELTIME_MULTIPLIER
                traci.edge.adaptTraveltime(
                    edge_id,
                    reduced_tt,
                    begin=float(self.scene_profile.evaluation_start_time),
                    end=float(self.scene_profile.evaluation_end_time),
                )
                traci.edge.setEffort(
                    edge_id,
                    reduced_tt,
                    begin=float(self.scene_profile.evaluation_start_time),
                    end=float(self.scene_profile.evaluation_end_time),
                )

    def _apply_blocked_edge(self, edge_id: str) -> None:
        min_speed = float(self.scene_profile.speed_profile.get("minimum_edge_speed_mps", DEFAULT_MIN_SPEED_MPS))
        for lane_id in _lane_ids_for_edge(edge_id):
            snapshot = self._lane_snapshots.get(lane_id)
            if snapshot is None:
                continue
            traci.lane.setMaxSpeed(lane_id, min_speed)
            self._close_lane(lane_id)
        traci.edge.adaptTraveltime(
            edge_id,
            DEFAULT_BLOCKED_TRAVELTIME_SECONDS,
            begin=float(self.scene_profile.evaluation_start_time),
            end=float(self.scene_profile.evaluation_end_time),
        )
        traci.edge.setEffort(
            edge_id,
            DEFAULT_BLOCKED_TRAVELTIME_SECONDS,
            begin=float(self.scene_profile.evaluation_start_time),
            end=float(self.scene_profile.evaluation_end_time),
        )

    def _close_lane(self, lane_id: str) -> None:
        try:
            traci.lane.setDisallowed(lane_id, ALL_DISALLOWED_VCLASSES)
        except Exception:
            pass

    def _restore_lane_state(self) -> None:
        for snapshot in self._lane_snapshots.values():
            try:
                traci.lane.setMaxSpeed(snapshot.lane_id, snapshot.max_speed)
                traci.lane.setDisallowed(snapshot.lane_id, snapshot.disallowed)
                if snapshot.allowed:
                    traci.lane.setAllowed(snapshot.lane_id, snapshot.allowed)
                elif snapshot.disallowed:
                    traci.lane.setDisallowed(snapshot.lane_id, snapshot.disallowed)
                else:
                    traci.lane.setDisallowed(snapshot.lane_id, ())
            except Exception:
                pass

    def _maybe_run_periodic_reroute(self, current_time: float) -> None:
        if not self.scene_profile.reroute_enabled:
            return
        if self._last_reroute_time is not None and current_time - self._last_reroute_time < float(self.scene_profile.reroute_period):
            return
        self._last_reroute_time = current_time
        for veh_id in tuple(traci.vehicle.getIDList()):
            self._maybe_reroute_vehicle(veh_id, current_time)

    def _maybe_reroute_vehicle(self, veh_id: str, current_time: float) -> None:
        try:
            route = tuple(traci.vehicle.getRoute(veh_id))
            if len(route) < 2:
                return
            route_index = int(traci.vehicle.getRouteIndex(veh_id))
            if route_index < 0 or route_index >= len(route):
                return
            current_edge = self._resolve_current_edge_for_reroute(veh_id, route, route_index)
            target_edge = self._resolve_target_edge(route)
            if not current_edge or not target_edge or current_edge.startswith(":") or target_edge.startswith(":"):
                return
            current_remaining_tt = self._estimate_remaining_route_travel_time(route, route_index)
            alt_edges, alt_tt = self._compute_best_route(current_edge, target_edge)
            if not alt_edges or alt_tt <= 0:
                return
            threshold = (alt_tt * float(self.scene_profile.reroute_threshold_factor)) + float(self.scene_profile.reroute_threshold_constant)
            if current_remaining_tt > threshold:
                traci.vehicle.setRoute(veh_id, alt_edges)
                traci.vehicle.setRoutingMode(veh_id, tc.ROUTING_MODE_AGGREGATED_CUSTOM)
        except Exception as exc:
            LOGGER.debug("Skip reroute for %s: %s", veh_id, exc)

    def _resolve_current_edge_for_reroute(self, veh_id: str, route: tuple[str, ...], route_index: int) -> str:
        road_id = str(traci.vehicle.getRoadID(veh_id))
        if road_id and not road_id.startswith(":"):
            return road_id
        for edge_id in route[route_index:]:
            if edge_id and not edge_id.startswith(":"):
                return edge_id
        return ""

    def _resolve_target_edge(self, route: tuple[str, ...]) -> str:
        for edge_id in reversed(route):
            if edge_id and not edge_id.startswith(":"):
                return edge_id
        return ""

    def _estimate_remaining_route_travel_time(self, route: tuple[str, ...], route_index: int) -> float:
        total = 0.0
        min_speed = float(self.scene_profile.speed_profile.get("minimum_edge_speed_mps", DEFAULT_MIN_SPEED_MPS))
        for edge_id in route[route_index:]:
            if not edge_id or edge_id.startswith(":"):
                continue
            try:
                total += float(traci.edge.getTraveltime(edge_id))
                continue
            except Exception:
                pass
            lane_id = f"{edge_id}_0"
            snapshot = self._lane_snapshots.get(lane_id)
            if snapshot is not None:
                total += snapshot.length / max(snapshot.max_speed, min_speed)
        return total

    def _build_fixed_tls_state(self, reference_state: str, state_mode: str) -> str:
        length = max(1, len(reference_state))
        mode = str(state_mode or "").strip().lower()
        if mode == "all_red":
            return "r" * length
        if mode == "all_green":
            return "G" * length
        if mode == "all_yellow":
            return "y" * length
        return reference_state

    def _load_network_if_available(self) -> None:
        net_path = self.config.get("SUMO_NET_PATH", "/root/autodl-tmp/SUMO/net/my_net.net.xml")
        try:
            if net_path and os.path.exists(net_path):
                self._net = sumolib.net.readNet(net_path)
                self._net_edges_by_id = {
                    edge.getID(): edge
                    for edge in self._net.getEdges()
                    if edge is not None and edge.getID() and not edge.getID().startswith(":")
                }
        except Exception as exc:
            LOGGER.warning("Failed to load SUMO net for custom reroute: %s", exc)
            self._net = None
            self._net_edges_by_id = {}

    def _compute_best_route(self, start_edge_id: str, target_edge_id: str) -> tuple[tuple[str, ...], float]:
        if self._net is None:
            return (), math.inf
        if start_edge_id == target_edge_id:
            return (start_edge_id,), self._edge_travel_time(start_edge_id)
        if start_edge_id not in self._net_edges_by_id or target_edge_id not in self._net_edges_by_id:
            return (), math.inf

        counter = 0
        start_cost = self._edge_travel_time(start_edge_id)
        frontier: list[tuple[float, int, str, tuple[str, ...]]] = [
            (start_cost, counter, start_edge_id, (start_edge_id,))
        ]
        best_cost = {start_edge_id: start_cost}

        while frontier:
            path_cost, _, edge_id, path = heapq.heappop(frontier)
            if path_cost > best_cost.get(edge_id, math.inf):
                continue
            if edge_id == target_edge_id:
                return path, path_cost

            edge_obj = self._net_edges_by_id.get(edge_id)
            if edge_obj is None:
                continue
            for next_edge in edge_obj.getOutgoing().keys():
                next_edge_id = next_edge.getID()
                if not next_edge_id or next_edge_id.startswith(":"):
                    continue
                next_cost = path_cost + self._edge_travel_time(next_edge_id)
                if next_cost >= best_cost.get(next_edge_id, math.inf):
                    continue
                best_cost[next_edge_id] = next_cost
                counter += 1
                heapq.heappush(frontier, (next_cost, counter, next_edge_id, path + (next_edge_id,)))

        return (), math.inf

    def _edge_travel_time(self, edge_id: str) -> float:
        min_speed = float(self.scene_profile.speed_profile.get("minimum_edge_speed_mps", DEFAULT_MIN_SPEED_MPS))
        try:
            return max(0.01, float(traci.edge.getTraveltime(edge_id)))
        except Exception:
            snapshot = self._lane_snapshots.get(f"{edge_id}_0")
            if snapshot is not None:
                return snapshot.length / max(snapshot.max_speed, min_speed)
        return DEFAULT_BLOCKED_TRAVELTIME_SECONDS
