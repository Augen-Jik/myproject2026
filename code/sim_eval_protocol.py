"""Unified SUMO evaluation protocol definitions and field mappings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from sim_scene_profiles import SimSceneProfile

SUPPORTED_REROUTE_TRIGGER_MODES = ("periodic", "event-driven")
DEFAULT_REROUTE_TRIGGER_MODE = "periodic"
DEFAULT_SIMULATION_SEED_POLICY = "scene_profile_fixed_seed_single_run"
DEFAULT_NUM_EVAL_SEEDS = 1

LEGACY_FIELD_MAPPING = {
    "planning_ms": "planning_time_ms",
    "plan_ms": "planning_time_ms",
    "time_s": "planning_time_s",
    "infer_time": "model_infer_time_s",
    "sumo_time": "travel_time_s",
    "travel_time": "travel_time_s",
    "true_travel_time": "analytical_travel_time_s_legacy",
}

LEGACY_FIELD_NOTES = {
    "planning_ms": "Legacy planning latency field in milliseconds.",
    "plan_ms": "Legacy planning latency field in milliseconds.",
    "time_s": "Legacy seconds field that was often used for inference or planning time, depending on the script.",
    "infer_time": "Model inference-only time; this is a subcomponent of planning time, not travel time.",
    "sumo_time": "SUMO-measured vehicle travel time in seconds.",
    "travel_time": "SUMO-measured vehicle travel time in seconds in the current app pipeline.",
    "true_travel_time": "Legacy analytical proxy from evaluate.py rather than a live SUMO trip duration.",
}


@dataclass(frozen=True)
class SimEvalProtocol:
    planning_time_definition: str
    travel_time_definition: str
    warmup_seconds: int
    evaluation_start_time: int
    evaluation_end_time: int
    reroute_trigger_mode: str
    reroute_period: int
    simulation_seed_policy: str
    num_eval_seeds: int
    vehicle_depart_window: tuple[int, int]
    legacy_field_mapping: dict[str, str] = field(default_factory=lambda: dict(LEGACY_FIELD_MAPPING))
    legacy_field_notes: dict[str, str] = field(default_factory=lambda: dict(LEGACY_FIELD_NOTES))

    def __post_init__(self) -> None:
        if self.reroute_trigger_mode not in SUPPORTED_REROUTE_TRIGGER_MODES:
            raise ValueError(
                f"Unsupported reroute_trigger_mode={self.reroute_trigger_mode}; "
                f"expected one of {SUPPORTED_REROUTE_TRIGGER_MODES}"
            )
        if self.warmup_seconds < 0:
            raise ValueError("warmup_seconds must be >= 0")
        if self.evaluation_start_time < 0:
            raise ValueError("evaluation_start_time must be >= 0")
        if self.evaluation_end_time <= self.evaluation_start_time:
            raise ValueError("evaluation_end_time must be > evaluation_start_time")
        if self.vehicle_depart_window[0] > self.vehicle_depart_window[1]:
            raise ValueError("vehicle_depart_window must be ordered (start <= end)")
        if self.num_eval_seeds <= 0:
            raise ValueError("num_eval_seeds must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_sim_eval_protocol(scene_profile: SimSceneProfile) -> SimEvalProtocol:
    depart_at = max(int(scene_profile.warmup_seconds), int(scene_profile.evaluation_start_time))
    return SimEvalProtocol(
        planning_time_definition=(
            "Planning Time (s) = model inference time + route solving time only. "
            "It excludes SUMO startup, warm-up, simulation stepping, and live vehicle travel."
        ),
        travel_time_definition=(
            "Travel Time (s) = SUMO trip duration for the evaluated vehicle only, "
            "measured from vehicle depart to vehicle arrival inside the simulator."
        ),
        warmup_seconds=int(scene_profile.warmup_seconds),
        evaluation_start_time=int(scene_profile.evaluation_start_time),
        evaluation_end_time=int(scene_profile.evaluation_end_time),
        reroute_trigger_mode=DEFAULT_REROUTE_TRIGGER_MODE,
        reroute_period=int(scene_profile.reroute_period),
        simulation_seed_policy=DEFAULT_SIMULATION_SEED_POLICY,
        num_eval_seeds=DEFAULT_NUM_EVAL_SEEDS,
        vehicle_depart_window=(depart_at, depart_at),
    )


def protocol_to_dict(protocol: SimEvalProtocol) -> dict[str, Any]:
    return protocol.to_dict()


def normalize_eval_time_fields(record: Mapping[str, Any] | None) -> dict[str, Any]:
    rec = dict(record or {})

    planning_time_s = rec.get("planning_time_s")
    if planning_time_s is None:
        planning_time_ms = rec.get("planning_time_ms", rec.get("planning_ms", rec.get("plan_ms")))
        if planning_time_ms is not None:
            planning_time_s = float(planning_time_ms) / 1000.0
        elif rec.get("time_s") is not None:
            planning_time_s = float(rec.get("time_s"))
    planning_time_ms = rec.get("planning_time_ms")
    if planning_time_ms is None and planning_time_s is not None:
        planning_time_ms = float(planning_time_s) * 1000.0

    model_infer_time_s = rec.get("model_infer_time_s", rec.get("infer_time"))
    route_solve_time_s = rec.get("route_solve_time_s")

    travel_time_s = rec.get("travel_time_s")
    if travel_time_s is None:
        travel_time_s = rec.get("travel_time", rec.get("sumo_time"))
    travel_time_ms = rec.get("travel_time_ms")
    if travel_time_ms is None and travel_time_s is not None:
        travel_time_ms = float(travel_time_s) * 1000.0

    analytical_travel_time_s_legacy = rec.get("analytical_travel_time_s_legacy", rec.get("true_travel_time"))

    return {
        "planning_time_s": None if planning_time_s is None else float(planning_time_s),
        "planning_time_ms": None if planning_time_ms is None else float(planning_time_ms),
        "model_infer_time_s": None if model_infer_time_s is None else float(model_infer_time_s),
        "route_solve_time_s": None if route_solve_time_s is None else float(route_solve_time_s),
        "travel_time_s": None if travel_time_s is None else float(travel_time_s),
        "travel_time_ms": None if travel_time_ms is None else float(travel_time_ms),
        "analytical_travel_time_s_legacy": (
            None if analytical_travel_time_s_legacy is None else float(analytical_travel_time_s_legacy)
        ),
    }


def build_protocol_snapshot(scene_profile: SimSceneProfile) -> dict[str, Any]:
    return protocol_to_dict(build_sim_eval_protocol(scene_profile))
