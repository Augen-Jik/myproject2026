"""Unified exogenous SUMO scene profiles for reproducible evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from congestion_propagation import build_spillover_weights

MODEL_CONTROLLED_PARAMETERS = (
    "edge_cost_estimates",
    "path_selection",
)

EXOGENOUS_ENVIRONMENT_PARAMETERS = (
    "speed_profile",
    "congestion_level_to_speed_factor",
    "incident_edges",
    "blocked_edges",
    "lane_reduction_edges",
    "tls_profile_name",
    "tls_fixed_program_id",
    "reroute_enabled",
    "reroute_period",
    "reroute_threshold_factor",
    "reroute_threshold_constant",
    "demand_profile_name",
    "depart_rate",
    "simulation_seed",
    "warmup_seconds",
    "evaluation_start_time",
    "evaluation_end_time",
)

DEFAULT_CONGESTION_LEVEL_TO_SPEED_FACTOR = {
    "normal": 1.00,
    "relief": 1.08,
    "mild": 0.82,
    "light": 0.82,
    "moderate": 0.62,
    "heavy": 0.45,
    "severe": 0.28,
    "blocked": 0.04,
    "lane_reduction": 0.72,
}

DEFAULT_CONGESTION_LEVEL_TO_WEIGHT = {
    "normal": 2.0,
    "light": 3.0,
    "moderate": 5.0,
    "heavy": 6.2,
    "severe": 7.5,
    "blocked": 9.5,
    "lane_reduction": 4.2,
}

DEFAULT_SPEED_PROFILE = {
    "base_speed_factor": 1.0,
    "blocked_speed_factor": 0.04,
    "lane_reduction_speed_factor": 0.72,
    "minimum_edge_speed_mps": 0.35,
}

BASE_ROUTE_LIBRARY = (
    ("Nongye_EW", ("R0C0_E", "R0C1_E", "R0C2_E", "R0C3_E", "R0C4_E")),
    ("Nongye_WE", ("R0C4_W", "R0C3_W", "R0C2_W", "R0C1_W", "R0C0_W")),
    ("Huanghe_EW", ("R3C0_E", "R3C1_E", "R3C2_E", "R3C3_E", "R3C4_E")),
    ("Huanghe_WE", ("R3C4_W", "R3C3_W", "R3C2_W", "R3C1_W", "R3C0_W")),
    ("Hongzhuan_EW", ("R1C0_E", "R1C1_E", "R1C2_E", "R1C3_E", "R1C4_E")),
    ("Hongzhuan_WE", ("R1C4_W", "R1C3_W", "R1C2_W", "R1C1_W", "R1C0_W")),
    ("Huayuan_NS", ("C4R0_N", "C4R1_N", "C4R2_N", "C4R3_N")),
    ("Huayuan_SN", ("C4R3_S", "C4R2_S", "C4R1_S", "C4R0_S")),
    ("Jing6_NS", ("C2R0_N", "C2R1_N", "C2R2_N", "C2R3_N")),
    ("Jing6_SN", ("C2R3_S", "C2R2_S", "C2R1_S", "C2R0_S")),
    ("CrossNW_SE", ("R0C0_E", "R0C1_E", "C1R0_N", "C1R1_N", "R2C1_E", "R2C2_E", "C2R2_N", "C2R3_N")),
    ("Short_R2", ("R2C2_E", "R2C3_E")),
    ("Short_C3", ("C3R1_N", "C3R2_N")),
)

BASE_VEHICLE_TYPES = (
    {
        "id": "car",
        "vClass": "passenger",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "length": "4.5",
        "minGap": "2.5",
        "maxSpeed": "16.67",
        "guiShape": "passenger",
    },
    {
        "id": "bus",
        "vClass": "bus",
        "accel": "1.2",
        "decel": "3.5",
        "sigma": "0.3",
        "length": "12.0",
        "minGap": "3.0",
        "maxSpeed": "13.89",
        "guiShape": "bus",
    },
    {
        "id": "truck",
        "vClass": "truck",
        "accel": "1.0",
        "decel": "3.0",
        "sigma": "0.3",
        "length": "9.0",
        "minGap": "3.5",
        "maxSpeed": "11.11",
        "guiShape": "truck",
    },
)

BASE_FLOW_LIBRARY = (
    ("f_Nongye_EW", "Nongye_EW", "car", 0, 3600, 15),
    ("f_Nongye_WE", "Nongye_WE", "car", 5, 3600, 15),
    ("f_Huanghe_EW", "Huanghe_EW", "car", 0, 3600, 18),
    ("f_Huanghe_WE", "Huanghe_WE", "car", 8, 3600, 18),
    ("f_Huayuan_NS", "Huayuan_NS", "car", 0, 3600, 20),
    ("f_Huayuan_SN", "Huayuan_SN", "car", 10, 3600, 20),
    ("f_Jing6_NS", "Jing6_NS", "car", 0, 3600, 22),
    ("f_Jing6_SN", "Jing6_SN", "car", 12, 3600, 22),
    ("f_Hongz_EW", "Hongzhuan_EW", "car", 0, 3600, 30),
    ("f_Hongz_WE", "Hongzhuan_WE", "car", 15, 3600, 30),
    ("f_cross1", "CrossNW_SE", "car", 0, 3600, 45),
    ("f_bus_N", "Nongye_EW", "bus", 60, 3600, 180),
    ("f_bus_H", "Huanghe_WE", "bus", 90, 3600, 180),
    ("f_truck", "Huanghe_EW", "truck", 0, 3600, 300),
)


@dataclass(frozen=True)
class SimSceneProfile:
    scene_name: str
    speed_profile: dict[str, float | str] = field(default_factory=lambda: dict(DEFAULT_SPEED_PROFILE))
    congestion_level_to_speed_factor: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CONGESTION_LEVEL_TO_SPEED_FACTOR)
    )
    incident_edges: dict[str, str] = field(default_factory=dict)
    blocked_edges: tuple[str, ...] = field(default_factory=tuple)
    lane_reduction_edges: dict[str, int] = field(default_factory=dict)
    tls_profile_name: str = "static_all_intersections"
    tls_fixed_program_id: str = "0"
    reroute_enabled: bool = False
    reroute_period: int = 60
    reroute_threshold_factor: float = 1.25
    reroute_threshold_constant: float = 5.0
    demand_profile_name: str = "urban_baseline"
    depart_rate: float = 1.0
    simulation_seed: int = 42
    warmup_seconds: int = 120
    evaluation_start_time: int = 120
    evaluation_end_time: int = 1800
    description: str = ""
    scene_type: str = "normal_baseline"

    def __post_init__(self) -> None:
        if self.depart_rate <= 0:
            raise ValueError(f"depart_rate must be > 0 for scene {self.scene_name}")
        if self.warmup_seconds < 0:
            raise ValueError(f"warmup_seconds must be >= 0 for scene {self.scene_name}")
        if self.evaluation_start_time < 0:
            raise ValueError(f"evaluation_start_time must be >= 0 for scene {self.scene_name}")
        if self.evaluation_end_time <= self.evaluation_start_time:
            raise ValueError(
                f"evaluation_end_time must be > evaluation_start_time for scene {self.scene_name}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _profile(
    *,
    scene_name: str,
    scene_type: str,
    incident_edges: dict[str, str] | None = None,
    blocked_edges: Iterable[str] = (),
    lane_reduction_edges: dict[str, int] | None = None,
    tls_profile_name: str = "static_all_intersections",
    tls_fixed_program_id: str = "0",
    reroute_enabled: bool = False,
    reroute_period: int = 60,
    reroute_threshold_factor: float = 1.25,
    reroute_threshold_constant: float = 5.0,
    demand_profile_name: str = "urban_baseline",
    depart_rate: float = 1.0,
    simulation_seed: int = 42,
    warmup_seconds: int = 120,
    evaluation_start_time: int = 120,
    evaluation_end_time: int = 1800,
    description: str = "",
) -> SimSceneProfile:
    return SimSceneProfile(
        scene_name=scene_name,
        scene_type=scene_type,
        speed_profile=dict(DEFAULT_SPEED_PROFILE),
        congestion_level_to_speed_factor=dict(DEFAULT_CONGESTION_LEVEL_TO_SPEED_FACTOR),
        incident_edges=dict(incident_edges or {}),
        blocked_edges=tuple(blocked_edges),
        lane_reduction_edges=dict(lane_reduction_edges or {}),
        tls_profile_name=tls_profile_name,
        tls_fixed_program_id=tls_fixed_program_id,
        reroute_enabled=reroute_enabled,
        reroute_period=reroute_period,
        reroute_threshold_factor=reroute_threshold_factor,
        reroute_threshold_constant=reroute_threshold_constant,
        demand_profile_name=demand_profile_name,
        depart_rate=depart_rate,
        simulation_seed=simulation_seed,
        warmup_seconds=warmup_seconds,
        evaluation_start_time=evaluation_start_time,
        evaluation_end_time=evaluation_end_time,
        description=description,
    )


SCENE_PROFILES = {
    "normal_baseline": _profile(
        scene_name="normal_baseline",
        scene_type="normal_baseline",
        tls_profile_name="static_program_0",
        demand_profile_name="urban_baseline",
        depart_rate=1.00,
        simulation_seed=101,
        warmup_seconds=120,
        evaluation_start_time=120,
        evaluation_end_time=1800,
        description="No exogenous incidents. Fixed static TLS, baseline demand, fixed seed.",
    ),
    "simple_local": _profile(
        scene_name="simple_local",
        scene_type="simple_local",
        tls_profile_name="static_program_0",
        incident_edges={"R3C0_E": "severe", "R3C1_E": "heavy"},
        lane_reduction_edges={"R3C0_E": 1},
        demand_profile_name="localized_peak",
        depart_rate=1.05,
        simulation_seed=111,
        warmup_seconds=180,
        evaluation_start_time=180,
        evaluation_end_time=2100,
        description="Localized eastbound disruption with one lane reduction.",
    ),
    "directional_asymmetry": _profile(
        scene_name="directional_asymmetry",
        scene_type="directional_asymmetry",
        tls_profile_name="static_program_0_phase_0",
        incident_edges={
            "R4C0_E": "severe",
            "R4C1_E": "severe",
            "R4C2_E": "heavy",
            "R4C3_E": "heavy",
            "C5R0_S": "moderate",
            "C5R1_S": "moderate",
            "C5R2_S": "light",
        },
        lane_reduction_edges={"R4C1_E": 1},
        demand_profile_name="directional_bias",
        depart_rate=1.10,
        simulation_seed=202,
        warmup_seconds=180,
        evaluation_start_time=180,
        evaluation_end_time=2100,
        description="Asymmetric directional demand with eastbound stress and southbound spillover.",
    ),
    "core_blockage": _profile(
        scene_name="core_blockage",
        scene_type="core_blockage",
        tls_profile_name="static_program_0_phase_2",
        incident_edges={
            "C1R1_S": "severe",
            "C1R2_S": "heavy",
            "C3R1_N": "light",
        },
        blocked_edges=("C4R0_N", "C4R1_N", "C2R1_N", "C2R2_N"),
        lane_reduction_edges={"C4R2_N": 1},
        demand_profile_name="core_blockage_peak",
        depart_rate=1.25,
        simulation_seed=303,
        warmup_seconds=240,
        evaluation_start_time=240,
        evaluation_end_time=2400,
        description="Central northbound corridors are blocked while adjacent links absorb spillover.",
    ),
    "propagation_range": _profile(
        scene_name="propagation_range",
        scene_type="propagation_range",
        tls_profile_name="static_program_0_phase_2",
        incident_edges={
            "C4R1_N": "moderate",
            "C4R2_N": "moderate",
            "R1C2_E": "light",
        },
        blocked_edges=("C2R0_N", "C2R1_N", "C2R2_N"),
        lane_reduction_edges={"C4R1_N": 1},
        reroute_enabled=False,
        reroute_period=45,
        reroute_threshold_factor=1.12,
        reroute_threshold_constant=8.0,
        demand_profile_name="spillover_heavy",
        depart_rate=1.20,
        simulation_seed=404,
        warmup_seconds=240,
        evaluation_start_time=240,
        evaluation_end_time=2400,
        description="Core blockage plus upstream spillover on nearby northbound approaches.",
    ),
    "compound_disaster": _profile(
        scene_name="compound_disaster",
        scene_type="compound_disaster",
        tls_profile_name="static_program_0_phase_2",
        incident_edges={
            "R3C0_E": "heavy",
            "R3C1_E": "heavy",
            "C4R2_N": "moderate",
            "C1R1_S": "severe",
            "C1R2_S": "heavy",
        },
        blocked_edges=("C2R1_N", "C2R1_S", "C2R2_N", "C2R2_S"),
        lane_reduction_edges={"C4R1_N": 1, "R3C2_E": 1},
        reroute_enabled=False,
        reroute_period=30,
        reroute_threshold_factor=1.10,
        reroute_threshold_constant=10.0,
        demand_profile_name="compound_disaster_peak",
        depart_rate=1.35,
        simulation_seed=505,
        warmup_seconds=300,
        evaluation_start_time=300,
        evaluation_end_time=2700,
        description="Multi-corridor blockage, heavy congestion, and capacity loss under peak demand.",
    ),
    "blockage_detour_showcase_conservative": _profile(
        scene_name="blockage_detour_showcase_conservative",
        scene_type="core_blockage",
        tls_profile_name="static_program_0_phase_2",
        incident_edges={
            "C1R1_S": "heavy",
            "C1R2_S": "moderate",
            "C3R1_N": "light",
        },
        blocked_edges=("C4R0_N", "C4R1_N"),
        lane_reduction_edges={"C4R2_N": 1},
        reroute_enabled=False,
        reroute_period=30,
        reroute_threshold_factor=1.12,
        reroute_threshold_constant=8.0,
        demand_profile_name="core_blockage_peak",
        depart_rate=0.35,
        simulation_seed=1303,
        warmup_seconds=30,
        evaluation_start_time=30,
        evaluation_end_time=900,
        description="Conservative showcase-only core blockage profile; keeps exogenous stress milder for runtime fairness checks.",
    ),
    "directional_asymmetry_showcase_conservative": _profile(
        scene_name="directional_asymmetry_showcase_conservative",
        scene_type="directional_asymmetry",
        tls_profile_name="static_program_0_phase_0",
        incident_edges={
            "R4C0_E": "heavy",
            "R4C1_E": "moderate",
            "R4C2_E": "moderate",
            "C5R0_S": "light",
            "C5R1_S": "light",
        },
        lane_reduction_edges={"R4C1_E": 1},
        reroute_enabled=True,
        reroute_period=30,
        reroute_threshold_factor=1.12,
        reroute_threshold_constant=8.0,
        demand_profile_name="directional_bias",
        depart_rate=0.40,
        simulation_seed=1202,
        warmup_seconds=30,
        evaluation_start_time=30,
        evaluation_end_time=900,
        description="Conservative showcase-only directional profile with reduced incident intensity and demand.",
    ),
    "compound_disaster_showcase_conservative": _profile(
        scene_name="compound_disaster_showcase_conservative",
        scene_type="compound_disaster",
        tls_profile_name="static_program_0_phase_2",
        incident_edges={
            "R3C0_E": "heavy",
            "R3C1_E": "moderate",
            "C4R2_N": "moderate",
            "C1R1_S": "heavy",
        },
        blocked_edges=("C2R1_N", "C2R2_N"),
        lane_reduction_edges={"C4R1_N": 1},
        reroute_enabled=True,
        reroute_period=30,
        reroute_threshold_factor=1.12,
        reroute_threshold_constant=8.0,
        demand_profile_name="compound_disaster_peak",
        depart_rate=0.40,
        simulation_seed=1505,
        warmup_seconds=45,
        evaluation_start_time=45,
        evaluation_end_time=1050,
        description="Conservative showcase-only compound profile; trims closures and stacked capacity loss without changing baseline logic.",
    ),
    "temporal_switch": _profile(
        scene_name="temporal_switch",
        scene_type="temporal_switch",
        tls_profile_name="static_program_0_phase_0",
        incident_edges={"R3C0_E": "severe", "R3C1_E": "heavy", "R3C2_E": "moderate"},
        demand_profile_name="rush_hour_slice",
        depart_rate=1.30,
        simulation_seed=606,
        warmup_seconds=300,
        evaluation_start_time=300,
        evaluation_end_time=1800,
        description="Fixed high-demand slice representing the peak window of a temporal event.",
    ),
    "anti_truncation_eval": _profile(
        scene_name="anti_truncation_eval",
        scene_type="anti_truncation_eval",
        tls_profile_name="static_program_0_phase_0",
        incident_edges={
            "R0C1_E": "moderate",
            "R0C2_E": "heavy",
            "R1C0_E": "light",
            "R1C3_E": "moderate",
            "R3C1_E": "severe",
            "R3C2_E": "heavy",
            "R4C0_E": "light",
            "R4C2_E": "moderate",
            "C1R0_N": "moderate",
            "C1R2_S": "heavy",
            "C3R1_N": "light",
            "C4R1_N": "moderate",
            "C5R2_S": "moderate",
        },
        blocked_edges=("C2R0_N", "C2R2_N"),
        lane_reduction_edges={"R3C1_E": 1, "C4R1_N": 1, "R1C3_E": 1},
        reroute_enabled=True,
        reroute_period=30,
        reroute_threshold_factor=1.08,
        reroute_threshold_constant=6.0,
        demand_profile_name="coverage_stress",
        depart_rate=1.15,
        simulation_seed=909,
        warmup_seconds=300,
        evaluation_start_time=300,
        evaluation_end_time=3000,
        description="Wide-coverage stress profile used to prevent evaluation from collapsing to a tiny active subset.",
    ),
}

STANDARD_SCENE_PROFILE_NAMES = (
    "normal_baseline",
    "simple_local",
    "directional_asymmetry",
    "core_blockage",
    "propagation_range",
    "compound_disaster",
)

SHOWCASE_CONSERVATIVE_PROFILE_NAMES = (
    "blockage_detour_showcase_conservative",
    "directional_asymmetry_showcase_conservative",
    "compound_disaster_showcase_conservative",
)


def _canonical_scene_profile_payload() -> dict[str, Any]:
    ordered_names = list(STANDARD_SCENE_PROFILE_NAMES) + [
        name for name in SCENE_PROFILES.keys() if name not in STANDARD_SCENE_PROFILE_NAMES
    ]
    return {
        name: SCENE_PROFILES[name].to_dict()
        for name in ordered_names
    }


SCENE_PROFILE_VERSION = "sha256:" + hashlib.sha256(
    json.dumps(
        _canonical_scene_profile_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()[:12]

SCENE_TYPE_TO_PROFILE_NAME = {
    "normal_baseline": "normal_baseline",
    "simple_local": "simple_local",
    "directional_asymmetry": "directional_asymmetry",
    "core_blockage": "core_blockage",
    "propagation_range": "propagation_range",
    "compound_disaster": "compound_disaster",
    "temporal_switch": "temporal_switch",
    "anti_truncation_eval": "anti_truncation_eval",
}


def list_scene_profiles() -> tuple[str, ...]:
    return tuple(SCENE_PROFILES.keys())


def get_scene_profile_version() -> str:
    return SCENE_PROFILE_VERSION


def scene_profile_to_dict(profile: SimSceneProfile) -> dict[str, Any]:
    return profile.to_dict()


def get_scene_profile(scene_name: str) -> SimSceneProfile:
    key = str(scene_name or "").strip()
    if key not in SCENE_PROFILES:
        raise KeyError(f"Unknown scene profile: {scene_name}")
    return SCENE_PROFILES[key]


def resolve_scene_profile_name(
    *,
    scene_name: str | None = None,
    scene_type: str | None = None,
    default_name: str = "normal_baseline",
) -> str:
    if scene_name:
        key = str(scene_name).strip()
        if key in SCENE_PROFILES:
            return key
    if scene_type:
        key = SCENE_TYPE_TO_PROFILE_NAME.get(str(scene_type).strip().lower())
        if key:
            return key
    return default_name


def resolve_scene_profile(
    *,
    scene_name: str | None = None,
    scene_type: str | None = None,
    default_name: str = "normal_baseline",
) -> SimSceneProfile:
    return get_scene_profile(
        resolve_scene_profile_name(scene_name=scene_name, scene_type=scene_type, default_name=default_name)
    )


def build_environment_weight_layers(
    edge_ids: Iterable[str],
    profile: SimSceneProfile,
    *,
    enable_spillover: bool = False,
) -> dict[str, Any]:
    return build_spillover_weights(
        edge_ids,
        profile,
        enable_spillover=enable_spillover,
    )


def build_environment_weights(
    edge_ids: Iterable[str],
    profile: SimSceneProfile,
    *,
    enable_spillover: bool = False,
) -> dict[str, float]:
    return dict(
        build_environment_weight_layers(
            edge_ids,
            profile,
            enable_spillover=enable_spillover,
        ).get("weights", {})
    )


def build_edge_speed_factors(edge_ids: Iterable[str], profile: SimSceneProfile) -> dict[str, float]:
    base_speed_factor = float(profile.speed_profile.get("base_speed_factor", 1.0))
    lane_reduction_speed_factor = float(
        profile.speed_profile.get("lane_reduction_speed_factor", DEFAULT_CONGESTION_LEVEL_TO_SPEED_FACTOR["lane_reduction"])
    )
    blocked_speed_factor = float(
        profile.speed_profile.get("blocked_speed_factor", DEFAULT_CONGESTION_LEVEL_TO_SPEED_FACTOR["blocked"])
    )
    factors = {edge_id: base_speed_factor for edge_id in edge_ids}
    for edge_id, remaining_lanes in profile.lane_reduction_edges.items():
        factor = lane_reduction_speed_factor
        if int(remaining_lanes) <= 1:
            factor = min(factor, 0.58)
        factors[edge_id] = min(factors.get(edge_id, 1.0), factor)
    for edge_id, level in profile.incident_edges.items():
        level_key = str(level).strip().lower()
        factor = float(profile.congestion_level_to_speed_factor.get(level_key, DEFAULT_CONGESTION_LEVEL_TO_SPEED_FACTOR["moderate"]))
        factors[edge_id] = min(factors.get(edge_id, 1.0), factor)
    for edge_id in profile.blocked_edges:
        factors[edge_id] = min(factors.get(edge_id, 1.0), blocked_speed_factor)
    return factors


def build_sumo_runtime_options(
    profile: SimSceneProfile,
    *,
    route_file_path: str | None = None,
    tripinfo_path: str | None = None,
    step_length: float | None = None,
) -> list[str]:
    options = [
        "--seed",
        str(profile.simulation_seed),
        "--ignore-route-errors",
        "true",
        "--no-step-log",
        "true",
        "--no-warnings",
        "--duration-log.disable",
        "--tripinfo-output.write-unfinished",
    ]
    if route_file_path:
        options.extend(["--route-files", route_file_path])
    if tripinfo_path:
        options.extend(["--tripinfo-output", tripinfo_path])
    if step_length is not None:
        options.extend(["--step-length", f"{float(step_length):.3f}"])
    return options


def build_reroute_policy(profile: SimSceneProfile) -> dict[str, float | int | bool | str]:
    reroute_enabled = bool(profile.reroute_enabled)
    if reroute_enabled:
        reason = (
            f"scene profile enables periodic reroute every {int(profile.reroute_period)} s "
            f"with threshold {float(profile.reroute_threshold_factor):.2f}x + "
            f"{float(profile.reroute_threshold_constant):.1f} s"
        )
        effective_status = "enabled"
    else:
        reason = (
            "current experiment protocol keeps the planner route fixed for this scene profile; "
            "dynamic reroute is intentionally disabled"
        )
        effective_status = "disabled"
    return {
        "reroute_enabled": reroute_enabled,
        "reroute_period": int(profile.reroute_period),
        "reroute_threshold_factor": float(profile.reroute_threshold_factor),
        "reroute_threshold_constant": float(profile.reroute_threshold_constant),
        "effective_status": effective_status,
        "reason": reason,
        "source": "scene_profile",
    }


def write_scene_route_file(path: str | Path, profile: SimSceneProfile) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>", ""]
    lines.append(f"  <!-- demand_profile={profile.demand_profile_name} depart_rate={profile.depart_rate:.3f} -->")
    lines.append("")
    for vtype in BASE_VEHICLE_TYPES:
        attrs = " ".join(f'{key}="{value}"' for key, value in vtype.items())
        lines.append(f"  <vType {attrs}/>")
    lines.append("")
    for route_name, edges in BASE_ROUTE_LIBRARY:
        lines.append(f'  <route id="{route_name}" edges="{" ".join(edges)}"/>')
    lines.append("")
    for flow_id, route_name, vehicle_type, begin, end, period in BASE_FLOW_LIBRARY:
        scaled_period = max(1.0, float(period) / float(profile.depart_rate))
        lines.append(
            f'  <flow id="{flow_id}" route="{route_name}" type="{vehicle_type}" '
            f'begin="{begin}" end="{end}" period="{scaled_period:.2f}" '
            'departLane="best" departSpeed="max"/>'
        )
    lines.append("</routes>")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)
