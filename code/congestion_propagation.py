from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Iterable

import sumolib

from roadnet_meta import SUMO_NET_PATH, load_roadnet_meta

DEFAULT_CONGESTION_LEVEL_TO_WEIGHT = {
    "normal": 2.0,
    "light": 3.0,
    "moderate": 5.0,
    "heavy": 6.2,
    "severe": 7.5,
    "blocked": 9.5,
    "lane_reduction": 4.2,
}


SCENE_SPILLOVER_CONFIG = {
    "normal_baseline": {
        "enabled": False,
        "hop_factors": {1: 0.0, 2: 0.0},
        "runtime": {
            "hop_speed_factor_bounds": {1: [1.0, 1.0], 2: [1.0, 1.0]},
            "hop_travel_time_multiplier": {1: 1.0, 2: 1.0},
        },
    },
    "simple_local": {
        "enabled": True,
        "hop_factors": {1: 0.38, 2: 0.14},
        "runtime": {
            "hop_speed_factor_bounds": {1: [0.72, 0.82], 2: [0.88, 0.94]},
            "hop_travel_time_multiplier": {1: 1.20, 2: 1.08},
        },
    },
    "directional_asymmetry": {
        "enabled": True,
        "hop_factors": {1: 0.42, 2: 0.16},
        "runtime": {
            "hop_speed_factor_bounds": {1: [0.68, 0.80], 2: [0.86, 0.94]},
            "hop_travel_time_multiplier": {1: 1.24, 2: 1.10},
        },
    },
    "core_blockage": {
        "enabled": True,
        "hop_factors": {1: 0.56, 2: 0.23},
        "runtime": {
            "hop_speed_factor_bounds": {1: [0.62, 0.76], 2: [0.84, 0.92]},
            "hop_travel_time_multiplier": {1: 1.30, 2: 1.12},
        },
    },
    "propagation_range": {
        "enabled": True,
        "hop_factors": {1: 0.60, 2: 0.25},
        "runtime": {
            "hop_speed_factor_bounds": {1: [0.60, 0.74], 2: [0.82, 0.90]},
            "hop_travel_time_multiplier": {1: 1.34, 2: 1.14},
        },
    },
    "compound_disaster": {
        "enabled": True,
        "hop_factors": {1: 0.52, 2: 0.21},
        "runtime": {
            "hop_speed_factor_bounds": {1: [0.60, 0.74], 2: [0.82, 0.90]},
            "hop_travel_time_multiplier": {1: 1.32, 2: 1.14},
        },
    },
    "blockage_detour_showcase_conservative": {
        "enabled": True,
        "hop_factors": {1: 0.34, 2: 0.12},
        "runtime": {
            "hop_speed_factor_bounds": {1: [0.76, 0.86], 2: [0.90, 0.96]},
            "hop_travel_time_multiplier": {1: 1.16, 2: 1.06},
        },
    },
    "directional_asymmetry_showcase_conservative": {
        "enabled": True,
        "hop_factors": {1: 0.30, 2: 0.10},
        "runtime": {
            "hop_speed_factor_bounds": {1: [0.78, 0.88], 2: [0.91, 0.97]},
            "hop_travel_time_multiplier": {1: 1.14, 2: 1.05},
        },
    },
    "compound_disaster_showcase_conservative": {
        "enabled": True,
        "hop_factors": {1: 0.32, 2: 0.11},
        "runtime": {
            "hop_speed_factor_bounds": {1: [0.76, 0.86], 2: [0.90, 0.96]},
            "hop_travel_time_multiplier": {1: 1.16, 2: 1.06},
        },
    },
    "temporal_switch": {
        "enabled": True,
        "hop_factors": {1: 0.40, 2: 0.14},
        "runtime": {
            "hop_speed_factor_bounds": {1: [0.72, 0.82], 2: [0.88, 0.95]},
            "hop_travel_time_multiplier": {1: 1.18, 2: 1.08},
        },
    },
    "anti_truncation_eval": {
        "enabled": True,
        "hop_factors": {1: 0.45, 2: 0.18},
        "runtime": {
            "hop_speed_factor_bounds": {1: [0.68, 0.80], 2: [0.86, 0.94]},
            "hop_travel_time_multiplier": {1: 1.22, 2: 1.10},
        },
    },
}


def _congestion_bucket(weight: float) -> str:
    value = float(weight)
    if value >= 8.8:
        return "closed"
    if value >= 7.0:
        return "severe"
    if value >= 4.5:
        return "moderate"
    if value >= 2.5:
        return "light"
    return "clear"


def _weight_range(weights: dict[str, float]) -> dict[str, float] | None:
    values = [float(weight) for weight in dict(weights or {}).values()]
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
    }


def _level_counts(weights: dict[str, float]) -> dict[str, int]:
    counts = {"light": 0, "moderate": 0, "severe": 0, "closed": 0}
    for weight in dict(weights or {}).values():
        bucket = _congestion_bucket(float(weight))
        if bucket in counts:
            counts[bucket] += 1
    return counts


def _config_for_scene(scene_name: str, scene_type: str) -> dict:
    return dict(
        SCENE_SPILLOVER_CONFIG.get(
            str(scene_name or "").strip(),
            SCENE_SPILLOVER_CONFIG.get(str(scene_type or "").strip(), SCENE_SPILLOVER_CONFIG["simple_local"]),
        )
    )


def resolve_spillover_runtime_config(scene_name: str, scene_type: str) -> dict:
    config = _config_for_scene(scene_name, scene_type)
    runtime_cfg = dict(config.get("runtime") or {})
    return {
        "hop_speed_factor_bounds": {
            int(hop): [float(bounds[0]), float(bounds[1])]
            for hop, bounds in dict(runtime_cfg.get("hop_speed_factor_bounds") or {}).items()
            if isinstance(bounds, (list, tuple)) and len(bounds) >= 2
        },
        "hop_travel_time_multiplier": {
            int(hop): float(multiplier)
            for hop, multiplier in dict(runtime_cfg.get("hop_travel_time_multiplier") or {}).items()
        },
    }


@lru_cache(maxsize=4)
def _topology_neighbors(net_path: str) -> dict[str, tuple[str, ...]]:
    meta = load_roadnet_meta(net_path)
    neighbors = {edge_id: set() for edge_id in meta.edge_ids}
    try:
        net = sumolib.net.readNet(net_path)
    except Exception:
        return {edge_id: tuple(sorted(items)) for edge_id, items in neighbors.items()}

    for edge_id in meta.edge_ids:
        try:
            edge_obj = net.getEdge(edge_id)
        except Exception:
            edge_obj = None
        if edge_obj is None:
            continue
        linked = neighbors[edge_id]
        for next_edge_obj in edge_obj.getOutgoing().keys():
            linked.add(next_edge_obj.getID())
        for prev_edge_obj in edge_obj.getIncoming().keys():
            linked.add(prev_edge_obj.getID())
    return {edge_id: tuple(sorted(items)) for edge_id, items in neighbors.items()}


def _raw_incident_weights(edge_ids: Iterable[str], profile) -> tuple[dict[str, float], dict[str, str]]:
    weights = {
        edge_id: float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["normal"])
        for edge_id in edge_ids
    }
    source_kind = {}
    for edge_id, remaining_lanes in dict(profile.lane_reduction_edges or {}).items():
        penalty = float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["lane_reduction"])
        if int(remaining_lanes) <= 1:
            penalty = max(penalty, float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["heavy"]))
        weights[edge_id] = max(weights.get(edge_id, 2.0), penalty)
        source_kind[edge_id] = "lane_reduction"
    for edge_id, level in dict(profile.incident_edges or {}).items():
        weights[edge_id] = max(
            weights.get(edge_id, 2.0),
            float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT.get(str(level).strip().lower(), DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["moderate"])),
        )
        source_kind[edge_id] = "incident"
    for edge_id in tuple(profile.blocked_edges or ()):
        weights[edge_id] = max(weights.get(edge_id, 2.0), float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["blocked"]))
        source_kind[edge_id] = "blocked"
    return weights, source_kind


def propagate_congestion_to_neighbors(
    *,
    raw_weights: dict[str, float],
    source_kind: dict[str, str],
    scene_name: str,
    scene_type: str,
    net_path: str = SUMO_NET_PATH,
    max_hops: int = 2,
) -> dict:
    road_meta = load_roadnet_meta(net_path)
    topology_neighbors = _topology_neighbors(net_path)
    config = _config_for_scene(scene_name, scene_type)
    hop_factors = dict(config.get("hop_factors") or {})
    base_weight = float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["normal"])

    spillover_weights: dict[str, float] = {}
    hop_by_edge: dict[str, int] = {}
    source_by_edge: dict[str, str] = {}
    seed_edges = [edge_id for edge_id, value in raw_weights.items() if float(value) > base_weight + 1e-9]

    for seed_edge in seed_edges:
        seed_weight = float(raw_weights.get(seed_edge, base_weight))
        seed_delta = max(seed_weight - base_weight, 0.0)
        if seed_delta <= 0.0:
            continue
        seed_meta = road_meta.edge_meta.get(seed_edge, {})
        queue = deque([(seed_edge, 0)])
        seen = {seed_edge}

        while queue:
            current_edge, current_hop = queue.popleft()
            if current_hop >= max_hops:
                continue
            for neighbor_edge in topology_neighbors.get(current_edge, ()):
                if neighbor_edge in seen:
                    continue
                seen.add(neighbor_edge)
                next_hop = current_hop + 1
                queue.append((neighbor_edge, next_hop))
                if neighbor_edge in source_kind:
                    continue
                neighbor_meta = road_meta.edge_meta.get(neighbor_edge, {})
                factor = float(hop_factors.get(next_hop, 0.0))
                if factor <= 0.0:
                    continue
                if seed_meta.get("dir_code") == neighbor_meta.get("dir_code"):
                    factor *= 1.10
                elif seed_meta.get("axis") == neighbor_meta.get("axis"):
                    factor *= 0.92
                else:
                    factor *= 0.62
                if seed_meta.get("road") == neighbor_meta.get("road"):
                    factor *= 1.12
                if seed_meta.get("rtype") == neighbor_meta.get("rtype"):
                    factor *= 1.03
                if neighbor_meta.get("rtype") == "A" and str(scene_type) in {"core_blockage", "propagation_range", "compound_disaster"}:
                    factor *= 1.05
                if source_kind.get(seed_edge) == "blocked" and next_hop == 1:
                    factor *= 1.04
                if next_hop == 2 and seed_meta.get("road") != neighbor_meta.get("road"):
                    factor *= 0.72
                if next_hop == 2 and seed_meta.get("axis") != neighbor_meta.get("axis"):
                    factor *= 0.66

                propagated_delta = seed_delta * factor
                if next_hop == 1:
                    propagated_delta = min(propagated_delta, seed_delta * 0.52)
                else:
                    propagated_delta = min(propagated_delta, seed_delta * 0.18)
                propagated_weight = base_weight + propagated_delta
                if propagated_weight < (2.55 if next_hop == 1 else 2.75):
                    continue
                propagated_weight = min(propagated_weight, seed_weight - (0.25 if next_hop == 1 else 0.6))
                if propagated_weight <= raw_weights.get(neighbor_edge, base_weight) + 1e-9:
                    continue
                if propagated_weight > spillover_weights.get(neighbor_edge, base_weight):
                    spillover_weights[neighbor_edge] = propagated_weight
                    hop_by_edge[neighbor_edge] = next_hop
                    source_by_edge[neighbor_edge] = seed_edge

    return {
        "spillover_weights": spillover_weights,
        "spillover_hop_by_edge": hop_by_edge,
        "spillover_source_by_edge": source_by_edge,
    }


def build_spillover_weights(
    edge_ids: Iterable[str],
    profile,
    *,
    enable_spillover: bool = False,
    net_path: str = SUMO_NET_PATH,
) -> dict:
    base_weight = float(DEFAULT_CONGESTION_LEVEL_TO_WEIGHT["normal"])
    edge_ids = list(edge_ids)
    raw_weights, source_kind = _raw_incident_weights(edge_ids, profile)
    raw_only_weights = {
        edge_id: float(weight)
        for edge_id, weight in raw_weights.items()
        if float(weight) > base_weight + 1e-9
    }

    if not enable_spillover:
        return {
            "raw_weights": dict(raw_weights),
            "raw_only_weights": raw_only_weights,
            "spillover_weights": {},
            "weights": dict(raw_weights),
            "direct_source_kind": source_kind,
            "spillover_hop_by_edge": {},
            "spillover_source_by_edge": {},
            "raw_incident_edge_ids": tuple(sorted(source_kind)),
            "spillover_edge_ids": tuple(),
            "spillover_enabled": False,
            "stats": {
                "raw_incident_count": len(source_kind),
                "spillover_1hop_count": 0,
                "spillover_2hop_count": 0,
                "spillover_total_count": 0,
            },
        }

    spillover = propagate_congestion_to_neighbors(
        raw_weights=raw_weights,
        source_kind=source_kind,
        scene_name=getattr(profile, "scene_name", ""),
        scene_type=getattr(profile, "scene_type", ""),
        net_path=net_path,
    )
    final_weights = dict(raw_weights)
    for edge_id, weight in spillover["spillover_weights"].items():
        final_weights[edge_id] = max(final_weights.get(edge_id, base_weight), float(weight))

    hop_by_edge = dict(spillover["spillover_hop_by_edge"])
    spillover_weights = {
        edge_id: float(weight)
        for edge_id, weight in spillover["spillover_weights"].items()
        if edge_id not in source_kind
    }
    spillover_1hop_weights = {
        edge_id: float(weight)
        for edge_id, weight in spillover_weights.items()
        if int(hop_by_edge.get(edge_id, 0)) == 1
    }
    spillover_2hop_weights = {
        edge_id: float(weight)
        for edge_id, weight in spillover_weights.items()
        if int(hop_by_edge.get(edge_id, 0)) == 2
    }
    spillover_1 = sum(1 for hop in hop_by_edge.values() if int(hop) == 1)
    spillover_2 = sum(1 for hop in hop_by_edge.values() if int(hop) == 2)
    return {
        "raw_weights": dict(raw_weights),
        "raw_only_weights": raw_only_weights,
        "spillover_weights": spillover_weights,
        "weights": final_weights,
        "direct_source_kind": source_kind,
        "spillover_hop_by_edge": hop_by_edge,
        "spillover_source_by_edge": dict(spillover["spillover_source_by_edge"]),
        "raw_incident_edge_ids": tuple(sorted(source_kind)),
        "spillover_edge_ids": tuple(sorted(spillover_weights)),
        "spillover_enabled": True,
        "stats": {
            "scene_name": str(getattr(profile, "scene_name", "") or ""),
            "scene_type": str(getattr(profile, "scene_type", "") or ""),
            "raw_incident_count": len(source_kind),
            "spillover_1hop_count": spillover_1,
            "spillover_2hop_count": spillover_2,
            "spillover_total_count": len(spillover_weights),
            "hop_factor_config": dict(_config_for_scene(getattr(profile, "scene_name", ""), getattr(profile, "scene_type", "")).get("hop_factors") or {}),
            "runtime_config": resolve_spillover_runtime_config(
                getattr(profile, "scene_name", ""),
                getattr(profile, "scene_type", ""),
            ),
            "raw_weight_range": _weight_range(raw_only_weights),
            "spillover_weight_range": _weight_range(spillover_weights),
            "spillover_1hop_weight_range": _weight_range(spillover_1hop_weights),
            "spillover_2hop_weight_range": _weight_range(spillover_2hop_weights),
            "raw_level_counts": _level_counts(raw_only_weights),
            "spillover_level_counts": _level_counts(spillover_weights),
            "spillover_1hop_level_counts": _level_counts(spillover_1hop_weights),
            "spillover_2hop_level_counts": _level_counts(spillover_2hop_weights),
        },
    }
