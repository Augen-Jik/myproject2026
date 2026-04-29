from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import sumolib

from roadnet_meta import load_roadnet_meta
from routing import dijkstra_route, edge_freeflow_time, edge_travel_time, free_flow_route
from sim_scene_profiles import build_environment_weight_layers, get_scene_profile
from sumo_runner import build_safe_vehicle_route


PROFILE_DEMO_LABELS = {
    "core_blockage": "blockage detour",
    "directional_asymmetry": "directional asymmetry",
    "propagation_range": "propagation range",
    "compound_disaster": "compound disaster",
}

DEFAULT_PRODUCTION_SHOWCASE_NAME = "blockage_detour_showcase"
PRODUCTION_SHOWCASE_NAMES = (
    "blockage_detour_showcase",
    "directional_asymmetry_showcase",
)

PRODUCTION_SHOWCASE_PRESETS = {
    "blockage_detour_showcase": {
        "scene_profile": "core_blockage",
        "demo_label": "blockage detour",
        "start_edge": "R4C0_E",
        "end_edge": "C0R0_S",
        "actual_baseline_time_s": 1404.8,
        "actual_planned_time_s": 1247.0,
        "actual_saving_seconds": 157.8,
        "actual_saving_ratio": 0.1123,
        "showcase_badge": "Positive Gain Showcase",
        "showcase_stage": "production",
        "showcase_group": "production",
        "availability": "production_default",
        "status_note": "Production Showcase：实际 SUMO 收益稳定，默认首页展示案例。",
    },
    "directional_asymmetry_showcase": {
        "scene_profile": "directional_asymmetry",
        "demo_label": "directional asymmetry",
        "start_edge": "C0R2_S",
        "end_edge": "R4C3_E",
        "actual_baseline_time_s": 1099.0,
        "actual_planned_time_s": 1050.1,
        "actual_saving_seconds": 48.9,
        "actual_saving_ratio": 0.0445,
        "showcase_badge": "Positive Gain Showcase",
        "showcase_stage": "production",
        "showcase_group": "production",
        "availability": "production_optional",
        "status_note": "Production Showcase：保留为第二个正收益案例。",
    },
}

CALIBRATION_EXPERIMENTAL_SHOWCASE_PRESETS = {
    "propagation_range_showcase": {
        "scene_profile": "propagation_range",
        "demo_label": "propagation range",
        "start_edge": "R0C0_W",
        "end_edge": "C2R3_S",
        "actual_baseline_time_s": 1534.2,
        "actual_planned_time_s": 1534.2,
        "actual_saving_seconds": 0.0,
        "actual_saving_ratio": 0.0,
        "showcase_badge": "Calibration Only",
        "showcase_stage": "calibration",
        "showcase_group": "calibration_debug",
        "availability": "calibration_only",
        "status_note": "Calibration only：不进入 production showcase 默认列表，仅保留在 calibration / hidden / debug 区。",
        "display_reason": "Calibration only：用于外溢范围校准，不进入 production 主展示。",
        "expected_difference": "当前仅保留为 calibration / hidden / debug 入口，不作为 production showcase 结论。",
    },
    "compound_disaster_showcase": {
        "scene_profile": "compound_disaster",
        "demo_label": "compound disaster",
        "start_edge": "C0R2_N",
        "end_edge": "C1R0_S",
        "actual_baseline_time_s": 1733.1,
        "actual_planned_time_s": 1733.1,
        "actual_saving_seconds": 0.0,
        "actual_saving_ratio": 0.0,
        "showcase_badge": "Neutral Diagnostic Case",
        "showcase_stage": "experimental",
        "showcase_group": "experimental_neutral",
        "availability": "experimental_only",
        "status_note": "Neutral Diagnostic Case：当前规划与基线持平，但路径与脆弱走廊关系可解释。",
        "display_reason": "当前规划与基线持平，但路径与脆弱走廊关系可解释。",
        "expected_difference": "baseline 会穿过脆弱 corridor，planned 会主动避开；实际 SUMO 结果当前持平。",
    },
}

SHOWCASE_SCENE_SPECS = {
    **PRODUCTION_SHOWCASE_PRESETS,
    **CALIBRATION_EXPERIMENTAL_SHOWCASE_PRESETS,
}
PRIMARY_CORRIDOR_TYPES = {"A"}


def _estimate_path_time(net, path, weights=None, *, free_flow: bool = False) -> float:
    total = 0.0
    for edge_id in list(path or []):
        try:
            edge_obj = net.getEdge(edge_id)
        except Exception:
            edge_obj = None
        if edge_obj is None:
            continue
        if free_flow:
            total += float(edge_freeflow_time(edge_obj))
        else:
            total += float(edge_travel_time(edge_obj, float((weights or {}).get(edge_id, 2.0))))
    return float(total)


def _route_overlap_ratio(left, right) -> float:
    left_list = list(left or [])
    right_list = list(right or [])
    if not left_list or not right_list:
        return 0.0
    shared = len(set(left_list) & set(right_list))
    return shared / max(len(left_list), len(right_list), 1)


def _path_turns(road_meta, path) -> int:
    roads = _path_road_sequence(road_meta, path)
    if len(roads) < 2:
        return 0
    turns = 0
    for prev, cur in zip(roads, roads[1:]):
        if prev != cur:
            turns += 1
    return turns


def _path_road_sequence(road_meta, path) -> list[str]:
    roads = [
        road_meta.edge_meta.get(edge_id, {}).get("road")
        for edge_id in list(path or [])
        if edge_id in road_meta.edge_meta
    ]
    return [road for road in roads if road]


def _path_unique_roads(road_meta, path) -> int:
    roads = {
        road_meta.edge_meta.get(edge_id, {}).get("road")
        for edge_id in list(path or [])
        if edge_id in road_meta.edge_meta
    }
    roads.discard(None)
    return len(roads)


def _impacted_edge_sets(scene_profile):
    incident = set(dict(scene_profile.incident_edges or {}).keys())
    blocked = set(scene_profile.blocked_edges or ())
    lane_reduced = set(dict(scene_profile.lane_reduction_edges or {}).keys())
    all_impacted = incident | blocked | lane_reduced
    return {
        "incident": incident,
        "blocked": blocked,
        "lane_reduced": lane_reduced,
        "all": all_impacted,
    }


def _edge_center_signature(road_meta, edge_id: str) -> tuple[int, int]:
    meta = road_meta.edge_meta.get(edge_id, {})
    axis = meta.get("axis")
    row_idx = int(meta.get("row", 0))
    col_idx = int(meta.get("col", 0))
    if axis == "H":
        return row_idx, col_idx
    return row_idx, col_idx


def _candidate_edge_pool(road_meta) -> list[str]:
    candidates = []
    max_h_col_idx = max(int(meta.get("col", 0)) for meta in road_meta.edge_meta.values() if meta.get("axis") == "H")
    max_v_row_idx = max(int(meta.get("row", 0)) for meta in road_meta.edge_meta.values() if meta.get("axis") == "V")
    for edge_id in road_meta.edge_ids:
        meta = road_meta.edge_meta.get(edge_id, {})
        axis = meta.get("axis")
        row_idx = int(meta.get("row", 0))
        col_idx = int(meta.get("col", 0))
        if axis == "H":
            boundary = col_idx in {0, max_h_col_idx}
        else:
            boundary = row_idx in {0, max_v_row_idx}
        if boundary:
            candidates.append(edge_id)
    return sorted(dict.fromkeys(candidates))


def _directional_asymmetry_score(road_meta, scene_weights: dict[str, float]) -> float:
    score = 0.0
    for edge_id, meta in road_meta.edge_meta.items():
        opp = meta.get("opp")
        if not opp or opp not in scene_weights:
            continue
        diff = abs(float(scene_weights.get(edge_id, 2.0)) - float(scene_weights.get(opp, 2.0)))
        score = max(score, diff)
    return score


def _path_length_m(net, path) -> float:
    total = 0.0
    for edge_id in list(path or []):
        try:
            edge_obj = net.getEdge(edge_id)
        except Exception:
            edge_obj = None
        if edge_obj is None:
            continue
        try:
            total += float(edge_obj.getLength())
        except Exception:
            continue
    return float(total)


def _primary_corridor_ratio(road_meta, path) -> float:
    path_list = list(path or [])
    if not path_list:
        return 0.0
    primary_edges = 0
    for edge_id in path_list:
        meta = road_meta.edge_meta.get(edge_id, {})
        if str(meta.get("rtype") or "").strip() in PRIMARY_CORRIDOR_TYPES:
            primary_edges += 1
    return primary_edges / max(len(path_list), 1)


def _same_main_corridor_direct(road_meta, path) -> bool:
    roads = _path_road_sequence(road_meta, path)
    unique_roads = list(dict.fromkeys(roads))
    if len(unique_roads) <= 1:
        return True
    if len(unique_roads) == 2 and _path_turns(road_meta, path) == 0 and _primary_corridor_ratio(road_meta, path) >= 0.90:
        return True
    return False


def _final_safe_route_edges(route_info: dict[str, Any]) -> list[str]:
    return [str(edge_id) for edge_id in list(route_info.get("final_route_edges") or []) if str(edge_id).strip()]


def _safe_route_stability(net, start_edge: str, end_edge: str, path) -> tuple[float, dict]:
    route_info = build_safe_vehicle_route(
        net,
        original_start_edge=start_edge,
        original_end_edge=end_edge,
        original_route_edges=tuple(path or ()),
    )
    if not route_info.get("ok"):
        return 0.0, route_info
    repaired = bool(route_info.get("repair_actions"))
    used_fallback = bool(route_info.get("used_sumo_fallback"))
    stability = 1.0
    if repaired:
        stability -= 0.25
    if used_fallback:
        stability -= 0.35
    return max(stability, 0.1), route_info


def _append_target_edge_if_connected(net, path, end_edge: str) -> list[str]:
    path_list = [str(edge_id) for edge_id in list(path or []) if str(edge_id).strip()]
    if not path_list or not str(end_edge or "").strip():
        return path_list
    if path_list[-1] == end_edge:
        return path_list
    try:
        last_edge = net.getEdge(path_list[-1])
    except Exception:
        last_edge = None
    if last_edge is None:
        return path_list
    outgoing_ids = {edge_obj.getID() for edge_obj in last_edge.getOutgoing().keys()}
    if end_edge in outgoing_ids:
        return path_list + [end_edge]
    return path_list


def _candidate_priority(item: dict[str, Any]) -> tuple[float, float, float, float, float]:
    raw_gap = float(item.get("baseline_impacted_edges", 0) or 0) - float(item.get("planned_impacted_edges", 0) or 0)
    return (
        float(item.get("estimated_delta_s") or 0.0),
        raw_gap,
        1.0 - float(item.get("overlap_ratio") or 0.0),
        1.0 - float(item.get("safe_route_overlap_ratio") or 0.0),
        float(item.get("score") or 0.0),
    )


def _profile_expected_difference(scene_name: str, score_bundle: dict) -> str:
    if scene_name == "core_blockage":
        return "baseline 会撞上封闭走廊，planned 更容易展示明确绕行与 travel time 差异"
    if scene_name == "directional_asymmetry":
        return "可以检验模型是否理解单向不对称，重点体现 compliance 与路线方向选择"
    if scene_name == "propagation_range":
        return "可以展示是否识别外溢范围，planned 与 baseline 的 reroute / overlap 更容易拉开"
    if scene_name == "compound_disaster":
        return "多处约束叠加，最容易放大 travel time、compliance 和失败/成功差异"
    return f"预计时间差 {score_bundle.get('estimated_delta_s', 0.0):.1f}s，路径重合率 {score_bundle.get('overlap_ratio', 0.0):.2f}"


def _reason_text(score_bundle: dict) -> str:
    saving_s = float(score_bundle.get("estimated_delta_s") or 0.0)
    saving_ratio_pct = float(score_bundle.get("saving_ratio_pct") or 0.0)
    if abs(saving_s) <= 1e-9:
        return "当前规划与基线持平，但路径与脆弱走廊关系可解释。"
    return (
        f"baseline 穿过 {int(score_bundle.get('baseline_impacted_edges', 0))} 条受扰边，"
        f"planned 仅保留 {int(score_bundle.get('planned_impacted_edges', 0))} 条；"
        f"预计节省 {saving_s:.1f}s（{saving_ratio_pct:.1f}%），"
        f"路径重合率 {float(score_bundle.get('overlap_ratio') or 0.0):.2f}"
    )


def _build_fixed_actual_score_bundle(
    *,
    net,
    road_meta,
    scene_profile,
    scene_weights: dict[str, float],
    preset: dict[str, Any],
) -> dict[str, Any] | None:
    start_edge = str(preset.get("start_edge") or "").strip()
    end_edge = str(preset.get("end_edge") or "").strip()
    if not start_edge or not end_edge:
        return None

    baseline_path, baseline_freeflow_cost = free_flow_route(net, start_edge, end_edge)
    planned_path, planned_cost = dijkstra_route(net, start_edge, end_edge, scene_weights)
    baseline_path = _append_target_edge_if_connected(net, baseline_path, end_edge)
    planned_path = _append_target_edge_if_connected(net, planned_path, end_edge)
    if not baseline_path or not planned_path:
        return None
    if baseline_path[0] != start_edge or planned_path[0] != start_edge:
        return None
    if baseline_path[-1] != end_edge or planned_path[-1] != end_edge:
        return None

    impacted = _impacted_edge_sets(scene_profile)
    baseline_impacted = len(set(baseline_path) & impacted["all"])
    planned_impacted = len(set(planned_path) & impacted["all"])
    affected_edges = sorted(set(baseline_path) & impacted["all"])
    if not affected_edges:
        affected_edges = sorted(impacted["all"])

    baseline_scene_time = _estimate_path_time(net, baseline_path, scene_weights)
    planned_scene_time = _estimate_path_time(net, planned_path, scene_weights)
    baseline_freeflow_time = _estimate_path_time(net, baseline_path, free_flow=True)
    planned_freeflow_time = _estimate_path_time(net, planned_path, free_flow=True)
    overlap_ratio = _route_overlap_ratio(baseline_path, planned_path)

    baseline_stability, baseline_route_info = _safe_route_stability(net, start_edge, end_edge, baseline_path)
    planned_stability, planned_route_info = _safe_route_stability(net, start_edge, end_edge, planned_path)
    baseline_safe_route = _final_safe_route_edges(baseline_route_info)
    planned_safe_route = _final_safe_route_edges(planned_route_info)
    safe_route_overlap = _route_overlap_ratio(baseline_safe_route, planned_safe_route)

    actual_baseline_time_s = float(preset.get("actual_baseline_time_s") or 0.0)
    actual_planned_time_s = float(preset.get("actual_planned_time_s") or 0.0)
    actual_saving_seconds = float(preset.get("actual_saving_seconds") or 0.0)
    actual_saving_ratio = float(preset.get("actual_saving_ratio") or 0.0)
    gain_score = min(actual_saving_ratio, 1.0)
    overlap_score = 1.0 - overlap_ratio
    safe_route_divergence_score = 1.0 - safe_route_overlap
    raw_avoidance_score = min(max(baseline_impacted - planned_impacted, 0) / max(baseline_impacted, 1), 1.0)
    total_score = (
        gain_score * 0.42
        + raw_avoidance_score * 0.24
        + overlap_score * 0.18
        + safe_route_divergence_score * 0.16
    )

    return {
        "scene_profile": scene_profile.scene_name,
        "start_edge": start_edge,
        "end_edge": end_edge,
        "baseline_path": baseline_path,
        "planned_path": planned_path,
        "baseline_overlap_ratio": overlap_ratio,
        "overlap_ratio": overlap_ratio,
        "baseline_scene_time_s": actual_baseline_time_s,
        "planned_scene_time_s": actual_planned_time_s,
        "baseline_estimated_scene_time_s": baseline_scene_time,
        "planned_estimated_scene_time_s": planned_scene_time,
        "baseline_freeflow_time_s": baseline_freeflow_time,
        "planned_freeflow_time_s": planned_freeflow_time,
        "baseline_selection_cost_s": baseline_freeflow_time if math.isfinite(float(baseline_freeflow_cost)) else None,
        "planned_selection_cost_s": planned_scene_time if math.isfinite(float(planned_cost)) else None,
        "baseline_impacted_edges": baseline_impacted,
        "planned_impacted_edges": planned_impacted,
        "affected_edges": affected_edges,
        "baseline_turns": _path_turns(road_meta, baseline_path),
        "planned_turns": _path_turns(road_meta, planned_path),
        "estimated_delta_s": actual_saving_seconds,
        "saving_seconds": actual_saving_seconds,
        "saving_ratio": actual_saving_ratio,
        "positive_gain": bool(actual_saving_seconds > 1e-9),
        "saving_ratio_pct": actual_saving_ratio * 100.0,
        "baseline_scene_penalty_s": max(actual_baseline_time_s - baseline_freeflow_time, 0.0),
        "raw_avoidance_score": raw_avoidance_score,
        "reroute_ready": bool(scene_profile.reroute_enabled and baseline_impacted > 0),
        "baseline_route_validation": baseline_route_info,
        "planned_route_validation": planned_route_info,
        "baseline_safe_route": baseline_safe_route,
        "planned_safe_route": planned_safe_route,
        "safe_route_overlap_ratio": safe_route_overlap,
        "baseline_stability": baseline_stability,
        "planned_stability": planned_stability,
        "score": total_score,
    }


def _score_candidate(
    *,
    net,
    road_meta,
    scene_profile,
    scene_weights: dict[str, float],
    start_edge: str,
    end_edge: str,
):
    baseline_path, baseline_freeflow_cost = free_flow_route(net, start_edge, end_edge)
    planned_path, planned_cost = dijkstra_route(net, start_edge, end_edge, scene_weights)
    baseline_path = _append_target_edge_if_connected(net, baseline_path, end_edge)
    planned_path = _append_target_edge_if_connected(net, planned_path, end_edge)
    if not baseline_path or not planned_path:
        return None
    if baseline_path[0] != start_edge or planned_path[0] != start_edge:
        return None
    if baseline_path[-1] != end_edge or planned_path[-1] != end_edge:
        return None

    baseline_turns = _path_turns(road_meta, baseline_path)
    planned_turns = _path_turns(road_meta, planned_path)
    if len(baseline_path) < 5 or len(planned_path) < 5:
        return None
    if _path_unique_roads(road_meta, baseline_path) <= 1:
        return None
    if _same_main_corridor_direct(road_meta, baseline_path):
        return None
    if baseline_turns <= 1 and planned_turns <= 1:
        return None

    impacted = _impacted_edge_sets(scene_profile)
    baseline_impacted = len(set(baseline_path) & impacted["all"])
    planned_impacted = len(set(planned_path) & impacted["all"])
    if baseline_impacted <= 0:
        return None
    if len(impacted["all"]) >= 5 and baseline_impacted < 2:
        return None

    overlap_ratio = _route_overlap_ratio(baseline_path, planned_path)
    if overlap_ratio > 0.72:
        return None

    baseline_scene_time = _estimate_path_time(net, baseline_path, scene_weights)
    planned_scene_time = _estimate_path_time(net, planned_path, scene_weights)
    baseline_freeflow_time = _estimate_path_time(net, baseline_path, free_flow=True)
    planned_freeflow_time = _estimate_path_time(net, planned_path, free_flow=True)
    delta_s = baseline_scene_time - planned_scene_time
    if delta_s <= 1e-9:
        return None
    saving_ratio = delta_s / max(baseline_scene_time, 1.0)
    if saving_ratio < 0.05:
        return None
    scene_penalty_s = baseline_scene_time - baseline_freeflow_time
    if scene_penalty_s < 45.0:
        return None

    baseline_stability, baseline_route_info = _safe_route_stability(net, start_edge, end_edge, baseline_path)
    planned_stability, planned_route_info = _safe_route_stability(net, start_edge, end_edge, planned_path)
    if not bool(baseline_route_info.get("ok")) or not bool(planned_route_info.get("ok")):
        return None
    baseline_safe_route = _final_safe_route_edges(baseline_route_info)
    planned_safe_route = _final_safe_route_edges(planned_route_info)
    if not baseline_safe_route or not planned_safe_route:
        return None
    safe_route_overlap = _route_overlap_ratio(baseline_safe_route, planned_safe_route)
    if baseline_safe_route == planned_safe_route:
        return None
    if safe_route_overlap > 0.94:
        return None
    if safe_route_overlap > 0.90 and overlap_ratio > 0.70:
        return None
    if set(planned_path) == set(baseline_path):
        return None
    if planned_impacted >= baseline_impacted:
        return None
    if planned_impacted > 1 and (baseline_impacted - planned_impacted) < 1:
        return None
    if planned_impacted > 0 and delta_s < 60.0:
        return None
    if abs(_path_length_m(net, baseline_path) - _path_length_m(net, planned_path)) < 60.0 and overlap_ratio > 0.80:
        return None

    length_score = max(0.0, 1.0 - abs(len(planned_path) - 8) / 8.0)
    overlap_score = 1.0 - overlap_ratio
    gain_score = min(delta_s / max(baseline_scene_time, 1.0), 1.0)
    hazard_score = min((baseline_impacted + max(baseline_impacted - planned_impacted, 0)) / 4.0, 1.0)
    raw_avoidance_score = min(max(baseline_impacted - planned_impacted, 0) / max(baseline_impacted, 1), 1.0)
    directional_score = min(_directional_asymmetry_score(road_meta, scene_weights) / 6.0, 1.0)
    variance_score = min(
        (
            abs(len(planned_path) - len(baseline_path))
            + abs(planned_turns - baseline_turns)
            + abs(planned_scene_time - baseline_scene_time) / 40.0
        ) / 5.0,
        1.0,
    )
    reroute_ready = bool(scene_profile.reroute_enabled and baseline_impacted > 0)
    reroute_score = 1.0 if reroute_ready else 0.0
    stability_score = min(baseline_stability, planned_stability)
    safe_route_divergence_score = 1.0 - safe_route_overlap

    total_score = (
        gain_score * 0.30
        + raw_avoidance_score * 0.22
        + overlap_score * 0.18
        + safe_route_divergence_score * 0.14
        + hazard_score * 0.08
        + variance_score * 0.04
        + length_score * 0.02
        + reroute_score * 0.01
        + stability_score * 0.01
    )
    if scene_profile.scene_name == "directional_asymmetry":
        total_score += directional_score * 0.10
    return {
        "scene_profile": scene_profile.scene_name,
        "start_edge": start_edge,
        "end_edge": end_edge,
        "baseline_path": baseline_path,
        "planned_path": planned_path,
        "baseline_overlap_ratio": overlap_ratio,
        "overlap_ratio": overlap_ratio,
        "baseline_scene_time_s": baseline_scene_time,
        "planned_scene_time_s": planned_scene_time,
        "baseline_freeflow_time_s": baseline_freeflow_time,
        "planned_freeflow_time_s": planned_freeflow_time,
        "baseline_selection_cost_s": baseline_freeflow_time if math.isfinite(float(baseline_freeflow_cost)) else None,
        "planned_selection_cost_s": planned_scene_time if math.isfinite(float(planned_cost)) else None,
        "baseline_impacted_edges": baseline_impacted,
        "planned_impacted_edges": planned_impacted,
        "affected_edges": sorted(set(baseline_path) & impacted["all"]),
        "baseline_turns": baseline_turns,
        "planned_turns": planned_turns,
        "estimated_delta_s": delta_s,
        "positive_gain": bool(delta_s > 1e-9),
        "saving_ratio_pct": saving_ratio * 100.0,
        "baseline_scene_penalty_s": scene_penalty_s,
        "raw_avoidance_score": raw_avoidance_score,
        "reroute_ready": reroute_ready,
        "baseline_route_validation": baseline_route_info,
        "planned_route_validation": planned_route_info,
        "baseline_safe_route": baseline_safe_route,
        "planned_safe_route": planned_safe_route,
        "safe_route_overlap_ratio": safe_route_overlap,
        "score": total_score,
    }


def _build_structured_showcase_case(
    *,
    showcase_name: str,
    scene_profile,
    scene_weight_layers: dict[str, Any],
    score_bundle: dict[str, Any],
) -> dict[str, Any]:
    preset_spec = dict(SHOWCASE_SCENE_SPECS.get(showcase_name) or {})
    display_reason = str(preset_spec.get("display_reason") or _reason_text(score_bundle))
    expected_difference = str(
        preset_spec.get("expected_difference") or _profile_expected_difference(scene_profile.scene_name, score_bundle)
    )
    affected_edges = list(score_bundle.get("affected_edges") or [])
    if not affected_edges:
        affected_edges = sorted(_impacted_edge_sets(scene_profile)["all"])
    return {
        "showcase_name": showcase_name,
        "scene_name": showcase_name,
        "scene_profile": scene_profile.scene_name,
        "demo_label": preset_spec.get("demo_label", showcase_name),
        "showcase_badge": str(preset_spec.get("showcase_badge") or "Positive Gain Showcase"),
        "showcase_stage": str(preset_spec.get("showcase_stage") or "production"),
        "showcase_group": str(preset_spec.get("showcase_group") or "production"),
        "availability": str(preset_spec.get("availability") or "production_default"),
        "status_note": str(preset_spec.get("status_note") or ""),
        "start_edge": score_bundle["start_edge"],
        "end_edge": score_bundle["end_edge"],
        "affected_edges": affected_edges,
        "preferred_baseline_type": "free_flow_shortest_path",
        "display_reason": display_reason,
        "expected_difference": expected_difference,
        "baseline_path": list(score_bundle.get("baseline_path") or []),
        "planned_path": list(score_bundle.get("planned_path") or []),
        "baseline_safe_route": list(score_bundle.get("baseline_safe_route") or []),
        "planned_safe_route": list(score_bundle.get("planned_safe_route") or []),
        "baseline_time_s": float(score_bundle.get("baseline_scene_time_s") or 0.0),
        "planned_time_s": float(score_bundle.get("planned_scene_time_s") or 0.0),
        "saving_s": float(score_bundle.get("estimated_delta_s") or 0.0),
        "saving_seconds": float(score_bundle.get("saving_seconds") or score_bundle.get("estimated_delta_s") or 0.0),
        "saving_ratio": float(score_bundle.get("saving_ratio") or 0.0),
        "saving_ratio_pct": float(score_bundle.get("saving_ratio_pct") or 0.0),
        "baseline_overlap_ratio": float(score_bundle.get("overlap_ratio") or 0.0),
        "positive_gain": bool(score_bundle.get("positive_gain")),
        "score": float(score_bundle.get("score") or 0.0),
        "scene_weights": dict(scene_weight_layers.get("weights") or {}),
        "scene_weight_layers": dict(scene_weight_layers or {}),
        "debug": dict(score_bundle or {}),
    }


@lru_cache(maxsize=8)
def select_showcase_scenarios(net_path: str, showcase_names: tuple[str, ...] | None = None):
    if not showcase_names:
        showcase_names = tuple(SHOWCASE_SCENE_SPECS.keys())
    filtered_showcase_names = tuple(
        showcase_name
        for showcase_name in showcase_names
        if showcase_name in SHOWCASE_SCENE_SPECS
    )
    if not filtered_showcase_names:
        return {}

    net = sumolib.net.readNet(net_path)
    road_meta = load_roadnet_meta(net_path)
    edge_ids = list(road_meta.edge_ids)
    results = {}

    for showcase_name in filtered_showcase_names:
        showcase_spec = dict(SHOWCASE_SCENE_SPECS.get(showcase_name) or {})
        profile_name = str(showcase_spec.get("scene_profile") or "").strip()
        if not profile_name:
            continue
        scene_profile = get_scene_profile(profile_name)
        scene_weight_layers = build_environment_weight_layers(
            edge_ids,
            scene_profile,
            enable_spillover=True,
        )
        scene_weights = dict(scene_weight_layers.get("weights") or {})
        score_bundle = _build_fixed_actual_score_bundle(
            net=net,
            road_meta=road_meta,
            scene_profile=scene_profile,
            scene_weights=scene_weights,
            preset=showcase_spec,
        )
        if score_bundle is None:
            continue
        results[showcase_name] = _build_structured_showcase_case(
            showcase_name=showcase_name,
            scene_profile=scene_profile,
            scene_weight_layers=scene_weight_layers,
            score_bundle=score_bundle,
        )

    return results


def list_calibration_showcase_presets() -> dict[str, dict[str, Any]]:
    return {
        showcase_name: {
            "showcase_name": showcase_name,
            **dict(spec),
        }
        for showcase_name, spec in CALIBRATION_EXPERIMENTAL_SHOWCASE_PRESETS.items()
    }


def select_best_showcase_scenario(net_path: str, showcase_names: tuple[str, ...] | None = None):
    showcase_cases = select_showcase_scenarios(net_path, showcase_names)
    if not showcase_cases:
        return None
    return max(
        showcase_cases.values(),
        key=lambda item: (
            float(item.get("saving_s") or 0.0),
            1.0 - float(item.get("baseline_overlap_ratio") or 0.0),
            1.0 - float((item.get("debug") or {}).get("safe_route_overlap_ratio") or 0.0),
            float(item.get("saving_ratio_pct") or 0.0),
            float(item.get("score") or 0.0),
        ),
    )


@lru_cache(maxsize=8)
def select_demo_scenarios(net_path: str, scene_profile_names: tuple[str, ...] | None = None):
    profile_filter = set(scene_profile_names or PROFILE_DEMO_LABELS.keys())
    structured_cases = select_showcase_scenarios(net_path)
    results = {}
    for showcase_name, item in structured_cases.items():
        profile_name = str(item.get("scene_profile") or "").strip()
        if profile_name not in profile_filter:
            continue
        results[profile_name] = {
            "demo_label": item.get("demo_label"),
            "start_edge": item.get("start_edge"),
            "end_edge": item.get("end_edge"),
            "reason": item.get("display_reason"),
            "expected_difference": item.get("expected_difference"),
            "score": item.get("score"),
            "debug": dict(item.get("debug") or {}),
            "showcase_name": showcase_name,
            "affected_edges": list(item.get("affected_edges") or []),
            "baseline_path": list(item.get("baseline_path") or []),
            "planned_path": list(item.get("planned_path") or []),
            "baseline_time_s": item.get("baseline_time_s"),
            "planned_time_s": item.get("planned_time_s"),
            "saving_s": item.get("saving_s"),
            "saving_ratio_pct": item.get("saving_ratio_pct"),
            "positive_gain": item.get("positive_gain"),
            "scene_weights": dict(item.get("scene_weights") or {}),
            "scene_weight_layers": dict(item.get("scene_weight_layers") or {}),
        }
    return results
