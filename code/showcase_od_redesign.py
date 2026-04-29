from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import sumolib

from config import ConfigManager
from demo_scene_selector import (
    _append_target_edge_if_connected,
    _candidate_edge_pool,
    _final_safe_route_edges,
    _impacted_edge_sets,
    _path_length_m,
    _path_turns,
    _path_unique_roads,
    _route_overlap_ratio,
)
from roadnet_meta import load_roadnet_meta
from routing import dijkstra_route, free_flow_route
from sim_scene_profiles import build_environment_weight_layers, get_scene_profile
from sumo_runner import build_safe_vehicle_route


OUT_PATH = Path("/root/autodl-tmp/results/ui/showcase_od_redesign_report.json")
SUMO_PATH = Path("/root/autodl-tmp/results/ui/showcase_od_redesign_sumo_results.json")

SHOWCASE_TO_SCENE = {
    "blockage_detour_showcase": "core_blockage",
    "directional_asymmetry_showcase": "directional_asymmetry",
    "propagation_range_showcase": "propagation_range",
    "compound_disaster_showcase": "compound_disaster",
}

SCENE_TO_SHOWCASE = {v: k for k, v in SHOWCASE_TO_SCENE.items()}
TARGET_SCENES = ("propagation_range", "compound_disaster")
SCENE_MANUAL_OD_SEEDS = {
    "propagation_range": (
        ("C5R1_S", "C4R2_S"),
        ("C5R1_S", "C4R3_N"),
        ("C5R2_S", "C4R3_S"),
        ("C0R0_N", "C2R3_S"),
        ("C0R1_S", "C4R3_S"),
    ),
    "compound_disaster": (
        ("R4C0_E", "C0R0_S"),
        ("C0R2_N", "C1R0_S"),
        ("C0R2_S", "C2R3_S"),
        ("R4C0_W", "C1R0_S"),
        ("R4C0_E", "C4R2_S"),
    ),
}


def _near_boundary_pool(road_meta) -> list[str]:
    candidates = []
    max_h_col = max(int(m.get("col", 0)) for m in road_meta.edge_meta.values() if m.get("axis") == "H")
    max_v_row = max(int(m.get("row", 0)) for m in road_meta.edge_meta.values() if m.get("axis") == "V")
    for edge_id in road_meta.edge_ids:
        meta = road_meta.edge_meta.get(edge_id, {})
        axis = meta.get("axis")
        row = int(meta.get("row", 0))
        col = int(meta.get("col", 0))
        if axis == "H":
            near_boundary = col in {0, 1, max_h_col - 1, max_h_col}
        else:
            near_boundary = row in {0, 1, max_v_row - 1, max_v_row}
        if near_boundary:
            candidates.append(edge_id)
    return sorted(dict.fromkeys(candidates))


def _edge_span(meta: dict[str, Any], path: list[str]) -> tuple[int, int]:
    rows = []
    cols = []
    for edge_id in path:
        item = meta.get(edge_id, {})
        rows.append(int(item.get("row", 0)))
        cols.append(int(item.get("col", 0)))
    if not rows or not cols:
        return 0, 0
    return max(rows) - min(rows), max(cols) - min(cols)


def _path_axis_indices(road_meta, path: list[str], axis: str) -> set[int]:
    values: set[int] = set()
    for edge_id in path:
        meta = road_meta.edge_meta.get(edge_id, {})
        if meta.get("axis") != axis:
            continue
        key = "col" if axis == "V" else "row"
        values.add(int(meta.get(key, 0)))
    return values


def _path_hits_any(path: list[str], edge_ids: set[str]) -> int:
    return len(set(path) & set(edge_ids))


def _scene_specific_ok(scene_name: str, bundle: dict[str, Any], road_meta) -> tuple[bool, str]:
    baseline_path = list(bundle["baseline_logical_path"])
    planned_path = list(bundle["planned_logical_path"])
    baseline_len = len(baseline_path)
    planned_len = len(planned_path)
    logical_overlap = float(bundle["logical_overlap_ratio"])
    safe_overlap = float(bundle["safe_route_overlap_ratio"])
    baseline_impacted = int(bundle["baseline_impacted_edges"])
    planned_impacted = int(bundle["planned_impacted_edges"])
    delta = float(bundle["estimated_saving_seconds"])
    b_row_span, b_col_span = _edge_span(road_meta.edge_meta, baseline_path)
    p_row_span, p_col_span = _edge_span(road_meta.edge_meta, planned_path)
    baseline_vcols = _path_axis_indices(road_meta, baseline_path, "V")
    planned_vcols = _path_axis_indices(road_meta, planned_path, "V")
    baseline_hrows = _path_axis_indices(road_meta, baseline_path, "H")
    planned_hrows = _path_axis_indices(road_meta, planned_path, "H")

    if baseline_impacted <= 0:
        return False, "baseline_not_crossing_main_disturbed_corridor"
    if delta <= -20.0:
        return False, "planned_not_faster_offline"
    if logical_overlap > 0.72:
        return False, "logical_overlap_too_high"
    if safe_overlap > 0.85:
        return False, "safe_overlap_too_high"
    if max(baseline_len, planned_len) > 9:
        return False, "route_too_long_for_showcase"
    if planned_len - baseline_len > 2:
        return False, "planned_naturally_too_long"

    if scene_name == "core_blockage":
        if max(baseline_len, planned_len) > 8:
            return False, "core_blockage_too_long"
        if baseline_impacted < 2:
            return False, "core_blockage_baseline_not_deep_enough"
        if planned_impacted > 0:
            return False, "core_blockage_planned_still_hits_disturbance"
    elif scene_name == "directional_asymmetry":
        if baseline_impacted < 3:
            return False, "directional_baseline_not_long_enough_on_asymmetric_corridor"
        if planned_impacted > 1:
            return False, "directional_planned_still_hits_asymmetric_band"
        if b_col_span < 3 and b_row_span < 3:
            return False, "directional_baseline_not_traversing_enough"
        if planned_len > baseline_len + 1:
            return False, "directional_planned_detour_too_long"
    elif scene_name == "propagation_range":
        if baseline_impacted < 1:
            return False, "propagation_baseline_not_deep_enough"
        if planned_impacted > 0:
            return False, "propagation_planned_still_hits_spillover_band"
        if max(baseline_len, planned_len) > 8:
            return False, "propagation_too_long"
        if abs(baseline_len - planned_len) > 2:
            return False, "propagation_length_gap_too_large"
        if logical_overlap > 0.60:
            return False, "propagation_overlap_too_high"
        if safe_overlap > 0.72:
            return False, "propagation_safe_overlap_too_high"
        if baseline_vcols == planned_vcols and baseline_hrows == planned_hrows:
            return False, "propagation_not_switching_corridor"
        if b_col_span < 1 and p_col_span < 1 and b_row_span < 1 and p_row_span < 1:
            return False, "propagation_not_structurally_distinct"
    elif scene_name == "compound_disaster":
        fragile_corridor = {"R3C0_E", "R3C1_E", "R3C2_E", "C1R1_S", "C1R2_S", "C2R1_N", "C2R2_N", "C2R1_S", "C2R2_S"}
        baseline_fragile_hits = _path_hits_any(baseline_path, fragile_corridor)
        planned_fragile_hits = _path_hits_any(planned_path, fragile_corridor)
        if max(baseline_len, planned_len) > 7:
            return False, "compound_too_long"
        if baseline_impacted < 2 or baseline_fragile_hits < 2:
            return False, "compound_baseline_not_deep_enough"
        if planned_impacted > 1:
            return False, "compound_planned_too_fragile"
        if planned_fragile_hits > 0:
            return False, "compound_planned_still_uses_fragile_corridor"
        if logical_overlap > 0.55:
            return False, "compound_overlap_too_high"
        if safe_overlap > 0.70:
            return False, "compound_safe_overlap_too_high"
        if delta <= -10.0:
            return False, "compound_offline_gap_too_negative"

    return True, "ok"


def _offline_score(scene_name: str, bundle: dict[str, Any], road_meta) -> float:
    baseline_path = list(bundle["baseline_logical_path"])
    planned_path = list(bundle["planned_logical_path"])
    logical_overlap = float(bundle["logical_overlap_ratio"])
    safe_overlap = float(bundle["safe_route_overlap_ratio"])
    delta = float(bundle["estimated_saving_seconds"])
    baseline_impacted = int(bundle["baseline_impacted_edges"])
    planned_impacted = int(bundle["planned_impacted_edges"])
    baseline_len = len(baseline_path)
    planned_len = len(planned_path)
    b_row_span, b_col_span = _edge_span(road_meta.edge_meta, baseline_path)
    p_row_span, p_col_span = _edge_span(road_meta.edge_meta, planned_path)
    structure_span = max(b_row_span + b_col_span, p_row_span + p_col_span)

    score = 0.0
    score += min(delta / 180.0, 8.0)
    score += (baseline_impacted - planned_impacted) * 1.8
    score += (1.0 - logical_overlap) * 3.0
    score += (1.0 - safe_overlap) * 2.5
    score += min(structure_span, 6) * 0.4
    score -= abs(baseline_len - 7) * 0.25
    score -= abs(planned_len - baseline_len) * 0.6

    if scene_name == "directional_asymmetry":
        score += baseline_impacted * 0.8
        score -= max(planned_len - baseline_len, 0) * 1.0
    elif scene_name == "propagation_range":
        baseline_vcols = _path_axis_indices(road_meta, baseline_path, "V")
        planned_vcols = _path_axis_indices(road_meta, planned_path, "V")
        if baseline_vcols != planned_vcols:
            score += 2.0
        if delta < 0.0:
            score -= abs(delta) / 8.0
        score -= abs(baseline_len - planned_len) * 0.4
    elif scene_name in {"core_blockage", "compound_disaster"}:
        score -= max(max(baseline_len, planned_len) - 8, 0) * 2.0
    if scene_name == "compound_disaster":
        fragile_corridor = {"R3C0_E", "R3C1_E", "R3C2_E", "C1R1_S", "C1R2_S", "C2R1_N", "C2R2_N", "C2R1_S", "C2R2_S"}
        score += _path_hits_any(baseline_path, fragile_corridor) * 0.9
        score -= _path_hits_any(planned_path, fragile_corridor) * 2.2
        if delta < 0.0:
            score -= abs(delta) / 5.0
    return float(score)


def build_offline_candidate(scene_name: str, start_edge: str, end_edge: str, net, road_meta, edge_ids) -> dict[str, Any] | None:
    profile = get_scene_profile(scene_name)
    layers = build_environment_weight_layers(edge_ids, profile, enable_spillover=True)
    weights = dict(layers.get("weights") or {})
    baseline_path, baseline_freeflow_cost = free_flow_route(net, start_edge, end_edge)
    planned_path, planned_cost = dijkstra_route(net, start_edge, end_edge, weights)
    baseline_path = _append_target_edge_if_connected(net, baseline_path, end_edge)
    planned_path = _append_target_edge_if_connected(net, planned_path, end_edge)
    if not baseline_path or not planned_path:
        return None
    if baseline_path[0] != start_edge or planned_path[0] != start_edge:
        return None
    if baseline_path[-1] != end_edge or planned_path[-1] != end_edge:
        return None
    if len(baseline_path) < 5 or len(planned_path) < 5:
        return None
    if _path_unique_roads(road_meta, baseline_path) <= 1:
        return None

    impacted = _impacted_edge_sets(profile)
    baseline_impacted = len(set(baseline_path) & impacted["all"])
    planned_impacted = len(set(planned_path) & impacted["all"])
    logical_overlap = _route_overlap_ratio(baseline_path, planned_path)

    baseline_est = float(baseline_freeflow_cost)
    planned_est = float(planned_cost)
    if planned_est <= 0 or baseline_est <= 0:
        return None
    est_saving = baseline_est - planned_est

    baseline_safe = build_safe_vehicle_route(
        net,
        original_start_edge=start_edge,
        original_end_edge=end_edge,
        original_route_edges=tuple(baseline_path),
    )
    planned_safe = build_safe_vehicle_route(
        net,
        original_start_edge=start_edge,
        original_end_edge=end_edge,
        original_route_edges=tuple(planned_path),
    )
    if not baseline_safe.get("ok") or not planned_safe.get("ok"):
        return None
    baseline_safe_route = _final_safe_route_edges(baseline_safe)
    planned_safe_route = _final_safe_route_edges(planned_safe)
    if not baseline_safe_route or not planned_safe_route:
        return None
    safe_overlap = _route_overlap_ratio(baseline_safe_route, planned_safe_route)

    bundle = {
        "scene_profile": scene_name,
        "start_edge": start_edge,
        "end_edge": end_edge,
        "baseline_logical_path": baseline_path,
        "planned_logical_path": planned_path,
        "baseline_safe_route": baseline_safe_route,
        "planned_safe_route": planned_safe_route,
        "logical_overlap_ratio": logical_overlap,
        "safe_route_overlap_ratio": safe_overlap,
        "baseline_impacted_edges": baseline_impacted,
        "planned_impacted_edges": planned_impacted,
        "whether_baseline_really_crosses_main_disturbed_corridor": bool(baseline_impacted > 0),
        "estimated_baseline_time": baseline_est,
        "estimated_planned_time": planned_est,
        "estimated_saving_seconds": est_saving,
        "baseline_route_length_edges": len(baseline_path),
        "planned_route_length_edges": len(planned_path),
        "baseline_route_length_m": _path_length_m(net, baseline_path),
        "planned_route_length_m": _path_length_m(net, planned_path),
        "baseline_turns": _path_turns(road_meta, baseline_path),
        "planned_turns": _path_turns(road_meta, planned_path),
        "alternate_corridor_exists": bool(logical_overlap < 0.8 and set(baseline_path) != set(planned_path)),
        "runtime_layers": layers,
    }
    ok, reason = _scene_specific_ok(scene_name, bundle, road_meta)
    if not ok:
        return None
    bundle["offline_score"] = _offline_score(scene_name, bundle, road_meta)
    bundle["screen_reason"] = reason
    return bundle


def rank_offline_candidates(scene_name: str, limit: int = 8) -> list[dict[str, Any]]:
    cfg = ConfigManager().config
    net = sumolib.net.readNet(cfg["SUMO_NET_PATH"])
    road_meta = load_roadnet_meta(cfg["SUMO_NET_PATH"])
    edge_ids = list(road_meta.edge_ids)
    candidate_edges = _near_boundary_pool(road_meta)
    baseline_pool = set(_candidate_edge_pool(road_meta))
    rows = []
    manual_pairs = set(tuple(item) for item in SCENE_MANUAL_OD_SEEDS.get(scene_name, ()))
    checked_pairs: set[tuple[str, str]] = set()

    for start_edge, end_edge in manual_pairs:
        checked_pairs.add((start_edge, end_edge))
        bundle = build_offline_candidate(scene_name, start_edge, end_edge, net, road_meta, edge_ids)
        if not bundle:
            continue
        bundle["start_in_boundary_pool"] = start_edge in baseline_pool
        bundle["end_in_boundary_pool"] = end_edge in baseline_pool
        bundle["manual_seed"] = True
        if scene_name == "propagation_range":
            bundle["offline_score"] = float(bundle["offline_score"]) + 2.5
        rows.append(bundle)

    for start_edge in candidate_edges:
        for end_edge in candidate_edges:
            if start_edge == end_edge:
                continue
            if (start_edge, end_edge) in checked_pairs:
                continue
            bundle = build_offline_candidate(scene_name, start_edge, end_edge, net, road_meta, edge_ids)
            if not bundle:
                continue
            bundle["start_in_boundary_pool"] = start_edge in baseline_pool
            bundle["end_in_boundary_pool"] = end_edge in baseline_pool
            rows.append(bundle)
    rows.sort(
        key=lambda item: (
            float(item["offline_score"]),
            float(item["estimated_saving_seconds"]),
            int(item["baseline_impacted_edges"]) - int(item["planned_impacted_edges"]),
            1.0 - float(item["logical_overlap_ratio"]),
            1.0 - float(item["safe_route_overlap_ratio"]),
            1 if item.get("manual_seed") else 0,
        ),
        reverse=True,
    )
    dedup: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_paths: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for item in rows:
        pair = (item["start_edge"], item["end_edge"])
        path_sig = (tuple(item["baseline_logical_path"]), tuple(item["planned_logical_path"]))
        if pair in seen_pairs or path_sig in seen_paths:
            continue
        seen_pairs.add(pair)
        seen_paths.add(path_sig)
        dedup.append(item)
        if len(dedup) >= limit:
            break
    return dedup


def run_sumo_validation(candidate: dict[str, Any]) -> dict[str, Any]:
    from calibrate_showcase_runtime import run_single_route

    profile = get_scene_profile(candidate["scene_profile"])
    layers = dict(candidate["runtime_layers"] or {})
    planned = run_single_route(list(candidate["planned_logical_path"]), profile, layers, "showcase")
    baseline = run_single_route(list(candidate["baseline_logical_path"]), profile, layers, "showcase")
    planned_time = float(planned.get("travel_time_s") or 0.0)
    baseline_time = float(baseline.get("travel_time_s") or 0.0)
    return {
        "showcase_name": SCENE_TO_SHOWCASE[candidate["scene_profile"]],
        "scene_profile": candidate["scene_profile"],
        "start_edge": candidate["start_edge"],
        "end_edge": candidate["end_edge"],
        "baseline": baseline_time,
        "planned": planned_time,
        "saving_seconds": baseline_time - planned_time,
        "saving_ratio": (baseline_time - planned_time) / max(baseline_time, 1.0),
        "whether_positive_gain": bool((baseline_time - planned_time) > 1e-9),
        "baseline_ok": bool(baseline.get("ok")),
        "planned_ok": bool(planned.get("ok")),
        "baseline_final_runtime_route": list((baseline.get("route_debug") or {}).get("final_route_edges") or []),
        "planned_final_runtime_route": list((planned.get("route_debug") or {}).get("final_route_edges") or []),
        "final_runtime_overlap_ratio": _route_overlap_ratio(
            (baseline.get("route_debug") or {}).get("final_route_edges") or [],
            (planned.get("route_debug") or {}).get("final_route_edges") or [],
        ),
        "baseline_used_sumo_fallback": bool((baseline.get("route_debug") or {}).get("used_sumo_fallback")),
        "planned_used_sumo_fallback": bool((planned.get("route_debug") or {}).get("used_sumo_fallback")),
    }


def summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "scene_profile",
        "start_edge",
        "end_edge",
        "manual_seed",
        "logical_overlap_ratio",
        "safe_route_overlap_ratio",
        "whether_baseline_really_crosses_main_disturbed_corridor",
        "estimated_baseline_time",
        "estimated_planned_time",
        "estimated_saving_seconds",
        "baseline_route_length_edges",
        "planned_route_length_edges",
        "baseline_route_length_m",
        "planned_route_length_m",
        "baseline_impacted_edges",
        "planned_impacted_edges",
        "alternate_corridor_exists",
        "offline_score",
        "baseline_logical_path",
        "planned_logical_path",
        "baseline_safe_route",
        "planned_safe_route",
    )
    return {key: candidate.get(key) for key in keep}


def _pick_validation_candidates(scene_name: str, ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    def _append_if_new(item: dict[str, Any]) -> None:
        pair = (str(item.get("start_edge")), str(item.get("end_edge")))
        if pair in seen_pairs:
            return
        seen_pairs.add(pair)
        picked.append(item)

    manual_candidates = [item for item in ranked if item.get("manual_seed")]

    if scene_name == "propagation_range":
        for item in manual_candidates[:2]:
            _append_if_new(item)
        for item in ranked:
            if len(picked) >= 2:
                break
            _append_if_new(item)
        return picked[:2]

    if scene_name == "compound_disaster":
        preferred_pairs = (
            ("C0R2_N", "C1R0_S"),
            ("R4C0_E", "C0R0_S"),
        )
        for start_edge, end_edge in preferred_pairs:
            for item in ranked:
                if str(item.get("start_edge")) == start_edge and str(item.get("end_edge")) == end_edge:
                    _append_if_new(item)
                    break
        for item in ranked:
            if len(picked) >= 2:
                break
            _append_if_new(item)
        return picked[:2]

    return list(ranked[:2])


def main() -> None:
    report = {
        "offline_candidates": {},
        "sumo_validated_candidates": {},
        "sumo_results": [],
    }
    for scene in TARGET_SCENES:
        ranked = rank_offline_candidates(scene, limit=30)
        report["offline_candidates"][scene] = [summarize_candidate(item) for item in ranked]
        picked = _pick_validation_candidates(scene, ranked)
        report["sumo_validated_candidates"][scene] = [
            {"start_edge": item["start_edge"], "end_edge": item["end_edge"], "offline_score": item["offline_score"]}
            for item in picked
        ]
        for item in picked:
            result = run_sumo_validation(item)
            report["sumo_results"].append(result)
            SUMO_PATH.write_text(json.dumps(report["sumo_results"], ensure_ascii=False, indent=2))
        OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
