from __future__ import annotations

from typing import Any

import pandas as pd

from utils_df_safe import flatten_json_record, make_arrow_safe


BYPASS_SOURCES = {"scene_aware_shortest_path", "showcase_bypass"}


def _safe_float(value, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return parsed


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _planned_path_source(result: dict[str, Any]) -> str:
    source = str(result.get("planned_path_source") or "").strip()
    if source:
        return source
    path_algo = str(result.get("path_algo") or "").lower()
    if result.get("showcase_structured_mode") or "scene-aware shortest path" in path_algo:
        return "scene_aware_shortest_path"
    if result.get("fallback_used"):
        return "fallback"
    if "rule" in path_algo:
        return "rule"
    return "parser"


def _sparse_parse(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("sparse_parse")
    return value if isinstance(value, dict) else {}


def _spillover_stats(result: dict[str, Any]) -> dict[str, Any]:
    direct = result.get("spillover_stats")
    if isinstance(direct, dict):
        return direct
    layers = result.get("scene_weight_layers")
    if isinstance(layers, dict) and isinstance(layers.get("stats"), dict):
        return dict(layers["stats"])
    return {}


def build_model_parsed_constraints(result: dict[str, Any]) -> dict[str, Any]:
    sparse_parse = _sparse_parse(result)
    planned_path_source = _planned_path_source(result)
    sparse_anchor_count = _safe_int(
        result.get("anchor_count", sparse_parse.get("anchor_count", len(sparse_parse.get("anchors") or [])))
    )
    parsed_changed_edges = _safe_int(
        result.get("parsed_edges", result.get("parsed_changed_edges", len(sparse_parse.get("mapped_weights") or {})))
    )
    total_edges = _safe_int(result.get("total_edges", result.get("edge_total", 96)), 96)
    if planned_path_source in BYPASS_SOURCES:
        parser_source = "showcase_bypass"
        model_contribution = "Bypassed"
    elif result.get("fallback_used"):
        parser_source = "fallback"
        model_contribution = "Fallback / No Anchors"
    elif "rule" in str(result.get("path_algo") or "").lower():
        parser_source = "rule"
        model_contribution = "Fallback / No Anchors"
    else:
        parser_source = str(result.get("parser_source") or "parser")
        model_contribution = "Active" if sparse_anchor_count > 0 else "Fallback / No Anchors"
    return {
        "parser_source": parser_source,
        "model_contribution": model_contribution,
        "sparse_anchor_count": sparse_anchor_count,
        "parsed_changed_edges": parsed_changed_edges,
        "total_edges": total_edges,
        "parse_confidence": _safe_float(result.get("parse_confidence", sparse_parse.get("parse_confidence")), 0.0),
        "planned_path_source": planned_path_source,
    }


def build_scene_runtime_constraints(result: dict[str, Any]) -> dict[str, Any]:
    stats = _spillover_stats(result)
    raw_incident_edges = _safe_int(stats.get("raw_incident_count"), 0)
    if raw_incident_edges <= 0:
        raw_incident_edges = len(result.get("showcase_affected_edges") or [])
    spillover_edges = _safe_int(stats.get("spillover_total_count"), 0)
    runtime_spillover_enabled = (
        str(result.get("scene_mode") or result.get("workspace_scene_mode") or "").strip() == "showcase"
        and (spillover_edges > 0 or bool(result.get("sumo_runtime_spillover_summary")))
    )
    return {
        "scene_profile": str(result.get("scene_profile") or "unknown"),
        "raw_incident_edges": raw_incident_edges,
        "spillover_edges": spillover_edges,
        "runtime_spillover_enabled": bool(runtime_spillover_enabled),
        "baseline_affected_overlap": _safe_float(result.get("baseline_affected_edge_overlap_ratio")),
        "planned_affected_overlap": _safe_float(result.get("planned_affected_edge_overlap_ratio", result.get("affected_edge_overlap_ratio"))),
        "baseline_incident_overlap": _safe_float(result.get("baseline_incident_edge_overlap_ratio")),
        "planned_incident_overlap": _safe_float(result.get("planned_incident_edge_overlap_ratio")),
    }


def build_route_gain_summary(result: dict[str, Any]) -> dict[str, Any]:
    model = build_model_parsed_constraints(result)
    baseline_travel_time_s = _safe_float(
        result.get("baseline_sumo_travel_time", result.get("baseline_sumo_travel_time_s", result.get("baseline_estimated_time")))
    )
    planned_travel_time_s = _safe_float(
        result.get("planned_sumo_travel_time", result.get("planned_sumo_travel_time_s", result.get("planned_estimated_time", result.get("travel_time"))))
    )
    time_saving_s = None
    saving_ratio_pct = None
    if baseline_travel_time_s is not None and planned_travel_time_s is not None:
        time_saving_s = baseline_travel_time_s - planned_travel_time_s
        saving_ratio_pct = time_saving_s / max(baseline_travel_time_s, 1.0) * 100.0
    if model["sparse_anchor_count"] == 0 and model["planned_path_source"] in BYPASS_SOURCES:
        gain_title = "Showcase Route Gain"
        gain_attribution = "Scene-aware route selection under injected SUMO profile; not parser-anchor gain."
    else:
        gain_title = "Sparse-LoRA-v2 Route Gain"
        gain_attribution = "Gain produced by parsed sparse anchors and route planning."
    return {
        "gain_title": gain_title,
        "baseline_travel_time_s": baseline_travel_time_s,
        "planned_travel_time_s": planned_travel_time_s,
        "time_saving_s": time_saving_s,
        "saving_ratio_pct": saving_ratio_pct,
        "gain_attribution": gain_attribution,
    }


def build_debug_table(result: dict[str, Any]) -> pd.DataFrame:
    return make_arrow_safe(pd.DataFrame([flatten_json_record(result)]))
