from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ui_schema import build_debug_table, build_model_parsed_constraints, build_scene_runtime_constraints
from utils_df_safe import flatten_json_record, make_arrow_safe


def _as_dict(value) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _kv_table(payload: dict[str, Any]) -> pd.DataFrame:
    return make_arrow_safe(pd.DataFrame([{"field": key, "value": value} for key, value in payload.items()]))


def _records_table(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return make_arrow_safe(value)
    if isinstance(value, list):
        if not value:
            return make_arrow_safe(pd.DataFrame(columns=["value"]))
        if all(isinstance(item, dict) for item in value):
            return make_arrow_safe(pd.DataFrame(value))
        return make_arrow_safe(pd.DataFrame({"value": value}))
    if isinstance(value, dict):
        return _kv_table(value)
    return make_arrow_safe(pd.DataFrame({"value": [] if value is None else [value]}))


def _show_dataframe(df: pd.DataFrame, *, key: str | None = None) -> None:
    if df.empty:
        st.caption("No data available")
    else:
        st.dataframe(make_arrow_safe(df), use_container_width=True, hide_index=True)


def render_debug_panel(result: dict[str, Any]) -> None:
    sparse_parse = _as_dict(result.get("sparse_parse"))
    scene_cfg = _as_dict(result.get("scene_profile_config"))
    scene_layers = _as_dict(result.get("scene_weight_layers"))
    runtime_spillover = _as_dict(result.get("sumo_runtime_spillover_summary"))
    model_schema = build_model_parsed_constraints(result)
    scene_schema = build_scene_runtime_constraints(result)

    with st.expander("Model / Parser Debug", expanded=False):
        _show_dataframe(
            _kv_table(
                {
                    "parser_source": model_schema["parser_source"],
                    "planned_path_source": model_schema["planned_path_source"],
                    "raw_model_output": result.get("raw_output") or result.get("raw_model_output") or result.get("llm_output") or "",
                    "parse_confidence": model_schema["parse_confidence"],
                    "sparse_anchor_count": model_schema["sparse_anchor_count"],
                    "parsed_changed_edges": model_schema["parsed_changed_edges"],
                    "model_contribution": model_schema["model_contribution"],
                }
            ),
            key="debug_model_summary",
        )
        st.markdown("**anchors**")
        _show_dataframe(_records_table(sparse_parse.get("anchors") or result.get("anchor_rows") or []), key="debug_anchors")
        st.markdown("**conflicts**")
        _show_dataframe(_records_table(sparse_parse.get("conflicts") or result.get("conflicts") or []), key="debug_conflicts")
        st.markdown("**protected_edges**")
        _show_dataframe(_records_table(sparse_parse.get("protected_edges") or result.get("protected_edges") or []), key="debug_protected_edges")
        st.markdown("**edge_confidence**")
        _show_dataframe(_records_table(sparse_parse.get("edge_confidence") or result.get("edge_confidence") or {}), key="debug_edge_confidence")

    with st.expander("Scene / SUMO Debug", expanded=False):
        _show_dataframe(
            _kv_table(
                {
                    "scene_profile": scene_schema["scene_profile"],
                    "raw_incident_edges": scene_schema["raw_incident_edges"],
                    "spillover_edges": scene_schema["spillover_edges"],
                    "runtime_spillover_enabled": scene_schema["runtime_spillover_enabled"],
                    "incident_edges": scene_cfg.get("incident_edges") or scene_cfg.get("incident") or result.get("incident_edges") or "",
                    "blocked_edges": scene_cfg.get("blocked_edges") or result.get("blocked_edges") or "",
                    "lane_reduction_edges": scene_cfg.get("lane_reduction_edges") or result.get("lane_reduction_edges") or "",
                    "spillover_runtime_summary": runtime_spillover,
                    "runtime_multipliers": scene_layers.get("stats") or result.get("spillover_stats") or "",
                    "reroute_period": result.get("reroute_period") or scene_cfg.get("reroute_period") or "",
                    "seed": result.get("seed") or scene_cfg.get("simulation_seed") or "",
                    "warmup_seconds": result.get("warmup_seconds") or scene_cfg.get("warmup_seconds") or "",
                    "evaluation_window": [
                        result.get("evaluation_start_time") or scene_cfg.get("evaluation_start_time"),
                        result.get("evaluation_end_time") or scene_cfg.get("evaluation_end_time"),
                    ],
                }
            ),
            key="debug_scene_summary",
        )
        st.markdown("**flattened scene profile**")
        _show_dataframe(make_arrow_safe(pd.DataFrame([flatten_json_record(scene_cfg)])), key="debug_scene_flat")

    with st.expander("Route / Runtime Debug", expanded=False):
        _show_dataframe(
            _kv_table(
                {
                    "baseline_path_edges": result.get("baseline_path") or result.get("baseline_path_edges") or [],
                    "planned_path_edges": result.get("planned_path") or result.get("planned_path_edges") or result.get("path") or [],
                    "baseline_travel_time": result.get("baseline_sumo_travel_time") or result.get("baseline_sumo_travel_time_s") or result.get("baseline_estimated_time"),
                    "planned_travel_time": result.get("planned_sumo_travel_time") or result.get("planned_sumo_travel_time_s") or result.get("planned_estimated_time") or result.get("travel_time"),
                    "waiting_time_s": result.get("waiting_time_s"),
                    "time_loss_s": result.get("time_loss_s"),
                    "stop_count": result.get("stop_count"),
                    "baseline_affected_overlap": scene_schema["baseline_affected_overlap"],
                    "planned_affected_overlap": scene_schema["planned_affected_overlap"],
                    "baseline_incident_overlap": scene_schema["baseline_incident_overlap"],
                    "planned_incident_overlap": scene_schema["planned_incident_overlap"],
                    "sumo_route_summary": result.get("sumo_route_summary") or "",
                    "baseline_sumo_route_summary": result.get("baseline_sumo_route_summary") or "",
                }
            ),
            key="debug_route_summary",
        )
        st.markdown("**flattened result payload**")
        _show_dataframe(build_debug_table(result), key="debug_flat_result")
