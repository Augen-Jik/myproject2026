#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/root/autodl-tmp")
CODE_DIR = ROOT / "code"
COMPARE_DIR = ROOT / "compare"
FINAL_TABLE_PATH = ROOT / "results" / "final_tables" / "method_comparison_v2.csv"
FINAL_TABLE_NO_PPO_PATH = ROOT / "results" / "final_tables" / "method_comparison_final.csv"
WARNING_REPORT_PATH = ROOT / "results" / "final_tables" / "method_comparison_v2_warnings.md"
BLOCKER_NOTE_PATH = ROOT / "results" / "final_tables" / "live_protocol_blockers.md"
OLD_TABLE_PATH = ROOT / "results" / "final_tables" / "method_comparison.csv"
PPO_SUMMARY_PATH = ROOT / "results" / "ppo_baseline" / "ppo_quick_summary.csv"
GAT_MODEL_PATH = ROOT / "gat_model_v2_gated.pt"

for path in (CODE_DIR, COMPARE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app_fixed as app
import run_compare as legacy_compare
from eval_metrics import compute_edge_metrics
from results_manager import create_result_bundle, default_config_snapshot, write_json, write_standard_artifacts
from scenarios import APP_SCENARIO_PRESETS
from sim_eval_protocol import build_sim_eval_protocol, protocol_to_dict
from sim_scene_profiles import (
    STANDARD_SCENE_PROFILE_NAMES,
    build_environment_weights,
    get_scene_profile,
    get_scene_profile_version,
)


app.st.error = lambda *args, **kwargs: None


SCENE_PROFILE_TO_PRESET_ID = {
    "normal_baseline": "N",
    "simple_local": "A",
    "directional_asymmetry": "E",
    # `B` starts from C4R0_N, which is blocked by the fixed core_blockage profile.
    # Use the alternate locked preset to keep the live evaluation route feasible.
    "core_blockage": "D",
    "propagation_range": "G",
    "compound_disaster": "C",
}

METHODS = (
    "Dijkstra",
    "Rule-A*",
    "DQN",
    "GCN-Weight",
    "Sparse-LoRA",
    "Sparse-LoRA+GAT",
    "PPO",
)

METHOD_META = {
    "Dijkstra": {
        "cn": "Dijkstra（等权）",
        "category": "Baseline",
        "notes": "旧 baseline 权重逻辑保持不变，仅切换为 scene-profile-fixed live SUMO 协议。",
    },
    "Rule-A*": {
        "cn": "规则 A*",
        "category": "Baseline",
        "notes": "旧规则权重生成保持不变，仅切换为 scene-profile-fixed live SUMO 协议。",
    },
    "DQN": {
        "cn": "DQN 强化学习",
        "category": "Baseline",
        "notes": "轻量 DQN baseline，按当前脚本固定 seed 逐场景训练并走 live SUMO 评测。",
    },
    "GCN-Weight": {
        "cn": "GCN 权重估计",
        "category": "Baseline",
        "notes": "轻量图特征权重估计 baseline，走 scene-profile-fixed live SUMO 评测。",
    },
    "Sparse-LoRA": {
        "cn": "Sparse-LoRA（stage4_fix）",
        "category": "Ours",
        "notes": "model_merged_sparse_v2_stage4_fix 主链路，A* + live SUMO。",
    },
    "Sparse-LoRA+GAT": {
        "cn": "Sparse-LoRA+GAT（stage4_fix）",
        "category": "Extended",
        "notes": "Exploratory graph-smoothing extension; not used as the frozen mainline method.",
    },
    "PPO": {
        "cn": "PPO（quick baseline，未调优）",
        "category": "Baseline",
        "notes": "直接读取 results/ppo_baseline/ppo_quick_summary.csv；未按 scene-profile-fixed live chain 重跑。",
    },
}

SUMMARY_METRICS = (
    ("Travel Time (s)", "travel_time_s"),
    ("Time Loss (s)", "time_loss_s"),
    ("Waiting Time (s)", "waiting_time_s"),
    ("Planning Time (s)", "planning_time_s"),
    ("Constraint Rate (%)", "constraint_rate"),
    ("Signal Ratio (%)", "signal_ratio"),
)

IDENTICAL_SCENE_FIELDS = (
    "travel_time_s",
    "time_loss_s",
    "waiting_time_s",
    "planning_time_s",
    "constraint_rate",
    "signal_ratio",
    "path_len",
    "path_cost",
    "start_edge",
    "end_edge",
)


def standard_scene_cases(scene_profiles: list[str]) -> list[dict[str, Any]]:
    preset_by_id = {str(item["id"]): item for item in APP_SCENARIO_PRESETS}
    cases: list[dict[str, Any]] = []
    for profile_name in scene_profiles:
        preset_id = SCENE_PROFILE_TO_PRESET_ID.get(profile_name)
        if not preset_id or preset_id not in preset_by_id:
            raise KeyError(f"Missing preset mapping for scene profile: {profile_name}")
        preset = preset_by_id[preset_id]
        scene_profile = get_scene_profile(profile_name)
        blocked_edges = set(scene_profile.blocked_edges or ())
        start_edge = str(preset["start"])
        end_edge = str(preset["end"])
        invalid_edges = [edge for edge in (start_edge, end_edge) if edge in blocked_edges]
        if invalid_edges:
            raise ValueError(
                "Invalid scene-profile preset mapping: "
                f"{profile_name} -> {preset_id} uses blocked endpoint(s) {invalid_edges}"
            )
        cases.append(
            {
                "preset_id": preset_id,
                "scenario": str(preset["name"]),
                "scene_profile_name": profile_name,
                "scene_profile": scene_profile,
                "start_edge": start_edge,
                "end_edge": end_edge,
                "constraint": str(preset["constraint"]),
            }
        )
    return cases


def load_old_method_means() -> dict[str, dict[str, float]]:
    if not OLD_TABLE_PATH.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    with OLD_TABLE_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            method = str(row.get("Method", "")).strip()
            if not method:
                continue
            out[method] = {
                "travel_time_s": _safe_float(row.get("Travel Time (s)")),
                "planning_time_s": _safe_float(row.get("Planning Time (s)")),
                "constraint_rate": _safe_float(row.get("Constraint Rate (%)")),
                "signal_ratio": _safe_float(row.get("Signal Ratio (%)")),
            }
    return out


def load_ppo_summary() -> dict[str, str]:
    if not PPO_SUMMARY_PATH.exists():
        return {}
    with PPO_SUMMARY_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[0] if rows else {}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _format_scalar(value: float | None, *, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _relative_delta(new_value: float | None, old_value: float | None) -> tuple[float | None, float | None]:
    if new_value is None or old_value is None:
        return None, None
    diff = float(new_value) - float(old_value)
    if abs(float(old_value)) < 1e-9:
        return diff, None
    return diff, diff / float(old_value) * 100.0


def _delta_text(new_value: float | None, old_value: float | None) -> str:
    diff, rel = _relative_delta(new_value, old_value)
    if diff is None:
        return "n/a"
    if rel is None:
        return f"{diff:+.2f}"
    flag = " [>5%]" if abs(rel) > 5.0 else ""
    return f"{diff:+.2f} ({rel:+.1f}%){flag}"


def _joined_protocol_values(cases: list[dict[str, Any]], field_name: str) -> str:
    return "|".join(str(getattr(case["scene_profile"], field_name)) for case in cases)


def _scene_protocol_map(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload = {}
    for case in cases:
        profile = case["scene_profile"]
        payload[case["scene_profile_name"]] = {
            "simulation_seed": int(profile.simulation_seed),
            "warmup_seconds": int(profile.warmup_seconds),
            "evaluation_start_time": int(profile.evaluation_start_time),
            "evaluation_end_time": int(profile.evaluation_end_time),
        }
    return payload


def _load_live_rows_from_metrics_csv(
    metrics_csv: str | Path,
    *,
    methods: list[str],
    scene_profiles: list[str],
) -> list[dict[str, Any]]:
    path = Path(metrics_csv)
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics csv: {path}")

    live_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw_row in reader:
            method_name = str(raw_row.get("method", "")).strip()
            scene_profile_name = str(raw_row.get("scene_profile", "")).strip()
            if method_name not in methods or scene_profile_name not in scene_profiles:
                continue
            live_rows.append(
                {
                    "scenario": str(raw_row.get("scenario", "")).strip(),
                    "preset_id": str(raw_row.get("preset_id", "")).strip(),
                    "scene_profile": scene_profile_name,
                    "method": method_name,
                    "start_edge": str(raw_row.get("start_edge", "")).strip(),
                    "end_edge": str(raw_row.get("end_edge", "")).strip(),
                    "travel_time_s": float(raw_row.get("travel_time_s", 0.0) or 0.0),
                    "time_loss_s": float(raw_row["time_loss_s"]) if raw_row.get("time_loss_s") not in (None, "", "None") else None,
                    "waiting_time_s": float(raw_row["waiting_time_s"]) if raw_row.get("waiting_time_s") not in (None, "", "None") else None,
                    "stop_count": int(float(raw_row["stop_count"])) if raw_row.get("stop_count") not in (None, "", "None") else None,
                    "vehicle_arrived": str(raw_row.get("vehicle_arrived", "")).strip().lower() == "true",
                    "simulation_truncated": str(raw_row.get("simulation_truncated", "")).strip().lower() == "true",
                    "completion_status": str(raw_row.get("completion_status", "")).strip(),
                    "tripinfo_arrival_s": (
                        float(raw_row["tripinfo_arrival_s"])
                        if raw_row.get("tripinfo_arrival_s") not in (None, "", "None", "nan")
                        else None
                    ),
                    "tripinfo_vaporized": str(raw_row.get("tripinfo_vaporized", "")).strip(),
                    "planning_time_s": float(raw_row.get("planning_time_s", 0.0) or 0.0),
                    "model_infer_time_s": float(raw_row.get("model_infer_time_s", 0.0) or 0.0),
                    "route_solve_time_s": float(raw_row.get("route_solve_time_s", 0.0) or 0.0),
                    "constraint_rate": float(raw_row.get("constraint_rate", 0.0) or 0.0),
                    "signal_ratio": float(raw_row.get("signal_ratio", 0.0) or 0.0),
                    "signal_edges": int(float(raw_row.get("signal_edges", 0) or 0)),
                    "parsed_edges": int(float(raw_row.get("parsed_edges", 0) or 0)),
                    "parsed_edge_ratio": float(raw_row.get("parsed_edge_ratio", 0.0) or 0.0),
                    "default_edges": int(float(raw_row.get("default_edges", 0) or 0)),
                    "path_len": int(float(raw_row.get("path_len", 0) or 0)),
                    "path_cost": float(raw_row.get("path_cost", 0.0) or 0.0),
                    "simulation_seed": int(float(raw_row.get("simulation_seed", 0) or 0)),
                    "warmup_seconds": int(float(raw_row.get("warmup_seconds", 0) or 0)),
                    "evaluation_start_time": int(float(raw_row.get("evaluation_start_time", 0) or 0)),
                    "evaluation_end_time": int(float(raw_row.get("evaluation_end_time", 0) or 0)),
                    "scene_profile_version": str(raw_row.get("scene_profile_version", "")).strip(),
                    "protocol_snapshot_json": str(raw_row.get("protocol_snapshot_json", "")).strip(),
                    "use_gat": str(raw_row.get("use_gat", "")).strip().lower() == "true",
                    "model_path": str(raw_row.get("model_path", "")).strip(),
                    "notes": str(raw_row.get("notes", "")).strip(),
                }
            )
    return live_rows


def _scene_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    signature: list[Any] = []
    for field in IDENTICAL_SCENE_FIELDS:
        value = row.get(field)
        if isinstance(value, float):
            signature.append(round(value, 6))
        else:
            signature.append(value)
    return tuple(signature)


def _detect_identical_scene_outputs(
    *,
    live_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in live_rows:
        by_method[row["method"]][row["scene_profile"]] = row

    ordered_scene_profiles = [case["scene_profile_name"] for case in cases]
    warnings: list[dict[str, Any]] = []
    for idx, left_method in enumerate(METHODS):
        if left_method not in by_method:
            continue
        for right_method in METHODS[idx + 1 :]:
            if right_method not in by_method:
                continue
            identical_scenes: list[str] = []
            for scene_profile_name in ordered_scene_profiles:
                left_row = by_method[left_method].get(scene_profile_name)
                right_row = by_method[right_method].get(scene_profile_name)
                if not left_row or not right_row:
                    identical_scenes = []
                    break
                if _scene_signature(left_row) != _scene_signature(right_row):
                    identical_scenes = []
                    break
                identical_scenes.append(scene_profile_name)
            if identical_scenes:
                warnings.append(
                    {
                        "left_method": left_method,
                        "right_method": right_method,
                        "scene_profiles": identical_scenes,
                        "checked_fields": list(IDENTICAL_SCENE_FIELDS),
                    }
                )
    return warnings


def _render_warning_report(warnings: list[dict[str, Any]]) -> str:
    lines = [
        "# Method Comparison Warnings",
        "",
        "- Checked duplicate scene-level signatures across "
        "`travel_time_s`, `planning_time_s`, `constraint_rate`, `signal_ratio`, "
        "`path_len`, `path_cost`, `start_edge`, `end_edge`.",
        "",
    ]
    if not warnings:
        lines.append("- No method pair is fully identical across every checked scene.")
        return "\n".join(lines) + "\n"

    lines.append("## Fully Identical Method Pairs")
    lines.append("")
    for item in warnings:
        scene_profiles = ", ".join(item["scene_profiles"])
        lines.append(
            f"- WARNING: `{item['left_method']}` and `{item['right_method']}` are "
            f"fully identical across scenes: {scene_profiles}"
        )
    return "\n".join(lines) + "\n"


def _render_blocker_note(include_ppo: bool) -> str:
    lines = [
        "# Live Protocol Blockers",
        "",
        "- Current scene-profile-fixed live runner coverage: "
        "`Dijkstra`, `Rule-A*`, `DQN`, `GCN-Weight`, `Sparse-LoRA`, `Sparse-LoRA+GAT`.",
        "- PPO still lacks a same-protocol live rerun path in this chain; the available PPO row "
        "remains the direct-read quick baseline summary under `results/ppo_baseline/ppo_quick_summary.csv`.",
        "- Therefore, the PPO row must stay explicitly marked as `quick baseline, untuned` and must "
        "not be promoted to a fully comparable live main-table result without a compatible runner.",
        "- Legacy prompt / LoRA rows such as `Raw-Qwen`, `CoT-Qwen`, `R1-Raw`, `Qwen-LoRA`, and "
        "`R1-LoRA` are not generated by this live scene-profile runner, so rebuilding an all-method "
        "paper table on one unified live protocol is blocked in the current code path.",
        "- Lowest viable alternative in this batch: keep `results/final_tables/method_comparison_v2.csv` "
        "as the protocol-clean main evidence table for the supported methods, and keep PPO as a "
        "separate coverage-only note.",
    ]
    if not include_ppo:
        lines.append("- This run was executed with `--skip-ppo`, so no PPO quick row was appended.")
    return "\n".join(lines) + "\n"


def _constraint_rate(final_weights: dict[str, float], scene_weights: dict[str, float]) -> float:
    sig_edges = [eid for eid, weight in scene_weights.items() if abs(float(weight) - 2.0) > 0.5]
    if not sig_edges:
        return 100.0
    correct = sum(
        1
        for eid in sig_edges
        if (float(final_weights.get(eid, 2.0)) >= 4.0) == (float(scene_weights[eid]) >= 4.0)
    )
    return round(correct / len(sig_edges) * 100.0, 2)


def _live_sparse_method(
    *,
    case: dict[str, Any],
    use_gat: bool,
    gat_alpha: float,
    gat_model_path: str,
) -> dict[str, Any]:
    app.config.config["RUN_MODE"] = "experiment_mode"
    app.config.config["MODEL_PATH"] = app.MAINLINE_MODEL_PATH
    setattr(app.path_engine, "_last_scene_type", case["scene_profile"].scene_type)
    setattr(app.path_engine, "_last_scene_profile_name", case["scene_profile"].scene_name)

    model_key = f"{app.MAINLINE_METHOD_NAME}|{app.MAINLINE_MODEL_PATH}"
    (
        _struct_output,
        raw_llm_weights,
        final_planning_weights,
        infer_time,
        _raw_output,
        _llm_parsed,
        _gat_applied,
        _gat_alpha_used,
        explicit_weight_dict,
        _mode_used,
        _fallback_used,
        _fallback_reason,
        _timing_info,
    ) = app.path_engine.generate_weights(
        case["constraint"],
        model_key,
        app.model_manager,
        use_gat=use_gat,
        gat_model_path=gat_model_path,
        gat_alpha=gat_alpha,
        mode="experiment_mode",
    )

    net, _ = app._load_net_cached(app.config.get("SUMO_NET_PATH"))
    route_t0 = time.perf_counter()
    path, total_cost = app.path_engine.astar_route(
        net,
        case["start_edge"],
        case["end_edge"],
        final_planning_weights,
    )
    route_solve_time_s = time.perf_counter() - route_t0
    return {
        "final_weights": dict(final_planning_weights),
        "explicit_weights": dict(explicit_weight_dict),
        "path": list(path or []),
        "total_cost": float(total_cost or 0.0),
        "model_infer_time_s": float(infer_time or 0.0),
        "route_solve_time_s": float(route_solve_time_s),
        "planning_time_s": float(infer_time or 0.0) + float(route_solve_time_s),
        "raw_weight_count": len(raw_llm_weights or {}),
    }


def _legacy_rule_weights(case: dict[str, Any]) -> dict[str, float]:
    weights, _path, _cost = legacy_compare.method_rule_astar(
        case["start_edge"],
        case["end_edge"],
        case["constraint"],
    )
    return dict(weights)


def _legacy_baseline_method(*, method_name: str, case: dict[str, Any], dqn_episodes: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    if method_name == "Dijkstra":
        weights, path, total_cost = legacy_compare.method_dijkstra(
            case["start_edge"],
            case["end_edge"],
            case["constraint"],
        )
        explicit_weights: dict[str, float] = {}
    elif method_name == "Rule-A*":
        weights, path, total_cost = legacy_compare.method_rule_astar(
            case["start_edge"],
            case["end_edge"],
            case["constraint"],
        )
        explicit_weights = dict(weights)
    elif method_name == "DQN":
        random.seed(42)
        np.random.seed(42)
        agent = legacy_compare.LightDQN()
        weights = _legacy_rule_weights(case)
        agent.train(legacy_compare.net, weights, episodes=dqn_episodes)
        path, total_cost = agent.find_path(
            legacy_compare.net,
            case["start_edge"],
            case["end_edge"],
            weights,
        )
        if not path:
            path, total_cost = legacy_compare.astar_route(
                legacy_compare.net,
                case["start_edge"],
                case["end_edge"],
                weights,
            )
        explicit_weights = dict(weights)
    elif method_name == "GCN-Weight":
        weights, path, total_cost = legacy_compare.method_gcn_weight(
            case["start_edge"],
            case["end_edge"],
            case["constraint"],
        )
        explicit_weights = dict(weights)
    else:
        raise ValueError(f"Unsupported baseline method: {method_name}")

    planning_time_s = time.perf_counter() - t0
    return {
        "final_weights": dict(weights),
        "explicit_weights": explicit_weights,
        "path": list(path or []),
        "total_cost": float(total_cost or 0.0),
        "model_infer_time_s": 0.0,
        "route_solve_time_s": float(planning_time_s),
        "planning_time_s": float(planning_time_s),
        "raw_weight_count": len(explicit_weights),
    }


def _evaluate_live_case(
    *,
    case: dict[str, Any],
    method_name: str,
    use_gat: bool,
    gat_alpha: float,
    gat_model_path: str,
    dqn_episodes: int,
) -> dict[str, Any]:
    if method_name in {"Dijkstra", "Rule-A*", "DQN", "GCN-Weight"}:
        method_payload = _legacy_baseline_method(
            method_name=method_name,
            case=case,
            dqn_episodes=dqn_episodes,
        )
    elif method_name == "Sparse-LoRA":
        method_payload = _live_sparse_method(
            case=case,
            use_gat=False,
            gat_alpha=gat_alpha,
            gat_model_path=gat_model_path,
        )
    elif method_name == "Sparse-LoRA+GAT":
        method_payload = _live_sparse_method(
            case=case,
            use_gat=use_gat,
            gat_alpha=gat_alpha,
            gat_model_path=gat_model_path,
        )
    else:
        raise ValueError(f"Unsupported live-eval method: {method_name}")

    scene_profile = case["scene_profile"]
    scene_weights = build_environment_weights(app.path_engine._ALL_EDGES, scene_profile)
    sim_result = app.sumo_engine.run_simulation(
        tuple(method_payload["path"]),
        scene_profile,
        tuple(sorted(scene_weights.items())),
    )
    travel_time_s = float(sim_result.get("travel_time", 0.0))
    avg_speed = float(sim_result.get("avg_speed", 0.0))
    _time_loss = sim_result.get("time_loss_s")
    _waiting_time = sim_result.get("waiting_time_s")
    _stop_count = sim_result.get("stop_count")
    _vehicle_arrived = bool(sim_result.get("vehicle_arrived"))
    _simulation_truncated = bool(sim_result.get("simulation_truncated"))
    _completion_status = str(sim_result.get("completion_status") or "")
    _tripinfo_arrival_s = sim_result.get("tripinfo_arrival_s")
    _tripinfo_vaporized = str(sim_result.get("tripinfo_vaporized") or "")
    edge_metrics = compute_edge_metrics(
        final_weights=method_payload["final_weights"],
        explicit_weights=method_payload["explicit_weights"],
        all_edges=app.path_engine._ALL_EDGES,
        method_name=method_name,
    )
    protocol_snapshot = protocol_to_dict(build_sim_eval_protocol(scene_profile))

    return {
        "scenario": case["scenario"],
        "preset_id": case["preset_id"],
        "scene_profile": case["scene_profile_name"],
        "start_edge": case["start_edge"],
        "end_edge": case["end_edge"],
        "constraint": case["constraint"],
        "method": method_name,
        "path": list(method_payload["path"]),
        "path_len": len(method_payload["path"]),
        "path_cost": round(float(method_payload["total_cost"]), 6),
        "travel_time_s": round(float(travel_time_s), 3),
        "time_loss_s": round(float(_time_loss), 3) if _time_loss is not None else None,
        "waiting_time_s": round(float(_waiting_time), 3) if _waiting_time is not None else None,
        "stop_count": int(_stop_count) if _stop_count is not None else None,
        "vehicle_arrived": _vehicle_arrived,
        "simulation_truncated": _simulation_truncated,
        "completion_status": _completion_status,
        "tripinfo_arrival_s": (
            round(float(_tripinfo_arrival_s), 3)
            if _tripinfo_arrival_s is not None and np.isfinite(float(_tripinfo_arrival_s))
            else None
        ),
        "tripinfo_vaporized": _tripinfo_vaporized,
        "planning_time_s": round(float(method_payload["planning_time_s"]), 6),
        "model_infer_time_s": round(float(method_payload["model_infer_time_s"]), 6),
        "route_solve_time_s": round(float(method_payload["route_solve_time_s"]), 6),
        "constraint_rate": _constraint_rate(method_payload["final_weights"], scene_weights),
        "signal_ratio": round(float(edge_metrics["signal_ratio"]), 3),
        "signal_edges": int(edge_metrics["signal_edges"]),
        "parsed_edges": int(edge_metrics["parsed_edges"]),
        "parsed_edge_ratio": round(float(edge_metrics["parsed_edge_ratio"]), 3),
        "default_edges": int(edge_metrics["default_edges"]),
        "avg_speed_kmh": round(float(avg_speed), 3),
        "simulation_seed": int(scene_profile.simulation_seed),
        "warmup_seconds": int(scene_profile.warmup_seconds),
        "evaluation_start_time": int(scene_profile.evaluation_start_time),
        "evaluation_end_time": int(scene_profile.evaluation_end_time),
        "scene_profile_version": get_scene_profile_version(),
        "protocol_snapshot": protocol_snapshot,
        "protocol_snapshot_json": json.dumps(protocol_snapshot, ensure_ascii=False, sort_keys=True),
        "use_gat": bool(use_gat and method_name == "Sparse-LoRA+GAT"),
        "model_path": (
            str(app.MAINLINE_MODEL_PATH)
            if method_name.startswith("Sparse-LoRA")
            else ""
        ),
        "notes": METHOD_META[method_name]["notes"],
    }


def _group_compare_results(rows: list[dict[str, Any]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[row["scenario"]].append(row)

    case_lookup = {case["scenario"]: case for case in cases}
    for case in cases:
        scenario = case["scenario"]
        protocol = protocol_to_dict(build_sim_eval_protocol(case["scene_profile"]))
        grouped.append(
            {
                "scenario": scenario,
                "scene_profile": case["scene_profile_name"],
                "preset_id": case["preset_id"],
                "start_edge": case["start_edge"],
                "end_edge": case["end_edge"],
                "constraint": case["constraint"],
                "protocol": protocol,
                "methods": sorted(by_scene.get(scenario, []), key=lambda item: METHODS.index(item["method"])),
            }
        )
    return grouped


def _build_method_comparison_rows(
    *,
    live_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    include_ppo: bool,
) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in live_rows:
        by_method[row["method"]].append(row)

    protocol_seed = _joined_protocol_values(cases, "simulation_seed")
    protocol_warmup = _joined_protocol_values(cases, "warmup_seconds")
    protocol_eval_start = _joined_protocol_values(cases, "evaluation_start_time")
    protocol_eval_end = _joined_protocol_values(cases, "evaluation_end_time")

    out: list[dict[str, Any]] = []
    for method_name in METHODS:
        if method_name == "PPO":
            continue
        rows = sorted(by_method.get(method_name, []), key=lambda item: STANDARD_SCENE_PROFILE_NAMES.index(item["scene_profile"]))
        if not rows:
            continue
        meta = METHOD_META[method_name]
        _tl_vals = [row["time_loss_s"] for row in rows if row.get("time_loss_s") is not None]
        _wt_vals = [row["waiting_time_s"] for row in rows if row.get("waiting_time_s") is not None]
        _sc_vals = [row["stop_count"] for row in rows if row.get("stop_count") is not None]
        out.append(
            {
                "Method": method_name,
                "Method (CN)": meta["cn"],
                "Category": meta["category"],
                "Travel Time (s)": round(_mean([row["travel_time_s"] for row in rows]), 6),
                "Time Loss (s)": round(_mean(_tl_vals), 6) if _tl_vals else None,
                "Waiting Time (s)": round(_mean(_wt_vals), 6) if _wt_vals else None,
                "Stop Count": round(_mean(_sc_vals), 3) if _sc_vals else None,
                "model_infer_time_s": round(_mean([row["model_infer_time_s"] for row in rows]), 6),
                "route_solve_time_s": round(_mean([row["route_solve_time_s"] for row in rows]), 6),
                "Planning Time (s)": round(_mean([row["planning_time_s"] for row in rows]), 6),
                "Constraint Rate (%)": round(_mean([row["constraint_rate"] for row in rows]), 6),
                "Signal Ratio (%)": round(_mean([row["signal_ratio"] for row in rows]), 6),
                "simulation_seed": protocol_seed,
                "warmup_seconds": protocol_warmup,
                "evaluation_start_time": protocol_eval_start,
                "evaluation_end_time": protocol_eval_end,
                "Notes": meta["notes"],
            }
        )

    if include_ppo:
        ppo_summary = load_ppo_summary()
        if ppo_summary:
            out.append(
                {
                    "Method": "PPO",
                    "Method (CN)": METHOD_META["PPO"]["cn"],
                    "Category": METHOD_META["PPO"]["category"],
                    "Travel Time (s)": _safe_float(ppo_summary.get("Travel Time (s)")),
                    "Time Loss (s)": None,
                    "Waiting Time (s)": None,
                    "Stop Count": None,
                    "model_infer_time_s": "",
                    "route_solve_time_s": "",
                    "Planning Time (s)": _safe_float(ppo_summary.get("Planning Time (s)")),
                    "Constraint Rate (%)": "",
                    "Signal Ratio (%)": "",
                    "simulation_seed": "",
                    "warmup_seconds": "",
                    "evaluation_start_time": "",
                    "evaluation_end_time": "",
                    "Notes": METHOD_META["PPO"]["notes"],
                }
            )

    return out


def _terminal_report(
    *,
    live_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    old_means: dict[str, dict[str, float]],
    include_ppo: bool,
    warnings: list[dict[str, Any]],
) -> str:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in live_rows:
        by_method[row["method"]].append(row)

    lines = []
    for method_name in METHODS:
        if method_name == "PPO":
            if not include_ppo:
                continue
            ppo_summary = load_ppo_summary()
            lines.append(f"[Method] PPO")
            if ppo_summary:
                lines.append(
                    "  - aggregate_only: "
                    f"travel={_format_scalar(_safe_float(ppo_summary.get('Travel Time (s)')))} s, "
                    f"planning={_format_scalar(_safe_float(ppo_summary.get('Planning Time (s)')), digits=6)} s, "
                    "constraint=n/a, signal=n/a, source=results/ppo_baseline/ppo_quick_summary.csv"
                )
            else:
                lines.append("  - aggregate_only: missing ppo_quick_summary.csv")
            lines.append("  - delta_vs_old_mean: n/a")
            lines.append("")
            continue

        rows = sorted(by_method.get(method_name, []), key=lambda item: STANDARD_SCENE_PROFILE_NAMES.index(item["scene_profile"]))
        if not rows:
            continue
        lines.append(f"[Method] {method_name}")
        for row in rows:
            _tl = row.get("time_loss_s")
            _wt = row.get("waiting_time_s")
            _sc = row.get("stop_count")
            _completion = str(row.get("completion_status") or "")
            _completion_suffix = f", completion={_completion}" if _completion else ""
            lines.append(
                "  - "
                f"{row['scene_profile']} / {row['scenario']}: "
                f"travel={row['travel_time_s']:.2f} s, "
                f"time_loss={f'{_tl:.2f} s' if _tl is not None else 'n/a'}, "
                f"waiting={f'{_wt:.2f} s' if _wt is not None else 'n/a'}, "
                f"stops={_sc if _sc is not None else 'n/a'}, "
                f"planning={row['planning_time_s']:.6f} s, "
                f"constraint={row['constraint_rate']:.2f}%, "
                f"signal={row['signal_ratio']:.2f}%, "
                f"seed={row['simulation_seed']}, "
                f"warmup={row['warmup_seconds']} s, "
                f"eval={row['evaluation_start_time']}-{row['evaluation_end_time']} s"
                f"{_completion_suffix}"
            )

        mean_travel = _mean([row["travel_time_s"] for row in rows])
        mean_planning = _mean([row["planning_time_s"] for row in rows])
        mean_constraint = _mean([row["constraint_rate"] for row in rows])
        mean_signal = _mean([row["signal_ratio"] for row in rows])
        _tl_vals = [row["time_loss_s"] for row in rows if row.get("time_loss_s") is not None]
        _wt_vals = [row["waiting_time_s"] for row in rows if row.get("waiting_time_s") is not None]
        _sc_vals = [row["stop_count"] for row in rows if row.get("stop_count") is not None]
        mean_time_loss = _mean(_tl_vals) if _tl_vals else None
        mean_waiting = _mean(_wt_vals) if _wt_vals else None
        mean_stops = _mean(_sc_vals) if _sc_vals else None
        lines.append(
            "  - mean: "
            f"travel={mean_travel:.2f} s, "
            f"time_loss={f'{mean_time_loss:.2f} s' if mean_time_loss is not None else 'n/a'}, "
            f"waiting={f'{mean_waiting:.2f} s' if mean_waiting is not None else 'n/a'}, "
            f"stops={f'{mean_stops:.2f}' if mean_stops is not None else 'n/a'}, "
            f"planning={mean_planning:.6f} s, "
            f"constraint={mean_constraint:.2f}%, "
            f"signal={mean_signal:.2f}%"
        )

        old = old_means.get(method_name)
        if old:
            lines.append(
                "  - delta_vs_old_mean: "
                f"travel={_delta_text(mean_travel, old.get('travel_time_s'))}, "
                f"planning={_delta_text(mean_planning, old.get('planning_time_s'))}, "
                f"constraint={_delta_text(mean_constraint, old.get('constraint_rate'))}, "
                f"signal={_delta_text(mean_signal, old.get('signal_ratio'))}"
            )
        else:
            lines.append("  - delta_vs_old_mean: n/a")
        lines.append("")
    lines.append("[Warnings]")
    if warnings:
        for item in warnings:
            lines.append(
                "  - WARNING: "
                f"{item['left_method']} vs {item['right_method']} are fully identical across "
                f"{len(item['scene_profiles'])} scenes"
            )
    else:
        lines.append("  - none")
    return "\n".join(lines).rstrip() + "\n"


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    scene_profiles = list(args.scene_profiles or STANDARD_SCENE_PROFILE_NAMES)
    methods = [method for method in METHODS if method != "PPO"]
    if args.methods:
        methods = [method for method in METHODS if method in args.methods and method != "PPO"]

    cases = standard_scene_cases(scene_profiles)
    old_means = load_old_method_means()
    scene_profile_version = get_scene_profile_version()

    app.config.config["RUN_MODE"] = "experiment_mode"
    app.config.config["MODEL_PATH"] = app.MAINLINE_MODEL_PATH

    if args.reuse_metrics_csv:
        live_rows = _load_live_rows_from_metrics_csv(
            args.reuse_metrics_csv,
            methods=methods,
            scene_profiles=scene_profiles,
        )
        print(f"[Reuse] metrics.csv: {args.reuse_metrics_csv}")
        print(f"[Reuse] loaded rows: {len(live_rows)}")
    else:
        live_rows = []
        for method_name in methods:
            print(f"[Run] {method_name}")
            for case in cases:
                print(
                    "  -> "
                    f"{case['scene_profile_name']} | {case['scenario']} | "
                    f"{case['start_edge']} -> {case['end_edge']}"
                )
                row = _evaluate_live_case(
                    case=case,
                    method_name=method_name,
                    use_gat=True,
                    gat_alpha=args.gat_alpha,
                    gat_model_path=args.gat_model_path,
                    dqn_episodes=args.dqn_episodes,
                )
                live_rows.append(row)
                _tl = row.get("time_loss_s")
                _wt = row.get("waiting_time_s")
                _sc = row.get("stop_count")
                _completion = str(row.get("completion_status") or "")
                _completion_suffix = f" | completion={_completion}" if _completion else ""
                print(
                    "     "
                    f"travel={row['travel_time_s']:.2f}s | "
                    f"time_loss={f'{_tl:.2f}s' if _tl is not None else 'n/a'} | "
                    f"waiting={f'{_wt:.2f}s' if _wt is not None else 'n/a'} | "
                    f"stops={_sc if _sc is not None else 'n/a'} | "
                    f"planning={row['planning_time_s']:.6f}s | "
                    f"constraint={row['constraint_rate']:.2f}% | "
                    f"signal={row['signal_ratio']:.2f}% | "
                    f"seed={row['simulation_seed']} | "
                    f"warmup={row['warmup_seconds']}s | "
                    f"eval={row['evaluation_start_time']}-{row['evaluation_end_time']}s"
                    f"{_completion_suffix}"
                )

    compare_results = _group_compare_results(live_rows, cases)
    warnings = _detect_identical_scene_outputs(live_rows=live_rows, cases=cases)
    summary_rows = _build_method_comparison_rows(
        live_rows=live_rows,
        cases=cases,
        include_ppo=not args.skip_ppo,
    )
    report_text = _terminal_report(
        live_rows=live_rows,
        cases=cases,
        old_means=old_means,
        include_ppo=not args.skip_ppo,
        warnings=warnings,
    )
    warning_report = _render_warning_report(warnings)
    blocker_note = _render_blocker_note(include_ppo=not args.skip_ppo)

    bundle = create_result_bundle(
        mode="experiment_mode",
        planner="compare_multi_method_live_scene_profile",
        model="method_comparison_v2",
    )
    compare_results_path = Path(bundle.run_dir) / "compare_results.json"
    write_json(str(compare_results_path), compare_results)

    metrics_field_order = [
        "scenario",
        "preset_id",
        "scene_profile",
        "method",
        "start_edge",
        "end_edge",
        "travel_time_s",
        "time_loss_s",
        "waiting_time_s",
        "stop_count",
        "vehicle_arrived",
        "simulation_truncated",
        "completion_status",
        "tripinfo_arrival_s",
        "tripinfo_vaporized",
        "planning_time_s",
        "model_infer_time_s",
        "route_solve_time_s",
        "constraint_rate",
        "signal_ratio",
        "signal_edges",
        "parsed_edges",
        "parsed_edge_ratio",
        "default_edges",
        "path_len",
        "path_cost",
        "simulation_seed",
        "warmup_seconds",
        "evaluation_start_time",
        "evaluation_end_time",
        "scene_profile_version",
        "use_gat",
        "model_path",
        "protocol_snapshot_json",
        "notes",
    ]

    summary = {
        "run_id": bundle.run_id,
        "mode": "experiment_mode",
        "scene_profiles": scene_profiles,
        "methods_rerun": methods,
        "include_ppo_summary": not args.skip_ppo,
        "scene_profile_version": scene_profile_version,
        "final_table_path": str(FINAL_TABLE_PATH),
        "final_table_no_ppo_path": str(FINAL_TABLE_NO_PPO_PATH),
        "compare_results_path": str(compare_results_path),
        "warning_report_path": str(WARNING_REPORT_PATH),
        "identical_scene_warnings": warnings,
    }
    config_snapshot = default_config_snapshot(
        timestamp=bundle.run_id.split("__")[0],
        mode="experiment_mode",
        seed=42,
        planner="compare_multi_method_live_scene_profile",
        model="method_comparison_v2",
        scenarios=[case["scenario"] for case in cases],
        fallback_policy="experiment_mode_no_rule_fallback_for_llm",
        default_weight_policy="method_specific_default_weight",
        timing_policy="planning_time_s=model_infer_time_s+route_solve_time_s",
        simulation_seed={key: value["simulation_seed"] for key, value in _scene_protocol_map(cases).items()},
        warmup_seconds={key: value["warmup_seconds"] for key, value in _scene_protocol_map(cases).items()},
        evaluation_start_time={key: value["evaluation_start_time"] for key, value in _scene_protocol_map(cases).items()},
        evaluation_end_time={key: value["evaluation_end_time"] for key, value in _scene_protocol_map(cases).items()},
        scene_profile_version=scene_profile_version,
        extra={
            "scene_profile_protocols": _scene_protocol_map(cases),
            "run_compare_results": str(compare_results_path),
            "mainline_model_path": str(app.MAINLINE_MODEL_PATH),
            "gat_model_path": str(args.gat_model_path),
            "ppo_summary_path": str(PPO_SUMMARY_PATH) if not args.skip_ppo else "",
            "reuse_metrics_csv": str(args.reuse_metrics_csv or ""),
            "warning_report_path": str(WARNING_REPORT_PATH),
            "identical_scene_warnings": warnings,
        },
    )

    write_standard_artifacts(
        bundle,
        summary=summary,
        metrics_rows=live_rows,
        report_text=report_text,
        config_snapshot=config_snapshot,
        metrics_field_order=metrics_field_order,
    )

    _SUMMARY_FIELDNAMES = [
        "Method",
        "Method (CN)",
        "Category",
        "Travel Time (s)",
        "Time Loss (s)",
        "Waiting Time (s)",
        "Stop Count",
        "model_infer_time_s",
        "route_solve_time_s",
        "Planning Time (s)",
        "Constraint Rate (%)",
        "Signal Ratio (%)",
        "simulation_seed",
        "warmup_seconds",
        "evaluation_start_time",
        "evaluation_end_time",
        "Notes",
    ]

    FINAL_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FINAL_TABLE_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    with FINAL_TABLE_NO_PPO_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([row for row in summary_rows if row["Method"] != "PPO"])

    WARNING_REPORT_PATH.write_text(warning_report, encoding="utf-8")
    BLOCKER_NOTE_PATH.write_text(blocker_note, encoding="utf-8")

    print("")
    print(report_text, end="")
    print(f"[Saved] compare_results: {compare_results_path}")
    print(f"[Saved] metrics.csv: {bundle.metrics_csv}")
    print(f"[Saved] config_snapshot.json: {bundle.config_snapshot_json}")
    print(f"[Saved] method_comparison_v2.csv: {FINAL_TABLE_PATH}")
    print(f"[Saved] method_comparison_final.csv: {FINAL_TABLE_NO_PPO_PATH}")
    print(f"[Saved] warnings: {WARNING_REPORT_PATH}")
    print(f"[Saved] blockers: {BLOCKER_NOTE_PATH}")

    return {
        "live_rows": live_rows,
        "summary_rows": summary_rows,
        "report_text": report_text,
        "warnings": warnings,
        "compare_results_path": str(compare_results_path),
        "final_table_path": str(FINAL_TABLE_PATH),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun the main comparison table with the scene-profile-fixed live SUMO chain.")
    parser.add_argument(
        "--scene-profiles",
        nargs="*",
        default=list(STANDARD_SCENE_PROFILE_NAMES),
        choices=list(STANDARD_SCENE_PROFILE_NAMES),
        help="Subset of the six standard scene profiles to run.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=list(METHODS),
        help="Subset of methods to run. PPO remains direct-read only.",
    )
    parser.add_argument(
        "--skip-ppo",
        action="store_true",
        help="Skip appending the direct-read PPO summary row.",
    )
    parser.add_argument(
        "--gat-alpha",
        type=float,
        default=0.85,
        help="Alpha passed to the live GAT smoother for Sparse-LoRA+GAT.",
    )
    parser.add_argument(
        "--gat-model-path",
        default=str(GAT_MODEL_PATH),
        help="Path to gat_model_v2_gated.pt.",
    )
    parser.add_argument(
        "--dqn-episodes",
        type=int,
        default=300,
        help="Training episodes for the lightweight DQN baseline per scene.",
    )
    parser.add_argument(
        "--reuse-metrics-csv",
        default="",
        help="Reuse a previous compatible metrics.csv to rebuild the table artifacts without rerunning live evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
