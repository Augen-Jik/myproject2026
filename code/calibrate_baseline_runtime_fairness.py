from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sumolib
import traci

from apply_scene_profile_to_sumo import SUMOSceneProfileApplier
from config import ConfigManager
from demo_scene_selector import select_showcase_scenarios
from roadnet_meta import live_edge_ids
from routing import dijkstra_route, free_flow_route
from sim_scene_profiles import build_environment_weight_layers, get_scene_profile
from sumo_runner import build_safe_vehicle_route


DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/results/runtime_audit"


def _cleanup_sumo() -> None:
    try:
        subprocess.run(["pkill", "-f", "sumo"], capture_output=True, timeout=5)
    except Exception:
        pass
    time.sleep(0.4)
    try:
        traci.close()
    except Exception:
        pass


def _nan() -> float:
    return float("nan")


def _ratio(path_edges: list[str], edge_set: set[str]) -> float:
    if not path_edges:
        return _nan()
    return len(set(path_edges) & set(edge_set or set())) / max(len(path_edges), 1)


def _append_target_edge_if_connected(net, path_edges: list[str], end_edge: str) -> list[str]:
    path = [str(edge_id) for edge_id in list(path_edges or []) if str(edge_id).strip()]
    if not path or not end_edge or path[-1] == end_edge:
        return path
    try:
        outgoing = {edge_obj.getID() for edge_obj in net.getEdge(path[-1]).getOutgoing().keys()}
    except Exception:
        outgoing = set()
    return path + [end_edge] if end_edge in outgoing else path


def _resolve_scene_od(scene: str) -> tuple[str, str, dict[str, Any], list[str]]:
    warnings: list[str] = []
    config = ConfigManager().config
    try:
        cases = select_showcase_scenarios(config.get("SUMO_NET_PATH"))
    except Exception as exc:
        cases = {}
        warnings.append(f"warning: could not load showcase cases: {exc}")
    case = dict(cases.get(scene) or {})
    if case:
        return str(case.get("start_edge") or ""), str(case.get("end_edge") or ""), case, warnings
    warnings.append(f"warning: no showcase OD/path bundle found for {scene!r}; calibration cannot run real routes.")
    return "", "", {}, warnings


def _tripinfo_metrics(tripinfo_path: str, vehicle_id: str) -> dict[str, Any]:
    metrics = {
        "waiting_time_s": _nan(),
        "time_loss_s": _nan(),
        "stop_count": _nan(),
        "depart_delay_s": _nan(),
        "duration_s": _nan(),
        "arrival_s": _nan(),
        "vaporized": "",
        "trip_completed": False,
    }
    if not os.path.exists(tripinfo_path) or os.path.getsize(tripinfo_path) <= 10:
        return metrics
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(tripinfo_path)
        for ti in tree.getroot().findall("tripinfo"):
            if ti.get("id") != vehicle_id:
                continue
            metrics.update(
                {
                    "waiting_time_s": float(ti.get("waitingTime", _nan())),
                    "time_loss_s": float(ti.get("timeLoss", _nan())),
                    "depart_delay_s": float(ti.get("departDelay", _nan())),
                    "duration_s": float(ti.get("duration", _nan())),
                    "arrival_s": float(ti.get("arrival", _nan())),
                    "vaporized": str(ti.get("vaporized", "") or "").strip(),
                }
            )
            metrics["trip_completed"] = (
                math.isfinite(float(metrics["arrival_s"]))
                and float(metrics["arrival_s"]) >= 0
                and not metrics["vaporized"]
            )
            break
    except Exception:
        pass
    return metrics


def run_quick_route(
    *,
    path_edges: list[str],
    scene_profile,
    scene_weight_layers: dict[str, Any],
    scene_mode: str,
    label: str,
) -> dict[str, Any]:
    config = ConfigManager().config
    net = sumolib.net.readNet(config.get("SUMO_NET_PATH"))
    planner_vehicle_id = f"{config.get('SUMO_PLANNER_VEHICLE_ID', 'llm_veh')}_{label}"
    planner_route_id = f"{config.get('SUMO_PLANNER_ROUTE_ID', 'llm_route')}_{label}"
    planner_vtype = str(config.get("SUMO_PLANNER_VTYPE", "car"))
    planner_vclass = str(config.get("SUMO_PLANNER_VCLASS", "passenger"))
    step_len = float(config.get("STEP_LENGTH", 0.1))
    tripinfo_path = os.path.join(DEFAULT_OUTPUT_DIR, f"_tripinfo_{label}.xml")
    os.makedirs(os.path.dirname(tripinfo_path), exist_ok=True)
    try:
        os.remove(tripinfo_path)
    except Exception:
        pass

    affected_edges = (
        set(dict(scene_profile.incident_edges or {}).keys())
        | set(tuple(scene_profile.blocked_edges or ()))
        | set(dict(scene_profile.lane_reduction_edges or {}).keys())
        | set(dict(scene_weight_layers.get("spillover_weights") or {}).keys())
    )
    result = {
        "ok": False,
        "travel_time_s": _nan(),
        "waiting_time_s": _nan(),
        "time_loss_s": _nan(),
        "stop_count": _nan(),
        "depart_delay_s": _nan(),
        "route_length_edges": len(path_edges or []),
        "affected_edge_overlap_ratio": _ratio(path_edges, affected_edges),
        "path_runtime_sum_s": _nan(),
        "final_route_edges": [],
        "route_debug": {},
        "runtime_spillover_summary": {},
        "simulation_truncated": False,
        "completion_status": "not_started",
        "stage": "init",
        "reason": "",
    }
    if not path_edges:
        result["stage"] = "empty_path"
        result["reason"] = "path_edges_empty"
        return result

    _cleanup_sumo()
    injector = SUMOSceneProfileApplier(
        config,
        scene_profile,
        runtime_spillover_payload=dict(scene_weight_layers or {}),
        scene_mode=scene_mode,
    )
    result["runtime_spillover_summary"] = injector.runtime_spillover_summary()
    sumo_cmd = injector.build_sumo_command(tripinfo_path=tripinfo_path, step_length=step_len)

    try:
        traci.start(sumo_cmd)
        injector.bootstrap_after_start()
        warmup_target = max(int(scene_profile.warmup_seconds), int(scene_profile.evaluation_start_time))
        while traci.simulation.getTime() < warmup_target:
            traci.simulationStep()
            injector.on_simulation_step()

        route_debug = build_safe_vehicle_route(
            net,
            original_start_edge=path_edges[0],
            original_end_edge=path_edges[-1],
            original_route_edges=path_edges,
            vtype_id=planner_vtype,
            vclass=planner_vclass,
        )
        result["route_debug"] = route_debug
        result["stage"] = route_debug.get("stage", "sumo_route_build")
        result["reason"] = route_debug.get("reason", "unknown")
        if not route_debug.get("ok"):
            result["travel_time_s"] = max(
                0.0,
                float(scene_profile.evaluation_end_time) - float(scene_profile.evaluation_start_time),
            )
            return result

        final_route = list(route_debug.get("final_route_edges") or [])
        result["final_route_edges"] = final_route
        result["route_length_edges"] = len(final_route)
        result["affected_edge_overlap_ratio"] = _ratio(final_route, affected_edges)
        edge_times: dict[str, float] = {}
        for edge_id in final_route:
            try:
                edge_times[edge_id] = float(traci.edge.getTraveltime(edge_id))
            except Exception:
                edge_times[edge_id] = _nan()
        finite_edge_times = [value for value in edge_times.values() if math.isfinite(float(value))]
        result["edge_runtime_times_s"] = edge_times
        result["path_runtime_sum_s"] = float(sum(finite_edge_times)) if finite_edge_times else _nan()
        if math.isfinite(float(result["path_runtime_sum_s"])):
            result["ok"] = True
            result["travel_time_s"] = float(result["path_runtime_sum_s"])
            result["stage"] = "sumo_edge_runtime_quick"
            result["reason"] = "quick_edge_traveltime_sum"
            return result

        traci.route.add(planner_route_id, final_route)
        traci.vehicle.add(
            planner_vehicle_id,
            planner_route_id,
            depart="now",
            departLane=str(config.get("SUMO_DEPART_LANE", "best")),
            departPos=str(config.get("SUMO_DEPART_POS", "base")),
            departSpeed=str(config.get("SUMO_DEPART_SPEED", "0")),
            typeID=planner_vtype,
        )
        injector.register_planner_vehicle(planner_vehicle_id)
        depart_time = float(traci.simulation.getTime())
        active_seen = False
        evaluation_end_time = max(float(scene_profile.evaluation_end_time), depart_time + step_len)
        step = 0
        vehicle_last_speed: dict[str, float] = {}
        vehicle_stop_count: dict[str, int] = {}
        while step < 36000 and float(traci.simulation.getTime()) <= evaluation_end_time:
            traci.simulationStep()
            injector.on_simulation_step()
            active_vehicle_ids = traci.vehicle.getIDList()
            if planner_vehicle_id in active_vehicle_ids:
                active_seen = True
                speed_mps = float(traci.vehicle.getSpeed(planner_vehicle_id))
                if planner_vehicle_id not in vehicle_last_speed:
                    vehicle_last_speed[planner_vehicle_id] = speed_mps
                    vehicle_stop_count[planner_vehicle_id] = 0
                else:
                    if vehicle_last_speed[planner_vehicle_id] >= 0.1 and speed_mps < 0.1:
                        vehicle_stop_count[planner_vehicle_id] += 1
                    vehicle_last_speed[planner_vehicle_id] = speed_mps
            elif active_seen:
                result["ok"] = True
                result["completion_status"] = "arrived"
                break
            step += 1

        result["travel_time_s"] = max(0.0, float(traci.simulation.getTime()) - depart_time)
        result["stop_count"] = int(vehicle_stop_count.get(planner_vehicle_id, 0))
    except Exception as exc:
        result["stage"] = "sumo_runtime"
        result["reason"] = str(exc)
    finally:
        try:
            injector.restore_base_state()
        except Exception:
            pass
        try:
            traci.close()
        except Exception:
            pass

    for _ in range(12):
        parsed = _tripinfo_metrics(tripinfo_path, planner_vehicle_id)
        if math.isfinite(float(parsed.get("duration_s", _nan()))):
            result["travel_time_s"] = float(parsed["duration_s"])
            result["waiting_time_s"] = parsed["waiting_time_s"]
            result["time_loss_s"] = parsed["time_loss_s"]
            result["depart_delay_s"] = parsed["depart_delay_s"]
            result["simulation_truncated"] = not bool(parsed.get("trip_completed"))
            result["completion_status"] = (
                "arrived" if parsed.get("trip_completed") else "truncated_at_evaluation_end"
            )
            break
        time.sleep(0.2)
    return result


def calibrate(scene: str, profile_name: str, output_dir: str) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    start_edge, end_edge, _case, od_warnings = _resolve_scene_od(scene)
    warnings.extend(od_warnings)
    profile = get_scene_profile(profile_name)
    config = ConfigManager().config
    net = sumolib.net.readNet(config.get("SUMO_NET_PATH"))
    edge_ids = tuple(live_edge_ids())
    scene_weight_layers = build_environment_weight_layers(edge_ids, profile, enable_spillover=True)
    scene_weights = dict(scene_weight_layers.get("weights") or {})

    baseline_path, _baseline_cost = free_flow_route(net, start_edge, end_edge)
    planned_path, _planned_cost = dijkstra_route(net, start_edge, end_edge, scene_weights)
    baseline_path = _append_target_edge_if_connected(net, list(baseline_path or []), end_edge)
    planned_path = _append_target_edge_if_connected(net, list(planned_path or []), end_edge)
    if not baseline_path:
        warnings.append("warning: baseline path could not be generated.")
    if not planned_path:
        warnings.append("warning: planned path could not be generated.")

    planned = run_quick_route(
        path_edges=planned_path,
        scene_profile=profile,
        scene_weight_layers=scene_weight_layers,
        scene_mode="showcase",
        label="planned",
    )
    baseline = run_quick_route(
        path_edges=baseline_path,
        scene_profile=profile,
        scene_weight_layers=scene_weight_layers,
        scene_mode="showcase",
        label="baseline",
    )

    baseline_time = float(baseline.get("travel_time_s") or _nan())
    planned_time = float(planned.get("travel_time_s") or _nan())
    relative_gap_pct = (
        ((baseline_time - planned_time) / max(planned_time, 1.0)) * 100.0
        if math.isfinite(baseline_time) and math.isfinite(planned_time)
        else _nan()
    )
    if not math.isfinite(relative_gap_pct):
        fairness = "NO_RUNTIME_RESULT"
    elif relative_gap_pct < 10.0:
        fairness = "TOO_WEAK"
    elif relative_gap_pct > 35.0:
        fairness = "TOO_STRONG"
    else:
        fairness = "PASS_SHOWCASE_FAIRNESS"

    direct_affected = (
        set(dict(profile.incident_edges or {}).keys())
        | set(tuple(profile.blocked_edges or ()))
        | set(dict(profile.lane_reduction_edges or {}).keys())
    )
    spillover_edges = set(dict(scene_weight_layers.get("spillover_weights") or {}).keys())
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scene": scene,
        "profile": profile.scene_name,
        "scene_type": profile.scene_type,
        "start_edge": start_edge,
        "end_edge": end_edge,
        "baseline_path_edges": baseline_path,
        "planned_path_edges": planned_path,
        "baseline_travel_time": baseline_time,
        "planned_travel_time": planned_time,
        "relative_gap_pct": relative_gap_pct,
        "baseline_waiting_time": baseline.get("waiting_time_s", _nan()),
        "planned_waiting_time": planned.get("waiting_time_s", _nan()),
        "baseline_time_loss": baseline.get("time_loss_s", _nan()),
        "planned_time_loss": planned.get("time_loss_s", _nan()),
        "baseline_stop_count": baseline.get("stop_count", _nan()),
        "planned_stop_count": planned.get("stop_count", _nan()),
        "baseline_depart_delay_s": baseline.get("depart_delay_s", _nan()),
        "planned_depart_delay_s": planned.get("depart_delay_s", _nan()),
        "baseline_route_length_edges": baseline.get("route_length_edges", len(baseline_path)),
        "planned_route_length_edges": planned.get("route_length_edges", len(planned_path)),
        "baseline_incident_overlap_ratio": _ratio(baseline_path, direct_affected),
        "planned_incident_overlap_ratio": _ratio(planned_path, direct_affected),
        "baseline_spillover_overlap_ratio": _ratio(baseline_path, spillover_edges),
        "planned_spillover_overlap_ratio": _ratio(planned_path, spillover_edges),
        "baseline_affected_edge_overlap_ratio": baseline.get("affected_edge_overlap_ratio", _nan()),
        "planned_affected_edge_overlap_ratio": planned.get("affected_edge_overlap_ratio", _nan()),
        "fairness_status": fairness,
        "baseline_runtime": baseline,
        "planned_runtime": planned,
        "warnings": warnings,
    }
    save_summary(summary, output_dir)
    return summary, warnings


def save_summary(summary: dict[str, Any], output_dir: str) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_key = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(summary.get("scene") or "scene"))
    json_path = out_dir / f"fairness_{scene_key}.json"
    csv_path = out_dir / f"fairness_{scene_key}.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    scalar = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
        for key, value in summary.items()
        if key not in {"baseline_runtime", "planned_runtime"}
    }
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar.keys()))
        writer.writeheader()
        writer.writerow(scalar)
    return json_path, csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate showcase baseline/planned SUMO runtime fairness.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary, warnings = calibrate(args.scene, args.profile, args.output_dir)
    for warning in warnings:
        print(warning, file=sys.stderr)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"fairness_status={summary.get('fairness_status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
