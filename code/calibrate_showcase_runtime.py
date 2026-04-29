from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import sumolib
import traci

from apply_scene_profile_to_sumo import SUMOSceneProfileApplier
from config import ConfigManager
from congestion_propagation import SCENE_SPILLOVER_CONFIG
from demo_scene_selector import (
    DEFAULT_PRODUCTION_SHOWCASE_NAME,
    PRODUCTION_SHOWCASE_NAMES,
    select_showcase_scenarios,
)
from sim_scene_profiles import get_scene_profile
from sumo_runner import build_safe_vehicle_route


RESULTS_PATH = Path("/root/autodl-tmp/results/ui/showcase_runtime_calibration.json")
LOG_PATH = Path("/root/autodl-tmp/results/ui/showcase_runtime_calibration.log")
CURRENT_ACTUAL_EVAL_PATH = Path("/root/autodl-tmp/results/ui/showcase_runtime_actual_eval.json")

SHOWCASE_ORDER = (
    "blockage_detour_showcase",
    "directional_asymmetry_showcase",
    "propagation_range_showcase",
    "compound_disaster_showcase",
)
PRODUCTION_SHOWCASE_ORDER = tuple(PRODUCTION_SHOWCASE_NAMES)

SHOWCASE_TO_SCENE = {
    "blockage_detour_showcase": "core_blockage",
    "directional_asymmetry_showcase": "directional_asymmetry",
    "propagation_range_showcase": "propagation_range",
    "compound_disaster_showcase": "compound_disaster",
}


def _log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


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


def _route_overlap_ratio(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    shared = len(set(left) & set(right))
    return shared / max(len(left), len(right), 1)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _clone_runtime_config(scene_key: str) -> dict[str, Any]:
    runtime_cfg = copy.deepcopy(dict(SCENE_SPILLOVER_CONFIG[scene_key]["runtime"]))
    runtime_cfg["hop_speed_factor_bounds"] = {
        int(hop): [float(bounds[0]), float(bounds[1])]
        for hop, bounds in runtime_cfg.get("hop_speed_factor_bounds", {}).items()
    }
    runtime_cfg["hop_travel_time_multiplier"] = {
        int(hop): float(multiplier)
        for hop, multiplier in runtime_cfg.get("hop_travel_time_multiplier", {}).items()
    }
    return runtime_cfg


def _set_runtime_config(scene_key: str, runtime_cfg: dict[str, Any]) -> None:
    SCENE_SPILLOVER_CONFIG[scene_key]["runtime"] = {
        "hop_speed_factor_bounds": {
            int(hop): [float(bounds[0]), float(bounds[1])]
            for hop, bounds in dict(runtime_cfg.get("hop_speed_factor_bounds") or {}).items()
        },
        "hop_travel_time_multiplier": {
            int(hop): float(multiplier)
            for hop, multiplier in dict(runtime_cfg.get("hop_travel_time_multiplier") or {}).items()
        },
    }


def _candidate_from_base(
    base_cfg: dict[str, Any],
    *,
    hop1_speed_shift: float = 0.0,
    hop2_speed_shift: float = 0.0,
    hop1_tt_shift: float = 0.0,
    hop2_tt_shift: float = 0.0,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for hop, shift in ((1, hop1_speed_shift), (2, hop2_speed_shift)):
        bounds = list(cfg["hop_speed_factor_bounds"][hop])
        lower = _clamp(bounds[0] + shift, 0.45, 0.98)
        upper = _clamp(bounds[1] + shift, lower + 0.04, 0.99)
        cfg["hop_speed_factor_bounds"][hop] = [round(lower, 3), round(upper, 3)]
    cfg["hop_travel_time_multiplier"][1] = round(
        _clamp(cfg["hop_travel_time_multiplier"][1] + hop1_tt_shift, 1.0, 1.55),
        3,
    )
    cfg["hop_travel_time_multiplier"][2] = round(
        _clamp(cfg["hop_travel_time_multiplier"][2] + hop2_tt_shift, 1.0, 1.35),
        3,
    )
    return cfg


def build_scene_candidates(scene_key: str) -> list[dict[str, Any]]:
    base_cfg = _clone_runtime_config(scene_key)
    if scene_key in {"core_blockage", "propagation_range"}:
        recipes = [
            ("hop1_plus_hop2_minus_mid", dict(hop1_speed_shift=-0.03, hop2_speed_shift=0.03, hop1_tt_shift=0.06, hop2_tt_shift=-0.03)),
            ("hop1_plus_hop2_minus_sharp", dict(hop1_speed_shift=-0.04, hop2_speed_shift=0.04, hop1_tt_shift=0.08, hop2_tt_shift=-0.04)),
            ("mixed_balance_b", dict(hop1_speed_shift=-0.01, hop2_speed_shift=0.03, hop1_tt_shift=0.05, hop2_tt_shift=-0.02)),
        ]
    elif scene_key == "directional_asymmetry":
        recipes = [
            ("weaken_hop1_mid", dict(hop1_speed_shift=0.03, hop2_speed_shift=0.01, hop1_tt_shift=-0.05, hop2_tt_shift=-0.01)),
            ("weaken_hop1_more", dict(hop1_speed_shift=0.04, hop2_speed_shift=0.02, hop1_tt_shift=-0.06, hop2_tt_shift=-0.02)),
            ("protect_2hop", dict(hop1_speed_shift=0.02, hop2_speed_shift=0.04, hop1_tt_shift=-0.04, hop2_tt_shift=-0.03)),
        ]
    elif scene_key == "compound_disaster":
        recipes = [
            ("hop2_minus_mid", dict(hop1_speed_shift=0.0, hop2_speed_shift=0.04, hop1_tt_shift=0.03, hop2_tt_shift=-0.04)),
            ("protect_planned_detour", dict(hop1_speed_shift=0.01, hop2_speed_shift=0.06, hop1_tt_shift=0.0, hop2_tt_shift=-0.05)),
            ("balanced_relief", dict(hop1_speed_shift=-0.01, hop2_speed_shift=0.04, hop1_tt_shift=0.02, hop2_tt_shift=-0.03)),
        ]
    else:
        recipes = []

    candidates: list[tuple[str, dict[str, Any]]] = []
    for label, recipe in recipes:
        candidates.append((label, _candidate_from_base(base_cfg, **recipe)))

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label, cfg in candidates:
        key = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"label": label, "runtime": cfg})
    return deduped


def load_current_actual_reference(showcase_name: str) -> dict[str, Any] | None:
    if not CURRENT_ACTUAL_EVAL_PATH.exists():
        return None
    try:
        payload = json.loads(CURRENT_ACTUAL_EVAL_PATH.read_text())
    except Exception:
        return None
    return dict((payload.get("cases") or {}).get(showcase_name) or {}) or None


def runtime_route_time(path_edges: list[str], profile, scene_weight_layers: dict[str, Any], scene_mode: str) -> dict[str, Any]:
    config = ConfigManager().config
    _cleanup_sumo()

    injector = SUMOSceneProfileApplier(
        config,
        profile,
        runtime_spillover_payload=dict(scene_weight_layers or {}),
        scene_mode=scene_mode,
    )
    cmd = injector.build_sumo_command(
        tripinfo_path=config["SUMO_TRIPINFO"],
        step_length=config["STEP_LENGTH"],
    )

    edge_times: dict[str, float] = {}
    final_route_edges: list[str] = []
    try:
        traci.start(cmd)
        injector.bootstrap_after_start()
        warmup_target = max(int(profile.warmup_seconds), int(profile.evaluation_start_time))
        while traci.simulation.getTime() < warmup_target:
            traci.simulationStep()
            injector.on_simulation_step()

        for edge_id in list(path_edges or []):
            edge_times[edge_id] = float(traci.edge.getTraveltime(edge_id))
        return {
            "path_runtime_sum": float(sum(edge_times.values())),
            "edge_times": edge_times,
            "runtime_spillover_summary": injector.runtime_spillover_summary(),
            "final_route_edges": final_route_edges,
        }
    finally:
        try:
            injector.restore_base_state()
        except Exception:
            pass
        try:
            traci.close()
        except Exception:
            pass


def run_single_route(path_edges: list[str], scene_profile, scene_weight_layers: dict[str, Any], scene_mode: str) -> dict[str, Any]:
    config = ConfigManager().config
    planner_vehicle_id = str(config.get("SUMO_PLANNER_VEHICLE_ID", "llm_veh"))
    planner_route_id = str(config.get("SUMO_PLANNER_ROUTE_ID", "llm_route"))
    planner_vtype = str(config.get("SUMO_PLANNER_VTYPE", "car"))
    planner_vclass = str(config.get("SUMO_PLANNER_VCLASS", "passenger"))
    depart_lane = str(config.get("SUMO_DEPART_LANE", "best"))
    depart_pos = str(config.get("SUMO_DEPART_POS", "base"))
    depart_speed = str(config.get("SUMO_DEPART_SPEED", "0"))
    step_len = float(config.get("STEP_LENGTH", 0.1))
    tripinfo_path = str(config.get("SUMO_TRIPINFO", "/root/autodl-tmp/SUMO/tripinfo.xml"))
    net_path = str(config.get("SUMO_NET_PATH"))
    net = sumolib.net.readNet(net_path)

    _cleanup_sumo()
    try:
        if os.path.exists(tripinfo_path):
            os.remove(tripinfo_path)
    except Exception:
        pass

    injector = SUMOSceneProfileApplier(
        config,
        scene_profile,
        runtime_spillover_payload=dict(scene_weight_layers or {}),
        scene_mode=scene_mode,
    )
    result = {
        "ok": False,
        "travel_time_s": None,
        "runtime_spillover_summary": injector.runtime_spillover_summary(),
        "route_debug": {},
        "stage": "init",
    }
    sumo_cmd = injector.build_sumo_command(
        tripinfo_path=tripinfo_path,
        step_length=step_len,
    )

    try:
        traci.start(sumo_cmd)
        injector.bootstrap_after_start()
        warmup_target = max(int(scene_profile.warmup_seconds), int(scene_profile.evaluation_start_time))
        while traci.simulation.getTime() < warmup_target:
            traci.simulationStep()
            injector.on_simulation_step()

        route_debug = build_safe_vehicle_route(
            net,
            original_start_edge=path_edges[0] if path_edges else "",
            original_end_edge=path_edges[-1] if path_edges else "",
            original_route_edges=path_edges,
            vtype_id=planner_vtype,
            vclass=planner_vclass,
        )
        result["route_debug"] = route_debug
        result["stage"] = route_debug.get("stage", "sumo_route_build")
        if not route_debug.get("ok"):
            result["travel_time_s"] = max(
                0.0,
                float(scene_profile.evaluation_end_time) - float(scene_profile.evaluation_start_time),
            )
            return result

        traci.route.add(planner_route_id, list(route_debug.get("final_route_edges") or []))
        traci.vehicle.add(
            planner_vehicle_id,
            planner_route_id,
            depart="now",
            departLane=depart_lane,
            departPos=depart_pos,
            departSpeed=depart_speed,
            typeID=planner_vtype,
        )
        injector.register_planner_vehicle(planner_vehicle_id)
        depart_time = float(traci.simulation.getTime())
        active_seen = False
        evaluation_end_time = max(float(scene_profile.evaluation_end_time), depart_time + step_len)
        step = 0

        while step < 36000 and float(traci.simulation.getTime()) <= evaluation_end_time:
            traci.simulationStep()
            injector.on_simulation_step()
            active_vehicle_ids = traci.vehicle.getIDList()
            if planner_vehicle_id in active_vehicle_ids:
                active_seen = True
            elif active_seen:
                result["ok"] = True
                break
            step += 1

        result["travel_time_s"] = max(0.0, float(traci.simulation.getTime()) - depart_time)
        return result
    finally:
        try:
            injector.restore_base_state()
        except Exception:
            pass
        try:
            traci.close()
        except Exception:
            pass


def evaluate_showcase_case(showcase_name: str, *, include_runtime_diagnostics: bool = False) -> dict[str, Any]:
    config = ConfigManager().config
    cases = select_showcase_scenarios(config["SUMO_NET_PATH"])
    item = cases[showcase_name]
    scene_profile = get_scene_profile(item["scene_profile"])
    planned_actual = run_single_route(list(item["planned_path"]), scene_profile, item["scene_weight_layers"], "showcase")
    baseline_actual = run_single_route(list(item["baseline_path"]), scene_profile, item["scene_weight_layers"], "showcase")

    planned_time = float(
        planned_actual.get("travel_time_s")
        if planned_actual.get("travel_time_s") is not None
        else max(0.0, float(scene_profile.evaluation_end_time) - float(scene_profile.evaluation_start_time))
    )
    baseline_time = float(
        baseline_actual.get("travel_time_s")
        if baseline_actual.get("travel_time_s") is not None
        else max(0.0, float(scene_profile.evaluation_end_time) - float(scene_profile.evaluation_start_time))
    )
    saving_s = baseline_time - planned_time
    saving_ratio = saving_s / max(baseline_time, 1.0)
    planned_final_route = list((planned_actual.get("route_debug") or {}).get("final_route_edges") or [])
    baseline_final_route = list((baseline_actual.get("route_debug") or {}).get("final_route_edges") or [])

    planned_payload = {
        "travel_time_s": planned_time,
        "ok": bool(planned_actual.get("ok")),
        "used_sumo_fallback": bool((planned_actual.get("route_debug") or {}).get("used_sumo_fallback")),
        "final_route_edges": planned_final_route,
        "runtime_spillover_summary": planned_actual.get("runtime_spillover_summary"),
    }
    baseline_payload = {
        "travel_time_s": baseline_time,
        "ok": bool(baseline_actual.get("ok")),
        "used_sumo_fallback": bool((baseline_actual.get("route_debug") or {}).get("used_sumo_fallback")),
        "final_route_edges": baseline_final_route,
        "runtime_spillover_summary": baseline_actual.get("runtime_spillover_summary"),
    }

    if include_runtime_diagnostics:
        raw_only_layers = {
            "raw_only_weights": dict(item["scene_weight_layers"].get("raw_only_weights") or {}),
            "spillover_weights": {},
            "spillover_hop_by_edge": {},
        }
        planned_runtime_sum = runtime_route_time(list(item["planned_path"]), scene_profile, item["scene_weight_layers"], "showcase")
        baseline_runtime_sum = runtime_route_time(list(item["baseline_path"]), scene_profile, item["scene_weight_layers"], "showcase")
        planned_runtime_sum_benchmark = runtime_route_time(list(item["planned_path"]), scene_profile, raw_only_layers, "benchmark")
        baseline_runtime_sum_benchmark = runtime_route_time(list(item["baseline_path"]), scene_profile, raw_only_layers, "benchmark")
        planned_payload["path_runtime_sum_s"] = float(planned_runtime_sum.get("path_runtime_sum") or 0.0)
        planned_payload["benchmark_path_runtime_sum_s"] = float(planned_runtime_sum_benchmark.get("path_runtime_sum") or 0.0)
        planned_payload["showcase_minus_benchmark_runtime_sum_s"] = float(
            float(planned_runtime_sum.get("path_runtime_sum") or 0.0)
            - float(planned_runtime_sum_benchmark.get("path_runtime_sum") or 0.0)
        )
        baseline_payload["path_runtime_sum_s"] = float(baseline_runtime_sum.get("path_runtime_sum") or 0.0)
        baseline_payload["benchmark_path_runtime_sum_s"] = float(baseline_runtime_sum_benchmark.get("path_runtime_sum") or 0.0)
        baseline_payload["showcase_minus_benchmark_runtime_sum_s"] = float(
            float(baseline_runtime_sum.get("path_runtime_sum") or 0.0)
            - float(baseline_runtime_sum_benchmark.get("path_runtime_sum") or 0.0)
        )

    return {
        "showcase_name": showcase_name,
        "scene_profile": item["scene_profile"],
        "scene_type": SHOWCASE_TO_SCENE[showcase_name],
        "start_edge": item["start_edge"],
        "end_edge": item["end_edge"],
        "planned": planned_payload,
        "baseline": baseline_payload,
        "saving_seconds": float(saving_s),
        "saving_ratio": float(saving_ratio),
        "whether_positive_gain": bool(saving_s > 1e-9),
        "actual_route_overlap_ratio": _route_overlap_ratio(planned_final_route, baseline_final_route),
        "actual_route_distinct": bool(planned_final_route != baseline_final_route),
    }


def _scene_objective(case_result: dict[str, Any]) -> tuple[Any, ...]:
    planned = dict(case_result.get("planned") or {})
    saving_s = float(case_result.get("saving_seconds") or 0.0)
    return (
        1 if saving_s > 1e-9 else 0,
        1 if bool(planned.get("ok")) else 0,
        round(saving_s, 6),
        1 if bool(case_result.get("actual_route_distinct")) else 0,
        round(float(case_result.get("saving_ratio") or 0.0), 6),
        -round(float(case_result.get("actual_route_overlap_ratio") or 0.0), 6),
    )


def evaluate_scene_candidates(scene_key: str) -> dict[str, Any]:
    showcase_name = next(name for name, key in SHOWCASE_TO_SCENE.items() if key == scene_key)
    base_cfg = _clone_runtime_config(scene_key)
    candidates = build_scene_candidates(scene_key)
    scene_report = {
        "scene_type": scene_key,
        "showcase_name": showcase_name,
        "base_runtime": copy.deepcopy(base_cfg),
        "reference_current_actual": load_current_actual_reference(showcase_name),
        "candidates": [],
        "best_candidate": None,
    }

    for index, candidate in enumerate(candidates, start=1):
        label = str(candidate["label"])
        runtime_cfg = copy.deepcopy(candidate["runtime"])
        _set_runtime_config(scene_key, runtime_cfg)
        _log(f"scene={scene_key} candidate={label} index={index}/{len(candidates)} start")
        case_result = evaluate_showcase_case(showcase_name, include_runtime_diagnostics=False)
        entry = {
            "label": label,
            "runtime": runtime_cfg,
            "result": case_result,
            "objective": list(_scene_objective(case_result)),
        }
        scene_report["candidates"].append(entry)
        if scene_report["best_candidate"] is None or tuple(entry["objective"]) > tuple(scene_report["best_candidate"]["objective"]):
            scene_report["best_candidate"] = entry
        RESULTS_PATH.write_text(json.dumps(_runtime_results_snapshot(scene_reports={scene_key: scene_report}), ensure_ascii=False, indent=2))
        _log(
            f"scene={scene_key} candidate={label} "
            f"saving={case_result['saving_seconds']:.1f} "
            f"positive={case_result['whether_positive_gain']} "
            f"planned_ok={case_result['planned']['ok']} "
            f"baseline_ok={case_result['baseline']['ok']} "
            f"overlap={case_result['actual_route_overlap_ratio']:.2f}"
        )

    _set_runtime_config(scene_key, base_cfg)
    return scene_report


def _runtime_results_snapshot(*, scene_reports: dict[str, Any], final_evaluation: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scene_reports": scene_reports,
    }
    if final_evaluation is not None:
        payload["final_evaluation"] = final_evaluation
    return payload


def evaluate_final_configs(scene_runtime_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    original = {scene_key: _clone_runtime_config(scene_key) for scene_key in scene_runtime_map}
    try:
        for scene_key, runtime_cfg in scene_runtime_map.items():
            _set_runtime_config(scene_key, runtime_cfg)
        cases = {}
        positives = []
        for showcase_name in SHOWCASE_ORDER:
            case_result = evaluate_showcase_case(showcase_name, include_runtime_diagnostics=True)
            cases[showcase_name] = case_result
            if showcase_name in PRODUCTION_SHOWCASE_ORDER and case_result["whether_positive_gain"]:
                positives.append(showcase_name)
        return {
            "recommended_runtime": scene_runtime_map,
            "cases": cases,
            "positive_showcases": positives,
            "default_best_showcase": DEFAULT_PRODUCTION_SHOWCASE_NAME,
        }
    finally:
        for scene_key, runtime_cfg in original.items():
            _set_runtime_config(scene_key, runtime_cfg)


def main() -> None:
    LOG_PATH.write_text("", encoding="utf-8")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log("calibration:start")

    original_configs = {
        scene_key: _clone_runtime_config(scene_key)
        for scene_key in set(SHOWCASE_TO_SCENE.values())
    }

    scene_reports: dict[str, Any] = {}
    try:
        for scene_key in ("core_blockage", "directional_asymmetry", "propagation_range", "compound_disaster"):
            scene_reports[scene_key] = evaluate_scene_candidates(scene_key)
            RESULTS_PATH.write_text(
                json.dumps(_runtime_results_snapshot(scene_reports=scene_reports), ensure_ascii=False, indent=2)
            )

        recommended_runtime = {
            scene_key: copy.deepcopy(scene_reports[scene_key]["best_candidate"]["runtime"])
            for scene_key in scene_reports
            if scene_reports[scene_key].get("best_candidate")
        }
        final_evaluation = evaluate_final_configs(recommended_runtime)
        RESULTS_PATH.write_text(
            json.dumps(
                _runtime_results_snapshot(scene_reports=scene_reports, final_evaluation=final_evaluation),
                ensure_ascii=False,
                indent=2,
            )
        )
        _log("calibration:done")
    finally:
        for scene_key, runtime_cfg in original_configs.items():
            _set_runtime_config(scene_key, runtime_cfg)
        _cleanup_sumo()


if __name__ == "__main__":
    main()
