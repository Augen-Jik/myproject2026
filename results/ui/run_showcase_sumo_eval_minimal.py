import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import sumolib
import traci

from apply_scene_profile_to_sumo import SUMOSceneProfileApplier
from config import ConfigManager
from demo_scene_selector import select_best_showcase_scenario, select_showcase_scenarios
from sim_scene_profiles import get_scene_profile
from sumo_runner import build_safe_vehicle_route


STATUS_PATH = Path("/root/autodl-tmp/results/ui/showcase_sumo_eval_status.log")


def _log(message: str) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def run_single_route(path_edges, scene_profile, scene_weight_layers, scene_mode: str) -> dict:
    _log(f"run_single_route:start scene={scene_profile.scene_name} mode={scene_mode} edges={len(path_edges)}")
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

    try:
        subprocess.run(["pkill", "-f", "sumo"], capture_output=True, timeout=3)
        time.sleep(0.6)
    except Exception:
        pass
    try:
        traci.close()
    except Exception:
        pass
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
        _log(f"traci:start_ok scene={scene_profile.scene_name}")
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
            _log(f"route_debug:not_ok scene={scene_profile.scene_name} reason={route_debug.get('reason')}")
            injector.restore_base_state()
            traci.close()
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
            active_vehicle_ids = set(traci.vehicle.getIDList())
            if planner_vehicle_id in active_vehicle_ids:
                active_seen = True
            elif active_seen:
                result["ok"] = True
                break
            step += 1

        result["travel_time_s"] = max(0.0, float(traci.simulation.getTime()) - depart_time)
        _log(
            f"run_single_route:done scene={scene_profile.scene_name} ok={result['ok']} travel_time={result['travel_time_s']}"
        )
        return result
    except Exception as exc:
        _log(f"run_single_route:exception scene={scene_profile.scene_name} exc={exc}")
        _log(traceback.format_exc())
        raise
    finally:
        try:
            injector.restore_base_state()
        except Exception:
            pass
        try:
            traci.close()
        except Exception:
            pass


def main() -> None:
    _log("main:start")
    net_path = "/root/autodl-tmp/SUMO/net/my_net.net.xml"
    output_path = Path("/root/autodl-tmp/results/ui/showcase_sumo_eval.json")
    showcase_cases = select_showcase_scenarios(net_path)
    best_case = select_best_showcase_scenario(net_path)
    payload = {
        "default_best_showcase": (best_case or {}).get("showcase_name"),
        "cases": {},
    }
    for showcase_name, item in showcase_cases.items():
        _log(f"main:case_start {showcase_name}")
        profile = get_scene_profile(item["scene_profile"])
        planned_result = run_single_route(
            list(item["planned_path"]),
            profile,
            item["scene_weight_layers"],
            "showcase",
        )
        baseline_result = run_single_route(
            list(item["baseline_path"]),
            profile,
            item["scene_weight_layers"],
            "showcase",
        )
        planned_time = planned_result.get("travel_time_s")
        baseline_time = baseline_result.get("travel_time_s")
        saving_s = None
        if planned_time is not None and baseline_time is not None:
            saving_s = float(baseline_time) - float(planned_time)
        payload["cases"][showcase_name] = {
            "scene_profile": item["scene_profile"],
            "start_edge": item["start_edge"],
            "end_edge": item["end_edge"],
            "planned_time_s": planned_time,
            "baseline_time_s": baseline_time,
            "saving_s": saving_s,
            "planned_ok": planned_result.get("ok"),
            "baseline_ok": baseline_result.get("ok"),
            "planned_runtime_spillover_summary": planned_result.get("runtime_spillover_summary"),
            "baseline_runtime_spillover_summary": baseline_result.get("runtime_spillover_summary"),
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        _log(f"main:case_done {showcase_name}")

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    _log("main:done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _log(f"main:exception {exc}")
        _log(traceback.format_exc())
        raise
