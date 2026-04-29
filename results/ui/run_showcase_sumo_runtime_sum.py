import json
import subprocess
import time
import traceback
from pathlib import Path

import traci

from apply_scene_profile_to_sumo import SUMOSceneProfileApplier
from config import ConfigManager
from demo_scene_selector import select_best_showcase_scenario, select_showcase_scenarios
from sim_scene_profiles import get_scene_profile


STATUS_PATH = Path("/root/autodl-tmp/results/ui/showcase_sumo_runtime_sum_status.log")
OUTPUT_PATH = Path("/root/autodl-tmp/results/ui/showcase_sumo_runtime_sum.json")


def _log(message: str) -> None:
    with STATUS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def runtime_route_time(path, profile, layers, scene_mode: str):
    config = ConfigManager().config
    try:
        subprocess.run(["pkill", "-f", "sumo"], capture_output=True, timeout=3)
        time.sleep(0.3)
    except Exception:
        pass
    try:
        traci.close()
    except Exception:
        pass

    injector = SUMOSceneProfileApplier(
        config,
        profile,
        runtime_spillover_payload=dict(layers or {}),
        scene_mode=scene_mode,
    )
    cmd = injector.build_sumo_command(
        tripinfo_path=config["SUMO_TRIPINFO"],
        step_length=config["STEP_LENGTH"],
    )
    _log(f"runtime_route_time:start scene={profile.scene_name} mode={scene_mode} path_edges={len(path)}")
    try:
        traci.start(cmd)
        injector.bootstrap_after_start()
        warmup_target = max(int(profile.warmup_seconds), int(profile.evaluation_start_time))
        while traci.simulation.getTime() < warmup_target:
            traci.simulationStep()
            injector.on_simulation_step()
        total = 0.0
        edge_times = {}
        for edge_id in path:
            travel_time = float(traci.edge.getTraveltime(edge_id))
            edge_times[edge_id] = travel_time
            total += travel_time
        summary = injector.runtime_spillover_summary()
        _log(f"runtime_route_time:done scene={profile.scene_name} total={total:.2f}")
        return total, edge_times, summary
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
    config = ConfigManager().config
    showcases = select_showcase_scenarios(config["SUMO_NET_PATH"])
    best_case = select_best_showcase_scenario(config["SUMO_NET_PATH"])
    payload = {
        "default_best_showcase": (best_case or {}).get("showcase_name"),
        "runtime_probe": {},
        "cases": {},
    }

    probe_case = showcases["propagation_range_showcase"]
    probe_profile = get_scene_profile(probe_case["scene_profile"])
    probe_layers = probe_case["scene_weight_layers"]
    raw_only_layers = {
        "raw_only_weights": dict(probe_layers.get("raw_only_weights") or {}),
        "spillover_weights": {},
        "spillover_hop_by_edge": {},
    }
    benchmark_total, benchmark_edge_times, benchmark_summary = runtime_route_time(
        probe_case["planned_path"], probe_profile, raw_only_layers, "benchmark"
    )
    showcase_total, showcase_edge_times, showcase_summary = runtime_route_time(
        probe_case["planned_path"], probe_profile, probe_layers, "showcase"
    )
    probe_1hop = next((edge_id for edge_id, hop in dict(probe_layers.get("spillover_hop_by_edge") or {}).items() if int(hop) == 1), None)
    probe_2hop = next((edge_id for edge_id, hop in dict(probe_layers.get("spillover_hop_by_edge") or {}).items() if int(hop) == 2), None)
    payload["runtime_probe"] = {
        "benchmark_runtime_spillover_summary": benchmark_summary,
        "showcase_runtime_spillover_summary": showcase_summary,
        "probe_1hop_edge": probe_1hop,
        "probe_1hop_benchmark_tt": benchmark_edge_times.get(probe_1hop),
        "probe_1hop_showcase_tt": showcase_edge_times.get(probe_1hop),
        "probe_2hop_edge": probe_2hop,
        "probe_2hop_benchmark_tt": benchmark_edge_times.get(probe_2hop),
        "probe_2hop_showcase_tt": showcase_edge_times.get(probe_2hop),
        "planned_path_benchmark_tt": benchmark_total,
        "planned_path_showcase_tt": showcase_total,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    for showcase_name, item in showcases.items():
        _log(f"main:case_start {showcase_name}")
        profile = get_scene_profile(item["scene_profile"])
        planned_time, _, runtime_summary = runtime_route_time(
            item["planned_path"], profile, item["scene_weight_layers"], "showcase"
        )
        baseline_time, _, _ = runtime_route_time(
            item["baseline_path"], profile, item["scene_weight_layers"], "showcase"
        )
        payload["cases"][showcase_name] = {
            "scene_profile": item["scene_profile"],
            "start_edge": item["start_edge"],
            "end_edge": item["end_edge"],
            "planned_time_s": planned_time,
            "baseline_time_s": baseline_time,
            "saving_s": baseline_time - planned_time,
            "runtime_spillover_summary": runtime_summary,
        }
        OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        _log(f"main:case_done {showcase_name}")

    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    _log("main:done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _log(f"main:exception {exc}")
        _log(traceback.format_exc())
        raise
