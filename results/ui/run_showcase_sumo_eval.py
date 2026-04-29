import json
from pathlib import Path

from demo_scene_selector import select_best_showcase_scenario, select_showcase_scenarios
from sim_scene_profiles import get_scene_profile

import app_fixed


def main() -> None:
    net_path = "/root/autodl-tmp/SUMO/net/my_net.net.xml"
    output_path = Path("/root/autodl-tmp/results/ui/showcase_sumo_eval.json")
    showcase_cases = select_showcase_scenarios(net_path)
    best_case = select_best_showcase_scenario(net_path)
    results = {
        "default_best_showcase": (best_case or {}).get("showcase_name"),
        "cases": {},
    }

    for showcase_name, item in showcase_cases.items():
        profile = get_scene_profile(item["scene_profile"])
        planned_result = app_fixed.sumo_engine.run_simulation(
            tuple(item["planned_path"]),
            profile,
            tuple(sorted(item["scene_weights"].items())),
            scene_weight_layer_payload=item["scene_weight_layers"],
            scene_mode="showcase",
        )
        baseline_result = app_fixed.sumo_engine.run_simulation(
            tuple(item["baseline_path"]),
            profile,
            tuple(sorted(item["scene_weights"].items())),
            scene_weight_layer_payload=item["scene_weight_layers"],
            scene_mode="showcase",
        )
        planned_time = planned_result.get("travel_time")
        baseline_time = baseline_result.get("travel_time")
        saving_s = None
        if planned_time is not None and baseline_time is not None:
            saving_s = float(baseline_time) - float(planned_time)
        results["cases"][showcase_name] = {
            "scene_profile": item["scene_profile"],
            "start_edge": item["start_edge"],
            "end_edge": item["end_edge"],
            "planned_time_s": planned_time,
            "baseline_time_s": baseline_time,
            "saving_s": saving_s,
            "planned_ok": planned_result.get("ok"),
            "baseline_ok": baseline_result.get("ok"),
            "planned_speed_status": planned_result.get("speed_status"),
            "baseline_speed_status": baseline_result.get("speed_status"),
            "planned_runtime_spillover_summary": planned_result.get("runtime_spillover_summary"),
            "baseline_runtime_spillover_summary": baseline_result.get("runtime_spillover_summary"),
        }

        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
