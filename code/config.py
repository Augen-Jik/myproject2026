"""Configuration utilities for the traffic planning app."""

import logging
import os

import torch


class ConfigManager:
    """Centralized runtime config with logging/bootstrap defaults."""

    def __init__(self):
        run_mode = os.getenv("RUN_MODE", "demo_mode")
        if run_mode not in {"demo_mode", "experiment_mode"}:
            logging.warning("Unknown RUN_MODE=%s, fallback to demo_mode", run_mode)
            run_mode = "demo_mode"
        self.config = {
            "MODEL_PATH": "/root/autodl-tmp/model_merged_sparse_v2_stage4_fix",
            "SUMO_NET_PATH": "/root/autodl-tmp/SUMO/net/my_net.net.xml",
            "SUMO_CONFIG_PATH": "/root/autodl-tmp/SUMO/config/my_config.sumocfg",
            "SUMO_TRIPINFO": "/root/autodl-tmp/SUMO/tripinfo.xml",
            "SUMO_BINARY": os.getenv("SUMO_BINARY", "sumo"),
            "SUMO_PLANNER_VEHICLE_ID": "llm_veh",
            "SUMO_PLANNER_ROUTE_ID": "llm_route",
            "SUMO_PLANNER_VTYPE": "car",
            "SUMO_PLANNER_VCLASS": "passenger",
            "SUMO_DEPART_LANE": "best",
            "SUMO_DEPART_POS": "base",
            "SUMO_DEPART_SPEED": "0",
            "SUMO_MIN_SPEED_SAMPLES": 5,
            "OSM_MAP_PATH": "/root/autodl-tmp/map.osm",
            "RUN_MODE": run_mode,
            "DEFAULT_SCENE_PROFILE": os.getenv("SIM_SCENE_PROFILE", "normal_baseline"),
            "STEP_LENGTH": 0.1,
            "FORMAT_CONSTRAINT": (
                "只输出权重，格式：路段ID:数字, 路段ID:数字\n"
                "示例：R0C0_E:8.5, C2R1_N:2.0\n"
                "权重范围0-10，越大越拥堵，不要输出其他文字。"
            ),
            "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
            "CACHE_DIR": "/root/autodl-tmp/.cache/huggingface",
            "LOG_FILE": "/root/autodl-tmp/logs/app.log",
            "RESULTS_DIR": "/root/autodl-tmp/results",
            "SCENE_PROFILE_NOTES_PATH": "/root/autodl-tmp/results/sim_scene_profiles_notes.md",
        }
        os.makedirs(self.config["RESULTS_DIR"], exist_ok=True)
        os.makedirs(os.path.dirname(self.config["LOG_FILE"]), exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            filename=self.config["LOG_FILE"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def get(self, key, default=None):
        return self.config.get(key, default)
