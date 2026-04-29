#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib
import importlib.util
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    spaces = None

ROOT = Path("/root/autodl-tmp")
CODE_DIR = ROOT / "code"
RESULT_DIR = ROOT / "results" / "ppo_baseline"
SUMMARY_PATH = RESULT_DIR / "ppo_quick_summary.csv"
NOTES_PATH = RESULT_DIR / "ppo_quick_notes.md"
INSTALL_NOTES_PATH = RESULT_DIR / "install_notes.md"
BLOCKER_PATH = RESULT_DIR / "ppo_blocker_notes.md"
MODEL_PATH = RESULT_DIR / "ppo_quick_model"

TRAIN_TIMESTEPS = 20_000
EVAL_REPEATS = 5
MAX_EPISODE_STEPS = 16
SEED = 20260409
TABLE_READY_SUCCESS_THRESHOLD = 0.50

ACTION_NAMES = ("N", "E", "S", "W")
ACTION_TO_DELTA = {
    0: (1, 0),   # row + 1 matches the repo's "N" edge naming.
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}


def ensure_result_dir() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def write_install_notes() -> None:
    ensure_result_dir()
    lines = [
        "# stable-baselines3 install notes",
        "",
        "- Initial environment check in this session: `stable-baselines3`, `gymnasium`, and `gym` were not installed.",
        "- Install command used:",
        "```bash",
        "python -m pip install stable-baselines3 gymnasium",
        "```",
    ]

    for package in ("stable_baselines3", "gymnasium", "gym"):
        if module_exists(package):
            try:
                version = importlib.import_module(package).__version__
            except Exception:
                version = "unknown"
            lines.append(f"- Installed version: `{package}=={version}`")
        else:
            lines.append(f"- `{package}` is still unavailable after setup.")

    lines.extend(
        [
            "- SUMO-side dependencies already present in the environment: `sumolib`, `traci`, and `/usr/bin/sumo`.",
            "- Re-run command for the baseline:",
            "```bash",
            "python /root/autodl-tmp/rl_baseline/run_ppo_quick_baseline.py",
            "```",
        ]
    )
    INSTALL_NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_blocker_notes(blocker: str, alternative: str, appendix_only: bool) -> None:
    ensure_result_dir()
    lines = [
        "# PPO blocker notes",
        "",
        f"- Current blocker: {blocker}",
        f"- Lowest viable fallback: {alternative}",
        f"- Recommendation: {'appendix only' if appendix_only else 'can still be considered for the main table'}",
        "- This blocker note was produced because the quick PPO baseline could not deliver a stable reportable result inside the fixed budget.",
    ]
    BLOCKER_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_repo_modules():
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    from baselines import all_directed_edges, all_nodes
    from evaluate import SCENARIOS, calc_true_travel_time

    return SCENARIOS, all_nodes, all_directed_edges, calc_true_travel_time


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def patch_sb3_obs_as_tensor() -> None:
    import torch as th
    from stable_baselines3.common import buffers as sb_buffers
    import stable_baselines3.common.on_policy_algorithm as sb3_on_policy
    import stable_baselines3.common.policies as sb3_policies
    import stable_baselines3.common.utils as sb3_utils

    def _numpy_to_torch(value, device):
        value_arr = np.asarray(value)
        if np.issubdtype(value_arr.dtype, np.floating):
            return th.tensor(value_arr, device=device, dtype=th.float32)
        if np.issubdtype(value_arr.dtype, np.integer):
            return th.tensor(value_arr, device=device, dtype=th.int64)
        if np.issubdtype(value_arr.dtype, np.bool_):
            return th.tensor(value_arr, device=device, dtype=th.bool)
        return th.tensor(value_arr.tolist(), device=device)

    def _safe_obs_as_tensor(obs, device):
        if isinstance(obs, np.ndarray):
            return _numpy_to_torch(obs, device)
        if isinstance(obs, dict):
            tensor_dict = {}
            for key, value in obs.items():
                tensor_dict[key] = _numpy_to_torch(value, device)
            return tensor_dict
        raise TypeError(f"Unrecognized type of observation {type(obs)}")

    def _safe_to_torch(self, array, copy: bool = True):
        return _numpy_to_torch(array, self.device)

    def _safe_compute_returns_and_advantage(self, last_values, dones):
        last_values_arr = np.asarray(last_values.clone().cpu().tolist(), dtype=np.float32).reshape(-1)
        last_gae_lam = np.zeros_like(last_values_arr, dtype=np.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - np.asarray(dones, dtype=np.float32)
                next_values = last_values_arr
            else:
                next_non_terminal = 1.0 - np.asarray(self.episode_starts[step + 1], dtype=np.float32)
                next_values = np.asarray(self.values[step + 1], dtype=np.float32)
            rewards = np.asarray(self.rewards[step], dtype=np.float32)
            values = np.asarray(self.values[step], dtype=np.float32)
            delta = rewards + self.gamma * next_values * next_non_terminal - values
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = np.asarray(last_gae_lam, dtype=np.float32)
        self.returns = np.asarray(self.advantages, dtype=np.float32) + np.asarray(self.values, dtype=np.float32)

    sb3_utils.obs_as_tensor = _safe_obs_as_tensor
    sb3_on_policy.obs_as_tensor = _safe_obs_as_tensor
    sb3_policies.obs_as_tensor = _safe_obs_as_tensor
    sb_buffers.BaseBuffer.to_torch = _safe_to_torch
    sb_buffers.RolloutBuffer.compute_returns_and_advantage = _safe_compute_returns_and_advantage


class SumoRoutePPOEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenarios: Sequence[Dict],
        nodes: Sequence[Tuple[int, int]],
        directed_edges: Sequence[Tuple[Tuple[int, int], Tuple[int, int], str]],
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ) -> None:
        super().__init__()

        self.scenarios = list(scenarios)
        self.nodes = list(nodes)
        self.directed_edges = list(directed_edges)
        self.max_episode_steps = max_episode_steps
        self.node_index = {node: idx for idx, node in enumerate(self.nodes)}
        self.edge_ids = [eid for _, _, eid in self.directed_edges]
        self.edge_index = {eid: idx for idx, eid in enumerate(self.edge_ids)}
        self.max_row = max(node[0] for node in self.nodes)
        self.max_col = max(node[1] for node in self.nodes)
        self.transitions: Dict[Tuple[int, int], Dict[int, Tuple[Tuple[int, int], str]]] = {
            node: {} for node in self.nodes
        }
        for src, dst, eid in self.directed_edges:
            delta = (dst[0] - src[0], dst[1] - src[1])
            action = self._action_from_delta(delta)
            if action is not None:
                self.transitions[src][action] = (dst, eid)

        self.current_scenario_idx = 0
        self.current_scenario = self.scenarios[0]
        self.current_node = self.current_scenario["start"]
        self.goal_node = self.current_scenario["end"]
        self.steps_taken = 0
        self.travel_time = 0.0
        self.episode_reward = 0.0
        self.path = [self.current_node]
        self.edge_path: List[str] = []

        sample_obs = self._build_observation()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=sample_obs.shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(ACTION_NAMES))

    @staticmethod
    def _action_from_delta(delta: Tuple[int, int]) -> int | None:
        for action, action_delta in ACTION_TO_DELTA.items():
            if delta == action_delta:
                return action
        return None

    def _normalized_node(self, node: Tuple[int, int]) -> np.ndarray:
        row, col = node
        return np.array(
            [
                row / max(self.max_row, 1),
                col / max(self.max_col, 1),
            ],
            dtype=np.float32,
        )

    def _one_hot_node(self, node: Tuple[int, int]) -> np.ndarray:
        vec = np.zeros(len(self.nodes), dtype=np.float32)
        vec[self.node_index[node]] = 1.0
        return vec

    def _global_weight_vector(self) -> np.ndarray:
        weights = np.zeros(len(self.edge_ids), dtype=np.float32)
        true_w = self.current_scenario["true_w"]
        for eid, idx in self.edge_index.items():
            weights[idx] = float(true_w.get(eid, 2.0)) / 10.0
        return weights

    def _local_action_features(self) -> Tuple[np.ndarray, np.ndarray]:
        local_weights = np.ones(len(ACTION_NAMES), dtype=np.float32)
        valid_mask = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        true_w = self.current_scenario["true_w"]
        for action in range(len(ACTION_NAMES)):
            transition = self.transitions[self.current_node].get(action)
            if transition is None:
                continue
            _dst, eid = transition
            local_weights[action] = float(true_w.get(eid, 2.0)) / 10.0
            valid_mask[action] = 1.0
        return local_weights, valid_mask

    def _build_observation(self) -> np.ndarray:
        current_xy = self._normalized_node(self.current_node)
        goal_xy = self._normalized_node(self.goal_node)
        delta_xy = goal_xy - current_xy
        manhattan = np.array(
            [
                (
                    abs(self.goal_node[0] - self.current_node[0])
                    + abs(self.goal_node[1] - self.current_node[1])
                )
                / max(self.max_row + self.max_col, 1)
            ],
            dtype=np.float32,
        )
        local_weights, valid_mask = self._local_action_features()
        parts = [
            self._one_hot_node(self.current_node),
            self._one_hot_node(self.goal_node),
            current_xy,
            goal_xy,
            delta_xy.astype(np.float32),
            manhattan,
            local_weights,
            valid_mask,
            self._global_weight_vector(),
        ]
        return np.concatenate(parts).astype(np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        if options and "scenario_index" in options:
            scenario_idx = int(options["scenario_index"])
        else:
            scenario_idx = int(self.np_random.integers(0, len(self.scenarios)))
        self.current_scenario_idx = scenario_idx
        self.current_scenario = self.scenarios[scenario_idx]
        self.current_node = self.current_scenario["start"]
        self.goal_node = self.current_scenario["end"]
        self.steps_taken = 0
        self.travel_time = 0.0
        self.episode_reward = 0.0
        self.path = [self.current_node]
        self.edge_path = []
        return self._build_observation(), self._build_info(False, False)

    def _build_info(self, success: bool, timeout: bool) -> Dict:
        return {
            "scenario_id": self.current_scenario["id"],
            "scenario_name": self.current_scenario["name"],
            "success": bool(success),
            "timeout": bool(timeout),
            "travel_time": round(self.travel_time, 4),
            "steps": self.steps_taken,
            "path": list(self.path),
            "edge_path": list(self.edge_path),
            "episode_reward": round(self.episode_reward, 4),
        }

    def step(self, action: int):
        action = int(action)
        self.steps_taken += 1
        success = False
        timeout = False
        reward = 0.0
        transition = self.transitions[self.current_node].get(action)

        if transition is None:
            reward = -2.0
        else:
            next_node, eid = transition
            weight = float(self.current_scenario["true_w"].get(eid, 2.0))
            edge_travel_time = 1.0 + weight / 5.0
            self.current_node = next_node
            self.travel_time += edge_travel_time
            self.path.append(next_node)
            self.edge_path.append(eid)
            reward = -edge_travel_time
            if self.current_node == self.goal_node:
                reward += 20.0
                success = True

        if not success and self.steps_taken >= self.max_episode_steps:
            timeout = True
            reward -= 10.0

        self.episode_reward += reward
        terminated = success or timeout
        truncated = False
        return self._build_observation(), reward, terminated, truncated, self._build_info(success, timeout)


def evaluate_model(model, env, scenarios: Sequence[Dict]) -> List[Dict]:
    import torch

    def _predict_action(observation: np.ndarray) -> int:
        obs_tensor, _vectorized = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            action_tensor = model.policy._predict(obs_tensor, deterministic=True)
        action_value = action_tensor.detach().cpu().tolist()
        if isinstance(action_value, list):
            if action_value and isinstance(action_value[0], list):
                return int(action_value[0][0])
            if action_value:
                return int(action_value[0])
        return int(action_value)

    records: List[Dict] = []
    for repeat in range(EVAL_REPEATS):
        for scenario_idx, scenario in enumerate(scenarios):
            obs, _ = env.reset(
                seed=SEED + repeat * 100 + scenario_idx,
                options={"scenario_index": scenario_idx},
            )
            done = False
            planning_time_s = 0.0
            while not done:
                t0 = time.perf_counter()
                action = _predict_action(obs)
                planning_time_s += time.perf_counter() - t0
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            record = {
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "success": bool(info["success"]),
                "timeout": bool(info["timeout"]),
                "travel_time_s": float(info["travel_time"]),
                "planning_time_s": planning_time_s,
                "episode_reward": float(info["episode_reward"]),
                "steps": int(info["steps"]),
            }
            records.append(record)
    return records


def summarise_records(records: Sequence[Dict]) -> Dict[str, float]:
    success_flags = [1.0 if rec["success"] else 0.0 for rec in records]
    timeout_flags = [1.0 if rec["timeout"] else 0.0 for rec in records]
    success_travel_times = [rec["travel_time_s"] for rec in records if rec["success"]]
    return {
        "success rate": round(float(np.mean(success_flags)), 4) if success_flags else float("nan"),
        "Planning Time (s)": round(float(np.mean([rec["planning_time_s"] for rec in records])), 6) if records else float("nan"),
        "Travel Time (s)": round(float(np.mean(success_travel_times)), 4) if success_travel_times else float("nan"),
        "average reward": round(float(np.mean([rec["episode_reward"] for rec in records])), 4) if records else float("nan"),
        "failure rate": round(1.0 - float(np.mean(success_flags)), 4) if success_flags else float("nan"),
        "timeout rate": round(float(np.mean(timeout_flags)), 4) if timeout_flags else float("nan"),
    }


def write_summary(summary: Dict[str, float], eval_episodes: int, train_seconds: float, device: str) -> None:
    ensure_result_dir()
    fieldnames = [
        "method",
        "success rate",
        "Planning Time (s)",
        "Travel Time (s)",
        "average reward",
        "failure rate",
        "timeout rate",
        "training_steps",
        "eval_episodes",
        "train_wall_time_s",
        "device",
        "table_ready",
    ]
    row = {
        "method": "PPO (quick baseline, untuned)",
        "training_steps": TRAIN_TIMESTEPS,
        "eval_episodes": eval_episodes,
        "train_wall_time_s": round(train_seconds, 3),
        "device": device,
        "table_ready": "yes" if is_table_ready(summary) else "no",
    }
    row.update(summary)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_notes(summary: Dict[str, float], eval_episodes: int, device: str) -> None:
    ensure_result_dir()
    lines = [
        "# PPO quick baseline notes",
        "",
        "- PPO (quick baseline, untuned)",
        f"- Training steps: {TRAIN_TIMESTEPS}",
        f"- Evaluation episodes: {eval_episodes}",
        f"- Device: `{device}`",
        "- Environment reuse: reused the existing SUMO road topology from `code/baselines.py` and the scenario weight definitions from `code/evaluate.py`, then wrapped them in a minimal Gymnasium environment instead of adding a new simulator stack.",
        "- State definition: concatenation of current-node one-hot, goal-node one-hot, normalized current/goal/delta coordinates, Manhattan distance, 4 local outgoing edge weights, 4 local validity-mask bits, and the normalized global directed-edge weight vector for the current scenario.",
        "- Action definition: `Discrete(4)` over `N / E / S / W` moves on the existing 5x6 SUMO-aligned road graph; invalid moves stay in place and receive a penalty.",
        "- Reward: per-step negative travel time `-(1 + weight / 5)`, `+20` on reaching the goal, `-2` for invalid moves, `-10` on timeout.",
        f"- Training budget: single run, fixed seed `{SEED}`, `max_episode_steps={MAX_EPISODE_STEPS}`, no sweep, no deep tuning.",
        "- Metric definition: `Travel Time (s)` is averaged over successful evaluation episodes; `Planning Time (s)` is average PPO action-selection wall time per episode.",
        f"- Quick result snapshot: success rate={summary['success rate']}, travel time={summary['Travel Time (s)']}, average reward={summary['average reward']}, timeout rate={summary['timeout rate']}.",
        "- This baseline is only for RL category coverage and does not represent optimal PPO performance.",
    ]
    NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def is_table_ready(summary: Dict[str, float]) -> bool:
    success_rate = summary.get("success rate", float("nan"))
    travel_time = summary.get("Travel Time (s)", float("nan"))
    return bool(
        not math.isnan(success_rate)
        and success_rate >= TABLE_READY_SUCCESS_THRESHOLD
        and not math.isnan(travel_time)
    )


def main() -> int:
    ensure_result_dir()
    write_install_notes()
    set_global_seed(SEED)

    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    try:
        import gymnasium as gym  # noqa: F401
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except Exception as exc:
        write_blocker_notes(
            blocker=f"stable-baselines3 / gymnasium import failed: {exc}",
            alternative="Keep a scripted RL-placeholder row or report the PPO slot only in the appendix until the RL dependencies are installed.",
            appendix_only=True,
        )
        print("PPO 是否已达到“可入表”标准: 否")
        return 1

    patch_sb3_obs_as_tensor()

    scenarios, all_nodes, all_directed_edges, _calc_true_travel_time = load_repo_modules()
    nodes = all_nodes()
    directed_edges = all_directed_edges()

    train_env = Monitor(
        SumoRoutePPOEnv(
            scenarios=scenarios,
            nodes=nodes,
            directed_edges=directed_edges,
            max_episode_steps=MAX_EPISODE_STEPS,
        )
    )
    eval_env = SumoRoutePPOEnv(
        scenarios=scenarios,
        nodes=nodes,
        directed_edges=directed_edges,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    device = "cpu"
    policy_kwargs = {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=0,
        seed=SEED,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=TRAIN_TIMESTEPS, progress_bar=False)
    train_seconds = time.perf_counter() - t0
    model.save(str(MODEL_PATH))

    records = evaluate_model(model, eval_env, scenarios)
    summary = summarise_records(records)
    write_summary(summary, eval_episodes=len(records), train_seconds=train_seconds, device=device)
    write_notes(summary, eval_episodes=len(records), device=device)

    if is_table_ready(summary):
        if BLOCKER_PATH.exists():
            BLOCKER_PATH.unlink()
        print("PPO 是否已达到“可入表”标准: 是")
        return 0

    write_blocker_notes(
        blocker=(
            "The PPO quick baseline finished training but stayed below the reportable threshold "
            f"(success rate={summary['success rate']}, timeout rate={summary['timeout rate']})."
        ),
        alternative=(
            "Keep this PPO result as an appendix-only RL coverage experiment, or replace it "
            "with a simpler RL-lite row such as a tabular shortest-path policy on the same graph."
        ),
        appendix_only=True,
    )
    print("PPO 是否已达到“可入表”标准: 否")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
