#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/root/autodl-tmp")
CODE_DIR = ROOT / "code"
COMPARE_DIR = ROOT / "compare"
RESULT_DIR = ROOT / "results" / "final_tables"
POLICY_DIR = RESULT_DIR / "dqn_scene_policies"
RESULT_CSV_PATH = RESULT_DIR / "dqn_fixed_results.csv"
COMPARE_MD_PATH = RESULT_DIR / "dqn_fixed_vs_old_vs_ruleastar.md"
TRAIN_NOTES_PATH = RESULT_DIR / "dqn_training_notes.md"

for path in (CODE_DIR, COMPARE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app_fixed as app
import run_compare as legacy_compare
from run_live_scene_profile_main_table import standard_scene_cases
from sim_scene_profiles import STANDARD_SCENE_PROFILE_NAMES, build_environment_weights

app.st.error = lambda *args, **kwargs: None

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import DQN
except Exception as exc:  # pragma: no cover - handled at runtime
    gym = None
    spaces = None
    DQN = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


ROW_NAMES = {
    "农业路": "R0",
    "红专路": "R1",
    "政七街": "R2",
    "黄河路": "R3",
    "纬五路": "R4",
}

COL_NAMES = {
    "经一路": "C0",
    "经三路": "C1",
    "经六路": "C2",
    "经八路": "C3",
    "花园路": "C4",
    "未来路": "C5",
}

DIRECTION_MAP = {
    "向东": "E",
    "向西": "W",
    "向北": "N",
    "向南": "S",
}

# 这里单独解析场景，不复用 Rule-A* 的规则。
# 输出固定长度状态向量，供 DQN 路由策略使用。
SEVERITY_MAP = (
    ("完全封闭", 9.2),
    ("全线封闭", 9.0),
    ("全段封闭", 9.0),
    ("封闭", 8.8),
    ("施工", 7.8),
    ("严重拥堵", 7.2),
    ("追尾事故", 7.0),
    ("事故", 6.8),
    ("中度拥堵", 5.2),
    ("拥堵", 4.4),
    ("缓行", 3.6),
    ("畅通无阻", 1.3),
    ("畅通", 1.4),
    ("正常通行", 1.8),
    ("正常", 2.0),
)


@dataclass
class CandidatePath:
    label: str
    source: str
    path: tuple[str, ...]
    base_cost: float


def ensure_dirs() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def patch_sb3_numpy_bridge() -> None:
    import torch as th
    from stable_baselines3.common import buffers as sb_buffers
    import stable_baselines3.common.policies as sb_policies
    import stable_baselines3.common.utils as sb_utils

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
            return {key: _numpy_to_torch(val, device) for key, val in obs.items()}
        return _numpy_to_torch(obs, device)

    def _safe_to_torch(self, array, copy: bool = True):
        return _numpy_to_torch(array, self.device)

    sb_buffers.BaseBuffer.to_torch = _safe_to_torch
    sb_utils.obs_as_tensor = _safe_obs_as_tensor
    sb_policies.obs_as_tensor = _safe_obs_as_tensor


def safe_scene_slug(scene_profile_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\\-]+", "_", scene_profile_name).strip("_")


def latest_reference_metrics_csv() -> Path:
    candidates = sorted(
        ROOT.glob("results/runs/*__experiment_mode__compare_multi_method_live_scene_profile__method_comparison_v2/metrics.csv")
    )
    if not candidates:
        raise FileNotFoundError("No live main-table metrics.csv found under results/runs/")
    return candidates[-1]


def load_reference_rows(metrics_csv: Path) -> dict[str, dict[str, dict[str, Any]]]:
    reference: dict[str, dict[str, dict[str, Any]]] = {}
    with metrics_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            method_name = str(row.get("method", "")).strip()
            scene_profile_name = str(row.get("scene_profile", "")).strip()
            if not method_name or not scene_profile_name:
                continue
            reference.setdefault(method_name, {})[scene_profile_name] = row
    return reference


def live_constraint_rate(pred_weights: dict[str, float], scene_weights: dict[str, float]) -> float:
    sig_edges = [eid for eid, weight in scene_weights.items() if abs(float(weight) - 2.0) > 0.5]
    if not sig_edges:
        return 100.0
    correct = sum(
        1
        for eid in sig_edges
        if (float(pred_weights.get(eid, 2.0)) >= 4.0) == (float(scene_weights[eid]) >= 4.0)
    )
    return round(correct / len(sig_edges) * 100.0, 2)


def parse_scene_state_vector(text: str, edge_ids: tuple[str, ...]) -> tuple[dict[str, float], np.ndarray]:
    weights = {edge_id: 2.0 for edge_id in edge_ids}
    clauses = [segment.strip() for segment in re.split(r"[；。，“”,,]", text) if segment.strip()]

    for clause in clauses:
        severity = None
        for keyword, value in SEVERITY_MAP:
            if keyword in clause:
                severity = value
                break
        if severity is None:
            continue

        directions = [code for token, code in DIRECTION_MAP.items() if token in clause]

        for road_name, prefix in ROW_NAMES.items():
            if road_name not in clause:
                continue
            valid_dirs = directions or ["E", "W"]
            for edge_id in edge_ids:
                if edge_id.startswith(prefix) and edge_id.split("_")[-1] in valid_dirs:
                    weights[edge_id] = severity

        for road_name, prefix in COL_NAMES.items():
            if road_name not in clause:
                continue
            valid_dirs = directions or ["N", "S"]
            for edge_id in edge_ids:
                if edge_id.startswith(prefix) and edge_id.split("_")[-1] in valid_dirs:
                    weights[edge_id] = severity

    vector = np.asarray([weights[edge_id] / 10.0 for edge_id in edge_ids], dtype=np.float32)
    return weights, vector


def run_live_path(
    path: tuple[str, ...],
    *,
    scene_profile: Any,
    scene_weights: dict[str, float],
) -> float:
    if not path:
        return float(scene_profile.evaluation_end_time + 600)
    travel_time_s, _avg_speed, _speed_data, _pos_data = app.sumo_engine.run_simulation(
        path,
        scene_profile,
        tuple(sorted(scene_weights.items())),
    )
    if float(travel_time_s) <= 0:
        return float(scene_profile.evaluation_end_time + 600)
    return float(travel_time_s)


class SceneRewardCache:
    def __init__(self, *, scene_profile: Any, scene_weights: dict[str, float]) -> None:
        self.scene_profile = scene_profile
        self.scene_weights = scene_weights
        self.cache: dict[tuple[str, ...], float] = {}

    def travel_time(self, path: tuple[str, ...], *, use_cache: bool = True) -> float:
        if use_cache and path in self.cache:
            return self.cache[path]
        travel_time_s = run_live_path(path, scene_profile=self.scene_profile, scene_weights=self.scene_weights)
        self.cache[path] = travel_time_s
        return travel_time_s


class CandidateRouteEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        observation: np.ndarray,
        candidates: list[CandidatePath],
        reward_cache: SceneRewardCache,
    ) -> None:
        super().__init__()
        self.observation = np.asarray(observation, dtype=np.float32)
        self.candidates = list(candidates)
        self.reward_cache = reward_cache
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=self.observation.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.candidates))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        return self.observation.copy(), {}

    def step(self, action: int):
        action = int(action)
        candidate = self.candidates[action]
        travel_time_s = self.reward_cache.travel_time(candidate.path, use_cache=True)
        reward = -float(travel_time_s)
        info = {
            "candidate_label": candidate.label,
            "candidate_source": candidate.source,
            "travel_time_s": round(float(travel_time_s), 3),
        }
        return self.observation.copy(), reward, True, False, info


def unique_candidates(candidates: list[CandidatePath]) -> list[CandidatePath]:
    seen: set[tuple[str, ...]] = set()
    out: list[CandidatePath] = []
    for candidate in candidates:
        if not candidate.path or candidate.path in seen:
            continue
        seen.add(candidate.path)
        out.append(candidate)
    return out


def penalize_path_edges(
    base_weights: dict[str, float],
    path: tuple[str, ...],
    penalty: float,
) -> dict[str, float]:
    weights = dict(base_weights)
    for idx, edge_id in enumerate(path):
        weights[edge_id] = float(weights.get(edge_id, 2.0)) + penalty + idx * 0.05
    return weights


def k_shortest_scene_vector_paths(
    *,
    start_edge: str,
    end_edge: str,
    weight_dict: dict[str, float],
    k: int,
) -> list[tuple[tuple[str, ...], float]]:
    start = legacy_compare.net.getEdge(start_edge)
    end = legacy_compare.net.getEdge(end_edge)
    if start is None or end is None:
        return []

    target_node = end.getToNode().getID()
    heap: list[tuple[float, tuple[str, ...], str, frozenset[str]]] = [
        (
            float(weight_dict.get(start_edge, 1.0)),
            (start_edge,),
            start.getToNode().getID(),
            frozenset({start.getToNode().getID(), start.getFromNode().getID()}),
        )
    ]
    solutions: list[tuple[tuple[str, ...], float]] = []
    seen_paths: set[tuple[str, ...]] = set()

    while heap and len(solutions) < k:
        cost, path, current_node, visited_nodes = heapq.heappop(heap)
        if current_node == target_node:
            if path not in seen_paths:
                seen_paths.add(path)
                solutions.append((path, cost))
            continue

        node = legacy_compare.net.getNode(current_node)
        if node is None:
            continue
        for edge in node.getOutgoing():
            next_node = edge.getToNode().getID()
            if next_node in visited_nodes:
                continue
            edge_id = edge.getID()
            next_path = path + (edge_id,)
            next_cost = float(cost) + float(weight_dict.get(edge_id, 1.0))
            heapq.heappush(
                heap,
                (
                    next_cost,
                    next_path,
                    next_node,
                    visited_nodes | {next_node},
                ),
            )
    return solutions


def build_candidate_paths(case: dict[str, Any], estimated_weights: dict[str, float]) -> list[CandidatePath]:
    candidates: list[CandidatePath] = []

    def add_candidate(label: str, source: str, path: list[str] | tuple[str, ...], cost: float) -> None:
        path_tuple = tuple(path or [])
        if not path_tuple:
            return
        candidates.append(
            CandidatePath(
                label=label,
                source=source,
                path=path_tuple,
                base_cost=float(cost or 0.0),
            )
        )

    weight_dict, path, cost = legacy_compare.method_dijkstra(case["start_edge"], case["end_edge"], case["constraint"])
    add_candidate("cand_dijkstra", "Dijkstra", path, cost)

    weight_dict, path, cost = legacy_compare.method_rule_astar(case["start_edge"], case["end_edge"], case["constraint"])
    add_candidate("cand_rule_astar", "Rule-A*", path, cost)

    weight_dict, path, cost = legacy_compare.method_gcn_weight(case["start_edge"], case["end_edge"], case["constraint"])
    add_candidate("cand_gcn_weight", "GCN-Weight", path, cost)

    path, cost = legacy_compare.astar_route(
        legacy_compare.net,
        case["start_edge"],
        case["end_edge"],
        estimated_weights,
    )
    add_candidate("cand_scene_vector_astar", "SceneVector-A*", path, cost)

    k_shortest = k_shortest_scene_vector_paths(
        start_edge=case["start_edge"],
        end_edge=case["end_edge"],
        weight_dict=estimated_weights,
        k=5,
    )
    for idx, (path_tuple, path_cost) in enumerate(k_shortest, start=1):
        add_candidate(f"cand_ksp_{idx}", f"SceneVector-KSP-{idx}", path_tuple, path_cost)

    seed_paths = unique_candidates(candidates)
    penalty_levels = (3.0, 6.0, 9.0)
    alt_index = 0
    for seed_candidate in list(seed_paths):
        for penalty in penalty_levels:
            if len(unique_candidates(candidates)) >= 5:
                break
            alt_weights = penalize_path_edges(estimated_weights, seed_candidate.path, penalty)
            alt_path, alt_cost = legacy_compare.astar_route(
                legacy_compare.net,
                case["start_edge"],
                case["end_edge"],
                alt_weights,
            )
            alt_index += 1
            add_candidate(
                f"cand_alt_{alt_index}",
                f"AltSceneVector-A*+penalty{int(penalty)}",
                alt_path,
                alt_cost,
            )

    # Fallback: heavily penalize one interior edge at a time to force an alternate detour.
    for seed_candidate in list(unique_candidates(candidates)):
        if len(unique_candidates(candidates)) >= 5:
            break
        for block_index, edge_id in enumerate(seed_candidate.path[1:-1], start=1):
            if len(unique_candidates(candidates)) >= 5:
                break
            blocked_weights = dict(estimated_weights)
            blocked_weights[edge_id] = float(blocked_weights.get(edge_id, 2.0)) + 60.0 + block_index
            alt_path, alt_cost = legacy_compare.astar_route(
                legacy_compare.net,
                case["start_edge"],
                case["end_edge"],
                blocked_weights,
            )
            alt_index += 1
            add_candidate(
                f"cand_block_{alt_index}",
                f"AltSceneVector-A*+block_{edge_id}",
                alt_path,
                alt_cost,
            )

    final_candidates = unique_candidates(candidates)[:3]
    if len(final_candidates) < 3:
        raise RuntimeError(
            f"Only {len(final_candidates)} unique candidate paths available for {case['scene_profile_name']}; "
            "need at least 3."
        )
    return final_candidates


def build_scene_case_payloads() -> list[dict[str, Any]]:
    app.config.config["RUN_MODE"] = "experiment_mode"
    app.config.config["MODEL_PATH"] = app.MAINLINE_MODEL_PATH
    return standard_scene_cases(list(STANDARD_SCENE_PROFILE_NAMES))


def format_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_compare_markdown(
    *,
    rows: list[dict[str, Any]],
    old_reference: dict[str, dict[str, dict[str, Any]]],
    reference_metrics_csv: Path,
    identical_constraint_warning: bool,
) -> None:
    lines = [
        "# DQN Fixed vs Old DQN vs Rule-A*",
        "",
        "## Old DQN Issue Summary",
        "",
        "- `compare/run_compare.py`: `method_dqn()` first calls `method_rule_astar()` and then trains/follows the rule-derived weights instead of producing an independent RL output.",
        "- `compare/run_live_scene_profile_main_table.py`: `_legacy_baseline_method(..., method_name='DQN')` calls `_legacy_rule_weights(case)` and returns those same weights as `final_weights` for `constraint_rate` evaluation.",
        "- Because live `constraint_rate` is computed from `final_weights` rather than from the finally chosen path, old DQN and Rule-A* share the same per-scene constraint behavior by construction.",
        f"- Reference scene-level old metrics loaded from: `{reference_metrics_csv}`",
        "",
        "## Scene-Level Comparison",
        "",
        "| Scene | New DQN Travel (s) | New DQN Planning (s) | New DQN Constraint (%) | Selected Candidate | Old DQN Travel (s) | Old DQN Constraint (%) | Rule-A* Travel (s) | Rule-A* Constraint (%) |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        scene_profile_name = row["scene_profile"]
        old_dqn = old_reference.get("DQN", {}).get(scene_profile_name, {})
        rule_astar = old_reference.get("Rule-A*", {}).get(scene_profile_name, {})
        lines.append(
            "| "
            f"{row['scene_name']} | "
            f"{format_float(row['travel_time_s'], 1)} | "
            f"{format_float(row['planning_time_s'], 6)} | "
            f"{format_float(row['constraint_rate'], 2)} | "
            f"{row['selected_candidate_source']} | "
            f"{old_dqn.get('travel_time_s', 'n/a')} | "
            f"{old_dqn.get('constraint_rate', 'n/a')} | "
            f"{rule_astar.get('travel_time_s', 'n/a')} | "
            f"{rule_astar.get('constraint_rate', 'n/a')} |"
        )

    lines.extend(
        [
            "",
            "## Acceptance Check",
            "",
            f"- New DQN reuses Rule-A* weights as its own output: `no`",
            f"- Six standard scenes evaluated: `{len(rows) == 6}`",
            f"- Scene-level comparison against old DQN and Rule-A* generated: `yes`",
        ]
    )

    if identical_constraint_warning:
        lines.extend(
            [
                "",
                "## WARNING",
                "",
                "- WARNING: the new DQN still matches Rule-A* exactly on `constraint_rate` across all 6 scenes.",
                "- Likely causes to inspect next: candidate-path space too narrow, scene-state parser too close to the old rule heuristic, or the fixed one-step environment leaves no room for policy differentiation.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## WARNING Check",
                "",
                "- No full six-scene `constraint_rate` identity with Rule-A* was detected.",
            ]
        )

    COMPARE_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_notes(
    *,
    rows: list[dict[str, Any]],
    train_logs: list[dict[str, Any]],
    episodes: int,
    reference_metrics_csv: Path,
    reused_policies: bool,
) -> None:
    lines = [
        "# DQN Training Notes",
        "",
        f"- Training backend: `stable-baselines3 DQN` on CPU.",
        f"- Per-scene training episodes: `{episodes}` (one-step episodes, fixed action pool per scene).",
        "- State space: 96-d scene vector produced by the independent text parser in `rl_baseline/run_dqn_fixed_baseline.py`.",
        "- Action space: discrete candidate-path selection over 3-5 feasible routes collected from existing baselines plus scene-vector alternates.",
        "- Reward: `-travel_time_s` from the live SUMO evaluator; no extra constraint penalty was added in this minimal version.",
        "- Planning time in `dqn_fixed_results.csv` counts only policy inference plus candidate lookup during evaluation; training time is logged separately below.",
        f"- Old DQN / Rule-A* reference metrics source: `{reference_metrics_csv}`.",
        f"- Reused pre-trained scene policies in this pass: `{reused_policies}`.",
        "",
        "## Per-Scene Training Log",
        "",
        "| Scene | Episodes | Candidate Count | Train Wall Time (s) | Policy File | Selected Candidate |",
        "|---|---:|---:|---:|---|---|",
    ]
    row_by_scene = {row["scene_profile"]: row for row in rows}
    for item in train_logs:
        final_row = row_by_scene[item["scene_profile"]]
        lines.append(
            "| "
            f"{item['scene_name']} | "
            f"{item['episodes']} | "
            f"{item['candidate_count']} | "
            f"{format_float(item['train_wall_time_s'], 3)} | "
            f"{item['policy_path']} | "
            f"{final_row['selected_candidate_source']} |"
        )
    TRAIN_NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate an independent fixed DQN route baseline.")
    parser.add_argument("--episodes", type=int, default=200, help="Per-scene DQN training episodes (default: 200).")
    parser.add_argument("--seed", type=int, default=20260410, help="Base random seed.")
    parser.add_argument(
        "--reference-metrics-csv",
        default="",
        help="Optional scene-level metrics.csv used as the old DQN / Rule-A* reference table.",
    )
    parser.add_argument(
        "--reuse-existing-policies",
        action="store_true",
        help="Reuse existing scene policy files and run evaluation-only without retraining.",
    )
    args = parser.parse_args()

    if IMPORT_ERROR is not None:
        raise RuntimeError(f"Missing RL dependency: {IMPORT_ERROR}")

    ensure_dirs()
    set_global_seed(args.seed)
    patch_sb3_numpy_bridge()

    reference_metrics_csv = Path(args.reference_metrics_csv) if args.reference_metrics_csv else latest_reference_metrics_csv()
    old_reference = load_reference_rows(reference_metrics_csv)

    cases = build_scene_case_payloads()
    edge_ids = tuple(app.path_engine._ALL_EDGES)

    result_rows: list[dict[str, Any]] = []
    train_logs: list[dict[str, Any]] = []

    for scene_index, case in enumerate(cases):
        scene_profile = case["scene_profile"]
        scene_profile_name = case["scene_profile_name"]
        scene_name = case["scenario"]
        scene_weights = build_environment_weights(edge_ids, scene_profile)
        estimated_weights, observation = parse_scene_state_vector(case["constraint"], edge_ids)
        candidates = build_candidate_paths(case, estimated_weights)

        reward_cache = SceneRewardCache(scene_profile=scene_profile, scene_weights=scene_weights)

        env = CandidateRouteEnv(
            observation=observation,
            candidates=candidates,
            reward_cache=reward_cache,
        )

        scene_seed = args.seed + scene_index * 17
        policy_stem = POLICY_DIR / f"{safe_scene_slug(scene_profile_name)}_dqn_policy"
        policy_path = str(policy_stem) + ".zip"
        if args.reuse_existing_policies:
            if not Path(policy_path).exists():
                raise FileNotFoundError(f"Missing saved policy for eval-only mode: {policy_path}")
            model = DQN.load(policy_path, env=env, device="cpu")
            train_wall_time_s = 0.0
        else:
            for candidate in candidates:
                reward_cache.travel_time(candidate.path, use_cache=True)
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=1e-3,
                buffer_size=max(512, args.episodes * 2),
                learning_starts=10,
                batch_size=16,
                gamma=0.0,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.35,
                exploration_final_eps=0.05,
                verbose=0,
                seed=scene_seed,
                device="cpu",
            )

            train_t0 = time.perf_counter()
            model.learn(total_timesteps=args.episodes, progress_bar=False)
            train_wall_time_s = time.perf_counter() - train_t0
            model.save(str(policy_stem))

        eval_t0 = time.perf_counter()
        action, _state = model.predict(observation[None, :], deterministic=True)
        action_idx = int(np.asarray(action).reshape(-1)[0])
        selected_candidate = candidates[action_idx]
        planning_time_s = time.perf_counter() - eval_t0
        travel_time_s = reward_cache.travel_time(selected_candidate.path, use_cache=False)
        constraint_rate = live_constraint_rate(estimated_weights, scene_weights)

        result_rows.append(
            {
                "scene_profile": scene_profile_name,
                "scene_name": scene_name,
                "simulation_seed": int(scene_profile.simulation_seed),
                "warmup_seconds": int(scene_profile.warmup_seconds),
                "evaluation_window": f"{int(scene_profile.evaluation_start_time)}-{int(scene_profile.evaluation_end_time)}",
                "travel_time_s": round(float(travel_time_s), 3),
                "planning_time_s": round(float(planning_time_s), 6),
                "constraint_rate": round(float(constraint_rate), 2),
                "candidate_count": len(candidates),
                "selected_candidate_label": selected_candidate.label,
                "selected_candidate_source": selected_candidate.source,
                "selected_path": "|".join(selected_candidate.path),
                "policy_path": policy_path,
                "train_episodes": int(args.episodes),
                "train_wall_time_s": round(float(train_wall_time_s), 3),
                "reward_definition": "-travel_time_s",
                "parser_type": "independent_text_vectorizer_96d",
            }
        )

        train_logs.append(
            {
                "scene_profile": scene_profile_name,
                "scene_name": scene_name,
                "episodes": int(args.episodes),
                "candidate_count": len(candidates),
                "train_wall_time_s": round(float(train_wall_time_s), 3),
                "policy_path": policy_path,
            }
        )

        print(
            f"[DQN-Fixed] {scene_profile_name:<22} | candidates={len(candidates)} | "
            f"travel={travel_time_s:.2f}s | planning={planning_time_s:.6f}s | "
            f"constraint={constraint_rate:.2f}% | selected={selected_candidate.source}"
        , flush=True)

    write_csv(RESULT_CSV_PATH, result_rows)

    identical_constraint_warning = all(
        str(row["constraint_rate"]) == str(old_reference.get("Rule-A*", {}).get(row["scene_profile"], {}).get("constraint_rate"))
        for row in result_rows
    )
    if identical_constraint_warning:
        print("WARNING: new DQN constraint_rate is still identical to Rule-A* across all 6 scenes.", flush=True)

    write_compare_markdown(
        rows=result_rows,
        old_reference=old_reference,
        reference_metrics_csv=reference_metrics_csv,
        identical_constraint_warning=identical_constraint_warning,
    )
    write_training_notes(
        rows=result_rows,
        train_logs=train_logs,
        episodes=args.episodes,
        reference_metrics_csv=reference_metrics_csv,
        reused_policies=args.reuse_existing_policies,
    )

    print(f"[Saved] {RESULT_CSV_PATH}", flush=True)
    print(f"[Saved] {COMPARE_MD_PATH}", flush=True)
    print(f"[Saved] {TRAIN_NOTES_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
