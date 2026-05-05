#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/root/autodl-tmp")
CODE_DIR = ROOT / "code"
COMPARE_DIR = ROOT / "compare"
FINAL_TABLE_DIR = ROOT / "results" / "final_tables"
FINAL_TABLE_PATH = FINAL_TABLE_DIR / "method_comparison_final.csv"
PREVIEW_PATH = FINAL_TABLE_DIR / "method_comparison_final_preview.md"
TABLE_NOTES_PATH = FINAL_TABLE_DIR / "table_notes.md"
DCRNN_RESULTS_PATH = FINAL_TABLE_DIR / "dcrnn_inspired_results.csv"
DCRNN_NOTES_PATH = FINAL_TABLE_DIR / "dcrnn_inspired_notes.md"
MAPPO_NOTE_PATH = FINAL_TABLE_DIR / "mappo_scope_note.md"
TEX_PATH = ROOT / "results" / "paper_ready" / "table1_method_comparison_final.tex"
LIVE_METRICS_PATH = ROOT / "results" / "runs" / "20260410T022825Z__experiment_mode__compare_multi_method_live_scene_profile__method_comparison_v2" / "metrics.csv"

for path in (CODE_DIR, COMPARE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app_fixed as app
import run_compare as legacy_compare
import run_live_scene_profile_main_table as live_main
from eval_metrics import compute_edge_metrics
from sim_scene_profiles import STANDARD_SCENE_PROFILE_NAMES, build_environment_weights, get_scene_profile_version


app.st.error = lambda *args, **kwargs: None

FIELD_ORDER = [
    "Method",
    "Method (CN)",
    "Category",
    "Travel Time (s)",
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

SCENE_FIELD_ORDER = [
    "method",
    "scenario",
    "preset_id",
    "scene_profile",
    "start_edge",
    "end_edge",
    "travel_time_s",
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
    "notes",
]

CATEGORY_ORDER = {
    "Baseline": 0,
    "GNN": 1,
    "RL": 2,
    "LLM-Prompt": 3,
    "Ours": 4,
    "Ours-Extended": 5,
}

METHOD_ORDER = {
    "Dijkstra": 0,
    "Rule-A*": 1,
    "GCN-Weight (Kipf & Welling)": 2,
    "DCRNN-inspired": 3,
    "DQN-RouteSelector": 4,
    "PPO": 5,
    "CoT-Qwen": 6,
    "R1-Raw": 7,
    "Sparse-LoRA": 8,
    "Sparse-LoRA+GAT": 9,
}

DCRNN_NOTE = (
    "This baseline is DCRNN-inspired and only reuses diffusion-style graph aggregation "
    "on a fixed graph; it is not a full DCRNN reproduction."
)

GCN_NOTE = (
    "Lightweight graph-weight baseline with citation anchor to Kipf & Welling (2017); "
    "the repo implementation remains a minimal graph estimator under the live SUMO chain."
)

COT_NOTE = (
    "Qwen2.5-1.5B-Instruct + chain-of-thought prompting baseline "
    "(Wei et al., 2022), rerun on the live scene-profile-fixed SUMO protocol."
)

R1_NOTE = (
    "DeepSeek-R1 raw reasoning prompt baseline, rerun on the live scene-profile-fixed "
    "SUMO protocol for citation-ready comparison."
)

MAPPO_NOTE = """# MAPPO Scope Note

- MAPPO is citation-relevant, but it is not a lowest-cost submission item in the current repository state.
- The current codebase does not define a stable multi-agent decomposition for the 96-edge task.
- Per-agent observation spaces are not specified.
- A joint action space or shared-policy coordination interface is not specified.
- A cooperative reward function for multiple agents is not specified.
- No compatible multi-agent SUMO wrapper was found; the current live chain is single-policy / single-route oriented.
- Therefore MAPPO is deferred in this batch instead of being implemented speculatively.

Minimal next step if revisited later:

- First freeze an agentization scheme over intersections or edge clusters.
- Then define per-agent observations, shared/global state, cooperative reward, and a PettingZoo-style or equivalent SUMO wrapper before model work starts.
"""


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _fmt(value: Any, digits: int) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _latex_escape(text: Any) -> str:
    return str(text).replace("&", "\\&")


def _standard_cases() -> list[dict[str, Any]]:
    return live_main.standard_scene_cases(list(STANDARD_SCENE_PROFILE_NAMES))


def _build_directed_neighbors() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    edge_set = set(legacy_compare.ALL_EDGES)
    forward: dict[str, tuple[str, ...]] = {}
    backward: dict[str, tuple[str, ...]] = {}
    for edge_id in legacy_compare.ALL_EDGES:
        edge = legacy_compare.net.getEdge(edge_id)
        forward[edge_id] = tuple(
            nbr.getID()
            for nbr in edge.getToNode().getOutgoing()
            if nbr.getID() in edge_set and nbr.getID() != edge_id
        )
        backward[edge_id] = tuple(
            nbr.getID()
            for nbr in edge.getFromNode().getIncoming()
            if nbr.getID() in edge_set and nbr.getID() != edge_id
        )
    return forward, backward


def _diffusion_step(values: dict[str, float], neighbors: dict[str, tuple[str, ...]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for edge_id in legacy_compare.ALL_EDGES:
        nbrs = neighbors[edge_id]
        if not nbrs:
            out[edge_id] = float(values[edge_id])
        else:
            out[edge_id] = float(sum(values[nbr] for nbr in nbrs) / len(nbrs))
    return out


def method_dcrnn_inspired(start_edge: str, end_edge: str, desc: str) -> tuple[dict[str, float], list[str], float]:
    base_weights, _base_path, _base_cost = legacy_compare.method_gcn_weight(start_edge, end_edge, desc)
    forward_neighbors, backward_neighbors = _build_directed_neighbors()

    forward_1 = _diffusion_step(base_weights, forward_neighbors)
    backward_1 = _diffusion_step(base_weights, backward_neighbors)
    forward_2 = _diffusion_step(forward_1, forward_neighbors)
    backward_2 = _diffusion_step(backward_1, backward_neighbors)

    final_weights: dict[str, float] = {}
    for edge_id in legacy_compare.ALL_EDGES:
        value = (
            0.55 * float(base_weights[edge_id])
            + 0.20 * float(forward_1[edge_id])
            + 0.15 * float(backward_1[edge_id])
            + 0.06 * float(forward_2[edge_id])
            + 0.04 * float(backward_2[edge_id])
        )
        if float(base_weights[edge_id]) >= 7.0:
            value = max(value, float(base_weights[edge_id]) * 0.92)
        final_weights[edge_id] = round(min(max(value, 0.5), 10.0), 2)

    path, total_cost = legacy_compare.astar_route(
        legacy_compare.net,
        start_edge,
        end_edge,
        final_weights,
    )
    return final_weights, list(path or []), float(total_cost or 0.0)


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    weights, path, total_cost = method_dcrnn_inspired(
        case["start_edge"],
        case["end_edge"],
        case["constraint"],
    )
    planning_time_s = time.perf_counter() - t0

    scene_profile = case["scene_profile"]
    scene_weights = build_environment_weights(app.path_engine._ALL_EDGES, scene_profile)
    travel_time_s, avg_speed, _speed_data, _pos_data = app.sumo_engine.run_simulation(
        tuple(path),
        scene_profile,
        tuple(sorted(scene_weights.items())),
    )
    edge_metrics = compute_edge_metrics(
        final_weights=weights,
        explicit_weights=weights,
        all_edges=app.path_engine._ALL_EDGES,
        method_name="DCRNN-inspired",
    )
    protocol_snapshot = live_main.protocol_to_dict(live_main.build_sim_eval_protocol(scene_profile))
    return {
        "method": "DCRNN-inspired",
        "scenario": case["scenario"],
        "preset_id": case["preset_id"],
        "scene_profile": case["scene_profile_name"],
        "start_edge": case["start_edge"],
        "end_edge": case["end_edge"],
        "travel_time_s": round(float(travel_time_s), 3),
        "planning_time_s": round(float(planning_time_s), 6),
        "model_infer_time_s": 0.0,
        "route_solve_time_s": round(float(planning_time_s), 6),
        "constraint_rate": live_main._constraint_rate(weights, scene_weights),
        "signal_ratio": round(float(edge_metrics["signal_ratio"]), 3),
        "signal_edges": int(edge_metrics["signal_edges"]),
        "parsed_edges": int(edge_metrics["parsed_edges"]),
        "parsed_edge_ratio": round(float(edge_metrics["parsed_edge_ratio"]), 3),
        "default_edges": int(edge_metrics["default_edges"]),
        "path_len": len(path),
        "path_cost": round(total_cost, 6),
        "avg_speed_kmh": round(float(avg_speed), 3),
        "simulation_seed": int(scene_profile.simulation_seed),
        "warmup_seconds": int(scene_profile.warmup_seconds),
        "evaluation_start_time": int(scene_profile.evaluation_start_time),
        "evaluation_end_time": int(scene_profile.evaluation_end_time),
        "scene_profile_version": get_scene_profile_version(),
        "protocol_snapshot_json": json.dumps(protocol_snapshot, ensure_ascii=False, sort_keys=True),
        "notes": DCRNN_NOTE,
    }


def _load_existing_rows() -> list[dict[str, str]]:
    with FINAL_TABLE_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_preview(rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["Category"]].append(row)

    lines = [
        "# Method Comparison Final Preview",
        "",
        "> Updated in this batch: citation-ready naming was normalized, `GCN-Weight (Kipf & Welling)` was relabeled, and `DCRNN-inspired` was added under `GNN`.",
        "",
    ]
    for category in ("Baseline", "GNN", "RL", "LLM-Prompt", "Ours", "Ours-Extended"):
        items = grouped.get(category, [])
        if not items:
            continue
        lines.extend(
            [
                f"## {category}",
                "",
                "| Method | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in items:
            lines.append(
                f"| {row['Method']} | {_fmt(row['Travel Time (s)'], 2)} | {_fmt(row['Planning Time (s)'], 6)} | "
                f"{_fmt(row['Constraint Rate (%)'], 2)} | {_fmt(row['Signal Ratio (%)'], 2)} |"
            )
        lines.append("")
    return "\n".join(lines)


def _render_tex(rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["Category"]].append(row)

    lines = [
        "% Requires: \\usepackage{booktabs,threeparttable}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Final method comparison under the scene-profile-fixed live SUMO protocol.}",
        "\\label{tab:table1_method_comparison_final}",
        "\\small",
        "\\begin{threeparttable}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Travel Time (s) & Planning Time (s) & Constraint Rate (\\%) & Signal Ratio (\\%) \\\\",
        "\\midrule",
    ]
    for category in ("Baseline", "GNN", "RL", "LLM-Prompt", "Ours", "Ours-Extended"):
        items = grouped.get(category, [])
        if not items:
            continue
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{category}}}}} \\\\")
        for row in items:
            lines.append(
                f"{_latex_escape(row['Method'])} & {_fmt(row['Travel Time (s)'], 2)} & {_fmt(row['Planning Time (s)'], 6)} & "
                f"{_fmt(row['Constraint Rate (%)'], 2)} & {_fmt(row['Signal Ratio (%)'], 2)} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]",
            "\\footnotesize",
            "\\item Signal Ratio (\\%) denotes the share of edges whose final planning weight departs from the method-specific default weight; it is not a traffic-light coverage metric.",
            "\\item Repaired DQN is a candidate-route selection policy and does not reuse rule-based weights as its own output; therefore its Signal Ratio is marked as n/a rather than treated as an edge-weight generator metric.",
            "\\item PPO is included here as an independent quick baseline from \\texttt{results/ppo\\_baseline/ppo\\_quick\\_summary.csv}; it is not a scene-profile-fixed live rerun, so constraint and signal fields remain n/a.",
            "\\item DCRNN-inspired only reuses diffusion-style graph aggregation on a fixed directed graph with K=2; it is not a full DCRNN reproduction.",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_dcrnn_notes(rows: list[dict[str, Any]], identical_warning: str) -> str:
    lines = [
        "# DCRNN-inspired Notes",
        "",
        "- Core design: forward/backward random-walk diffusion aggregation over the fixed directed 96-edge graph.",
        "- Diffusion depth: `K=2`.",
        "- Seed weights: reuse the existing `method_gcn_weight()` output as the initial graph signal, then diffuse over directed edge-neighborhoods.",
        f"- {DCRNN_NOTE}",
        "",
        "## Scene-Level Results",
        "",
        "| Scene | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene_profile']} | {row['travel_time_s']:.3f} | {row['planning_time_s']:.6f} | "
            f"{row['constraint_rate']:.2f} | {row['signal_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Duplicate Check",
            "",
            f"- {identical_warning}",
            "",
        ]
    )
    return "\n".join(lines)


def _load_gcn_scene_rows() -> list[dict[str, Any]]:
    if not LIVE_METRICS_PATH.exists():
        return []
    with LIVE_METRICS_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        if str(row.get("method", "")).strip() != "GCN-Weight":
            continue
        out.append(
            {
                "scene_profile": str(row.get("scene_profile", "")).strip(),
                "travel_time_s": float(row.get("travel_time_s", 0.0) or 0.0),
                "constraint_rate": float(row.get("constraint_rate", 0.0) or 0.0),
                "signal_ratio": float(row.get("signal_ratio", 0.0) or 0.0),
                "path_len": int(float(row.get("path_len", 0) or 0)),
                "path_cost": float(row.get("path_cost", 0.0) or 0.0),
            }
        )
    return out


def _duplicate_warning(rows: list[dict[str, Any]]) -> str:
    gcn_rows = {row["scene_profile"]: row for row in _load_gcn_scene_rows()}
    if len(gcn_rows) != len(rows):
        return "GCN scene-level reference rows were incomplete; duplicate check skipped."
    for row in rows:
        ref = gcn_rows.get(row["scene_profile"])
        if not ref:
            return "GCN scene-level reference rows were incomplete; duplicate check skipped."
        signature = (
            round(float(row["travel_time_s"]), 6),
            round(float(row["constraint_rate"]), 6),
            round(float(row["signal_ratio"]), 6),
            int(row["path_len"]),
            round(float(row["path_cost"]), 6),
        )
        ref_signature = (
            round(float(ref["travel_time_s"]), 6),
            round(float(ref["constraint_rate"]), 6),
            round(float(ref["signal_ratio"]), 6),
            int(ref["path_len"]),
            round(float(ref["path_cost"]), 6),
        )
        if signature != ref_signature:
            return "No full six-scene identity with `GCN-Weight (Kipf & Welling)` was detected."
    return "WARNING: `DCRNN-inspired` is fully identical to `GCN-Weight (Kipf & Welling)` across all checked scenes."


def _update_table_notes() -> None:
    existing = TABLE_NOTES_PATH.read_text(encoding="utf-8")
    extra = f"""

# # 7. 2026-04-10 citation-ready baseline naming update

- 当前主表中的图基线命名统一为 `GCN-Weight (Kipf & Welling)`，用于明确其 citation anchor；但本仓库实现仍是轻量图权重估计器，而不是完整监督式 GCN 复现。
- `CoT-Qwen` 保留在 `LLM-Prompt` 分组，并在主表注释中明确为 chain-of-thought prompting baseline。
- `R1-Raw` 保留在 `LLM-Prompt` 分组，并在主表注释中明确为 raw reasoning prompt baseline。
- `DQN-RouteSelector` 保持为 repaired RL baseline。

# # 8. 2026-04-10 DCRNN-inspired baseline

- `DCRNN-inspired` 已按最低代价接入现有 live evaluation pipeline。
- 该方法只在固定有向图上复用 forward/backward random-walk diffusion aggregation，`K=2`。
- It is not a full DCRNN reproduction.
- 为避免夸大方法复现程度，主表与说明文件统一使用 `DCRNN-inspired` 命名。

# # 9. 2026-04-10 MAPPO 暂缓理由

- 当前仓库缺少清晰的多智能体定义、每个 agent 的观测空间、联合动作空间或共享策略接口、协作奖励设计，以及兼容的多智能体 SUMO 环境包装。
- 因此 `MAPPO` 虽然可以作为论文引用对象，但本轮不属于最低代价可提交项，已单独写入 `results/final_tables/mappo_scope_note.md`。
"""
    if "## 7. 2026-04-10 citation-ready baseline naming update" in existing:
        TABLE_NOTES_PATH.write_text(existing, encoding="utf-8")
        return
    TABLE_NOTES_PATH.write_text(existing.rstrip() + "\n" + extra, encoding="utf-8")


def main() -> None:
    cases = _standard_cases()
    scene_rows: list[dict[str, Any]] = []
    for case in cases:
        row = _evaluate_case(case)
        scene_rows.append(row)
        print(
            f"[DCRNN-inspired] {row['scene_profile']} | travel={row['travel_time_s']:.2f}s | "
            f"planning={row['planning_time_s']:.6f}s | constraint={row['constraint_rate']:.2f}% | "
            f"signal={row['signal_ratio']:.2f}%"
        )

    _write_rows(DCRNN_RESULTS_PATH, scene_rows, SCENE_FIELD_ORDER)

    existing_rows = _load_existing_rows()
    updated_rows: list[dict[str, Any]] = []
    dcrnn_summary = {
        "Method": "DCRNN-inspired",
        "Method (CN)": "DCRNN-inspired（扩散聚合）",
        "Category": "GNN",
        "Travel Time (s)": round(_mean([row["travel_time_s"] for row in scene_rows]), 6),
        "model_infer_time_s": 0.0,
        "route_solve_time_s": round(_mean([row["route_solve_time_s"] for row in scene_rows]), 6),
        "Planning Time (s)": round(_mean([row["planning_time_s"] for row in scene_rows]), 6),
        "Constraint Rate (%)": round(_mean([row["constraint_rate"] for row in scene_rows]), 6),
        "Signal Ratio (%)": round(_mean([row["signal_ratio"] for row in scene_rows]), 6),
        "simulation_seed": "|".join(str(row["simulation_seed"]) for row in scene_rows),
        "warmup_seconds": "|".join(str(row["warmup_seconds"]) for row in scene_rows),
        "evaluation_start_time": "|".join(str(row["evaluation_start_time"]) for row in scene_rows),
        "evaluation_end_time": "|".join(str(row["evaluation_end_time"]) for row in scene_rows),
        "Notes": DCRNN_NOTE,
    }

    replace_methods = {
        "GCN-Weight",
        "GCN-Weight (Kipf & Welling)",
        "DCRNN-inspired",
        "CoT-Qwen",
        "R1-Raw",
    }
    for row in existing_rows:
        method = row.get("Method", "")
        if method in replace_methods:
            continue
        updated_rows.append({key: row.get(key, "") for key in FIELD_ORDER})

    for row in existing_rows:
        method = row.get("Method", "")
        if method not in {"GCN-Weight", "GCN-Weight (Kipf & Welling)", "CoT-Qwen", "R1-Raw"}:
            continue
        new_row = {key: row.get(key, "") for key in FIELD_ORDER}
        if method in {"GCN-Weight", "GCN-Weight (Kipf & Welling)"}:
            new_row["Method"] = "GCN-Weight (Kipf & Welling)"
            new_row["Method (CN)"] = "GCN-Weight（Kipf & Welling）"
            new_row["Category"] = "GNN"
            new_row["Notes"] = GCN_NOTE
        elif method == "CoT-Qwen":
            new_row["Notes"] = COT_NOTE
        elif method == "R1-Raw":
            new_row["Notes"] = R1_NOTE
        updated_rows.append(new_row)

    updated_rows.append(dcrnn_summary)
    updated_rows.sort(
        key=lambda row: (
            CATEGORY_ORDER.get(str(row.get("Category", "")), 99),
            METHOD_ORDER.get(str(row.get("Method", "")), 999),
        )
    )

    _write_rows(FINAL_TABLE_PATH, updated_rows, FIELD_ORDER)
    PREVIEW_PATH.write_text(_render_preview(updated_rows), encoding="utf-8")
    TEX_PATH.write_text(_render_tex(updated_rows), encoding="utf-8")

    identical_warning = _duplicate_warning(scene_rows)
    DCRNN_NOTES_PATH.write_text(_render_dcrnn_notes(scene_rows, identical_warning), encoding="utf-8")
    MAPPO_NOTE_PATH.write_text(MAPPO_NOTE, encoding="utf-8")
    _update_table_notes()

    print(f"[Saved] {DCRNN_RESULTS_PATH}")
    print(f"[Saved] {DCRNN_NOTES_PATH}")
    print(f"[Saved] {MAPPO_NOTE_PATH}")
    print(f"[Saved] {FINAL_TABLE_PATH}")
    print(f"[Saved] {PREVIEW_PATH}")
    print(f"[Saved] {TEX_PATH}")
    print(f"[Check] {identical_warning}")


if __name__ == "__main__":
    main()
