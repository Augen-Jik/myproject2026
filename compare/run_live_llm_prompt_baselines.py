#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
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
SCENE_RESULTS_PATH = FINAL_TABLE_DIR / "llm_prompt_baselines_scene_results.csv"
NOTES_PATH = FINAL_TABLE_DIR / "llm_prompt_baselines_notes.md"
PREVIEW_PATH = FINAL_TABLE_DIR / "method_comparison_final_preview.md"
TEX_PATH = ROOT / "results" / "paper_ready" / "table1_method_comparison_final.tex"

for path in (CODE_DIR, COMPARE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app_fixed as app
import run_compare as legacy_compare
import run_live_scene_profile_main_table as live_main
from eval_metrics import compute_edge_metrics
from sim_scene_profiles import STANDARD_SCENE_PROFILE_NAMES, build_environment_weights, get_scene_profile_version


app.st.error = lambda *args, **kwargs: None

METHOD_SPECS = (
    {
        "method": "CoT-Qwen",
        "method_cn": "CoT-Qwen（提示推理）",
        "category": "LLM-Prompt",
        "model_path": str(ROOT / "Qwen2.5-1.5B-Instruct"),
        "load_fn": lambda: legacy_compare.load_llm(str(ROOT / "Qwen2.5-1.5B-Instruct")),
        "infer_fn": lambda tok, mdl, constraint: legacy_compare.llm_infer(
            tok,
            mdl,
            legacy_compare.COT_PROMPT_TMPL.format(desc=constraint),
        ),
        "notes": "Qwen2.5-1.5B-Instruct + CoT prompt，经 live scene-profile-fixed SUMO rerun。",
    },
    {
        "method": "R1-Raw",
        "method_cn": "R1-Raw（DeepSeek-R1 基座）",
        "category": "LLM-Prompt",
        "model_path": str(ROOT / "DeepSeek-R1-1.5B"),
        "load_fn": lambda: legacy_compare.load_llm(str(ROOT / "DeepSeek-R1-1.5B")),
        "infer_fn": lambda tok, mdl, constraint: legacy_compare.llm_infer_r1(tok, mdl, constraint),
        "notes": "DeepSeek-R1-1.5B raw prompt chain，经 live scene-profile-fixed SUMO rerun。",
    },
)

CSV_FIELD_ORDER = [
    "method",
    "method_cn",
    "category",
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
    "anchor_coverage",
    "signal_edges",
    "parsed_edges",
    "parsed_edge_ratio",
    "default_edges",
    "path_len",
    "path_cost",
    "avg_speed_kmh",
    "simulation_seed",
    "warmup_seconds",
    "evaluation_start_time",
    "evaluation_end_time",
    "scene_profile_version",
    "model_path",
    "protocol_snapshot_json",
    "notes",
]

FINAL_TABLE_FIELD_ORDER = [
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

GROUP_ORDER = {
    "Baseline": 0,
    "LLM-Prompt": 1,
    "Ours": 2,
    "Ours-Extended": 3,
}

METHOD_ORDER = {
    "Dijkstra": 0,
    "Rule-A*": 1,
    "DQN": 2,
    "GCN-Weight": 3,
    "CoT-Qwen": 4,
    "R1-Raw": 5,
    "Sparse-LoRA": 6,
    "Sparse-LoRA+GAT": 7,
}


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _constraint_rate(final_weights: dict[str, float], scene_weights: dict[str, float]) -> float:
    return live_main._constraint_rate(final_weights, scene_weights)


def _standard_cases() -> list[dict[str, Any]]:
    return live_main.standard_scene_cases(list(STANDARD_SCENE_PROFILE_NAMES))


def _complete_weights(parsed_weights: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    explicit_weights = dict(parsed_weights)
    final_weights = dict(parsed_weights)
    if not final_weights:
        final_weights = {edge: 2.0 for edge in legacy_compare.ALL_EDGES}
    else:
        for edge in legacy_compare.ALL_EDGES:
            final_weights.setdefault(edge, 2.0)
    return explicit_weights, final_weights


def _evaluate_method(spec: dict[str, Any], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not Path(spec["model_path"]).exists():
        raise FileNotFoundError(f"Missing model path for {spec['method']}: {spec['model_path']}")

    tok, mdl = spec["load_fn"]()
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            infer_t0 = time.perf_counter()
            raw_text = spec["infer_fn"](tok, mdl, case["constraint"])
            infer_time_s = time.perf_counter() - infer_t0

            parsed_weights = legacy_compare.parse_weights(raw_text)
            explicit_weights, final_weights = _complete_weights(parsed_weights)

            route_t0 = time.perf_counter()
            path, path_cost = legacy_compare.astar_route(
                legacy_compare.net,
                case["start_edge"],
                case["end_edge"],
                final_weights,
            )
            route_solve_time_s = time.perf_counter() - route_t0
            planning_time_s = infer_time_s + route_solve_time_s

            scene_profile = case["scene_profile"]
            scene_weights = build_environment_weights(app.path_engine._ALL_EDGES, scene_profile)
            travel_time_s, avg_speed, _speed_data, _pos_data = app.sumo_engine.run_simulation(
                tuple(path),
                scene_profile,
                tuple(sorted(scene_weights.items())),
            )
            edge_metrics = compute_edge_metrics(
                final_weights=final_weights,
                explicit_weights=explicit_weights,
                all_edges=app.path_engine._ALL_EDGES,
                method_name=spec["method"],
            )
            protocol_snapshot = live_main.protocol_to_dict(live_main.build_sim_eval_protocol(scene_profile))

            rows.append(
                {
                    "method": spec["method"],
                    "method_cn": spec["method_cn"],
                    "category": spec["category"],
                    "scenario": case["scenario"],
                    "preset_id": case["preset_id"],
                    "scene_profile": case["scene_profile_name"],
                    "start_edge": case["start_edge"],
                    "end_edge": case["end_edge"],
                    "travel_time_s": round(float(travel_time_s), 3),
                    "planning_time_s": round(float(planning_time_s), 6),
                    "model_infer_time_s": round(float(infer_time_s), 6),
                    "route_solve_time_s": round(float(route_solve_time_s), 6),
                    "constraint_rate": _constraint_rate(final_weights, scene_weights),
                    "signal_ratio": round(float(edge_metrics["signal_ratio"]), 3),
                    "anchor_coverage": "",
                    "signal_edges": int(edge_metrics["signal_edges"]),
                    "parsed_edges": int(edge_metrics["parsed_edges"]),
                    "parsed_edge_ratio": round(float(edge_metrics["parsed_edge_ratio"]), 3),
                    "default_edges": int(edge_metrics["default_edges"]),
                    "path_len": len(path or []),
                    "path_cost": round(float(path_cost or 0.0), 6),
                    "avg_speed_kmh": round(float(avg_speed), 3),
                    "simulation_seed": int(scene_profile.simulation_seed),
                    "warmup_seconds": int(scene_profile.warmup_seconds),
                    "evaluation_start_time": int(scene_profile.evaluation_start_time),
                    "evaluation_end_time": int(scene_profile.evaluation_end_time),
                    "scene_profile_version": get_scene_profile_version(),
                    "model_path": spec["model_path"],
                    "protocol_snapshot_json": json.dumps(protocol_snapshot, ensure_ascii=False, sort_keys=True),
                    "notes": spec["notes"],
                }
            )
    finally:
        del tok
        del mdl
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return rows


def _write_scene_results(rows: list[dict[str, Any]]) -> None:
    FINAL_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with SCENE_RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELD_ORDER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    out: list[dict[str, Any]] = []
    for spec in METHOD_SPECS:
        method_rows = sorted(
            by_method.get(spec["method"], []),
            key=lambda item: list(STANDARD_SCENE_PROFILE_NAMES).index(item["scene_profile"]),
        )
        if not method_rows:
            continue
        out.append(
            {
                "Method": spec["method"],
                "Method (CN)": spec["method_cn"],
                "Category": spec["category"],
                "Travel Time (s)": round(_mean([row["travel_time_s"] for row in method_rows]), 6),
                "model_infer_time_s": round(_mean([row["model_infer_time_s"] for row in method_rows]), 6),
                "route_solve_time_s": round(_mean([row["route_solve_time_s"] for row in method_rows]), 6),
                "Planning Time (s)": round(_mean([row["planning_time_s"] for row in method_rows]), 6),
                "Constraint Rate (%)": round(_mean([row["constraint_rate"] for row in method_rows]), 6),
                "Signal Ratio (%)": round(_mean([row["signal_ratio"] for row in method_rows]), 6),
                "simulation_seed": "|".join(str(row["simulation_seed"]) for row in method_rows),
                "warmup_seconds": "|".join(str(row["warmup_seconds"]) for row in method_rows),
                "evaluation_start_time": "|".join(str(row["evaluation_start_time"]) for row in method_rows),
                "evaluation_end_time": "|".join(str(row["evaluation_end_time"]) for row in method_rows),
                "Notes": spec["notes"],
            }
        )
    return out


def _read_existing_final_table() -> list[dict[str, str]]:
    if not FINAL_TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing existing final table: {FINAL_TABLE_PATH}")
    with FINAL_TABLE_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _update_final_table(new_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_rows = _read_existing_final_table()
    replace_methods = {row["Method"] for row in new_summary_rows}
    kept_rows: list[dict[str, Any]] = [row for row in existing_rows if row.get("Method") not in replace_methods]
    merged_rows: list[dict[str, Any]] = kept_rows + [{key: row.get(key, "") for key in FINAL_TABLE_FIELD_ORDER} for row in new_summary_rows]
    merged_rows.sort(key=lambda row: (GROUP_ORDER.get(row.get("Category", ""), 99), METHOD_ORDER.get(row.get("Method", ""), 999)))

    with FINAL_TABLE_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FINAL_TABLE_FIELD_ORDER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)
    return merged_rows


def _fmt(value: Any, digits: int) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def _render_notes(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> str:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    lines = [
        "# LLM Prompt Baselines Notes",
        "",
        "## Entry Points",
        "",
        "- `CoT-Qwen`: `compare/run_compare.py` -> `COT_PROMPT_TMPL` + `llm_infer()` + `parse_weights()` + `astar_route()`.",
        "- `R1-Raw`: `compare/run_compare.py` -> `llm_infer_r1()` + `parse_weights()` + `astar_route()`.",
        "- Live evaluation protocol reused from `compare/run_live_scene_profile_main_table.py` via `standard_scene_cases()` and `app_fixed.sumo_engine.run_simulation()`.",
        "",
        "## Compatibility",
        "",
        "- Both methods can be attached to the six standard scenes without changing Sparse-LoRA / GAT / DQN / PPO code.",
        "- `planning_time` in this batch counts only prompt inference + downstream A* planning; model loading is excluded.",
        "- `signal_ratio` is applicable through `compute_edge_metrics()`.",
        "- `anchor_coverage` is left blank because these prompt baselines do not emit anchor-level structures in the current evaluation chain.",
        "",
        "## Scene-Level Results",
        "",
        "| Method | Scene | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) | Anchor Coverage |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for spec in METHOD_SPECS:
        for row in sorted(by_method.get(spec["method"], []), key=lambda item: list(STANDARD_SCENE_PROFILE_NAMES).index(item["scene_profile"])):
            lines.append(
                f"| {row['method']} | {row['scene_profile']} | {row['travel_time_s']:.3f} | "
                f"{row['planning_time_s']:.6f} | {row['constraint_rate']:.2f} | {row['signal_ratio']:.3f} |  |"
            )
    lines.extend(
        [
            "",
            "## Mean Rows Added To Main Table",
            "",
            "| Method | Category | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['Method']} | {row['Category']} | {float(row['Travel Time (s)']):.6f} | "
            f"{float(row['Planning Time (s)']):.6f} | {float(row['Constraint Rate (%)']):.6f} | "
            f"{float(row['Signal Ratio (%)']):.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_preview(final_rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        grouped[row["Category"]].append(row)

    lines = [
        "# Method Comparison Final Preview",
        "",
        "> Newly added rows in this batch: `CoT-Qwen`, `R1-Raw`.",
        "",
    ]
    for category in ("Baseline", "LLM-Prompt", "Ours", "Ours-Extended"):
        rows = grouped.get(category, [])
        if not rows:
            continue
        lines.extend(
            [
                f"## {category}",
                "",
                "| Method | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['Method']} | {_fmt(row['Travel Time (s)'], 2)} | {_fmt(row['Planning Time (s)'], 6)} | "
                f"{_fmt(row['Constraint Rate (%)'], 2)} | {_fmt(row['Signal Ratio (%)'], 2)} |"
            )
        lines.append("")
    return "\n".join(lines)


def _render_tex(final_rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
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
    for category in ("Baseline", "LLM-Prompt", "Ours", "Ours-Extended"):
        rows = grouped.get(category, [])
        if not rows:
            continue
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{category}}}}} \\\\")
        for row in rows:
            lines.append(
                f"{row['Method']} & {_fmt(row['Travel Time (s)'], 2)} & {_fmt(row['Planning Time (s)'], 6)} & "
                f"{_fmt(row['Constraint Rate (%)'], 2)} & {_fmt(row['Signal Ratio (%)'], 2)} \\\\"
            )
        if category != "Ours-Extended":
            lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]",
            "\\footnotesize",
            "\\item Signal Ratio (\\%) denotes the share of edges whose final planning weight departs from the method-specific default weight; it is not a traffic-light coverage metric.",
            "\\item PPO is intentionally excluded from this main table because its available result comes from the simplified quick baseline environment in \\texttt{results/ppo\\_baseline/ppo\\_quick\\_summary.csv}, not from the scene-profile-fixed live SUMO protocol.",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live rerun CoT-Qwen and R1-Raw on the six standard scene profiles.")
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=[spec["method"] for spec in METHOD_SPECS],
        default=[spec["method"] for spec in METHOD_SPECS],
        help="Subset of prompt baselines to rerun.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_methods = [spec for spec in METHOD_SPECS if spec["method"] in set(args.methods)]
    cases = _standard_cases()

    all_rows: list[dict[str, Any]] = []
    for spec in selected_methods:
        print(f"[Run] {spec['method']}")
        method_rows = _evaluate_method(spec, cases)
        all_rows.extend(method_rows)
        for row in method_rows:
            print(
                "  - "
                f"{row['scene_profile']} | travel={row['travel_time_s']:.2f}s | "
                f"planning={row['planning_time_s']:.6f}s | constraint={row['constraint_rate']:.2f}% | "
                f"signal={row['signal_ratio']:.2f}%"
            )

    summary_rows = _summary_rows(all_rows)
    final_rows = _update_final_table(summary_rows)

    _write_scene_results(all_rows)
    NOTES_PATH.write_text(_render_notes(all_rows, summary_rows), encoding="utf-8")
    PREVIEW_PATH.write_text(_render_preview(final_rows), encoding="utf-8")
    if TEX_PATH.exists():
        TEX_PATH.write_text(_render_tex(final_rows), encoding="utf-8")

    print(f"[Saved] {SCENE_RESULTS_PATH}")
    print(f"[Saved] {FINAL_TABLE_PATH}")
    print(f"[Saved] {NOTES_PATH}")
    print(f"[Saved] {PREVIEW_PATH}")
    if TEX_PATH.exists():
        print(f"[Saved] {TEX_PATH}")


if __name__ == "__main__":
    main()
