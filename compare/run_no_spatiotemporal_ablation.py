#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/root/autodl-tmp")
CODE_DIR = ROOT / "code"
COMPARE_DIR = ROOT / "compare"
FINAL_DIR = ROOT / "results" / "final_tables"
PAPER_DIR = ROOT / "results" / "paper_ready"

for path in (CODE_DIR, COMPARE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app_fixed as app
from app_fixed import MAINLINE_MODEL_PATH, PathPlanningEngine
from config import ConfigManager
from eval_metrics import compute_edge_metrics
from inference import generate_weights
from run_compare import TEST_CASES, astar_route, calc_constraint_rate, method_rule_astar, net, run_sumo


ABLATION_CSV_PATH = FINAL_DIR / "ablation_study.csv"
PREVIEW_MD_PATH = FINAL_DIR / "table2_ablation_preview.md"
NOTES_PATH = FINAL_DIR / "no_spatiotemporal_notes.md"
RAW_RESULTS_PATH = FINAL_DIR / "no_spatiotemporal_results.csv"
TEX_PATH = PAPER_DIR / "table2_ablation.tex"

VARIANT = "No-SpatioTemporal"
VARIANT_CN = "无时空增强输入"
VARIANT_ORDER = {
    "Qwen-LoRA": 0,
    "LoRA+GAT": 1,
    "Sparse-LoRA": 2,
    "No-SpatioTemporal": 3,
    "Sparse-LoRA+GAT": 4,
}
SCENARIO_ORDER = {f"场景{tag}": idx for idx, tag in enumerate(["A", "B", "C", "D", "E", "F"])}


class SingleModelManager:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def get_model(self, _model_name: str):
        if self.model is None or self.tokenizer is None:
            self._load()
        return self.tokenizer, self.model

    def _load(self):
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def close(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _fmt(value, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _scenario_sort_key(name: str) -> tuple[int, str]:
    for prefix, idx in SCENARIO_ORDER.items():
        if str(name).startswith(prefix):
            return idx, str(name)
    return 99, str(name)


def run_no_st_ablation() -> list[dict]:
    config = ConfigManager()
    config.config["RUN_MODE"] = "experiment_mode"
    engine = PathPlanningEngine(config)
    manager = SingleModelManager(MAINLINE_MODEL_PATH)
    rows: list[dict] = []

    try:
        manager.get_model("Sparse-LoRA")
        for tc in TEST_CASES:
            (
                _struct,
                _raw_llm_weights,
                final_planning_weights,
                infer_time,
                _raw_output,
                _llm_parsed_count,
                _gat_applied,
                _gat_alpha_used,
                explicit_wd,
                _mode_used,
                _fallback_used,
                _fallback_reason,
                _timing_info,
            ) = generate_weights(
                engine,
                tc["desc"],
                "Sparse-LoRA",
                manager,
                use_gat=False,
                mode="experiment_mode",
                prompt_mode="baseline_sparse_prompt",
            )
            route_t0 = time.perf_counter()
            path, cost = astar_route(net, tc["start"], tc["end"], final_planning_weights)
            route_solve_time_s = time.perf_counter() - route_t0
            planning_time_s = float(infer_time) + float(route_solve_time_s)

            scene_w, _, _ = method_rule_astar(tc["start"], tc["end"], tc["desc"])
            travel_time_s = run_sumo(path, scene_w)
            edge_metrics = compute_edge_metrics(
                final_planning_weights,
                engine._ALL_EDGES,
                explicit_weights=explicit_wd,
                method_name="Sparse-LoRA",
            )

            row = {
                "Variant": VARIANT,
                "Variant (CN)": VARIANT_CN,
                "Scenario": tc["name"],
                "Travel Time (s)": round(float(travel_time_s), 3),
                "model_infer_time_s": round(float(infer_time), 6),
                "route_solve_time_s": round(float(route_solve_time_s), 6),
                "Planning Time (s)": round(float(planning_time_s), 6),
                "Constraint Rate (%)": round(float(calc_constraint_rate(final_planning_weights, tc["ground_truth"])), 1),
                "Signal Ratio (%)": round(float(edge_metrics["signal_ratio"]), 1),
                "Parsed Edge Ratio (%)": round(float(edge_metrics["parsed_edge_ratio"]), 1),
                "Status": "ok" if int(edge_metrics["parsed_edges"]) > 0 else "parse_failed",
                "LLM Component": "Yes",
                "GAT Component": "No",
                "Sparse Mode": "Yes",
            }
            rows.append(row)
            print(
                f"[{VARIANT}] {tc['name']} | travel={row['Travel Time (s)']:.1f}s | "
                f"planning={row['Planning Time (s)']:.6f}s | constraint={row['Constraint Rate (%)']:.1f}% | "
                f"signal={row['Signal Ratio (%)']:.1f}% | parsed={row['Parsed Edge Ratio (%)']:.1f}%"
            )
    finally:
        manager.close()

    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _update_ablation_csv(new_rows: list[dict]) -> list[dict]:
    with ABLATION_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        existing_rows = list(csv.DictReader(fh))
        fieldnames = list(existing_rows[0].keys()) if existing_rows else list(new_rows[0].keys())

    kept = [row for row in existing_rows if row.get("Variant") != VARIANT]
    merged = kept + new_rows
    merged.sort(key=lambda row: (VARIANT_ORDER.get(row.get("Variant", ""), 99), _scenario_sort_key(row.get("Scenario", ""))))
    _write_csv(ABLATION_CSV_PATH, merged, fieldnames)
    return merged


def _aggregate_variant_rows(rows: list[dict]) -> list[dict]:
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(row["Variant"], []).append(row)

    out = []
    for variant, items in by_variant.items():
        sample = items[0]
        out.append(
            {
                "Variant": variant,
                "Variant (CN)": sample["Variant (CN)"],
                "Travel Time (s)": round(_mean([float(r["Travel Time (s)"]) for r in items]), 6),
                "Planning Time (s)": round(_mean([float(r["Planning Time (s)"]) for r in items]), 6),
                "Constraint Rate (%)": round(_mean([float(r["Constraint Rate (%)"]) for r in items]), 6),
                "Signal Ratio (%)": round(_mean([float(r["Signal Ratio (%)"]) for r in items]), 6),
                "Parsed Edge Ratio (%)": round(_mean([float(r["Parsed Edge Ratio (%)"]) for r in items]), 6),
                "LLM Component": sample["LLM Component"],
                "GAT Component": sample["GAT Component"],
                "Sparse Mode": sample["Sparse Mode"],
            }
        )
    out.sort(key=lambda row: VARIANT_ORDER.get(row["Variant"], 99))
    return out


def _render_preview(scene_rows: list[dict], mean_rows: list[dict]) -> str:
    lines = [
        "# Table2 Ablation Preview",
        "",
        "> Newly added variant in this batch: `No-SpatioTemporal`.",
        "",
        "## Mean Rows",
        "",
        "| Variant | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) | Parsed Edge Ratio (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in mean_rows:
        lines.append(
            f"| {row['Variant']} | {_fmt(row['Travel Time (s)'], 2)} | {_fmt(row['Planning Time (s)'], 6)} | "
            f"{_fmt(row['Constraint Rate (%)'], 2)} | {_fmt(row['Signal Ratio (%)'], 2)} | {_fmt(row['Parsed Edge Ratio (%)'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## No-SpatioTemporal Scene Rows",
            "",
            "| Scenario | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) | Parsed Edge Ratio (%) | Status |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted([r for r in scene_rows if r["Variant"] == VARIANT], key=lambda r: _scenario_sort_key(r["Scenario"])):
        lines.append(
            f"| {row['Scenario']} | {_fmt(row['Travel Time (s)'], 1)} | {_fmt(row['Planning Time (s)'], 6)} | "
            f"{_fmt(row['Constraint Rate (%)'], 1)} | {_fmt(row['Signal Ratio (%)'], 1)} | {_fmt(row['Parsed Edge Ratio (%)'], 1)} | {row['Status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_tex(mean_rows: list[dict]) -> str:
    lines = [
        "% Requires: \\usepackage{booktabs,threeparttable}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation study with the No-SpatioTemporal prompt-input variant.}",
        "\\label{tab:table2_ablation}",
        "\\small",
        "\\begin{threeparttable}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Variant & Travel Time (s) & Planning Time (s) & Constraint Rate (\\%) & Signal Ratio (\\%) & Parsed Edge Ratio (\\%) \\\\",
        "\\midrule",
    ]
    for row in mean_rows:
        label = str(row["Variant"]).replace("&", "\\&")
        lines.append(
            f"{label} & {_fmt(row['Travel Time (s)'], 2)} & {_fmt(row['Planning Time (s)'], 6)} & "
            f"{_fmt(row['Constraint Rate (%)'], 2)} & {_fmt(row['Signal Ratio (%)'], 2)} & {_fmt(row['Parsed Edge Ratio (%)'], 2)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]",
            "\\footnotesize",
            "\\item No-SpatioTemporal removes only the spatiotemporal structured-field injection from the Sparse-LoRA prompt. Model weights, sparse parser, path planning, GAT setting, and metric definitions remain unchanged.",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_notes(new_rows: list[dict], mean_row: dict) -> str:
    lines = [
        "# No-SpatioTemporal Notes",
        "",
        "## Sparse-LoRA Main Prompt Chain",
        "",
        "- `code/inference.py` uses `build_sparse_task_prompt()` for the normal Sparse-LoRA path.",
        "- `build_sparse_task_prompt()` delegates to `code/sparse_utils.py::build_sparse_task_prompt()` which injects `SCENE_TYPE`, `SCHEMA`, `NARRATIVE`, and per-event structured fields such as `ROAD`, `DIR`, `RANGE`, `LEVEL`, `TIME`, `PROPAGATION`, and `CONFLICT`.",
        "- Sparse inference then stays on the same chain: prompt -> `infer_sparse()` -> `parse_sparse_output_bundle()` -> A* path planning -> SUMO evaluation.",
        "",
        "## No-SpatioTemporal Change",
        "",
        "- This ablation switches only `prompt_mode` from `spatiotemporal_enhanced_prompt` to `baseline_sparse_prompt`.",
        "- The same model weights are reused from the Sparse-LoRA mainline model path.",
        "- The same sparse parser, path planner, no-GAT setting, and evaluation metrics are reused unchanged.",
        "",
        "## Scene Results",
        "",
        "| Scenario | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) | Parsed Edge Ratio (%) | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(new_rows, key=lambda r: _scenario_sort_key(r["Scenario"])):
        lines.append(
            f"| {row['Scenario']} | {_fmt(row['Travel Time (s)'], 1)} | {_fmt(row['Planning Time (s)'], 6)} | "
            f"{_fmt(row['Constraint Rate (%)'], 1)} | {_fmt(row['Signal Ratio (%)'], 1)} | {_fmt(row['Parsed Edge Ratio (%)'], 1)} | {row['Status']} |"
        )
    lines.extend(
        [
            "",
            "## Mean Row",
            "",
            "| Variant | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) | Parsed Edge Ratio (%) |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {mean_row['Variant']} | {_fmt(mean_row['Travel Time (s)'], 6)} | {_fmt(mean_row['Planning Time (s)'], 6)} | "
            f"{_fmt(mean_row['Constraint Rate (%)'], 6)} | {_fmt(mean_row['Signal Ratio (%)'], 6)} | {_fmt(mean_row['Parsed Edge Ratio (%)'], 6)} |",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    new_rows = run_no_st_ablation()
    raw_fieldnames = list(new_rows[0].keys())
    _write_csv(RAW_RESULTS_PATH, new_rows, raw_fieldnames)

    merged_scene_rows = _update_ablation_csv(new_rows)
    mean_rows = _aggregate_variant_rows(merged_scene_rows)
    mean_row = next(row for row in mean_rows if row["Variant"] == VARIANT)

    PREVIEW_MD_PATH.write_text(_render_preview(merged_scene_rows, mean_rows), encoding="utf-8")
    TEX_PATH.write_text(_render_tex(mean_rows), encoding="utf-8")
    NOTES_PATH.write_text(_render_notes(new_rows, mean_row), encoding="utf-8")

    print(f"[Saved] {RAW_RESULTS_PATH}")
    print(f"[Saved] {ABLATION_CSV_PATH}")
    print(f"[Saved] {PREVIEW_MD_PATH}")
    print(f"[Saved] {TEX_PATH}")
    print(f"[Saved] {NOTES_PATH}")


if __name__ == "__main__":
    main()
