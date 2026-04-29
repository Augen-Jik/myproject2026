#!/usr/bin/env python3
"""Run Sparse-LoRA vs Always-GAT vs Gated-GAT with scene-type summaries."""

from __future__ import annotations

import argparse
import copy
import gc
import os
import re
import statistics
import sys
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/autodl-tmp/code")
sys.path.insert(0, "/root/autodl-tmp/compare")

from app_fixed import PathPlanningEngine
from config import ConfigManager
from eval_metrics import compute_edge_metrics
from gat_smoother import EDGES, EDGE_IDX, N_EDGES, describe_gat_checkpoint, load_gat_model, smooth_weights
from inference import generate_weights
from results_manager import (
    create_result_bundle,
    default_config_snapshot,
    write_json,
    write_metrics_csv,
    write_standard_artifacts,
)
from run_compare import (
    TEST_CASES,
    astar_route,
    calc_constraint_rate,
    method_rule_astar,
    net,
    run_sumo,
)
from scenarios import APP_SCENARIO_PRESETS, classify_scene_type, get_special_stress_cases


SCENE_TYPE_BY_NAME = {
    item["name"]: item["scene_type"]
    for item in APP_SCENARIO_PRESETS
}
SCENE_TYPE_BY_ID = {
    item["id"]: item["scene_type"]
    for item in APP_SCENARIO_PRESETS
}

SPECIAL_SCENES = {"core_blockage", "directional_asymmetry", "compound_disaster"}


def build_ablation_cases():
    normal_cases = []
    for tc in TEST_CASES:
        normal_cases.append({
            "id": tc["name"].split(" ")[0],
            "name": tc["name"],
            "start": tc["start"],
            "end": tc["end"],
            "desc": tc["desc"],
            "ground_truth": dict(tc["ground_truth"]),
            "scene_type": resolve_scene_type(tc["name"], tc["desc"]),
            "scene_bucket": "normal",
            "stress_tags": [],
        })

    special_cases = []
    for tc in get_special_stress_cases():
        special_cases.append({
            "id": tc["id"],
            "name": tc["name"],
            "start": tc["start"],
            "end": tc["end"],
            "desc": tc["desc"],
            "ground_truth": dict(tc["ground_truth"]),
            "scene_type": tc["scene_type"],
            "scene_bucket": tc.get("scene_bucket", "special"),
            "stress_tags": list(tc.get("stress_tags", [])),
        })
    return normal_cases + special_cases


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


def _adaptive_alpha(llm_weight_dict: dict[str, float], alpha: float) -> float:
    llm_parsed_n = len([eid for eid in llm_weight_dict if eid in EDGE_IDX])
    if llm_parsed_n >= 90:
        alpha = 0.90
    elif llm_parsed_n >= 60:
        alpha = 0.75
    else:
        alpha = 0.55

    parsed_vals = [v for k, v in llm_weight_dict.items() if k in EDGE_IDX and v != 2.0]
    if len(parsed_vals) >= 3:
        std = float(np.std(parsed_vals))
        if std > 1.5:
            alpha = max(alpha - 0.10, 0.45)
    return round(alpha, 3)


def smooth_weights_always(
    llm_weight_dict: dict[str, float],
    *,
    model_path: str,
    alpha: float = 0.65,
    device: str = "cpu",
    protected_edges: list[str] | set[str] | None = None,
):
    model = load_gat_model(model_path, device)
    alpha = _adaptive_alpha(llm_weight_dict, alpha)

    raw_vec = torch.zeros(N_EDGES)
    parsed_mask = torch.zeros(N_EDGES, dtype=torch.bool)
    protected_mask = torch.zeros(N_EDGES, dtype=torch.bool)
    protected_set = set(protected_edges or [])

    for eid, weight in llm_weight_dict.items():
        if eid not in EDGE_IDX:
            continue
        idx = EDGE_IDX[eid]
        raw_vec[idx] = float(weight)
        parsed_mask[idx] = True
        if eid in protected_set or float(weight) >= 9.0:
            protected_mask[idx] = True

    with torch.no_grad():
        gat_vec = model(raw_vec.to(device)).cpu()

    blend_mask = parsed_mask & ~protected_mask
    missing_mask = ~parsed_mask
    refined = gat_vec.clone()
    refined[blend_mask] = alpha * raw_vec[blend_mask] + (1 - alpha) * gat_vec[blend_mask]
    refined[protected_mask] = raw_vec[protected_mask]
    refined = refined.clamp(0.3, 10.0)

    result = {EDGES[i]: round(refined[i].item(), 2) for i in range(N_EDGES)}
    for eid in protected_set:
        if eid in llm_weight_dict:
            result[eid] = max(result.get(eid, llm_weight_dict[eid]), float(llm_weight_dict[eid]))

    metadata = {
        "gat_mode": "always_topology_completion",
        "high_conf_edges": [],
        "low_conf_edges": [EDGES[i] for i in range(N_EDGES) if blend_mask[i]],
        "missing_edges": [EDGES[i] for i in range(N_EDGES) if missing_mask[i]],
        "protected_edges": sorted(protected_set),
        "alpha_used": alpha,
    }
    return result, alpha, metadata


def mean_or_zero(values):
    return round(statistics.mean(values), 3) if values else 0.0


def resolve_scene_type(name: str, desc: str) -> str:
    direct = SCENE_TYPE_BY_NAME.get(name)
    if direct:
        return direct
    match = re.search(r"场景([A-F])", name)
    if match and match.group(1) in SCENE_TYPE_BY_ID:
        return SCENE_TYPE_BY_ID[match.group(1)]
    return classify_scene_type(name=name, text=desc, event_count=desc.count("；") + 1)


def rank_methods(rows: list[dict], metric: str) -> list[tuple[str, float]]:
    bucket = defaultdict(list)
    for row in rows:
        bucket[row["method"]].append(float(row[metric]))
    reverse = metric in {"constraint_rate", "parsed_edge_ratio", "signal_ratio"}
    ranked = [(method, mean_or_zero(vals)) for method, vals in bucket.items()]
    return sorted(ranked, key=lambda item: item[1], reverse=reverse)


def summarize_subset(rows: list[dict], subset_name: str):
    bucket = defaultdict(list)
    for row in rows:
        bucket[row["method"]].append(row)
    summary_rows = []
    for method, items in sorted(bucket.items()):
        summary_rows.append({
            "subset": subset_name,
            "method": method,
            "samples": len(items),
            "avg_path_cost": mean_or_zero([r["path_cost"] for r in items]),
            "avg_constraint_rate": mean_or_zero([r["constraint_rate"] for r in items]),
            "avg_parsed_edge_ratio": mean_or_zero([r["parsed_edge_ratio"] for r in items]),
            "avg_signal_ratio": mean_or_zero([r["signal_ratio"] for r in items]),
            "avg_infer_time_s": mean_or_zero([r["infer_time_s"] for r in items]),
            "avg_sumo_time": mean_or_zero([r["sumo_time"] for r in items if r["sumo_time"] > 0]),
        })
    return summary_rows


def compare_against_sparse(rows: list[dict]):
    by_scene = defaultdict(dict)
    for row in rows:
        by_scene[row["scene"]][row["method"]] = row
    out = []
    for scene, items in sorted(by_scene.items()):
        sparse = items.get("Sparse-LoRA")
        if not sparse:
            continue
        base = float(sparse["sumo_time"]) if float(sparse["sumo_time"]) > 0 else float(sparse["path_cost"])
        base_metric = "sumo_time" if float(sparse["sumo_time"]) > 0 else "path_cost"
        for method in ("Sparse-LoRA+Always-GAT", "Sparse-LoRA+Gated-GAT"):
            cur = items.get(method)
            if not cur:
                continue
            value = float(cur["sumo_time"]) if base_metric == "sumo_time" and float(cur["sumo_time"]) > 0 else float(cur["path_cost"])
            delta = round(value - base, 3)
            status = "tie"
            if delta < -1.0:
                status = "helpful"
            elif delta > 1.0:
                status = "negative"
            out.append({
                "scene": scene,
                "scene_type": cur["scene_type"],
                "baseline_metric": base_metric,
                "method": method,
                "delta_vs_sparse": delta,
                "status": status,
            })
    return out


def compare_gate_vs_always(rows: list[dict]):
    by_scene = defaultdict(dict)
    for row in rows:
        by_scene[row["scene"]][row["method"]] = row
    out = []
    for scene, items in sorted(by_scene.items()):
        always = items.get("Sparse-LoRA+Always-GAT")
        gated = items.get("Sparse-LoRA+Gated-GAT")
        if not always or not gated:
            continue
        base = float(always["sumo_time"]) if float(always["sumo_time"]) > 0 else float(always["path_cost"])
        cur = float(gated["sumo_time"]) if float(gated["sumo_time"]) > 0 else float(gated["path_cost"])
        delta = round(cur - base, 3)
        status = "tie"
        if delta < -1.0:
            status = "helpful"
        elif delta > 1.0:
            status = "negative"
        out.append({
            "scene": scene,
            "scene_type": gated["scene_type"],
            "scene_bucket": gated.get("scene_bucket", "normal"),
            "always_metric": round(base, 3),
            "gated_metric": round(cur, 3),
            "delta_gated_vs_always": delta,
            "status": status,
        })
    return out


def extract_failure_cases(rows: list[dict]):
    failures = []
    for row in rows:
        if row["method"] not in {"Sparse-LoRA+Always-GAT", "Sparse-LoRA+Gated-GAT"}:
            continue
        if float(row["constraint_rate"]) < 40.0 or (float(row["sumo_time"]) > 0 and float(row["sumo_time"]) >= 800.0):
            failures.append({
                "scene": row["scene"],
                "scene_type": row["scene_type"],
                "scene_bucket": row.get("scene_bucket", "normal"),
                "method": row["method"],
                "sumo_time": row["sumo_time"],
                "constraint_rate": row["constraint_rate"],
                "parsed_edge_ratio": row["parsed_edge_ratio"],
                "signal_ratio": row["signal_ratio"],
                "gat_mode": row["gat_mode"],
                "high_conf_edges": row["high_conf_edges"],
                "low_conf_edges": row["low_conf_edges"],
                "missing_edges": row["missing_edges"],
            })
    return failures


def run_ablation(gat_model_path: str, with_sumo: bool = True, alpha: float = 0.65):
    gat_device = "cpu"
    checkpoint_info = describe_gat_checkpoint(gat_model_path)
    checkpoint_exists = bool(checkpoint_info.get("exists"))
    checkpoint_error = None
    if checkpoint_exists:
        try:
            load_gat_model(gat_model_path, gat_device)
        except Exception as exc:
            checkpoint_error = f"{type(exc).__name__}: {exc}"

    config = ConfigManager()
    config.config["RUN_MODE"] = "experiment_mode"
    engine = PathPlanningEngine(config)
    manager = SingleModelManager("/root/autodl-tmp/model_merged_sparse")
    case_rows = []
    all_cases = build_ablation_cases()

    try:
        manager.get_model("Sparse-LoRA")
        for tc in all_cases:
            scene_type = tc.get("scene_type") or resolve_scene_type(tc["name"], tc["desc"])
            (
                _struct,
                raw_sparse_weights,
                _filled_weights,
                infer_time,
                _raw,
                _parsed_count,
                _gat_applied,
                _gat_alpha,
                _explicit_wd,
                _mode,
                _fallback_used,
                _fallback_reason,
                timing_info,
            ) = generate_weights(
                engine,
                tc["desc"],
                "Sparse-LoRA",
                manager,
                use_gat=False,
                mode="experiment_mode",
            )
            sparse_bundle = getattr(engine, "_last_sparse_parse", {})
            base_sparse = {eid: 2.0 for eid in engine._ALL_EDGES}
            base_sparse.update(copy.deepcopy(raw_sparse_weights))

            method_outputs = [("Sparse-LoRA", base_sparse, {"gat_mode": "disabled", "alpha_used": 0.0})]

            if checkpoint_exists and checkpoint_error is None:
                always_weights, always_alpha, always_meta = smooth_weights_always(
                    copy.deepcopy(raw_sparse_weights),
                    model_path=gat_model_path,
                    alpha=alpha,
                    device=gat_device,
                    protected_edges=sparse_bundle.get("protected_edges", []),
                )
                gated_weights, gated_alpha, gated_meta = smooth_weights(
                    copy.deepcopy(raw_sparse_weights),
                    model_path=gat_model_path,
                    alpha=alpha,
                    device=gat_device,
                    edge_confidence=sparse_bundle.get("edge_confidence"),
                    protected_edges=sparse_bundle.get("protected_edges"),
                    return_metadata=True,
                )
                method_outputs.extend([
                    ("Sparse-LoRA+Always-GAT", always_weights, {**always_meta, "alpha_used": always_alpha}),
                    ("Sparse-LoRA+Gated-GAT", gated_weights, {**gated_meta, "alpha_used": gated_alpha}),
                ])

            env_weights, _, _ = method_rule_astar(tc["start"], tc["end"], tc["desc"])

            for method_name, final_weights, gat_meta in method_outputs:
                path, cost = astar_route(net, tc["start"], tc["end"], final_weights)
                sumo_time = run_sumo(path, env_weights) if with_sumo else 0.0
                edge_metrics = compute_edge_metrics(
                    final_weights,
                    engine._ALL_EDGES,
                    explicit_weights=raw_sparse_weights,
                    method_name="Sparse-LoRA",
                )
                case_rows.append({
                    "scene": tc["name"],
                    "scene_type": scene_type,
                    "scene_bucket": tc.get("scene_bucket", "normal"),
                    "stress_tags": ",".join(tc.get("stress_tags", [])),
                    "method": method_name,
                    "path_cost": round(float(cost), 3),
                    "path_len": len(path),
                    "constraint_rate": calc_constraint_rate(final_weights, tc["ground_truth"]),
                    "parsed_edges": edge_metrics["parsed_edges"],
                    "parsed_edge_ratio": edge_metrics["parsed_edge_ratio"],
                    "signal_edges": edge_metrics["signal_edges"],
                    "signal_ratio": edge_metrics["signal_ratio"],
                    "infer_time_s": round(float(infer_time), 3),
                    "sumo_time": round(float(sumo_time), 3),
                    "anchor_count": int(sparse_bundle.get("anchor_count", 0)),
                    "parse_confidence": round(float(sparse_bundle.get("parse_confidence", 0.0)), 3),
                    "gat_mode": gat_meta.get("gat_mode", "disabled"),
                    "alpha_used": float(gat_meta.get("alpha_used", 0.0)),
                    "high_conf_edges": len(gat_meta.get("high_conf_edges", [])),
                    "low_conf_edges": len(gat_meta.get("low_conf_edges", [])),
                    "missing_edges": len(gat_meta.get("missing_edges", [])),
                })
                print(
                    f"{tc['name']} | {method_name:<23} | bucket={tc.get('scene_bucket', 'normal'):<7} | scene={scene_type:<22} | "
                    f"cost={float(cost):6.2f} | sumo={float(sumo_time):6.1f} | gate={gat_meta.get('gat_mode', 'disabled')}"
                )
    finally:
        manager.close()

    overall = summarize_subset(case_rows, "all_scenes")
    normal = summarize_subset([row for row in case_rows if row["scene_bucket"] == "normal"], "normal_scenes")
    special = summarize_subset([row for row in case_rows if row["scene_bucket"] == "special"], "special_scenes")
    scene_type_tables = {
        scene_type: summarize_subset([row for row in case_rows if row["scene_type"] == scene_type], scene_type)
        for scene_type in sorted({row["scene_type"] for row in case_rows})
    }
    delta_rows = compare_against_sparse(case_rows)
    gate_vs_always_rows = compare_gate_vs_always(case_rows)
    failure_rows = extract_failure_cases(case_rows)

    rankings = {
        scene_type: {
            "sumo_ranking": rank_methods([row for row in case_rows if row["scene_type"] == scene_type], "sumo_time"),
            "constraint_ranking": rank_methods([row for row in case_rows if row["scene_type"] == scene_type], "constraint_rate"),
        }
        for scene_type in sorted({row["scene_type"] for row in case_rows})
    }

    checkpoint_status = {
        **checkpoint_info,
        "load_error": checkpoint_error,
        "mode": checkpoint_info.get("message") if checkpoint_exists and checkpoint_error is None else "logic_only",
        "fallback_recommendation": None if checkpoint_exists and checkpoint_error is None else "Use Sparse-LoRA and a no-op / fixed-weight gate as the minimal ablation baseline.",
    }

    return case_rows, overall, normal, special, scene_type_tables, delta_rows, gate_vs_always_rows, failure_rows, rankings, checkpoint_status


def render_report(overall, normal, special, rankings, delta_rows, gate_vs_always_rows, failure_rows, checkpoint_status):
    helpful = sum(1 for row in gate_vs_always_rows if row["status"] == "helpful")
    ties = sum(1 for row in gate_vs_always_rows if row["status"] == "tie")
    negative = sum(1 for row in gate_vs_always_rows if row["status"] == "negative")
    lines = [
        "Gated GAT Ablation",
        "===================",
        "",
        f"Checkpoint: {checkpoint_status['path']}",
        f"Mode: {checkpoint_status['mode']}",
        f"Load detail: {checkpoint_status.get('message', 'unknown')}",
        f"Load error: {checkpoint_status['load_error'] or 'none'}",
        "",
        "Overall Summary",
        "---------------",
    ]
    for row in overall:
        lines.append(
            f"{row['method']}: sumo={row['avg_sumo_time']:.1f}s, "
            f"constraint={row['avg_constraint_rate']:.1f}%, "
            f"parsed={row['avg_parsed_edge_ratio']:.1f}%, signal={row['avg_signal_ratio']:.1f}%"
        )
    lines.extend(["", "Normal Scene Summary", "--------------------"])
    for row in normal:
        lines.append(
            f"{row['method']}: sumo={row['avg_sumo_time']:.1f}s, "
            f"constraint={row['avg_constraint_rate']:.1f}%"
        )
    lines.extend(["", "Special Scene Summary", "---------------------"])
    for row in special:
        lines.append(
            f"{row['method']}: sumo={row['avg_sumo_time']:.1f}s, "
            f"constraint={row['avg_constraint_rate']:.1f}%"
        )
    lines.extend([
        "",
        "Gate vs Always",
        "--------------",
        f"helpful={helpful}, tie={ties}, negative={negative}",
        "结论口径：当前结果用于解释 gate 何时有效、何时仅持平，不据此宣称 Gated-GAT 全局优于 Always-GAT。",
    ])
    for row in gate_vs_always_rows:
        lines.append(
            f"{row['scene']} | bucket={row['scene_bucket']} | delta={row['delta_gated_vs_always']:+.1f} | {row['status']}"
        )
    lines.extend(["", "Scene Type Rankings", "------------------"])
    for scene_type, payload in rankings.items():
        sumo_rank = ", ".join(f"{name}={value:.1f}" for name, value in payload["sumo_ranking"])
        lines.append(f"{scene_type}: {sumo_rank}")
    lines.extend(["", "Delta vs Sparse", "---------------"])
    for row in delta_rows:
        lines.append(
            f"{row['scene']} | {row['method']} | delta={row['delta_vs_sparse']:.1f} | {row['status']}"
        )
    lines.extend(["", "Failure Cases", "-------------"])
    if not failure_rows:
        lines.append("none")
    else:
        for row in failure_rows:
            lines.append(
                f"{row['scene']} | {row['method']} | sumo={row['sumo_time']:.1f}s | "
                f"constraint={row['constraint_rate']:.1f}% | gate={row['gat_mode']}"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Sparse-LoRA vs Always/Gated GAT ablation.")
    parser.add_argument("--gat-model", default="/root/autodl-tmp/gat_model.pt")
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--without-sumo", action="store_true")
    args = parser.parse_args()

    (
        case_rows,
        overall,
        normal,
        special,
        scene_type_tables,
        delta_rows,
        gate_vs_always_rows,
        failure_rows,
        rankings,
        checkpoint_status,
    ) = run_ablation(args.gat_model, with_sumo=not args.without_sumo, alpha=args.alpha)

    bundle = create_result_bundle(
        mode="experiment_mode",
        planner="gated_gat_ablation",
        model="sparse_lora_gat",
    )

    case_csv = os.path.join(bundle.run_dir, "gated_gat_case_metrics.csv")
    overall_csv = os.path.join(bundle.run_dir, "gated_gat_overall_summary.csv")
    normal_csv = os.path.join(bundle.run_dir, "gated_gat_normal_summary.csv")
    special_csv = os.path.join(bundle.run_dir, "gated_gat_special_summary.csv")
    delta_csv = os.path.join(bundle.run_dir, "gated_gat_delta_vs_sparse.csv")
    gate_vs_always_csv = os.path.join(bundle.run_dir, "gated_gat_vs_always.csv")
    failure_csv = os.path.join(bundle.run_dir, "gated_gat_failure_cases.csv")

    write_metrics_csv(case_csv, case_rows)
    write_metrics_csv(overall_csv, overall)
    write_metrics_csv(normal_csv, normal)
    write_metrics_csv(special_csv, special)
    write_metrics_csv(delta_csv, delta_rows)
    write_metrics_csv(gate_vs_always_csv, gate_vs_always_rows)
    write_metrics_csv(failure_csv, failure_rows)
    write_json(os.path.join(bundle.run_dir, "scene_type_tables.json"), scene_type_tables)
    write_json(os.path.join(bundle.run_dir, "rankings.json"), rankings)
    write_json(os.path.join(bundle.run_dir, "checkpoint_status.json"), checkpoint_status)

    report_text = render_report(overall, normal, special, rankings, delta_rows, gate_vs_always_rows, failure_rows, checkpoint_status)
    config_snapshot = default_config_snapshot(
        timestamp=bundle.run_id.split("__")[0],
        mode="experiment_mode",
        seed=42,
        planner="gated_gat_ablation",
        model="sparse_lora_gat",
        scenarios=[row["scene"] for row in case_rows],
        fallback_policy="no_checkpoint_means_logic_only_and_no_fake_quant",
        default_weight_policy="Sparse-LoRA defaults to 2.0 for non-explicit edges",
        timing_policy="generate_weights infer time + compare SUMO time",
        extra={
            "gat_model": args.gat_model,
            "checkpoint_status": checkpoint_status,
            "overall_csv": overall_csv,
            "normal_csv": normal_csv,
            "special_csv": special_csv,
            "delta_csv": delta_csv,
            "gate_vs_always_csv": gate_vs_always_csv,
            "failure_csv": failure_csv,
            "case_csv": case_csv,
        },
    )

    write_standard_artifacts(
        bundle,
        summary={
            "run_id": bundle.run_id,
            "checkpoint_status": checkpoint_status,
            "case_csv": case_csv,
            "overall_csv": overall_csv,
            "normal_csv": normal_csv,
            "special_csv": special_csv,
            "delta_csv": delta_csv,
            "gate_vs_always_csv": gate_vs_always_csv,
            "failure_csv": failure_csv,
        },
        metrics_rows=overall,
        report_text=report_text,
        config_snapshot=config_snapshot,
        metrics_field_order=[
            "subset",
            "method",
            "samples",
            "avg_path_cost",
            "avg_constraint_rate",
            "avg_parsed_edge_ratio",
            "avg_signal_ratio",
            "avg_infer_time_s",
            "avg_sumo_time",
        ],
    )

    print(f"\n✅ gated gat overall summary -> {overall_csv}")
    for row in overall:
        print(
            f"{row['method']}: sumo={row['avg_sumo_time']:.1f}s | "
            f"constraint={row['avg_constraint_rate']:.1f}% | parsed={row['avg_parsed_edge_ratio']:.1f}%"
        )


if __name__ == "__main__":
    main()
