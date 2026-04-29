#!/usr/bin/env python3
"""Compare baseline sparse prompt vs spatiotemporal-enhanced sparse prompt."""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/autodl-tmp/code")

from app_fixed import PathPlanningEngine
from config import ConfigManager
from eval_metrics import compute_edge_metrics
from inference import build_sparse_baseline_prompt, build_sparse_task_prompt
from results_manager import create_result_bundle, default_config_snapshot, write_json, write_metrics_csv, write_standard_artifacts
from scenarios import get_spatiotemporal_prompt_cases


MODEL_PATH = "/root/autodl-tmp/model_merged_sparse"


class SingleModelManager:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is not None and self.tokenizer is not None:
            return self.tokenizer, self.model

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
        return self.tokenizer, self.model

    def close(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def mean_or_zero(values):
    return round(statistics.mean(values), 3) if values else 0.0


def build_prompt(engine, prompt_mode: str, constraint: str) -> str:
    if prompt_mode == "baseline_sparse_prompt":
        return build_sparse_baseline_prompt(engine, constraint)
    if prompt_mode == "spatiotemporal_enhanced_prompt":
        return build_sparse_task_prompt(engine, constraint)
    raise ValueError(f"unknown prompt_mode: {prompt_mode}")


def infer_sparse_with_prompt(engine, tokenizer, model, instruction: str):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=False).to(model.device)
    prompt_tokens = int(encoded["input_ids"].shape[1])
    in_len = prompt_tokens
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=320,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    infer_time = time.time() - t0
    raw = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
    parsed = engine._parse_sparse_output_bundle(raw)
    final_weights = {eid: 2.0 for eid in engine._ALL_EDGES}
    final_weights.update(parsed.get("mapped_weights", {}))
    return prompt, raw, parsed, final_weights, infer_time, prompt_tokens


def score_case(case: dict, parsed_bundle: dict, final_weights: dict) -> dict:
    gt = case["ground_truth"]
    expected_edges = list(gt.keys())
    forbidden_edges = list(case.get("forbidden_edges", []))

    target_hits = 0
    for eid in expected_edges:
        pred = float(final_weights.get(eid, 2.0))
        gold = float(gt[eid])
        if abs(pred - 2.0) > 0.75 and (gold >= 8.5 and pred >= 6.5 or gold >= 5.0 and pred >= 4.0 or gold <= 1.3 and pred <= 2.2):
            target_hits += 1

    reverse_leak = 0
    for eid in forbidden_edges:
        pred = float(final_weights.get(eid, 2.0))
        if pred >= 4.0:
            reverse_leak += 1

    mae = 0.0
    if expected_edges:
        mae = statistics.mean(abs(float(final_weights.get(eid, 2.0)) - float(gt[eid])) for eid in expected_edges)

    target_recall = round(100.0 * target_hits / len(expected_edges), 1) if expected_edges else 0.0
    reverse_leakage = round(100.0 * reverse_leak / len(forbidden_edges), 1) if forbidden_edges else 0.0
    success = int(target_recall >= 60.0 and reverse_leakage <= 25.0 and mae <= 3.0)

    return {
        "target_recall": target_recall,
        "reverse_leakage": reverse_leakage,
        "weight_mae": round(float(mae), 3),
        "scene_success": success,
        "anchor_count": int(parsed_bundle.get("anchor_count", 0)),
        "parse_confidence": round(float(parsed_bundle.get("parse_confidence", 0.0)), 3),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["focus"], row["prompt_mode"])].append(row)

    out = []
    for focus, prompt_mode in sorted(buckets):
        items = buckets[(focus, prompt_mode)]
        out.append({
            "focus": focus,
            "prompt_mode": prompt_mode,
            "samples": len(items),
            "avg_prompt_tokens": mean_or_zero([r["prompt_tokens"] for r in items]),
            "avg_infer_time_s": mean_or_zero([r["infer_time_s"] for r in items]),
            "avg_parsed_edge_ratio": mean_or_zero([r["parsed_edge_ratio"] for r in items]),
            "avg_signal_ratio": mean_or_zero([r["signal_ratio"] for r in items]),
            "avg_target_recall": mean_or_zero([r["target_recall"] for r in items]),
            "avg_reverse_leakage": mean_or_zero([r["reverse_leakage"] for r in items]),
            "avg_weight_mae": mean_or_zero([r["weight_mae"] for r in items]),
            "avg_anchor_count": mean_or_zero([r["anchor_count"] for r in items]),
            "avg_parse_confidence": mean_or_zero([r["parse_confidence"] for r in items]),
            "success_rate": round(100.0 * sum(r["scene_success"] for r in items) / len(items), 1),
        })
    return out


def build_case_deltas(case_rows: list[dict]) -> list[dict]:
    by_case = defaultdict(dict)
    for row in case_rows:
        by_case[row["case_name"]][row["prompt_mode"]] = row

    out = []
    for case_name, payload in sorted(by_case.items()):
        base = payload.get("baseline_sparse_prompt")
        enh = payload.get("spatiotemporal_enhanced_prompt")
        if not base or not enh:
            continue
        delta = round(float(enh["target_recall"]) - float(base["target_recall"]), 1)
        leak_delta = round(float(enh["reverse_leakage"]) - float(base["reverse_leakage"]), 1)
        status = "tie"
        if delta >= 10.0 and leak_delta <= 5.0:
            status = "helpful"
        elif delta <= -10.0 or leak_delta >= 15.0:
            status = "negative"
        out.append({
            "case_name": case_name,
            "focus": enh["focus"],
            "baseline_target_recall": base["target_recall"],
            "enhanced_target_recall": enh["target_recall"],
            "delta_target_recall": delta,
            "baseline_reverse_leakage": base["reverse_leakage"],
            "enhanced_reverse_leakage": enh["reverse_leakage"],
            "delta_reverse_leakage": leak_delta,
            "status": status,
        })
    return out


def render_report(summary_rows: list[dict], delta_rows: list[dict], failures: list[dict]) -> str:
    lines = [
        "Spatiotemporal Prompt Ablation",
        "==============================",
        "",
        "Summary",
        "-------",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['focus']} | {row['prompt_mode']}: "
            f"recall={row['avg_target_recall']:.1f}%, "
            f"leak={row['avg_reverse_leakage']:.1f}%, "
            f"success={row['success_rate']:.1f}%, "
            f"parse_conf={row['avg_parse_confidence']:.2f}"
        )
    lines.extend(["", "Case Deltas", "-----------"])
    for row in delta_rows:
        lines.append(
            f"{row['case_name']} | delta_recall={row['delta_target_recall']:+.1f} | "
            f"delta_leak={row['delta_reverse_leakage']:+.1f} | {row['status']}"
        )
    lines.extend(["", "Failure Cases", "-------------"])
    if not failures:
        lines.append("none")
    else:
        for row in failures:
            lines.append(
                f"{row['case_name']} | {row['prompt_mode']} | "
                f"recall={row['target_recall']:.1f}% | leak={row['reverse_leakage']:.1f}% | mae={row['weight_mae']:.2f}"
            )
    return "\n".join(lines) + "\n"


def run_prompt_ablation(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"missing sparse model: {model_path}")

    config = ConfigManager()
    config.config["RUN_MODE"] = "experiment_mode"
    engine = PathPlanningEngine(config)
    cases = get_spatiotemporal_prompt_cases()
    manager = SingleModelManager(model_path)
    tokenizer, model = manager.load()

    case_rows = []
    try:
        for case in cases:
            for prompt_mode in ("baseline_sparse_prompt", "spatiotemporal_enhanced_prompt"):
                instruction = build_prompt(engine, prompt_mode, case["constraint"])
                prompt, raw, parsed, final_weights, infer_time, prompt_tokens = infer_sparse_with_prompt(
                    engine, tokenizer, model, instruction
                )
                edge_metrics = compute_edge_metrics(
                    final_weights,
                    engine._ALL_EDGES,
                    explicit_weights=parsed.get("mapped_weights", {}),
                    method_name="Sparse-LoRA",
                )
                score = score_case(case, parsed, final_weights)
                row = {
                    "case_id": case["id"],
                    "case_name": case["name"],
                    "focus": case["focus"],
                    "scene_type": case["scene_type"],
                    "prompt_mode": prompt_mode,
                    "prompt_chars": len(prompt),
                    "prompt_tokens": prompt_tokens,
                    "infer_time_s": round(float(infer_time), 3),
                    "parsed_edges": edge_metrics["parsed_edges"],
                    "parsed_edge_ratio": edge_metrics["parsed_edge_ratio"],
                    "signal_edges": edge_metrics["signal_edges"],
                    "signal_ratio": edge_metrics["signal_ratio"],
                    "anchor_count": score["anchor_count"],
                    "parse_confidence": score["parse_confidence"],
                    "target_recall": score["target_recall"],
                    "reverse_leakage": score["reverse_leakage"],
                    "weight_mae": score["weight_mae"],
                    "scene_success": score["scene_success"],
                    "raw_preview": raw[:280],
                    "instruction_preview": instruction[:280],
                }
                case_rows.append(row)
                print(
                    f"{case['name']} | {prompt_mode:<30} | recall={row['target_recall']:5.1f}% | "
                    f"leak={row['reverse_leakage']:5.1f}% | anchors={row['anchor_count']}"
                )
    finally:
        manager.close()

    summary_rows = aggregate(case_rows)
    delta_rows = build_case_deltas(case_rows)
    failures = [
        row for row in case_rows
        if row["scene_success"] == 0 or row["reverse_leakage"] >= 25.0 or row["target_recall"] < 60.0
    ]
    return case_rows, summary_rows, delta_rows, failures


def main():
    parser = argparse.ArgumentParser(description="Spatiotemporal prompt ablation for Sparse-LoRA.")
    parser.add_argument("--model-path", default=MODEL_PATH)
    args = parser.parse_args()

    case_rows, summary_rows, delta_rows, failures = run_prompt_ablation(args.model_path)

    bundle = create_result_bundle(
        mode="experiment_mode",
        planner="spatiotemporal_prompt_ablation",
        model="sparse_lora",
    )
    case_csv = os.path.join(bundle.run_dir, "spatiotemporal_prompt_case_metrics.csv")
    summary_csv = os.path.join(bundle.run_dir, "spatiotemporal_prompt_summary.csv")
    delta_csv = os.path.join(bundle.run_dir, "spatiotemporal_prompt_deltas.csv")
    failure_json = os.path.join(bundle.run_dir, "spatiotemporal_prompt_failures.json")

    write_metrics_csv(case_csv, case_rows)
    write_metrics_csv(summary_csv, summary_rows)
    write_metrics_csv(delta_csv, delta_rows)
    write_json(failure_json, failures)

    report_text = render_report(summary_rows, delta_rows, failures)
    config_snapshot = default_config_snapshot(
        timestamp=bundle.run_id.split("__")[0],
        mode="experiment_mode",
        seed=42,
        planner="spatiotemporal_prompt_ablation",
        model="sparse_lora",
        scenarios=[row["case_name"] for row in case_rows],
        fallback_policy="none",
        default_weight_policy="explicit_sparse_anchor_defaults_to_2.0",
        timing_policy="prompt_inference_time_separate_from_scene_score",
        extra={
            "model_path": args.model_path,
            "case_csv": case_csv,
            "summary_csv": summary_csv,
            "delta_csv": delta_csv,
            "failure_json": failure_json,
        },
    )

    write_standard_artifacts(
        bundle,
        summary={
            "run_id": bundle.run_id,
            "case_csv": case_csv,
            "summary_csv": summary_csv,
            "delta_csv": delta_csv,
            "failure_json": failure_json,
        },
        metrics_rows=summary_rows,
        report_text=report_text,
        config_snapshot=config_snapshot,
    )
    print(f"\n✅ spatiotemporal prompt summary -> {summary_csv}")


if __name__ == "__main__":
    main()
