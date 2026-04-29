#!/usr/bin/env python3
"""Run minimal truncation-resilience experiments for full vs sparse outputs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/autodl-tmp/code")
sys.path.insert(0, "/root/autodl-tmp/compare")

from app_fixed import PathPlanningEngine
from config import ConfigManager
from eval_metrics import compute_edge_metrics
from inference import build_sparse_task_prompt, generate_weights
from results_manager import (
    create_result_bundle,
    default_config_snapshot,
    write_json,
    write_metrics_csv,
    write_standard_artifacts,
)
from run_compare import TEST_CASES
from scenarios import APP_SCENARIO_PRESETS, classify_scene_type


MODEL_SPECS = [
    {"model_name": "Qwen-LoRA", "path": "/root/autodl-tmp/model_merged_qwen"},
    {"model_name": "Sparse-LoRA", "path": "/root/autodl-tmp/model_merged_sparse"},
    {"model_name": "R1-LoRA", "path": "/root/autodl-tmp/model_merged_r1"},
]

VARIANT_ORDER = ["short", "medium", "long", "noisy_long"]
VARIANT_COMPLEXITY = {"short": 1, "medium": 2, "long": 3, "noisy_long": 4}

SCENE_TYPE_BY_NAME = {
    item["name"]: item["scene_type"]
    for item in APP_SCENARIO_PRESETS
}
SCENE_TYPE_BY_ID = {
    item["id"]: item["scene_type"]
    for item in APP_SCENARIO_PRESETS
}


NOISE_UNIT = (
    "补充背景：以下为值班安排、热线提醒、会务通知、停车倡议、天气播报、志愿服务登记和社区公告，"
    "仅用于制造长文本干扰，与真实道路状态无关。"
)

NOISE_UNIT_ALT = (
    "附加信息：这一段只包含行政说明、宣讲口径、乘车礼仪和公共服务提示，不包含任何真实路况，"
    "也不对应具体道路。"
)


@dataclass
class PromptStats:
    prompt_tokens: int
    used_tokens: int
    max_length: int
    truncated: bool
    prompt_chars: int


class SingleModelManager:
    """Load exactly one merged model at a time to keep memory bounded."""

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
        if "R1" in self.model_path or "DeepSeek" in self.model_path:
            self.tokenizer.padding_side = "left"
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


def build_variant_description(desc: str, variant: str) -> str:
    if variant == "short":
        return desc
    if variant == "medium":
        prefix = " ".join([NOISE_UNIT, NOISE_UNIT_ALT] * 6)
        return f"{prefix}\n最终请只以最后一段真实路况为准：{desc}"
    if variant == "long":
        prefix = " ".join([NOISE_UNIT, NOISE_UNIT_ALT] * 36)
        return f"{prefix}\n最终仅以下真实路况有效：{desc}"
    if variant == "noisy_long":
        chunks = []
        for idx in range(140):
            unit = NOISE_UNIT if idx % 2 == 0 else NOISE_UNIT_ALT
            chunks.append(f"第{idx+1}条补充说明：{unit}")
        prefix = " ".join(chunks)
        return f"{prefix}\n最后一段才是真实交通描述，请忽略前文无关信息：{desc}"
    raise ValueError(f"unknown variant: {variant}")


def resolve_scene_type(name: str, desc: str) -> str:
    direct = SCENE_TYPE_BY_NAME.get(name)
    if direct:
        return direct
    match = re.search(r"场景([A-F])", name)
    if match and match.group(1) in SCENE_TYPE_BY_ID:
        return SCENE_TYPE_BY_ID[match.group(1)]
    return classify_scene_type(name=name, text=desc, event_count=desc.count("；") + 1)


def build_prompt_for_measurement(engine, tokenizer, model_name: str, constraint: str):
    if "Sparse" in model_name or "sparse" in model_name:
        instruction = build_sparse_task_prompt(engine, constraint)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return instruction, prompt, 768

    total_edges = len(engine._ALL_EDGES)
    instruction = constraint
    if not any(keyword in instruction for keyword in ("生成权重", "输出权重", "分配", "路段权重", "权重值")):
        instruction = instruction.rstrip("。") + f"。请为全部{total_edges}条路段生成权重（0-10，越大越拥堵）。"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return instruction, prompt, 4096


def measure_prompt(tokenizer, prompt: str, max_length: int) -> PromptStats:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False, padding=False)
    prompt_tokens = int(encoded["input_ids"].shape[1])
    return PromptStats(
        prompt_tokens=prompt_tokens,
        used_tokens=min(prompt_tokens, max_length),
        max_length=max_length,
        truncated=prompt_tokens > max_length,
        prompt_chars=len(prompt),
    )


def parse_failure(parsed_edges: int, anchor_count: int | None = None) -> int:
    if parsed_edges <= 0:
        return 1
    if anchor_count is not None and anchor_count <= 0:
        return 1
    return 0


def mean_or_zero(values):
    return round(statistics.mean(values), 3) if values else 0.0


def aggregate_rows(case_rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in case_rows:
        buckets.setdefault((row["model"], row["variant"]), []).append(row)

    out = []
    for model, variant in sorted(buckets.keys(), key=lambda item: (item[0], VARIANT_ORDER.index(item[1]))):
        rows = buckets[(model, variant)]
        out.append({
            "model": model,
            "variant": variant,
            "complexity_level": VARIANT_COMPLEXITY[variant],
            "samples": len(rows),
            "avg_prompt_tokens": mean_or_zero([r["prompt_tokens"] for r in rows]),
            "avg_used_tokens": mean_or_zero([r["used_tokens"] for r in rows]),
            "truncation_rate": round(100.0 * sum(r["truncated"] for r in rows) / len(rows), 1),
            "parse_failure_rate": round(100.0 * sum(r["parse_failure"] for r in rows) / len(rows), 1),
            "avg_parsed_edge_ratio": mean_or_zero([r["parsed_edge_ratio"] for r in rows]),
            "avg_signal_ratio": mean_or_zero([r["signal_ratio"] for r in rows]),
            "avg_inference_time_s": mean_or_zero([r["infer_time_s"] for r in rows]),
            "avg_anchor_count": mean_or_zero([r["anchor_count"] for r in rows]),
            "avg_parse_confidence": mean_or_zero([r["parse_confidence"] for r in rows]),
        })
    return out


def render_markdown_table(summary_rows: list[dict]) -> str:
    headers = [
        "Model",
        "Variant",
        "Truncation%",
        "ParseFail%",
        "ParsedEdge%",
        "Signal%",
        "Infer(s)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {variant} | {truncation_rate:.1f} | {parse_failure_rate:.1f} | "
            "{avg_parsed_edge_ratio:.1f} | {avg_signal_ratio:.1f} | {avg_inference_time_s:.2f} |".format(**row)
        )
    return "\n".join(lines)


def run_experiment(include_r1: bool = True):
    config = ConfigManager()
    config.config["RUN_MODE"] = "experiment_mode"
    engine = PathPlanningEngine(config)

    model_specs = [spec for spec in MODEL_SPECS if include_r1 or spec["model_name"] != "R1-LoRA"]
    model_specs = [spec for spec in model_specs if os.path.exists(spec["path"])]

    case_rows: list[dict] = []
    missing_models = [spec["model_name"] for spec in MODEL_SPECS if not os.path.exists(spec["path"])]

    for spec in model_specs:
        print(f"\n=== Loading {spec['model_name']} ===")
        manager = SingleModelManager(spec["path"])
        tokenizer, _ = manager.get_model(spec["model_name"])
        try:
            for tc in TEST_CASES:
                scene_type = resolve_scene_type(tc["name"], tc["desc"])
                for variant in VARIANT_ORDER:
                    constraint = build_variant_description(tc["desc"], variant)
                    _, prompt, max_length = build_prompt_for_measurement(engine, tokenizer, spec["model_name"], constraint)
                    prompt_stats = measure_prompt(tokenizer, prompt, max_length)

                    (
                        _struct,
                        raw_llm_weights,
                        final_planning_weights,
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
                        constraint,
                        spec["model_name"],
                        manager,
                        use_gat=False,
                        mode="experiment_mode",
                    )

                    edge_metrics = compute_edge_metrics(
                        final_planning_weights,
                        engine._ALL_EDGES,
                        explicit_weights=raw_llm_weights,
                        method_name=spec["model_name"],
                    )
                    sparse_bundle = getattr(engine, "_last_sparse_parse", {}) if "Sparse" in spec["model_name"] else {}
                    anchor_count = int(sparse_bundle.get("anchor_count", 0)) if sparse_bundle else 0
                    parse_conf = float(sparse_bundle.get("parse_confidence", 0.0)) if sparse_bundle else 0.0

                    case_rows.append({
                        "model": spec["model_name"],
                        "scene": tc["name"],
                        "scene_type": scene_type,
                        "variant": variant,
                        "complexity_level": VARIANT_COMPLEXITY[variant],
                        "prompt_chars": prompt_stats.prompt_chars,
                        "prompt_tokens": prompt_stats.prompt_tokens,
                        "used_tokens": prompt_stats.used_tokens,
                        "max_length": prompt_stats.max_length,
                        "truncated": int(prompt_stats.truncated),
                        "parse_failure": parse_failure(edge_metrics["parsed_edges"], anchor_count if "Sparse" in spec["model_name"] else None),
                        "parsed_edges": edge_metrics["parsed_edges"],
                        "parsed_edge_ratio": edge_metrics["parsed_edge_ratio"],
                        "signal_edges": edge_metrics["signal_edges"],
                        "signal_ratio": edge_metrics["signal_ratio"],
                        "infer_time_s": round(float(infer_time), 3),
                        "total_ms": float(timing_info.get("total_ms", 0.0)),
                        "anchor_count": anchor_count,
                        "parse_confidence": round(parse_conf, 3),
                    })
                    print(
                        f"{spec['model_name']:>11} | {variant:<10} | {tc['name']} | "
                        f"tokens={prompt_stats.prompt_tokens}/{prompt_stats.max_length} | "
                        f"trunc={int(prompt_stats.truncated)} | parsed={edge_metrics['parsed_edges']}"
                    )
        finally:
            manager.close()

    summary_rows = aggregate_rows(case_rows)
    return case_rows, summary_rows, missing_models


def main():
    parser = argparse.ArgumentParser(description="Run minimal truncation evidence experiments.")
    parser.add_argument("--without-r1", action="store_true", help="Skip R1-LoRA even if present.")
    args = parser.parse_args()

    case_rows, summary_rows, missing_models = run_experiment(include_r1=not args.without_r1)

    bundle = create_result_bundle(
        mode="experiment_mode",
        planner="truncation_experiment",
        model="llm_length_ablation",
    )

    case_csv = os.path.join(bundle.run_dir, "truncation_case_metrics.csv")
    summary_csv = os.path.join(bundle.run_dir, "truncation_summary.csv")

    write_metrics_csv(case_csv, case_rows)
    write_metrics_csv(summary_csv, summary_rows)
    write_json(os.path.join(bundle.run_dir, "truncation_summary.json"), {
        "missing_models": missing_models,
        "summary_rows": summary_rows,
    })

    report_text = (
        "Truncation Evidence Experiment\n"
        "============================\n\n"
        f"Missing models: {', '.join(missing_models) if missing_models else 'none'}\n\n"
        "Summary Table\n"
        "-------------\n"
        f"{render_markdown_table(summary_rows)}\n"
    )

    config_snapshot = default_config_snapshot(
        timestamp=bundle.run_id.split("__")[0],
        mode="experiment_mode",
        seed=42,
        planner="truncation_experiment",
        model="llm_length_ablation",
        scenarios=[row["scene"] for row in case_rows],
        fallback_policy="experiment_mode_no_rule_fallback",
        default_weight_policy="explicit_only_metrics_default_fill_2.0",
        timing_policy="generate_weights_return_infer_time_and_total_ms",
        extra={
            "variants": VARIANT_ORDER,
            "models": sorted({row["model"] for row in case_rows}),
            "missing_models": missing_models,
            "case_metrics_csv": case_csv,
            "summary_csv": summary_csv,
        },
    )

    write_standard_artifacts(
        bundle,
        summary={
            "run_id": bundle.run_id,
            "total_cases": len(case_rows),
            "models": sorted({row["model"] for row in case_rows}),
            "variants": VARIANT_ORDER,
            "missing_models": missing_models,
            "case_metrics_csv": case_csv,
            "summary_csv": summary_csv,
        },
        metrics_rows=summary_rows,
        report_text=report_text,
        config_snapshot=config_snapshot,
        metrics_field_order=[
            "model",
            "variant",
            "complexity_level",
            "samples",
            "avg_prompt_tokens",
            "avg_used_tokens",
            "truncation_rate",
            "parse_failure_rate",
            "avg_parsed_edge_ratio",
            "avg_signal_ratio",
            "avg_inference_time_s",
            "avg_anchor_count",
            "avg_parse_confidence",
        ],
    )

    print(f"\n✅ truncation summary -> {summary_csv}")
    print(render_markdown_table(summary_rows))


if __name__ == "__main__":
    main()
