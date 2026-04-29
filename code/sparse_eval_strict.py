from __future__ import annotations

"""Strict evaluation helpers for Sparse-LoRA-v2."""

import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_metrics import compute_edge_metrics
from roadnet_meta import NORMAL_DEFAULT_WEIGHT, live_edge_ids
from sparse_utils import parse_sparse_output_bundle

CHANGED_EDGE_TOLERANCE = 0.05
TARGET_EDGE_DELTA = 2.0

_DIR_RULE = "DIR_RULE=ANCHOR中DIR字段须与EVENT中DIR完全一致；不得颠倒方向（向东≠向西，向北≠向南）；无明确方向时填写双向。"


def _inject_dir_rule(prompt: str) -> str:
    marker = "OUTPUT_RULE="
    if _DIR_RULE in prompt:
        return prompt
    lines = prompt.splitlines()
    out = []
    for line in lines:
        out.append(line)
        if line.startswith(marker):
            out.append(_DIR_RULE)
    return "\n".join(out)


def load_rows(parquet_dir: str, limit: int | None = None) -> list[dict]:
    table = pq.read_table(os.path.join(parquet_dir, "data.parquet"))
    rows = table.to_pylist()
    return rows[:limit] if limit else rows


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_path: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=(
            torch.bfloat16
            if device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
            if device == "cuda"
            else torch.float32
        ),
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def unload_model(model) -> None:
    try:
        del model
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def infer_sparse_row(
    *,
    model,
    tokenizer,
    prompt_text: str,
    scene_type: str | None,
    device: str,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=False).to(device)
    in_len = inputs["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    infer_time = time.time() - t0
    raw_generation = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
    bundle = parse_sparse_output_bundle(raw_generation, scene_type=scene_type)
    return {
        "prompt": prompt,
        "raw_generation": raw_generation,
        "bundle": bundle,
        "infer_time_s": infer_time,
    }


def dense_prediction(bundle: dict, edge_ids: list[str]) -> dict[str, float]:
    pred = {edge_id: NORMAL_DEFAULT_WEIGHT for edge_id in edge_ids}
    pred.update(bundle.get("mapped_weights", {}))
    return pred


def changed_edge_ids(
    truth: dict[str, float],
    *,
    default_weight: float = NORMAL_DEFAULT_WEIGHT,
    tolerance: float = CHANGED_EDGE_TOLERANCE,
) -> list[str]:
    return [
        edge_id
        for edge_id, value in truth.items()
        if abs(float(value) - float(default_weight)) > tolerance
    ]


def target_edge_ids(
    truth: dict[str, float],
    *,
    default_weight: float = NORMAL_DEFAULT_WEIGHT,
    delta_threshold: float = TARGET_EDGE_DELTA,
) -> list[str]:
    return [
        edge_id
        for edge_id, value in truth.items()
        if abs(float(value) - float(default_weight)) > delta_threshold
    ]


def is_no_anchor(bundle: dict) -> bool:
    return int(bundle.get("anchor_count", 0)) == 0 and not bool(bundle.get("parse_fail", False))


def edge_recall_hit(pred: float, gold: float, *, default_weight: float = NORMAL_DEFAULT_WEIGHT) -> bool:
    gold_delta = float(gold) - float(default_weight)
    pred_delta = float(pred) - float(default_weight)
    if abs(gold_delta) <= CHANGED_EDGE_TOLERANCE:
        return False
    strength = max(0.25, min(0.6, abs(gold_delta) * 0.45))
    return pred_delta >= strength if gold_delta > 0 else pred_delta <= -strength


def target_recall_hit(pred: float, gold: float, *, default_weight: float = NORMAL_DEFAULT_WEIGHT) -> bool:
    value = float(gold)
    pred_value = float(pred)
    if value >= default_weight + TARGET_EDGE_DELTA:
        if value >= 8.5:
            return pred_value >= 6.5
        if value >= 6.0:
            return pred_value >= 4.8
        return pred_value >= 3.8
    if value <= default_weight - TARGET_EDGE_DELTA:
        return pred_value <= 1.2
    return False


def mae(values_a: dict[str, float], values_b: dict[str, float], edge_ids: list[str]) -> float:
    if not edge_ids:
        return 0.0
    return sum(abs(float(values_a.get(edge_id, NORMAL_DEFAULT_WEIGHT)) - float(values_b.get(edge_id, NORMAL_DEFAULT_WEIGHT))) for edge_id in edge_ids) / len(edge_ids)


def _changed_edge_abs_error_sum(pred: dict[str, float], truth: dict[str, float], edge_ids: list[str]) -> float:
    return sum(abs(float(pred.get(edge_id, NORMAL_DEFAULT_WEIGHT)) - float(truth.get(edge_id, NORMAL_DEFAULT_WEIGHT))) for edge_id in edge_ids)


def _bundle_view(bundle: dict) -> dict[str, Any]:
    return {
        "anchor_count": int(bundle.get("anchor_count", 0)),
        "parse_confidence": round(float(bundle.get("parse_confidence", 0.0)), 3),
        "parse_fail": bool(bundle.get("parse_fail", False)),
        "no_anchor": is_no_anchor(bundle),
        "mapped_edge_count": len(bundle.get("mapped_weights", {})),
        "anchors": bundle.get("anchors", []),
        "conflicts": bundle.get("conflicts", []),
    }


def score_prediction(
    *,
    row: dict,
    bundle: dict,
    pred_weights: dict[str, float],
    edge_ids: list[str],
    sample_id: str,
    split: str,
    split_index: int,
    raw_generation: str,
    infer_time_s: float,
) -> dict[str, Any]:
    truth = json.loads(row["ground_truth_json"])
    changed_edges = changed_edge_ids(truth)
    target_edges = target_edge_ids(truth)
    no_anchor = is_no_anchor(bundle)
    strict_parse_fail = bool(bundle.get("parse_fail", False)) or (no_anchor and bool(changed_edges))
    strict_failure_reason = "ok"
    if bundle.get("parse_fail", False):
        strict_failure_reason = "parse_fail"
    elif no_anchor and target_edges:
        strict_failure_reason = "no_anchor_with_target_edges"
    elif no_anchor and changed_edges:
        strict_failure_reason = "no_anchor_with_changed_edges"

    changed_hits = sum(1 for edge_id in changed_edges if edge_recall_hit(pred_weights.get(edge_id, NORMAL_DEFAULT_WEIGHT), truth[edge_id]))
    target_hits = sum(1 for edge_id in target_edges if target_recall_hit(pred_weights.get(edge_id, NORMAL_DEFAULT_WEIGHT), truth[edge_id]))
    changed_abs_error_sum = _changed_edge_abs_error_sum(pred_weights, truth, changed_edges)
    target_abs_error_sum = _changed_edge_abs_error_sum(pred_weights, truth, target_edges)

    edge_metrics = compute_edge_metrics(
        pred_weights,
        edge_ids,
        explicit_weights=bundle.get("mapped_weights", {}),
        explicit_edge_ids=bundle.get("mapped_weights", {}).keys(),
        default_weight=NORMAL_DEFAULT_WEIGHT,
    )

    changed_mae = changed_abs_error_sum / len(changed_edges) if changed_edges else 0.0
    target_mae = target_abs_error_sum / len(target_edges) if target_edges else 0.0

    return {
        "sample_id": sample_id,
        "split": split,
        "split_index": split_index,
        "scene_type": row.get("scene_type", ""),
        "length_bucket": row.get("length_bucket", ""),
        "constraint_text": row.get("constraint_text", ""),
        "prompt_text": _inject_dir_rule(json.loads(row["messages"])[0]["content"]),
        "ground_truth_json": row["ground_truth_json"],
        "gt_changed_edge_count": len(changed_edges),
        "gt_target_edge_count": len(target_edges),
        "gt_changed_edges": changed_edges,
        "gt_target_edges": target_edges,
        "no_anchor": no_anchor,
        "parse_fail": bool(bundle.get("parse_fail", False)),
        "strict_parse_fail": strict_parse_fail,
        "strict_failure_reason": strict_failure_reason,
        "parse_confidence": round(float(bundle.get("parse_confidence", 0.0)), 3),
        "anchor_count": int(bundle.get("anchor_count", 0)),
        "mapped_edge_count": len(bundle.get("mapped_weights", {})),
        "changed_edge_hits": changed_hits,
        "target_edge_hits": target_hits,
        "gt_changed_edge_recall": round(100.0 * changed_hits / len(changed_edges), 1) if changed_edges else 0.0,
        "target_recall": round(100.0 * target_hits / len(target_edges), 1) if target_edges else 0.0,
        "dense_weight_mae": round(mae(pred_weights, truth, edge_ids), 4),
        "mae_on_changed_edges": round(changed_mae, 4),
        "mae_on_target_edges": round(target_mae, 4),
        "changed_edge_abs_error_sum": round(changed_abs_error_sum, 6),
        "target_edge_abs_error_sum": round(target_abs_error_sum, 6),
        "edge_metrics": edge_metrics,
        "prediction_raw": raw_generation,
        "prediction_raw_preview": raw_generation[:240].replace("\n", "\\n"),
        "parser_output": _bundle_view(bundle),
        "pred_abnormal_edges": {
            edge_id: pred_weights[edge_id]
            for edge_id in edge_ids
            if abs(float(pred_weights.get(edge_id, NORMAL_DEFAULT_WEIGHT)) - NORMAL_DEFAULT_WEIGHT) > CHANGED_EDGE_TOLERANCE
        },
        "infer_time_s": round(float(infer_time_s), 4),
    }


def evaluate_rows(
    *,
    model,
    tokenizer,
    device: str,
    rows: list[dict],
    split: str,
    edge_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    edge_ids = edge_ids or live_edge_ids()
    details = []
    for split_index, row in enumerate(rows):
        run = infer_sparse_row(
            model=model,
            tokenizer=tokenizer,
            prompt_text=_inject_dir_rule(json.loads(row["messages"])[0]["content"]),
            scene_type=row.get("scene_type"),
            device=device,
        )
        pred_weights = dense_prediction(run["bundle"], edge_ids)
        details.append(
            score_prediction(
                row=row,
                bundle=run["bundle"],
                pred_weights=pred_weights,
                edge_ids=edge_ids,
                sample_id=f"{split}:{split_index}",
                split=split,
                split_index=split_index,
                raw_generation=run["raw_generation"],
                infer_time_s=run["infer_time_s"],
            )
        )
    return details


def summarize_details(details: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(details) or 1
    changed_samples = [item for item in details if item["gt_changed_edge_count"] > 0]
    target_samples = [item for item in details if item["gt_target_edge_count"] > 0]
    strict_fail_samples = [item for item in details if item["strict_parse_fail"]]
    no_anchor_samples = [item for item in details if item["no_anchor"]]
    no_anchor_changed_samples = [item for item in changed_samples if item["no_anchor"]]
    no_anchor_target_samples = [item for item in target_samples if item["no_anchor"]]

    changed_edge_total = sum(item["gt_changed_edge_count"] for item in details)
    target_edge_total = sum(item["gt_target_edge_count"] for item in details)
    changed_hit_total = sum(item["changed_edge_hits"] for item in details)
    target_hit_total = sum(item["target_edge_hits"] for item in details)
    changed_abs_error_sum = sum(item["changed_edge_abs_error_sum"] for item in details)
    target_abs_error_sum = sum(item["target_edge_abs_error_sum"] for item in details)
    parsed_edge_total = sum(item["edge_metrics"]["parsed_edges"] for item in details)
    signal_edge_total = sum(item["edge_metrics"]["signal_edges"] for item in details)
    total_edge_total = sum(item["edge_metrics"]["total_edges"] for item in details)

    return {
        "count": len(details),
        "no_anchor_rate": round(len(no_anchor_samples) / count, 4),
        "no_anchor_when_gt_changed_rate": round(len(no_anchor_changed_samples) / max(len(changed_samples), 1), 4),
        "no_anchor_when_gt_target_changed_rate": round(len(no_anchor_target_samples) / max(len(target_samples), 1), 4),
        "parse_fail_rate": round(sum(int(item["parse_fail"]) for item in details) / count, 4),
        "parse_fail_rate_strict": round(len(strict_fail_samples) / count, 4),
        "target_recall": round(100.0 * target_hit_total / max(target_edge_total, 1), 1) if target_edge_total else 0.0,
        "gt_changed_edge_recall": round(100.0 * changed_hit_total / max(changed_edge_total, 1), 1) if changed_edge_total else 0.0,
        "MAE_on_changed_edges": round(changed_abs_error_sum / max(changed_edge_total, 1), 4) if changed_edge_total else 0.0,
        "MAE_on_target_edges": round(target_abs_error_sum / max(target_edge_total, 1), 4) if target_edge_total else 0.0,
        "dense_weight_mae": round(sum(item["dense_weight_mae"] for item in details) / count, 4),
        "parse_confidence": round(sum(item["parse_confidence"] for item in details) / count, 4),
        "anchor_count": round(sum(item["anchor_count"] for item in details) / count, 4),
        "parsed_edge_ratio": round(parsed_edge_total / max(total_edge_total, 1), 4) if total_edge_total else 0.0,
        "signal_edge_ratio": round(signal_edge_total / max(total_edge_total, 1), 4) if total_edge_total else 0.0,
        "inference_time": round(sum(item["infer_time_s"] for item in details) / count, 4),
        "gt_changed_sample_rate": round(len(changed_samples) / count, 4),
        "gt_target_sample_rate": round(len(target_samples) / count, 4),
        "avg_gt_changed_edge_count": round(changed_edge_total / count, 4),
        "avg_gt_target_edge_count": round(target_edge_total / count, 4),
        "gt_changed_edge_total": int(changed_edge_total),
        "gt_target_edge_total": int(target_edge_total),
        "strict_fail_reasons": dict(Counter(item["strict_failure_reason"] for item in details)),
    }


def group_details(details: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in details:
        buckets[str(item.get(key, "") or "unknown")].append(item)
    return {
        bucket: summarize_details(items)
        for bucket, items in sorted(buckets.items(), key=lambda pair: pair[0])
    }


def build_report(
    *,
    model_path: str,
    dataset_root: str,
    split_details: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": model_path,
        "dataset_root": dataset_root,
        "metric_formulas": {
            "no_anchor_rate": "samples with parser anchor_count=0 and parse_fail=false divided by total samples",
            "no_anchor_when_gt_changed_rate": "samples with at least one GT changed edge (abs(gt-2.0)>0.05) and no_anchor divided by GT-changed samples",
            "no_anchor_when_gt_target_changed_rate": "samples with task-relevant GT target edges (abs(gt-2.0)>2.0) and no_anchor divided by GT-target samples",
            "target_recall": "edge-level recall over GT target edges (abs(gt-2.0)>2.0) using severity-aware hit thresholds",
            "gt_changed_edge_recall": "edge-level recall over all GT changed edges (abs(gt-2.0)>0.05) using sign-aware non-default detection",
            "MAE_on_changed_edges": "absolute error summed only on GT changed edges divided by GT changed edge count",
            "MAE_on_target_edges": "absolute error summed only on GT target edges divided by GT target edge count",
            "parse_fail_rate_strict": "parser parse_fail OR no_anchor on a sample with any GT changed edge, divided by total samples",
            "dense_weight_mae": "legacy dense MAE over all edges after default-2.0 fill, retained only for comparison",
        },
        "splits": {},
        "overall": {},
        "by_scene_type_overall": {},
        "by_length_bucket_overall": {},
    }

    all_details: list[dict[str, Any]] = []
    for split, details in split_details.items():
        all_details.extend(details)
        report["splits"][split] = {
            "summary": summarize_details(details),
            "by_scene_type": group_details(details, "scene_type"),
            "by_length_bucket": group_details(details, "length_bucket"),
        }

    report["overall"] = summarize_details(all_details)
    report["by_scene_type_overall"] = group_details(all_details, "scene_type")
    report["by_length_bucket_overall"] = group_details(all_details, "length_bucket")
    return report


def write_summary_csv(report: dict[str, Any], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "scope_name",
        "count",
        "no_anchor_rate",
        "no_anchor_when_gt_changed_rate",
        "no_anchor_when_gt_target_changed_rate",
        "parse_fail_rate",
        "parse_fail_rate_strict",
        "target_recall",
        "gt_changed_edge_recall",
        "MAE_on_changed_edges",
        "MAE_on_target_edges",
        "dense_weight_mae",
        "parse_confidence",
        "anchor_count",
        "parsed_edge_ratio",
        "signal_edge_ratio",
        "inference_time",
        "gt_changed_sample_rate",
        "gt_target_sample_rate",
        "avg_gt_changed_edge_count",
        "avg_gt_target_edge_count",
    ]
    rows: list[dict[str, Any]] = []
    rows.append({"scope": "overall", "scope_name": "overall", **report["overall"]})
    for split, payload in report["splits"].items():
        rows.append({"scope": "split", "scope_name": split, **payload["summary"]})
        for scene_type, summary in payload["by_scene_type"].items():
            rows.append({"scope": f"{split}:scene_type", "scope_name": scene_type, **summary})
        for length_bucket, summary in payload["by_length_bucket"].items():
            rows.append({"scope": f"{split}:length_bucket", "scope_name": length_bucket, **summary})
    for scene_type, summary in report["by_scene_type_overall"].items():
        rows.append({"scope": "overall:scene_type", "scope_name": scene_type, **summary})
    for length_bucket, summary in report["by_length_bucket_overall"].items():
        rows.append({"scope": "overall:length_bucket", "scope_name": length_bucket, **summary})

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_detail_csv(details: list[dict[str, Any]], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "split_index",
        "scene_type",
        "length_bucket",
        "gt_changed_edge_count",
        "gt_target_edge_count",
        "no_anchor",
        "parse_fail",
        "strict_parse_fail",
        "strict_failure_reason",
        "parse_confidence",
        "anchor_count",
        "mapped_edge_count",
        "target_recall",
        "gt_changed_edge_recall",
        "MAE_on_changed_edges",
        "MAE_on_target_edges",
        "dense_weight_mae",
        "parsed_edge_ratio",
        "signal_edge_ratio",
        "infer_time_s",
        "prediction_raw_preview",
        "constraint_text",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in details:
            row = dict(item)
            row["parsed_edge_ratio"] = round(float(item["edge_metrics"]["parsed_edge_ratio"]) / 100.0, 4)
            row["signal_edge_ratio"] = round(float(item["edge_metrics"]["signal_ratio"]) / 100.0, 4)
            writer.writerow({key: row.get(key, "") for key in fieldnames})
