#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import torch

from roadnet_meta import live_edge_ids
from sparse_eval_strict import (
    build_report,
    dense_prediction,
    evaluate_rows,
    group_details,
    infer_sparse_row,
    load_model,
    load_rows,
    load_tokenizer,
    score_prediction,
    summarize_details,
    unload_model,
    write_detail_csv,
    write_summary_csv,
)


METRIC_COLUMNS = [
    "count",
    "no_anchor_when_gt_changed_rate",
    "target_recall",
    "gt_changed_edge_recall",
    "MAE_on_changed_edges",
    "parse_fail_rate_strict",
]

TARGET_STRUCTURE_TAGS = {"short_plain", "prefixed_short"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stage4d micro patch and produce final release decision.")
    parser.add_argument("--model-path", default="/root/autodl-tmp/model_merged_sparse_v2_stage4d_micro")
    parser.add_argument("--lora-root", default="/root/autodl-tmp/model_lora_sparse_v2_stage4d_micro")
    parser.add_argument("--heldout-root", default="/root/autodl-tmp/dataset_sparse_v2")
    parser.add_argument("--stage4d-root", default="/root/autodl-tmp/dataset_sparse_v2_stage4d_micro")
    parser.add_argument("--results-root", default="/root/autodl-tmp/results")
    parser.add_argument("--pre-model-path", default="/root/autodl-tmp/model_merged_sparse_v2_stage4_fix")
    parser.add_argument("--stage4fix-summary-csv", default="/root/autodl-tmp/results/sparse_v2_validation_strict_stage4_fix_summary.csv")
    parser.add_argument("--stage4c-summary-csv", default="/root/autodl-tmp/results/sparse_v2_validation_strict_stage4c_fix_summary.csv")
    parser.add_argument("--stage4c-guard-summary-csv", default="/root/autodl-tmp/results/stage4c_guard_eval_summary.csv")
    parser.add_argument("--remaining-hard-json", default="/root/autodl-tmp/results/stage4c_remaining_hard_cases.json")
    return parser.parse_args()


def enrich_details(details: list[dict[str, Any]], rows: list[dict[str, Any]], extra_fields: list[str]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for detail, row in zip(details, rows, strict=True):
        merged = dict(detail)
        for field in extra_fields:
            merged[field] = row.get(field, "")
        enriched.append(merged)
    return enriched


def write_slice_summary_csv(
    *,
    output_path: Path,
    overall: dict[str, Any],
    grouped: dict[str, dict[str, dict[str, Any]]],
) -> None:
    fieldnames = ["scope", "scope_name", *METRIC_COLUMNS]
    rows = [{"scope": "overall", "scope_name": "overall", **overall}]
    for group_name, payload in grouped.items():
        for scope_name, summary in payload.items():
            rows.append({"scope": group_name, "scope_name": scope_name, **summary})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def evaluate_slice(
    *,
    model,
    tokenizer,
    device: str,
    rows: list[dict[str, Any]],
    split_name: str,
    group_fields: list[str],
    extra_fields: list[str],
    output_summary_csv: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    details = evaluate_rows(
        model=model,
        tokenizer=tokenizer,
        device=device,
        rows=rows,
        split=split_name,
        edge_ids=live_edge_ids(),
    )
    details = enrich_details(details, rows, extra_fields)
    grouped = {field: group_details(details, field) for field in group_fields}
    overall = summarize_details(details)
    write_slice_summary_csv(output_path=output_summary_csv, overall=overall, grouped=grouped)
    return overall, grouped, details


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def lookup_summary(rows: list[dict[str, str]], *, scope: str, scope_name: str) -> dict[str, str]:
    for row in rows:
        if row["scope"] == scope and row["scope_name"] == scope_name:
            return row
    raise KeyError(f"Missing summary row {scope}/{scope_name} in {rows}")


def to_float(summary_row: dict[str, Any], key: str) -> float:
    return float(summary_row.get(key, 0.0))


def latest_checkpoint_state(lora_root: Path) -> dict[str, Any]:
    checkpoints = sorted(
        [path for path in lora_root.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found under {lora_root}")
    state_path = checkpoints[-1] / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing trainer_state.json: {state_path}")
    data = json.loads(state_path.read_text(encoding="utf-8"))
    data["_trainer_state_path"] = str(state_path)
    return data


def extract_training_readout(trainer_state: dict[str, Any]) -> dict[str, Any]:
    history = trainer_state.get("log_history", []) or []
    loss_entries = [item for item in history if "loss" in item]
    eval_entries = [item for item in history if "eval_loss" in item]
    lr_entries = [item for item in history if "learning_rate" in item]

    final_train_loss = float(loss_entries[-1]["loss"]) if loss_entries else None
    final_eval_loss = float(eval_entries[-1]["eval_loss"]) if eval_entries else None
    final_lr = float(lr_entries[-1]["learning_rate"]) if lr_entries else None
    max_steps = int(trainer_state.get("max_steps", 0) or 0)
    global_step = int(trainer_state.get("global_step", 0) or 0)
    learning_rates = [float(item["learning_rate"]) for item in lr_entries if "learning_rate" in item]
    stable = all(
        str(item.get("loss", "")).lower() not in {"nan", "inf", "-inf"}
        for item in loss_entries
    ) and all(
        str(item.get("eval_loss", "")).lower() not in {"nan", "inf", "-inf"}
        for item in eval_entries
    )
    return {
        "trainer_state_path": trainer_state["_trainer_state_path"],
        "actual_global_step": global_step,
        "max_steps": max_steps,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "final_learning_rate": final_lr,
        "first_learning_rate": learning_rates[0] if learning_rates else None,
        "min_learning_rate_logged": min(learning_rates) if learning_rates else None,
        "stable": stable,
    }


def scope_name(value: Any) -> str:
    text = str(value).strip()
    return text if text else "unknown"


def summarize_negative_details(details: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(details) or 1
    false_positive = [item for item in details if int(item.get("anchor_count", 0)) > 0 and item.get("gt_changed_edge_count", 0) == 0]
    parse_fail = [item for item in details if bool(item.get("parse_fail", False))]
    no_anchor = [item for item in details if int(item.get("anchor_count", 0)) == 0 and not bool(item.get("parse_fail", False))]
    return {
        "count": len(details),
        "false_positive_anchor_count": len(false_positive),
        "false_positive_anchor_rate": round(len(false_positive) / count, 4),
        "exact_no_anchor_count": len(no_anchor),
        "exact_no_anchor_rate": round(len(no_anchor) / count, 4),
        "parse_fail_count": len(parse_fail),
        "parse_fail_rate": round(len(parse_fail) / count, 4),
        "avg_anchor_count": round(sum(float(item.get("anchor_count", 0)) for item in details) / count, 4),
        "avg_parse_confidence": round(sum(float(item.get("parse_confidence", 0.0)) for item in details) / count, 4),
    }


def group_negative_details(details: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in details:
        buckets[scope_name(item.get(field, ""))].append(item)
    return {
        bucket: summarize_negative_details(items)
        for bucket, items in sorted(buckets.items(), key=lambda pair: pair[0])
    }


def write_negative_summary_csv(
    *,
    output_path: Path,
    overall: dict[str, Any],
    grouped: dict[str, dict[str, dict[str, Any]]],
) -> None:
    fieldnames = [
        "scope",
        "scope_name",
        "count",
        "false_positive_anchor_count",
        "false_positive_anchor_rate",
        "exact_no_anchor_count",
        "exact_no_anchor_rate",
        "parse_fail_count",
        "parse_fail_rate",
        "avg_anchor_count",
        "avg_parse_confidence",
    ]
    rows = [{"scope": "overall", "scope_name": "overall", **overall}]
    for group_name, payload in grouped.items():
        for scope_name_value, summary in payload.items():
            rows.append({"scope": group_name, "scope_name": scope_name_value, **summary})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def normalize_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def load_rows_by_sample_ids(dataset_root: Path, sample_ids: list[str]) -> pd.DataFrame:
    split_to_indices: dict[str, set[int]] = defaultdict(set)
    for sample_id in sample_ids:
        split, split_index_text = str(sample_id).split(":")
        split_to_indices[split].add(int(split_index_text))

    frames: list[pd.DataFrame] = []
    for split, indices in sorted(split_to_indices.items()):
        table = pq.read_table(dataset_root / split / "data.parquet")
        frame = pd.DataFrame(table.to_pylist())
        frame["split_index"] = range(len(frame))
        frame["sample_id"] = frame["split"].astype(str) + ":" + frame["split_index"].astype(str)
        frame = frame[frame["split_index"].isin(indices)].copy()
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    ordering = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    merged["_sample_order"] = merged["sample_id"].map(ordering)
    merged = merged.sort_values("_sample_order").drop(columns=["_sample_order"]).reset_index(drop=True)
    return merged


def classify_output(bundle: dict[str, Any]) -> str:
    return "ANCHOR" if int(bundle.get("anchor_count", 0)) > 0 else "NO_ANCHOR"


def canonical_anchor_signature(bundle: dict[str, Any]) -> list[str]:
    anchors = bundle.get("anchors", []) or []
    signatures = []
    for anchor in anchors:
        signatures.append(
            "ROAD={road}|DIR={direction}|RANGE={range_}|LEVEL={level}".format(
                road=str(anchor.get("road", "")).strip(),
                direction=str(anchor.get("dir", "")).strip(),
                range_=str(anchor.get("range", "")).strip(),
                level=str(anchor.get("level", "")).strip(),
            )
        )
    return sorted(signatures)


def post_is_fully_correct(scored: dict[str, Any]) -> bool:
    if bool(scored.get("parse_fail", False)):
        return False
    if int(scored.get("changed_edge_hits", 0)) != int(scored.get("gt_changed_edge_count", 0)):
        return False
    if int(scored.get("target_edge_hits", 0)) != int(scored.get("gt_target_edge_count", 0)):
        return False
    return True


def infer_for_model(
    *,
    model_path: str,
    dataset_rows: pd.DataFrame,
    device: str,
    edge_ids: list[str],
) -> dict[str, dict[str, Any]]:
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path, device)
    results: dict[str, dict[str, Any]] = {}
    try:
        for row in dataset_rows.itertuples(index=False):
            prompt_text = json.loads(row.messages)[0]["content"]
            run = infer_sparse_row(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                scene_type=row.scene_type,
                device=device,
            )
            pred_weights = dense_prediction(run["bundle"], edge_ids)
            scored = score_prediction(
                row={
                    "scene_type": row.scene_type,
                    "length_bucket": getattr(row, "length_bucket", ""),
                    "constraint_text": row.constraint_text,
                    "messages": row.messages,
                    "ground_truth_json": row.ground_truth_json,
                },
                bundle=run["bundle"],
                pred_weights=pred_weights,
                edge_ids=edge_ids,
                sample_id=row.sample_id,
                split=row.split,
                split_index=int(row.split_index),
                raw_generation=run["raw_generation"],
                infer_time_s=run["infer_time_s"],
            )
            results[str(row.sample_id)] = {
                "raw_output": run["raw_generation"],
                "output_label": classify_output(run["bundle"]),
                "anchor_signature": canonical_anchor_signature(run["bundle"]),
                "bundle": run["bundle"],
                "score": scored,
            }
    finally:
        unload_model(model)
    return results


def derive_change_type(pre_item: dict[str, Any], post_item: dict[str, Any]) -> tuple[bool, str]:
    pre_label = pre_item["output_label"]
    post_label = post_item["output_label"]
    if pre_label == "NO_ANCHOR" and post_label == "NO_ANCHOR":
        return False, "no_change"
    if pre_label == "NO_ANCHOR" and post_label == "ANCHOR":
        return True, "no_anchor_to_anchor"
    if pre_label == "ANCHOR" and post_label == "NO_ANCHOR":
        return True, "anchor_to_no_anchor"
    if pre_item["anchor_signature"] == post_item["anchor_signature"]:
        return False, "no_change"
    return True, "anchor_changed"


def build_target_diff_report(
    *,
    remaining_hard_json: Path,
    dataset_root: Path,
    pre_model_path: str,
    post_model_path: str,
) -> dict[str, Any]:
    payload = json.loads(remaining_hard_json.read_text(encoding="utf-8"))
    target_cases = [
        item
        for item in payload.get("still_failed_single_edge_cases", [])
        if str(item.get("semantic_gap_subtype", "")) == "relief_normal_single_edge"
        and str(item.get("structure_tag", "")) in TARGET_STRUCTURE_TAGS
    ]
    negative_reference_cases = payload.get("new_false_positive_normal_only_anchors", [])

    target_sample_ids = [str(item["sample_id"]) for item in target_cases]
    negative_sample_ids = [str(item["sample_id"]) for item in negative_reference_cases]
    target_rows = load_rows_by_sample_ids(dataset_root, target_sample_ids)
    negative_rows = load_rows_by_sample_ids(dataset_root, negative_sample_ids)

    if len(target_rows) != len(target_sample_ids):
        raise ValueError(f"Expected {len(target_sample_ids)} target rows, found {len(target_rows)}")
    if len(negative_rows) != len(negative_sample_ids):
        raise ValueError(f"Expected {len(negative_sample_ids)} negative rows, found {len(negative_rows)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    edge_ids = live_edge_ids()

    pre_target = infer_for_model(model_path=pre_model_path, dataset_rows=target_rows, device=device, edge_ids=edge_ids)
    post_target = infer_for_model(model_path=post_model_path, dataset_rows=target_rows, device=device, edge_ids=edge_ids)
    pre_negative = infer_for_model(model_path=pre_model_path, dataset_rows=negative_rows, device=device, edge_ids=edge_ids)
    post_negative = infer_for_model(model_path=post_model_path, dataset_rows=negative_rows, device=device, edge_ids=edge_ids)

    target_lookup = {str(item["sample_id"]): item for item in target_cases}
    samples = []
    for row in target_rows.itertuples(index=False):
        sample_id = str(row.sample_id)
        pre_item = pre_target[sample_id]
        post_item = post_target[sample_id]
        meta = target_lookup[sample_id]
        changed, change_type = derive_change_type(pre_item, post_item)
        post_anchor_still_wrong = post_item["output_label"] == "ANCHOR" and not post_is_fully_correct(post_item["score"])
        samples.append(
            {
                "sample_id": sample_id,
                "split": row.split,
                "split_index": int(row.split_index),
                "constraint_text": row.constraint_text,
                "length_bucket": meta.get("length_bucket", ""),
                "phrase_variant": meta.get("phrase_variant", ""),
                "structure_tag": meta.get("structure_tag", ""),
                "pre_output": pre_item["output_label"],
                "post_output": post_item["output_label"],
                "changed": changed,
                "change_type": change_type,
                "post_anchor_still_wrong": post_anchor_still_wrong,
                "pre_anchor_signature": pre_item["anchor_signature"],
                "post_anchor_signature": post_item["anchor_signature"],
                "pre_parse_fail": bool(pre_item["score"]["parse_fail"]),
                "post_parse_fail": bool(post_item["score"]["parse_fail"]),
                "pre_changed_recall": float(pre_item["score"]["gt_changed_edge_recall"]),
                "post_changed_recall": float(post_item["score"]["gt_changed_edge_recall"]),
                "pre_target_recall": float(pre_item["score"]["target_recall"]),
                "post_target_recall": float(post_item["score"]["target_recall"]),
                "pre_raw_output": pre_item["raw_output"],
                "post_raw_output": post_item["raw_output"],
            }
        )

    negative_lookup = {str(item["sample_id"]): item for item in negative_reference_cases}
    negative_samples = []
    for row in negative_rows.itertuples(index=False):
        sample_id = str(row.sample_id)
        pre_item = pre_negative[sample_id]
        post_item = post_negative[sample_id]
        meta = negative_lookup[sample_id]
        negative_samples.append(
            {
                "sample_id": sample_id,
                "constraint_text": row.constraint_text,
                "phrase_variant": meta.get("phrase_variant", ""),
                "structure_tag": meta.get("structure_tag", ""),
                "pre_output": pre_item["output_label"],
                "post_output": post_item["output_label"],
                "pre_anchor_signature": pre_item["anchor_signature"],
                "post_anchor_signature": post_item["anchor_signature"],
                "pre_is_false_positive": pre_item["output_label"] == "ANCHOR",
                "post_is_false_positive": post_item["output_label"] == "ANCHOR",
                "pre_raw_output": pre_item["raw_output"],
                "post_raw_output": post_item["raw_output"],
            }
        )

    stats = {
        "sample_count": len(samples),
        "completely_unchanged_count": sum(1 for item in samples if item["change_type"] == "no_change"),
        "no_anchor_to_anchor_count": sum(1 for item in samples if item["change_type"] == "no_anchor_to_anchor"),
        "anchor_but_still_wrong_count": sum(1 for item in samples if item["post_anchor_still_wrong"]),
        "false_positive_new_count": sum(1 for item in negative_samples if item["post_is_false_positive"]),
        "anchor_changed_count": sum(1 for item in samples if item["change_type"] == "anchor_changed"),
        "anchor_to_no_anchor_count": sum(1 for item in samples if item["change_type"] == "anchor_to_no_anchor"),
        "post_anchor_and_fully_correct_count": sum(
            1 for item in samples if item["post_output"] == "ANCHOR" and not item["post_anchor_still_wrong"]
        ),
        "negative_reference_sample_count": len(negative_samples),
        "negative_reference_pre_false_positive_count": sum(1 for item in negative_samples if item["pre_is_false_positive"]),
        "negative_reference_post_false_positive_count": sum(1 for item in negative_samples if item["post_is_false_positive"]),
        "changed_phrase_variant_breakdown": dict(Counter(item["phrase_variant"] for item in samples if item["changed"])),
        "changed_structure_breakdown": dict(Counter(item["structure_tag"] for item in samples if item["changed"])),
    }

    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_remaining_hard_json": str(remaining_hard_json),
        "pre_model_path": pre_model_path,
        "post_model_path": post_model_path,
        "target_bucket_name": "single_edge_changed_relief__short_plain_or_prefixed",
        "sample_count": stats["sample_count"],
        "completely_unchanged_count": stats["completely_unchanged_count"],
        "no_anchor_to_anchor_count": stats["no_anchor_to_anchor_count"],
        "anchor_but_still_wrong_count": stats["anchor_but_still_wrong_count"],
        "false_positive_new_count": stats["false_positive_new_count"],
        "anchor_changed_count": stats["anchor_changed_count"],
        "anchor_to_no_anchor_count": stats["anchor_to_no_anchor_count"],
        "stats": stats,
        "samples": samples,
        "negative_reference_samples": negative_samples,
    }


def bool_text(value: bool) -> str:
    return "是" if value else "否"


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    heldout_json = results_root / "sparse_v2_validation_strict_stage4d_micro.json"
    heldout_summary_csv = results_root / "sparse_v2_validation_strict_stage4d_micro_summary.csv"
    heldout_detail_csv = results_root / "sparse_v2_validation_strict_stage4d_micro_details.csv"
    target_summary_csv = results_root / "stage4d_micro_target_bucket_summary.csv"
    hard_negative_summary_csv = results_root / "stage4d_micro_hard_negative_guard_summary.csv"
    canary_summary_csv = results_root / "stage4d_micro_canary_summary.csv"
    target_diff_json = results_root / "stage4d_micro_target_diff.json"
    final_decision_md = results_root / "stage4d_micro_final_decision.md"

    trainer_state = latest_checkpoint_state(Path(args.lora_root))
    training_readout = extract_training_readout(trainer_state)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path, device)

    try:
        edge_ids = live_edge_ids()
        split_details: dict[str, list[dict[str, Any]]] = {}
        all_heldout_details: list[dict[str, Any]] = []
        for split in ("val_normal", "val_special", "val_anti_truncation"):
            rows = load_rows(str(Path(args.heldout_root) / split))
            details = evaluate_rows(
                model=model,
                tokenizer=tokenizer,
                device=device,
                rows=rows,
                split=split,
                edge_ids=edge_ids,
            )
            split_details[split] = details
            all_heldout_details.extend(details)

        heldout_report = build_report(
            model_path=args.model_path,
            dataset_root=args.heldout_root,
            split_details=split_details,
        )
        heldout_json.write_text(json.dumps(heldout_report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_summary_csv(heldout_report, str(heldout_summary_csv))
        write_detail_csv(all_heldout_details, str(heldout_detail_csv))

        stage4d_eval_rows = load_rows(str(Path(args.stage4d_root) / "eval"))
        target_rows = [row for row in stage4d_eval_rows if str(row.get("mixture_component", "")) == "micro_patch_positive_core"]
        hard_negative_rows = [row for row in stage4d_eval_rows if str(row.get("mixture_component", "")) == "hard_negative_guard"]
        canary_rows = [row for row in stage4d_eval_rows if str(row.get("mixture_component", "")) == "minimal_stability_canary"]

        target_overall, _, _ = evaluate_slice(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=target_rows,
            split_name="stage4d_micro_target_eval",
            group_fields=["phrase_variant", "structure_variant", "source_type", "anchor_style"],
            extra_fields=["mixture_component", "phrase_variant", "structure_variant", "source_type", "anchor_style"],
            output_summary_csv=target_summary_csv,
        )

        hard_negative_details = evaluate_rows(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=hard_negative_rows,
            split="stage4d_micro_hard_negative_guard",
            edge_ids=edge_ids,
        )
        hard_negative_details = enrich_details(
            hard_negative_details,
            hard_negative_rows,
            ["mixture_component", "phrase_variant", "structure_variant", "source_type", "anchor_style"],
        )
        hard_negative_overall = summarize_negative_details(hard_negative_details)
        hard_negative_grouped = {
            field: group_negative_details(hard_negative_details, field)
            for field in ["phrase_variant", "structure_variant", "source_type"]
        }
        write_negative_summary_csv(
            output_path=hard_negative_summary_csv,
            overall=hard_negative_overall,
            grouped=hard_negative_grouped,
        )

        canary_overall, _, _ = evaluate_slice(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=canary_rows,
            split_name="stage4d_micro_canary_eval",
            group_fields=["canary_kind", "structure_variant", "source_type"],
            extra_fields=["mixture_component", "canary_kind", "structure_variant", "source_type"],
            output_summary_csv=canary_summary_csv,
        )
    finally:
        unload_model(model)

    diff_report = build_target_diff_report(
        remaining_hard_json=Path(args.remaining_hard_json),
        dataset_root=Path(args.heldout_root),
        pre_model_path=args.pre_model_path,
        post_model_path=args.model_path,
    )
    policy_aligned_false_positive_count = int(hard_negative_overall["false_positive_anchor_count"])
    policy_aligned_negative_count = int(hard_negative_overall["count"])
    diff_report["false_positive_new_count"] = policy_aligned_false_positive_count
    diff_report["stats"]["false_positive_new_count"] = policy_aligned_false_positive_count
    diff_report["stats"]["policy_aligned_negative_reference_count"] = policy_aligned_negative_count
    target_diff_json.write_text(json.dumps(diff_report, ensure_ascii=False, indent=2), encoding="utf-8")

    stage4fix_rows = read_summary_csv(Path(args.stage4fix_summary_csv))
    stage4c_rows = read_summary_csv(Path(args.stage4c_summary_csv))
    stage4d_rows = read_summary_csv(heldout_summary_csv)
    stage4c_guard_rows = read_summary_csv(Path(args.stage4c_guard_summary_csv))

    simple_local_fix = lookup_summary(stage4fix_rows, scope="overall:scene_type", scope_name="simple_local")
    simple_local_stage4c = lookup_summary(stage4c_rows, scope="overall:scene_type", scope_name="simple_local")
    simple_local_stage4d = lookup_summary(stage4d_rows, scope="overall:scene_type", scope_name="simple_local")
    stage4c_guard_overall = lookup_summary(stage4c_guard_rows, scope="overall", scope_name="overall")

    target_sample_count = int(diff_report["sample_count"])
    no_anchor_to_anchor_count = int(diff_report["no_anchor_to_anchor_count"])
    unchanged_count = int(diff_report["completely_unchanged_count"])
    anchor_but_still_wrong_count = int(diff_report["anchor_but_still_wrong_count"])
    stage4d_false_positive_count = policy_aligned_false_positive_count
    remaining_hard_payload = json.loads(Path(args.remaining_hard_json).read_text(encoding="utf-8"))
    stage4c_false_positive_count = len(remaining_hard_payload.get("new_false_positive_normal_only_anchors", []))

    target_bucket_moved = no_anchor_to_anchor_count > 0
    target_bucket_moved_clearly = no_anchor_to_anchor_count >= 3
    normal_only_lower_than_stage4c = stage4d_false_positive_count < stage4c_false_positive_count
    simple_local_no_anchor_delta = to_float(simple_local_stage4c, "no_anchor_when_gt_changed_rate") - to_float(simple_local_stage4d, "no_anchor_when_gt_changed_rate")
    simple_local_recall_delta = to_float(simple_local_stage4d, "gt_changed_edge_recall") - to_float(simple_local_stage4c, "gt_changed_edge_recall")
    simple_local_parse_delta = to_float(simple_local_stage4d, "parse_fail_rate_strict") - to_float(simple_local_stage4c, "parse_fail_rate_strict")
    simple_local_net_gain = (
        simple_local_no_anchor_delta >= 0.01
        and simple_local_recall_delta >= -1.0
        and simple_local_parse_delta <= 0.02
    )
    canary_stable = (
        to_float(canary_overall, "parse_fail_rate_strict") <= 0.05
        and to_float(canary_overall, "gt_changed_edge_recall") >= max(70.0, to_float(stage4c_guard_overall, "gt_changed_edge_recall") - 10.0)
    )

    decision = (
        "RELEASE_TO_GAT_V2"
        if target_bucket_moved_clearly
        and normal_only_lower_than_stage4c
        and simple_local_net_gain
        and canary_stable
        and training_readout["stable"]
        else "FREEZE_STAGE4_FIX_AND_STOP_PATCHING"
    )

    md_lines = [
        "# stage4d_micro Final Decision",
        "",
        "## Required Answers",
        f"- 目标桶是否终于明显动了：`{bool_text(target_bucket_moved_clearly)}`（sample_count=`{target_sample_count}`，no_anchor_to_anchor=`{no_anchor_to_anchor_count}`，completely_unchanged=`{unchanged_count}`）",
        f"- no_anchor_to_anchor 有多少条：`{no_anchor_to_anchor_count}`",
        f"- normal-only 假阳性是否少于 stage4c：`{bool_text(normal_only_lower_than_stage4c)}`（stage4d=`{stage4d_false_positive_count}`，stage4c=`{stage4c_false_positive_count}`）",
        f"- simple_local overall 是否有可观净收益：`{bool_text(simple_local_net_gain)}`（no_anchor_when_gt_changed_rate: stage4c=`{simple_local_stage4c['no_anchor_when_gt_changed_rate']}` -> stage4d=`{simple_local_stage4d['no_anchor_when_gt_changed_rate']}`；gt_changed_edge_recall: stage4c=`{simple_local_stage4c['gt_changed_edge_recall']}` -> stage4d=`{simple_local_stage4d['gt_changed_edge_recall']}`；parse_fail_rate_strict: stage4c=`{simple_local_stage4c['parse_fail_rate_strict']}` -> stage4d=`{simple_local_stage4d['parse_fail_rate_strict']}`）",
        f"- 是否达到放行条件：`{bool_text(decision == 'RELEASE_TO_GAT_V2')}`",
        f"- 若仍未达到，是否正式停止所有后续 patch：`{bool_text(decision == 'FREEZE_STAGE4_FIX_AND_STOP_PATCHING')}`",
        "",
        "## Supporting Readout",
        f"- target_bucket_moved_any: `{target_bucket_moved}`",
        f"- anchor_but_still_wrong_count: `{anchor_but_still_wrong_count}`",
        f"- stage4_fix simple_local: `no_anchor_when_gt_changed_rate={simple_local_fix['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={simple_local_fix['gt_changed_edge_recall']}`, `parse_fail_rate_strict={simple_local_fix['parse_fail_rate_strict']}`",
        f"- stage4c simple_local: `no_anchor_when_gt_changed_rate={simple_local_stage4c['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={simple_local_stage4c['gt_changed_edge_recall']}`, `parse_fail_rate_strict={simple_local_stage4c['parse_fail_rate_strict']}`",
        f"- stage4d simple_local: `no_anchor_when_gt_changed_rate={simple_local_stage4d['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={simple_local_stage4d['gt_changed_edge_recall']}`, `parse_fail_rate_strict={simple_local_stage4d['parse_fail_rate_strict']}`",
        f"- target bucket eval overall: `no_anchor_when_gt_changed_rate={target_overall['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={target_overall['gt_changed_edge_recall']}`, `parse_fail_rate_strict={target_overall['parse_fail_rate_strict']}`",
        f"- hard negative guard overall: `false_positive_anchor_count={hard_negative_overall['false_positive_anchor_count']}`, `false_positive_anchor_rate={hard_negative_overall['false_positive_anchor_rate']}`, `exact_no_anchor_rate={hard_negative_overall['exact_no_anchor_rate']}`",
        f"- canary overall: `no_anchor_when_gt_changed_rate={canary_overall['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={canary_overall['gt_changed_edge_recall']}`, `parse_fail_rate_strict={canary_overall['parse_fail_rate_strict']}`",
        f"- training steps: `{training_readout['actual_global_step']}` / `{training_readout['max_steps']}`",
        f"- final train loss: `{training_readout['final_train_loss']}`",
        f"- final eval loss: `{training_readout['final_eval_loss']}`",
        f"- trainer_state: `{training_readout['trainer_state_path']}`",
        "",
        "## Final Decision",
        f"- `{decision}`",
    ]
    final_decision_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "decision": decision,
                "heldout_summary_csv": str(heldout_summary_csv),
                "heldout_detail_csv": str(heldout_detail_csv),
                "target_summary_csv": str(target_summary_csv),
                "hard_negative_summary_csv": str(hard_negative_summary_csv),
                "canary_summary_csv": str(canary_summary_csv),
                "target_diff_json": str(target_diff_json),
                "final_decision_md": str(final_decision_md),
                "target_bucket_moved_clearly": target_bucket_moved_clearly,
                "no_anchor_to_anchor_count": no_anchor_to_anchor_count,
                "stage4d_false_positive_count": stage4d_false_positive_count,
                "stage4c_false_positive_count": stage4c_false_positive_count,
                "simple_local_net_gain": simple_local_net_gain,
                "canary_stable": canary_stable,
                "training_stable": training_readout["stable"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
