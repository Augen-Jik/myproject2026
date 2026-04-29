#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from stage4b_common import (
    BASE_COLUMNS,
    OUTPUT_EXTRA_COLUMNS,
    PACK_COLUMNS,
    RELIEF_BUCKETS,
    anchor_style,
    clean_text,
    label_is_valid,
    read_parquet_rows,
    sample_id_from_split,
    validate_policy_text,
    validate_simple_local_policy,
)


REQUIRED_GUARD_BUCKETS = {
    "anti_truncation_guard",
    "directional_asymmetry_stable",
    "hard_negative_no_anchor",
    "existing_simple_local_buffer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stage4b dataset built from relabel + relief.")
    parser.add_argument("--stage4-train", default="/root/autodl-tmp/dataset_sparse_v2_stage4/train")
    parser.add_argument("--stage4-eval", default="/root/autodl-tmp/dataset_sparse_v2_stage4/eval")
    parser.add_argument("--relabel-train", default="/root/autodl-tmp/dataset_sparse_v2_stage4b_relabel/train/data.parquet")
    parser.add_argument("--relabel-eval", default="/root/autodl-tmp/dataset_sparse_v2_stage4b_relabel/eval/data.parquet")
    parser.add_argument("--relabel-mapping", default="/root/autodl-tmp/dataset_sparse_v2_stage4b_relabel/relabel_mapping.csv")
    parser.add_argument("--policy-md", default="/root/autodl-tmp/results/simple_local_v2_label_policy.md")
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/dataset_sparse_v2_stage4b")
    parser.add_argument("--output", default="/root/autodl-tmp/dataset_sparse_v2_stage4b/validation_report.json")
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {row["sample_id"]: row for row in rows}
    if len(rows) != len(mapping):
        raise ValueError("relabel_mapping.csv contains duplicated sample_id values.")
    return mapping


def load_reference_labels(rows: list[dict[str, Any]], split_name: str) -> dict[str, str]:
    return {
        sample_id_from_split(str(row.get("split", split_name)), index): str(row["response_text"])
        for index, row in enumerate(rows)
    }


def collect_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows, _ = read_parquet_rows(root / "train")
    eval_rows, _ = read_parquet_rows(root / "eval")
    return train_rows, eval_rows


def main() -> None:
    args = parse_args()

    policy_text = Path(args.policy_md).read_text(encoding="utf-8")
    validate_policy_text(policy_text)

    report: dict[str, Any] = {
        "dataset_root": args.dataset_root,
        "checks": {},
        "counts": {},
        "warnings": [],
        "fatal_errors": [],
    }

    dataset_root = Path(args.dataset_root)
    required_paths = [
        dataset_root / "train" / "data.parquet",
        dataset_root / "eval" / "data.parquet",
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        report["fatal_errors"].append(f"train/eval structure abnormal; missing {missing_paths}")
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit("stage4b validation failed: train/eval structure abnormal")

    train_rows, eval_rows = collect_rows(dataset_root)
    all_rows = train_rows + eval_rows

    stage4_train_rows, _ = read_parquet_rows(args.stage4_train)
    stage4_eval_rows, _ = read_parquet_rows(args.stage4_eval)
    relabel_train_rows, _ = read_parquet_rows(args.relabel_train)
    relabel_eval_rows, _ = read_parquet_rows(args.relabel_eval)
    relabel_mapping = load_mapping(Path(args.relabel_mapping))
    stage4_labels = {
        **load_reference_labels(stage4_train_rows, "train"),
        **load_reference_labels(stage4_eval_rows, "eval"),
    }
    relabel_labels = {
        **load_reference_labels(relabel_train_rows, "train"),
        **load_reference_labels(relabel_eval_rows, "eval"),
    }

    required_columns = set(BASE_COLUMNS) | set(OUTPUT_EXTRA_COLUMNS) | set(PACK_COLUMNS)
    missing_columns = sorted(required_columns - set(all_rows[0].keys())) if all_rows else sorted(required_columns)
    report["checks"]["required_columns_present"] = not missing_columns
    if missing_columns:
        report["fatal_errors"].append(f"train/eval structure abnormal; missing columns {missing_columns}")

    non_empty_labels = all(clean_text(str(row.get("response_text", ""))) for row in all_rows)
    valid_labels = all(label_is_valid(str(row.get("response_text", ""))) for row in all_rows)
    report["checks"]["label_non_empty"] = non_empty_labels
    report["checks"]["anchor_format_valid"] = valid_labels

    component_complete = all(clean_text(str(row.get("mixture_component", ""))) for row in all_rows)
    report["checks"]["mixture_component_complete"] = component_complete

    temporal_switch_rows = [row["stage4b_sample_id"] for row in all_rows if str(row.get("scene_type", "")) == "temporal_switch"]
    report["checks"]["temporal_switch_absent"] = not temporal_switch_rows
    if temporal_switch_rows:
        report["fatal_errors"].append(f"temporal_switch mixed in: {temporal_switch_rows[:8]}")

    simple_local_policy_errors: list[dict[str, str]] = []
    relief_policy_errors: list[dict[str, str]] = []
    for row in all_rows:
        if str(row.get("scene_type", "")) != "simple_local":
            continue
        ok, reason = validate_simple_local_policy(row)
        if not ok:
            simple_local_policy_errors.append({"stage4b_sample_id": row["stage4b_sample_id"], "reason": reason})
            if str(row.get("mixture_component")) == "relief_supervision_pack":
                relief_policy_errors.append({"stage4b_sample_id": row["stage4b_sample_id"], "reason": reason})
    report["checks"]["simple_local_policy_aligned"] = not simple_local_policy_errors
    report["checks"]["relief_policy_aligned"] = not relief_policy_errors
    if relief_policy_errors:
        report["fatal_errors"].append(f"relief label conflicts with policy-A: {relief_policy_errors[:8]}")

    relabel_component_rows = [row for row in all_rows if str(row.get("mixture_component")) == "relabeled_simple_local_repair"]
    relabel_errors: list[dict[str, str]] = []
    missing_mapping_ids: list[str] = []
    for row in relabel_component_rows:
        source_id = str(row.get("source_sample_id", ""))
        mapping_row = relabel_mapping.get(source_id)
        if mapping_row is None:
            missing_mapping_ids.append(source_id)
            continue
        if mapping_row["changed"].lower() != "true":
            relabel_errors.append({"source_sample_id": source_id, "reason": "mapping says changed=false"})
            continue
        if mapping_row["relabel_bucket"] != "anomaly_plus_normal_side_event":
            relabel_errors.append({"source_sample_id": source_id, "reason": f"unexpected mapping bucket {mapping_row['relabel_bucket']}"})
            continue
        if str(row["response_text"]) != str(mapping_row["new_label"]):
            relabel_errors.append({"source_sample_id": source_id, "reason": "row label does not match relabel mapping new_label"})
            continue
        if source_id in stage4_labels and str(stage4_labels[source_id]) == str(row["response_text"]):
            relabel_errors.append({"source_sample_id": source_id, "reason": "row label was not actually replaced from old stage4 label"})
        if source_id in relabel_labels and str(relabel_labels[source_id]) != str(row["response_text"]):
            relabel_errors.append({"source_sample_id": source_id, "reason": "row label diverges from relabel parquet source"})
    report["checks"]["relabel_mapping_complete"] = not missing_mapping_ids
    report["checks"]["relabel_repair_correct"] = not relabel_errors and not missing_mapping_ids
    if missing_mapping_ids:
        report["fatal_errors"].append(f"relabel mapping missing for source_sample_id values: {missing_mapping_ids[:8]}")

    guard_rows = [row for row in all_rows if str(row.get("mixture_component")) == "anti_regression_guard"]
    guard_bucket_counts = Counter(str(row.get("bucket", "")) for row in guard_rows)
    missing_guard_buckets = sorted(REQUIRED_GUARD_BUCKETS - set(guard_bucket_counts))
    report["checks"]["guard_bucket_coverage"] = not missing_guard_buckets
    if missing_guard_buckets:
        report["warnings"].append(f"anti_regression_guard missing buckets: {missing_guard_buckets}")

    anti_truncation_long_ok = all(
        str(row.get("length_bucket", "")) in {"long", "noisy_long"}
        for row in guard_rows
        if str(row.get("bucket", "")) == "anti_truncation_guard"
    )
    report["checks"]["anti_truncation_guard_is_long_context"] = anti_truncation_long_ok

    no_anchor_ratio = sum(1 for row in all_rows if anchor_style(str(row.get("response_text", ""))) == "NO_ANCHOR") / max(len(all_rows), 1)
    no_anchor_ratio_train = sum(1 for row in train_rows if anchor_style(str(row.get("response_text", ""))) == "NO_ANCHOR") / max(len(train_rows), 1)
    report["counts"]["rows"] = {"train": len(train_rows), "eval": len(eval_rows), "total": len(all_rows)}
    report["counts"]["mixture_component_counts"] = dict(sorted(Counter(str(row.get("mixture_component", "")) for row in all_rows).items()))
    report["counts"]["bucket_counts"] = dict(sorted(Counter(str(row.get("bucket", "")) for row in all_rows).items()))
    report["counts"]["scene_type_counts"] = dict(sorted(Counter(str(row.get("scene_type", "")) for row in all_rows).items()))
    report["counts"]["guard_bucket_counts"] = dict(sorted(guard_bucket_counts.items()))
    report["counts"]["no_anchor_ratio"] = round(no_anchor_ratio, 4)
    report["counts"]["no_anchor_ratio_train"] = round(no_anchor_ratio_train, 4)
    report["checks"]["no_anchor_ratio_controlled"] = no_anchor_ratio <= 0.15 and no_anchor_ratio_train <= 0.15
    if not report["checks"]["no_anchor_ratio_controlled"]:
        report["warnings"].append(
            f"NO_ANCHOR share looks high: overall={no_anchor_ratio:.4f}, train={no_anchor_ratio_train:.4f}"
        )

    report["counts"]["relief_bucket_counts"] = dict(sorted(
        Counter(str(row.get("bucket", "")) for row in all_rows if str(row.get("bucket", "")) in RELIEF_BUCKETS).items()
    ))
    report["counts"]["anchor_style_counts"] = dict(sorted(Counter(anchor_style(str(row.get("response_text", ""))) for row in all_rows).items()))
    report["checks"]["train_eval_non_empty"] = bool(train_rows) and bool(eval_rows)
    report["checks"]["relabel_component_present"] = bool(relabel_component_rows)
    report["checks"]["relief_component_present"] = any(str(row.get("mixture_component")) == "relief_supervision_pack" for row in all_rows)
    report["checks"]["guard_component_present"] = bool(guard_rows)

    if relabel_errors:
        report["warnings"].append(f"relabel issues: {relabel_errors[:8]}")
    if simple_local_policy_errors and not relief_policy_errors:
        report["warnings"].append(f"simple_local policy mismatches: {simple_local_policy_errors[:8]}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if report["fatal_errors"]:
        raise SystemExit("stage4b validation failed")
    if not report["checks"]["train_eval_non_empty"]:
        raise SystemExit("stage4b validation failed")
    if not report["checks"]["required_columns_present"]:
        raise SystemExit("stage4b validation failed")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
