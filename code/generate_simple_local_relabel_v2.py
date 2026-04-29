#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


NEUTRAL_LEVELS = {"正常", "畅通"}
TARGET_BUCKETS = [
    "anomaly_plus_normal_side_event",
    "relief_normal_single_edge",
    "relief_normal_short_range",
    "multi_edge_relief",
]
ANCHOR_RE = re.compile(r"^ANCHOR\|ROAD=([^|]+)\|DIR=([^|]+)\|RANGE=([^|]+)\|LEVEL=([^|]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strict-aligned simple_local relabel dataset for ceiling evaluation.")
    parser.add_argument("--input-train", default="/root/autodl-tmp/dataset_sparse_v2_stage4/train")
    parser.add_argument("--input-eval", default="/root/autodl-tmp/dataset_sparse_v2_stage4/eval")
    parser.add_argument(
        "--failure-summary-csv",
        default="/root/autodl-tmp/results/simple_local_failure_bucket_summary.csv",
    )
    parser.add_argument(
        "--failure-examples-json",
        default="/root/autodl-tmp/results/simple_local_failure_examples.json",
    )
    parser.add_argument(
        "--policy-md",
        default="/root/autodl-tmp/results/simple_local_v2_label_policy.md",
    )
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/dataset_sparse_v2_stage4b_relabel",
    )
    return parser.parse_args()


def clean_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.replace("\\n", "\n").strip()


def safe_json_loads(raw: str):
    if not isinstance(raw, str) or not raw:
        return []
    return json.loads(raw)


def event_is_changed(event: dict) -> bool:
    try:
        weight = float(event.get("weight", 2.0))
    except (TypeError, ValueError):
        weight = 2.0
    return abs(weight - 2.0) > 0.05


def event_signature(event: dict) -> str:
    return (
        f"ROAD={event.get('road', '')}|DIR={event.get('dir', '')}|"
        f"RANGE={event.get('range', '')}|LEVEL={event.get('level', '')}"
    )


def event_anchor_line(event: dict) -> str:
    return (
        f"ANCHOR|ROAD={event.get('road', '')}|DIR={event.get('dir', '')}|"
        f"RANGE={event.get('range', '')}|LEVEL={event.get('level', '')}"
    )


def event_edge_union_count(events: list[dict]) -> int:
    edge_ids = set()
    for event in events:
        for edge_id in event.get("anchor_edges", []) or []:
            if edge_id:
                edge_ids.add(str(edge_id))
    return len(edge_ids)


def parse_label_body(label: str) -> list[str]:
    lines = [line.strip() for line in clean_text(label).splitlines() if line.strip()]
    body = []
    in_think = False
    for line in lines:
        if line == "<think>":
            in_think = True
            continue
        if line == "</think>":
            in_think = False
            continue
        if in_think:
            continue
        body.append(line)
    return body


def label_anchor_lines(label: str) -> list[str]:
    body = parse_label_body(label)
    return [line for line in body if line.startswith("ANCHOR|")]


def label_is_valid(label: str) -> bool:
    body = parse_label_body(label)
    if not body:
        return False
    if body == ["NO_ANCHOR"]:
        return True
    return all(ANCHOR_RE.match(line) for line in body)


def render_label(changed_events: list[dict]) -> str:
    anchor_lines = [event_anchor_line(event) for event in changed_events]
    think_lines = [
        "<think>",
        "scene=simple_local",
        f"anchor_count={len(anchor_lines)}",
        "simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。",
        "</think>",
    ]
    if not anchor_lines:
        body = ["NO_ANCHOR"]
    else:
        body = anchor_lines
    return "\n".join([*think_lines, *body])


def update_messages(messages_raw: str, new_label: str) -> str:
    messages = safe_json_loads(messages_raw)
    if isinstance(messages, list):
        updated = False
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                message["content"] = new_label
                updated = True
        if not updated:
            messages.append({"role": "assistant", "content": new_label})
        return json.dumps(messages, ensure_ascii=False)
    return messages_raw


def load_split_dir(split_dir: Path) -> tuple[pd.DataFrame, Path]:
    parquet_path = split_dir / "data.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing input parquet: {parquet_path}")
    table = pq.read_table(parquet_path)
    df = pd.DataFrame(table.to_pydict())
    return df, parquet_path


def infer_bucket(changed_events: list[dict]) -> str:
    changed_neutral = [event for event in changed_events if str(event.get("level", "")).strip() in NEUTRAL_LEVELS]
    changed_abnormal = [event for event in changed_events if str(event.get("level", "")).strip() not in NEUTRAL_LEVELS]

    if changed_abnormal and changed_neutral:
        return "anomaly_plus_normal_side_event"
    if not changed_abnormal and changed_neutral:
        edge_count = event_edge_union_count(changed_events)
        if edge_count <= 1:
            return "relief_normal_single_edge"
        if edge_count == 2:
            return "relief_normal_short_range"
        return "multi_edge_relief"
    return "other"


def relabel_reason(bucket: str, changed: bool, target_anchor_count: int) -> str:
    if not changed:
        if target_anchor_count == 0:
            return "no active changed edges; NO_ANCHOR remains valid under strict-aligned labeling"
        return "existing label already covers all active changed-event anchors"
    if bucket == "anomaly_plus_normal_side_event":
        return "add missing normal/relief side-event anchor so all active changed events are explicitly labeled"
    if bucket == "relief_normal_single_edge":
        return "replace NO_ANCHOR with one explicit relief/normal anchor for the single changed edge"
    if bucket == "relief_normal_short_range":
        return "replace NO_ANCHOR with one relief/normal anchor covering the full 2-edge changed range"
    if bucket == "multi_edge_relief":
        return "replace NO_ANCHOR with one relief/normal anchor covering the full multi-edge changed range"
    return "rewrite label to match strict-aligned policy: output all active changed-event anchors"


def validate_policy(policy_text: str) -> None:
    required_phrases = [
        "strict-aligned labeling",
        "输出所有 active changed-event anchors",
        "anomaly_plus_normal 双事件样本是否要求同时锚定异常主事件和正常 side event：`是`",
        "multi-edge relief 是否必须覆盖完整范围：`是`",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in policy_text]
    if missing:
        raise ValueError(f"Policy markdown is missing required A-scheme phrases: {missing}")


def collect_reference_inputs(summary_csv: Path, examples_json: Path, policy_md: Path) -> dict:
    summary_df = pd.read_csv(summary_csv)
    with examples_json.open("r", encoding="utf-8") as fh:
        examples_data = json.load(fh)
    policy_text = policy_md.read_text(encoding="utf-8")
    validate_policy(policy_text)
    return {
        "failure_summary_rows": int(len(summary_df)),
        "failure_examples_meta": examples_data.get("meta", {}),
        "policy_path": str(policy_md),
        "policy_scheme": "A. strict-aligned labeling",
    }


def process_dataset_split(df: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    df = df.copy()
    df["split_index"] = range(len(df))
    df["sample_id"] = df["split"].astype(str) + ":" + df["split_index"].astype(str)

    mapping_rows: list[dict] = []
    example_rows: list[dict] = []

    for idx, row in df.iterrows():
        if row["scene_type"] != "simple_local":
            continue

        sample_id = row["sample_id"]
        old_label = clean_text(row["response_text"])
        old_anchor_lines = label_anchor_lines(old_label)
        events = safe_json_loads(row["event_records_json"])
        active_changed = [event for event in events if event.get("active", True) and event_is_changed(event)]
        bucket = infer_bucket(active_changed)
        target_anchor_lines = [event_anchor_line(event) for event in active_changed]
        target_anchor_set = set(target_anchor_lines)
        old_anchor_set = set(old_anchor_lines)

        target_label = render_label(active_changed)
        should_change = False
        if not label_is_valid(old_label):
            should_change = True
        elif target_anchor_set != old_anchor_set:
            should_change = True
        elif not target_anchor_lines and parse_label_body(old_label) != ["NO_ANCHOR"]:
            should_change = True

        new_label = target_label if should_change else old_label
        if not clean_text(new_label):
            raise ValueError(f"Generated empty new_label for {sample_id}")
        if not label_is_valid(new_label):
            raise ValueError(f"Generated invalid new_label format for {sample_id}: {new_label}")

        new_anchor_count = len(label_anchor_lines(new_label))
        df.at[idx, "response_text"] = new_label
        df.at[idx, "anchor_count"] = new_anchor_count
        df.at[idx, "messages"] = update_messages(row["messages"], new_label)
        if should_change:
            df.at[idx, "parse_confidence"] = 1.0

        reason = relabel_reason(bucket, should_change, len(target_anchor_lines))
        relabel_bucket = bucket if bucket in TARGET_BUCKETS else ("unchanged" if not should_change else "other")
        if should_change and bucket == "other":
            relabel_bucket = "other"
        elif not should_change and bucket in TARGET_BUCKETS:
            relabel_bucket = f"unchanged_{bucket}"

        added_anchor_count = max(0, new_anchor_count - len(old_anchor_lines))
        mapping_row = {
            "sample_id": sample_id,
            "dataset_split": split_name,
            "scene_type": row["scene_type"],
            "old_label": old_label,
            "new_label": new_label,
            "relabel_reason": reason,
            "relabel_bucket": relabel_bucket,
            "changed": bool(should_change),
            "old_anchor_count": len(old_anchor_lines),
            "new_anchor_count": new_anchor_count,
            "added_anchor_count": added_anchor_count,
            "input_text": clean_text(row["constraint_text"]),
        }
        mapping_rows.append(mapping_row)

        if should_change or bucket in TARGET_BUCKETS:
            example_rows.append(
                {
                    **mapping_row,
                    "bucket": bucket,
                }
            )

    df = df.drop(columns=["split_index", "sample_id"])
    return df, mapping_rows, example_rows


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    data = {column: df[column].tolist() for column in df.columns}
    table = pa.Table.from_pydict(data)
    pq.write_table(table, path)


def main() -> None:
    args = parse_args()

    input_train = Path(args.input_train)
    input_eval = Path(args.input_eval)
    failure_summary_csv = Path(args.failure_summary_csv)
    failure_examples_json = Path(args.failure_examples_json)
    policy_md = Path(args.policy_md)
    output_root = Path(args.output_root)

    reference_meta = collect_reference_inputs(failure_summary_csv, failure_examples_json, policy_md)

    train_df, train_path = load_split_dir(input_train)
    eval_df, eval_path = load_split_dir(input_eval)

    new_train_df, train_mapping, train_examples = process_dataset_split(train_df, "train")
    new_eval_df, eval_mapping, eval_examples = process_dataset_split(eval_df, "eval")

    output_train_dir = output_root / "train"
    output_eval_dir = output_root / "eval"
    output_train_dir.mkdir(parents=True, exist_ok=True)
    output_eval_dir.mkdir(parents=True, exist_ok=True)

    write_parquet(new_train_df, output_train_dir / "data.parquet")
    write_parquet(new_eval_df, output_eval_dir / "data.parquet")

    mapping_rows = train_mapping + eval_mapping
    mapping_df = pd.DataFrame(mapping_rows)
    simple_local_df = mapping_df.copy()
    changed_df = simple_local_df[simple_local_df["changed"]].copy()
    unchanged_df = simple_local_df[~simple_local_df["changed"]].copy()

    mapping_path = output_root / "relabel_mapping.csv"
    mapping_df[
        [
            "sample_id",
            "scene_type",
            "old_label",
            "new_label",
            "relabel_reason",
            "relabel_bucket",
            "dataset_split",
            "changed",
            "old_anchor_count",
            "new_anchor_count",
            "added_anchor_count",
            "input_text",
        ]
    ].to_csv(mapping_path, index=False, encoding="utf-8")

    bucket_counter_all = Counter()
    bucket_counter_changed = Counter()
    unchanged_sample_ids = []
    anomaly_single_to_double = 0
    added_relief_normal_anchors = 0

    for row in mapping_rows:
        bucket = row["relabel_bucket"]
        bucket_counter_all[bucket] += 1
        if row["changed"]:
            bucket_counter_changed[bucket] += 1
            if row["relabel_bucket"] == "anomaly_plus_normal_side_event" and row["old_anchor_count"] == 1 and row["new_anchor_count"] == 2:
                anomaly_single_to_double += 1
            old_lines = set(label_anchor_lines(row["old_label"]))
            new_lines = set(label_anchor_lines(row["new_label"]))
            for line in new_lines - old_lines:
                match = ANCHOR_RE.match(line)
                if match and match.group(4) in NEUTRAL_LEVELS:
                    added_relief_normal_anchors += 1
        else:
            unchanged_sample_ids.append(row["sample_id"])

    examples_by_bucket: dict[str, list[dict]] = defaultdict(list)
    for example in train_examples + eval_examples:
        bucket = example["bucket"] if example["bucket"] in TARGET_BUCKETS else example["relabel_bucket"]
        if len(examples_by_bucket[bucket]) >= 5:
            continue
        examples_by_bucket[bucket].append(
            {
                "sample_id": example["sample_id"],
                "dataset_split": example["dataset_split"],
                "input_text": example["input_text"],
                "old_label": example["old_label"],
                "new_label": example["new_label"],
                "relabel_reason": example["relabel_reason"],
                "changed": example["changed"],
                "old_anchor_count": example["old_anchor_count"],
                "new_anchor_count": example["new_anchor_count"],
            }
        )

    validations = {
        "new_label_non_empty": bool((mapping_df["new_label"].map(lambda text: clean_text(text) != "")).all()),
        "new_label_format_valid": bool(mapping_df["new_label"].map(label_is_valid).all()),
        "mapping_traceable_one_to_one": bool(mapping_df["sample_id"].is_unique and len(mapping_df) == len(simple_local_df)),
        "train_row_count_matches_input": len(train_df) == len(new_train_df),
        "eval_row_count_matches_input": len(eval_df) == len(new_eval_df),
        "train_schema_matches_input": list(train_df.columns) == list(new_train_df.columns),
        "eval_schema_matches_input": list(eval_df.columns) == list(new_eval_df.columns),
    }
    if not all(validations.values()):
        raise ValueError(f"Validation failed: {validations}")

    summary = {
        "source_inputs": {
            "input_train": str(train_path),
            "input_eval": str(eval_path),
            "failure_summary_csv": str(failure_summary_csv),
            "failure_examples_json": str(failure_examples_json),
            "policy_md": str(policy_md),
            **reference_meta,
        },
        "policy_applied": "A. strict-aligned labeling",
        "dataset_stats": {
            "train_rows": int(len(new_train_df)),
            "eval_rows": int(len(new_eval_df)),
            "simple_local_total": int(len(simple_local_df)),
            "simple_local_relabeled_count": int(len(changed_df)),
            "simple_local_unchanged_count": int(len(unchanged_df)),
        },
        "relabel_stats": {
            "anomaly_plus_normal_single_to_double_anchor": int(anomaly_single_to_double),
            "added_relief_normal_anchor_count": int(added_relief_normal_anchors),
            "relabel_bucket_counts_all_simple_local": dict(sorted(bucket_counter_all.items())),
            "relabel_bucket_counts_changed_only": dict(sorted(bucket_counter_changed.items())),
            "target_bucket_counts_changed_only": {bucket: int(bucket_counter_changed.get(bucket, 0)) for bucket in TARGET_BUCKETS},
            "target_bucket_counts_all_simple_local": {
                bucket: int(
                    sum(
                        1
                        for row in mapping_rows
                        if row["relabel_bucket"] == bucket or row["relabel_bucket"] == f"unchanged_{bucket}"
                    )
                )
                for bucket in TARGET_BUCKETS
            },
        },
        "unchanged_sample_ids": unchanged_sample_ids,
        "validation": validations,
        "boundary_ambiguity_note": "multi_edge_relief remains the most likely boundary-ambiguous bucket because full-range coverage depends on correctly resolving continuous edge spans.",
    }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    relabel_examples = {
        "meta": {
            "policy_applied": "A. strict-aligned labeling",
            "simple_local_relabeled_count": int(len(changed_df)),
            "target_buckets": TARGET_BUCKETS,
        },
        "examples_by_bucket": {bucket: examples_by_bucket.get(bucket, []) for bucket in TARGET_BUCKETS + ["other", "unchanged"]},
    }
    relabel_examples_path = output_root / "relabel_examples.json"
    relabel_examples_path.write_text(json.dumps(relabel_examples, ensure_ascii=False, indent=2), encoding="utf-8")

    present_target_gap = sum(summary["relabel_stats"]["target_bucket_counts_all_simple_local"][bucket] for bucket in TARGET_BUCKETS)
    changed_target_gap = sum(summary["relabel_stats"]["target_bucket_counts_changed_only"][bucket] for bucket in TARGET_BUCKETS)

    print(
        f"stage4 simple_local relabeled={len(changed_df)}/{len(simple_local_df)}; "
        f"present target-gap samples covered={changed_target_gap}/{present_target_gap}"
    )
    print(
        "remaining boundary ambiguity risk: "
        "multi_edge_relief (full-range coverage and edge-span boundaries are still easiest to mis-specify)"
    )
    print(f"output_root={output_root}")


if __name__ == "__main__":
    main()
