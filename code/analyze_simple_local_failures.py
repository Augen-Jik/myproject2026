#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


NEUTRAL_LEVELS = {"正常", "畅通"}
DIMENSION_BUCKETS = {
    "failure_type": ["no_anchor", "parse_fail", "target_miss", "changed_edge_miss", "mixed"],
    "changed_edge_count_bucket": ["1", "2", "3+"],
    "input_length_bucket": ["short", "medium", "long"],
    "anomaly_strength_bucket": ["mild", "moderate", "severe"],
    "pattern_bucket": [
        "single_anomaly_only",
        "anomaly_plus_normal",
        "same_road_multi_dir",
        "short_range_local",
        "other",
    ],
}


@dataclass(frozen=True)
class Paths:
    details_csv: Path
    dataset_root: Path
    summary_csv: Path
    examples_json: Path


def parse_args() -> Paths:
    parser = argparse.ArgumentParser(description="Analyze simple_local failures from strict stage4_fix details.")
    parser.add_argument(
        "--details-csv",
        default="/root/autodl-tmp/results/sparse_v2_validation_strict_stage4_fix_details.csv",
    )
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/dataset_sparse_v2")
    parser.add_argument(
        "--summary-csv",
        default="/root/autodl-tmp/results/simple_local_failure_bucket_summary.csv",
    )
    parser.add_argument(
        "--examples-json",
        default="/root/autodl-tmp/results/simple_local_failure_examples.json",
    )
    args = parser.parse_args()
    return Paths(
        details_csv=Path(args.details_csv),
        dataset_root=Path(args.dataset_root),
        summary_csv=Path(args.summary_csv),
        examples_json=Path(args.examples_json),
    )


def normalize_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def safe_json_loads(raw: str):
    if not isinstance(raw, str) or not raw:
        return []
    return json.loads(raw)


def clean_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.replace("\\n", "\n").strip()


def load_details(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["scene_type"].astype(str) == "simple_local"].copy()
    for col in ["no_anchor", "parse_fail", "strict_parse_fail"]:
        df[col] = df[col].map(normalize_bool)
    numeric_cols = [
        "gt_changed_edge_count",
        "gt_target_edge_count",
        "target_recall",
        "gt_changed_edge_recall",
        "anchor_count",
        "mapped_edge_count",
        "parse_confidence",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parquet_to_frame(path: Path) -> pd.DataFrame:
    table = pq.read_table(
        path,
        columns=[
            "scene_type",
            "length_bucket",
            "split",
            "constraint_text",
            "prompt_text",
            "response_text",
            "ground_truth_json",
            "event_records_json",
        ],
    )
    data = table.to_pydict()
    frame = pd.DataFrame(data)
    frame["split_index"] = range(len(frame))
    frame["sample_id"] = frame["split"].astype(str) + ":" + frame["split_index"].astype(str)
    return frame


def load_dataset(dataset_root: Path, splits: Iterable[str]) -> pd.DataFrame:
    frames = []
    for split in sorted(set(splits)):
        split_path = dataset_root / split / "data.parquet"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing dataset split parquet: {split_path}")
        frames.append(parquet_to_frame(split_path))
    return pd.concat(frames, ignore_index=True)


def level_strength(level: str, weight: float | None) -> int:
    text = (level or "").strip()
    if any(token in text for token in ["封闭", "严重"]):
        return 3
    if "中度" in text:
        return 2
    if text in NEUTRAL_LEVELS:
        return 1
    if weight is None:
        return 1
    if weight >= 7.0:
        return 3
    if weight >= 4.5:
        return 2
    return 1


def event_is_changed(event: dict) -> bool:
    weight = event.get("weight", 2.0)
    try:
        weight = float(weight)
    except (TypeError, ValueError):
        weight = 2.0
    return abs(weight - 2.0) > 0.05


def derive_anomaly_strength(events: list[dict]) -> str:
    changed = [event for event in events if event.get("active", True) and event_is_changed(event)]
    if not changed:
        return "mild"
    max_rank = max(level_strength(event.get("level", ""), event.get("weight")) for event in changed)
    return {1: "mild", 2: "moderate", 3: "severe"}[max_rank]


def derive_pattern_bucket(events: list[dict], gt_changed_edge_count: int) -> str:
    active_changed = [event for event in events if event.get("active", True) and event_is_changed(event)]
    if not active_changed:
        return "other"

    road_dirs: dict[str, set[str]] = defaultdict(set)
    for event in active_changed:
        road_dirs[str(event.get("road", ""))].add(str(event.get("dir", "")))
    if any(len(dirs) >= 2 for dirs in road_dirs.values()):
        return "same_road_multi_dir"

    abnormal_events = [event for event in active_changed if str(event.get("level", "")).strip() not in NEUTRAL_LEVELS]
    normal_events = [event for event in active_changed if str(event.get("level", "")).strip() in NEUTRAL_LEVELS]
    if abnormal_events and normal_events:
        return "anomaly_plus_normal"

    local_keywords = ("局部", "路口", "短时", "附近", "临时")
    text_blob = " ".join(str(event.get("text", "")) for event in active_changed)
    if gt_changed_edge_count <= 1 or any(keyword in text_blob for keyword in local_keywords):
        return "short_range_local"

    if len(active_changed) == 1:
        return "single_anomaly_only"

    return "other"


def derive_input_length_bucket(length_bucket: str, constraint_text: str) -> str:
    bucket = str(length_bucket or "").strip()
    if bucket in {"short", "medium", "long"}:
        return bucket
    if bucket == "noisy_long":
        return "long"

    length = len(constraint_text or "")
    if length <= 35:
        return "short"
    if length <= 70:
        return "medium"
    return "long"


def derive_changed_edge_count_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def compute_failure_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["no_anchor_when_gt_changed"] = df["no_anchor"] & (df["gt_changed_edge_count"] > 0)
    df["parse_fail_strict"] = df["strict_parse_fail"]
    df["target_miss_raw"] = (df["gt_target_edge_count"] > 0) & (df["target_recall"] < 100.0)
    df["changed_edge_miss_raw"] = (df["gt_changed_edge_count"] > 0) & (df["gt_changed_edge_recall"] < 100.0)
    df["parse_fail_root"] = df["parse_fail"] | (df["parse_fail_strict"] & ~df["no_anchor_when_gt_changed"])
    df["is_failure"] = (
        df["no_anchor_when_gt_changed"]
        | df["parse_fail_strict"]
        | df["target_miss_raw"]
        | df["changed_edge_miss_raw"]
    )

    def classify_failure(row: pd.Series) -> str:
        if not row["is_failure"]:
            return "ok"
        if row["no_anchor_when_gt_changed"]:
            return "no_anchor"
        if row["parse_fail_root"]:
            return "parse_fail"
        if row["target_miss_raw"] and row["changed_edge_miss_raw"]:
            return "mixed"
        if row["target_miss_raw"]:
            return "target_miss"
        if row["changed_edge_miss_raw"]:
            return "changed_edge_miss"
        return "mixed"

    df["failure_type"] = df.apply(classify_failure, axis=1)
    return df


def enrich_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    event_records = df["event_records_json"].map(safe_json_loads)
    df["events"] = event_records
    df["changed_edge_count_bucket"] = df["gt_changed_edge_count"].fillna(0).astype(int).map(derive_changed_edge_count_bucket)
    df["input_length_bucket"] = [
        derive_input_length_bucket(length_bucket, text)
        for length_bucket, text in zip(df["length_bucket"], df["constraint_text"], strict=False)
    ]
    df["anomaly_strength_bucket"] = event_records.map(derive_anomaly_strength)
    df["pattern_bucket"] = [
        derive_pattern_bucket(events, int(count or 0))
        for events, count in zip(event_records, df["gt_changed_edge_count"], strict=False)
    ]
    df["input_char_len"] = df["constraint_text"].fillna("").map(len)
    return df


def summarize_dimension(
    all_simple_df: pd.DataFrame,
    fail_df: pd.DataFrame,
    dimension: str,
    bucket_order: list[str],
) -> list[dict]:
    all_bucket_counts = all_simple_df[dimension].value_counts().to_dict()
    fail_bucket_counts = fail_df[dimension].value_counts().to_dict()
    rows = []
    for rank, bucket in enumerate(bucket_order, start=1):
        bucket_total = int(all_bucket_counts.get(bucket, 0))
        bucket_fail = int(fail_bucket_counts.get(bucket, 0))
        row = {
            "dimension": dimension,
            "bucket": bucket,
            "bucket_rank": rank,
            "bucket_total_simple_local": bucket_total if dimension != "failure_type" else bucket_fail,
            "bucket_failed_simple_local": bucket_fail,
            "failure_rate_within_bucket": round(bucket_fail / bucket_total, 4) if bucket_total and dimension != "failure_type" else (1.0 if bucket_fail and dimension == "failure_type" else 0.0),
            "share_of_failed_simple_local": round(bucket_fail / len(fail_df), 4) if len(fail_df) else 0.0,
        }
        slice_df = fail_df[fail_df[dimension] == bucket]
        row["pred_matches_gt_label_count"] = int(slice_df["pred_matches_gt_label"].sum()) if not slice_df.empty else 0
        for failure_type in DIMENSION_BUCKETS["failure_type"]:
            row[f"{failure_type}_count"] = int((slice_df["failure_type"] == failure_type).sum())
        rows.append(row)
    return rows


def score_for_examples(row: pd.Series) -> tuple:
    failure_priority = {
        "no_anchor": 0,
        "parse_fail": 1,
        "mixed": 2,
        "target_miss": 3,
        "changed_edge_miss": 4,
    }.get(row["failure_type"], 9)
    strength_priority = {"severe": 0, "moderate": 1, "mild": 2}.get(row["anomaly_strength_bucket"], 9)
    return (
        failure_priority,
        strength_priority,
        -int(row["gt_changed_edge_count"]),
        -int(row["gt_target_edge_count"]),
        -int(row["input_char_len"]),
        str(row["sample_id"]),
    )


def pick_examples(bucket_df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    if len(bucket_df) <= limit:
        return bucket_df.sort_values("sample_id") if len(bucket_df) else bucket_df

    ordered = bucket_df.copy()
    ordered["_score"] = ordered.apply(score_for_examples, axis=1)
    ordered = ordered.sort_values("_score")

    selected_indices = []
    for _, group in ordered.groupby("failure_type", sort=False):
        selected_indices.append(group.index[0])
        if len(selected_indices) >= limit:
            break

    for idx in ordered.index:
        if idx not in selected_indices:
            selected_indices.append(idx)
        if len(selected_indices) >= limit:
            break

    return ordered.loc[selected_indices].drop(columns="_score")


def example_record(row: pd.Series) -> dict:
    parse_result = {
        "no_anchor": bool(row["no_anchor"]),
        "parse_fail": bool(row["parse_fail"]),
        "strict_parse_fail": bool(row["strict_parse_fail"]),
        "strict_error_reason": row["strict_failure_reason"],
        "anchor_count": int(row["anchor_count"]) if pd.notna(row["anchor_count"]) else None,
        "mapped_edge_count": int(row["mapped_edge_count"]) if pd.notna(row["mapped_edge_count"]) else None,
        "parse_confidence": round(float(row["parse_confidence"]), 4) if pd.notna(row["parse_confidence"]) else None,
    }
    failure_flags = []
    if row["no_anchor_when_gt_changed"]:
        failure_flags.append("no_anchor_when_gt_changed")
    if row["parse_fail_strict"]:
        failure_flags.append("parse_fail_strict")
    if row["target_miss_raw"]:
        failure_flags.append("target_miss")
    if row["changed_edge_miss_raw"]:
        failure_flags.append("changed_edge_miss")
    return {
        "sample_id": row["sample_id"],
        "failure_type": row["failure_type"],
        "pred_matches_gt_label": bool(row["pred_matches_gt_label"]),
        "input_text": clean_text(row["constraint_text"]),
        "gt_label": clean_text(row["response_text"]),
        "pred_text": clean_text(row["prediction_raw_preview"]),
        "parse_result": parse_result,
        "target_recall": float(row["target_recall"]) if pd.notna(row["target_recall"]) else None,
        "gt_changed_edge_recall": float(row["gt_changed_edge_recall"]) if pd.notna(row["gt_changed_edge_recall"]) else None,
        "gt_changed_edge_count": int(row["gt_changed_edge_count"]) if pd.notna(row["gt_changed_edge_count"]) else None,
        "gt_target_edge_count": int(row["gt_target_edge_count"]) if pd.notna(row["gt_target_edge_count"]) else None,
        "changed_edge_count_bucket": row["changed_edge_count_bucket"],
        "input_length_bucket": row["input_length_bucket"],
        "anomaly_strength_bucket": row["anomaly_strength_bucket"],
        "pattern_bucket": row["pattern_bucket"],
        "failure_flags": failure_flags,
    }


def build_examples_payload(fail_df: pd.DataFrame, paths: Paths) -> dict:
    payload = {
        "meta": {
            "source_details_csv": str(paths.details_csv),
            "dataset_root": str(paths.dataset_root),
            "simple_local_failure_count": int(len(fail_df)),
            "pred_matches_gt_label_failure_count": int(fail_df["pred_matches_gt_label"].sum()),
            "selection_policy": "Per dimension bucket, keep up to 5 representative failures; if a bucket has fewer than 5 samples, keep all available samples.",
        },
        "dimension_buckets": {},
    }

    for dimension, bucket_order in DIMENSION_BUCKETS.items():
        payload["dimension_buckets"][dimension] = {}
        for bucket in bucket_order:
            bucket_df = fail_df[fail_df[dimension] == bucket].copy()
            chosen = pick_examples(bucket_df, limit=5)
            payload["dimension_buckets"][dimension][bucket] = {
                "bucket_count": int(len(bucket_df)),
                "selected_example_count": int(len(chosen)),
                "examples": [example_record(row) for _, row in chosen.iterrows()],
            }
    return payload


def top_augmentation_patterns(fail_df: pd.DataFrame) -> list[tuple[tuple, int]]:
    combo_cols = ["failure_type", "pattern_bucket", "anomaly_strength_bucket", "changed_edge_count_bucket"]
    combo_counts = fail_df.groupby(combo_cols).size().sort_values(ascending=False)
    return list(combo_counts.head(3).items())


def main() -> None:
    paths = parse_args()
    details_df = load_details(paths.details_csv)
    dataset_df = load_dataset(paths.dataset_root, details_df["split"].unique())
    merged = details_df.merge(
        dataset_df[
            [
                "sample_id",
                "prompt_text",
                "response_text",
                "ground_truth_json",
                "event_records_json",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if merged[["constraint_text", "response_text", "event_records_json"]].isna().any().any():
        missing = merged[merged["constraint_text"].isna()]["sample_id"].tolist()
        raise ValueError(f"Failed to join dataset rows for sample_ids: {missing[:5]}")

    merged = compute_failure_flags(merged)
    merged = enrich_rows(merged)
    merged["pred_matches_gt_label"] = (
        merged["prediction_raw_preview"].map(clean_text) == merged["response_text"].map(clean_text)
    )
    fail_df = merged[merged["is_failure"]].copy()

    summary_rows = []
    summary_rows.extend(
        [
            {
                "dimension": "overall",
                "bucket": "all_simple_local",
                "bucket_rank": 1,
                "bucket_total_simple_local": int(len(merged)),
                "bucket_failed_simple_local": int(len(fail_df)),
                "failure_rate_within_bucket": round(len(fail_df) / len(merged), 4) if len(merged) else 0.0,
                "share_of_failed_simple_local": 1.0,
                "pred_matches_gt_label_count": int(fail_df["pred_matches_gt_label"].sum()),
                **{f"{ft}_count": int((fail_df["failure_type"] == ft).sum()) for ft in DIMENSION_BUCKETS["failure_type"]},
            }
        ]
    )
    for dimension, bucket_order in DIMENSION_BUCKETS.items():
        source_df = fail_df if dimension == "failure_type" else merged
        summary_rows.extend(summarize_dimension(source_df, fail_df, dimension, bucket_order))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(paths.summary_csv, index=False, encoding="utf-8")

    examples_payload = build_examples_payload(fail_df, paths)
    with paths.examples_json.open("w", encoding="utf-8") as fh:
        json.dump(examples_payload, fh, ensure_ascii=False, indent=2)

    failure_counts = fail_df["failure_type"].value_counts()
    top_failure = failure_counts.index[0] if not failure_counts.empty else "none"
    top_failure_count = int(failure_counts.iloc[0]) if not failure_counts.empty else 0
    top_failure_share = (top_failure_count / len(fail_df)) if len(fail_df) else 0.0
    top_patterns = top_augmentation_patterns(fail_df)
    matched_gt_count = int(fail_df["pred_matches_gt_label"].sum())

    print(f"simple_local total={len(merged)}, failures={len(fail_df)}, failure_rate={len(fail_df) / len(merged):.2%}")
    print(f"strict失败中 pred==gt_label: {matched_gt_count}/{len(fail_df)} ({matched_gt_count / len(fail_df):.1%})")
    print(
        "当前 simple_local 主失败模式："
        f"{top_failure} ({top_failure_count}/{len(fail_df)}, {top_failure_share:.1%})"
    )
    print("下一轮数据增强优先覆盖的 3 类模式：")
    for rank, (combo, count) in enumerate(top_patterns, start=1):
        failure_type, pattern_bucket, anomaly_strength_bucket, changed_edge_count_bucket = combo
        print(
            f"{rank}. {failure_type} | {pattern_bucket} | {anomaly_strength_bucket} | "
            f"changed_edges={changed_edge_count_bucket} ({count} samples)"
        )
    print(f"summary_csv={paths.summary_csv}")
    print(f"examples_json={paths.examples_json}")


if __name__ == "__main__":
    main()
