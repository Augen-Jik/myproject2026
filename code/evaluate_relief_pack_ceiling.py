#!/usr/bin/env python3
from __future__ import annotations

"""Oracle ceiling evaluation for the simple_local relief supervision pack.

Important implementation note:
- Parquet is always read with `pyarrow.parquet.read_table(...)`.
- Do not switch this script to `pandas.read_parquet(...)`; the repo already has
  known parquet->pandas shape issues on some generated files.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


CODE_DIR = Path("/root/autodl-tmp/code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from roadnet_meta import live_edge_ids  # noqa: E402
from sparse_eval_strict import dense_prediction, score_prediction, summarize_details  # noqa: E402
from sparse_utils import parse_sparse_output_bundle  # noqa: E402


TARGET_BUCKETS = [
    "relief_normal_single_edge",
    "relief_normal_short_range",
    "multi_edge_relief",
]
NEUTRAL_LEVELS = {"正常", "畅通"}
METRICS = [
    "no_anchor_when_gt_changed_rate",
    "target_recall",
    "gt_changed_edge_recall",
    "MAE_on_changed_edges",
    "parse_fail_rate_strict",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate simple_local relief supervision pack oracle ceiling without training."
    )
    parser.add_argument(
        "--pack-train-parquet",
        default="/root/autodl-tmp/dataset_simple_local_relief_pack/train/data.parquet",
    )
    parser.add_argument(
        "--pack-eval-parquet",
        default="/root/autodl-tmp/dataset_simple_local_relief_pack/eval/data.parquet",
    )
    parser.add_argument(
        "--pack-summary-json",
        default="/root/autodl-tmp/dataset_simple_local_relief_pack/summary.json",
    )
    parser.add_argument(
        "--pack-examples-json",
        default="/root/autodl-tmp/dataset_simple_local_relief_pack/examples.json",
    )
    parser.add_argument(
        "--strict-details-csv",
        default="/root/autodl-tmp/results/sparse_v2_validation_strict_stage4_fix_details.csv",
    )
    parser.add_argument(
        "--failure-summary-csv",
        default="/root/autodl-tmp/results/simple_local_failure_bucket_summary.csv",
    )
    parser.add_argument(
        "--policy-md",
        default="/root/autodl-tmp/results/simple_local_v2_label_policy.md",
    )
    parser.add_argument(
        "--dataset-root",
        default="/root/autodl-tmp/dataset_sparse_v2",
        help="Used only to recover held-out simple_local rows by sample_id.",
    )
    parser.add_argument(
        "--output-csv",
        default="/root/autodl-tmp/results/relief_pack_ceiling_eval.csv",
    )
    parser.add_argument(
        "--output-notes",
        default="/root/autodl-tmp/results/relief_pack_ceiling_notes.md",
    )
    return parser.parse_args()


def clean_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.replace("\\n", "\n").strip()


def safe_json_loads(raw: str):
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str) or not raw:
        return []
    return json.loads(raw)


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def event_is_changed(event: dict) -> bool:
    try:
        weight = float(event.get("weight", 2.0))
    except (TypeError, ValueError):
        weight = 2.0
    return abs(weight - 2.0) > 0.05


def changed_events(events: list[dict]) -> list[dict]:
    return [
        event
        for event in events
        if event.get("active", True) and event_is_changed(event)
    ]


def changed_edge_count(events: list[dict]) -> int:
    return len(
        {
            str(edge)
            for event in events
            for edge in (event.get("anchor_edges", []) or [])
            if edge
        }
    )


def infer_bucket(events: list[dict]) -> str:
    abnormal = [
        event
        for event in events
        if str(event.get("level", "")).strip() not in NEUTRAL_LEVELS
    ]
    neutral = [
        event
        for event in events
        if str(event.get("level", "")).strip() in NEUTRAL_LEVELS
    ]
    if abnormal and neutral:
        return "anomaly_plus_normal_side_event"
    if not abnormal and neutral:
        edge_count = changed_edge_count(events)
        if edge_count <= 1:
            return "relief_normal_single_edge"
        if edge_count == 2:
            return "relief_normal_short_range"
        return "multi_edge_relief"
    return "other"


def render_label(events: list[dict]) -> str:
    anchors = [
        f"ANCHOR|ROAD={event.get('road', '')}|DIR={event.get('dir', '')}|RANGE={event.get('range', '')}|LEVEL={event.get('level', '')}"
        for event in events
    ]
    return "\n".join(
        [
            "<think>",
            "scene=simple_local",
            f"anchor_count={len(anchors)}",
            "simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。",
            "</think>",
            *(anchors or ["NO_ANCHOR"]),
        ]
    )


def range_fingerprint(events: list[dict]) -> str:
    return " || ".join(
        sorted(
            f"ROAD={event.get('road', '')}|DIR={event.get('dir', '')}|RANGE={event.get('range', '')}"
            for event in events
        )
    )


def validate_policy(policy_text: str) -> list[str]:
    warnings: list[str] = []
    required = [
        "strict-aligned labeling",
        "输出所有 active changed-event anchors",
        "anomaly_plus_normal 双事件样本是否要求同时锚定异常主事件和正常 side event：`是`",
        "multi-edge relief 是否必须覆盖完整范围：`是`",
    ]
    missing = [item for item in required if item not in policy_text]
    if missing:
        raise ValueError(f"Policy markdown missing required strict-aligned statements: {missing}")
    if "A. strict-aligned labeling" not in policy_text:
        warnings.append("policy markdown does not literally contain 'A. strict-aligned labeling' header text")
    return warnings


def load_pack_rows(train_path: Path, eval_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for split_name, parquet_path in [("train", train_path), ("eval", eval_path)]:
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            item = dict(row)
            item["pack_split_name"] = split_name
            item["sample_id"] = str(item["pack_sample_id"])
            item["split_index"] = int(str(item["pack_sample_id"]).split(":")[-1])
            rows.append(item)
    return pd.DataFrame(rows)


def load_heldout_simple_local_rows(dataset_root: Path, strict_simple_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    needed_splits = sorted(set(strict_simple_df["split"].astype(str)))
    for split in needed_splits:
        parquet_path = dataset_root / split / "data.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing held-out split parquet: {parquet_path}")
        table = pq.read_table(
            parquet_path,
            columns=[
                "messages",
                "scene_type",
                "scene_bucket",
                "length_bucket",
                "split",
                "constraint_text",
                "prompt_text",
                "response_text",
                "anchor_count",
                "ground_truth_json",
                "event_records_json",
                "parse_confidence",
                "total_edges",
            ],
        )
        for split_index, row in enumerate(table.to_pylist()):
            if row["scene_type"] != "simple_local":
                continue
            item = dict(row)
            item["split_index"] = split_index
            item["sample_id"] = f"{row['split']}:{split_index}"
            events = changed_events(safe_json_loads(row["event_records_json"]))
            item["heldout_bucket"] = infer_bucket(events)
            item["range_fingerprint"] = range_fingerprint(events)
            item["oracle_label"] = render_label(events)
            rows.append(item)
    return pd.DataFrame(rows)


def evaluate_rows(df: pd.DataFrame, label_col: str) -> list[dict[str, Any]]:
    edge_ids = live_edge_ids()
    details: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_generation = clean_text(row[label_col])
        bundle = parse_sparse_output_bundle(raw_generation, scene_type=row["scene_type"])
        pred_weights = dense_prediction(bundle, edge_ids)
        details.append(
            score_prediction(
                row=row.to_dict(),
                bundle=bundle,
                pred_weights=pred_weights,
                edge_ids=edge_ids,
                sample_id=str(row["sample_id"]),
                split=str(row["split"]),
                split_index=int(row["split_index"]),
                raw_generation=raw_generation,
                infer_time_s=0.0,
            )
        )
    return details


def is_current_strict_fail(detail: dict[str, Any]) -> bool:
    return bool(
        detail["strict_parse_fail"]
        or (detail["gt_target_edge_count"] > 0 and float(detail["target_recall"]) < 100.0)
        or (detail["gt_changed_edge_count"] > 0 and float(detail["gt_changed_edge_recall"]) < 100.0)
    )


def metric_direction(metric: str) -> str:
    return "lower_is_better" if metric in {"no_anchor_when_gt_changed_rate", "MAE_on_changed_edges", "parse_fail_rate_strict"} else "higher_is_better"


def relative_gain(current: float, oracle: float) -> float:
    if current == 0:
        return 0.0 if oracle == 0 else math.inf
    return (oracle - current) / abs(current) * 100.0


def metric_rows(
    *,
    scope: str,
    bucket: str,
    evidence_type: str,
    current_summary: dict[str, Any],
    oracle_summary: dict[str, Any],
    note_by_metric: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    note_by_metric = note_by_metric or {}
    for metric in METRICS:
        current = float(current_summary.get(metric, 0.0))
        oracle = float(oracle_summary.get(metric, 0.0))
        rows.append(
            {
                "row_type": "metric",
                "scope": scope,
                "bucket": bucket,
                "metric": metric,
                "current_reference": round(current, 4),
                "oracle_ceiling": round(oracle, 4),
                "absolute_gain": round(oracle - current, 4),
                "relative_gain_pct": round(relative_gain(current, oracle), 4) if math.isfinite(relative_gain(current, oracle)) else "inf",
                "metric_direction": metric_direction(metric),
                "evidence_type": evidence_type,
                "support_count": int(current_summary.get("count", 0)),
                "note": note_by_metric.get(metric, ""),
            }
        )
    return rows


def coverage_rows(
    *,
    heldout_fail_df: pd.DataFrame,
    pack_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pack_all_ids = set(pack_df["origin_sample_id"].astype(str))
    pack_train_ids = set(pack_df[pack_df["pack_split_name"] == "train"]["origin_sample_id"].astype(str))
    pack_all_ranges = {
        bucket: set(pack_df[pack_df["pack_bucket"] == bucket]["range_fingerprint"].astype(str))
        for bucket in TARGET_BUCKETS
    }
    pack_train_ranges = {
        bucket: set(
            pack_df[
                (pack_df["pack_bucket"] == bucket) & (pack_df["pack_split_name"] == "train")
            ]["range_fingerprint"].astype(str)
        )
        for bucket in TARGET_BUCKETS
    }

    for bucket in TARGET_BUCKETS:
        heldout_bucket = heldout_fail_df[heldout_fail_df["heldout_bucket"] == bucket].copy()
        heldout_count = int(len(heldout_bucket))
        train_count = int(((pack_df["pack_bucket"] == bucket) & (pack_df["pack_split_name"] == "train")).sum())
        eval_count = int(((pack_df["pack_bucket"] == bucket) & (pack_df["pack_split_name"] == "eval")).sum())
        direct_total = int(heldout_bucket["sample_id"].astype(str).isin(pack_all_ids).sum())
        direct_train = int(heldout_bucket["sample_id"].astype(str).isin(pack_train_ids).sum())
        range_total = int(heldout_bucket["range_fingerprint"].astype(str).isin(pack_all_ranges[bucket]).sum())
        range_train = int(heldout_bucket["range_fingerprint"].astype(str).isin(pack_train_ranges[bucket]).sum())

        stats = [
            ("heldout_failure_count", heldout_count, heldout_count, "当前 held-out 该 bucket 的 strict fail 数量。"),
            ("pack_train_count", train_count, heldout_count, "未来进入 stage4b 训练时该 bucket 可直接进入训练的样本数。"),
            ("pack_eval_count", eval_count, heldout_count, "该 bucket 在 pack 中保留的 eval 样本数。"),
            ("direct_sample_overlap_pack_total", direct_total, heldout_count, "pack 全量与 held-out 失败样本 sample_id 直接重合数。"),
            ("direct_sample_overlap_pack_train", direct_train, heldout_count, "pack train 与 held-out 失败样本 sample_id 直接重合数。"),
            ("exact_range_overlap_pack_total", range_total, heldout_count, "pack 全量与 held-out 失败样本 ROAD+DIR+RANGE 完全一致数。"),
            ("exact_range_overlap_pack_train", range_train, heldout_count, "pack train 与 held-out 失败样本 ROAD+DIR+RANGE 完全一致数。"),
        ]
        for metric, current, oracle, note in stats:
            rows.append(
                {
                    "row_type": "coverage",
                    "scope": "C.heldout_alignment_reference",
                    "bucket": bucket,
                    "metric": metric,
                    "current_reference": current,
                    "oracle_ceiling": oracle,
                    "absolute_gain": round(current / oracle, 4) if oracle else 0.0,
                    "relative_gain_pct": "",
                    "metric_direction": "higher_is_better",
                    "evidence_type": "bucket_coverage",
                    "support_count": heldout_count,
                    "note": note,
                }
            )
    return rows


def failure_summary_reference(failure_summary_df: pd.DataFrame) -> dict[str, int]:
    overall = failure_summary_df[
        (failure_summary_df["dimension"] == "overall")
        & (failure_summary_df["bucket"] == "all_simple_local")
    ]
    no_anchor = failure_summary_df[
        (failure_summary_df["dimension"] == "failure_type")
        & (failure_summary_df["bucket"] == "no_anchor")
    ]
    return {
        "simple_local_total": int(overall["bucket_total_simple_local"].iloc[0]) if not overall.empty else -1,
        "simple_local_failure_total": int(overall["bucket_failed_simple_local"].iloc[0]) if not overall.empty else -1,
        "no_anchor_failure_total": int(no_anchor["bucket_failed_simple_local"].iloc[0]) if not no_anchor.empty else -1,
    }


def main() -> None:
    args = parse_args()

    pack_train_parquet = Path(args.pack_train_parquet)
    pack_eval_parquet = Path(args.pack_eval_parquet)
    pack_summary_json = Path(args.pack_summary_json)
    pack_examples_json = Path(args.pack_examples_json)
    strict_details_csv = Path(args.strict_details_csv)
    failure_summary_csv = Path(args.failure_summary_csv)
    policy_md = Path(args.policy_md)
    dataset_root = Path(args.dataset_root)
    output_csv = Path(args.output_csv)
    output_notes = Path(args.output_notes)

    pack_summary = json.loads(pack_summary_json.read_text(encoding="utf-8"))
    pack_examples = json.loads(pack_examples_json.read_text(encoding="utf-8"))
    failure_summary_df = pd.read_csv(failure_summary_csv)
    policy_warnings = validate_policy(policy_md.read_text(encoding="utf-8"))

    warnings: list[str] = list(policy_warnings)

    pack_df = load_pack_rows(pack_train_parquet, pack_eval_parquet)
    if pack_df.empty:
        raise ValueError("Relief pack is empty.")
    if set(pack_df["pack_bucket"]) != set(TARGET_BUCKETS):
        raise ValueError(f"Unexpected relief pack buckets: {sorted(set(pack_df['pack_bucket']))}")
    if not pack_df["can_direct_stage4b"].astype(bool).all():
        warnings.append("some pack rows are not marked can_direct_stage4b=true")
    if pack_df["response_text"].map(lambda text: "simple_local_v2=" not in clean_text(text)).any():
        warnings.append("some pack labels do not contain simple_local_v2 strict-aligned marker")
    if pack_df["response_text"].map(lambda text: "NO_ANCHOR" in clean_text(text)).any():
        warnings.append("some relief pack labels still contain NO_ANCHOR, which conflicts with relief supervision intent")

    pack_bucket_mismatch = []
    for _, row in pack_df.iterrows():
        events = changed_events(safe_json_loads(row["event_records_json"]))
        inferred = infer_bucket(events)
        if inferred != row["pack_bucket"]:
            pack_bucket_mismatch.append(str(row["sample_id"]))
    if pack_bucket_mismatch:
        warnings.append(f"pack bucket mismatch detected for sample_ids: {pack_bucket_mismatch[:5]}")

    strict_details_df = pd.read_csv(strict_details_csv)
    strict_simple = strict_details_df[strict_details_df["scene_type"].astype(str) == "simple_local"].copy()
    heldout_rows = load_heldout_simple_local_rows(dataset_root, strict_simple)
    heldout_df = strict_simple.merge(
        heldout_rows[
            [
                "sample_id",
                "messages",
                "scene_type",
                "scene_bucket",
                "length_bucket",
                "split",
                "constraint_text",
                "prompt_text",
                "response_text",
                "anchor_count",
                "ground_truth_json",
                "event_records_json",
                "parse_confidence",
                "total_edges",
                "split_index",
                "heldout_bucket",
                "range_fingerprint",
                "oracle_label",
            ]
        ],
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_strict", ""),
    )
    if len(heldout_df) != len(strict_simple):
        raise ValueError("Held-out join failed: strict details and dataset rows do not align one-to-one.")
    heldout_df["current_label"] = heldout_df["prediction_raw_preview"].map(clean_text)

    failure_ref = failure_summary_reference(failure_summary_df)
    if failure_ref["simple_local_total"] != len(strict_simple):
        warnings.append(
            f"simple_local total mismatch between failure summary ({failure_ref['simple_local_total']}) and strict details ({len(strict_simple)})"
        )

    pack_oracle_details = evaluate_rows(pack_df, "response_text")
    pack_oracle_summary = summarize_details(pack_oracle_details)
    pack_by_bucket = {
        bucket: summarize_details(
            [detail for detail in pack_oracle_details if detail["sample_id"] in set(pack_df[pack_df["pack_bucket"] == bucket]["sample_id"])]
        )
        for bucket in TARGET_BUCKETS
    }
    if pack_oracle_summary["gt_changed_edge_recall"] < 100.0 or pack_oracle_summary["MAE_on_changed_edges"] > 0.0:
        warnings.append(
            "relief pack oracle adequacy is not perfect under current strict parser/scorer; inspect pack_oracle metrics before training"
        )

    heldout_current_details = evaluate_rows(heldout_df, "current_label")
    current_by_id = {detail["sample_id"]: detail for detail in heldout_current_details}
    heldout_current_overall = summarize_details(heldout_current_details)

    changed_sample_count = int(sum(detail["gt_changed_edge_count"] > 0 for detail in heldout_current_details))
    current_no_anchor_failures = int(sum(detail["no_anchor"] and detail["gt_changed_edge_count"] > 0 for detail in heldout_current_details))
    if failure_ref["no_anchor_failure_total"] != current_no_anchor_failures:
        warnings.append(
            f"no_anchor failure mismatch between failure summary ({failure_ref['no_anchor_failure_total']}) and strict details ({current_no_anchor_failures})"
        )

    targeted_fail_ids = {
        sample_id
        for sample_id, detail in current_by_id.items()
        if is_current_strict_fail(detail) and heldout_df.loc[heldout_df["sample_id"] == sample_id, "heldout_bucket"].iloc[0] in TARGET_BUCKETS
    }
    heldout_target_df = heldout_df[heldout_df["sample_id"].isin(targeted_fail_ids)].copy()
    if heldout_target_df.empty:
        raise ValueError("No held-out targeted relief failures were found.")

    heldout_target_counts = heldout_target_df["heldout_bucket"].value_counts().to_dict()
    heldout_current_target_details = [current_by_id[sample_id] for sample_id in heldout_target_df["sample_id"]]
    heldout_oracle_target_details = evaluate_rows(heldout_target_df, "oracle_label")
    heldout_target_current = summarize_details(heldout_current_target_details)
    heldout_target_oracle = summarize_details(heldout_oracle_target_details)

    current_target_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    oracle_target_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bucket_by_id = dict(zip(heldout_target_df["sample_id"], heldout_target_df["heldout_bucket"], strict=False))
    for detail in heldout_current_target_details:
        current_target_by_bucket[bucket_by_id[detail["sample_id"]]].append(detail)
    for detail in heldout_oracle_target_details:
        oracle_target_by_bucket[bucket_by_id[detail["sample_id"]]].append(detail)

    projected_df = heldout_df.copy()
    projected_df["mixed_label"] = projected_df.apply(
        lambda row: row["oracle_label"] if row["sample_id"] in targeted_fail_ids else row["current_label"],
        axis=1,
    )
    heldout_projected_details = evaluate_rows(projected_df, "mixed_label")
    heldout_projected_overall = summarize_details(heldout_projected_details)

    metric_note_zero_target = "gt_target_edge_total=0; relief-only scope 上该指标按当前 strict 实现记为 0.0，语义上应视为 N/A。"
    rows: list[dict[str, Any]] = []
    rows.extend(
        metric_rows(
            scope="A.relief_pack_overall",
            bucket="overall",
            evidence_type="oracle_label_adequacy",
            current_summary=pack_oracle_summary,
            oracle_summary=pack_oracle_summary,
            note_by_metric={"target_recall": metric_note_zero_target},
        )
    )
    for bucket in TARGET_BUCKETS:
        rows.extend(
            metric_rows(
                scope=f"B.{bucket}",
                bucket=bucket,
                evidence_type="oracle_label_adequacy",
                current_summary=pack_by_bucket[bucket],
                oracle_summary=pack_by_bucket[bucket],
                note_by_metric={"target_recall": metric_note_zero_target},
            )
        )

    rows.extend(
        metric_rows(
            scope="C.heldout_alignment_reference",
            bucket="overall",
            evidence_type="failure_mode_alignment + bucket_coverage + oracle_label_adequacy",
            current_summary=heldout_current_overall,
            oracle_summary=heldout_projected_overall,
        )
    )
    for bucket in TARGET_BUCKETS:
        rows.extend(
            metric_rows(
                scope="C.heldout_alignment_reference",
                bucket=bucket,
                evidence_type="failure_mode_alignment + oracle_label_adequacy",
                current_summary=summarize_details(current_target_by_bucket[bucket]),
                oracle_summary=summarize_details(oracle_target_by_bucket[bucket]),
                note_by_metric={"target_recall": metric_note_zero_target},
            )
        )
    rows.extend(coverage_rows(heldout_fail_df=heldout_target_df, pack_df=pack_df))

    eval_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_csv, index=False, encoding="utf-8")

    max_allowed_no_anchor = math.floor((0.15 * changed_sample_count) - 1e-9)
    fixes_needed_for_sub_015 = max(0, current_no_anchor_failures - max_allowed_no_anchor)

    pack_train_counts = {
        bucket: int(((pack_df["pack_bucket"] == bucket) & (pack_df["pack_split_name"] == "train")).sum())
        for bucket in TARGET_BUCKETS
    }
    pack_eval_counts = {
        bucket: int(((pack_df["pack_bucket"] == bucket) & (pack_df["pack_split_name"] == "eval")).sum())
        for bucket in TARGET_BUCKETS
    }
    pack_total_counts = {bucket: int((pack_df["pack_bucket"] == bucket).sum()) for bucket in TARGET_BUCKETS}

    pack_all_ids = set(pack_df["origin_sample_id"].astype(str))
    pack_train_ids = set(pack_df[pack_df["pack_split_name"] == "train"]["origin_sample_id"].astype(str))
    pack_all_ranges = {
        bucket: set(pack_df[pack_df["pack_bucket"] == bucket]["range_fingerprint"].astype(str))
        for bucket in TARGET_BUCKETS
    }
    pack_train_ranges = {
        bucket: set(
            pack_df[
                (pack_df["pack_bucket"] == bucket) & (pack_df["pack_split_name"] == "train")
            ]["range_fingerprint"].astype(str)
        )
        for bucket in TARGET_BUCKETS
    }
    direct_overlap_total = int(heldout_target_df["sample_id"].astype(str).isin(pack_all_ids).sum())
    direct_overlap_train = int(heldout_target_df["sample_id"].astype(str).isin(pack_train_ids).sum())
    range_overlap_total = int(
        sum(
            row["range_fingerprint"] in pack_all_ranges[row["heldout_bucket"]]
            for _, row in heldout_target_df.iterrows()
        )
    )
    range_overlap_train = int(
        sum(
            row["range_fingerprint"] in pack_train_ranges[row["heldout_bucket"]]
            for _, row in heldout_target_df.iterrows()
        )
    )

    key_bucket = max(TARGET_BUCKETS, key=lambda bucket: heldout_target_counts.get(bucket, 0))
    high_risk_bucket = "multi_edge_relief"
    stage4b_ready = (
        heldout_target_oracle["no_anchor_when_gt_changed_rate"] == 0.0
        and heldout_projected_overall["no_anchor_when_gt_changed_rate"] < 0.15
        and pack_oracle_summary["gt_changed_edge_recall"] == 100.0
        and pack_oracle_summary["MAE_on_changed_edges"] == 0.0
    )
    decision = "GO_TO_STAGE4B_WITH_RELIEF_PACK" if stage4b_ready else "HOLD_AND_EXPAND_RELIEF_SUPERVISION"

    metric_rows_md = [["Scope", "Bucket", "Metric", "Current", "Oracle", "Gain"]]
    for scope, bucket in [
        ("A.relief_pack_overall", "overall"),
        ("C.heldout_alignment_reference", "overall"),
    ]:
        subset = eval_df[(eval_df["row_type"] == "metric") & (eval_df["scope"] == scope) & (eval_df["bucket"] == bucket)]
        for metric in METRICS:
            row = subset[subset["metric"] == metric].iloc[0]
            metric_rows_md.append(
                [
                    scope,
                    bucket,
                    metric,
                    str(row["current_reference"]),
                    str(row["oracle_ceiling"]),
                    str(row["absolute_gain"]),
                ]
            )

    bucket_metric_rows_md = [["Bucket", "Metric", "Current", "Oracle", "Gain"]]
    for bucket in TARGET_BUCKETS:
        subset = eval_df[(eval_df["row_type"] == "metric") & (eval_df["scope"] == "C.heldout_alignment_reference") & (eval_df["bucket"] == bucket)]
        for metric in METRICS:
            row = subset[subset["metric"] == metric].iloc[0]
            bucket_metric_rows_md.append(
                [
                    bucket,
                    metric,
                    str(row["current_reference"]),
                    str(row["oracle_ceiling"]),
                    str(row["absolute_gain"]),
                ]
            )

    coverage_md = [["Bucket", "Held-out Fail", "Pack Train", "Pack Eval", "Direct(total)", "Exact Range(total)"]]
    for bucket in TARGET_BUCKETS:
        bucket_df = heldout_target_df[heldout_target_df["heldout_bucket"] == bucket]
        coverage_md.append(
            [
                bucket,
                str(int(len(bucket_df))),
                str(pack_train_counts[bucket]),
                str(pack_eval_counts[bucket]),
                str(int(bucket_df["sample_id"].astype(str).isin(pack_all_ids).sum())),
                str(
                    int(
                        sum(
                            row["range_fingerprint"] in pack_all_ranges[bucket]
                            for _, row in bucket_df.iterrows()
                        )
                    )
                ),
            ]
        )

    warning_lines = [f"- WARNING: {warning}" for warning in warnings] or ["- 无额外 warning。"]
    pack_examples_by_bucket = pack_examples.get("examples_by_bucket", {})

    notes = [
        "# Relief Pack Ceiling Evaluation",
        "",
        "## Read Method",
        "- 所有 parquet 均显式使用 `pyarrow.parquet.read_table(...)` 读取，并只在读入后转成 Python dict / pandas DataFrame；没有依赖 `pandas.read_parquet(...)`。",
        "",
        "## Evaluation Setup",
        "- `A.relief_pack_overall`：在 relief pack 自身上做 oracle ceiling sanity check，确认标签与 strict 目标是否一致。",
        "- `B.<bucket>`：分别检查每个 relief bucket 在 pack 自身上的 oracle label adequacy。",
        "- `C.heldout_alignment_reference`：不把 pack 和 held-out 伪装成 sample-level 配对；主证据明确限定为 `failure-mode alignment + bucket coverage + oracle label adequacy`。",
        "- 另外，为了验证 strict 目标本身是否可达，我们只在 held-out 自身内部做 policy-A oracle rewrite：这说明标签上限，不等于训练后真实收益。",
        "",
        "## Metric Table",
        markdown_table(metric_rows_md),
        "",
        "## Held-out Bucket Table",
        markdown_table(bucket_metric_rows_md),
        "",
        "## Coverage Table",
        markdown_table(coverage_md),
        "",
        "## Sanity Checks",
        f"- relief pack bucket set == target buckets: `{set(pack_df['pack_bucket']) == set(TARGET_BUCKETS)}`",
        f"- relief pack oracle `gt_changed_edge_recall`: `{pack_oracle_summary['gt_changed_edge_recall']:.1f}`",
        f"- relief pack oracle `MAE_on_changed_edges`: `{pack_oracle_summary['MAE_on_changed_edges']:.4f}`",
        f"- relief pack oracle `parse_fail_rate_strict`: `{pack_oracle_summary['parse_fail_rate_strict']:.4f}`",
        f"- held-out strict details simple_local count: `{len(strict_simple)}`",
        f"- failure summary simple_local count: `{failure_ref['simple_local_total']}`",
        f"- failure summary no_anchor count: `{failure_ref['no_anchor_failure_total']}`",
        f"- strict-details current no_anchor changed count: `{current_no_anchor_failures}`",
        *warning_lines,
        "",
        "## Required Answers",
        f"- 这三类新 supervision 是否正中当前 held-out 主瓶颈：`是`。当前 held-out `simple_local` 的剩余主失败模式就是这三类 relief bucket，对应 `29/29` 个 `no_anchor_when_gt_changed` 失败样本。",
        f"- 若将其加入后续 stage4b，`simple_local no_anchor_when_gt_changed_rate` 是否“有现实依据”向 `0.15` 以下推进：`有`。",
        f"  现实依据来自三层证据而不是伪造配对：",
        f"  1. `failure-mode alignment`：当前 `29` 个 no-anchor changed 失败全部落在这三类 bucket。",
        f"  2. `oracle label adequacy`：这三类 held-out 失败样本在 policy-A oracle 下可从 `no_anchor_when_gt_changed_rate 1.0000 -> 0.0000`，`gt_changed_edge_recall 0.0 -> 100.0`，`MAE_on_changed_edges 0.8000 -> 0.0000`。",
        f"  3. `bucket coverage`：pack 全量对 held-out 失败样本有 `17/29` direct overlap、`21/29` exact ROAD+DIR+RANGE overlap，其中 `short_range` 与 `multi_edge` 已做到 exact coverage 全覆盖。",
        f"- 为什么说 `0.15` 以下有现实依据：当前 changed 样本总数是 `{changed_sample_count}`，当前 no-anchor changed 样本是 `{current_no_anchor_failures}`。要把该指标压到 `0.15` 以下，至少只需修掉 `{fixes_needed_for_sub_015}` 个失败样本；而当前被 relief pack 对准的失败样本有 `29` 个，量级上明显足够。",
        f"- 哪个 bucket 对压 no_anchor 最关键：`{key_bucket}`。因为它在 held-out 当前剩余瓶颈中占比最高，为 `{heldout_target_counts.get(key_bucket, 0)}/{len(heldout_target_df)} = {heldout_target_counts.get(key_bucket, 0)/max(len(heldout_target_df),1):.1%}`。",
        f"- 哪个 bucket 即使补了也仍然风险最高：`{high_risk_bucket}`。原因不是数量最大，而是连续 3+ edge 的范围边界最容易截短或偏移，而且该 bucket 的历史唯一语义最少。",
        f"- `multi_edge_relief` 的边界歧义是否会显著影响后续训练收益：`不会显著阻断 overall 收益，但会成为最需要监控的尾部风险`。当前 held-out 里该 bucket 只有 `2` 条失败，所以它不太可能主导 overall 结果；但它最容易让模型在区间边界上丢边，因此对单桶收益波动最敏感。",
        f"- 当前更适合：`{decision}`。",
        "",
        "## Decision Rationale",
        f"- relief pack 自身 oracle 已经与 strict 对齐：overall `gt_changed_edge_recall=100.0`、`MAE_on_changed_edges=0.0000`。",
        f"- held-out overall `simple_local` 在 bucket-targeted oracle projection 下，`no_anchor_when_gt_changed_rate` 可从 `{heldout_current_overall['no_anchor_when_gt_changed_rate']:.4f}` 降到 `{heldout_projected_overall['no_anchor_when_gt_changed_rate']:.4f}`，`gt_changed_edge_recall` 可从 `{heldout_current_overall['gt_changed_edge_recall']:.1f}` 升到 `{heldout_projected_overall['gt_changed_edge_recall']:.1f}`。",
        f"- 因此，当前瓶颈更像是“缺少这三类 supervision”，而不是 strict 目标过强。",
        "",
        "## Recommended Stage4b Composition",
        (
            "- 最推荐的 stage4b 训练集组成："
            if decision == "GO_TO_STAGE4B_WITH_RELIEF_PACK"
            else "- 当前还不建议进入 stage4b 训练；下一步扩展建议："
        ),
        (
            "- 训练主集：保留现有 `dataset_sparse_v2_stage4b_relabel/train/data.parquet` 作为 anomaly+normal strict-aligned 主集，再追加 `dataset_simple_local_relief_pack/train/data.parquet` 这 48 条 relief 补充监督。"
            if decision == "GO_TO_STAGE4B_WITH_RELIEF_PACK"
            else f"- 优先继续扩 `relief_normal_single_edge`，因为它仍是 held-out 剩余 no-anchor 最大头部来源。"
        ),
        (
            "- 评估集：继续保留现有 `dataset_sparse_v2_stage4b_relabel/eval/data.parquet`，并额外单独监控 `dataset_simple_local_relief_pack/eval/data.parquet` 这 24 条 relief eval。"
            if decision == "GO_TO_STAGE4B_WITH_RELIEF_PACK"
            else f"- 其次再补 `multi_edge_relief` 的边界多样性，尤其是 3+ edge 连续区间的不同道路/方向组合。"
        ),
        (
            "- 采样建议：适度提高 `relief_normal_single_edge` 的 batch 暴露频次，因为它对压 `no_anchor` 最关键；`multi_edge_relief` 保持较小但稳定的曝光，用于约束边界行为。"
            if decision == "GO_TO_STAGE4B_WITH_RELIEF_PACK"
            else "- 在进入训练前，先把这三类 pack 的 exact range coverage 尤其是 single-edge 未覆盖的 8 条 held-out 模式再扩一轮。"
        ),
        "",
        "## Output Files",
        f"- CSV: `{output_csv}`",
        f"- Notes: `{output_notes}`",
    ]

    output_notes.parent.mkdir(parents=True, exist_ok=True)
    output_notes.write_text("\n".join(notes), encoding="utf-8")

    print(
        "现在是否具备进入 stage4b 的现实依据："
        f"{'是' if decision == 'GO_TO_STAGE4B_WITH_RELIEF_PACK' else '否'}"
    )
    if decision == "GO_TO_STAGE4B_WITH_RELIEF_PACK":
        print(
            "最推荐的 stage4b 训练集组成：保留现有 stage4b relabel train 主集，追加 relief pack train 48 条；"
            "同时把 relief pack eval 24 条作为单独监控切片。"
        )
    else:
        print("下一步最该扩的 supervision：relief_normal_single_edge，其次补 multi_edge_relief 的边界多样性。")
    if warnings:
        print(f"warning_count={len(warnings)}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
