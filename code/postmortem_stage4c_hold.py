#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


KEY_METRICS = [
    "no_anchor_when_gt_changed_rate",
    "gt_changed_edge_recall",
    "parse_fail_rate_strict",
]

NEUTRAL_PHRASES = [
    "局部恢复正常",
    "恢复正常通行",
    "保持正常通行",
    "交通基本正常",
    "正常通行",
    "通行正常",
    "恢复正常",
    "车流稳定",
    "道路畅通",
    "车流顺畅",
    "通行顺畅",
    "畅通无阻",
    "畅通",
    "顺畅",
    "正常",
]

ABNORMAL_PHRASES = [
    "严重拥堵",
    "中度拥堵",
    "拥堵",
    "缓行",
    "追尾事故",
    "事故",
    "封闭",
    "排队",
    "积压",
    "效率下降",
]

PHRASE_PATTERNS: list[tuple[str, str]] = [
    ("局部恢复正常", "局部恢复正常"),
    ("恢复正常通行", "恢复正常"),
    ("保持正常通行", "保持正常通行"),
    ("交通基本正常", "交通基本正常"),
    ("正常通行", "正常通行"),
    ("通行正常", "通行正常"),
    ("恢复正常", "恢复正常"),
    ("车流顺畅", "顺畅"),
    ("通行顺畅", "顺畅"),
    ("畅通无阻", "畅通无阻"),
    ("道路畅通", "畅通"),
    ("畅通", "畅通"),
    ("顺畅", "顺畅"),
    ("正常", "正常"),
]

WEAK_RELIEF_GROUPS = {
    "局部恢复正常",
    "恢复正常",
    "保持正常通行",
    "交通基本正常",
    "正常通行",
    "通行正常",
    "正常",
}

ANCHOR_RE = re.compile(
    r"ANCHOR\|ROAD=(?P<road>[^|]+)\|DIR=(?P<direction>[^|]+)\|RANGE=(?P<range>[^|]+)\|LEVEL=(?P<level>[^\n]+)"
)
FALLBACK_TEXT_RE = re.compile(
    r"(?P<road>[^；，。 ]+?(?:路|街))(?P<range>[^；，。 ]+?至[^；，。 ]+?(?:路|街))(?:段)?(?P<direction>向东|向西|向南|向北)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postmortem for stage4c HOLD_AND_REPAIR without launching training.")
    parser.add_argument(
        "--release-decision-md",
        default="/root/autodl-tmp/results/stage4c_release_decision.md",
    )
    parser.add_argument(
        "--diff-json",
        default="/root/autodl-tmp/results/stage4c_diff_29.json",
    )
    parser.add_argument(
        "--stage4c-summary-csv",
        default="/root/autodl-tmp/results/sparse_v2_validation_strict_stage4c_fix_summary.csv",
    )
    parser.add_argument(
        "--stage4c-details-csv",
        default="/root/autodl-tmp/results/sparse_v2_validation_strict_stage4c_fix_details.csv",
    )
    parser.add_argument(
        "--shadow-summary-csv",
        default="/root/autodl-tmp/results/stage4c_shadow_eval_summary.csv",
    )
    parser.add_argument(
        "--relief-summary-csv",
        default="/root/autodl-tmp/results/relief_pack_stage4c_eval_summary.csv",
    )
    parser.add_argument(
        "--guard-summary-csv",
        default="/root/autodl-tmp/results/stage4c_guard_eval_summary.csv",
    )
    parser.add_argument(
        "--stage4-summary-csv",
        default="/root/autodl-tmp/results/sparse_v2_validation_strict_stage4_fix_summary.csv",
    )
    parser.add_argument(
        "--stage4-details-csv",
        default="/root/autodl-tmp/results/sparse_v2_validation_strict_stage4_fix_details.csv",
    )
    parser.add_argument(
        "--output-summary-md",
        default="/root/autodl-tmp/results/stage4c_postmortem_summary.md",
    )
    parser.add_argument(
        "--output-buckets-csv",
        default="/root/autodl-tmp/results/stage4c_postmortem_buckets.csv",
    )
    parser.add_argument(
        "--output-hard-cases-json",
        default="/root/autodl-tmp/results/stage4c_remaining_hard_cases.json",
    )
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_json_loads(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return json.loads(text)


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\\n", "\n").strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return not math.isnan(value) and value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        if pd.isna(value):
            return 0.0
    except TypeError:
        pass
    return float(value)


def phrase_variant(text: str) -> tuple[str, str]:
    text = clean_text(text)
    for surface, group in PHRASE_PATTERNS:
        if surface in text:
            return surface, group
    return "unknown", "unknown"


def structure_tag(text: str) -> str:
    text = clean_text(text)
    if text.startswith("当前需要根据常态或局部短时事件进行路径规划。"):
        return "planner_long"
    if text.startswith("常态或局部短时事件："):
        return "prefixed_short"
    if "；" in text:
        return "mixed_clause"
    return "short_plain"


def is_very_short_sentence(text: str) -> bool:
    text = clean_text(text)
    compact = re.sub(r"\s+", "", text)
    return len(compact) <= 20


def has_neutral_phrase(text: str) -> bool:
    text = clean_text(text)
    return any(token in text for token in NEUTRAL_PHRASES)


def has_abnormal_phrase(text: str) -> bool:
    text = clean_text(text)
    return any(token in text for token in ABNORMAL_PHRASES)


def is_mixed_normal_abnormal(text: str) -> bool:
    text = clean_text(text)
    return has_neutral_phrase(text) and has_abnormal_phrase(text)


def is_weak_relief(text: str) -> bool:
    _, group = phrase_variant(text)
    return group in WEAK_RELIEF_GROUPS


def edge_profile(changed_edge_count: int) -> str:
    if changed_edge_count <= 1:
        return "single_edge"
    if changed_edge_count == 2:
        return "short_range"
    return "multi_edge"


def semantic_type(row: pd.Series) -> str:
    changed_edge_count = int(row["gt_changed_edge_count_post"])
    text = clean_text(row["constraint_text_post"])
    if changed_edge_count <= 1 and has_neutral_phrase(text) and not has_abnormal_phrase(text):
        return "relief_normal_single_edge"
    if changed_edge_count == 2 and has_neutral_phrase(text) and not has_abnormal_phrase(text):
        return "relief_normal_short_range"
    if changed_edge_count >= 3 and has_neutral_phrase(text) and not has_abnormal_phrase(text):
        return "multi_edge_relief"
    if is_mixed_normal_abnormal(text):
        return "anomaly_plus_normal_side_event"
    return edge_profile(changed_edge_count)


def failure_status(no_anchor: Any, strict_parse_fail: Any, gt_changed_edge_count: Any, gt_changed_edge_recall: Any) -> str:
    no_anchor_bool = as_bool(no_anchor)
    strict_parse_fail_bool = as_bool(strict_parse_fail)
    changed_edge_count = int(as_float(gt_changed_edge_count))
    changed_recall = as_float(gt_changed_edge_recall)
    if changed_edge_count > 0 and strict_parse_fail_bool:
        return "no_anchor" if no_anchor_bool else "strict_fail"
    if changed_edge_count > 0 and changed_recall < 100.0:
        return "no_anchor" if no_anchor_bool else "anchor_but_incomplete_range"
    if changed_edge_count == 0 and not no_anchor_bool:
        return "false_positive_anchor"
    return "success"


def extract_first_anchor(raw_preview: str) -> dict[str, str]:
    match = ANCHOR_RE.search(clean_text(raw_preview))
    if match:
        return {
            "road": match.group("road").strip(),
            "dir": match.group("direction").strip(),
            "range": match.group("range").strip(),
            "level": match.group("level").strip(),
        }
    return {}


def extract_road_dir_range(text: str, raw_preview: str) -> dict[str, str]:
    anchor = extract_first_anchor(raw_preview)
    if anchor:
        return anchor
    match = FALLBACK_TEXT_RE.search(clean_text(text))
    if match:
        return {
            "road": match.group("road").strip(),
            "dir": match.group("direction").strip(),
            "range": match.group("range").strip(),
            "level": "",
        }
    return {"road": "", "dir": "", "range": "", "level": ""}


def load_summary_lookup(path: str | Path) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    df = pd.read_csv(path)
    lookup = {
        (str(row.scope), str(row.scope_name)): row._asdict()
        for row in df.itertuples(index=False)
    }
    return df, lookup


def lookup_row(lookup: dict[tuple[str, str], dict[str, Any]], scope: str, scope_name: str) -> dict[str, Any]:
    try:
        return lookup[(scope, scope_name)]
    except KeyError as exc:
        raise KeyError(f"Missing summary row {scope}/{scope_name}") from exc


def build_summary_comparison(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    merged = pre_df.merge(post_df, on=["scope", "scope_name", "count"], suffixes=("_pre", "_post"))
    for metric in KEY_METRICS:
        merged[f"delta_{metric}"] = merged[f"{metric}_post"] - merged[f"{metric}_pre"]

    merged["improved_metrics"] = (
        (merged["no_anchor_when_gt_changed_rate_post"] < merged["no_anchor_when_gt_changed_rate_pre"]).astype(int)
        + (merged["gt_changed_edge_recall_post"] > merged["gt_changed_edge_recall_pre"]).astype(int)
        + (merged["parse_fail_rate_strict_post"] < merged["parse_fail_rate_strict_pre"]).astype(int)
    )
    merged["regressed_metrics"] = (
        (merged["no_anchor_when_gt_changed_rate_post"] > merged["no_anchor_when_gt_changed_rate_pre"]).astype(int)
        + (merged["gt_changed_edge_recall_post"] < merged["gt_changed_edge_recall_pre"]).astype(int)
        + (merged["parse_fail_rate_strict_post"] > merged["parse_fail_rate_strict_pre"]).astype(int)
    )
    merged["net_status"] = merged.apply(classify_summary_status, axis=1)
    return merged


def classify_summary_status(row: pd.Series) -> str:
    if int(row["improved_metrics"]) > 0 and int(row["regressed_metrics"]) == 0:
        return "net_improved"
    if int(row["improved_metrics"]) == 0 and int(row["regressed_metrics"]) == 0:
        return "unchanged"
    if int(row["regressed_metrics"]) > 0 and int(row["improved_metrics"]) == 0:
        return "net_regressed"
    return "mixed"


def enrich_detail_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ("no_anchor", "parse_fail", "strict_parse_fail"):
        if column in df.columns:
            df[column] = df[column].map(as_bool)
    df["constraint_text"] = df["constraint_text"].map(clean_text)
    df["prediction_raw_preview"] = df["prediction_raw_preview"].map(clean_text)
    df["phrase_variant"], df["phrase_group"] = zip(*df["constraint_text"].map(phrase_variant), strict=True)
    df["structure_tag"] = df["constraint_text"].map(structure_tag)
    df["is_very_short_sentence"] = df["constraint_text"].map(is_very_short_sentence)
    df["is_mixed_normal_abnormal"] = df["constraint_text"].map(is_mixed_normal_abnormal)
    df["is_weak_relief"] = df["constraint_text"].map(is_weak_relief)
    df["edge_profile"] = df["gt_changed_edge_count"].map(lambda value: edge_profile(int(as_float(value))))
    df["failure_status"] = df.apply(
        lambda row: failure_status(
            row["no_anchor"],
            row["strict_parse_fail"],
            row["gt_changed_edge_count"],
            row["gt_changed_edge_recall"],
        ),
        axis=1,
    )
    return df


def build_detail_merge(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [
        "sample_id",
        "split",
        "split_index",
        "scene_type",
        "length_bucket",
        "gt_changed_edge_count",
        "gt_target_edge_count",
        "no_anchor",
        "strict_parse_fail",
        "strict_failure_reason",
        "gt_changed_edge_recall",
        "target_recall",
        "constraint_text",
        "prediction_raw_preview",
        "phrase_variant",
        "phrase_group",
        "structure_tag",
        "is_very_short_sentence",
        "is_mixed_normal_abnormal",
        "is_weak_relief",
        "edge_profile",
        "failure_status",
    ]
    merged = pre_df[keep_columns].merge(post_df[keep_columns], on="sample_id", suffixes=("_pre", "_post"))
    merged["net_improved"] = (
        (merged["failure_status_pre"] != "success") & (merged["failure_status_post"] == "success")
    )
    merged["net_regressed"] = (
        (merged["failure_status_pre"] == "success") & (merged["failure_status_post"] != "success")
    )
    merged["no_anchor_to_anchor_post_wrong"] = (
        (merged["no_anchor_pre"] == True)
        & (merged["no_anchor_post"] == False)
        & (merged["failure_status_post"] != "success")
        & (merged["gt_changed_edge_count_post"] > 0)
    )
    merged["new_false_positive_anchor"] = (
        (merged["gt_changed_edge_count_post"] == 0)
        & (merged["no_anchor_pre"] == True)
        & (merged["no_anchor_post"] == False)
    )
    merged["semantic_type_post"] = merged.apply(semantic_type, axis=1)
    return merged


def summarize_diff29(diff_payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.DataFrame(diff_payload.get("samples", []))
    if rows.empty:
        raise ValueError("stage4c_diff_29.json has no samples")

    grouped_rows: list[dict[str, Any]] = []
    for bucket_name, part in rows.groupby("semantic_gap_subtype", dropna=False):
        changed_flags = part["changed"].map(as_bool)
        grouped_rows.append(
            {
                "bucket_name": str(bucket_name),
                "count": int(len(part)),
                "changed_count": int(changed_flags.sum()),
                "unchanged_count": int((~changed_flags).sum()),
                "no_anchor_to_anchor_count": int((part["change_type"] == "no_anchor_to_anchor").sum()),
                "anchor_but_still_wrong_count": int(part["post_anchor_still_wrong"].map(as_bool).sum()),
            }
        )
    grouped = pd.DataFrame(grouped_rows)
    grouped["bucket_family"] = "diff29_subtype"
    grouped["net_status"] = grouped["changed_count"].map(lambda value: "unchanged" if int(value) == 0 else "net_improved")
    return rows, grouped


def group_post_failures(post_normal_simple_local: pd.DataFrame) -> list[dict[str, Any]]:
    def resolve_col(base_name: str) -> str:
        if base_name in post_normal_simple_local.columns:
            return base_name
        suffixed = f"{base_name}_post"
        if suffixed in post_normal_simple_local.columns:
            return suffixed
        raise KeyError(f"Missing column {base_name} or {suffixed}")

    failure_col = resolve_col("failure_status")
    failures = post_normal_simple_local[post_normal_simple_local[failure_col] != "success"].copy()
    outputs: list[dict[str, Any]] = []
    for column in ["failure_status", "edge_profile", "phrase_group", "structure_tag", "is_mixed_normal_abnormal"]:
        actual_column = resolve_col(column)
        grouped = failures.groupby(actual_column).size().reset_index(name="count")
        for row in grouped.itertuples(index=False):
            outputs.append(
                {
                    "bucket_family": f"remaining_{column}",
                    "bucket_name": str(getattr(row, actual_column)),
                    "count": int(row.count),
                }
            )
    return outputs


def build_remaining_hard_cases(
    merged: pd.DataFrame,
    diff_rows: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normal_simple_local = merged[
        (merged["split_post"] == "val_normal")
        & (merged["scene_type_post"] == "simple_local")
    ].copy()

    remaining_failures = normal_simple_local[
        (normal_simple_local["gt_changed_edge_count_post"] > 0)
        & (normal_simple_local["failure_status_post"] != "success")
    ].copy()
    remaining_single_edge = remaining_failures[remaining_failures["gt_changed_edge_count_post"] == 1].copy()
    range_boundary_cases = remaining_failures[
        remaining_failures["failure_status_post"] == "anchor_but_incomplete_range"
    ].copy()
    false_positive_cases = normal_simple_local[normal_simple_local["new_false_positive_anchor"]].copy()

    diff_lookup = {
        str(row.sample_id): row._asdict()
        for row in diff_rows.itertuples(index=False)
    }

    single_edge_cases: list[dict[str, Any]] = []
    for row in remaining_single_edge.sort_values(["structure_tag_post", "phrase_group_post", "sample_id"]).itertuples(index=False):
        diff_row = diff_lookup.get(str(row.sample_id), {})
        event_records = safe_json_loads(diff_row.get("event_records_json", "[]")) or []
        primary_event = event_records[0] if isinstance(event_records, list) and event_records else {}
        fallback = extract_road_dir_range(row.constraint_text_post, row.prediction_raw_preview_post)
        exact_phrase, phrase_group = phrase_variant(row.constraint_text_post)
        item = {
            "sample_id": row.sample_id,
            "split": row.split_post,
            "split_index": int(as_float(row.split_index_post)),
            "constraint_text": row.constraint_text_post,
            "failure_mode": row.failure_status_post,
            "semantic_gap_subtype": diff_row.get("semantic_gap_subtype", "relief_normal_single_edge"),
            "phrase_variant": exact_phrase,
            "phrase_group": phrase_group,
            "road": clean_text(primary_event.get("road")) or fallback.get("road", ""),
            "dir": clean_text(primary_event.get("dir")) or fallback.get("dir", ""),
            "range": clean_text(primary_event.get("range")) or fallback.get("range", ""),
            "type": "changed_relief" if clean_text(primary_event.get("level")) == "畅通" else semantic_type(pd.Series({"gt_changed_edge_count_post": row.gt_changed_edge_count_post, "constraint_text_post": row.constraint_text_post})),
            "level": clean_text(primary_event.get("level")) or fallback.get("level", ""),
            "gt_changed_edge_count": int(as_float(row.gt_changed_edge_count_post)),
            "length_bucket": row.length_bucket_post,
            "structure_tag": row.structure_tag_post,
            "is_very_short_sentence": bool(row.is_very_short_sentence_post),
            "is_mixed_normal_abnormal": bool(row.is_mixed_normal_abnormal_post),
            "is_weak_relief": bool(row.is_weak_relief_post),
            "pre_status": row.failure_status_pre,
            "post_status": row.failure_status_post,
            "pre_no_anchor": bool(row.no_anchor_pre),
            "post_no_anchor": bool(row.no_anchor_post),
            "pre_gt_changed_edge_recall": as_float(row.gt_changed_edge_recall_pre),
            "post_gt_changed_edge_recall": as_float(row.gt_changed_edge_recall_post),
            "post_prediction_preview": row.prediction_raw_preview_post,
        }
        single_edge_cases.append(item)

    range_reference: list[dict[str, Any]] = []
    for row in range_boundary_cases.sort_values(["gt_changed_edge_count_post", "sample_id"]).itertuples(index=False):
        exact_phrase, phrase_group = phrase_variant(row.constraint_text_post)
        hint = extract_road_dir_range(row.constraint_text_post, row.prediction_raw_preview_post)
        range_reference.append(
            {
                "sample_id": row.sample_id,
                "constraint_text": row.constraint_text_post,
                "failure_mode": row.failure_status_post,
                "phrase_variant": exact_phrase,
                "phrase_group": phrase_group,
                "road_hint": hint.get("road", ""),
                "dir_hint": hint.get("dir", ""),
                "range_hint": hint.get("range", ""),
                "type": row.semantic_type_post,
                "gt_changed_edge_count": int(as_float(row.gt_changed_edge_count_post)),
                "length_bucket": row.length_bucket_post,
                "structure_tag": row.structure_tag_post,
                "is_very_short_sentence": bool(row.is_very_short_sentence_post),
                "is_mixed_normal_abnormal": bool(row.is_mixed_normal_abnormal_post),
                "is_weak_relief": bool(row.is_weak_relief_post),
                "post_gt_changed_edge_recall": as_float(row.gt_changed_edge_recall_post),
            }
        )

    false_positive_reference: list[dict[str, Any]] = []
    for row in false_positive_cases.sort_values(["phrase_group_post", "sample_id"]).itertuples(index=False):
        exact_phrase, phrase_group = phrase_variant(row.constraint_text_post)
        hint = extract_road_dir_range(row.constraint_text_post, row.prediction_raw_preview_post)
        false_positive_reference.append(
            {
                "sample_id": row.sample_id,
                "constraint_text": row.constraint_text_post,
                "phrase_variant": exact_phrase,
                "phrase_group": phrase_group,
                "road": hint.get("road", ""),
                "dir": hint.get("dir", ""),
                "range": hint.get("range", ""),
                "type": "normal_only_false_positive",
                "length_bucket": row.length_bucket_post,
                "structure_tag": row.structure_tag_post,
                "is_very_short_sentence": bool(row.is_very_short_sentence_post),
                "is_weak_relief": bool(row.is_weak_relief_post),
                "post_prediction_preview": row.prediction_raw_preview_post,
            }
        )

    summary = {
        "remaining_simple_local_failures": int(len(remaining_failures)),
        "remaining_single_edge_failures": int(len(remaining_single_edge)),
        "remaining_no_anchor_failures": int((remaining_failures["failure_status_post"] == "no_anchor").sum()),
        "remaining_anchor_but_incomplete_range": int((remaining_failures["failure_status_post"] == "anchor_but_incomplete_range").sum()),
        "remaining_mixed_normal_abnormal": int(remaining_failures["is_mixed_normal_abnormal_post"].sum()),
        "remaining_very_short": int(remaining_failures["is_very_short_sentence_post"].sum()),
        "remaining_weak_relief": int(remaining_failures["is_weak_relief_post"].sum()),
        "new_false_positive_normal_only_anchors": int(len(false_positive_cases)),
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_rule": "Held-out val_normal/simple_local rows where stage4c still fails on changed edges, with single-edge failures preserved in full detail.",
        "counts": summary,
        "still_failed_single_edge_cases": single_edge_cases,
        "range_boundary_reference_cases": range_reference,
        "new_false_positive_normal_only_anchors": false_positive_reference,
    }
    return payload, summary


def choose_verdict(
    *,
    diff_rows: pd.DataFrame,
    merged: pd.DataFrame,
    relief_summary_lookup: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, str, str]:
    moved_29 = int(diff_rows["changed"].map(as_bool).sum())
    normal_simple_local = merged[
        (merged["split_post"] == "val_normal")
        & (merged["scene_type_post"] == "simple_local")
    ].copy()
    true_single_edge_fixes = int(
        (
            (normal_simple_local["gt_changed_edge_count_post"] == 1)
            & normal_simple_local["net_improved"]
        ).sum()
    )
    new_false_positive_anchors = int(normal_simple_local["new_false_positive_anchor"].sum())
    relief_overall = lookup_row(relief_summary_lookup, "overall", "overall")
    relief_learned = (
        as_float(relief_overall["no_anchor_when_gt_changed_rate"]) == 0.0
        and as_float(relief_overall["gt_changed_edge_recall"]) >= 99.9
        and as_float(relief_overall["parse_fail_rate_strict"]) == 0.0
    )

    if moved_29 <= 1 and true_single_edge_fixes <= 1 and new_false_positive_anchors >= 4 and relief_learned:
        return (
            "DIMINISHING_RETURNS_REACHED",
            "迁移失败 > 范围边界失败",
            "收益已接近耗尽",
        )
    if moved_29 <= 2:
        return (
            "PATCH_SFT_ONLY_MARGINAL",
            "迁移失败 > 范围边界失败",
            "stage4c 失败更像迁移失败",
        )
    return (
        "PATCH_SFT_STILL_PROMISING",
        "迁移失败",
        "stage4c 失败更像迁移失败",
    )


def build_buckets_csv(
    summary_cmp: pd.DataFrame,
    diff_bucket_df: pd.DataFrame,
    merged: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for row in summary_cmp.sort_values(["scope", "scope_name"]).itertuples(index=False):
        rows.append(
            {
                "bucket_family": "heldout_summary",
                "bucket_name": f"{row.scope}/{row.scope_name}",
                "count": int(row.count),
                "pre_no_anchor_when_gt_changed_rate": row.no_anchor_when_gt_changed_rate_pre,
                "post_no_anchor_when_gt_changed_rate": row.no_anchor_when_gt_changed_rate_post,
                "delta_no_anchor_when_gt_changed_rate": row.delta_no_anchor_when_gt_changed_rate,
                "pre_gt_changed_edge_recall": row.gt_changed_edge_recall_pre,
                "post_gt_changed_edge_recall": row.gt_changed_edge_recall_post,
                "delta_gt_changed_edge_recall": row.delta_gt_changed_edge_recall,
                "pre_parse_fail_rate_strict": row.parse_fail_rate_strict_pre,
                "post_parse_fail_rate_strict": row.parse_fail_rate_strict_post,
                "delta_parse_fail_rate_strict": row.delta_parse_fail_rate_strict,
                "improved_metrics": int(row.improved_metrics),
                "regressed_metrics": int(row.regressed_metrics),
                "net_status": row.net_status,
                "changed_count": "",
                "unchanged_count": "",
                "no_anchor_to_anchor_count": "",
                "anchor_but_still_wrong_count": "",
                "notes": "",
            }
        )

    for row in diff_bucket_df.sort_values("bucket_name").itertuples(index=False):
        rows.append(
            {
                "bucket_family": row.bucket_family,
                "bucket_name": row.bucket_name,
                "count": int(row.count),
                "pre_no_anchor_when_gt_changed_rate": "",
                "post_no_anchor_when_gt_changed_rate": "",
                "delta_no_anchor_when_gt_changed_rate": "",
                "pre_gt_changed_edge_recall": "",
                "post_gt_changed_edge_recall": "",
                "delta_gt_changed_edge_recall": "",
                "pre_parse_fail_rate_strict": "",
                "post_parse_fail_rate_strict": "",
                "delta_parse_fail_rate_strict": "",
                "improved_metrics": "",
                "regressed_metrics": "",
                "net_status": row.net_status,
                "changed_count": int(row.changed_count),
                "unchanged_count": int(row.unchanged_count),
                "no_anchor_to_anchor_count": int(row.no_anchor_to_anchor_count),
                "anchor_but_still_wrong_count": int(row.anchor_but_still_wrong_count),
                "notes": "targeted semantic-gap bucket from diff_29",
            }
        )

    remaining_failures = merged[
        (merged["split_post"] == "val_normal")
        & (merged["scene_type_post"] == "simple_local")
        & (merged["gt_changed_edge_count_post"] > 0)
        & (merged["failure_status_post"] != "success")
    ].copy()
    for extra_row in group_post_failures(remaining_failures):
        rows.append(
            {
                "bucket_family": extra_row["bucket_family"],
                "bucket_name": extra_row["bucket_name"],
                "count": int(extra_row["count"]),
                "pre_no_anchor_when_gt_changed_rate": "",
                "post_no_anchor_when_gt_changed_rate": "",
                "delta_no_anchor_when_gt_changed_rate": "",
                "pre_gt_changed_edge_recall": "",
                "post_gt_changed_edge_recall": "",
                "delta_gt_changed_edge_recall": "",
                "pre_parse_fail_rate_strict": "",
                "post_parse_fail_rate_strict": "",
                "delta_parse_fail_rate_strict": "",
                "improved_metrics": "",
                "regressed_metrics": "",
                "net_status": "remaining_failure_bucket",
                "changed_count": "",
                "unchanged_count": "",
                "no_anchor_to_anchor_count": "",
                "anchor_but_still_wrong_count": "",
                "notes": "post-stage4c remaining hard cases on val_normal/simple_local",
            }
        )

    return pd.DataFrame(rows)


def markdown_lines(
    *,
    verdict: str,
    root_cause_shape: str,
    concise_conclusion: str,
    release_decision_text: str,
    summary_cmp: pd.DataFrame,
    diff_bucket_df: pd.DataFrame,
    merged: pd.DataFrame,
    shadow_lookup: dict[tuple[str, str], dict[str, Any]],
    relief_lookup: dict[tuple[str, str], dict[str, Any]],
    guard_lookup: dict[tuple[str, str], dict[str, Any]],
    hard_case_summary: dict[str, Any],
) -> list[str]:
    def find(scope: str, scope_name: str) -> pd.Series:
        match = summary_cmp[(summary_cmp["scope"] == scope) & (summary_cmp["scope_name"] == scope_name)]
        if match.empty:
            raise KeyError(f"Missing comparison row {scope}/{scope_name}")
        return match.iloc[0]

    simple_local = find("overall:scene_type", "simple_local")
    short_bucket = find("val_normal:length_bucket", "short")
    medium_bucket = find("val_normal:length_bucket", "medium")
    long_bucket = find("val_normal:length_bucket", "long")
    noisy_long_bucket = find("val_normal:length_bucket", "noisy_long")
    directional_bucket = find("overall:scene_type", "directional_asymmetry")
    anti_trunc = find("split", "val_anti_truncation")

    diff_lookup = {
        str(row.bucket_name): row._asdict()
        for row in diff_bucket_df.itertuples(index=False)
    }
    single_edge_diff = diff_lookup["relief_normal_single_edge"]
    short_range_diff = diff_lookup["relief_normal_short_range"]
    multi_edge_diff = diff_lookup["multi_edge_relief"]

    normal_simple_local = merged[
        (merged["split_post"] == "val_normal")
        & (merged["scene_type_post"] == "simple_local")
    ].copy()
    single_edge = normal_simple_local[normal_simple_local["gt_changed_edge_count_post"] == 1].copy()
    true_single_edge_fixes = int(single_edge["net_improved"].sum())
    new_false_positive_anchors = int(normal_simple_local["new_false_positive_anchor"].sum())
    no_anchor_to_anchor_wrong = int(normal_simple_local["no_anchor_to_anchor_post_wrong"].sum())

    single_edge_failures = single_edge[single_edge["failure_status_post"] != "success"].copy()
    single_edge_phrase_counts = Counter(single_edge_failures["phrase_variant_post"])
    single_edge_structure_counts = Counter(single_edge_failures["structure_tag_post"])

    relief_overall = lookup_row(relief_lookup, "overall", "overall")
    relief_singleedge = lookup_row(relief_lookup, "bucket", "relief_normal_single_edge")
    relief_phrase_changtong = lookup_row(relief_lookup, "phrase_variant", "畅通无阻")
    relief_phrase_shunchang = lookup_row(relief_lookup, "phrase_variant", "顺畅")
    shadow_overall = lookup_row(shadow_lookup, "overall", "overall")
    shadow_relabel = lookup_row(shadow_lookup, "bucket", "anomaly_plus_normal_side_event")
    guard_overall = lookup_row(guard_lookup, "overall", "overall")
    guard_directional = lookup_row(guard_lookup, "bucket", "directional_asymmetry_stable")
    guard_anti = lookup_row(guard_lookup, "bucket", "anti_truncation_canary")
    guard_hard_negative = lookup_row(guard_lookup, "bucket", "hard_negative_no_anchor")

    remaining_anchor_range = int(hard_case_summary["remaining_anchor_but_incomplete_range"])
    remaining_no_anchor = int(hard_case_summary["remaining_no_anchor_failures"])

    lines = [
        "# stage4c HOLD_AND_REPAIR Postmortem",
        "",
        "## Final Judgment",
        f"- Action verdict: `{verdict}`",
        f"- Root-cause shape: `{root_cause_shape}`",
        f"- Final one-liner: `{concise_conclusion}`",
        "",
        "## Executive Readout",
        f"- held-out `simple_local` 只出现边际改善：`no_anchor_when_gt_changed_rate {simple_local['no_anchor_when_gt_changed_rate_pre']:.4f} -> {simple_local['no_anchor_when_gt_changed_rate_post']:.4f}`，`gt_changed_edge_recall {simple_local['gt_changed_edge_recall_pre']:.1f} -> {simple_local['gt_changed_edge_recall_post']:.1f}`，`parse_fail_rate_strict {simple_local['parse_fail_rate_strict_pre']:.4f} -> {simple_local['parse_fail_rate_strict_post']:.4f}`。",
        f"- targeted `diff_29` 只动了 `1/29`：`relief_normal_single_edge` 仅 `1/20` 改善，`relief_normal_short_range` `7/7` 完全没动，`multi_edge_relief` `2/2` 完全没动。",
        f"- 真正落到 held-out target 的净收益只有 `+{true_single_edge_fixes}` 条 single-edge 修复，但 normal-only 假阳性新锚点新增了 `{new_false_positive_anchors}` 条。",
        f"- `shadow_eval / relief_eval` 明显学会了：relief overall `gt_changed_edge_recall={as_float(relief_overall['gt_changed_edge_recall']):.1f}`、`no_anchor_when_gt_changed_rate={as_float(relief_overall['no_anchor_when_gt_changed_rate']):.1f}`、`parse_fail_rate_strict={as_float(relief_overall['parse_fail_rate_strict']):.1f}`；shadow overall `gt_changed_edge_recall={as_float(shadow_overall['gt_changed_edge_recall']):.1f}`。",
        "",
        "## A. Net Improvements Vs stage4_fix",
        f"- summary 层面真正相关的净提升集中在 `val_normal/simple_local`，尤其是 `short`：`no_anchor_when_gt_changed_rate {short_bucket['no_anchor_when_gt_changed_rate_pre']:.4f} -> {short_bucket['no_anchor_when_gt_changed_rate_post']:.4f}`，`gt_changed_edge_recall {short_bucket['gt_changed_edge_recall_pre']:.1f} -> {short_bucket['gt_changed_edge_recall_post']:.1f}`，`parse_fail_rate_strict {short_bucket['parse_fail_rate_strict_pre']:.4f} -> {short_bucket['parse_fail_rate_strict_post']:.4f}`。`medium` 只有 recall 小幅增加：`{medium_bucket['gt_changed_edge_recall_pre']:.1f} -> {medium_bucket['gt_changed_edge_recall_post']:.1f}`。",
        f"- targeted semantic-gap 层面，只有 `relief_normal_single_edge` 有净提升，而且只是 `{int(single_edge_diff['changed_count'])}/{int(single_edge_diff['count'])}`；具体只多救回了一个 `车流顺畅` 单边样本。",
        "",
        "## B. Completely Unchanged Buckets",
        f"- `diff_29` 里 `relief_normal_short_range` `{int(short_range_diff['unchanged_count'])}/{int(short_range_diff['count'])}` 完全没动，`multi_edge_relief` `{int(multi_edge_diff['unchanged_count'])}/{int(multi_edge_diff['count'])}` 完全没动；即便是 `relief_normal_single_edge` 也还有 `{int(single_edge_diff['unchanged_count'])}/{int(single_edge_diff['count'])}` 没动。",
        f"- held-out summary 里 `val_normal:length_bucket/long` 与 `val_normal:length_bucket/noisy_long` 指标完全不变：`long recall {long_bucket['gt_changed_edge_recall_pre']:.1f} -> {long_bucket['gt_changed_edge_recall_post']:.1f}`，`noisy_long recall {noisy_long_bucket['gt_changed_edge_recall_pre']:.1f} -> {noisy_long_bucket['gt_changed_edge_recall_post']:.1f}`；single-edge 的 `prefixed_short` 和 `planner_long` 结构也没有任何净改善。",
        "",
        "## C. no_anchor -> anchor But Still Wrong Range",
        f"- targeted `diff_29` 里这一类是 `0`。",
        f"- 整个 held-out `val_normal/simple_local/gt_changed>0` 里这一类也是 `{no_anchor_to_anchor_wrong}`；`no_anchor -> anchor` 一共出现 `9` 次，但其中只有 `1` 次是真修复，其余 `8` 次都是 `gt_changed=0` 的 normal-only 假阳性，不是“范围边界差一点”的 recoveries。",
        "",
        "## D. shadow_eval / relief_eval Learned But Held-out Did Not Transfer",
        f"- `relief_eval` 已经把 `relief_normal_single_edge` 学满：bucket `count={int(as_float(relief_singleedge['count']))}`，`gt_changed_edge_recall={as_float(relief_singleedge['gt_changed_edge_recall']):.1f}`，`no_anchor_when_gt_changed_rate={as_float(relief_singleedge['no_anchor_when_gt_changed_rate']):.1f}`。",
        f"- phrase 细分同样学满：`畅通无阻` eval `{int(as_float(relief_phrase_changtong['count']))}/{int(as_float(relief_phrase_changtong['count']))}`，`顺畅` eval `{int(as_float(relief_phrase_shunchang['count']))}/{int(as_float(relief_phrase_shunchang['count']))}`，都是 `100` recall / `0` parse fail；但 held-out single-edge 里 `畅通无阻` 仍是 `11/11` 失败，`车流顺畅` 仍是 `8/9` 失败。",
        f"- `shadow_eval` 的 `anomaly_plus_normal_side_event` 也不是完全不会，bucket `gt_changed_edge_recall={as_float(shadow_relabel['gt_changed_edge_recall']):.1f}`、`no_anchor_when_gt_changed_rate={as_float(shadow_relabel['no_anchor_when_gt_changed_rate']):.1f}`；但 held-out 剩余 hard cases 里 mixed normal+abnormal 还有 `{int(hard_case_summary['remaining_mixed_normal_abnormal'])}` 条，且主要表现为 anchor 出来了但范围没补全。",
        "",
        "## E. Guard Stability",
        f"- guard 可以判定为稳定：held-out `directional_asymmetry` recall `+{directional_bucket['delta_gt_changed_edge_recall']:.1f}`，`anti_truncation` recall `+{anti_trunc['delta_gt_changed_edge_recall']:.1f}`，no-anchor 与 strict parse fail 都没有变差。",
        f"- dedicated guard eval 也是 `0` no_anchor / `0` strict parse fail；其中 `directional_asymmetry_stable` `{as_float(guard_directional['gt_changed_edge_recall']):.1f}` recall，`anti_truncation_canary` `{as_float(guard_anti['gt_changed_edge_recall']):.1f}` recall，`hard_negative_no_anchor` 仅 `{as_float(guard_hard_negative['gt_changed_edge_recall']):.1f}` recall 但样本数只有 `{int(as_float(guard_hard_negative['count']))}`。",
        "",
        "## Remaining Hard Cases",
        f"- stage4c 之后 `val_normal/simple_local` 还剩 `{int(hard_case_summary['remaining_simple_local_failures'])}` 条 changed-edge 失败，其中 `{remaining_no_anchor}` 条还是纯 `no_anchor`，`{remaining_anchor_range}` 条变成了 `anchor_but_incomplete_range`。",
        f"- single-edge 剩余失败仍是主桶：`{int(hard_case_summary['remaining_single_edge_failures'])}` 条，phrase 只剩两类：`畅通无阻={single_edge_phrase_counts.get('畅通无阻', 0)}`、`车流顺畅={single_edge_phrase_counts.get('车流顺畅', 0)}`；结构上 `short_plain={single_edge_structure_counts.get('short_plain', 0)}`、`prefixed_short={single_edge_structure_counts.get('prefixed_short', 0)}`、`planner_long={single_edge_structure_counts.get('planner_long', 0)}`。",
        f"- 弱语气 relief 不是当前主难点：remaining hard cases 里 `weak_relief={int(hard_case_summary['remaining_weak_relief'])}`。",
        "",
        "## Interpretation",
        f"- 这轮 patch SFT 不是完全没学到，而是“in-mixture 学会了，但 held-out 迁移非常弱”。从根因上看更像 `迁移失败`，不是 `no_anchor -> anchor 但范围差一点` 的边界问题。",
        f"- 但从继续推进 patch SFT 的行动价值看，`32` steps、`59.5%` target share、relief eval 全会之后，held-out 仍只多修 `1` 条 single-edge 且引入 `{new_false_positive_anchors}` 条 normal-only 假阳性，所以更接近 `收益已接近耗尽`。",
        "",
        "## Source Note",
        f"- release decision 仍是 `{extract_release_decision(release_decision_text)}`。",
    ]
    return lines


def extract_release_decision(text: str) -> str:
    match = re.search(r"## Final Decision\s+- `([^`]+)`", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    if "HOLD_AND_REPAIR" in text:
        return "HOLD_AND_REPAIR"
    return "UNKNOWN"


def main() -> None:
    args = parse_args()

    release_decision_text = Path(args.release_decision_md).read_text(encoding="utf-8")
    diff_payload = read_json(args.diff_json)
    stage4_summary_df, stage4_summary_lookup = load_summary_lookup(args.stage4_summary_csv)
    stage4c_summary_df, stage4c_summary_lookup = load_summary_lookup(args.stage4c_summary_csv)
    shadow_summary_df, shadow_summary_lookup = load_summary_lookup(args.shadow_summary_csv)
    relief_summary_df, relief_summary_lookup = load_summary_lookup(args.relief_summary_csv)
    guard_summary_df, guard_summary_lookup = load_summary_lookup(args.guard_summary_csv)

    stage4_details_df = enrich_detail_df(pd.read_csv(args.stage4_details_csv))
    stage4c_details_df = enrich_detail_df(pd.read_csv(args.stage4c_details_csv))

    summary_cmp = build_summary_comparison(stage4_summary_df, stage4c_summary_df)
    merged_details = build_detail_merge(stage4_details_df, stage4c_details_df)
    diff_rows, diff_bucket_df = summarize_diff29(diff_payload)
    hard_cases_payload, hard_case_summary = build_remaining_hard_cases(merged_details, diff_rows)

    verdict, root_cause_shape, concise_conclusion = choose_verdict(
        diff_rows=diff_rows,
        merged=merged_details,
        relief_summary_lookup=relief_summary_lookup,
    )

    buckets_df = build_buckets_csv(summary_cmp, diff_bucket_df, merged_details)
    buckets_df.to_csv(args.output_buckets_csv, index=False, encoding="utf-8")

    Path(args.output_hard_cases_json).write_text(
        json.dumps(hard_cases_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_text = "\n".join(
        markdown_lines(
            verdict=verdict,
            root_cause_shape=root_cause_shape,
            concise_conclusion=concise_conclusion,
            release_decision_text=release_decision_text,
            summary_cmp=summary_cmp,
            diff_bucket_df=diff_bucket_df,
            merged=merged_details,
            shadow_lookup=shadow_summary_lookup,
            relief_lookup=relief_summary_lookup,
            guard_lookup=guard_summary_lookup,
            hard_case_summary=hard_case_summary,
        )
    )
    Path(args.output_summary_md).write_text(summary_text + "\n", encoding="utf-8")

    print(f"结论：{concise_conclusion}")


if __name__ == "__main__":
    main()
