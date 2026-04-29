#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


NEUTRAL_LEVELS = {"正常", "畅通"}
SEMANTIC_FAILURE_TYPES = {"no_anchor", "changed_edge_miss", "target_miss"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit semantic-gap failures for simple_local strict failures.")
    parser.add_argument(
        "--details-csv",
        default="/root/autodl-tmp/results/sparse_v2_validation_strict_stage4_fix_details.csv",
    )
    parser.add_argument(
        "--examples-json",
        default="/root/autodl-tmp/results/simple_local_failure_examples.json",
    )
    parser.add_argument(
        "--output-csv",
        default="/root/autodl-tmp/results/simple_local_semantic_gap_audit.csv",
    )
    parser.add_argument(
        "--output-notes",
        default="/root/autodl-tmp/results/simple_local_semantic_gap_notes.md",
    )
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Optional override. If omitted, dataset_root is read from the examples JSON meta.",
    )
    return parser.parse_args()


def normalize_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


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


def event_edge_set(event: dict) -> set[str]:
    edges = event.get("anchor_edges", []) or []
    return {str(edge) for edge in edges if edge}


def load_examples_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("meta", {})


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
        columns=["split", "constraint_text", "response_text", "event_records_json"],
    )
    data = table.to_pydict()
    frame = pd.DataFrame(data)
    frame["split_index"] = range(len(frame))
    frame["sample_id"] = frame["split"].astype(str) + ":" + frame["split_index"].astype(str)
    return frame


def load_dataset_rows(dataset_root: Path, splits: list[str]) -> pd.DataFrame:
    frames = []
    for split in sorted(set(splits)):
        split_path = dataset_root / split / "data.parquet"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing dataset parquet for split {split}: {split_path}")
        frames.append(parquet_to_frame(split_path))
    return pd.concat(frames, ignore_index=True)


def compute_failure_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["no_anchor_when_gt_changed"] = df["no_anchor"] & (df["gt_changed_edge_count"] > 0)
    df["target_miss"] = (df["gt_target_edge_count"] > 0) & (df["target_recall"] < 100.0)
    df["changed_edge_miss"] = (df["gt_changed_edge_count"] > 0) & (df["gt_changed_edge_recall"] < 100.0)
    df["parse_fail_root"] = df["parse_fail"] | (df["strict_parse_fail"] & ~df["no_anchor_when_gt_changed"])
    df["strict_fail"] = (
        df["no_anchor_when_gt_changed"]
        | df["strict_parse_fail"]
        | df["target_miss"]
        | df["changed_edge_miss"]
    )

    def classify(row: pd.Series) -> str:
        if row["no_anchor_when_gt_changed"]:
            return "no_anchor"
        if row["parse_fail_root"]:
            return "parse_fail"
        if row["target_miss"] and row["changed_edge_miss"]:
            return "mixed"
        if row["target_miss"]:
            return "target_miss"
        if row["changed_edge_miss"]:
            return "changed_edge_miss"
        return "ok"

    df["failure_type"] = df.apply(classify, axis=1)
    return df


def has_format_issue(pred_text: str) -> bool:
    text = clean_text(pred_text)
    if not text:
        return True
    if "NO_ANCHOR" in text:
        return False
    if "ANCHOR|ROAD=" in text:
        return False
    return True


def derive_gap_subtype(row: pd.Series) -> str:
    if row["audit_bucket"] != "label_semantic_gap":
        return ""

    active_events = row["active_changed_events"]
    active_neutral = [event for event in active_events if str(event.get("level", "")) in NEUTRAL_LEVELS]
    active_abnormal = [event for event in active_events if str(event.get("level", "")) not in NEUTRAL_LEVELS]
    unanchored_events = row["unanchored_changed_events"]
    unanchored_neutral = [event for event in unanchored_events if str(event.get("level", "")) in NEUTRAL_LEVELS]
    anchored_abnormal = [
        event for event in active_abnormal if event_signature(event) in row["gt_label_clean"]
    ]
    unanchored_edge_count = row["unanchored_changed_edge_count"]

    if active_abnormal and active_neutral and anchored_abnormal and unanchored_neutral:
        return "anomaly_plus_normal_side_event"

    if active_events and len(active_events) == len(active_neutral):
        if unanchored_edge_count <= 1:
            return "relief_normal_single_edge"
        if unanchored_edge_count == 2:
            return "relief_normal_short_range"
        if unanchored_edge_count >= 3:
            return "multi_edge_relief"

    return "other"


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    divider = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, divider, *body])


def main() -> None:
    args = parse_args()
    details_csv = Path(args.details_csv)
    examples_json = Path(args.examples_json)
    output_csv = Path(args.output_csv)
    output_notes = Path(args.output_notes)

    meta = load_examples_meta(examples_json)
    dataset_root = Path(args.dataset_root or meta.get("dataset_root") or "/root/autodl-tmp/dataset_sparse_v2")

    details_df = load_details(details_csv)
    details_df = compute_failure_flags(details_df)
    details_df = details_df[details_df["strict_fail"]].copy()

    dataset_df = load_dataset_rows(dataset_root, details_df["split"].tolist())
    merged = details_df.merge(
        dataset_df[["sample_id", "constraint_text", "response_text", "event_records_json"]],
        on="sample_id",
        how="left",
        suffixes=("_details", ""),
        validate="one_to_one",
    )
    if merged[["response_text", "event_records_json"]].isna().any().any():
        missing = merged[merged["response_text"].isna()]["sample_id"].tolist()
        raise ValueError(f"Missing GT join rows for sample_ids: {missing[:5]}")

    merged["input_text"] = merged["constraint_text"].fillna(merged["constraint_text_details"]).map(clean_text)
    merged["gt_label_clean"] = merged["response_text"].map(clean_text)
    merged["pred_text_clean"] = merged["prediction_raw_preview"].map(clean_text)
    merged["pred_matches_gt_label"] = merged["pred_text_clean"] == merged["gt_label_clean"]
    merged["format_issue"] = merged["prediction_raw_preview"].map(has_format_issue)
    merged["events"] = merged["event_records_json"].map(safe_json_loads)
    merged["active_changed_events"] = merged["events"].map(
        lambda events: [
            event for event in events if event.get("active", True) and event_is_changed(event)
        ]
    )
    merged["active_changed_event_count"] = merged["active_changed_events"].map(len)
    merged["active_changed_edge_count_derived"] = merged["active_changed_events"].map(
        lambda events: len(set().union(*(event_edge_set(event) for event in events)) if events else set())
    )
    merged["unanchored_changed_events"] = merged.apply(
        lambda row: [
            event
            for event in row["active_changed_events"]
            if event_signature(event) not in row["gt_label_clean"]
        ],
        axis=1,
    )
    merged["unanchored_changed_event_count"] = merged["unanchored_changed_events"].map(len)
    merged["unanchored_changed_edge_count"] = merged["unanchored_changed_events"].map(
        lambda events: len(set().union(*(event_edge_set(event) for event in events)) if events else set())
    )
    merged["active_changed_levels"] = merged["active_changed_events"].map(
        lambda events: "|".join(event.get("level", "") for event in events)
    )
    merged["unanchored_changed_levels"] = merged["unanchored_changed_events"].map(
        lambda events: "|".join(event.get("level", "") for event in events)
    )
    merged["active_changed_event_texts"] = merged["active_changed_events"].map(
        lambda events: " || ".join(event.get("text", "") for event in events)
    )
    merged["unanchored_changed_event_texts"] = merged["unanchored_changed_events"].map(
        lambda events: " || ".join(event.get("text", "") for event in events)
    )

    def audit_bucket(row: pd.Series) -> str:
        if row["pred_matches_gt_label"] and row["failure_type"] in SEMANTIC_FAILURE_TYPES:
            return "label_semantic_gap"
        if (not row["pred_matches_gt_label"]) or row["parse_fail_root"] or row["format_issue"]:
            return "generation_or_parse_issue"
        return "mixed"

    merged["audit_bucket"] = merged.apply(audit_bucket, axis=1)
    merged["semantic_gap_subtype"] = merged.apply(derive_gap_subtype, axis=1)

    output_df = merged[
        [
            "sample_id",
            "split",
            "scene_type",
            "strict_fail",
            "failure_type",
            "audit_bucket",
            "semantic_gap_subtype",
            "pred_matches_gt_label",
            "format_issue",
            "no_anchor",
            "parse_fail",
            "strict_parse_fail",
            "strict_failure_reason",
            "no_anchor_when_gt_changed",
            "target_miss",
            "changed_edge_miss",
            "gt_changed_edge_count",
            "gt_target_edge_count",
            "active_changed_event_count",
            "active_changed_edge_count_derived",
            "unanchored_changed_event_count",
            "unanchored_changed_edge_count",
            "target_recall",
            "gt_changed_edge_recall",
            "active_changed_levels",
            "unanchored_changed_levels",
            "active_changed_event_texts",
            "unanchored_changed_event_texts",
            "input_text",
            "gt_label_clean",
            "pred_text_clean",
        ]
    ].rename(
        columns={
            "gt_label_clean": "gt_label",
            "pred_text_clean": "pred_text",
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False, encoding="utf-8")

    strict_fail_count = len(merged)
    bucket_counts = merged["audit_bucket"].value_counts()
    subtype_counts = merged[merged["audit_bucket"] == "label_semantic_gap"]["semantic_gap_subtype"].value_counts()

    label_semantic_gap_count = int(bucket_counts.get("label_semantic_gap", 0))
    generation_issue_count = int(bucket_counts.get("generation_or_parse_issue", 0))
    mixed_count = int(bucket_counts.get("mixed", 0))

    gap_df = merged[merged["audit_bucket"] == "label_semantic_gap"].copy()
    level_counter = Counter()
    level_edge_counter = Counter()
    for events in gap_df["unanchored_changed_events"]:
        for event in events:
            level = str(event.get("level", "")).strip() or "UNKNOWN"
            level_counter[level] += 1
            level_edge_counter[level] += len(event_edge_set(event)) or 1

    unanchored_type_rows = [["Type", "Samples", "Share"]]
    for subtype, count in subtype_counts.items():
        unanchored_type_rows.append(
            [subtype, str(int(count)), f"{count / strict_fail_count:.1%}"]
        )

    level_rows = [["Level", "Unanchored Events", "Approx Unanchored Edges"]]
    for level, count in level_counter.most_common():
        level_rows.append([level, str(count), str(level_edge_counter[level])])

    sample_lines = []
    for _, row in gap_df.head(5).iterrows():
        sample_lines.append(
            f"- `{row['sample_id']}` | {row['semantic_gap_subtype']} | {row['failure_type']} | "
            f"input: {row['input_text']} | GT: {row['gt_label_clean']}"
        )

    notes = [
        "# Simple Local Semantic Gap Audit",
        "",
        "## Scope",
        f"- Source details: `{details_csv}`",
        f"- Source examples: `{examples_json}`",
        f"- Dataset root resolved from examples meta: `{dataset_root}`",
        f"- Analyzed samples: `scene_type == simple_local` and strict fail, total `{strict_fail_count}`",
        "",
        "## Audit Split",
        markdown_table(
            [
                ["Bucket", "Count", "Share of strict fail"],
                ["label_semantic_gap", str(label_semantic_gap_count), f"{label_semantic_gap_count / strict_fail_count:.1%}" if strict_fail_count else "0.0%"],
                ["generation_or_parse_issue", str(generation_issue_count), f"{generation_issue_count / strict_fail_count:.1%}" if strict_fail_count else "0.0%"],
                ["mixed", str(mixed_count), f"{mixed_count / strict_fail_count:.1%}" if strict_fail_count else "0.0%"],
            ]
        ),
        "",
        "## Semantic Gap Subtypes",
        markdown_table(unanchored_type_rows),
        "",
        "## Unanchored Changed-Edge Levels",
        markdown_table(level_rows),
        "",
        "## Required Answers",
        f"- strict fail 中有多少比例其实是 label semantic gap：`{label_semantic_gap_count}/{strict_fail_count}`，即 `{label_semantic_gap_count / strict_fail_count:.1%}`。当前审计里，generation/parse 问题样本数为 `{generation_issue_count}`，说明 strict fail 几乎全部来自标签语义覆盖缺口，而不是模型没学到标签格式。",
        "- 哪些 changed-edge 类型在 GT 标签里长期未被显式锚定：",
        f"  1. 纯 relief/normal changed edges，尤其 `正常` / `畅通` 的单边样本、2-edge 短范围样本、3+ edge 多边样本。对应 subtype 主要是 `relief_normal_single_edge`、`relief_normal_short_range`、`multi_edge_relief`。",
        f"  2. `anomaly_plus_normal_side_event` 类型里，GT 标签通常只锚定异常主事件，把同时存在的正常/缓解 side event 留空，导致 changed-edge recall 丢失。",
        f"  3. 从未显式锚定的 level 以 `{', '.join(level for level, _ in level_counter.most_common())}` 为主；这些 level 在 strict 评估里仍对应 changed edges，所以会持续被判 miss。",
        "- 继续用当前标签做 SFT 的收益为什么会受限：",
        "  1. 当 `pred_text == gt_label` 仍然被 strict 判错时，继续喂同一套标签只会强化当前的漏锚策略，而不会提升 strict recall。",
        "  2. 训练目标与评估目标不一致。标签在 `simple_local` 中默认忽略 relief/normal changed edges 或 anomaly 的正常 side event，但 strict 指标把这些边都算作必须覆盖的 changed edges。",
        "  3. 结果上会出现学习饱和：模型越来越稳定地复现标签语义，但 strict 分数卡在同一个上限，尤其是 `no_anchor` 和 `changed_edge_miss` 这两类系统性失败。",
        "",
        "## Representative Semantic-Gap Samples",
        *sample_lines,
        "",
        "## Output Files",
        f"- CSV: `{output_csv}`",
        f"- Notes: `{output_notes}`",
    ]

    output_notes.parent.mkdir(parents=True, exist_ok=True)
    output_notes.write_text("\n".join(notes), encoding="utf-8")

    print(f"strict_fail_samples={strict_fail_count}")
    print(f"label_semantic_gap={label_semantic_gap_count} ({label_semantic_gap_count / strict_fail_count:.1%})")
    print(f"generation_or_parse_issue={generation_issue_count}")
    print(f"mixed={mixed_count}")
    print("semantic_gap_subtypes:")
    for subtype, count in subtype_counts.items():
        print(f"- {subtype}: {count}")
    print(f"output_csv={output_csv}")
    print(f"output_notes={output_notes}")


if __name__ == "__main__":
    main()
