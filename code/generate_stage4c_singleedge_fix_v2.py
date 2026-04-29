#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from stage4b_common import (
    LENGTH_RANK,
    anchor_style,
    changed_edge_count,
    clean_text,
    get_changed_events,
    has_explicit_anchor_label,
    has_mixed_abnormal_and_neutral,
    infer_simple_local_bucket,
    read_parquet_rows,
    safe_json_loads,
    sample_id_from_split,
    severity_from_edge_count,
    validate_policy_text,
    validate_simple_local_policy,
    write_parquet_rows,
)


COMPONENT_SINGLEEDGE = "singleedge_relief_core"
COMPONENT_SHORT_RANGE = "short_range_relief_support"
COMPONENT_RELABEL = "relabeled_anomaly_plus_normal_repair"
COMPONENT_GUARD = "guard_minimal"
COMPONENT_MULTI_EDGE = "multi_edge_relief_tail"
COMPONENT_ORDER = [
    COMPONENT_SINGLEEDGE,
    COMPONENT_SHORT_RANGE,
    COMPONENT_RELABEL,
    COMPONENT_GUARD,
    COMPONENT_MULTI_EDGE,
]

EDGE_PROFILE_SINGLE = "single_edge"
EDGE_PROFILE_SHORT = "short_range"
EDGE_PROFILE_MULTI = "multi_edge"
EDGE_PROFILE_MIXED = "mixed_or_other"

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"

SHORT_RANGE_BUCKET = "relief_normal_short_range"
SINGLEEDGE_BUCKET = "relief_normal_single_edge"
MULTI_EDGE_BUCKET = "multi_edge_relief"
ANOMALY_PLUS_NORMAL_BUCKET = "anomaly_plus_normal_side_event"

TARGET_COMPONENT_COUNTS = {
    "train": {
        COMPONENT_SINGLEEDGE: 91,
        COMPONENT_SHORT_RANGE: 23,
        COMPONENT_RELABEL: 16,
        COMPONENT_GUARD: 16,
        COMPONENT_MULTI_EDGE: 7,
    },
    "eval": {
        COMPONENT_SINGLEEDGE: 34,
        COMPONENT_SHORT_RANGE: 10,
        COMPONENT_RELABEL: 6,
        COMPONENT_GUARD: 7,
        COMPONENT_MULTI_EDGE: 3,
    },
}

COMPONENT_SHARE_BOUNDS = {
    COMPONENT_SINGLEEDGE: (0.55, 0.65),
    COMPONENT_SHORT_RANGE: (0.15, 0.20),
    COMPONENT_RELABEL: (0.10, 0.15),
    COMPONENT_GUARD: (0.10, 0.15),
    COMPONENT_MULTI_EDGE: (0.00, 0.05),
}

ANOMALY_LENGTH_QUOTAS = {
    "train": {"short": 8, "medium": 5, "longish": 3},
    "eval": {"short": 3, "medium": 2, "longish": 1},
}

GUARD_BUCKET_QUOTAS = {
    "train": {
        "directional_asymmetry_stable": 8,
        "anti_truncation_canary": 4,
        "hard_negative_no_anchor": 4,
    },
    "eval": {
        "directional_asymmetry_stable": 4,
        "anti_truncation_canary": 2,
        "hard_negative_no_anchor": 1,
    },
}

MIN_EXAMPLES_PER_COMPONENT = 10
OUTPUT_SPLIT_PREFIX = "stage4c_singleedge_fix_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a narrower stage4c dataset where single-edge is the absolute main signal.")
    parser.add_argument("--singleedge-train", default="/root/autodl-tmp/dataset_stage4c_singleedge_core/train/data.parquet")
    parser.add_argument("--singleedge-eval", default="/root/autodl-tmp/dataset_stage4c_singleedge_core/eval/data.parquet")
    parser.add_argument("--relabel-train", default="/root/autodl-tmp/dataset_sparse_v2_stage4b_relabel/train/data.parquet")
    parser.add_argument("--relabel-eval", default="/root/autodl-tmp/dataset_sparse_v2_stage4b_relabel/eval/data.parquet")
    parser.add_argument("--relief-train", default="/root/autodl-tmp/dataset_simple_local_relief_pack/train/data.parquet")
    parser.add_argument("--relief-eval", default="/root/autodl-tmp/dataset_simple_local_relief_pack/eval/data.parquet")
    parser.add_argument("--audit-md", default="/root/autodl-tmp/results/stage4c_training_signal_audit.md")
    parser.add_argument("--policy-md", default="/root/autodl-tmp/results/simple_local_v2_label_policy.md")
    parser.add_argument("--output-root", default="/root/autodl-tmp/dataset_sparse_v2_stage4c")
    return parser.parse_args()


def length_group_for_anomaly(length_bucket: str) -> str:
    if length_bucket in {"long", "noisy_long"}:
        return "longish"
    return length_bucket


def sort_key_by_length(row_or_entry: dict[str, Any]) -> tuple[Any, ...]:
    length_bucket = str(row_or_entry.get("length_bucket", ""))
    return (
        LENGTH_RANK.get(length_bucket, 99),
        clean_text(str(row_or_entry.get("constraint_text", ""))),
    )


def output_split_name(split_name: str) -> str:
    return f"{OUTPUT_SPLIT_PREFIX}_{split_name}"


def row_text(row: dict[str, Any]) -> str:
    return clean_text(str(row.get("constraint_text", "")))


def ensure_label_valid(row: dict[str, Any], row_name: str) -> None:
    style = anchor_style(str(row.get("response_text", "")))
    if style not in {"explicit_anchor", "NO_ANCHOR"}:
        raise ValueError(f"{row_name}: invalid label style {style}")


def validate_singleedge_core_label(row: dict[str, Any], row_name: str) -> None:
    ensure_label_valid(row, row_name)
    if anchor_style(str(row.get("response_text", ""))) != "explicit_anchor":
        raise ValueError(f"{row_name}: singleedge core rows must keep explicit anchors.")
    if int(row.get("anchor_count", 0) or 0) != 1:
        raise ValueError(f"{row_name}: expected anchor_count=1, got {row.get('anchor_count')}")
    if "simple_local_v2=输出所有 active changed-event anchors" not in str(row.get("response_text", "")):
        raise ValueError(f"{row_name}: missing simple_local_v2 marker in response_text.")


def compute_changed_edges(row: dict[str, Any]) -> int:
    existing = row.get("changed_edge_count")
    if existing not in {None, ""}:
        return int(existing)
    events = safe_json_loads(row.get("event_records_json", "[]"))
    if not isinstance(events, list):
        return 0
    return int(changed_edge_count(get_changed_events(events)))


def build_relabel_registry(split_name: str, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        sample_id = sample_id_from_split(str(row.get("split", split_name)), index)
        events = safe_json_loads(row.get("event_records_json", "[]"))
        changed_events = get_changed_events(events) if isinstance(events, list) else []
        simple_bucket = ""
        if str(row.get("scene_type", "")) == "simple_local":
            ok, reason = validate_simple_local_policy(row)
            if not ok:
                raise ValueError(f"{sample_id}: relabel row violates simple_local_v2 policy: {reason}")
            simple_bucket = infer_simple_local_bucket(changed_events)
        registry[sample_id] = {
            "sample_id": sample_id,
            "row": row,
            "changed_events": changed_events,
            "changed_edge_count": int(changed_edge_count(changed_events)),
            "severity": severity_from_edge_count(changed_edge_count(changed_events)),
            "simple_bucket": simple_bucket,
            "length_rank": LENGTH_RANK.get(str(row.get("length_bucket", "")), 99),
            "has_explicit_anchor": has_explicit_anchor_label(str(row.get("response_text", ""))),
            "has_mixed_abnormal_and_neutral": has_mixed_abnormal_and_neutral(changed_events),
        }
    return registry


def validate_singleedge_core_rows(rows: list[dict[str, Any]], split_name: str) -> None:
    for index, row in enumerate(rows):
        row_name = f"singleedge_core:{split_name}:{index}"
        validate_singleedge_core_label(row, row_name)
        if str(row.get("core_bucket", "")) != SINGLEEDGE_BUCKET:
            raise ValueError(f"{row_name}: unexpected core_bucket={row.get('core_bucket')}")
        if str(row.get("scene_type", "")) != "simple_local":
            raise ValueError(f"{row_name}: unexpected scene_type={row.get('scene_type')}")
        if bool(row.get("is_direct_heldout_text_copy")):
            raise ValueError(f"{row_name}: direct held-out text copy leaked into singleedge core.")
        if not clean_text(str(row.get("surface_group", ""))):
            raise ValueError(f"{row_name}: missing surface_group for singleedge phrase_variant.")


def validate_relief_pack_rows(rows: list[dict[str, Any]], split_name: str) -> None:
    for index, row in enumerate(rows):
        row_name = f"relief_pack:{split_name}:{index}"
        ensure_label_valid(row, row_name)
        ok, reason = validate_simple_local_policy(row)
        if not ok:
            raise ValueError(f"{row_name}: policy mismatch: {reason}")
        if str(row.get("scene_type", "")) != "simple_local":
            raise ValueError(f"{row_name}: unexpected scene_type={row.get('scene_type')}")


def validate_training_audit(audit_text: str) -> float:
    required_fragments = [
        "建议提高到至少 40-60% 的 exposure",
        "建议压到 10-20%",
        "single-edge 成为主信号",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in audit_text]
    if missing:
        raise ValueError(f"stage4c_training_signal_audit.md is missing expected guidance fragments: {missing}")

    match = re.search(r"guard 当前占\s*`?([0-9]+(?:\.[0-9]+)?)%`?", audit_text)
    if not match:
        raise ValueError("Unable to parse stage4b guard share from stage4c_training_signal_audit.md")
    return float(match.group(1)) / 100.0


def counter_share(counter: Counter[str], total: int) -> dict[str, float]:
    return {
        key: round(value / max(total, 1), 4)
        for key, value in sorted(counter.items())
    }


def row_source_row_id(row: dict[str, Any]) -> str:
    for key in ("core_sample_id", "pack_sample_id", "stage4c_sample_id"):
        value = clean_text(str(row.get(key, "")))
        if value:
            return value
    return clean_text(str(row.get("source_row_id", "")))


def balanced_repeat_rows(
    base_rows: list[dict[str, Any]],
    *,
    extra_count: int,
    group_key: Callable[[dict[str, Any]], str],
    sort_key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[dict[str, Any]]:
    if extra_count <= 0:
        return []

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_counts: Counter[str] = Counter()
    row_repeat_counts: Counter[str] = Counter()

    for row in sorted(base_rows, key=sort_key):
        key = group_key(row)
        groups[key].append(row)
        group_counts[key] += 1

    repeated: list[dict[str, Any]] = []
    while len(repeated) < extra_count:
        available_groups = [key for key, rows in groups.items() if rows]
        if not available_groups:
            raise ValueError("No rows available for balanced repeat selection.")
        chosen_group = min(available_groups, key=lambda key: (group_counts[key], key))
        chosen_row = min(
            groups[chosen_group],
            key=lambda row: (row_repeat_counts[row_source_row_id(row)], sort_key(row)),
        )
        repeated.append(chosen_row)
        row_repeat_counts[row_source_row_id(chosen_row)] += 1
        group_counts[chosen_group] += 1
    return repeated


def take_unique_rows(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    sort_key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for row in sorted(rows, key=sort_key):
        text = row_text(row)
        if text in seen_texts:
            continue
        selected.append(row)
        seen_texts.add(text)
        if len(selected) >= target_count:
            break
    if len(selected) < target_count:
        raise ValueError(f"Unable to take {target_count} unique rows; only found {len(selected)}.")
    return selected


def take_unique_entries(
    entries: list[dict[str, Any]],
    *,
    target_count: int,
    used_ids: set[str] | None = None,
    sort_key: Callable[[dict[str, Any]], tuple[Any, ...]] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    working_used_ids = used_ids if used_ids is not None else set()
    ordered = sorted(entries, key=sort_key) if sort_key is not None else list(entries)
    for entry in ordered:
        sample_id = str(entry["sample_id"])
        text = row_text(entry["row"])
        if sample_id in working_used_ids or text in seen_texts:
            continue
        selected.append(entry)
        working_used_ids.add(sample_id)
        seen_texts.add(text)
        if len(selected) >= target_count:
            break
    if len(selected) < target_count:
        raise ValueError(f"Unable to take {target_count} unique entries; only found {len(selected)}.")
    return selected


def select_singleedge_component(split_name: str, core_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_count = TARGET_COMPONENT_COUNTS[split_name][COMPONENT_SINGLEEDGE]
    if target_count < len(core_rows):
        raise ValueError(f"{split_name}: singleedge target {target_count} is smaller than core row count {len(core_rows)}.")

    selected: list[dict[str, Any]] = list(sorted(core_rows, key=lambda row: clean_text(str(row.get("core_sample_id", "")))))
    repeats = balanced_repeat_rows(
        selected,
        extra_count=target_count - len(selected),
        group_key=lambda row: clean_text(str(row.get("surface_group", ""))),
        sort_key=lambda row: (
            clean_text(str(row.get("surface_group", ""))),
            0 if bool(row.get("heldout_like_target")) else 1,
            -int(row.get("heldout_frequency", 0) or 0),
            clean_text(str(row.get("core_sample_id", ""))),
        ),
    )
    selected.extend(repeats)
    return selected


def select_short_range_component(split_name: str, relief_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_count = TARGET_COMPONENT_COUNTS[split_name][COMPONENT_SHORT_RANGE]
    candidates = [row for row in relief_rows if str(row.get("pack_bucket", "")) == SHORT_RANGE_BUCKET]
    base_rows = take_unique_rows(
        candidates,
        target_count=min(target_count, len(candidates)),
        sort_key=lambda row: (
            clean_text(str(row.get("pack_sample_id", ""))),
            row_text(row),
        ),
    )
    if target_count <= len(base_rows):
        return base_rows[:target_count]
    repeats = balanced_repeat_rows(
        base_rows,
        extra_count=target_count - len(base_rows),
        group_key=lambda row: clean_text(str(row.get("lexical_tag", ""))) or clean_text(str(row.get("length_bucket", ""))),
        sort_key=lambda row: (
            clean_text(str(row.get("lexical_tag", ""))),
            LENGTH_RANK.get(str(row.get("length_bucket", "")), 99),
            clean_text(str(row.get("pack_sample_id", ""))),
        ),
    )
    return base_rows + repeats


def select_multi_edge_component(split_name: str, relief_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_count = TARGET_COMPONENT_COUNTS[split_name][COMPONENT_MULTI_EDGE]
    candidates = [row for row in relief_rows if str(row.get("pack_bucket", "")) == MULTI_EDGE_BUCKET]
    return take_unique_rows(
        candidates,
        target_count=target_count,
        sort_key=lambda row: (
            clean_text(str(row.get("lexical_tag", ""))),
            0 if str(row.get("length_bucket", "")) in {"long", "noisy_long"} else 1,
            clean_text(str(row.get("pack_sample_id", ""))),
        ),
    )


def select_anomaly_plus_normal_component(split_name: str, registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    quotas = ANOMALY_LENGTH_QUOTAS[split_name]
    candidates = [
        entry
        for entry in registry.values()
        if str(entry["row"].get("scene_type", "")) == "simple_local"
        and entry["simple_bucket"] == ANOMALY_PLUS_NORMAL_BUCKET
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in candidates:
        grouped[length_group_for_anomaly(str(entry["row"].get("length_bucket", "")))].append(entry)

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for group_name, quota in quotas.items():
        group_rows = grouped.get(group_name, [])
        selected.extend(
            take_unique_entries(
                group_rows,
                target_count=quota,
                used_ids=used_ids,
                sort_key=lambda entry: (
                    LENGTH_RANK.get(str(entry["row"].get("length_bucket", "")), 99),
                    clean_text(str(entry["sample_id"])),
                ),
            )
        )
    if len(selected) != TARGET_COMPONENT_COUNTS[split_name][COMPONENT_RELABEL]:
        raise ValueError(
            f"{split_name}: anomaly_plus_normal component size mismatch. expected {TARGET_COMPONENT_COUNTS[split_name][COMPONENT_RELABEL]}, got {len(selected)}"
        )
    return selected


def select_guard_component(split_name: str, registry: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    quotas = GUARD_BUCKET_QUOTAS[split_name]
    used_ids: set[str] = set()

    stable_candidates = [
        entry
        for entry in registry.values()
        if str(entry["row"].get("scene_type", "")) == "directional_asymmetry"
        and entry["has_explicit_anchor"]
        and not entry["has_mixed_abnormal_and_neutral"]
    ]
    stable_rows = take_unique_entries(
        stable_candidates,
        target_count=quotas["directional_asymmetry_stable"],
        used_ids=used_ids,
        sort_key=lambda entry: (
            entry["length_rank"],
            clean_text(str(entry["sample_id"])),
        ),
    )

    anti_candidates = [
        entry
        for entry in registry.values()
        if entry["has_explicit_anchor"]
        and entry["length_rank"] >= LENGTH_RANK["long"]
        and str(entry["row"].get("scene_type", "")) in {"simple_local", "directional_asymmetry"}
        and (
            str(entry["row"].get("scene_type", "")) != "simple_local"
            or entry["simple_bucket"] == "anomaly_only"
        )
    ]
    anti_rows = take_unique_entries(
        anti_candidates,
        target_count=quotas["anti_truncation_canary"],
        used_ids=used_ids,
        sort_key=lambda entry: (
            0 if str(entry["row"].get("scene_type", "")) == "directional_asymmetry" else 1,
            0 if str(entry["row"].get("length_bucket", "")) == "noisy_long" else 1,
            -int(entry["changed_edge_count"]),
            clean_text(str(entry["sample_id"])),
        ),
    )

    hard_candidates = [
        entry
        for entry in registry.values()
        if str(entry["row"].get("scene_type", "")) == "directional_asymmetry"
        and entry["has_explicit_anchor"]
        and entry["has_mixed_abnormal_and_neutral"]
    ]
    hard_rows = take_unique_entries(
        hard_candidates,
        target_count=quotas["hard_negative_no_anchor"],
        used_ids=used_ids,
        sort_key=lambda entry: (
            entry["length_rank"],
            -int(entry["changed_edge_count"]),
            clean_text(str(entry["sample_id"])),
        ),
    )

    selected = {
        "directional_asymmetry_stable": stable_rows,
        "anti_truncation_canary": anti_rows,
        "hard_negative_no_anchor": hard_rows,
    }
    total = sum(len(rows) for rows in selected.values())
    if total != TARGET_COMPONENT_COUNTS[split_name][COMPONENT_GUARD]:
        raise ValueError(
            f"{split_name}: guard component size mismatch. expected {TARGET_COMPONENT_COUNTS[split_name][COMPONENT_GUARD]}, got {total}"
        )
    return selected


def make_output_row(
    *,
    split_name: str,
    component: str,
    component_index: int,
    base_row: dict[str, Any],
    bucket: str,
    phrase_variant: str,
    risk_flag: str,
    source_type: str,
    source_dataset: str,
    source_split: str,
    source_sample_id: str,
    source_row_id: str,
    sampling_mode: str,
    repeat_index: int,
    edge_profile: str,
    guard_focus: str,
    why_included: str,
) -> dict[str, Any]:
    output = dict(base_row)
    changed_edges = compute_changed_edges(base_row)
    output.update(
        {
            "split": output_split_name(split_name),
            "stage4c_sample_id": f"{output_split_name(split_name)}:{component}:{component_index}",
            "mixture_component": component,
            "bucket": bucket,
            "guard_focus": guard_focus,
            "phrase_variant": phrase_variant,
            "risk_flag": risk_flag,
            "source_type": source_type,
            "source_dataset": source_dataset,
            "source_split": source_split,
            "source_sample_id": source_sample_id,
            "source_row_id": source_row_id,
            "sampling_mode": sampling_mode,
            "repeat_index": int(repeat_index),
            "edge_profile": edge_profile,
            "anchor_style": anchor_style(str(base_row.get("response_text", ""))),
            "changed_edge_count": changed_edges,
            "severity": str(base_row.get("severity", "")) or str(base_row.get("relief_severity", "")) or severity_from_edge_count(changed_edges),
            "why_included": why_included,
            "policy_semantics": (
                "simple_local_v2_strict_aligned"
                if str(base_row.get("scene_type", "")) == "simple_local"
                else "stage4c_guard_minimal"
            ),
        }
    )
    return output


def build_split_rows(
    *,
    split_name: str,
    core_rows: list[dict[str, Any]],
    relief_rows: list[dict[str, Any]],
    relabel_registry: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_rows: list[dict[str, Any]] = []

    singleedge_rows = select_singleedge_component(split_name, core_rows)
    singleedge_repeat_counts: Counter[str] = Counter()
    for row in singleedge_rows:
        row_id = clean_text(str(row.get("core_sample_id", "")))
        sampling_mode = "base_unique" if singleedge_repeat_counts[row_id] == 0 else "phrase_boost_repeat"
        source_type = "singleedge_core_base" if sampling_mode == "base_unique" else "singleedge_core_phrase_boost"
        why = (
            "stage4c_singleedge_core 全量主集，作为 single-edge 的绝对主训练信号。"
            if sampling_mode == "base_unique"
            else "为抬高稀缺 single-edge phrase_variant 的曝光，对 core 稀缺表达做定向重复加权。"
        )
        selected_rows.append(
            make_output_row(
                split_name=split_name,
                component=COMPONENT_SINGLEEDGE,
                component_index=sum(1 for row_ in selected_rows if row_.get("mixture_component") == COMPONENT_SINGLEEDGE),
                base_row=row,
                bucket=SINGLEEDGE_BUCKET,
                phrase_variant=clean_text(str(row.get("surface_group", ""))),
                risk_flag=RISK_LOW,
                source_type=source_type,
                source_dataset="dataset_stage4c_singleedge_core",
                source_split=clean_text(str(row.get("core_split", ""))) or clean_text(str(row.get("split", ""))),
                source_sample_id=clean_text(str(row.get("source_sample_id", ""))),
                source_row_id=row_id,
                sampling_mode=sampling_mode,
                repeat_index=int(singleedge_repeat_counts[row_id]),
                edge_profile=EDGE_PROFILE_SINGLE,
                guard_focus="",
                why_included=why,
            )
        )
        singleedge_repeat_counts[row_id] += 1

    short_rows = select_short_range_component(split_name, relief_rows)
    short_repeat_counts: Counter[str] = Counter()
    for row in short_rows:
        row_id = clean_text(str(row.get("pack_sample_id", "")))
        sampling_mode = "base_unique" if short_repeat_counts[row_id] == 0 else "support_repeat"
        source_type = "short_range_relief_base" if sampling_mode == "base_unique" else "short_range_relief_repeat"
        why = (
            "从 relief_pack 抽取 short-range relief 监督，给 single-edge 主信号提供邻近范围支持。"
            if sampling_mode == "base_unique"
            else "为把 short-range support 提到 15%+，对 relief_pack short-range 做小幅均衡重复。"
        )
        selected_rows.append(
            make_output_row(
                split_name=split_name,
                component=COMPONENT_SHORT_RANGE,
                component_index=sum(1 for row_ in selected_rows if row_.get("mixture_component") == COMPONENT_SHORT_RANGE),
                base_row=row,
                bucket=SHORT_RANGE_BUCKET,
                phrase_variant="",
                risk_flag=RISK_LOW,
                source_type=source_type,
                source_dataset="dataset_simple_local_relief_pack",
                source_split=clean_text(str(row.get("split", ""))),
                source_sample_id=clean_text(str(row.get("origin_sample_id", ""))),
                source_row_id=row_id,
                sampling_mode=sampling_mode,
                repeat_index=int(short_repeat_counts[row_id]),
                edge_profile=EDGE_PROFILE_SHORT,
                guard_focus="",
                why_included=why,
            )
        )
        short_repeat_counts[row_id] += 1

    relabel_entries = select_anomaly_plus_normal_component(split_name, relabel_registry)
    for entry in relabel_entries:
        row = entry["row"]
        selected_rows.append(
            make_output_row(
                split_name=split_name,
                component=COMPONENT_RELABEL,
                component_index=sum(1 for row_ in selected_rows if row_.get("mixture_component") == COMPONENT_RELABEL),
                base_row=row,
                bucket=ANOMALY_PLUS_NORMAL_BUCKET,
                phrase_variant="",
                risk_flag=RISK_LOW,
                source_type="relabel_anomaly_plus_normal",
                source_dataset="dataset_sparse_v2_stage4b_relabel",
                source_split=clean_text(str(row.get("split", ""))),
                source_sample_id=clean_text(str(entry["sample_id"])),
                source_row_id=clean_text(str(entry["sample_id"])),
                sampling_mode="curated_subset",
                repeat_index=0,
                edge_profile=EDGE_PROFILE_MIXED,
                guard_focus="",
                why_included="保留少量 anomaly+normal 双锚点 relabel 样本，防止已修好的 side-event 锚点语义回退。",
            )
        )

    guard_entries = select_guard_component(split_name, relabel_registry)
    guard_source_type = {
        "directional_asymmetry_stable": "guard_directional_stable",
        "anti_truncation_canary": "guard_anti_truncation_canary",
        "hard_negative_no_anchor": "guard_hard_negative_no_anchor",
    }
    guard_why = {
        "directional_asymmetry_stable": "保留少量稳定 directional_asymmetry 显式锚点样本，防止方向区分能力回退。",
        "anti_truncation_canary": "只留少量 long/noisy_long anti-truncation canary，避免长上下文截断回退。",
        "hard_negative_no_anchor": "只保留极少量 hard_negative_no_anchor guard，用于压制 no-anchor 旧习惯在非目标场景回流。",
    }
    for guard_focus, entries in guard_entries.items():
        for entry in entries:
            row = entry["row"]
            selected_rows.append(
                make_output_row(
                    split_name=split_name,
                    component=COMPONENT_GUARD,
                    component_index=sum(1 for row_ in selected_rows if row_.get("mixture_component") == COMPONENT_GUARD),
                    base_row=row,
                    bucket=guard_focus,
                    phrase_variant="",
                    risk_flag=RISK_MEDIUM,
                    source_type=guard_source_type[guard_focus],
                    source_dataset="dataset_sparse_v2_stage4b_relabel",
                    source_split=clean_text(str(row.get("split", ""))),
                    source_sample_id=clean_text(str(entry["sample_id"])),
                    source_row_id=clean_text(str(entry["sample_id"])),
                    sampling_mode="minimal_guard",
                    repeat_index=0,
                    edge_profile=EDGE_PROFILE_MIXED,
                    guard_focus=guard_focus,
                    why_included=guard_why[guard_focus],
                )
            )

    multi_rows = select_multi_edge_component(split_name, relief_rows)
    for row in multi_rows:
        selected_rows.append(
            make_output_row(
                split_name=split_name,
                component=COMPONENT_MULTI_EDGE,
                component_index=sum(1 for row_ in selected_rows if row_.get("mixture_component") == COMPONENT_MULTI_EDGE),
                base_row=row,
                bucket=MULTI_EDGE_BUCKET,
                phrase_variant="",
                risk_flag=RISK_HIGH,
                source_type="multi_edge_relief_tail",
                source_dataset="dataset_simple_local_relief_pack",
                source_split=clean_text(str(row.get("split", ""))),
                source_sample_id=clean_text(str(row.get("origin_sample_id", ""))),
                source_row_id=clean_text(str(row.get("pack_sample_id", ""))),
                sampling_mode="tail_unique",
                repeat_index=0,
                edge_profile=EDGE_PROFILE_MULTI,
                guard_focus="",
                why_included="multi-edge relief 只保留很小尾巴，用来维持边界感知，但不再作为主优化目标。",
            )
        )

    return selected_rows


def validate_component_shares(rows: list[dict[str, Any]], split_name: str) -> dict[str, float]:
    total = len(rows)
    counts = Counter(str(row["mixture_component"]) for row in rows)
    shares = {component: counts[component] / max(total, 1) for component in COMPONENT_ORDER}
    for component, (lower, upper) in COMPONENT_SHARE_BOUNDS.items():
        share = shares.get(component, 0.0)
        if component == COMPONENT_MULTI_EDGE:
            if share > upper + 1e-9:
                raise ValueError(f"{split_name}: {component} share={share:.4f} exceeds max {upper:.4f}")
        elif share < lower - 1e-9 or share > upper + 1e-9:
            raise ValueError(
                f"{split_name}: {component} share={share:.4f} outside bounds [{lower:.4f}, {upper:.4f}]"
            )
    return shares


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    component_counts = Counter(str(row.get("mixture_component", "")) for row in rows)
    phrase_counts = Counter(
        clean_text(str(row.get("phrase_variant", "")))
        for row in rows
        if clean_text(str(row.get("phrase_variant", "")))
    )
    source_type_counts = Counter(str(row.get("source_type", "")) for row in rows)
    edge_profile_counts = Counter(str(row.get("edge_profile", "")) for row in rows)
    risk_flag_counts = Counter(str(row.get("risk_flag", "")) for row in rows)
    anchor_style_counts = Counter(str(row.get("anchor_style", "")) for row in rows)
    bucket_counts = Counter(str(row.get("bucket", "")) for row in rows)
    guard_focus_counts = Counter(
        str(row.get("guard_focus", ""))
        for row in rows
        if clean_text(str(row.get("guard_focus", "")))
    )

    singleedge_count = int(edge_profile_counts.get(EDGE_PROFILE_SINGLE, 0))
    short_range_count = int(edge_profile_counts.get(EDGE_PROFILE_SHORT, 0))
    multi_edge_count = int(edge_profile_counts.get(EDGE_PROFILE_MULTI, 0))
    no_anchor_count = int(anchor_style_counts.get("NO_ANCHOR", 0))

    return {
        "total_rows": total,
        "mixture_component_counts": dict(sorted(component_counts.items())),
        "mixture_component_share": counter_share(component_counts, total),
        "single_edge_count": singleedge_count,
        "short_range_count": short_range_count,
        "multi_edge_count": multi_edge_count,
        "single_edge_share": round(singleedge_count / max(total, 1), 4),
        "short_range_share": round(short_range_count / max(total, 1), 4),
        "multi_edge_share": round(multi_edge_count / max(total, 1), 4),
        "phrase_variant_counts": dict(sorted(phrase_counts.items())),
        "phrase_variant_share_within_singleedge": {
            key: round(value / max(singleedge_count, 1), 4)
            for key, value in sorted(phrase_counts.items())
        },
        "NO_ANCHOR_count": no_anchor_count,
        "NO_ANCHOR_share": round(no_anchor_count / max(total, 1), 4),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "source_type_share": counter_share(source_type_counts, total),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "guard_focus_counts": dict(sorted(guard_focus_counts.items())),
        "risk_flag_counts": dict(sorted(risk_flag_counts.items())),
        "anchor_style_counts": dict(sorted(anchor_style_counts.items())),
    }


def build_mixture_breakdown_rows(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for split_name, rows in (("train", train_rows), ("eval", eval_rows)):
        total = max(len(rows), 1)
        scopes = {
            "mixture_component": Counter(str(row.get("mixture_component", "")) for row in rows),
            "source_type": Counter(str(row.get("source_type", "")) for row in rows),
            "bucket": Counter(str(row.get("bucket", "")) for row in rows),
            "edge_profile": Counter(str(row.get("edge_profile", "")) for row in rows),
            "risk_flag": Counter(str(row.get("risk_flag", "")) for row in rows),
            "phrase_variant": Counter(clean_text(str(row.get("phrase_variant", ""))) or "__non_singleedge__" for row in rows),
        }
        for scope, counter in scopes.items():
            for value, count in sorted(counter.items()):
                output_rows.append(
                    {
                        "split": split_name,
                        "scope": scope,
                        "value": value,
                        "count": int(count),
                        "share_of_split": round(count / total, 4),
                    }
                )
    return output_rows


def select_examples(rows: list[dict[str, Any]], *, target_count: int, group_key: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda row: clean_text(str(row.get("stage4c_sample_id", "")))):
        grouped[group_key(row)].append(row)

    selected: list[dict[str, Any]] = []
    used_per_group: Counter[str] = Counter()
    indices: Counter[str] = Counter()
    while len(selected) < target_count:
        available = [key for key, items in grouped.items() if indices[key] < len(items)]
        if not available:
            break
        chosen_group = min(available, key=lambda key: (used_per_group[key], key))
        selected.append(grouped[chosen_group][indices[chosen_group]])
        indices[chosen_group] += 1
        used_per_group[chosen_group] += 1
    if len(selected) < target_count:
        raise ValueError(f"Unable to collect {target_count} representative examples; only got {len(selected)}.")
    return selected


def build_examples_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_fn = {
        COMPONENT_SINGLEEDGE: lambda row: clean_text(str(row.get("phrase_variant", ""))) or "__unknown__",
        COMPONENT_SHORT_RANGE: lambda row: clean_text(str(row.get("lexical_tag", ""))) or clean_text(str(row.get("length_bucket", ""))) or "__unknown__",
        COMPONENT_RELABEL: lambda row: clean_text(str(row.get("length_bucket", ""))) or "__unknown__",
        COMPONENT_GUARD: lambda row: clean_text(str(row.get("source_type", ""))) or "__unknown__",
        COMPONENT_MULTI_EDGE: lambda row: clean_text(str(row.get("lexical_tag", ""))) or clean_text(str(row.get("length_bucket", ""))) or "__unknown__",
    }

    payload = {
        "meta": {
            "required_examples_per_component": MIN_EXAMPLES_PER_COMPONENT,
            "components": COMPONENT_ORDER,
        },
        "examples_by_component": {},
    }
    for component in COMPONENT_ORDER:
        component_rows = [row for row in rows if str(row.get("mixture_component", "")) == component]
        selected = select_examples(component_rows, target_count=MIN_EXAMPLES_PER_COMPONENT, group_key=group_fn[component])
        payload["examples_by_component"][component] = [
            {
                "stage4c_sample_id": row["stage4c_sample_id"],
                "split": row["split"],
                "text": clean_text(str(row.get("constraint_text", ""))),
                "label": clean_text(str(row.get("response_text", ""))),
                "mixture_component": component,
                "bucket": clean_text(str(row.get("bucket", ""))),
                "phrase_variant": clean_text(str(row.get("phrase_variant", ""))),
                "risk_flag": clean_text(str(row.get("risk_flag", ""))),
                "source_type": clean_text(str(row.get("source_type", ""))),
                "why_included": clean_text(str(row.get("why_included", ""))),
            }
            for row in selected
        ]
    return payload


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_final_rows(rows: list[dict[str, Any]], split_name: str) -> None:
    for row in rows:
        for key in ("mixture_component", "risk_flag", "source_type"):
            if not clean_text(str(row.get(key, ""))):
                raise ValueError(f"{split_name}: missing required field {key} in {row.get('stage4c_sample_id')}")
        if str(row.get("mixture_component", "")) == COMPONENT_SINGLEEDGE and not clean_text(str(row.get("phrase_variant", ""))):
            raise ValueError(f"{split_name}: singleedge row missing phrase_variant: {row.get('stage4c_sample_id')}")
        if str(row.get("mixture_component", "")) == COMPONENT_MULTI_EDGE and str(row.get("risk_flag", "")) != RISK_HIGH:
            raise ValueError(f"{split_name}: multi_edge row must be high risk: {row.get('stage4c_sample_id')}")
        if str(row.get("scene_type", "")) == "simple_local":
            if str(row.get("mixture_component", "")) == COMPONENT_SINGLEEDGE:
                validate_singleedge_core_label(row, f"{split_name}:{row.get('stage4c_sample_id')}")
            else:
                ok, reason = validate_simple_local_policy(row)
                if not ok:
                    raise ValueError(f"{split_name}: simple_local policy mismatch in {row.get('stage4c_sample_id')}: {reason}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    policy_text = Path(args.policy_md).read_text(encoding="utf-8")
    validate_policy_text(policy_text)

    audit_text = Path(args.audit_md).read_text(encoding="utf-8")
    stage4b_guard_share = validate_training_audit(audit_text)

    singleedge_train_rows, singleedge_train_path = read_parquet_rows(args.singleedge_train)
    singleedge_eval_rows, singleedge_eval_path = read_parquet_rows(args.singleedge_eval)
    relief_train_rows, relief_train_path = read_parquet_rows(args.relief_train)
    relief_eval_rows, relief_eval_path = read_parquet_rows(args.relief_eval)
    relabel_train_rows, relabel_train_path = read_parquet_rows(args.relabel_train)
    relabel_eval_rows, relabel_eval_path = read_parquet_rows(args.relabel_eval)

    validate_singleedge_core_rows(singleedge_train_rows, "train")
    validate_singleedge_core_rows(singleedge_eval_rows, "eval")
    validate_relief_pack_rows(relief_train_rows, "train")
    validate_relief_pack_rows(relief_eval_rows, "eval")

    relabel_train_registry = build_relabel_registry("train", relabel_train_rows)
    relabel_eval_registry = build_relabel_registry("eval", relabel_eval_rows)

    train_rows = build_split_rows(
        split_name="train",
        core_rows=singleedge_train_rows,
        relief_rows=relief_train_rows,
        relabel_registry=relabel_train_registry,
    )
    eval_rows = build_split_rows(
        split_name="eval",
        core_rows=singleedge_eval_rows,
        relief_rows=relief_eval_rows,
        relabel_registry=relabel_eval_registry,
    )

    validate_final_rows(train_rows, "train")
    validate_final_rows(eval_rows, "eval")

    train_shares = validate_component_shares(train_rows, "train")
    eval_shares = validate_component_shares(eval_rows, "eval")

    if not (150 <= len(train_rows) <= 190):
        raise ValueError(f"train total must stay in 150~190; got {len(train_rows)}")

    write_parquet_rows(train_rows, output_root / "train")
    write_parquet_rows(eval_rows, output_root / "eval")

    breakdown_rows = build_mixture_breakdown_rows(train_rows, eval_rows)
    write_csv(breakdown_rows, output_root / "mixture_breakdown.csv")

    examples_payload = build_examples_payload(train_rows + eval_rows)
    (output_root / "examples.json").write_text(
        json.dumps(examples_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_summary = summarize_rows(train_rows)
    eval_summary = summarize_rows(eval_rows)
    overall_summary = summarize_rows(train_rows + eval_rows)

    train_singleedge_share = train_summary["mixture_component_share"].get(COMPONENT_SINGLEEDGE, 0.0)
    train_guard_share = train_summary["mixture_component_share"].get(COMPONENT_GUARD, 0.0)
    singleedge_absolute_main = (
        train_summary["mixture_component_counts"].get(COMPONENT_SINGLEEDGE, 0)
        > sum(
            count
            for component, count in train_summary["mixture_component_counts"].items()
            if component != COMPONENT_SINGLEEDGE
        )
    )
    guard_clearly_lower = train_guard_share + 0.10 <= stage4b_guard_share

    summary = {
        "paths": {
            "singleedge_train": str(singleedge_train_path),
            "singleedge_eval": str(singleedge_eval_path),
            "relabel_train": str(relabel_train_path),
            "relabel_eval": str(relabel_eval_path),
            "relief_train": str(relief_train_path),
            "relief_eval": str(relief_eval_path),
            "audit_md": args.audit_md,
            "policy_md": args.policy_md,
            "output_train": str(output_root / "train"),
            "output_eval": str(output_root / "eval"),
        },
        "reference_from_audit": {
            "stage4b_guard_share": round(stage4b_guard_share, 4),
            "target_singleedge_min_share": 0.55,
            "target_guard_band": [0.10, 0.15],
        },
        "target_component_counts": TARGET_COMPONENT_COUNTS,
        "component_share_check": {
            "train": {component: round(train_shares[component], 4) for component in COMPONENT_ORDER},
            "eval": {component: round(eval_shares[component], 4) for component in COMPONENT_ORDER},
        },
        "total_rows": {
            "train": len(train_rows),
            "eval": len(eval_rows),
            "overall": len(train_rows) + len(eval_rows),
        },
        "split_summary": {
            "train": train_summary,
            "eval": eval_summary,
            "overall": overall_summary,
        },
        "final_checks": {
            "singleedge_exposure_ge_55pct": bool(train_singleedge_share >= 0.55),
            "guard_clearly_lower_than_stage4b": bool(guard_clearly_lower),
            "singleedge_is_absolute_main_signal": bool(singleedge_absolute_main and train_singleedge_share >= 0.55),
            "train_total_in_preferred_band": bool(150 <= len(train_rows) <= 190),
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"train_rows={len(train_rows)} eval_rows={len(eval_rows)}")
    print(
        "singleedge exposure 55%+ "
        f"=> {train_singleedge_share:.2%} "
        f"({'PASS' if train_singleedge_share >= 0.55 else 'FAIL'})"
    )
    print(
        "guard clearly below stage4b "
        f"=> stage4c={train_guard_share:.2%}, stage4b={stage4b_guard_share:.2%} "
        f"({'PASS' if guard_clearly_lower else 'FAIL'})"
    )
    print(
        "single-edge is absolute main signal "
        f"=> {train_summary['mixture_component_counts'].get(COMPONENT_SINGLEEDGE, 0)}/{len(train_rows)} "
        f"({'PASS' if singleedge_absolute_main and train_singleedge_share >= 0.55 else 'FAIL'})"
    )


if __name__ == "__main__":
    main()
