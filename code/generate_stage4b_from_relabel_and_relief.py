#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stage4b_common import (
    BASE_COLUMNS,
    LENGTH_RANK,
    OUTPUT_EXTRA_COLUMNS,
    PACK_COLUMNS,
    RELIEF_BUCKETS,
    anchor_style,
    changed_edge_count,
    clean_text,
    get_changed_events,
    has_explicit_anchor_label,
    has_mixed_abnormal_and_neutral,
    infer_simple_local_bucket,
    read_parquet_rows,
    sample_id_from_split,
    severity_from_edge_count,
    summarize_anchor_styles,
    summarize_counter,
    validate_policy_text,
    validate_simple_local_policy,
    write_parquet_rows,
)


TARGET_RELABEL_COMPONENT = "anomaly_plus_normal_side_event"
COMPONENT_A = "relabeled_simple_local_repair"
COMPONENT_B = "relief_supervision_pack"
COMPONENT_C = "anti_regression_guard"
GUARD_BUCKETS = [
    "directional_asymmetry_stable",
    "anti_truncation_guard",
    "hard_negative_no_anchor",
    "existing_simple_local_buffer",
]
GUARD_QUOTAS = {
    "train": {
        "directional_asymmetry_stable": 16,
        "anti_truncation_guard": 12,
        "hard_negative_no_anchor": 10,
        "existing_simple_local_buffer": 10,
    },
    "eval": {
        "directional_asymmetry_stable": 8,
        "anti_truncation_guard": 4,
        "hard_negative_no_anchor": 4,
        "existing_simple_local_buffer": 4,
    },
}
REQUIRED_BUCKET_EXAMPLES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stage4b from relabel repair + relief supervision + anti-regression guard.")
    parser.add_argument("--stage4-train", default="/root/autodl-tmp/dataset_sparse_v2_stage4/train")
    parser.add_argument("--stage4-eval", default="/root/autodl-tmp/dataset_sparse_v2_stage4/eval")
    parser.add_argument("--relabel-train", default="/root/autodl-tmp/dataset_sparse_v2_stage4b_relabel/train/data.parquet")
    parser.add_argument("--relabel-eval", default="/root/autodl-tmp/dataset_sparse_v2_stage4b_relabel/eval/data.parquet")
    parser.add_argument("--relabel-mapping", default="/root/autodl-tmp/dataset_sparse_v2_stage4b_relabel/relabel_mapping.csv")
    parser.add_argument("--relief-train", default="/root/autodl-tmp/dataset_simple_local_relief_pack/train/data.parquet")
    parser.add_argument("--relief-eval", default="/root/autodl-tmp/dataset_simple_local_relief_pack/eval/data.parquet")
    parser.add_argument("--policy-md", default="/root/autodl-tmp/results/simple_local_v2_label_policy.md")
    parser.add_argument("--output-root", default="/root/autodl-tmp/dataset_sparse_v2_stage4b")
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {row["sample_id"]: row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("relabel_mapping.csv contains duplicated sample_id values.")
    return mapping


def build_stage4_registry(
    *,
    split_name: str,
    stage4_rows: list[dict[str, Any]],
    relabel_rows: list[dict[str, Any]],
    mapping: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    if len(stage4_rows) != len(relabel_rows):
        raise ValueError(f"{split_name}: stage4 and relabel row counts do not match.")

    registry: dict[str, dict[str, Any]] = {}
    for index, (stage4_row, relabel_row) in enumerate(zip(stage4_rows, relabel_rows, strict=True)):
        source_split = str(relabel_row.get("split", stage4_row.get("split", split_name)))
        sample_id = sample_id_from_split(source_split, index)
        events = json.loads(relabel_row["event_records_json"])
        changed_events = get_changed_events(events)
        simple_bucket = ""
        if relabel_row["scene_type"] == "simple_local":
            simple_bucket = infer_simple_local_bucket(changed_events)
            ok, reason = validate_simple_local_policy(relabel_row)
            if not ok:
                raise ValueError(f"{sample_id}: relabel simple_local policy invalid before stage4b build: {reason}")

        registry[sample_id] = {
            "sample_id": sample_id,
            "split_name": split_name,
            "stage4_row": stage4_row,
            "relabel_row": relabel_row,
            "mapping_row": mapping.get(sample_id),
            "events": events,
            "changed_events": changed_events,
            "changed_edge_count": changed_edge_count(changed_events),
            "severity": severity_from_edge_count(changed_edge_count(changed_events)),
            "simple_bucket": simple_bucket,
            "has_explicit_anchor": has_explicit_anchor_label(str(relabel_row["response_text"])),
            "anchor_style": anchor_style(str(relabel_row["response_text"])),
            "has_mixed_abnormal_and_neutral": has_mixed_abnormal_and_neutral(changed_events),
            "length_rank": LENGTH_RANK.get(str(relabel_row.get("length_bucket", "")), 99),
        }
    return registry


def ordered_keys(keys: list[str], registry: dict[str, dict[str, Any]], *, unique_text: bool = True) -> list[str]:
    seen_texts: set[str] = set()
    ordered: list[str] = []
    for sample_id in keys:
        row = registry[sample_id]["relabel_row"]
        text = clean_text(str(row.get("constraint_text", "")))
        if unique_text and text in seen_texts:
            continue
        seen_texts.add(text)
        ordered.append(sample_id)
    return ordered


def take_keys(
    *,
    candidate_ids: list[str],
    registry: dict[str, dict[str, Any]],
    used_ids: set[str],
    target_count: int,
) -> list[str]:
    selected: list[str] = []
    seen_texts: set[str] = set()
    for sample_id in candidate_ids:
        if sample_id in used_ids:
            continue
        text = clean_text(str(registry[sample_id]["relabel_row"].get("constraint_text", "")))
        if text in seen_texts:
            continue
        used_ids.add(sample_id)
        seen_texts.add(text)
        selected.append(sample_id)
        if len(selected) >= target_count:
            break
    if len(selected) < target_count:
        raise ValueError(f"Unable to fill quota {target_count}; only selected {len(selected)} rows.")
    return selected


def make_output_row(
    *,
    stage4b_sample_id: str,
    stage4b_split_name: str,
    base_row: dict[str, Any],
    mixture_component: str,
    bucket: str,
    guard_focus: str,
    policy_semantics: str,
    source_dataset: str,
    source_split: str,
    source_sample_id: str,
    source_origin_split: str,
    source_origin_sample_id: str,
    relabel_bucket: str,
    relabel_changed: bool,
    stage4_label_changed: bool,
    changed_edges: int,
    severity: str,
    anchor_style_name: str,
    high_risk: bool,
    why_included: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for column in BASE_COLUMNS:
        output[column] = base_row.get(column, "")
    output["split"] = stage4b_split_name
    for column in OUTPUT_EXTRA_COLUMNS:
        output[column] = ""
    for column in PACK_COLUMNS:
        output[column] = base_row.get(column, None)
    output.update(
        {
            "stage4b_sample_id": stage4b_sample_id,
            "mixture_component": mixture_component,
            "bucket": bucket,
            "guard_focus": guard_focus,
            "policy_semantics": policy_semantics,
            "source_dataset": source_dataset,
            "source_split": source_split,
            "source_sample_id": source_sample_id,
            "source_origin_split": source_origin_split,
            "source_origin_sample_id": source_origin_sample_id,
            "relabel_bucket": relabel_bucket,
            "relabel_changed": bool(relabel_changed),
            "stage4_label_changed": bool(stage4_label_changed),
            "changed_edge_count": int(changed_edges),
            "severity": severity,
            "anchor_style": anchor_style_name,
            "high_risk": bool(high_risk),
            "why_included": why_included,
        }
    )
    return output


def build_component_a_rows(split_name: str, registry: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    stage4b_split_name = f"stage4b_{split_name}"
    candidate_ids = [
        sample_id
        for sample_id, entry in registry.items()
        if entry["relabel_row"]["scene_type"] == "simple_local"
        and entry["mapping_row"] is not None
        and entry["mapping_row"]["changed"].lower() == "true"
        and entry["mapping_row"]["relabel_bucket"] == TARGET_RELABEL_COMPONENT
    ]
    candidate_ids = ordered_keys(sorted(candidate_ids), registry, unique_text=False)
    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(candidate_ids):
        entry = registry[sample_id]
        mapping_row = entry["mapping_row"]
        rows.append(
            make_output_row(
                stage4b_sample_id=f"{stage4b_split_name}:{COMPONENT_A}:{idx}",
                stage4b_split_name=stage4b_split_name,
                base_row=entry["relabel_row"],
                mixture_component=COMPONENT_A,
                bucket=TARGET_RELABEL_COMPONENT,
                guard_focus="",
                policy_semantics="simple_local_v2_strict_aligned",
                source_dataset="dataset_sparse_v2_stage4b_relabel",
                source_split=str(entry["relabel_row"].get("split", "")),
                source_sample_id=sample_id,
                source_origin_split=str(entry["relabel_row"].get("split", "")),
                source_origin_sample_id=sample_id,
                relabel_bucket=str(mapping_row["relabel_bucket"]),
                relabel_changed=True,
                stage4_label_changed=True,
                changed_edges=int(entry["changed_edge_count"]),
                severity=str(entry["severity"]),
                anchor_style_name=str(entry["anchor_style"]),
                high_risk=False,
                why_included=str(mapping_row["relabel_reason"]),
            )
        )
    return rows, candidate_ids


def build_component_b_rows(split_name: str, relief_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage4b_split_name = f"stage4b_{split_name}"
    rows: list[dict[str, Any]] = []
    for idx, base_row in enumerate(relief_rows):
        ok, reason = validate_simple_local_policy(base_row)
        if not ok:
            raise ValueError(f"relief row {base_row.get('pack_sample_id', idx)} conflicts with policy-A: {reason}")
        changed_edges = int(base_row.get("changed_edge_count", 0))
        rows.append(
            make_output_row(
                stage4b_sample_id=f"{stage4b_split_name}:{COMPONENT_B}:{idx}",
                stage4b_split_name=stage4b_split_name,
                base_row=base_row,
                mixture_component=COMPONENT_B,
                bucket=str(base_row["pack_bucket"]),
                guard_focus="",
                policy_semantics="simple_local_v2_strict_aligned",
                source_dataset="dataset_simple_local_relief_pack",
                source_split=str(base_row.get("split", "")),
                source_sample_id=str(base_row.get("pack_sample_id", f"relief:{idx}")),
                source_origin_split=str(base_row.get("origin_split", "")),
                source_origin_sample_id=str(base_row.get("origin_sample_id", "")),
                relabel_bucket="",
                relabel_changed=False,
                stage4_label_changed=False,
                changed_edges=changed_edges,
                severity=str(base_row.get("relief_severity", severity_from_edge_count(changed_edges))),
                anchor_style_name=anchor_style(str(base_row.get("response_text", ""))),
                high_risk=str(base_row.get("pack_bucket", "")) == "multi_edge_relief",
                why_included=str(base_row.get("why_included", "")),
            )
        )
    return rows


def build_guard_rows(split_name: str, registry: dict[str, dict[str, Any]], component_a_ids: list[str]) -> list[dict[str, Any]]:
    quotas = GUARD_QUOTAS[split_name]
    used_ids = set(component_a_ids)
    stage4b_split_name = f"stage4b_{split_name}"

    stable_directional_ids = [
        sample_id
        for sample_id, entry in registry.items()
        if entry["relabel_row"]["scene_type"] == "directional_asymmetry"
        and entry["has_explicit_anchor"]
        and not entry["has_mixed_abnormal_and_neutral"]
    ]
    stable_directional_ids.sort(
        key=lambda sample_id: (
            registry[sample_id]["length_rank"],
            registry[sample_id]["changed_edge_count"],
            sample_id,
        )
    )
    selected_directional = take_keys(
        candidate_ids=stable_directional_ids,
        registry=registry,
        used_ids=used_ids,
        target_count=quotas["directional_asymmetry_stable"],
    )

    hard_negative_ids = [
        sample_id
        for sample_id, entry in registry.items()
        if entry["relabel_row"]["scene_type"] == "directional_asymmetry"
        and entry["has_explicit_anchor"]
        and entry["has_mixed_abnormal_and_neutral"]
    ]
    hard_negative_ids.sort(
        key=lambda sample_id: (
            registry[sample_id]["length_rank"],
            -registry[sample_id]["changed_edge_count"],
            sample_id,
        )
    )
    selected_hard_negative = take_keys(
        candidate_ids=hard_negative_ids,
        registry=registry,
        used_ids=used_ids,
        target_count=quotas["hard_negative_no_anchor"],
    )

    anti_truncation_ids = [
        sample_id
        for sample_id, entry in registry.items()
        if entry["length_rank"] >= LENGTH_RANK["long"]
        and entry["has_explicit_anchor"]
        and entry["relabel_row"]["scene_type"] in {"simple_local", "directional_asymmetry"}
        and (
            entry["relabel_row"]["scene_type"] != "simple_local"
            or entry["simple_bucket"] == "anomaly_only"
        )
    ]
    anti_truncation_ids.sort(
        key=lambda sample_id: (
            0 if registry[sample_id]["relabel_row"]["length_bucket"] == "noisy_long" else 1,
            0 if registry[sample_id]["relabel_row"]["scene_type"] == "directional_asymmetry" else 1,
            -registry[sample_id]["changed_edge_count"],
            sample_id,
        )
    )
    selected_anti_truncation = take_keys(
        candidate_ids=anti_truncation_ids,
        registry=registry,
        used_ids=used_ids,
        target_count=quotas["anti_truncation_guard"],
    )

    simple_buffer_ids = [
        sample_id
        for sample_id, entry in registry.items()
        if entry["relabel_row"]["scene_type"] == "simple_local"
        and entry["simple_bucket"] == "no_changed"
        and entry["anchor_style"] == "NO_ANCHOR"
    ]
    simple_buffer_ids.sort(
        key=lambda sample_id: (
            registry[sample_id]["length_rank"],
            sample_id,
        )
    )
    selected_simple_buffer = take_keys(
        candidate_ids=simple_buffer_ids,
        registry=registry,
        used_ids=used_ids,
        target_count=quotas["existing_simple_local_buffer"],
    )

    selected_plan = [
        (
            "directional_asymmetry_stable",
            selected_directional,
            "稳定的 directional_asymmetry 显式锚点样本，用于抑制方向区分能力回退。",
        ),
        (
            "hard_negative_no_anchor",
            selected_hard_negative,
            "保留旧 stage4 中最容易触发 no_anchor 回退的方向不对称 mixed 结构，但不与新的 simple_local_v2 语义冲突。",
        ),
        (
            "anti_truncation_guard",
            selected_anti_truncation,
            "选择 long/noisy_long 的非 temporal_switch 显式锚点样本，作为 anti-truncation 保守 guard。",
        ),
        (
            "existing_simple_local_buffer",
            selected_simple_buffer,
            "少量 existing_simple_local 非 relief 正常样本作为分布缓冲，同时继续把 NO_ANCHOR 占比压低。",
        ),
    ]

    rows: list[dict[str, Any]] = []
    running_index = 0
    for bucket, sample_ids, why_template in selected_plan:
        for sample_id in sample_ids:
            entry = registry[sample_id]
            policy_semantics = (
                "simple_local_v2_strict_aligned"
                if entry["relabel_row"]["scene_type"] == "simple_local"
                else "stage4_directional_guard"
            )
            rows.append(
                make_output_row(
                    stage4b_sample_id=f"{stage4b_split_name}:{COMPONENT_C}:{running_index}",
                    stage4b_split_name=stage4b_split_name,
                    base_row=entry["relabel_row"],
                    mixture_component=COMPONENT_C,
                    bucket=bucket,
                    guard_focus=bucket,
                    policy_semantics=policy_semantics,
                    source_dataset="dataset_sparse_v2_stage4b_relabel",
                    source_split=str(entry["relabel_row"].get("split", "")),
                    source_sample_id=sample_id,
                    source_origin_split=str(entry["relabel_row"].get("split", "")),
                    source_origin_sample_id=sample_id,
                    relabel_bucket=str(entry["mapping_row"]["relabel_bucket"]) if entry["mapping_row"] else "",
                    relabel_changed=bool(entry["mapping_row"] and entry["mapping_row"]["changed"].lower() == "true"),
                    stage4_label_changed=bool(entry["mapping_row"] and entry["mapping_row"]["changed"].lower() == "true"),
                    changed_edges=int(entry["changed_edge_count"]),
                    severity=str(entry["severity"]),
                    anchor_style_name=str(entry["anchor_style"]),
                    high_risk=False,
                    why_included=why_template,
                )
            )
            running_index += 1
    return rows


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    relief_bucket_counts = Counter(
        str(row["bucket"])
        for row in rows
        if str(row.get("bucket", "")) in RELIEF_BUCKETS
    )
    return {
        "total_rows": len(rows),
        "mixture_component_counts": summarize_counter(rows, "mixture_component"),
        "mixture_component_share": {
            key: round(value / max(len(rows), 1), 4)
            for key, value in Counter(str(row["mixture_component"]) for row in rows).items()
        },
        "bucket_counts": summarize_counter(rows, "bucket"),
        "scene_type_counts": summarize_counter(rows, "scene_type"),
        "guard_focus_counts": {
            key: value
            for key, value in summarize_counter([row for row in rows if row.get("guard_focus")], "guard_focus").items()
            if key
        },
        "relief_bucket_counts": dict(sorted(relief_bucket_counts.items())),
        "anchor_style_counts": summarize_anchor_styles(rows),
        "severity_counts": summarize_counter(rows, "severity"),
        "length_bucket_counts": summarize_counter(rows, "length_bucket"),
        "high_risk_bucket_count": sum(1 for row in rows if bool(row.get("high_risk"))),
        "no_anchor_count": sum(1 for row in rows if str(row.get("anchor_style")) == "NO_ANCHOR"),
        "explicit_anchor_count": sum(1 for row in rows if str(row.get("anchor_style")) == "explicit_anchor"),
    }


def build_mixture_breakdown_rows(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    breakdown_rows: list[dict[str, Any]] = []
    for split_name, rows in (("train", train_rows), ("eval", eval_rows)):
        total = max(len(rows), 1)
        scopes = {
            "mixture_component": Counter(str(row["mixture_component"]) for row in rows),
            "bucket": Counter(str(row["bucket"]) for row in rows),
            "scene_type": Counter(str(row["scene_type"]) for row in rows),
            "anchor_style": Counter(str(row["anchor_style"]) for row in rows),
            "severity": Counter(str(row["severity"]) for row in rows),
            "guard_focus": Counter(str(row["guard_focus"]) for row in rows if row.get("guard_focus")),
        }
        for scope, counter in scopes.items():
            for key, count in sorted(counter.items()):
                breakdown_rows.append(
                    {
                        "split": split_name,
                        "scope": scope,
                        "value": key,
                        "count": count,
                        "share_of_split": round(count / total, 4),
                    }
                )
        combo_counter = Counter((str(row["mixture_component"]), str(row["bucket"])) for row in rows)
        for (component, bucket), count in sorted(combo_counter.items()):
            breakdown_rows.append(
                {
                    "split": split_name,
                    "scope": "mixture_component_bucket",
                    "value": f"{component}::{bucket}",
                    "count": count,
                    "share_of_split": round(count / total, 4),
                }
            )
    return breakdown_rows


def build_examples(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    examples_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = str(row["bucket"])
        if bucket not in {TARGET_RELABEL_COMPONENT, *RELIEF_BUCKETS, *GUARD_BUCKETS}:
            continue
        if len(examples_by_bucket[bucket]) >= REQUIRED_BUCKET_EXAMPLES:
            continue
        examples_by_bucket[bucket].append(
            {
                "stage4b_sample_id": row["stage4b_sample_id"],
                "text": clean_text(str(row["constraint_text"])),
                "label": clean_text(str(row["response_text"])),
                "source_split": str(row["source_split"]),
                "mixture_component": str(row["mixture_component"]),
                "bucket": bucket,
                "why_included": str(row["why_included"]),
            }
        )
    missing = [bucket for bucket in [TARGET_RELABEL_COMPONENT, *RELIEF_BUCKETS, *GUARD_BUCKETS] if len(examples_by_bucket.get(bucket, [])) < REQUIRED_BUCKET_EXAMPLES]
    if missing:
        raise ValueError(f"examples.json bucket coverage is insufficient: {missing}")
    return {bucket: examples_by_bucket[bucket] for bucket in [TARGET_RELABEL_COMPONENT, *RELIEF_BUCKETS, *GUARD_BUCKETS]}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    policy_text = Path(args.policy_md).read_text(encoding="utf-8")
    validate_policy_text(policy_text)

    stage4_train_rows, stage4_train_path = read_parquet_rows(args.stage4_train)
    stage4_eval_rows, stage4_eval_path = read_parquet_rows(args.stage4_eval)
    relabel_train_rows, relabel_train_path = read_parquet_rows(args.relabel_train)
    relabel_eval_rows, relabel_eval_path = read_parquet_rows(args.relabel_eval)
    relief_train_rows, relief_train_path = read_parquet_rows(args.relief_train)
    relief_eval_rows, relief_eval_path = read_parquet_rows(args.relief_eval)
    relabel_mapping = load_mapping(Path(args.relabel_mapping))

    registry_train = build_stage4_registry(
        split_name="train",
        stage4_rows=stage4_train_rows,
        relabel_rows=relabel_train_rows,
        mapping=relabel_mapping,
    )
    registry_eval = build_stage4_registry(
        split_name="eval",
        stage4_rows=stage4_eval_rows,
        relabel_rows=relabel_eval_rows,
        mapping=relabel_mapping,
    )

    component_a_train, component_a_train_ids = build_component_a_rows("train", registry_train)
    component_a_eval, component_a_eval_ids = build_component_a_rows("eval", registry_eval)
    component_b_train = build_component_b_rows("train", relief_train_rows)
    component_b_eval = build_component_b_rows("eval", relief_eval_rows)
    component_c_train = build_guard_rows("train", registry_train, component_a_train_ids)
    component_c_eval = build_guard_rows("eval", registry_eval, component_a_eval_ids)

    train_rows = component_a_train + component_b_train + component_c_train
    eval_rows = component_a_eval + component_b_eval + component_c_eval

    if any(str(row.get("scene_type", "")) == "temporal_switch" for row in train_rows + eval_rows):
        raise ValueError("temporal_switch leaked into stage4b output.")
    for row in train_rows + eval_rows:
        if str(row.get("scene_type", "")) == "simple_local":
            ok, reason = validate_simple_local_policy(row)
            if not ok:
                raise ValueError(f"simple_local row {row['stage4b_sample_id']} violates policy-A: {reason}")

    write_parquet_rows(train_rows, output_root / "train")
    write_parquet_rows(eval_rows, output_root / "eval")

    breakdown_rows = build_mixture_breakdown_rows(train_rows, eval_rows)
    write_csv(breakdown_rows, output_root / "mixture_breakdown.csv")

    examples_payload = {
        "meta": {
            "required_examples_per_bucket": REQUIRED_BUCKET_EXAMPLES,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "mixture_components": [COMPONENT_A, COMPONENT_B, COMPONENT_C],
        },
        "examples_by_bucket": build_examples(train_rows + eval_rows),
    }
    (output_root / "examples.json").write_text(json.dumps(examples_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "paths": {
            "stage4_train": str(stage4_train_path),
            "stage4_eval": str(stage4_eval_path),
            "relabel_train": str(relabel_train_path),
            "relabel_eval": str(relabel_eval_path),
            "relabel_mapping": str(args.relabel_mapping),
            "relief_train": str(relief_train_path),
            "relief_eval": str(relief_eval_path),
            "policy_md": str(args.policy_md),
            "output_train": str(output_root / "train"),
            "output_eval": str(output_root / "eval"),
        },
        "constraints": {
            "temporal_switch_mixed_in": False,
            "simple_local_semantics": "simple_local_v2_strict_aligned",
            "no_anchor_controlled": True,
            "multi_edge_relief_high_risk_tagged": True,
            "pyarrow_io": True,
        },
        "old_stage4_reference": {
            "train_rows": len(stage4_train_rows),
            "eval_rows": len(stage4_eval_rows),
            "train_scene_type_counts": summarize_counter(stage4_train_rows, "scene_type"),
            "eval_scene_type_counts": summarize_counter(stage4_eval_rows, "scene_type"),
        },
        "stage4b_summary": {
            "train": summarize_split(train_rows),
            "eval": summarize_split(eval_rows),
        },
        "major_increment_sources_vs_old_stage4": {
            "train": {
                "relabeled_simple_local_repair": len(component_a_train),
                "relief_supervision_pack": len(component_b_train),
                "anti_regression_guard": len(component_c_train),
            },
            "eval": {
                "relabeled_simple_local_repair": len(component_a_eval),
                "relief_supervision_pack": len(component_b_eval),
                "anti_regression_guard": len(component_c_eval),
            },
            "notes": [
                "stage4b is a curated conservative patch set, not a superset of old stage4.",
                "relabel repair carries over only the changed anomaly_plus_normal simple_local samples.",
                "relief supervision pack is the major new strict-aligned addition for the held-out bottleneck buckets.",
                "anti_regression_guard reuses stable stage4-derived samples without mixing temporal_switch.",
            ],
        },
        "component_ratio_check": {
            "train": {
                COMPONENT_A: round(len(component_a_train) / len(train_rows), 4),
                COMPONENT_B: round(len(component_b_train) / len(train_rows), 4),
                COMPONENT_C: round(len(component_c_train) / len(train_rows), 4),
            },
            "eval": {
                COMPONENT_A: round(len(component_a_eval) / len(eval_rows), 4),
                COMPONENT_B: round(len(component_b_eval) / len(eval_rows), 4),
                COMPONENT_C: round(len(component_c_eval) / len(eval_rows), 4),
            },
        },
        "guard_bucket_quota": GUARD_QUOTAS,
        "high_risk_bucket": "multi_edge_relief",
        "boundary_risk_note": "multi_edge_relief is kept with explicit high-risk tagging because full-range boundary coverage remains the most fragile tail behavior.",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"stage4b train={len(train_rows)} eval={len(eval_rows)} "
        f"| ratios train=({len(component_a_train)}/{len(train_rows)}, {len(component_b_train)}/{len(train_rows)}, {len(component_c_train)}/{len(train_rows)}) "
        f"| high_risk_multi_edge={sum(1 for row in train_rows + eval_rows if bool(row.get('high_risk')))}"
    )
    print(f"output_root={output_root}")


if __name__ == "__main__":
    main()
