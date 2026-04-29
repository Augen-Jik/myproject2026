#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from stage4b_common import (
    LENGTH_RANK,
    anchor_style,
    changed_edge_count,
    clean_text,
    get_changed_events,
    read_parquet_rows,
    render_simple_local_v2_label,
    safe_json_loads,
    validate_policy_text,
    validate_simple_local_policy,
    write_parquet_rows,
)


TARGET_BUCKET = "single_edge_changed_relief__short_plain_or_prefixed"
TARGET_POSITIVE_STRUCTURES = ("short_plain", "prefixed_short")
TARGET_POSITIVE_PHRASES = ("畅通无阻", "车流顺畅", "顺畅")
TARGET_NEGATIVE_PHRASES = ("交通基本正常", "保持正常通行")

COMPONENT_POSITIVE = "micro_patch_positive_core"
COMPONENT_NEGATIVE = "hard_negative_guard"
COMPONENT_CANARY = "minimal_stability_canary"
COMPONENT_ORDER = [COMPONENT_POSITIVE, COMPONENT_NEGATIVE, COMPONENT_CANARY]

TRAIN_COMPONENT_TARGETS = {
    COMPONENT_POSITIVE: 65,
    COMPONENT_NEGATIVE: 25,
    COMPONENT_CANARY: 10,
}
EVAL_COMPONENT_TARGETS = {
    COMPONENT_POSITIVE: 18,
    COMPONENT_NEGATIVE: 8,
    COMPONENT_CANARY: 4,
}

POSITIVE_TRAIN_CELL_QUOTAS: list[tuple[str, str, int]] = [
    ("畅通无阻", "short_plain", 12),
    ("畅通无阻", "prefixed_short", 12),
    ("车流顺畅", "short_plain", 12),
    ("车流顺畅", "prefixed_short", 12),
    ("顺畅", "short_plain", 8),
    ("顺畅", "prefixed_short", 9),
]

NEGATIVE_TRAIN_CELL_QUOTAS: list[tuple[str, str, int]] = [
    ("交通基本正常", "short_plain", 6),
    ("交通基本正常", "prefixed_short", 4),
    ("交通基本正常", "planner_long", 2),
    ("保持正常通行", "short_plain", 5),
    ("保持正常通行", "prefixed_short", 5),
    ("保持正常通行", "planner_long", 3),
]

CANARY_TRAIN_QUOTAS = {
    "directional_asymmetry": 5,
    "anti_truncation": 5,
}
CANARY_EVAL_QUOTAS = {
    "directional_asymmetry": 2,
    "anti_truncation": 2,
}

MIN_EXAMPLES_PER_COMPONENT = 10
OUTPUT_SPLIT_PREFIX = "stage4d_micro_patch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the last tiny stage4d patch dataset with one dominant bucket and hard-negative guard.")
    parser.add_argument("--remaining-hard-json", default="/root/autodl-tmp/results/stage4c_remaining_hard_cases.json")
    parser.add_argument("--priority-csv", default="/root/autodl-tmp/results/stage4c_repair_priority.csv")
    parser.add_argument("--policy-md", default="/root/autodl-tmp/results/simple_local_v2_label_policy.md")
    parser.add_argument("--core-train", default="/root/autodl-tmp/dataset_stage4c_singleedge_core/train/data.parquet")
    parser.add_argument("--core-eval", default="/root/autodl-tmp/dataset_stage4c_singleedge_core/eval/data.parquet")
    parser.add_argument("--optional-stage4-fix-train", default="/root/autodl-tmp/dataset_sparse_v2_stage4_fix/train/data.parquet")
    parser.add_argument("--optional-stage4-fix-eval", default="/root/autodl-tmp/dataset_sparse_v2_stage4_fix/eval/data.parquet")
    parser.add_argument("--original-root", default="/root/autodl-tmp/dataset_sparse_v2")
    parser.add_argument("--output-root", default="/root/autodl-tmp/dataset_sparse_v2_stage4d_micro")
    return parser.parse_args()


def output_split_name(split_name: str) -> str:
    return f"{OUTPUT_SPLIT_PREFIX}_{split_name}"


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_priority_row(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("stage4c_repair_priority.csv is empty")
    top_row = rows[0]
    if clean_text(top_row.get("bucket_name")) != TARGET_BUCKET:
        raise ValueError(
            f"Top priority bucket changed: expected {TARGET_BUCKET}, got {top_row.get('bucket_name')}"
        )
    if clean_text(top_row.get("count")) != "18":
        raise ValueError(f"Expected top priority bucket count=18, got {top_row.get('count')}")
    if clean_text(top_row.get("patchable_with_micro_patch")) != "True":
        raise ValueError("Top priority bucket is no longer marked micro-patchable.")
    return top_row


def resolve_original_paths(args: argparse.Namespace) -> dict[str, str]:
    original_root = Path(args.original_root)
    fallback_paths = {
        "train": str(original_root / "train" / "data.parquet"),
        "eval": str(original_root / "eval" / "data.parquet"),
        "val_normal": str(original_root / "val_normal" / "data.parquet"),
        "val_special": str(original_root / "val_special" / "data.parquet"),
        "val_anti_truncation": str(original_root / "val_anti_truncation" / "data.parquet"),
    }
    stage4_fix_train = Path(args.optional_stage4_fix_train)
    stage4_fix_eval = Path(args.optional_stage4_fix_eval)
    return {
        "stage4_fix_train_requested": str(stage4_fix_train),
        "stage4_fix_eval_requested": str(stage4_fix_eval),
        "stage4_fix_train_exists": str(stage4_fix_train.exists()),
        "stage4_fix_eval_exists": str(stage4_fix_eval.exists()),
        **fallback_paths,
    }


def classify_structure_variant(text: str) -> str:
    normalized = clean_text(text)
    if normalized.startswith("当前需要根据"):
        return "planner_long"
    if normalized.startswith("常态或局部短时事件：") or normalized.startswith("方向不对称约束："):
        return "prefixed_short"
    return "short_plain"


def normalize_positive_phrase(row: dict[str, Any]) -> str:
    surface_phrase = clean_text(row.get("surface_phrase"))
    surface_group = clean_text(row.get("surface_group"))
    text = clean_text(row.get("constraint_text"))

    if surface_phrase == "车流顺畅" or "车流顺畅" in text:
        return "车流顺畅"
    if surface_phrase == "畅通无阻" or "畅通无阻" in text:
        return "畅通无阻"
    if surface_phrase in {"顺畅", "通行顺畅"} or surface_group == "顺畅" or "顺畅" in text:
        return "顺畅"
    return ""


def normalize_negative_phrase(text: str) -> str:
    normalized = clean_text(text)
    if "交通基本正常" in normalized:
        return "交通基本正常"
    if "保持正常通行" in normalized:
        return "保持正常通行"
    return ""


def make_lookup_rows(path: str | Path, split_prefix: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows, resolved = read_parquet_rows(path)
    output_rows: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        entry = dict(row)
        row_id = f"{split_prefix}:{index}"
        entry["_source_row_id"] = row_id
        output_rows.append(entry)
        lookup[row_id] = entry
    return output_rows, lookup


def render_simple_local_v2_prompt(row: dict[str, Any]) -> str:
    events = safe_json_loads(row.get("event_records_json", "[]"))
    if not isinstance(events, list):
        raise ValueError("event_records_json must decode to a list")

    lines = [
        "TASK=sparse_output",
        "SCENE_TYPE=simple_local",
        "OUTPUT_OBJECT=当前生效的 changed-event anchors",
        "OUTPUT_FORMAT=ANCHOR|ROAD=道路名|DIR=方向|RANGE=范围|LEVEL=等级",
        "OUTPUT_RULE=输出所有当前生效且可定位的 changed-event anchors；包含异常主事件、畅通/顺畅/正常 relief side event；不要补全全量边权；不要解释。",
        "POLICY=simple_local_v2_strict_aligned",
        "SCHEMA=ROAD|DIR|RANGE|LEVEL|TIME|PROPAGATION|CONFLICT",
        f"NARRATIVE={clean_text(row.get('constraint_text'))}",
    ]

    for index, event in enumerate(events, start=1):
        lines.append(
            "EVENT_"
            f"{index}: ROAD={clean_text(event.get('road'))} | "
            f"DIR={clean_text(event.get('dir'))} | "
            f"RANGE={clean_text(event.get('range'))} | "
            f"LEVEL={clean_text(event.get('level'))} | "
            f"TIME={clean_text(event.get('time'))} | "
            f"PROPAGATION={clean_text(event.get('propagation'))} | "
            f"CONFLICT={clean_text(event.get('conflict'))} | "
            f"TEXT={clean_text(event.get('text'))}"
        )
    lines.append("仅输出锚点，不要解释。")
    return "\n".join(lines)


def render_messages(prompt_text: str, response_text: str) -> str:
    return json.dumps(
        [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ],
        ensure_ascii=False,
    )


def relabel_simple_local_v2(row: dict[str, Any]) -> dict[str, Any]:
    events = safe_json_loads(row.get("event_records_json", "[]"))
    if not isinstance(events, list):
        raise ValueError(f"{row.get('_source_row_id')}: event_records_json is not a list")
    changed_events = get_changed_events(events)
    prompt_text = render_simple_local_v2_prompt(row)
    response_text = render_simple_local_v2_label(changed_events)
    output = dict(row)
    output["prompt_text"] = prompt_text
    output["response_text"] = response_text
    output["messages"] = render_messages(prompt_text, response_text)
    output["anchor_count"] = len(changed_events)
    return output


def compute_changed_edge_count_from_row(row: dict[str, Any]) -> int:
    events = safe_json_loads(row.get("event_records_json", "[]"))
    if not isinstance(events, list):
        return 0
    return int(changed_edge_count(get_changed_events(events)))


def positive_bucket_match(row: dict[str, Any]) -> tuple[bool, str, str]:
    if clean_text(row.get("scene_type")) != "simple_local":
        return False, "", ""
    structure_variant = clean_text(row.get("structure_tag")) or classify_structure_variant(clean_text(row.get("constraint_text")))
    if structure_variant not in TARGET_POSITIVE_STRUCTURES:
        return False, "", ""
    phrase_variant = normalize_positive_phrase(row)
    if phrase_variant not in TARGET_POSITIVE_PHRASES:
        return False, "", ""
    events = safe_json_loads(row.get("event_records_json", "[]"))
    if not isinstance(events, list):
        return False, "", ""
    changed_events = get_changed_events(events)
    if len(changed_events) != 1:
        return False, "", ""
    if int(changed_edge_count(changed_events)) != 1:
        return False, "", ""
    if clean_text(changed_events[0].get("level")) != "畅通":
        return False, "", ""
    return True, phrase_variant, structure_variant


def hard_negative_match(row: dict[str, Any]) -> tuple[bool, str, str]:
    if clean_text(row.get("scene_type")) != "simple_local":
        return False, "", ""
    phrase_variant = normalize_negative_phrase(clean_text(row.get("constraint_text")))
    if phrase_variant not in TARGET_NEGATIVE_PHRASES:
        return False, "", ""
    structure_variant = classify_structure_variant(clean_text(row.get("constraint_text")))
    events = safe_json_loads(row.get("event_records_json", "[]"))
    if not isinstance(events, list):
        return False, "", ""
    changed_events = get_changed_events(events)
    if changed_events:
        return False, "", ""
    return True, phrase_variant, structure_variant


def build_entry(
    *,
    row: dict[str, Any],
    phrase_variant: str,
    structure_variant: str,
    source_type: str,
    source_dataset: str,
    source_split: str,
    source_sample_id: str,
    source_row_id: str,
    canary_kind: str = "",
    exact_seed_type: str = "",
) -> dict[str, Any]:
    return {
        "row": row,
        "phrase_variant": phrase_variant,
        "structure_variant": structure_variant,
        "source_type": source_type,
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_sample_id": source_sample_id,
        "source_row_id": source_row_id,
        "canary_kind": canary_kind,
        "exact_seed_type": exact_seed_type,
        "dedupe_text": clean_text(row.get("constraint_text")),
    }


def unique_entries(entries: list[dict[str, Any]], sort_key: Callable[[dict[str, Any]], tuple[Any, ...]]) -> list[dict[str, Any]]:
    seen_texts: set[str] = set()
    output: list[dict[str, Any]] = []
    for entry in sorted(entries, key=sort_key):
        text = clean_text(entry.get("dedupe_text"))
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        output.append(entry)
    return output


def take_quota_with_repeats(
    entries: list[dict[str, Any]],
    *,
    quota: int,
    sort_key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[dict[str, Any]]:
    base = unique_entries(entries, sort_key=sort_key)
    if len(base) >= quota:
        return base[:quota]
    if not base:
        raise ValueError(f"Unable to satisfy quota={quota}; no candidates available.")
    output = list(base)
    repeat_index = 0
    while len(output) < quota:
        output.append(base[repeat_index % len(base)])
        repeat_index += 1
    return output


def select_examples(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    group_key: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: clean_text(item.get("stage4d_sample_id"))):
        grouped[group_key(row)].append(row)

    selected: list[dict[str, Any]] = []
    used_per_group: Counter[str] = Counter()
    indices: Counter[str] = Counter()
    while len(selected) < target_count:
        available = [key for key, group_rows in grouped.items() if indices[key] < len(group_rows)]
        if not available:
            break
        chosen_group = min(available, key=lambda key: (used_per_group[key], key))
        selected.append(grouped[chosen_group][indices[chosen_group]])
        indices[chosen_group] += 1
        used_per_group[chosen_group] += 1
    if len(selected) < target_count:
        raise ValueError(f"Unable to collect {target_count} examples; only found {len(selected)}.")
    return selected


def counter_share(counter: Counter[str], total: int) -> dict[str, float]:
    return {key: round(value / max(total, 1), 4) for key, value in sorted(counter.items())}


def make_output_row(
    *,
    split_name: str,
    component: str,
    component_index: int,
    entry: dict[str, Any],
    repeat_index: int,
    is_hard_negative: bool,
) -> dict[str, Any]:
    row = dict(entry["row"])
    row.update(
        {
            "split": output_split_name(split_name),
            "stage4d_sample_id": f"{output_split_name(split_name)}:{component}:{component_index}",
            "mixture_component": component,
            "phrase_variant": entry["phrase_variant"],
            "structure_variant": entry["structure_variant"],
            "is_hard_negative": bool(is_hard_negative),
            "source_type": entry["source_type"],
            "source_dataset": entry["source_dataset"],
            "source_split": entry["source_split"],
            "source_sample_id": entry["source_sample_id"],
            "source_row_id": entry["source_row_id"],
            "repeat_index": int(repeat_index),
            "canary_kind": entry.get("canary_kind", ""),
            "exact_seed_type": entry.get("exact_seed_type", ""),
            "changed_edge_count": compute_changed_edge_count_from_row(row),
            "anchor_style": anchor_style(clean_text(row.get("response_text"))),
        }
    )
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    component_counts = Counter(clean_text(row.get("mixture_component")) for row in rows)
    phrase_counts = Counter(clean_text(row.get("phrase_variant")) for row in rows if clean_text(row.get("phrase_variant")))
    structure_counts = Counter(clean_text(row.get("structure_variant")) for row in rows if clean_text(row.get("structure_variant")))
    source_type_counts = Counter(clean_text(row.get("source_type")) for row in rows if clean_text(row.get("source_type")))
    anchor_style_counts = Counter(clean_text(row.get("anchor_style")) for row in rows)
    canary_kind_counts = Counter(clean_text(row.get("canary_kind")) for row in rows if clean_text(row.get("canary_kind")))

    hard_negative_count = sum(1 for row in rows if bool(row.get("is_hard_negative")))
    no_anchor_count = int(anchor_style_counts.get("NO_ANCHOR", 0))

    component_phrase_counts: dict[str, dict[str, int]] = {}
    component_structure_counts: dict[str, dict[str, int]] = {}
    for component in COMPONENT_ORDER:
        component_rows = [row for row in rows if clean_text(row.get("mixture_component")) == component]
        component_phrase_counts[component] = dict(
            sorted(
                Counter(clean_text(row.get("phrase_variant")) for row in component_rows if clean_text(row.get("phrase_variant"))).items()
            )
        )
        component_structure_counts[component] = dict(
            sorted(
                Counter(clean_text(row.get("structure_variant")) for row in component_rows if clean_text(row.get("structure_variant"))).items()
            )
        )

    return {
        "total_rows": total,
        "mixture_component_counts": dict(sorted(component_counts.items())),
        "mixture_component_share": counter_share(component_counts, total),
        "phrase_variant_counts": dict(sorted(phrase_counts.items())),
        "structure_variant_counts": dict(sorted(structure_counts.items())),
        "target_structure_counts": {
            key: int(structure_counts.get(key, 0))
            for key in TARGET_POSITIVE_STRUCTURES
        },
        "hard_negative_count": hard_negative_count,
        "hard_negative_share": round(hard_negative_count / max(total, 1), 4),
        "NO_ANCHOR_count": no_anchor_count,
        "NO_ANCHOR_share": round(no_anchor_count / max(total, 1), 4),
        "anchor_style_counts": dict(sorted(anchor_style_counts.items())),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "canary_kind_counts": dict(sorted(canary_kind_counts.items())),
        "component_phrase_variant_counts": component_phrase_counts,
        "component_structure_variant_counts": component_structure_counts,
    }


def build_mixture_breakdown_rows(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    scopes = (
        "mixture_component",
        "phrase_variant",
        "structure_variant",
        "source_type",
        "is_hard_negative",
        "canary_kind",
    )
    for split_name, rows in (("train", train_rows), ("eval", eval_rows)):
        total = max(len(rows), 1)
        for scope in scopes:
            counter = Counter(str(row.get(scope, "")) for row in rows if clean_text(row.get(scope, "")) or scope == "is_hard_negative")
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


def build_examples_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_key_map: dict[str, Callable[[dict[str, Any]], str]] = {
        COMPONENT_POSITIVE: lambda row: f"{clean_text(row.get('phrase_variant'))}|{clean_text(row.get('structure_variant'))}",
        COMPONENT_NEGATIVE: lambda row: f"{clean_text(row.get('phrase_variant'))}|{clean_text(row.get('structure_variant'))}",
        COMPONENT_CANARY: lambda row: f"{clean_text(row.get('canary_kind'))}|{clean_text(row.get('length_bucket'))}",
    }
    payload = {
        "meta": {
            "required_examples_per_component": MIN_EXAMPLES_PER_COMPONENT,
            "components": COMPONENT_ORDER,
        },
        "examples_by_component": {},
    }
    for component in COMPONENT_ORDER:
        component_rows = [row for row in rows if clean_text(row.get("mixture_component")) == component]
        selected = select_examples(component_rows, target_count=MIN_EXAMPLES_PER_COMPONENT, group_key=group_key_map[component])
        payload["examples_by_component"][component] = [
            {
                "stage4d_sample_id": row["stage4d_sample_id"],
                "text": clean_text(row.get("constraint_text")),
                "label": clean_text(row.get("response_text")),
                "phrase_variant": clean_text(row.get("phrase_variant")),
                "structure_variant": clean_text(row.get("structure_variant")),
                "source_type": clean_text(row.get("source_type")),
                "is_hard_negative": bool(row.get("is_hard_negative")),
                "canary_kind": clean_text(row.get("canary_kind")),
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
        component = clean_text(row.get("mixture_component"))
        phrase_variant = clean_text(row.get("phrase_variant"))
        structure_variant = clean_text(row.get("structure_variant"))
        sample_id = clean_text(row.get("stage4d_sample_id"))

        for required in ("mixture_component", "phrase_variant", "structure_variant", "source_type"):
            if not clean_text(row.get(required)):
                raise ValueError(f"{split_name}: {sample_id} missing required field {required}")

        if component == COMPONENT_POSITIVE:
            if phrase_variant not in TARGET_POSITIVE_PHRASES:
                raise ValueError(f"{split_name}: {sample_id} has unexpected positive phrase_variant={phrase_variant}")
            if structure_variant not in TARGET_POSITIVE_STRUCTURES:
                raise ValueError(f"{split_name}: {sample_id} has unexpected positive structure_variant={structure_variant}")
            if bool(row.get("is_hard_negative")):
                raise ValueError(f"{split_name}: {sample_id} positive row cannot be hard negative")
            if int(row.get("changed_edge_count", 0) or 0) != 1:
                raise ValueError(f"{split_name}: {sample_id} positive row must stay single-edge")
            ok, reason = validate_simple_local_policy(row)
            if not ok:
                raise ValueError(f"{split_name}: {sample_id} positive row violates policy-A: {reason}")

        elif component == COMPONENT_NEGATIVE:
            if phrase_variant not in TARGET_NEGATIVE_PHRASES:
                raise ValueError(f"{split_name}: {sample_id} has unexpected negative phrase_variant={phrase_variant}")
            if not bool(row.get("is_hard_negative")):
                raise ValueError(f"{split_name}: {sample_id} hard-negative row must set is_hard_negative=true")
            if clean_text(row.get("anchor_style")) != "NO_ANCHOR":
                raise ValueError(f"{split_name}: {sample_id} hard-negative row must be NO_ANCHOR")
            if int(row.get("changed_edge_count", 0) or 0) != 0:
                raise ValueError(f"{split_name}: {sample_id} hard-negative row cannot contain changed edges")
            ok, reason = validate_simple_local_policy(row)
            if not ok:
                raise ValueError(f"{split_name}: {sample_id} hard-negative row violates policy-A: {reason}")

        elif component == COMPONENT_CANARY:
            if bool(row.get("is_hard_negative")):
                raise ValueError(f"{split_name}: {sample_id} canary row cannot be hard negative")
            if clean_text(row.get("scene_type")) != "directional_asymmetry":
                raise ValueError(f"{split_name}: {sample_id} canary row must stay directional_asymmetry")
            if clean_text(row.get("canary_kind")) not in {"directional_asymmetry", "anti_truncation"}:
                raise ValueError(f"{split_name}: {sample_id} canary_kind missing or invalid")

        else:
            raise ValueError(f"{split_name}: {sample_id} unknown mixture_component={component}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    policy_text = Path(args.policy_md).read_text(encoding="utf-8")
    validate_policy_text(policy_text)

    priority_row = load_priority_row(args.priority_csv)
    remaining_payload = load_json(args.remaining_hard_json)

    target_hard_cases = [
        item
        for item in remaining_payload.get("still_failed_single_edge_cases", [])
        if clean_text(item.get("semantic_gap_subtype")) == "relief_normal_single_edge"
        and clean_text(item.get("structure_tag")) in TARGET_POSITIVE_STRUCTURES
        and clean_text(item.get("phrase_variant")) in {"畅通无阻", "车流顺畅"}
    ]
    if len(target_hard_cases) != EVAL_COMPONENT_TARGETS[COMPONENT_POSITIVE]:
        raise ValueError(f"Expected 18 exact target hard cases, got {len(target_hard_cases)}")

    hard_negative_seeds = [
        item
        for item in remaining_payload.get("new_false_positive_normal_only_anchors", [])
        if clean_text(item.get("phrase_variant")) in TARGET_NEGATIVE_PHRASES
    ]
    if len(hard_negative_seeds) != EVAL_COMPONENT_TARGETS[COMPONENT_NEGATIVE]:
        raise ValueError(f"Expected 8 exact hard-negative seeds, got {len(hard_negative_seeds)}")

    original_paths = resolve_original_paths(args)

    core_train_rows, core_train_path = read_parquet_rows(args.core_train)
    core_eval_rows, core_eval_path = read_parquet_rows(args.core_eval)
    train_rows_raw, train_lookup = make_lookup_rows(original_paths["train"], "train")
    eval_rows_raw, _ = make_lookup_rows(original_paths["eval"], "eval")
    val_normal_rows, val_normal_lookup = make_lookup_rows(original_paths["val_normal"], "val_normal")
    val_special_rows, _ = make_lookup_rows(original_paths["val_special"], "val_special")

    hard_case_texts = {clean_text(item.get("constraint_text")) for item in target_hard_cases}

    positive_train_entries: list[dict[str, Any]] = []
    for row in core_train_rows:
        matched, phrase_variant, structure_variant = positive_bucket_match(row)
        if not matched:
            continue
        positive_train_entries.append(
            build_entry(
                row=row,
                phrase_variant=phrase_variant,
                structure_variant=structure_variant,
                source_type="stage4c_core_targeted",
                source_dataset="dataset_stage4c_singleedge_core",
                source_split=clean_text(row.get("core_split")) or clean_text(row.get("split")),
                source_sample_id=clean_text(row.get("source_sample_id")) or clean_text(row.get("core_sample_id")),
                source_row_id=clean_text(row.get("core_sample_id")),
                exact_seed_type="heldout_like_rewrite" if bool(row.get("heldout_like_target")) else "",
            )
        )

    for row in train_rows_raw:
        matched, phrase_variant, structure_variant = positive_bucket_match(row)
        if not matched:
            continue
        if clean_text(row.get("constraint_text")) in hard_case_texts:
            continue
        relabeled_row = relabel_simple_local_v2(row)
        positive_train_entries.append(
            build_entry(
                row=relabeled_row,
                phrase_variant=phrase_variant,
                structure_variant=structure_variant,
                source_type="orig_train_policy_a_relabel",
                source_dataset="dataset_sparse_v2",
                source_split="train",
                source_sample_id=clean_text(row.get("_source_row_id")),
                source_row_id=clean_text(row.get("_source_row_id")),
            )
        )

    positive_train_cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in positive_train_entries:
        positive_train_cells[(entry["phrase_variant"], entry["structure_variant"])].append(entry)

    selected_positive_train: list[dict[str, Any]] = []
    for phrase_variant, structure_variant, quota in POSITIVE_TRAIN_CELL_QUOTAS:
        preferred_core = 0 if phrase_variant == "顺畅" else 1
        selected_positive_train.extend(
            take_quota_with_repeats(
                positive_train_cells[(phrase_variant, structure_variant)],
                quota=quota,
                sort_key=lambda item, preferred_core=preferred_core: (
                    0
                    if (preferred_core == 0 and clean_text(item.get("source_type")) == "stage4c_core_targeted")
                    or (preferred_core == 1 and clean_text(item.get("source_type")) == "orig_train_policy_a_relabel")
                    else 1,
                    clean_text(item.get("dedupe_text")),
                    clean_text(item.get("source_row_id")),
                ),
            )
        )
    if len(selected_positive_train) != TRAIN_COMPONENT_TARGETS[COMPONENT_POSITIVE]:
        raise ValueError("Positive train quota mismatch")

    selected_positive_eval: list[dict[str, Any]] = []
    for item in target_hard_cases:
        source_row_id = clean_text(item.get("sample_id"))
        source_row = val_normal_lookup.get(source_row_id)
        if source_row is None:
            raise ValueError(f"Missing hard case source row for {source_row_id}")
        relabeled_row = relabel_simple_local_v2(source_row)
        selected_positive_eval.append(
            build_entry(
                row=relabeled_row,
                phrase_variant=normalize_positive_phrase(relabeled_row),
                structure_variant=clean_text(item.get("structure_tag")),
                source_type="remaining_hard_case_exact_eval_only",
                source_dataset="dataset_sparse_v2",
                source_split="val_normal",
                source_sample_id=source_row_id,
                source_row_id=source_row_id,
                exact_seed_type="remaining_hard_case_exact",
            )
        )

    negative_train_entries: list[dict[str, Any]] = []
    for row in train_rows_raw:
        matched, phrase_variant, structure_variant = hard_negative_match(row)
        if not matched:
            continue
        relabeled_row = relabel_simple_local_v2(row)
        negative_train_entries.append(
            build_entry(
                row=relabeled_row,
                phrase_variant=phrase_variant,
                structure_variant=structure_variant,
                source_type="orig_train_hard_negative_guard",
                source_dataset="dataset_sparse_v2",
                source_split="train",
                source_sample_id=clean_text(row.get("_source_row_id")),
                source_row_id=clean_text(row.get("_source_row_id")),
            )
        )

    negative_train_cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in negative_train_entries:
        negative_train_cells[(entry["phrase_variant"], entry["structure_variant"])].append(entry)

    selected_negative_train: list[dict[str, Any]] = []
    for phrase_variant, structure_variant, quota in NEGATIVE_TRAIN_CELL_QUOTAS:
        selected_negative_train.extend(
            take_quota_with_repeats(
                negative_train_cells[(phrase_variant, structure_variant)],
                quota=quota,
                sort_key=lambda item: (
                    LENGTH_RANK.get(clean_text(item["row"].get("length_bucket")), 99),
                    clean_text(item.get("dedupe_text")),
                    clean_text(item.get("source_row_id")),
                ),
            )
        )
    if len(selected_negative_train) != TRAIN_COMPONENT_TARGETS[COMPONENT_NEGATIVE]:
        raise ValueError("Negative train quota mismatch")

    selected_negative_eval: list[dict[str, Any]] = []
    for item in hard_negative_seeds:
        source_row_id = clean_text(item.get("sample_id"))
        source_row = val_normal_lookup.get(source_row_id)
        if source_row is None:
            raise ValueError(f"Missing hard-negative source row for {source_row_id}")
        relabeled_row = relabel_simple_local_v2(source_row)
        matched, phrase_variant, structure_variant = hard_negative_match(relabeled_row)
        if not matched:
            raise ValueError(f"Hard-negative seed no longer matches: {source_row_id}")
        selected_negative_eval.append(
            build_entry(
                row=relabeled_row,
                phrase_variant=phrase_variant,
                structure_variant=structure_variant,
                source_type="remaining_false_positive_seed_eval_only",
                source_dataset="dataset_sparse_v2",
                source_split="val_normal",
                source_sample_id=source_row_id,
                source_row_id=source_row_id,
                exact_seed_type="remaining_false_positive_seed",
            )
        )

    canary_train_entries: list[dict[str, Any]] = []
    for row in train_rows_raw:
        if clean_text(row.get("scene_type")) != "directional_asymmetry":
            continue
        if int(row.get("anchor_count", 0) or 0) <= 0:
            continue
        canary_kind = "anti_truncation" if clean_text(row.get("length_bucket")) in {"long", "noisy_long"} else "directional_asymmetry"
        canary_train_entries.append(
            build_entry(
                row=row,
                phrase_variant=f"{canary_kind}_canary",
                structure_variant=classify_structure_variant(clean_text(row.get("constraint_text"))),
                source_type=f"{canary_kind}_canary",
                source_dataset="dataset_sparse_v2",
                source_split="train",
                source_sample_id=clean_text(row.get("_source_row_id")),
                source_row_id=clean_text(row.get("_source_row_id")),
                canary_kind=canary_kind,
            )
        )

    selected_canary_train: list[dict[str, Any]] = []
    for canary_kind, quota in CANARY_TRAIN_QUOTAS.items():
        pool = [entry for entry in canary_train_entries if clean_text(entry.get("canary_kind")) == canary_kind]
        selected_canary_train.extend(
            take_quota_with_repeats(
                pool,
                quota=quota,
                sort_key=lambda item, canary_kind=canary_kind: (
                    0 if canary_kind == "anti_truncation" and clean_text(item["row"].get("length_bucket")) == "noisy_long" else 1,
                    0 if canary_kind == "directional_asymmetry" and clean_text(item["row"].get("length_bucket")) == "short" else 1,
                    clean_text(item.get("dedupe_text")),
                    clean_text(item.get("source_row_id")),
                ),
            )
        )
    if len(selected_canary_train) != TRAIN_COMPONENT_TARGETS[COMPONENT_CANARY]:
        raise ValueError("Canary train quota mismatch")

    canary_eval_entries: list[dict[str, Any]] = []
    for row in val_special_rows:
        if clean_text(row.get("scene_type")) != "directional_asymmetry":
            continue
        if int(row.get("anchor_count", 0) or 0) <= 0:
            continue
        canary_kind = "anti_truncation" if clean_text(row.get("length_bucket")) in {"long", "noisy_long"} else "directional_asymmetry"
        canary_eval_entries.append(
            build_entry(
                row=row,
                phrase_variant=f"{canary_kind}_canary",
                structure_variant=classify_structure_variant(clean_text(row.get("constraint_text"))),
                source_type=f"{canary_kind}_canary_eval",
                source_dataset="dataset_sparse_v2",
                source_split="val_special",
                source_sample_id=clean_text(row.get("_source_row_id")),
                source_row_id=clean_text(row.get("_source_row_id")),
                canary_kind=canary_kind,
            )
        )

    selected_canary_eval: list[dict[str, Any]] = []
    for canary_kind, quota in CANARY_EVAL_QUOTAS.items():
        pool = [entry for entry in canary_eval_entries if clean_text(entry.get("canary_kind")) == canary_kind]
        selected_canary_eval.extend(
            take_quota_with_repeats(
                pool,
                quota=quota,
                sort_key=lambda item, canary_kind=canary_kind: (
                    0 if canary_kind == "anti_truncation" and clean_text(item["row"].get("length_bucket")) == "noisy_long" else 1,
                    0 if canary_kind == "directional_asymmetry" and clean_text(item["row"].get("length_bucket")) == "short" else 1,
                    clean_text(item.get("dedupe_text")),
                    clean_text(item.get("source_row_id")),
                ),
            )
        )
    if len(selected_canary_eval) != EVAL_COMPONENT_TARGETS[COMPONENT_CANARY]:
        raise ValueError("Canary eval quota mismatch")

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    def append_component_rows(
        output_rows: list[dict[str, Any]],
        *,
        split_name: str,
        component: str,
        entries: list[dict[str, Any]],
        is_hard_negative: bool,
    ) -> None:
        repeats: Counter[str] = Counter()
        for entry in entries:
            component_index = sum(1 for row in output_rows if clean_text(row.get("mixture_component")) == component)
            output_rows.append(
                make_output_row(
                    split_name=split_name,
                    component=component,
                    component_index=component_index,
                    entry=entry,
                    repeat_index=int(repeats[clean_text(entry.get("source_row_id"))]),
                    is_hard_negative=is_hard_negative,
                )
            )
            repeats[clean_text(entry.get("source_row_id"))] += 1

    append_component_rows(train_rows, split_name="train", component=COMPONENT_POSITIVE, entries=selected_positive_train, is_hard_negative=False)
    append_component_rows(train_rows, split_name="train", component=COMPONENT_NEGATIVE, entries=selected_negative_train, is_hard_negative=True)
    append_component_rows(train_rows, split_name="train", component=COMPONENT_CANARY, entries=selected_canary_train, is_hard_negative=False)

    append_component_rows(eval_rows, split_name="eval", component=COMPONENT_POSITIVE, entries=selected_positive_eval, is_hard_negative=False)
    append_component_rows(eval_rows, split_name="eval", component=COMPONENT_NEGATIVE, entries=selected_negative_eval, is_hard_negative=True)
    append_component_rows(eval_rows, split_name="eval", component=COMPONENT_CANARY, entries=selected_canary_eval, is_hard_negative=False)

    validate_final_rows(train_rows, "train")
    validate_final_rows(eval_rows, "eval")

    if not (80 <= len(train_rows) <= 120):
        raise ValueError(f"train total must stay in 80~120; got {len(train_rows)}")

    train_summary = summarize_rows(train_rows)
    eval_summary = summarize_rows(eval_rows)
    overall_summary = summarize_rows(train_rows + eval_rows)

    positive_train_share = train_summary["mixture_component_share"].get(COMPONENT_POSITIVE, 0.0)
    negative_train_share = train_summary["mixture_component_share"].get(COMPONENT_NEGATIVE, 0.0)
    canary_train_share = train_summary["mixture_component_share"].get(COMPONENT_CANARY, 0.0)
    positive_is_absolute_main = (
        train_summary["mixture_component_counts"].get(COMPONENT_POSITIVE, 0)
        > train_summary["mixture_component_counts"].get(COMPONENT_NEGATIVE, 0)
        + train_summary["mixture_component_counts"].get(COMPONENT_CANARY, 0)
        and 0.60 <= positive_train_share <= 0.70
    )
    hard_negative_guard_sufficient = (
        0.20 <= negative_train_share <= 0.30
        and train_summary["hard_negative_count"] >= 20
        and train_summary["phrase_variant_counts"].get("交通基本正常", 0) >= 10
        and train_summary["phrase_variant_counts"].get("保持正常通行", 0) >= 10
        and len(selected_negative_eval) == 8
    )

    write_parquet_rows(train_rows, output_root / "train")
    write_parquet_rows(eval_rows, output_root / "eval")

    breakdown_rows = build_mixture_breakdown_rows(train_rows, eval_rows)
    write_csv(breakdown_rows, output_root / "mixture_breakdown.csv")

    examples_payload = build_examples_payload(train_rows + eval_rows)
    (output_root / "examples.json").write_text(json.dumps(examples_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "dataset": "dataset_sparse_v2_stage4d_micro",
        "target_bucket": TARGET_BUCKET,
        "paths": {
            "remaining_hard_json": args.remaining_hard_json,
            "priority_csv": args.priority_csv,
            "policy_md": args.policy_md,
            "core_train": str(core_train_path),
            "core_eval": str(core_eval_path),
            "original_train": original_paths["train"],
            "original_eval": original_paths["eval"],
            "original_val_normal": original_paths["val_normal"],
            "original_val_special": original_paths["val_special"],
            "output_train": str(output_root / "train"),
            "output_eval": str(output_root / "eval"),
        },
        "optional_input_status": {
            "requested_stage4_fix_train": original_paths["stage4_fix_train_requested"],
            "requested_stage4_fix_eval": original_paths["stage4_fix_eval_requested"],
            "stage4_fix_train_exists": original_paths["stage4_fix_train_exists"],
            "stage4_fix_eval_exists": original_paths["stage4_fix_eval_exists"],
            "fallback_to_dataset_sparse_v2": True,
        },
        "priority_reference": top_priority_reference(priority_row),
        "selection_reference": {
            "remaining_target_hard_cases": len(target_hard_cases),
            "remaining_false_positive_seeds": len(hard_negative_seeds),
            "train_target_total": len(train_rows),
            "eval_target_total": len(eval_rows),
        },
        "train_total": len(train_rows),
        "eval_total": len(eval_rows),
        "split_summary": {
            "train": train_summary,
            "eval": eval_summary,
            "overall": overall_summary,
        },
        "final_checks": {
            "train_total_in_requested_band": bool(80 <= len(train_rows) <= 120),
            "micro_patch_positive_core_share_in_band": bool(0.60 <= positive_train_share <= 0.70),
            "hard_negative_guard_share_in_band": bool(0.20 <= negative_train_share <= 0.30),
            "minimal_stability_canary_share_in_band": bool(0.10 <= canary_train_share <= 0.15),
            "single_edge_bucket_is_absolute_main_signal": bool(positive_is_absolute_main),
            "hard_negative_guard_sufficient_for_normal_only_false_positive_risk": bool(hard_negative_guard_sufficient),
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"train_rows={len(train_rows)} eval_rows={len(eval_rows)}")
    print(
        "是否已经把唯一值得修的桶变成绝对主信号："
        f"{'YES' if positive_is_absolute_main else 'NO'} "
        f"(micro_patch_positive_core={positive_train_share:.1%}, count={train_summary['mixture_component_counts'].get(COMPONENT_POSITIVE, 0)})"
    )
    print(
        "hard negative guard 是否足以压 normal-only 假阳性风险："
        f"{'YES' if hard_negative_guard_sufficient else 'NO'} "
        f"(hard_negative_guard={negative_train_share:.1%}, train_hard_negative={train_summary['hard_negative_count']}, eval_fp_seeds={len(selected_negative_eval)})"
    )


def top_priority_reference(priority_row: dict[str, str]) -> dict[str, Any]:
    return {
        "priority_rank": clean_text(priority_row.get("priority_rank")),
        "bucket_name": clean_text(priority_row.get("bucket_name")),
        "count": clean_text(priority_row.get("count")),
        "phrases": clean_text(priority_row.get("phrases")),
        "structures": clean_text(priority_row.get("structures")),
        "guardrail": clean_text(priority_row.get("guardrail")),
        "stop_reason": clean_text(priority_row.get("stop_reason")),
        "final_recommendation": clean_text(priority_row.get("final_recommendation")),
    }


if __name__ == "__main__":
    main()
