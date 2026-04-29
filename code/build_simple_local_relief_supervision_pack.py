#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


NEUTRAL_LEVELS = {"正常", "畅通"}
TARGET_BUCKETS = [
    "relief_normal_single_edge",
    "relief_normal_short_range",
    "multi_edge_relief",
]
STRICT_POLICY_NAME = "A. strict-aligned labeling"
STRICT_POLICY_RULE = "输出所有当前生效且可定位的 changed-event anchors，包含异常主事件与畅通/正常 relief side event。"
MANUAL_VARIANTS = [
    {"level": "畅通", "phrase": "通行顺畅", "weight": 1.2},
    {"level": "畅通", "phrase": "畅通无阻", "weight": 1.0},
    {"level": "畅通", "phrase": "车流顺畅", "weight": 1.1},
    {"level": "畅通", "phrase": "道路畅通", "weight": 1.15},
]
MANUAL_TEXT_TEMPLATES = [
    "常态或局部短时事件：{event_text}",
    "{event_text}",
    "当前需要根据常态或局部短时事件进行路径规划。 请优先按当前生效约束理解，不要被已解除或背景信息误导。 {event_text} 其余未明确提及路段按常态理解，不要额外扩大影响范围。",
]
TEMPLATE_WRAPPERS = [
    "常态或局部短时事件：{event_text} 周边商圈有零星出库车流播报，但不改变本段主判断。",
    "当前需要根据常态或局部短时事件进行路径规划。 请优先按当前生效约束理解，不要被已解除或背景信息误导。 {event_text} 导航播报中还提到相邻支路车流波动，但不直接改变本段主判断。",
    "{event_text} 部分播报会提到周边停车场信息，但主判断仍以道路方向、区间和当前生效约束为核心。",
]
BASE_COLUMNS = [
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
]
EXTRA_COLUMNS = [
    "pack_sample_id",
    "pack_bucket",
    "pack_source_type",
    "pack_source_detail",
    "origin_dataset_root",
    "origin_split",
    "origin_sample_id",
    "seed_sample_id",
    "changed_edge_count",
    "relief_severity",
    "lexical_tag",
    "semantic_fingerprint",
    "range_fingerprint",
    "is_failure_library_match",
    "is_new_supervision",
    "can_direct_stage4b",
    "why_included",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a strict-aligned simple_local relief supervision pack without launching training."
    )
    parser.add_argument(
        "--historical-root",
        default="/root/autodl-tmp/dataset_sparse_v2",
        help="Historical dataset root containing old train/eval/val splits.",
    )
    parser.add_argument(
        "--failure-audit-csv",
        default="/root/autodl-tmp/results/simple_local_semantic_gap_audit.csv",
    )
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
        default="/root/autodl-tmp/dataset_simple_local_relief_pack",
    )
    parser.add_argument("--train-per-bucket", type=int, default=16)
    parser.add_argument("--eval-per-bucket", type=int, default=8)
    parser.add_argument("--examples-per-bucket", type=int, default=10)
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


def event_is_changed(event: dict) -> bool:
    try:
        weight = float(event.get("weight", 2.0))
    except (TypeError, ValueError):
        weight = 2.0
    return abs(weight - 2.0) > 0.05


def get_changed_events(events: list[dict]) -> list[dict]:
    return [
        event
        for event in events
        if event.get("active", True) and event_is_changed(event)
    ]


def event_edge_count(event: dict) -> int:
    return len({str(edge) for edge in (event.get("anchor_edges", []) or []) if edge})


def changed_edge_count(events: list[dict]) -> int:
    return len(
        {
            str(edge)
            for event in events
            for edge in (event.get("anchor_edges", []) or [])
            if edge
        }
    )


def infer_bucket(changed_events: list[dict]) -> str:
    abnormal = [
        event
        for event in changed_events
        if str(event.get("level", "")).strip() not in NEUTRAL_LEVELS
    ]
    neutral = [
        event
        for event in changed_events
        if str(event.get("level", "")).strip() in NEUTRAL_LEVELS
    ]
    if abnormal and neutral:
        return "anomaly_plus_normal_side_event"
    if not abnormal and neutral:
        edge_count = changed_edge_count(changed_events)
        if edge_count <= 1:
            return "relief_normal_single_edge"
        if edge_count == 2:
            return "relief_normal_short_range"
        return "multi_edge_relief"
    return "other"


def derive_relief_severity(edge_count: int) -> str:
    if edge_count <= 1:
        return "mild"
    if edge_count == 2:
        return "moderate"
    return "severe"


def derive_length_bucket(text: str) -> str:
    length = len(clean_text(text))
    if length <= 35:
        return "short"
    if length <= 70:
        return "medium"
    return "long"


def lexical_tag(text: str) -> str:
    text = clean_text(text)
    for token in [
        "畅通无阻",
        "车流顺畅",
        "通行顺畅",
        "恢复正常通行",
        "保持正常通行",
        "交通基本正常",
        "正常通行",
        "车流稳定",
        "道路畅通",
    ]:
        if token in text:
            return token
    return "other"


def event_anchor_line(event: dict) -> str:
    return (
        f"ANCHOR|ROAD={event.get('road', '')}|DIR={event.get('dir', '')}|"
        f"RANGE={event.get('range', '')}|LEVEL={event.get('level', '')}"
    )


def label_is_valid(label: str) -> bool:
    lines = [line.strip() for line in clean_text(label).splitlines() if line.strip()]
    if not lines:
        return False
    body: list[str] = []
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
    if not body:
        return False
    if body == ["NO_ANCHOR"]:
        return True
    return all(line.startswith("ANCHOR|ROAD=") for line in body)


def render_label(changed_events: list[dict]) -> str:
    anchor_lines = [event_anchor_line(event) for event in changed_events]
    think_lines = [
        "<think>",
        "scene=simple_local",
        f"anchor_count={len(anchor_lines)}",
        "simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。",
        "</think>",
    ]
    return "\n".join([*think_lines, *(anchor_lines or ["NO_ANCHOR"])])


def render_event_text(event: dict, phrase_override: str | None = None) -> str:
    road = str(event.get("road", "")).strip()
    range_label = str(event.get("range", "")).strip()
    dir_label = str(event.get("dir", "")).strip()
    if phrase_override:
        return f"{road}{range_label}段{dir_label}{phrase_override}"

    level = str(event.get("level", "")).strip()
    if level == "畅通":
        return f"{road}{range_label}段{dir_label}畅通无阻"
    if level == "正常":
        return f"{road}{range_label}段{dir_label}保持正常通行"
    return clean_text(str(event.get("text", ""))) or f"{road}{range_label}段{dir_label}{level}"


def render_prompt(constraint_text: str, changed_events: list[dict]) -> str:
    event_lines = []
    for idx, event in enumerate(changed_events, start=1):
        event_text = clean_text(str(event.get("text", ""))) or render_event_text(event)
        event_lines.append(
            "EVENT_{idx}: ROAD={road} | DIR={dir_label} | RANGE={range_label} | LEVEL={level} | "
            "TIME={time_label} | PROPAGATION={propagation} | CONFLICT={conflict} | TEXT={text}".format(
                idx=idx,
                road=event.get("road", ""),
                dir_label=event.get("dir", ""),
                range_label=event.get("range", ""),
                level=event.get("level", ""),
                time_label=event.get("time", "当前"),
                propagation=event.get("propagation", "无明显传播"),
                conflict=event.get("conflict", "单约束"),
                text=event_text,
            )
        )
    prompt_lines = [
        "TASK=sparse_output",
        "SCENE_TYPE=simple_local",
        "OUTPUT_OBJECT=当前生效的 changed-event anchors",
        "OUTPUT_FORMAT=ANCHOR|ROAD=道路名|DIR=方向|RANGE=范围|LEVEL=等级",
        "OUTPUT_RULE=输出所有当前生效且可定位的 changed-event anchors；包含异常主事件、畅通/顺畅/正常 relief side event；不要补全全量边权；不要解释。",
        "POLICY=simple_local_v2_strict_aligned",
        "SCHEMA=ROAD|DIR|RANGE|LEVEL|TIME|PROPAGATION|CONFLICT",
        f"NARRATIVE={clean_text(constraint_text)}",
        *event_lines,
        "仅输出锚点，不要解释。",
    ]
    return "\n".join(prompt_lines)


def build_messages(prompt_text: str, response_text: str) -> str:
    return json.dumps(
        [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ],
        ensure_ascii=False,
    )


def semantic_fingerprint(events: list[dict]) -> str:
    return " || ".join(sorted(event_anchor_line(event) for event in events))


def range_fingerprint(events: list[dict]) -> str:
    return " || ".join(
        sorted(
            f"ROAD={event.get('road', '')}|DIR={event.get('dir', '')}|RANGE={event.get('range', '')}"
            for event in events
        )
    )


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


def load_failure_audit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["semantic_gap_subtype"].isin(TARGET_BUCKETS)].copy()
    return df


def load_reference_meta(summary_csv: Path, examples_json: Path) -> dict:
    summary_df = pd.read_csv(summary_csv)
    with examples_json.open("r", encoding="utf-8") as fh:
        examples_data = json.load(fh)
    return {
        "failure_summary_rows": int(len(summary_df)),
        "failure_examples_meta": examples_data.get("meta", {}),
    }


def read_history_rows(dataset_root: Path) -> list[dict]:
    rows: list[dict] = []
    for split_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        parquet_path = split_dir / "data.parquet"
        if not parquet_path.exists():
            continue
        table = pq.read_table(parquet_path, columns=BASE_COLUMNS)
        for idx, row in enumerate(table.to_pylist()):
            row = dict(row)
            row["origin_dataset_root"] = str(dataset_root)
            row["origin_split_name"] = split_dir.name
            row["split_index"] = idx
            rows.append(row)
    return rows


def historical_reason(bucket: str, source_detail: str, is_failure_library_match: bool) -> str:
    reason_prefix = "失败样本库优先回收" if is_failure_library_match else "历史数据集直接检索"
    bucket_reason = {
        "relief_normal_single_edge": "单边 relief-only changed edge，直接补 no_anchor 主瓶颈。",
        "relief_normal_short_range": "2-edge relief 范围覆盖完整，可直接补短范围漏锚监督。",
        "multi_edge_relief": "3+ edge relief 范围覆盖完整，可直接补 multi-edge 边界监督。",
    }[bucket]
    return f"{reason_prefix}（{source_detail}）：{bucket_reason}"


def build_historical_candidates(
    history_rows: list[dict],
    failure_audit_df: pd.DataFrame,
) -> tuple[dict[str, list[dict]], dict[str, set[str]], list[dict]]:
    failure_rows = {
        str(row["sample_id"]): row
        for row in failure_audit_df.to_dict(orient="records")
    }
    candidates_by_bucket: dict[str, list[dict]] = defaultdict(list)
    target_range_fps: dict[str, set[str]] = defaultdict(set)
    event_seed_rows: list[dict] = []

    for row in history_rows:
        if row["scene_type"] != "simple_local":
            continue

        origin_split = str(row["split"])
        origin_sample_id = f"{origin_split}:{int(row['split_index'])}"
        events = safe_json_loads(row["event_records_json"])
        changed_events = get_changed_events(events)
        if not changed_events:
            continue

        bucket = infer_bucket(changed_events)
        if bucket in TARGET_BUCKETS:
            response_text = render_label(changed_events)
            if not label_is_valid(response_text):
                raise ValueError(f"Generated invalid label for {origin_sample_id}")
            constraint_text = clean_text(row["constraint_text"])
            prompt_text = render_prompt(constraint_text, changed_events)
            source_detail = f"historical_{origin_split}"
            is_failure_library_match = origin_sample_id in failure_rows
            candidate = {
                "messages": build_messages(prompt_text, response_text),
                "scene_type": "simple_local",
                "scene_bucket": row["scene_bucket"],
                "length_bucket": derive_length_bucket(constraint_text),
                "split": row["split"],
                "constraint_text": constraint_text,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "anchor_count": len(changed_events),
                "ground_truth_json": row["ground_truth_json"],
                "event_records_json": json.dumps(changed_events, ensure_ascii=False),
                "parse_confidence": 1.0,
                "total_edges": int(row["total_edges"]),
                "pack_bucket": bucket,
                "pack_source_type": "historical_retrieval",
                "pack_source_detail": source_detail,
                "origin_dataset_root": row["origin_dataset_root"],
                "origin_split": origin_split,
                "origin_sample_id": origin_sample_id,
                "seed_sample_id": origin_sample_id,
                "changed_edge_count": changed_edge_count(changed_events),
                "relief_severity": derive_relief_severity(changed_edge_count(changed_events)),
                "lexical_tag": lexical_tag(constraint_text),
                "semantic_fingerprint": semantic_fingerprint(changed_events),
                "range_fingerprint": range_fingerprint(changed_events),
                "is_failure_library_match": bool(is_failure_library_match),
                "is_new_supervision": True,
                "can_direct_stage4b": True,
                "why_included": historical_reason(bucket, source_detail, is_failure_library_match),
            }
            candidates_by_bucket[bucket].append(candidate)
            target_range_fps[bucket].add(candidate["range_fingerprint"])

        for event in changed_events:
            event_seed_rows.append(
                {
                    "origin_dataset_root": row["origin_dataset_root"],
                    "origin_split": origin_split,
                    "origin_sample_id": origin_sample_id,
                    "ground_truth_json": row["ground_truth_json"],
                    "total_edges": int(row["total_edges"]),
                    "changed_event": copy.deepcopy(event),
                    "event_edge_count": event_edge_count(event),
                    "range_fingerprint": (
                        f"ROAD={event.get('road', '')}|DIR={event.get('dir', '')}|RANGE={event.get('range', '')}"
                    ),
                }
            )

    return candidates_by_bucket, target_range_fps, event_seed_rows


def sort_candidates_for_eval(candidates: list[dict]) -> list[dict]:
    return sorted(
        candidates,
        key=lambda item: (
            0 if item["pack_source_type"] == "historical_retrieval" else 1,
            0 if item["is_failure_library_match"] else 1,
            0 if item["origin_split"] in {"val_normal", "eval", "val_special"} else 1,
            item["lexical_tag"],
            item["origin_sample_id"],
        ),
    )


def sort_candidates_for_train(candidates: list[dict]) -> list[dict]:
    return sorted(
        candidates,
        key=lambda item: (
            0 if item["pack_source_type"] == "historical_retrieval" and item["origin_split"] == "train" else 1,
            0 if item["pack_source_type"] == "historical_retrieval" else 1,
            0 if not item["is_failure_library_match"] else 1,
            item["lexical_tag"],
            item["origin_sample_id"],
        ),
    )


def round_robin_select(
    candidates: list[dict],
    target_count: int,
    *,
    used_semantics: set[str],
    used_row_ids: set[str],
    allow_duplicate_semantics: bool = False,
) -> list[dict]:
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        by_tag[item["lexical_tag"]].append(item)

    selected: list[dict] = []
    tag_order = sorted(by_tag)
    while len(selected) < target_count and tag_order:
        progressed = False
        next_order: list[str] = []
        for tag in tag_order:
            pool = by_tag[tag]
            picked = None
            while pool:
                candidate = pool.pop(0)
                row_key = f"{candidate['pack_source_type']}::{candidate['origin_sample_id']}::{candidate['constraint_text']}"
                if row_key in used_row_ids:
                    continue
                if not allow_duplicate_semantics and candidate["semantic_fingerprint"] in used_semantics:
                    continue
                picked = candidate
                break
            if picked is not None:
                selected.append(picked)
                used_row_ids.add(
                    f"{picked['pack_source_type']}::{picked['origin_sample_id']}::{picked['constraint_text']}"
                )
                if not allow_duplicate_semantics:
                    used_semantics.add(picked["semantic_fingerprint"])
                progressed = True
                if len(selected) >= target_count:
                    break
            if pool:
                next_order.append(tag)
        if not progressed:
            break
        tag_order = next_order
    return selected


def reset_ground_truth(base_ground_truth_json: str, target_edges: list[str], weight: float) -> tuple[str, int]:
    base_gt = safe_json_loads(base_ground_truth_json)
    if not isinstance(base_gt, dict):
        raise ValueError("ground_truth_json must decode to a dict")
    new_gt = {str(edge_id): 2.0 for edge_id in base_gt}
    for edge_id in target_edges:
        new_gt[str(edge_id)] = float(weight)
    return json.dumps(new_gt, ensure_ascii=False, sort_keys=True), len(new_gt)


def make_manual_constraint_text(event_text: str, variant_index: int) -> str:
    template = MANUAL_TEXT_TEMPLATES[variant_index % len(MANUAL_TEXT_TEMPLATES)]
    return template.format(event_text=event_text)


def build_manual_candidates(
    bucket: str,
    seed_rows: list[dict],
    blocked_range_fps: set[str],
    already_selected_semantics: set[str],
    needed_count: int,
) -> list[dict]:
    manual_candidates: list[dict] = []
    seen_ranges: set[str] = set()
    target_span = {
        "relief_normal_single_edge": 1,
        "relief_normal_short_range": 2,
        "multi_edge_relief": 3,
    }[bucket]

    for seed in seed_rows:
        span = int(seed["event_edge_count"])
        if bucket == "multi_edge_relief":
            span_ok = span >= 3
        else:
            span_ok = span == target_span
        if not span_ok:
            continue

        range_fp = seed["range_fingerprint"]
        if range_fp in blocked_range_fps or range_fp in seen_ranges:
            continue

        base_event = copy.deepcopy(seed["changed_event"])
        base_edges = [str(edge) for edge in (base_event.get("anchor_edges", []) or []) if edge]
        if not base_edges:
            continue

        for variant_index, variant in enumerate(MANUAL_VARIANTS):
            event = copy.deepcopy(base_event)
            event["level"] = variant["level"]
            event["weight"] = variant["weight"]
            event["time"] = "当前"
            event["propagation"] = "无明显传播"
            event["conflict"] = "单约束"
            event["active"] = True
            event_text = render_event_text(event, phrase_override=variant["phrase"])
            event["text"] = event_text
            changed_events = [event]
            manual_semantic_fp = semantic_fingerprint(changed_events)
            if manual_semantic_fp in already_selected_semantics:
                continue

            constraint_text = make_manual_constraint_text(event_text, variant_index)
            ground_truth_json, total_edges = reset_ground_truth(
                seed["ground_truth_json"],
                base_edges,
                float(variant["weight"]),
            )
            response_text = render_label(changed_events)
            prompt_text = render_prompt(constraint_text, changed_events)
            manual_candidates.append(
                {
                    "messages": build_messages(prompt_text, response_text),
                    "scene_type": "simple_local",
                    "scene_bucket": "simple_local",
                    "length_bucket": derive_length_bucket(constraint_text),
                    "split": seed["origin_split"],
                    "constraint_text": constraint_text,
                    "prompt_text": prompt_text,
                    "response_text": response_text,
                    "anchor_count": 1,
                    "ground_truth_json": ground_truth_json,
                    "event_records_json": json.dumps(changed_events, ensure_ascii=False),
                    "parse_confidence": 1.0,
                    "total_edges": total_edges,
                    "pack_bucket": bucket,
                    "pack_source_type": "manual_generation",
                    "pack_source_detail": "manual_generation_from_structured_seed",
                    "origin_dataset_root": seed["origin_dataset_root"],
                    "origin_split": seed["origin_split"],
                    "origin_sample_id": seed["origin_sample_id"],
                    "seed_sample_id": seed["origin_sample_id"],
                    "changed_edge_count": changed_edge_count(changed_events),
                    "relief_severity": derive_relief_severity(changed_edge_count(changed_events)),
                    "lexical_tag": lexical_tag(constraint_text),
                    "semantic_fingerprint": manual_semantic_fp,
                    "range_fingerprint": range_fingerprint(changed_events),
                    "is_failure_library_match": False,
                    "is_new_supervision": True,
                    "can_direct_stage4b": True,
                    "why_included": (
                        "历史唯一语义不足后，基于结构化事件程序化生成新的 "
                        f"{bucket} relief-only strict-aligned 样本。"
                    ),
                }
            )
            already_selected_semantics.add(manual_semantic_fp)
            seen_ranges.add(range_fp)
            if len(manual_candidates) >= needed_count:
                return manual_candidates
            break
    return manual_candidates


def build_template_candidates(
    bucket: str,
    source_candidates: list[dict],
    used_texts: set[str],
    needed_count: int,
) -> list[dict]:
    template_candidates: list[dict] = []
    for source in source_candidates:
        changed_events = safe_json_loads(source["event_records_json"])
        if not isinstance(changed_events, list) or not changed_events:
            continue
        event_text = clean_text(str(changed_events[0].get("text", ""))) or render_event_text(changed_events[0])
        for wrapper in TEMPLATE_WRAPPERS:
            constraint_text = wrapper.format(event_text=event_text)
            if constraint_text in used_texts:
                continue
            prompt_text = render_prompt(constraint_text, changed_events)
            response_text = render_label(changed_events)
            candidate = copy.deepcopy(source)
            candidate["messages"] = build_messages(prompt_text, response_text)
            candidate["length_bucket"] = derive_length_bucket(constraint_text)
            candidate["constraint_text"] = constraint_text
            candidate["prompt_text"] = prompt_text
            candidate["response_text"] = response_text
            candidate["pack_source_type"] = "template_transformation"
            candidate["pack_source_detail"] = "template_wrap_existing_simple_local"
            candidate["why_included"] = (
                "历史检索与程序化手工生成仍不足后，对已有 simple_local 模板做语义保持变换补齐。"
            )
            template_candidates.append(candidate)
            used_texts.add(constraint_text)
            if len(template_candidates) >= needed_count:
                return template_candidates
    return template_candidates


def assign_pack_fields(rows: list[dict], split_name: str) -> list[dict]:
    assigned: list[dict] = []
    for idx, row in enumerate(rows):
        item = copy.deepcopy(row)
        item["pack_sample_id"] = f"{split_name}:{idx}"
        item["split"] = f"simple_local_relief_pack_{split_name}"
        assigned.append(item)
    return assigned


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pydict({column: df[column].tolist() for column in df.columns})
    pq.write_table(table, path)


def build_examples(rows: list[dict], per_bucket: int) -> dict[str, list[dict]]:
    examples: dict[str, list[dict]] = defaultdict(list)
    ordered = sorted(
        rows,
        key=lambda row: (
            TARGET_BUCKETS.index(row["pack_bucket"]),
            0 if row["split"].endswith("eval") else 1,
            0 if row["pack_source_type"] == "historical_retrieval" else 1,
            row["lexical_tag"],
            row["pack_sample_id"],
        ),
    )
    for row in ordered:
        bucket = row["pack_bucket"]
        if len(examples[bucket]) >= per_bucket:
            continue
        examples[bucket].append(
            {
                "text": row["constraint_text"],
                "label": row["response_text"],
                "bucket": bucket,
                "source_type": row["pack_source_type"],
                "changed_edge_count": int(row["changed_edge_count"]),
                "why_included": row["why_included"],
                "pack_split": row["split"],
                "origin_sample_id": row["origin_sample_id"],
            }
        )
    return {bucket: examples.get(bucket, []) for bucket in TARGET_BUCKETS}


def main() -> None:
    args = parse_args()

    historical_root = Path(args.historical_root)
    failure_audit_csv = Path(args.failure_audit_csv)
    failure_summary_csv = Path(args.failure_summary_csv)
    failure_examples_json = Path(args.failure_examples_json)
    policy_md = Path(args.policy_md)
    output_root = Path(args.output_root)

    policy_text = policy_md.read_text(encoding="utf-8")
    validate_policy(policy_text)
    failure_audit_df = load_failure_audit(failure_audit_csv)
    reference_meta = load_reference_meta(failure_summary_csv, failure_examples_json)
    history_rows = read_history_rows(historical_root)
    candidates_by_bucket, target_range_fps, event_seed_rows = build_historical_candidates(
        history_rows,
        failure_audit_df,
    )

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    bucket_selection_meta: dict[str, dict] = {}
    history_unique_available = {
        bucket: len({item["semantic_fingerprint"] for item in items})
        for bucket, items in candidates_by_bucket.items()
    }

    for bucket in TARGET_BUCKETS:
        history_pool = candidates_by_bucket.get(bucket, [])
        if not history_pool:
            raise ValueError(f"No historical candidates found for bucket {bucket}")

        used_semantics: set[str] = set()
        used_row_ids: set[str] = set()

        eval_history = round_robin_select(
            sort_candidates_for_eval(history_pool),
            args.eval_per_bucket,
            used_semantics=used_semantics,
            used_row_ids=used_row_ids,
        )

        remaining_history = [
            item
            for item in sort_candidates_for_train(history_pool)
            if item["semantic_fingerprint"] not in used_semantics
        ]
        train_history = round_robin_select(
            remaining_history,
            args.train_per_bucket,
            used_semantics=used_semantics,
            used_row_ids=used_row_ids,
        )

        manual_rows: list[dict] = []
        template_rows: list[dict] = []

        train_short = args.train_per_bucket - len(train_history)
        if train_short > 0:
            manual_rows = build_manual_candidates(
                bucket,
                event_seed_rows,
                blocked_range_fps=set(target_range_fps.get(bucket, set())),
                already_selected_semantics=set(used_semantics),
                needed_count=train_short,
            )
            train_manual = round_robin_select(
                sort_candidates_for_train(manual_rows),
                train_short,
                used_semantics=used_semantics,
                used_row_ids=used_row_ids,
            )
            train_history.extend(train_manual)

        train_short = args.train_per_bucket - len(train_history)
        if train_short > 0:
            template_rows = build_template_candidates(
                bucket,
                history_pool + manual_rows,
                used_texts={item["constraint_text"] for item in train_history + eval_history},
                needed_count=train_short,
            )
            train_template = round_robin_select(
                sort_candidates_for_train(template_rows),
                train_short,
                used_semantics=used_semantics,
                used_row_ids=used_row_ids,
                allow_duplicate_semantics=True,
            )
            train_history.extend(train_template)

        if len(eval_history) != args.eval_per_bucket or len(train_history) != args.train_per_bucket:
            raise ValueError(
                f"Unable to satisfy quota for {bucket}: "
                f"eval={len(eval_history)}/{args.eval_per_bucket}, "
                f"train={len(train_history)}/{args.train_per_bucket}"
            )

        eval_rows.extend(eval_history)
        train_rows.extend(train_history)
        bucket_selection_meta[bucket] = {
            "history_unique_available": int(history_unique_available.get(bucket, 0)),
            "train_selected": int(len(train_history)),
            "eval_selected": int(len(eval_history)),
            "selected_by_source_type": dict(
                Counter(item["pack_source_type"] for item in train_history + eval_history)
            ),
            "selected_failure_library_count": int(
                sum(1 for item in train_history + eval_history if item["is_failure_library_match"])
            ),
        }

    train_rows = assign_pack_fields(train_rows, "train")
    eval_rows = assign_pack_fields(eval_rows, "eval")
    all_rows = train_rows + eval_rows

    expected_per_bucket = {
        bucket: args.train_per_bucket + args.eval_per_bucket for bucket in TARGET_BUCKETS
    }
    actual_per_bucket = Counter(row["pack_bucket"] for row in all_rows)
    for bucket, expected_count in expected_per_bucket.items():
        if actual_per_bucket.get(bucket, 0) != expected_count:
            raise ValueError(
                f"Bucket count mismatch for {bucket}: "
                f"{actual_per_bucket.get(bucket, 0)} != {expected_count}"
            )

    for row in all_rows:
        if not label_is_valid(row["response_text"]):
            raise ValueError(f"Invalid response_text for {row['pack_sample_id']}")
        if infer_bucket(safe_json_loads(row["event_records_json"])) != row["pack_bucket"]:
            raise ValueError(f"Bucket drift detected for {row['pack_sample_id']}")

    train_df = pd.DataFrame(train_rows)
    eval_df = pd.DataFrame(eval_rows)
    ordered_columns = [*BASE_COLUMNS, *EXTRA_COLUMNS]
    train_df = train_df[ordered_columns]
    eval_df = eval_df[ordered_columns]

    output_train_dir = output_root / "train"
    output_eval_dir = output_root / "eval"
    output_train_dir.mkdir(parents=True, exist_ok=True)
    output_eval_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(train_df, output_train_dir / "data.parquet")
    write_parquet(eval_df, output_eval_dir / "data.parquet")

    source_breakdown_df = pd.DataFrame(
        [
            {
                "pack_sample_id": row["pack_sample_id"],
                "pack_split": row["split"],
                "bucket": row["pack_bucket"],
                "source_type": row["pack_source_type"],
                "source_detail": row["pack_source_detail"],
                "origin_dataset_root": row["origin_dataset_root"],
                "origin_split": row["origin_split"],
                "origin_sample_id": row["origin_sample_id"],
                "seed_sample_id": row["seed_sample_id"],
                "changed_edge_count": int(row["changed_edge_count"]),
                "relief_severity": row["relief_severity"],
                "is_failure_library_match": bool(row["is_failure_library_match"]),
                "is_new_supervision": bool(row["is_new_supervision"]),
                "can_direct_stage4b": bool(row["can_direct_stage4b"]),
                "text": row["constraint_text"],
                "label": row["response_text"],
            }
            for row in all_rows
        ]
    )
    source_breakdown_df.to_csv(output_root / "source_breakdown.csv", index=False, encoding="utf-8")

    examples = {
        "meta": {
            "policy_applied": STRICT_POLICY_NAME,
            "examples_per_bucket": int(args.examples_per_bucket),
            "bucket_total_counts": dict(actual_per_bucket),
        },
        "examples_by_bucket": build_examples(all_rows, args.examples_per_bucket),
    }
    for bucket in TARGET_BUCKETS:
        if len(examples["examples_by_bucket"].get(bucket, [])) < args.examples_per_bucket:
            raise ValueError(f"examples.json does not have enough examples for {bucket}")
    (output_root / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    source_type_counts = Counter(row["pack_source_type"] for row in all_rows)
    severity_counts = Counter(row["relief_severity"] for row in all_rows)
    bucket_split_counts: dict[str, dict[str, int]] = {
        bucket: {
            "train": int(sum(1 for row in train_rows if row["pack_bucket"] == bucket)),
            "eval": int(sum(1 for row in eval_rows if row["pack_bucket"] == bucket)),
        }
        for bucket in TARGET_BUCKETS
    }
    quota_met = all(
        bucket_split_counts[bucket]["train"] >= args.train_per_bucket
        and bucket_split_counts[bucket]["eval"] >= args.eval_per_bucket
        for bucket in TARGET_BUCKETS
    )
    scarcest_bucket = min(
        TARGET_BUCKETS,
        key=lambda bucket: (
            history_unique_available.get(bucket, 0),
            bucket_split_counts[bucket]["train"] + bucket_split_counts[bucket]["eval"],
        ),
    )

    summary = {
        "policy_applied": STRICT_POLICY_NAME,
        "policy_rule": STRICT_POLICY_RULE,
        "source_inputs": {
            "historical_root": str(historical_root),
            "failure_audit_csv": str(failure_audit_csv),
            "failure_summary_csv": str(failure_summary_csv),
            "failure_examples_json": str(failure_examples_json),
            "policy_md": str(policy_md),
            **reference_meta,
        },
        "quota": {
            "train_per_bucket": int(args.train_per_bucket),
            "eval_per_bucket": int(args.eval_per_bucket),
            "examples_per_bucket": int(args.examples_per_bucket),
        },
        "dataset_stats": {
            "train_rows": int(len(train_rows)),
            "eval_rows": int(len(eval_rows)),
            "bucket_counts_total": {bucket: int(actual_per_bucket.get(bucket, 0)) for bucket in TARGET_BUCKETS},
            "bucket_counts_by_split": bucket_split_counts,
            "single_edge_count": int(actual_per_bucket.get("relief_normal_single_edge", 0)),
            "short_range_count": int(actual_per_bucket.get("relief_normal_short_range", 0)),
            "multi_edge_count": int(actual_per_bucket.get("multi_edge_relief", 0)),
            "severity_counts": {
                "mild": int(severity_counts.get("mild", 0)),
                "moderate": int(severity_counts.get("moderate", 0)),
                "severe": int(severity_counts.get("severe", 0)),
            },
            "source_type_counts": dict(source_type_counts),
            "source_type_share": {
                source_type: round(count / len(all_rows), 4)
                for source_type, count in sorted(source_type_counts.items())
            },
        },
        "bucket_selection_meta": bucket_selection_meta,
        "readiness": {
            "supports_next_ceiling_eval": bool(quota_met),
            "scarcest_bucket": scarcest_bucket,
            "boundary_ambiguity_bucket": "multi_edge_relief",
            "boundary_ambiguity_reason": "multi_edge_relief 需要覆盖完整连续 3+ edge 范围，最容易在区间边界上出现歧义。",
            "severity_definition": "mild/moderate/severe 按 relief changed-edge 跨度定义：1 edge / 2 edges / 3+ edges。",
        },
        "validation": {
            "train_bucket_quota_met": bool(
                all(bucket_split_counts[bucket]["train"] == args.train_per_bucket for bucket in TARGET_BUCKETS)
            ),
            "eval_bucket_quota_met": bool(
                all(bucket_split_counts[bucket]["eval"] == args.eval_per_bucket for bucket in TARGET_BUCKETS)
            ),
            "all_labels_valid": bool(all(label_is_valid(row["response_text"]) for row in all_rows)),
            "all_rows_stage4b_ready": bool(all(row["can_direct_stage4b"] for row in all_rows)),
            "all_rows_new_supervision": bool(all(row["is_new_supervision"] for row in all_rows)),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "三类 relief supervision 是否已补齐到足以支撑下一轮 ceiling 评估："
        f"{'是' if quota_met else '否'}"
    )
    print(f"仍最稀缺的 bucket：{scarcest_bucket}")
    print("最容易有边界歧义的 bucket：multi_edge_relief（完整连续范围最容易在起止边界上漏锚或截短）")
    print(f"output_root={output_root}")


if __name__ == "__main__":
    main()
