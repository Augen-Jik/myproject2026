from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


NEUTRAL_LEVELS = {"正常", "畅通"}
RELIEF_BUCKETS = [
    "relief_normal_single_edge",
    "relief_normal_short_range",
    "multi_edge_relief",
]
REQUIRED_POLICY_PHRASES = [
    "strict-aligned labeling",
    "输出所有 active changed-event anchors",
    "anomaly_plus_normal 双事件样本是否要求同时锚定异常主事件和正常 side event：`是`",
    "multi-edge relief 是否必须覆盖完整范围：`是`",
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
OUTPUT_EXTRA_COLUMNS = [
    "stage4b_sample_id",
    "mixture_component",
    "bucket",
    "guard_focus",
    "policy_semantics",
    "source_dataset",
    "source_split",
    "source_sample_id",
    "source_origin_split",
    "source_origin_sample_id",
    "relabel_bucket",
    "relabel_changed",
    "stage4_label_changed",
    "changed_edge_count",
    "severity",
    "anchor_style",
    "high_risk",
    "why_included",
]
PACK_COLUMNS = [
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
]
ANCHOR_RE = re.compile(r"^ANCHOR\|ROAD=([^|]+)\|DIR=([^|]+)\|RANGE=([^|]+)\|LEVEL=([^|]+)$")
LENGTH_RANK = {"short": 0, "medium": 1, "long": 2, "noisy_long": 3}


def clean_text(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.replace("\\n", "\n").strip()


def safe_json_loads(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str) or not raw:
        return []
    return json.loads(raw)


def read_parquet_rows(path_or_dir: str | Path) -> tuple[list[dict[str, Any]], Path]:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pq.read_table(path).to_pylist(), path


def write_parquet_rows(rows: list[dict[str, Any]], out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    parquet_path = out_path / "data.parquet"
    pq.write_table(table, parquet_path)
    return parquet_path


def sample_id_from_split(split_name: str, index: int) -> str:
    return f"{split_name}:{index}"


def validate_policy_text(policy_text: str) -> None:
    missing = [phrase for phrase in REQUIRED_POLICY_PHRASES if phrase not in policy_text]
    if missing:
        raise ValueError(f"Policy markdown is missing required phrases: {missing}")


def parse_label_body(label: str) -> list[str]:
    lines = [line.strip() for line in clean_text(label).splitlines() if line.strip()]
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
    return body


def label_anchor_lines(label: str) -> list[str]:
    return [line for line in parse_label_body(label) if line.startswith("ANCHOR|")]


def label_anchor_set(label: str) -> set[str]:
    return set(label_anchor_lines(label))


def label_is_valid(label: str) -> bool:
    body = parse_label_body(label)
    if not body:
        return False
    if body == ["NO_ANCHOR"]:
        return True
    return all(ANCHOR_RE.match(line) for line in body)


def anchor_style(label: str) -> str:
    body = parse_label_body(label)
    if not body:
        return "invalid"
    if body == ["NO_ANCHOR"]:
        return "NO_ANCHOR"
    return "explicit_anchor"


def has_explicit_anchor_label(label: str) -> bool:
    return anchor_style(label) == "explicit_anchor"


def event_is_changed(event: dict[str, Any]) -> bool:
    try:
        weight = float(event.get("weight", 2.0))
    except (TypeError, ValueError):
        weight = 2.0
    return abs(weight - 2.0) > 0.05


def get_changed_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("active", True) and event_is_changed(event)
    ]


def changed_edge_count(events: list[dict[str, Any]]) -> int:
    return len(
        {
            str(edge_id)
            for event in events
            for edge_id in (event.get("anchor_edges", []) or [])
            if edge_id
        }
    )


def severity_from_edge_count(edge_count: int) -> str:
    if edge_count <= 0:
        return "none"
    if edge_count == 1:
        return "mild"
    if edge_count == 2:
        return "moderate"
    return "severe"


def event_anchor_line(event: dict[str, Any]) -> str:
    return (
        f"ANCHOR|ROAD={event.get('road', '')}|DIR={event.get('dir', '')}|"
        f"RANGE={event.get('range', '')}|LEVEL={event.get('level', '')}"
    )


def infer_simple_local_bucket(changed_events: list[dict[str, Any]]) -> str:
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
    if abnormal:
        return "anomaly_only"
    return "no_changed"


def has_mixed_abnormal_and_neutral(changed_events: list[dict[str, Any]]) -> bool:
    levels = {str(event.get("level", "")).strip() for event in changed_events}
    return any(level in NEUTRAL_LEVELS for level in levels) and any(level not in NEUTRAL_LEVELS for level in levels)


def render_simple_local_v2_label(changed_events: list[dict[str, Any]]) -> str:
    anchor_lines = [event_anchor_line(event) for event in changed_events]
    think_lines = [
        "<think>",
        "scene=simple_local",
        f"anchor_count={len(anchor_lines)}",
        "simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。",
        "</think>",
    ]
    return "\n".join([*think_lines, *(anchor_lines or ["NO_ANCHOR"])])


def validate_simple_local_policy(row: dict[str, Any]) -> tuple[bool, str]:
    events = safe_json_loads(row.get("event_records_json", "[]"))
    if not isinstance(events, list):
        return False, "event_records_json is not a list"
    changed_events = get_changed_events(events)
    target_lines = [event_anchor_line(event) for event in changed_events]
    actual_body = parse_label_body(str(row.get("response_text", "")))
    if not actual_body:
        return False, "label body is empty"
    if not label_is_valid(str(row.get("response_text", ""))):
        return False, "label format invalid"
    if not target_lines:
        if actual_body != ["NO_ANCHOR"]:
            return False, "no-changed simple_local sample must be NO_ANCHOR under policy-A"
        return True, "ok"
    actual_lines = label_anchor_lines(str(row.get("response_text", "")))
    if len(actual_lines) != len(target_lines):
        return False, "anchor count does not match policy-A target"
    if set(actual_lines) != set(target_lines):
        return False, "anchor set does not match policy-A target"
    return True, "ok"


def summarize_counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "") or "unknown") for row in rows).items()))


def summarize_anchor_styles(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(anchor_style(str(row.get("response_text", ""))) for row in rows).items()))

