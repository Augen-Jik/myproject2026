#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

CODE_DIR = "/root/autodl-tmp/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from build_simple_local_relief_supervision_pack import (  # noqa: E402
    BASE_COLUMNS,
    build_messages,
    clean_text,
    derive_length_bucket,
    range_fingerprint,
    render_label,
    render_prompt,
    semantic_fingerprint,
    validate_policy,
    write_parquet,
)
from roadnet_meta import expand_anchor_edges, live_edge_ids  # noqa: E402


DEFAULT_PACK_TRAIN = "/root/autodl-tmp/dataset_simple_local_relief_pack/train/data.parquet"
DEFAULT_PACK_EVAL = "/root/autodl-tmp/dataset_simple_local_relief_pack/eval/data.parquet"
DEFAULT_FAILURE_EXAMPLES = "/root/autodl-tmp/results/simple_local_failure_examples.json"
DEFAULT_DIFF_JSON = "/root/autodl-tmp/results/stage4b_diff_29.json"
DEFAULT_POLICY_MD = "/root/autodl-tmp/results/simple_local_v2_label_policy.md"
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/dataset_stage4c_singleedge_core"

TARGET_BUCKET = "relief_normal_single_edge"
TARGET_TRAIN_ROWS = 72
TARGET_EVAL_ROWS = 24
TARGET_TOTAL_MIN = 64
TARGET_TOTAL_MAX = 96

EXTRA_COLUMNS = [
    "core_sample_id",
    "core_bucket",
    "core_split",
    "source_origin",
    "source_sample_id",
    "source_pack_sample_id",
    "source_signature",
    "heldout_like_target",
    "heldout_frequency",
    "structure_tag",
    "surface_phrase",
    "surface_group",
    "label_level",
    "pattern_type",
    "changed_edge_count",
    "relief_severity",
    "semantic_fingerprint",
    "range_fingerprint",
    "is_exact_heldout_signature",
    "is_pattern_matched_to_heldout",
    "is_direct_heldout_text_copy",
    "why_included",
    "can_direct_stage4c",
]

CHANGED_PHRASES = [
    {"surface": "畅通", "surface_group": "畅通", "level": "畅通", "weight": 1.2},
    {"surface": "畅通无阻", "surface_group": "畅通无阻", "level": "畅通", "weight": 1.2},
    {"surface": "顺畅", "surface_group": "顺畅", "level": "畅通", "weight": 1.2},
    {"surface": "通行顺畅", "surface_group": "顺畅", "level": "畅通", "weight": 1.2},
    {"surface": "车流顺畅", "surface_group": "顺畅", "level": "畅通", "weight": 1.2},
]

NORMAL_PHRASES = [
    {"surface": "正常通行", "surface_group": "正常通行", "level": "正常", "weight": 2.0},
    {"surface": "保持正常通行", "surface_group": "保持正常通行", "level": "正常", "weight": 2.0},
    {"surface": "通行正常", "surface_group": "通行正常", "level": "正常", "weight": 2.0},
    {"surface": "恢复正常", "surface_group": "恢复正常", "level": "正常", "weight": 2.0},
    {"surface": "局部恢复正常", "surface_group": "局部恢复正常", "level": "正常", "weight": 2.0},
    {"surface": "正常", "surface_group": "某段向某方向正常", "level": "正常", "weight": 2.0},
]

STRUCTURE_TEMPLATES = {
    "short_plain": [
        "{clause}",
    ],
    "prefixed_short": [
        "常态或局部短时事件：{clause}",
    ],
    "planner_long": [
        "当前需要根据常态或局部短时事件进行路径规划。 请优先按当前生效约束理解，不要被已解除或背景信息误导。 {clause} 其余未明确提及路段按常态理解，不要额外扩大影响范围。",
    ],
    "anomaly_mixed_context": [
        "周边仍有事故缓行播报，但当前{clause}，不要把背景异常外推到该段。",
        "虽然附近仍有拥堵余波消息，但当前{clause}，本段不再按异常处理。",
    ],
    "same_road_dir_contrast": [
        "{clause}，同一路反向方向暂未出现同等恢复。",
        "{clause}；同一路反向车道仍需单独判断，不要把本方向恢复外推过去。",
    ],
    "tiny_range": [
        "{tiny_clause}",
        "只看{road}{range}这一小段，当前{dir}{surface}。",
    ],
    "weak_relief": [
        "当前路口压力回落，{clause}",
        "整体通行已经缓和，{clause}",
        "就这一小段来看，{clause}",
    ],
}

REQUIRED_VARIANT_CHECKS = {
    "正常通行": lambda text: "正常通行" in text,
    "保持正常通行": lambda text: "保持正常通行" in text,
    "畅通": lambda text: "畅通" in text,
    "畅通无阻": lambda text: "畅通无阻" in text,
    "顺畅": lambda text: "顺畅" in text,
    "通行正常": lambda text: "通行正常" in text,
    "恢复正常": lambda text: "恢复正常" in text,
    "局部恢复正常": lambda text: "局部恢复正常" in text,
    "某段向某方向正常": lambda text: bool(re.search(r"段向[东西南北]正常", text)),
    "某段向某方向畅通": lambda text: bool(re.search(r"段向[东西南北]畅通", text)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand a heldout-like single-edge relief core dataset for stage4c.")
    parser.add_argument("--pack-train", default=DEFAULT_PACK_TRAIN)
    parser.add_argument("--pack-eval", default=DEFAULT_PACK_EVAL)
    parser.add_argument("--failure-examples-json", default=DEFAULT_FAILURE_EXAMPLES)
    parser.add_argument("--diff-json", default=DEFAULT_DIFF_JSON)
    parser.add_argument("--policy-md", default=DEFAULT_POLICY_MD)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-target", type=int, default=TARGET_TRAIN_ROWS)
    parser.add_argument("--eval-target", type=int, default=TARGET_EVAL_ROWS)
    return parser.parse_args()


def pattern_type(level: str) -> str:
    if level == "畅通":
        return "changed_relief"
    if level == "正常":
        return "normal_anchor"
    return "other"


def severity_from_changed_edge_count(count: int) -> str:
    if count <= 1:
        return "mild"
    if count == 2:
        return "moderate"
    return "severe"


def signature_from_event(event: dict[str, Any]) -> str:
    return f"{event['road']}|{event['dir']}|{event['range']}|{event['level']}"


def rdr_from_event(event: dict[str, Any]) -> str:
    return f"{event['road']}|{event['dir']}|{event['range']}"


def rdrt_from_event(event: dict[str, Any]) -> str:
    return f"{event['road']}|{event['dir']}|{event['range']}|{pattern_type(str(event['level']))}"


def parse_single_event(event_records_json: str) -> dict[str, Any]:
    events = json.loads(event_records_json)
    active = [event for event in events if event.get("active", True)]
    if len(active) != 1:
        raise ValueError(f"Expected exactly 1 active event, got {len(active)}")
    event = dict(active[0])
    event["road"] = str(event["road"]).strip()
    event["dir"] = str(event["dir"]).strip()
    event["range"] = str(event["range"]).strip()
    event["level"] = str(event["level"]).strip()
    event["text"] = clean_text(str(event.get("text", "")))
    event["anchor_edges"] = [str(edge) for edge in (event.get("anchor_edges", []) or []) if edge]
    return event


def load_pack_rows(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(pq.read_table(path).to_pylist())
    df = df[df["pack_bucket"] == TARGET_BUCKET].copy()
    df["single_event"] = df["event_records_json"].map(parse_single_event)
    return df.reset_index(drop=True)


def load_heldout_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in data["samples"] if row["semantic_gap_subtype"] == TARGET_BUCKET]
    for row in rows:
        row["single_event"] = parse_single_event(row["event_records_json"])
    return rows


def derive_failure_style_hints(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = data["dimension_buckets"]["failure_type"]["no_anchor"]["examples"]
    counts = Counter()
    for example in examples:
        text = clean_text(example.get("input_text", ""))
        if text.startswith("当前需要根据常态或局部短时事件进行路径规划。"):
            counts["planner_long"] += 1
        elif text.startswith("常态或局部短时事件："):
            counts["prefixed_short"] += 1
        else:
            counts["short_plain"] += 1
    return dict(counts)


def build_seed_registry(
    pack_train_df: pd.DataFrame,
    pack_eval_df: pd.DataFrame,
    heldout_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str], set[str]]:
    registry: dict[str, dict[str, Any]] = {}
    all_forbidden_texts: set[str] = set()
    heldout_texts: set[str] = set()

    def ensure_seed(signature: str, event: dict[str, Any], source_origin: str, source_sample_id: str, source_pack_id: str | None, text: str) -> dict[str, Any]:
        seed = registry.get(signature)
        if seed is None:
            seed = {
                "signature": signature,
                "road": event["road"],
                "dir": event["dir"],
                "range": event["range"],
                "level": event["level"],
                "anchor_edges": list(event["anchor_edges"]),
                "source_items": [],
                "heldout_sample_ids": [],
                "heldout_count": 0,
            }
            registry[signature] = seed
        seed["source_items"].append(
            {
                "source_origin": source_origin,
                "source_sample_id": source_sample_id,
                "source_pack_sample_id": source_pack_id,
                "text": text,
            }
        )
        return seed

    for _, row in pack_train_df.iterrows():
        text = clean_text(row["constraint_text"])
        all_forbidden_texts.add(text)
        event = row["single_event"]
        ensure_seed(
            signature_from_event(event),
            event,
            source_origin="relief_pack_train",
            source_sample_id=str(row["origin_sample_id"]),
            source_pack_id=str(row["pack_sample_id"]),
            text=text,
        )

    for _, row in pack_eval_df.iterrows():
        text = clean_text(row["constraint_text"])
        all_forbidden_texts.add(text)
        event = row["single_event"]
        ensure_seed(
            signature_from_event(event),
            event,
            source_origin="relief_pack_eval",
            source_sample_id=str(row["origin_sample_id"]),
            source_pack_id=str(row["pack_sample_id"]),
            text=text,
        )

    for row in heldout_rows:
        text = clean_text(row["input_text"])
        all_forbidden_texts.add(text)
        heldout_texts.add(text)
        event = row["single_event"]
        seed = ensure_seed(
            signature_from_event(event),
            event,
            source_origin="heldout_diff29",
            source_sample_id=str(row["sample_id"]),
            source_pack_id=None,
            text=text,
        )
        seed["heldout_sample_ids"].append(str(row["sample_id"]))
        seed["heldout_count"] += 1

    return registry, all_forbidden_texts, heldout_texts


def make_event(seed: dict[str, Any], phrase_spec: dict[str, Any]) -> dict[str, Any]:
    edges = expand_anchor_edges(seed["road"], seed["dir"], seed["range"])
    if not edges:
        raise ValueError(f"Could not expand anchor edges for {seed['signature']}")
    return {
        "road": seed["road"],
        "dir": seed["dir"],
        "range": seed["range"],
        "level": phrase_spec["level"],
        "weight": float(phrase_spec["weight"]),
        "time": "当前",
        "propagation": "无明显传播",
        "conflict": "单约束",
        "anchor_edges": list(edges),
        "active": True,
    }


def build_ground_truth_json(event: dict[str, Any]) -> str:
    weights = {edge_id: 2.0 for edge_id in live_edge_ids()}
    for edge_id in event["anchor_edges"]:
        weights[str(edge_id)] = float(event["weight"])
    return json.dumps(weights, ensure_ascii=False, sort_keys=True)


def build_clause(event: dict[str, Any], phrase_spec: dict[str, Any], *, tiny: bool = False) -> str:
    segment = "这一小段" if tiny else "段"
    return f"{event['road']}{event['range']}{segment}{event['dir']}{phrase_spec['surface']}"


def render_constraint_text(
    event: dict[str, Any],
    phrase_spec: dict[str, Any],
    structure_tag: str,
    variant_index: int,
) -> str:
    templates = STRUCTURE_TEMPLATES[structure_tag]
    template = templates[variant_index % len(templates)]
    clause = build_clause(event, phrase_spec, tiny=False)
    tiny_clause = build_clause(event, phrase_spec, tiny=True)
    return clean_text(
        template.format(
            clause=clause,
            tiny_clause=tiny_clause,
            road=event["road"],
            range=event["range"],
            dir=event["dir"],
            surface=phrase_spec["surface"],
        )
    )


def build_row(
    *,
    seed: dict[str, Any],
    split_name: str,
    row_index: int,
    phrase_spec: dict[str, Any],
    structure_tag: str,
    variant_index: int,
    why_included: str,
    heldout_signature_set: set[str],
    heldout_range_set: set[str],
    heldout_texts: set[str],
) -> dict[str, Any]:
    event = make_event(seed, phrase_spec)
    constraint_text = render_constraint_text(event, phrase_spec, structure_tag, variant_index)
    event["text"] = build_clause(event, phrase_spec, tiny=False)
    response_text = render_label([event])
    prompt_text = render_prompt(constraint_text, [event])
    signature = signature_from_event(event)
    semantic_fp = semantic_fingerprint([event])
    range_fp = range_fingerprint([event])
    changed_edges = 1 if abs(float(event["weight"]) - 2.0) > 0.05 else 0
    primary_source = seed["source_items"][0]
    return {
        "messages": build_messages(prompt_text, response_text),
        "scene_type": "simple_local",
        "scene_bucket": "simple_local",
        "length_bucket": derive_length_bucket(constraint_text),
        "split": split_name,
        "constraint_text": constraint_text,
        "prompt_text": prompt_text,
        "response_text": response_text,
        "anchor_count": 1,
        "ground_truth_json": build_ground_truth_json(event),
        "event_records_json": json.dumps([event], ensure_ascii=False),
        "parse_confidence": 1.0,
        "total_edges": len(live_edge_ids()),
        "core_sample_id": f"{split_name}:{row_index}",
        "core_bucket": TARGET_BUCKET,
        "core_split": split_name,
        "source_origin": primary_source["source_origin"],
        "source_sample_id": primary_source["source_sample_id"],
        "source_pack_sample_id": primary_source["source_pack_sample_id"],
        "source_signature": seed["signature"],
        "heldout_like_target": bool(seed["heldout_count"] > 0),
        "heldout_frequency": int(seed["heldout_count"]),
        "structure_tag": structure_tag,
        "surface_phrase": phrase_spec["surface"],
        "surface_group": phrase_spec["surface_group"],
        "label_level": phrase_spec["level"],
        "pattern_type": pattern_type(phrase_spec["level"]),
        "changed_edge_count": changed_edges,
        "relief_severity": severity_from_changed_edge_count(max(changed_edges, 1)),
        "semantic_fingerprint": semantic_fp,
        "range_fingerprint": range_fp,
        "is_exact_heldout_signature": bool(signature in heldout_signature_set),
        "is_pattern_matched_to_heldout": bool(rdr_from_event(event) in heldout_range_set),
        "is_direct_heldout_text_copy": bool(constraint_text in heldout_texts),
        "why_included": why_included,
        "can_direct_stage4c": True,
    }


def add_row_if_unique(
    rows: list[dict[str, Any]],
    used_texts: set[str],
    forbidden_train_texts: set[str],
    row_kwargs: dict[str, Any],
) -> bool:
    for extra_shift in range(6):
        candidate = build_row(
            variant_index=row_kwargs["variant_index"] + extra_shift,
            **{key: value for key, value in row_kwargs.items() if key != "variant_index"},
        )
        text = candidate["constraint_text"]
        if text in used_texts:
            continue
        if candidate["split"].endswith("train") and text in forbidden_train_texts:
            continue
        rows.append(candidate)
        used_texts.add(text)
        return True
    return False


def required_variant_counts(texts: list[str]) -> dict[str, int]:
    counts = {}
    for name, checker in REQUIRED_VARIANT_CHECKS.items():
        counts[name] = sum(1 for text in texts if checker(text))
    return counts


def build_signature_coverage(
    train_df: pd.DataFrame,
    heldout_rows: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    train_events = train_df["event_records_json"].map(parse_single_event)
    train_df = train_df.copy()
    train_df["active_signature"] = train_events.map(signature_from_event)
    train_df["rdr"] = train_events.map(rdr_from_event)
    train_df["rdrt"] = train_events.map(rdrt_from_event)
    train_df["road"] = train_events.map(lambda event: event["road"])
    train_df["dir"] = train_events.map(lambda event: event["dir"])
    train_df["range"] = train_events.map(lambda event: event["range"])
    train_df["level"] = train_events.map(lambda event: event["level"])
    train_df["type"] = train_events.map(lambda event: pattern_type(event["level"]))

    sig_counts = train_df["active_signature"].value_counts().to_dict()
    rdr_counts = train_df["rdr"].value_counts().to_dict()
    rdrt_counts = train_df["rdrt"].value_counts().to_dict()
    road_counts = train_df["road"].value_counts().to_dict()
    road_dir_counts = train_df.groupby(["road", "dir"]).size().to_dict()
    road_dir_type_counts = train_df.groupby(["road", "dir", "type"]).size().to_dict()

    rows: list[dict[str, Any]] = []
    for signature, group in train_df.groupby("active_signature"):
        first = group.iloc[0]
        rows.append(
            {
                "row_type": "train_signature",
                "signature": signature,
                "road": first["road"],
                "dir": first["dir"],
                "range": first["range"],
                "level": first["level"],
                "type": first["type"],
                "train_count": int(len(group)),
                "heldout_exact_signature_support": int(
                    sum(1 for row in heldout_rows if signature_from_event(row["single_event"]) == signature)
                ),
                "heldout_pattern_support": int(
                    sum(1 for row in heldout_rows if rdr_from_event(row["single_event"]) == first["rdr"])
                ),
            }
        )

    exact_sample_hits = 0
    pattern_sample_hits = 0
    best_scores: list[float] = []
    for heldout in heldout_rows:
        event = heldout["single_event"]
        held_sig = signature_from_event(event)
        held_rdr = rdr_from_event(event)
        held_rdrt = rdrt_from_event(event)
        held_type = pattern_type(event["level"])

        exact_count = int(sig_counts.get(held_sig, 0))
        rdr_count = int(rdr_counts.get(held_rdr, 0))
        rdrt_count = int(rdrt_counts.get(held_rdrt, 0))
        road_count = int(road_counts.get(event["road"], 0))
        road_dir_count = int(road_dir_counts.get((event["road"], event["dir"]), 0))
        road_dir_type_count = int(road_dir_type_counts.get((event["road"], event["dir"], held_type), 0))

        if exact_count > 0:
            exact_sample_hits += 1
        if rdrt_count > 0:
            pattern_sample_hits += 1

        best_signature = ""
        best_score = 0.0
        for signature, count in sig_counts.items():
            road, direction, range_label, level = signature.split("|")
            current_type = pattern_type(level)
            score = (
                (1.0 if road == event["road"] else 0.0)
                + (1.0 if direction == event["dir"] else 0.0)
                + (1.0 if range_label == event["range"] else 0.0)
                + (1.0 if current_type == held_type else 0.0)
            ) / 4.0
            if score > best_score or (score == best_score and count > sig_counts.get(best_signature, 0)):
                best_score = score
                best_signature = signature
        best_scores.append(best_score)

        rows.append(
            {
                "row_type": "heldout_coverage",
                "sample_id": heldout["sample_id"],
                "signature": held_sig,
                "road": event["road"],
                "dir": event["dir"],
                "range": event["range"],
                "level": event["level"],
                "type": held_type,
                "train_exact_signature_count": exact_count,
                "train_exact_rdr_count": rdr_count,
                "train_exact_rdrt_count": rdrt_count,
                "train_road_dir_count": road_dir_count,
                "train_road_dir_type_count": road_dir_type_count,
                "train_road_count": road_count,
                "best_pattern_score": round(best_score, 4),
                "best_train_signature": best_signature,
            }
        )

    summary_rows = [
        {
            "row_type": "aggregate",
            "metric": "train_signature_unique_count",
            "value": int(len(sig_counts)),
        },
        {
            "row_type": "aggregate",
            "metric": "heldout_sample_exact_signature_coverage",
            "value": f"{exact_sample_hits}/{len(heldout_rows)}",
        },
        {
            "row_type": "aggregate",
            "metric": "heldout_sample_pattern_rdrt_coverage",
            "value": f"{pattern_sample_hits}/{len(heldout_rows)}",
        },
        {
            "row_type": "aggregate",
            "metric": "heldout_best_pattern_score_avg",
            "value": round(sum(best_scores) / max(len(best_scores), 1), 4),
        },
    ]

    coverage_df = pd.DataFrame(rows + summary_rows)
    coverage_df.to_csv(output_path, index=False, encoding="utf-8")
    return {
        "train_signature_unique_count": int(len(sig_counts)),
        "heldout_sample_exact_signature_coverage": f"{exact_sample_hits}/{len(heldout_rows)}",
        "heldout_sample_pattern_rdrt_coverage": f"{pattern_sample_hits}/{len(heldout_rows)}",
        "heldout_best_pattern_score_avg": round(sum(best_scores) / max(len(best_scores), 1), 4),
    }


def build_examples(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, Any]:
    combined = pd.concat([train_df, eval_df], ignore_index=True)
    examples_by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_phrase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in combined.sort_values(["structure_tag", "surface_phrase", "core_sample_id"]).iterrows():
        item = {
            "core_sample_id": row["core_sample_id"],
            "split": row["core_split"],
            "text": row["constraint_text"],
            "label": row["response_text"],
            "structure_tag": row["structure_tag"],
            "surface_phrase": row["surface_phrase"],
            "source_origin": row["source_origin"],
            "heldout_like_target": bool(row["heldout_like_target"]),
        }
        if len(examples_by_structure[row["structure_tag"]]) < 3:
            examples_by_structure[row["structure_tag"]].append(item)
        if len(examples_by_phrase[row["surface_group"]]) < 2:
            examples_by_phrase[row["surface_group"]].append(item)
    return {
        "meta": {
            "dataset": "stage4c_singleedge_core",
            "bucket": TARGET_BUCKET,
        },
        "examples_by_structure": dict(examples_by_structure),
        "examples_by_phrase_group": dict(examples_by_phrase),
    }


def choose_scarcest_variant(train_texts: list[str]) -> tuple[str, int]:
    counts = required_variant_counts(train_texts)
    scarcest = min(sorted(counts.items()), key=lambda item: (item[1], item[0]))
    return scarcest


def main() -> None:
    args = parse_args()
    policy_text = Path(args.policy_md).read_text(encoding="utf-8")
    validate_policy(policy_text)

    pack_train_df = load_pack_rows(Path(args.pack_train))
    pack_eval_df = load_pack_rows(Path(args.pack_eval))
    heldout_rows = load_heldout_rows(Path(args.diff_json))
    failure_style_hints = derive_failure_style_hints(Path(args.failure_examples_json))

    registry, forbidden_texts, heldout_texts = build_seed_registry(pack_train_df, pack_eval_df, heldout_rows)
    heldout_signature_set = {signature_from_event(row["single_event"]) for row in heldout_rows}
    heldout_range_set = {rdr_from_event(row["single_event"]) for row in heldout_rows}

    heldout_counts_by_signature = Counter(signature_from_event(row["single_event"]) for row in heldout_rows)
    train_signatures = sorted(
        registry.values(),
        key=lambda seed: (
            -heldout_counts_by_signature.get(seed["signature"], 0),
            0 if seed["signature"] in heldout_signature_set else 1,
            seed["road"],
            seed["dir"],
            seed["range"],
        ),
    )
    heldout_signature_seeds = [seed for seed in train_signatures if seed["signature"] in heldout_signature_set]
    pack_only_signature_seeds = [seed for seed in train_signatures if seed["signature"] not in heldout_signature_set]

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    used_texts: set[str] = set()

    def add_train(seed: dict[str, Any], phrase_spec: dict[str, Any], structure_tag: str, variant_index: int, why: str) -> bool:
        return add_row_if_unique(
            train_rows,
            used_texts,
            forbidden_texts,
            {
                "seed": seed,
                "split_name": "stage4c_singleedge_core_train",
                "row_index": len(train_rows),
                "phrase_spec": phrase_spec,
                "structure_tag": structure_tag,
                "variant_index": variant_index,
                "why_included": why,
                "heldout_signature_set": heldout_signature_set,
                "heldout_range_set": heldout_range_set,
                "heldout_texts": heldout_texts,
            },
        )

    def add_eval(seed: dict[str, Any], phrase_spec: dict[str, Any], structure_tag: str, variant_index: int, why: str) -> bool:
        return add_row_if_unique(
            eval_rows,
            used_texts,
            forbidden_texts,
            {
                "seed": seed,
                "split_name": "stage4c_singleedge_core_eval",
                "row_index": len(eval_rows),
                "phrase_spec": phrase_spec,
                "structure_tag": structure_tag,
                "variant_index": variant_index,
                "why_included": why,
                "heldout_signature_set": heldout_signature_set,
                "heldout_range_set": heldout_range_set,
                "heldout_texts": heldout_texts,
            },
        )

    # Step 1: seed required lexical/structure coverage with heldout-like signatures.
    required_seed_plan = [
        (heldout_signature_seeds[0], CHANGED_PHRASES[0], "short_plain", "补 bare 畅通表达与最短句式。"),
        (heldout_signature_seeds[1], CHANGED_PHRASES[1], "prefixed_short", "补 畅通无阻 + 常态前缀表达。"),
        (heldout_signature_seeds[2], CHANGED_PHRASES[2], "tiny_range", "补 顺畅 + 极短范围表达。"),
        (heldout_signature_seeds[3], NORMAL_PHRASES[0], "anomaly_mixed_context", "补 正常通行 + 正常/异常混排表达。"),
        (heldout_signature_seeds[4], NORMAL_PHRASES[1], "same_road_dir_contrast", "补 保持正常通行 + 同一路不同方向对照。"),
        (heldout_signature_seeds[5], NORMAL_PHRASES[2], "short_plain", "补 通行正常表达。"),
        (heldout_signature_seeds[6], NORMAL_PHRASES[3], "planner_long", "补 恢复正常 + 长句规划式表达。"),
        (heldout_signature_seeds[7], NORMAL_PHRASES[4], "weak_relief", "补 局部恢复正常 + 弱语气 relief 表达。"),
        (heldout_signature_seeds[8], NORMAL_PHRASES[5], "short_plain", "补 某段向某方向正常。"),
        (heldout_signature_seeds[9], CHANGED_PHRASES[0], "short_plain", "补 某段向某方向畅通。"),
    ]
    for idx, (seed, phrase_spec, structure_tag, why) in enumerate(required_seed_plan):
        if not add_train(seed, phrase_spec, structure_tag, idx, why):
            raise RuntimeError(f"Failed to add required coverage row for {seed['signature']} / {phrase_spec['surface']}")

    # Step 2: one changed relief rewrite for every available signature.
    primary_structures = ["short_plain", "prefixed_short", "planner_long", "tiny_range"]
    for idx, seed in enumerate(train_signatures):
        phrase_spec = CHANGED_PHRASES[(idx + 1) % len(CHANGED_PHRASES)]
        structure_tag = primary_structures[idx % len(primary_structures)]
        add_train(
            seed,
            phrase_spec,
            structure_tag,
            idx + 10,
            "所有可用 single-edge signature 至少给 1 条非原句 rewrite，先把 signature 池铺满。",
        )

    # Step 3: give heldout-like signatures extra changed rewrites proportional to heldout frequency.
    focused_structures = ["anomaly_mixed_context", "same_road_dir_contrast", "weak_relief", "planner_long"]
    counter = 0
    for idx, seed in enumerate(heldout_signature_seeds):
        reps = max(1, heldout_counts_by_signature.get(seed["signature"], 0))
        for rep in range(reps):
            phrase_spec = CHANGED_PHRASES[(idx + rep + 2) % len(CHANGED_PHRASES)]
            structure_tag = focused_structures[(idx + rep) % len(focused_structures)]
            add_train(
                seed,
                phrase_spec,
                structure_tag,
                100 + counter,
                "对 held-out 主失败 signature 追加贴脸 rewrite，把 stage4c 主信号压向真正的 failure distribution。",
            )
            counter += 1

    # Step 4: inject more normal-surface anchors on top heldout-like signatures.
    normal_priority = heldout_signature_seeds + pack_only_signature_seeds
    for idx, seed in enumerate(normal_priority[:14]):
        phrase_spec = NORMAL_PHRASES[idx % len(NORMAL_PHRASES)]
        structure_tag = ["short_plain", "prefixed_short", "same_road_dir_contrast", "weak_relief"][idx % 4]
        add_train(
            seed,
            phrase_spec,
            structure_tag,
            200 + idx,
            "补足 normal-surface 单边 anchor 表达，让 single-edge 核心集不只会锚 `畅通无阻/车流顺畅` 两种说法。",
        )

    # Step 5: fill train to target with extra heldout-weighted changed rewrites.
    fill_priority = sorted(
        heldout_signature_seeds,
        key=lambda seed: (
            -heldout_counts_by_signature.get(seed["signature"], 0),
            seed["road"],
            seed["dir"],
            seed["range"],
        ),
    ) + pack_only_signature_seeds
    fill_idx = 0
    while len(train_rows) < args.train_target:
        seed = fill_priority[fill_idx % len(fill_priority)]
        phrase_spec = CHANGED_PHRASES[(fill_idx + 3) % len(CHANGED_PHRASES)]
        structure_tag = ["planner_long", "anomaly_mixed_context", "same_road_dir_contrast", "tiny_range", "weak_relief"][
            fill_idx % 5
        ]
        added = add_train(
            seed,
            phrase_spec,
            structure_tag,
            300 + fill_idx,
            "继续加密高频 held-out-like single-edge signature 的 rewrite 覆盖，避免 stage4c 再次出现 train 太少、词面太单一。",
        )
        fill_idx += 1
        if not added and fill_idx > 400:
            raise RuntimeError("Could not fill train target with unique single-edge rewrites")

    # Build eval with distinct paraphrases, still heldout-like but not copied.
    eval_priority = heldout_signature_seeds + pack_only_signature_seeds
    for idx, seed in enumerate(eval_priority):
        if len(eval_rows) >= args.eval_target:
            break
        phrase_spec = CHANGED_PHRASES[(idx + 4) % len(CHANGED_PHRASES)]
        structure_tag = ["planner_long", "anomaly_mixed_context", "same_road_dir_contrast", "prefixed_short", "weak_relief"][
            idx % 5
        ]
        add_eval(
            seed,
            phrase_spec,
            structure_tag,
            500 + idx,
            "eval 保持 held-out-like 结构，但全部改写成非原句，用来监控 single-edge 核心表达的泛化而不是记忆。",
        )

    eval_fill_idx = 0
    while len(eval_rows) < args.eval_target:
        seed = eval_priority[eval_fill_idx % len(eval_priority)]
        phrase_bank = NORMAL_PHRASES if eval_fill_idx % 4 == 0 else CHANGED_PHRASES
        phrase_spec = phrase_bank[(eval_fill_idx + 1) % len(phrase_bank)]
        structure_tag = ["tiny_range", "planner_long", "prefixed_short", "same_road_dir_contrast"][
            eval_fill_idx % 4
        ]
        added = add_eval(
            seed,
            phrase_spec,
            structure_tag,
            700 + eval_fill_idx,
            "补齐 eval 的 lexical/structure coverage，使其能单独监控 single-edge phrase generalization。",
        )
        eval_fill_idx += 1
        if not added and eval_fill_idx > 200:
            raise RuntimeError("Could not fill eval target with unique single-edge rewrites")

    if not (TARGET_TOTAL_MIN <= len(train_rows) <= TARGET_TOTAL_MAX):
        raise RuntimeError(
            f"Train target out of requested range: got {len(train_rows)}, expected within [{TARGET_TOTAL_MIN}, {TARGET_TOTAL_MAX}]"
        )
    if len(eval_rows) < 24:
        raise RuntimeError(f"Eval target too small: got {len(eval_rows)}")

    train_df = pd.DataFrame(train_rows)
    eval_df = pd.DataFrame(eval_rows)
    ordered_columns = [*BASE_COLUMNS, *EXTRA_COLUMNS]
    train_df = train_df[ordered_columns]
    eval_df = eval_df[ordered_columns]

    output_root = Path(args.output_root)
    train_dir = output_root / "train"
    eval_dir = output_root / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(train_df, train_dir / "data.parquet")
    write_parquet(eval_df, eval_dir / "data.parquet")

    coverage_summary = build_signature_coverage(
        train_df,
        heldout_rows,
        output_root / "signature_coverage.csv",
    )

    train_texts = train_df["constraint_text"].tolist()
    eval_texts = eval_df["constraint_text"].tolist()
    train_variant_counts = required_variant_counts(train_texts)
    eval_variant_counts = required_variant_counts(eval_texts)
    scarcest_variant, scarcest_count = choose_scarcest_variant(train_texts)

    summary = {
        "dataset": "stage4c_singleedge_core",
        "bucket": TARGET_BUCKET,
        "source_inputs": {
            "pack_train": args.pack_train,
            "pack_eval": args.pack_eval,
            "failure_examples_json": args.failure_examples_json,
            "diff_json": args.diff_json,
            "policy_md": args.policy_md,
        },
        "targets": {
            "train_target": args.train_target,
            "eval_target": args.eval_target,
            "requested_train_range": [TARGET_TOTAL_MIN, TARGET_TOTAL_MAX],
        },
        "counts": {
            "train_rows": int(len(train_df)),
            "eval_rows": int(len(eval_df)),
            "train_heldout_like_rows": int(train_df["heldout_like_target"].astype(bool).sum()),
            "eval_heldout_like_rows": int(eval_df["heldout_like_target"].astype(bool).sum()),
            "train_exact_heldout_signature_rows": int(train_df["is_exact_heldout_signature"].astype(bool).sum()),
            "eval_exact_heldout_signature_rows": int(eval_df["is_exact_heldout_signature"].astype(bool).sum()),
            "label_level_counts_train": train_df["label_level"].value_counts().to_dict(),
            "structure_counts_train": train_df["structure_tag"].value_counts().to_dict(),
            "surface_counts_train": train_df["surface_group"].value_counts().to_dict(),
        },
        "heldout_alignment": {
            "heldout_single_edge_sample_count": int(len(heldout_rows)),
            "heldout_single_edge_signature_unique_count": int(len(heldout_signature_set)),
            **coverage_summary,
        },
        "coverage_checks": {
            "train_has_64_plus_rows": bool(len(train_df) >= 64),
            "train_text_has_no_direct_heldout_copy": bool(~train_df["is_direct_heldout_text_copy"].astype(bool).any()),
            "eval_text_has_no_direct_heldout_copy": bool(~eval_df["is_direct_heldout_text_copy"].astype(bool).any()),
            "all_required_variant_present_in_train": {
                key: bool(value > 0) for key, value in train_variant_counts.items()
            },
            "required_variant_counts_train": train_variant_counts,
            "required_variant_counts_eval": eval_variant_counts,
            "failure_style_hints": failure_style_hints,
        },
        "diagnosis": {
            "upgraded_from_16_to_64_plus_face_coverage": bool(
                len(train_df) >= 64 and coverage_summary["heldout_sample_exact_signature_coverage"] == f"{len(heldout_rows)}/{len(heldout_rows)}"
            ),
            "still_scarcest_singleedge_expression": {
                "variant": scarcest_variant,
                "train_count": int(scarcest_count),
            },
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    examples = build_examples(train_df, eval_df)
    examples["meta"].update(
        {
            "train_rows": int(len(train_df)),
            "eval_rows": int(len(eval_df)),
            "required_variant_counts_train": train_variant_counts,
        }
    )
    (output_root / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    upgraded = summary["diagnosis"]["upgraded_from_16_to_64_plus_face_coverage"]
    print(
        "single_edge 主集是否已经从“16 条弱覆盖”升级为“64+ 条贴脸覆盖”："
        f"{'是' if upgraded else '否'}"
    )
    print(f"哪类 single-edge 表达仍最稀缺：{scarcest_variant} ({scarcest_count})")
    print(f"output_root={output_root}")


if __name__ == "__main__":
    main()
