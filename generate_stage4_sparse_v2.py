#!/usr/bin/env python3
"""Generate Stage4 targeted repair data for Sparse-LoRA-v2."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = "/root/autodl-tmp"
CODE_DIR = f"{ROOT_DIR}/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from generate_dataset_sparse import SparseDatasetBuilder, save_parquet, summarize_samples  # noqa: E402
from roadnet_meta import COL_NAMES, NORMAL_DEFAULT_WEIGHT, ROW_NAMES, live_edge_ids  # noqa: E402
from scenarios import scene_bucket  # noqa: E402
from sparse_eval_strict import (  # noqa: E402
    changed_edge_ids,
    evaluate_rows,
    load_model,
    load_rows,
    load_tokenizer,
    unload_model,
)
from sparse_utils import build_anchor_response, format_sparse_prompt, parse_sparse_output_bundle, validate_sparse_response  # noqa: E402


def row_to_sample(row: dict, *, split_name: str) -> dict[str, Any]:
    messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    return {
        "messages": messages,
        "scene_type": row["scene_type"],
        "scene_bucket": row["scene_bucket"],
        "length_bucket": row["length_bucket"],
        "split": split_name,
        "constraint_text": row["constraint_text"],
        "prompt_text": row["prompt_text"],
        "response_text": row["response_text"],
        "anchor_count": int(row["anchor_count"]),
        "ground_truth_json": row["ground_truth_json"],
        "event_records_json": row["event_records_json"],
        "parse_confidence": float(row["parse_confidence"]),
        "total_edges": int(row["total_edges"]),
    }


def sample_key(sample: dict[str, Any]) -> str:
    return str(sample["prompt_text"]).strip()


def has_changed_edges(sample: dict[str, Any]) -> bool:
    truth = json.loads(sample["ground_truth_json"])
    return bool(changed_edge_ids(truth))


def has_explicit_anchor_label(sample: dict[str, Any]) -> bool:
    return int(sample.get("anchor_count", 0)) > 0 and "NO_ANCHOR" not in str(sample.get("response_text", ""))


def is_focus_scene(sample: dict[str, Any]) -> bool:
    return sample.get("scene_type") in {"simple_local", "directional_asymmetry"}


def make_custom_sample(
    builder: SparseDatasetBuilder,
    *,
    split_name: str,
    scene_type: str,
    length_bucket: str,
    records: list[dict],
    narrative: str | None = None,
) -> dict[str, Any]:
    narrative = narrative or builder._render_constraint(scene_type, records, length_bucket)
    prompt = format_sparse_prompt(scene_type=scene_type, events=records, narrative=narrative)
    response, anchor_count = build_anchor_response(records, scene_type=scene_type)
    ok, issues = validate_sparse_response(response)
    if not ok:
        raise ValueError(f"invalid handcrafted sparse response: {issues}")

    parsed = parse_sparse_output_bundle(response, scene_type=scene_type)
    weights = {edge_id: NORMAL_DEFAULT_WEIGHT for edge_id in builder.edge_ids}
    for record in records:
        builder._apply_event(weights, record)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "scene_type": scene_type,
        "scene_bucket": scene_bucket(scene_type),
        "length_bucket": length_bucket,
        "split": split_name,
        "constraint_text": narrative,
        "prompt_text": prompt,
        "response_text": response,
        "anchor_count": anchor_count,
        "ground_truth_json": json.dumps(weights, ensure_ascii=False),
        "event_records_json": json.dumps(records, ensure_ascii=False),
        "parse_confidence": parsed.get("parse_confidence", 0.0),
        "total_edges": len(builder.edge_ids),
    }


def get_meta_for_road_dir(builder: SparseDatasetBuilder, road: str, dir_label: str) -> dict:
    for edge_id in builder.edge_ids:
        meta = builder.edge_meta[edge_id]
        if meta["road"] == road and meta["dir"] == dir_label:
            return dict(meta)
    raise KeyError(f"meta not found for {road} {dir_label}")


def make_explicit_record(
    builder: SparseDatasetBuilder,
    *,
    road: str,
    dir_label: str,
    range_label: str,
    level: str,
    text: str,
    time_label: str = "未指明",
    propagation: str = "无明显传播",
    conflict: str = "单约束",
    active: bool = True,
    propagation_strength: float = 0.0,
) -> dict[str, Any]:
    meta = get_meta_for_road_dir(builder, road, dir_label)
    if range_label == "局部":
        anchor_edges = [edge_id for edge_id in builder.edge_ids if builder.edge_meta[edge_id]["road"] == road and builder.edge_meta[edge_id]["dir"] == dir_label][:1]
    elif range_label == "全线":
        anchor_edges = [
            edge_id
            for edge_id in builder.edge_ids
            if builder.edge_meta[edge_id]["road"] == road and builder.edge_meta[edge_id]["dir"] == dir_label
        ]
    else:
        start_name, end_name = range_label.split("至", 1)
        if meta["axis"] == "H":
            start_idx = COL_NAMES.index(start_name)
            end_idx = COL_NAMES.index(end_name)
            anchor_edges = [
                edge_id
                for edge_id in builder.edge_ids
                if builder.edge_meta[edge_id]["road"] == road
                and builder.edge_meta[edge_id]["dir"] == dir_label
                and start_idx <= int(builder.edge_meta[edge_id]["col"]) < end_idx
            ]
        else:
            start_idx = ROW_NAMES.index(start_name)
            end_idx = ROW_NAMES.index(end_name)
            anchor_edges = [
                edge_id
                for edge_id in builder.edge_ids
                if builder.edge_meta[edge_id]["road"] == road
                and builder.edge_meta[edge_id]["dir"] == dir_label
                and start_idx <= int(builder.edge_meta[edge_id]["row"]) < end_idx
            ]
    return {
        "road": road,
        "dir": dir_label,
        "range": range_label,
        "level": level,
        "weight": builder._make_event(meta, level=level, scope="segment")["weight"],
        "time": time_label,
        "propagation": propagation,
        "conflict": conflict,
        "text": text,
        "anchor_edges": anchor_edges,
        "active": active,
        "propagation_strength": propagation_strength,
    }


def generate_handcrafted_candidates(builder: SparseDatasetBuilder, *, seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)

    fixed_cases = [
        (
            "handcrafted_clear_short",
            "simple_local",
            "short",
            [
                make_explicit_record(
                    builder,
                    road="黄河路",
                    dir_label="向东",
                    range_label="经一路至经三路",
                    level="严重拥堵",
                    propagation_strength=0.0,
                    text="黄河路经一路至经三路段向东因追尾事故严重拥堵",
                ),
                make_explicit_record(
                    builder,
                    road="花园路",
                    dir_label="向北",
                    range_label="局部",
                    level="畅通",
                    propagation_strength=0.0,
                    text="花园路向北畅通无阻",
                ),
            ],
            "黄河路经一路至经三路段向东因追尾事故严重拥堵；花园路向北畅通无阻",
        ),
        (
            "handcrafted_clear_short",
            "simple_local",
            "short",
            [
                make_explicit_record(
                    builder,
                    road="经六路",
                    dir_label="向北",
                    range_label="农业路至红专路",
                    level="封闭",
                    propagation_strength=0.0,
                    text="经六路农业路至红专路段向北施工封闭",
                ),
                make_explicit_record(
                    builder,
                    road="经八路",
                    dir_label="向北",
                    range_label="局部",
                    level="正常",
                    propagation_strength=0.0,
                    text="经八路向北正常通行",
                ),
            ],
            "经六路农业路至红专路段向北施工封闭；经八路向北正常通行",
        ),
    ]

    for category, scene_type, length_bucket, records, narrative in fixed_cases:
        categories[category].append(
            make_custom_sample(
                builder,
                split_name="stage4_train",
                scene_type=scene_type,
                length_bucket=length_bucket,
                records=records,
                narrative=narrative,
            )
        )

    while len(categories["handcrafted_single_segment"]) < 90:
        _, meta = builder._pick_edge()
        records = [
            builder._make_event(
                meta,
                level=rng.choice(["中度拥堵", "严重拥堵", "封闭"]),
                scope="segment",
                propagation_strength=0.0,
            )
        ]
        if rng.random() < 0.35:
            _, relief = builder._pick_edge(axis="V" if meta["axis"] == "H" else "H")
            records.append(builder._make_event(relief, level=rng.choice(["正常", "畅通"]), scope="segment", propagation_strength=0.0))
        categories["handcrafted_single_segment"].append(
            make_custom_sample(
                builder,
                split_name="stage4_train",
                scene_type="simple_local",
                length_bucket=rng.choice(["short", "medium"]),
                records=records,
            )
        )

    while len(categories["handcrafted_single_direction"]) < 90:
        axis = rng.choice(["H", "V"])
        dir_a, dir_b = ("E", "W") if axis == "H" else ("N", "S")
        _, meta_a = builder._pick_edge(axis=axis, dir_code=dir_a)
        meta_b = dict(meta_a)
        meta_b["dir_code"] = dir_b
        meta_b["dir"] = {"E": "向东", "W": "向西", "N": "向北", "S": "向南"}[dir_b]
        records = [
            builder._make_event(
                meta_a,
                level=rng.choice(["严重拥堵", "封闭"]),
                scope=rng.choice(["segment", "range"]),
                propagation_strength=0.0,
            ),
            builder._make_event(
                meta_b,
                level=rng.choice(["正常", "畅通"]),
                scope="segment",
                propagation_strength=0.0,
            ),
        ]
        categories["handcrafted_single_direction"].append(
            make_custom_sample(
                builder,
                split_name="stage4_train",
                scene_type="directional_asymmetry",
                length_bucket=rng.choice(["short", "medium"]),
                records=records,
            )
        )

    while len(categories["handcrafted_single_range"]) < 90:
        _, meta = builder._pick_edge()
        records = [
            builder._make_event(
                meta,
                level=rng.choice(["中度拥堵", "严重拥堵", "封闭"]),
                scope="range",
                propagation_strength=0.08,
            )
        ]
        if rng.random() < 0.25:
            _, relief = builder._pick_edge(axis="V" if meta["axis"] == "H" else "H")
            records.append(builder._make_event(relief, level="正常", scope="segment", propagation_strength=0.0))
        categories["handcrafted_single_range"].append(
            make_custom_sample(
                builder,
                split_name="stage4_train",
                scene_type="simple_local",
                length_bucket=rng.choice(["short", "medium"]),
                records=records,
            )
        )

    while len(categories["handcrafted_clear_short"]) < 90:
        use_directional = rng.random() < 0.45
        if use_directional:
            axis = rng.choice(["H", "V"])
            dir_a, dir_b = ("E", "W") if axis == "H" else ("N", "S")
            _, meta_a = builder._pick_edge(axis=axis, dir_code=dir_a)
            meta_b = dict(meta_a)
            meta_b["dir_code"] = dir_b
            meta_b["dir"] = {"E": "向东", "W": "向西", "N": "向北", "S": "向南"}[dir_b]
            records = [
                builder._make_event(meta_a, level=rng.choice(["严重拥堵", "封闭"]), scope="segment", propagation_strength=0.0),
                builder._make_event(meta_b, level=rng.choice(["正常", "畅通"]), scope="segment", propagation_strength=0.0),
            ]
            scene_type = "directional_asymmetry"
        else:
            _, meta = builder._pick_edge()
            records = [builder._make_event(meta, level=rng.choice(["中度拥堵", "严重拥堵", "封闭"]), scope="segment", propagation_strength=0.0)]
            scene_type = "simple_local"
        categories["handcrafted_clear_short"].append(
            make_custom_sample(
                builder,
                split_name="stage4_train",
                scene_type=scene_type,
                length_bucket="short",
                records=records,
            )
        )

    return categories


def collect_existing_category_candidates(dataset_root: str) -> dict[str, list[dict[str, Any]]]:
    candidate_specs = [
        ("stage1_train", f"{dataset_root}/stage1/train"),
        ("stage1_eval", f"{dataset_root}/stage1/eval"),
        ("stage2_train", f"{dataset_root}/stage2/train"),
        ("stage2_eval", f"{dataset_root}/stage2/eval"),
        ("stage3_train", f"{dataset_root}/stage3/train"),
        ("stage3_eval", f"{dataset_root}/stage3/eval"),
        ("val_normal", f"{dataset_root}/val_normal"),
        ("val_special", f"{dataset_root}/val_special"),
        ("val_anti_truncation", f"{dataset_root}/val_anti_truncation"),
    ]

    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_name, source_dir in candidate_specs:
        rows = load_rows(source_dir)
        for row in rows:
            sample = row_to_sample(row, split_name="stage4_train")
            item = {
                "sample": sample,
                "category": "uncategorized",
                "source_split": source_name,
                "reason": "existing_data",
            }
            if sample["scene_type"] == "simple_local" and has_changed_edges(sample) and has_explicit_anchor_label(sample):
                out["existing_simple_local"].append(dict(item, category="existing_simple_local"))
            if sample["scene_type"] == "directional_asymmetry" and has_changed_edges(sample) and has_explicit_anchor_label(sample):
                out["existing_directional_asymmetry"].append(dict(item, category="existing_directional_asymmetry"))
            if is_focus_scene(sample) and not has_changed_edges(sample):
                out["balance_normal"].append(dict(item, category="balance_normal"))
    return out


def collect_hard_negative_candidates(
    *,
    model_path: str,
    dataset_root: str,
) -> list[dict[str, Any]]:
    source_specs = [
        ("stage1_eval", f"{dataset_root}/stage1/eval"),
        ("stage2_eval", f"{dataset_root}/stage2/eval"),
        ("stage3_eval", f"{dataset_root}/stage3/eval"),
        ("val_normal", f"{dataset_root}/val_normal"),
        ("val_special", f"{dataset_root}/val_special"),
        ("val_anti_truncation", f"{dataset_root}/val_anti_truncation"),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path, device)
    edge_ids = live_edge_ids()
    candidates: list[dict[str, Any]] = []
    try:
        for split_name, source_dir in source_specs:
            rows = load_rows(source_dir)
            details = evaluate_rows(
                model=model,
                tokenizer=tokenizer,
                device=device,
                rows=rows,
                split=split_name,
                edge_ids=edge_ids,
            )
            for row, detail in zip(rows, details):
                sample = row_to_sample(row, split_name="stage4_train")
                if not is_focus_scene(sample):
                    continue
                if not detail["no_anchor"] or detail["gt_changed_edge_count"] <= 0:
                    continue
                if not has_explicit_anchor_label(sample):
                    continue
                candidates.append(
                    {
                        "sample": sample,
                        "category": "hard_negative_no_anchor",
                        "source_split": split_name,
                        "reason": detail["strict_failure_reason"],
                        "detail": {
                            "target_recall": detail["target_recall"],
                            "gt_changed_edge_recall": detail["gt_changed_edge_recall"],
                            "MAE_on_changed_edges": detail["mae_on_changed_edges"],
                            "prediction_raw_preview": detail["prediction_raw_preview"],
                        },
                    }
                )
    finally:
        unload_model(model)
    return candidates


def clone_sample(sample: dict[str, Any], *, split_name: str) -> dict[str, Any]:
    cloned = dict(sample)
    cloned["messages"] = list(sample["messages"])
    cloned["split"] = split_name
    return cloned


def take_candidates(
    *,
    candidates: list[dict[str, Any]],
    train_cap: int,
    eval_cap: int,
    rng: random.Random,
    train_items: list[dict[str, Any]],
    eval_items: list[dict[str, Any]],
    seen_keys: set[str],
) -> None:
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    train_count = 0
    eval_count = 0
    for item in shuffled:
        key = sample_key(item["sample"])
        if key in seen_keys:
            continue
        if eval_count < eval_cap:
            eval_items.append(dict(item, sample=clone_sample(item["sample"], split_name="stage4_eval")))
            eval_count += 1
            seen_keys.add(key)
            continue
        if train_count < train_cap:
            train_items.append(dict(item, sample=clone_sample(item["sample"], split_name="stage4_train")))
            train_count += 1
            seen_keys.add(key)
        if train_count >= train_cap and eval_count >= eval_cap:
            break


def build_summary(
    *,
    train_items: list[dict[str, Any]],
    eval_items: list[dict[str, Any]],
    train_samples: list[dict[str, Any]],
    eval_samples: list[dict[str, Any]],
    output_root: str,
) -> dict[str, Any]:
    category_stats = {
        "train": dict(Counter(item["category"] for item in train_items)),
        "eval": dict(Counter(item["category"] for item in eval_items)),
    }
    source_stats = {
        "train": dict(Counter(item["source_split"] for item in train_items)),
        "eval": dict(Counter(item["source_split"] for item in eval_items)),
    }
    response_stats = {
        "train": {
            "no_anchor_count": sum(1 for sample in train_samples if "NO_ANCHOR" in str(sample.get("response_text", ""))),
            "anchor_labeled_count": sum(1 for sample in train_samples if has_explicit_anchor_label(sample)),
        },
        "eval": {
            "no_anchor_count": sum(1 for sample in eval_samples if "NO_ANCHOR" in str(sample.get("response_text", ""))),
            "anchor_labeled_count": sum(1 for sample in eval_samples if has_explicit_anchor_label(sample)),
        },
    }
    preview = []
    for item in (train_items[:6] + eval_items[:6]):
        preview.append(
            {
                "category": item["category"],
                "source_split": item["source_split"],
                "scene_type": item["sample"]["scene_type"],
                "length_bucket": item["sample"]["length_bucket"],
                "constraint_text": item["sample"]["constraint_text"],
                "response_text": item["sample"]["response_text"],
                "reason": item.get("reason", ""),
            }
        )
    return {
        "output_root": output_root,
        "train_path": f"{output_root}/train",
        "eval_path": f"{output_root}/eval",
        "train_summary": summarize_samples(train_samples),
        "eval_summary": summarize_samples(eval_samples),
        "category_stats": category_stats,
        "source_stats": source_stats,
        "response_stats": response_stats,
        "preview": preview,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=f"{ROOT_DIR}/dataset_sparse_v2")
    parser.add_argument("--model-path", default=f"{ROOT_DIR}/model_merged_sparse_v2")
    parser.add_argument("--out-dir", default=f"{ROOT_DIR}/dataset_sparse_v2_stage4")
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    builder = SparseDatasetBuilder(seed=args.seed)

    handcrafted = generate_handcrafted_candidates(builder, seed=args.seed)
    existing = collect_existing_category_candidates(args.dataset_root)
    hard_negatives = collect_hard_negative_candidates(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
    )

    category_pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    category_pools["hard_negative_no_anchor"].extend(hard_negatives)
    for category, items in existing.items():
        category_pools[category].extend(items)
    for category, samples in handcrafted.items():
        category_pools[category].extend(
            {
                "sample": sample,
                "category": category,
                "source_split": "handcrafted",
                "reason": "manual_enhancement",
            }
            for sample in samples
        )

    train_items: list[dict[str, Any]] = []
    eval_items: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    take_candidates(
        candidates=category_pools["hard_negative_no_anchor"],
        train_cap=120,
        eval_cap=30,
        rng=rng,
        train_items=train_items,
        eval_items=eval_items,
        seen_keys=seen_keys,
    )
    take_candidates(
        candidates=category_pools["existing_simple_local"],
        train_cap=180,
        eval_cap=40,
        rng=rng,
        train_items=train_items,
        eval_items=eval_items,
        seen_keys=seen_keys,
    )
    take_candidates(
        candidates=category_pools["existing_directional_asymmetry"],
        train_cap=180,
        eval_cap=40,
        rng=rng,
        train_items=train_items,
        eval_items=eval_items,
        seen_keys=seen_keys,
    )
    take_candidates(
        candidates=category_pools["handcrafted_single_segment"],
        train_cap=60,
        eval_cap=12,
        rng=rng,
        train_items=train_items,
        eval_items=eval_items,
        seen_keys=seen_keys,
    )
    take_candidates(
        candidates=category_pools["handcrafted_single_direction"],
        train_cap=60,
        eval_cap=12,
        rng=rng,
        train_items=train_items,
        eval_items=eval_items,
        seen_keys=seen_keys,
    )
    take_candidates(
        candidates=category_pools["handcrafted_single_range"],
        train_cap=60,
        eval_cap=12,
        rng=rng,
        train_items=train_items,
        eval_items=eval_items,
        seen_keys=seen_keys,
    )
    take_candidates(
        candidates=category_pools["handcrafted_clear_short"],
        train_cap=60,
        eval_cap=12,
        rng=rng,
        train_items=train_items,
        eval_items=eval_items,
        seen_keys=seen_keys,
    )
    take_candidates(
        candidates=category_pools["balance_normal"],
        train_cap=60,
        eval_cap=20,
        rng=rng,
        train_items=train_items,
        eval_items=eval_items,
        seen_keys=seen_keys,
    )

    train_samples = [item["sample"] for item in train_items]
    eval_samples = [item["sample"] for item in eval_items]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    save_parquet(train_samples, str(out_root / "train"))
    save_parquet(eval_samples, str(out_root / "eval"))

    summary = build_summary(
        train_items=train_items,
        eval_items=eval_items,
        train_samples=train_samples,
        eval_samples=eval_samples,
        output_root=str(out_root),
    )
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
