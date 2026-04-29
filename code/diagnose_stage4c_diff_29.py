#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import torch

from roadnet_meta import live_edge_ids
from sparse_eval_strict import (
    dense_prediction,
    infer_sparse_row,
    load_model,
    load_tokenizer,
    score_prediction,
    unload_model,
)


DEFAULT_PRE_MODEL = "/root/autodl-tmp/model_merged_sparse_v2_stage4_fix"
DEFAULT_POST_MODEL = "/root/autodl-tmp/model_merged_sparse_v2_stage4c_fix"
DEFAULT_DATASET_ROOT = "/root/autodl-tmp/dataset_sparse_v2"
DEFAULT_SAMPLE_SOURCE = "/root/autodl-tmp/results/simple_local_semantic_gap_audit.csv"
DEFAULT_OUTPUT_JSON = "/root/autodl-tmp/results/stage4c_diff_29.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff diagnose stage4_fix vs stage4c on the 29 held-out main failures.")
    parser.add_argument("--pre-model-path", default=DEFAULT_PRE_MODEL)
    parser.add_argument("--post-model-path", default=DEFAULT_POST_MODEL)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sample-source-csv", default=DEFAULT_SAMPLE_SOURCE)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def normalize_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def load_target_samples(sample_source_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(sample_source_csv)
    df["no_anchor_when_gt_changed"] = df["no_anchor_when_gt_changed"].map(normalize_bool)
    target = df[df["no_anchor_when_gt_changed"]].copy()
    target["split_index"] = target["sample_id"].astype(str).str.split(":").str[-1].astype(int)
    target["split"] = target["sample_id"].astype(str).str.split(":").str[0]
    target = target.sort_values(["split", "split_index"]).reset_index(drop=True)
    return target


def load_dataset_rows(dataset_root: Path, target_samples: pd.DataFrame) -> pd.DataFrame:
    frames = []
    columns = [
        "split",
        "scene_type",
        "constraint_text",
        "messages",
        "response_text",
        "ground_truth_json",
        "event_records_json",
    ]
    for split in sorted(target_samples["split"].unique()):
        table = pq.read_table(dataset_root / split / "data.parquet", columns=columns)
        frame = pd.DataFrame(table.to_pylist())
        frame["split_index"] = range(len(frame))
        frame["sample_id"] = frame["split"].astype(str) + ":" + frame["split_index"].astype(str)
        frames.append(frame)
    dataset_df = pd.concat(frames, ignore_index=True)
    keep_ids = set(target_samples["sample_id"].astype(str))
    return dataset_df[dataset_df["sample_id"].astype(str).isin(keep_ids)].copy()


def classify_output(bundle: dict[str, Any]) -> str:
    return "ANCHOR" if int(bundle.get("anchor_count", 0)) > 0 else "NO_ANCHOR"


def canonical_anchor_signature(bundle: dict[str, Any]) -> list[str]:
    anchors = bundle.get("anchors", []) or []
    signatures = []
    for anchor in anchors:
        signatures.append(
            "ROAD={road}|DIR={direction}|RANGE={range_}|LEVEL={level}".format(
                road=str(anchor.get("road", "")).strip(),
                direction=str(anchor.get("dir", "")).strip(),
                range_=str(anchor.get("range", "")).strip(),
                level=str(anchor.get("level", "")).strip(),
            )
        )
    return sorted(signatures)


def post_is_fully_correct(scored: dict[str, Any]) -> bool:
    if bool(scored.get("parse_fail", False)):
        return False
    if int(scored.get("changed_edge_hits", 0)) != int(scored.get("gt_changed_edge_count", 0)):
        return False
    if int(scored.get("target_edge_hits", 0)) != int(scored.get("gt_target_edge_count", 0)):
        return False
    return True


def infer_for_model(
    *,
    model_path: str,
    dataset_rows: pd.DataFrame,
    device: str,
    edge_ids: list[str],
) -> dict[str, dict[str, Any]]:
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path, device)
    results: dict[str, dict[str, Any]] = {}
    try:
        for row in dataset_rows.itertuples(index=False):
            prompt_text = json.loads(row.messages)[0]["content"]
            run = infer_sparse_row(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                scene_type=row.scene_type,
                device=device,
            )
            pred_weights = dense_prediction(run["bundle"], edge_ids)
            scored = score_prediction(
                row={
                    "scene_type": row.scene_type,
                    "length_bucket": "",
                    "constraint_text": row.constraint_text,
                    "messages": row.messages,
                    "ground_truth_json": row.ground_truth_json,
                },
                bundle=run["bundle"],
                pred_weights=pred_weights,
                edge_ids=edge_ids,
                sample_id=row.sample_id,
                split=row.split,
                split_index=int(row.split_index),
                raw_generation=run["raw_generation"],
                infer_time_s=run["infer_time_s"],
            )
            results[str(row.sample_id)] = {
                "raw_output": run["raw_generation"],
                "output_label": classify_output(run["bundle"]),
                "anchor_signature": canonical_anchor_signature(run["bundle"]),
                "bundle": run["bundle"],
                "score": scored,
            }
    finally:
        unload_model(model)
    return results


def derive_change_type(pre_item: dict[str, Any], post_item: dict[str, Any]) -> tuple[bool, str]:
    pre_label = pre_item["output_label"]
    post_label = post_item["output_label"]
    if pre_label == "NO_ANCHOR" and post_label == "NO_ANCHOR":
        return False, "no_change"
    if pre_label == "NO_ANCHOR" and post_label == "ANCHOR":
        return True, "no_anchor_to_anchor"
    if pre_label == "ANCHOR" and post_label == "NO_ANCHOR":
        return True, "anchor_to_no_anchor"
    if pre_item["anchor_signature"] == post_item["anchor_signature"]:
        return False, "no_change"
    return True, "anchor_changed"


def build_report(
    *,
    target_samples: pd.DataFrame,
    dataset_rows: pd.DataFrame,
    pre_results: dict[str, dict[str, Any]],
    post_results: dict[str, dict[str, Any]],
    pre_model_path: str,
    post_model_path: str,
    sample_source_csv: str,
) -> dict[str, Any]:
    merged = target_samples.merge(
        dataset_rows[
            [
                "sample_id",
                "constraint_text",
                "response_text",
                "ground_truth_json",
                "event_records_json",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    samples = []
    for row in merged.itertuples(index=False):
        sample_id = str(row.sample_id)
        pre_item = pre_results[sample_id]
        post_item = post_results[sample_id]
        changed, change_type = derive_change_type(pre_item, post_item)
        post_anchor_still_wrong = (
            post_item["output_label"] == "ANCHOR" and not post_is_fully_correct(post_item["score"])
        )
        samples.append(
            {
                "sample_id": sample_id,
                "split": row.split,
                "split_index": int(row.split_index),
                "scene_type": row.scene_type,
                "semantic_gap_subtype": row.semantic_gap_subtype,
                "input_text": row.input_text,
                "gt_label": row.gt_label,
                "constraint_text": row.constraint_text,
                "event_records_json": row.event_records_json,
                "ground_truth_json": row.ground_truth_json,
                "pre_raw_output": pre_item["raw_output"],
                "post_raw_output": post_item["raw_output"],
                "pre_output": pre_item["output_label"],
                "post_output": post_item["output_label"],
                "changed": changed,
                "change_type": change_type,
                "post_anchor_still_wrong": post_anchor_still_wrong,
                "pre_anchor_signature": pre_item["anchor_signature"],
                "post_anchor_signature": post_item["anchor_signature"],
                "pre_parse_fail": bool(pre_item["score"]["parse_fail"]),
                "post_parse_fail": bool(post_item["score"]["parse_fail"]),
                "pre_changed_recall": float(pre_item["score"]["gt_changed_edge_recall"]),
                "post_changed_recall": float(post_item["score"]["gt_changed_edge_recall"]),
                "pre_target_recall": float(pre_item["score"]["target_recall"]),
                "post_target_recall": float(post_item["score"]["target_recall"]),
            }
        )

    subtype_breakdown = (
        pd.Series([item["semantic_gap_subtype"] for item in samples]).value_counts().to_dict()
        if samples
        else {}
    )
    changed_samples = [item for item in samples if item["changed"]]
    changed_subtype_breakdown = (
        pd.Series([item["semantic_gap_subtype"] for item in changed_samples]).value_counts().to_dict()
        if changed_samples
        else {}
    )

    stats = {
        "sample_count": len(samples),
        "completely_unchanged_count": sum(1 for item in samples if item["change_type"] == "no_change"),
        "no_anchor_to_anchor_count": sum(1 for item in samples if item["change_type"] == "no_anchor_to_anchor"),
        "anchor_to_no_anchor_count": sum(1 for item in samples if item["change_type"] == "anchor_to_no_anchor"),
        "anchor_changed_count": sum(1 for item in samples if item["change_type"] == "anchor_changed"),
        "anchor_but_still_wrong_count": sum(1 for item in samples if item["post_anchor_still_wrong"]),
        "post_anchor_and_fully_correct_count": sum(
            1 for item in samples if item["post_output"] == "ANCHOR" and not item["post_anchor_still_wrong"]
        ),
        "subtype_breakdown": subtype_breakdown,
        "changed_subtype_breakdown": changed_subtype_breakdown,
        "majority_answer": (
            "completely_unchanged_is_majority"
            if sum(1 for item in samples if item["change_type"] == "no_change") > 15
            else "completely_unchanged_is_minority"
        ),
    }

    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_source_csv": sample_source_csv,
        "pre_model_path": pre_model_path,
        "post_model_path": post_model_path,
        "sample_count": stats["sample_count"],
        "completely_unchanged_count": stats["completely_unchanged_count"],
        "no_anchor_to_anchor_count": stats["no_anchor_to_anchor_count"],
        "anchor_but_still_wrong_count": stats["anchor_but_still_wrong_count"],
        "anchor_to_no_anchor_count": stats["anchor_to_no_anchor_count"],
        "anchor_changed_count": stats["anchor_changed_count"],
        "stats": stats,
        "samples": samples,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("STAGE4C_DIFF_29_SUMMARY")
    print(f"sample_count={report['sample_count']}")
    print(f"completely_unchanged_count={report['completely_unchanged_count']}")
    print(f"no_anchor_to_anchor_count={report['no_anchor_to_anchor_count']}")
    print(f"anchor_but_still_wrong_count={report['anchor_but_still_wrong_count']}")
    print(f"anchor_to_no_anchor_count={report['anchor_to_no_anchor_count']}")
    print(f"anchor_changed_count={report['anchor_changed_count']}")
    print(f"majority_answer={report['stats']['majority_answer']}")


def main() -> None:
    args = parse_args()
    sample_source_csv = Path(args.sample_source_csv)
    dataset_root = Path(args.dataset_root)
    output_json = Path(args.output_json)

    target_samples = load_target_samples(sample_source_csv)
    if len(target_samples) != 29:
        raise ValueError(f"Expected 29 target samples, found {len(target_samples)} from {sample_source_csv}")

    dataset_rows = load_dataset_rows(dataset_root, target_samples)
    if len(dataset_rows) != 29:
        raise ValueError(f"Expected 29 dataset rows, found {len(dataset_rows)} from {dataset_root}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    edge_ids = live_edge_ids()

    pre_results = infer_for_model(
        model_path=args.pre_model_path,
        dataset_rows=dataset_rows,
        device=device,
        edge_ids=edge_ids,
    )
    post_results = infer_for_model(
        model_path=args.post_model_path,
        dataset_rows=dataset_rows,
        device=device,
        edge_ids=edge_ids,
    )

    report = build_report(
        target_samples=target_samples,
        dataset_rows=dataset_rows,
        pre_results=pre_results,
        post_results=post_results,
        pre_model_path=args.pre_model_path,
        post_model_path=args.post_model_path,
        sample_source_csv=str(sample_source_csv),
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)


if __name__ == "__main__":
    main()
