#!/usr/bin/env python3
"""Validate Sparse-LoRA-v2 + GAT-v2 and export paper-facing reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

CODE_DIR = "/root/autodl-tmp/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from gat_smoother import (  # noqa: E402
    DEFAULT_ALWAYS_GAT_PATH,
    DEFAULT_GATED_GAT_PATH,
    describe_gat_checkpoint,
    smooth_weights,
)
from gat_v2_defaults import DEFAULT_SPARSE_UPSTREAM  # noqa: E402
from roadnet_meta import live_edge_ids  # noqa: E402
from sparse_eval_strict import (  # noqa: E402
    build_report,
    dense_prediction,
    infer_sparse_row,
    load_model,
    load_rows,
    load_tokenizer,
    score_prediction,
    unload_model,
)

METHOD_LABELS = {
    "sparse": "Sparse-LoRA-v2",
    "always_gat_v2": "Sparse-LoRA-v2 + Always-GAT-v2",
    "gated_gat_v2": "Sparse-LoRA-v2 + Gated-GAT-v2",
}
PRIMARY_SCENE_TYPES = [
    "simple_local",
    "directional_asymmetry",
    "propagation_range",
    "compound_disaster",
]
SPECIAL_SCENE_TYPES = [
    "core_blockage",
    "directional_asymmetry",
    "propagation_range",
    "compound_disaster",
    "temporal_switch",
]
MAIN_SCOPE_SPECS = [
    ("overall", "overall"),
    ("overall:scene_type", "simple_local"),
    ("overall:scene_type", "directional_asymmetry"),
    ("overall:scene_type", "propagation_range"),
    ("overall:scene_type", "compound_disaster"),
    ("split", "val_anti_truncation"),
]
SPECIAL_SCOPE_SPECS = [
    ("overall:scene_type", "core_blockage"),
    ("overall:scene_type", "directional_asymmetry"),
    ("overall:scene_type", "propagation_range"),
    ("overall:scene_type", "compound_disaster"),
    ("overall:scene_type", "temporal_switch"),
    ("split", "val_anti_truncation"),
]
SUMMARY_METRICS = [
    "dense_weight_mae",
    "MAE_on_changed_edges",
    "MAE_on_target_edges",
    "gt_changed_edge_recall",
    "target_recall",
    "parse_fail_rate_strict",
]


def evaluate_rows(
    *,
    model,
    tokenizer,
    device: str,
    rows: list[dict[str, Any]],
    split: str,
    edge_ids: list[str],
    always_path: str,
    gated_path: str,
) -> dict[str, list[dict[str, Any]]]:
    details_by_method = {
        "sparse": [],
        "always_gat_v2": [],
        "gated_gat_v2": [],
    }

    for split_index, row in enumerate(rows):
        run = infer_sparse_row(
            model=model,
            tokenizer=tokenizer,
            prompt_text=json.loads(row["messages"])[0]["content"],
            scene_type=row.get("scene_type"),
            device=device,
        )
        bundle = run["bundle"]
        sparse_pred = dense_prediction(bundle, edge_ids)
        always_pred, always_alpha = smooth_weights(
            bundle.get("mapped_weights", {}),
            model_path=always_path,
            edge_confidence=bundle.get("edge_confidence", {}),
            protected_edges=bundle.get("protected_edges", []),
            parse_confidence=bundle.get("parse_confidence", 0.0),
            scene_type=row.get("scene_type"),
            mode="always",
            device=device,
        )
        gated_pred, gated_alpha = smooth_weights(
            bundle.get("mapped_weights", {}),
            model_path=gated_path,
            edge_confidence=bundle.get("edge_confidence", {}),
            protected_edges=bundle.get("protected_edges", []),
            parse_confidence=bundle.get("parse_confidence", 0.0),
            scene_type=row.get("scene_type"),
            mode="gated",
            device=device,
        )

        method_preds = {
            "sparse": (sparse_pred, None),
            "always_gat_v2": (always_pred, always_alpha),
            "gated_gat_v2": (gated_pred, gated_alpha),
        }
        for method_name, (pred_weights, alpha_used) in method_preds.items():
            detail = score_prediction(
                row=row,
                bundle=bundle,
                pred_weights=pred_weights,
                edge_ids=edge_ids,
                sample_id=f"{split}:{split_index}",
                split=split,
                split_index=split_index,
                raw_generation=run["raw_generation"],
                infer_time_s=run["infer_time_s"],
            )
            detail["method"] = method_name
            detail["method_label"] = METHOD_LABELS[method_name]
            detail["gat_alpha_used"] = "" if alpha_used is None else round(float(alpha_used), 4)
            details_by_method[method_name].append(detail)

    return details_by_method


def get_scope_summary(report: dict[str, Any], scope: str, scope_name: str) -> dict[str, Any]:
    if scope == "overall":
        return report["overall"]
    if scope == "overall:scene_type":
        return report["by_scene_type_overall"].get(scope_name, {})
    if scope == "split":
        return report["splits"].get(scope_name, {}).get("summary", {})
    raise ValueError(f"unknown scope: {scope}")


def stability_tuple(report: dict[str, Any]) -> tuple[float, ...]:
    overall = report["overall"]
    dense_values = []
    changed_values = []
    for scene_type in PRIMARY_SCENE_TYPES:
        summary = report["by_scene_type_overall"].get(scene_type, {})
        dense_values.append(float(summary.get("dense_weight_mae", 1e9)))
        changed_values.append(float(summary.get("MAE_on_changed_edges", 1e9)))

    anti_truncation = report["splits"].get("val_anti_truncation", {}).get("summary", {})
    dense_values.append(float(anti_truncation.get("dense_weight_mae", 1e9)))
    changed_values.append(float(anti_truncation.get("MAE_on_changed_edges", 1e9)))

    return (
        float(overall.get("dense_weight_mae", 1e9)),
        float(overall.get("MAE_on_changed_edges", 1e9)),
        max(dense_values) if dense_values else 1e9,
        max(changed_values) if changed_values else 1e9,
        -float(overall.get("target_recall", 0.0)),
        -float(overall.get("gt_changed_edge_recall", 0.0)),
    )


def pick_mainline(method_reports: dict[str, dict[str, Any]]) -> tuple[str, str]:
    candidates = ["always_gat_v2", "gated_gat_v2"]
    ordered = sorted(candidates, key=lambda name: stability_tuple(method_reports[name]))
    return ordered[0], ordered[1]


def summary_row(
    *,
    scope: str,
    scope_name: str,
    sparse_summary: dict[str, Any],
    main_method: str,
    main_summary: dict[str, Any],
    appendix_method: str,
    appendix_summary: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "scope": scope,
        "scope_name": scope_name,
        "count": sparse_summary.get("count", 0),
        "baseline_method": METHOD_LABELS["sparse"],
        "main_method": METHOD_LABELS[main_method],
        "appendix_method": METHOD_LABELS[appendix_method],
    }
    for prefix, summary in (
        ("baseline", sparse_summary),
        ("main", main_summary),
        ("appendix", appendix_summary),
    ):
        for metric in SUMMARY_METRICS:
            row[f"{prefix}_{metric}"] = summary.get(metric, "")

    for prefix, summary in (("main", main_summary), ("appendix", appendix_summary)):
        row[f"{prefix}_dense_mae_delta_vs_sparse"] = round(
            float(summary.get("dense_weight_mae", 0.0)) - float(sparse_summary.get("dense_weight_mae", 0.0)),
            4,
        )
        row[f"{prefix}_changed_mae_delta_vs_sparse"] = round(
            float(summary.get("MAE_on_changed_edges", 0.0)) - float(sparse_summary.get("MAE_on_changed_edges", 0.0)),
            4,
        )
        row[f"{prefix}_target_mae_delta_vs_sparse"] = round(
            float(summary.get("MAE_on_target_edges", 0.0)) - float(sparse_summary.get("MAE_on_target_edges", 0.0)),
            4,
        )
        row[f"{prefix}_gt_changed_recall_delta_vs_sparse"] = round(
            float(summary.get("gt_changed_edge_recall", 0.0)) - float(sparse_summary.get("gt_changed_edge_recall", 0.0)),
            1,
        )
        row[f"{prefix}_target_recall_delta_vs_sparse"] = round(
            float(summary.get("target_recall", 0.0)) - float(sparse_summary.get("target_recall", 0.0)),
            1,
        )
    return row


def write_csv_rows(output_path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_main_summary_csv(
    *,
    reports: dict[str, dict[str, Any]],
    main_method: str,
    appendix_method: str,
    output_path: str,
) -> list[dict[str, Any]]:
    rows = []
    for scope, scope_name in MAIN_SCOPE_SPECS:
        sparse_summary = get_scope_summary(reports["sparse"], scope, scope_name)
        if not sparse_summary:
            continue
        main_summary = get_scope_summary(reports[main_method], scope, scope_name)
        appendix_summary = get_scope_summary(reports[appendix_method], scope, scope_name)
        rows.append(
            summary_row(
                scope=scope,
                scope_name=scope_name,
                sparse_summary=sparse_summary,
                main_method=main_method,
                main_summary=main_summary,
                appendix_method=appendix_method,
                appendix_summary=appendix_summary,
            )
        )

    fieldnames = [
        "scope",
        "scope_name",
        "count",
        "baseline_method",
        "baseline_dense_weight_mae",
        "baseline_MAE_on_changed_edges",
        "baseline_MAE_on_target_edges",
        "baseline_gt_changed_edge_recall",
        "baseline_target_recall",
        "baseline_parse_fail_rate_strict",
        "main_method",
        "main_dense_weight_mae",
        "main_MAE_on_changed_edges",
        "main_MAE_on_target_edges",
        "main_gt_changed_edge_recall",
        "main_target_recall",
        "main_parse_fail_rate_strict",
        "main_dense_mae_delta_vs_sparse",
        "main_changed_mae_delta_vs_sparse",
        "main_target_mae_delta_vs_sparse",
        "main_gt_changed_recall_delta_vs_sparse",
        "main_target_recall_delta_vs_sparse",
        "appendix_method",
        "appendix_dense_weight_mae",
        "appendix_MAE_on_changed_edges",
        "appendix_MAE_on_target_edges",
        "appendix_gt_changed_edge_recall",
        "appendix_target_recall",
        "appendix_parse_fail_rate_strict",
        "appendix_dense_mae_delta_vs_sparse",
        "appendix_changed_mae_delta_vs_sparse",
        "appendix_target_mae_delta_vs_sparse",
        "appendix_gt_changed_recall_delta_vs_sparse",
        "appendix_target_recall_delta_vs_sparse",
    ]
    write_csv_rows(output_path, fieldnames, rows)
    return rows


def write_main_detail_csv(
    *,
    split_details_by_method: dict[str, dict[str, list[dict[str, Any]]]],
    main_method: str,
    appendix_method: str,
    output_path: str,
) -> None:
    sparse_lookup = {
        item["sample_id"]: item
        for split_items in split_details_by_method["sparse"].values()
        for item in split_items
    }
    main_lookup = {
        item["sample_id"]: item
        for split_items in split_details_by_method[main_method].values()
        for item in split_items
    }
    appendix_lookup = {
        item["sample_id"]: item
        for split_items in split_details_by_method[appendix_method].values()
        for item in split_items
    }

    rows = []
    for sample_id in sorted(sparse_lookup.keys()):
        sparse_item = sparse_lookup[sample_id]
        main_item = main_lookup[sample_id]
        appendix_item = appendix_lookup[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "split": sparse_item["split"],
                "split_index": sparse_item["split_index"],
                "scene_type": sparse_item["scene_type"],
                "length_bucket": sparse_item["length_bucket"],
                "gt_changed_edge_count": sparse_item["gt_changed_edge_count"],
                "gt_target_edge_count": sparse_item["gt_target_edge_count"],
                "no_anchor": sparse_item["no_anchor"],
                "parse_fail": sparse_item["parse_fail"],
                "strict_parse_fail": sparse_item["strict_parse_fail"],
                "parse_confidence": sparse_item["parse_confidence"],
                "anchor_count": sparse_item["anchor_count"],
                "mapped_edge_count": sparse_item["mapped_edge_count"],
                "infer_time_s": sparse_item["infer_time_s"],
                "sparse_dense_weight_mae": sparse_item["dense_weight_mae"],
                "sparse_MAE_on_changed_edges": sparse_item["mae_on_changed_edges"],
                "sparse_MAE_on_target_edges": sparse_item["mae_on_target_edges"],
                "sparse_gt_changed_edge_recall": sparse_item["gt_changed_edge_recall"],
                "sparse_target_recall": sparse_item["target_recall"],
                "main_method": METHOD_LABELS[main_method],
                "main_dense_weight_mae": main_item["dense_weight_mae"],
                "main_MAE_on_changed_edges": main_item["mae_on_changed_edges"],
                "main_MAE_on_target_edges": main_item["mae_on_target_edges"],
                "main_gt_changed_edge_recall": main_item["gt_changed_edge_recall"],
                "main_target_recall": main_item["target_recall"],
                "main_dense_mae_delta_vs_sparse": round(
                    float(main_item["dense_weight_mae"]) - float(sparse_item["dense_weight_mae"]),
                    4,
                ),
                "main_changed_mae_delta_vs_sparse": round(
                    float(main_item["mae_on_changed_edges"]) - float(sparse_item["mae_on_changed_edges"]),
                    4,
                ),
                "main_target_mae_delta_vs_sparse": round(
                    float(main_item["mae_on_target_edges"]) - float(sparse_item["mae_on_target_edges"]),
                    4,
                ),
                "main_gt_changed_recall_delta_vs_sparse": round(
                    float(main_item["gt_changed_edge_recall"]) - float(sparse_item["gt_changed_edge_recall"]),
                    1,
                ),
                "main_target_recall_delta_vs_sparse": round(
                    float(main_item["target_recall"]) - float(sparse_item["target_recall"]),
                    1,
                ),
                "appendix_method": METHOD_LABELS[appendix_method],
                "appendix_dense_weight_mae": appendix_item["dense_weight_mae"],
                "appendix_MAE_on_changed_edges": appendix_item["mae_on_changed_edges"],
                "appendix_MAE_on_target_edges": appendix_item["mae_on_target_edges"],
                "appendix_gt_changed_edge_recall": appendix_item["gt_changed_edge_recall"],
                "appendix_target_recall": appendix_item["target_recall"],
                "appendix_dense_mae_delta_vs_sparse": round(
                    float(appendix_item["dense_weight_mae"]) - float(sparse_item["dense_weight_mae"]),
                    4,
                ),
                "appendix_changed_mae_delta_vs_sparse": round(
                    float(appendix_item["mae_on_changed_edges"]) - float(sparse_item["mae_on_changed_edges"]),
                    4,
                ),
                "appendix_target_mae_delta_vs_sparse": round(
                    float(appendix_item["mae_on_target_edges"]) - float(sparse_item["mae_on_target_edges"]),
                    4,
                ),
                "appendix_gt_changed_recall_delta_vs_sparse": round(
                    float(appendix_item["gt_changed_edge_recall"]) - float(sparse_item["gt_changed_edge_recall"]),
                    1,
                ),
                "appendix_target_recall_delta_vs_sparse": round(
                    float(appendix_item["target_recall"]) - float(sparse_item["target_recall"]),
                    1,
                ),
                "prediction_raw_preview": sparse_item["prediction_raw_preview"],
                "constraint_text": sparse_item["constraint_text"],
            }
        )

    fieldnames = [
        "sample_id",
        "split",
        "split_index",
        "scene_type",
        "length_bucket",
        "gt_changed_edge_count",
        "gt_target_edge_count",
        "no_anchor",
        "parse_fail",
        "strict_parse_fail",
        "parse_confidence",
        "anchor_count",
        "mapped_edge_count",
        "infer_time_s",
        "sparse_dense_weight_mae",
        "sparse_MAE_on_changed_edges",
        "sparse_MAE_on_target_edges",
        "sparse_gt_changed_edge_recall",
        "sparse_target_recall",
        "main_method",
        "main_dense_weight_mae",
        "main_MAE_on_changed_edges",
        "main_MAE_on_target_edges",
        "main_gt_changed_edge_recall",
        "main_target_recall",
        "main_dense_mae_delta_vs_sparse",
        "main_changed_mae_delta_vs_sparse",
        "main_target_mae_delta_vs_sparse",
        "main_gt_changed_recall_delta_vs_sparse",
        "main_target_recall_delta_vs_sparse",
        "appendix_method",
        "appendix_dense_weight_mae",
        "appendix_MAE_on_changed_edges",
        "appendix_MAE_on_target_edges",
        "appendix_gt_changed_edge_recall",
        "appendix_target_recall",
        "appendix_dense_mae_delta_vs_sparse",
        "appendix_changed_mae_delta_vs_sparse",
        "appendix_target_mae_delta_vs_sparse",
        "appendix_gt_changed_recall_delta_vs_sparse",
        "appendix_target_recall_delta_vs_sparse",
        "prediction_raw_preview",
        "constraint_text",
    ]
    write_csv_rows(output_path, fieldnames, rows)


def better_method_name(
    *,
    sparse_summary: dict[str, Any],
    main_method: str,
    main_summary: dict[str, Any],
    appendix_method: str,
    appendix_summary: dict[str, Any],
) -> str:
    scores = {
        "sparse": (
            float(sparse_summary.get("dense_weight_mae", 1e9)),
            float(sparse_summary.get("MAE_on_changed_edges", 1e9)),
            -float(sparse_summary.get("target_recall", 0.0)),
            -float(sparse_summary.get("gt_changed_edge_recall", 0.0)),
        ),
        main_method: (
            float(main_summary.get("dense_weight_mae", 1e9)),
            float(main_summary.get("MAE_on_changed_edges", 1e9)),
            -float(main_summary.get("target_recall", 0.0)),
            -float(main_summary.get("gt_changed_edge_recall", 0.0)),
        ),
        appendix_method: (
            float(appendix_summary.get("dense_weight_mae", 1e9)),
            float(appendix_summary.get("MAE_on_changed_edges", 1e9)),
            -float(appendix_summary.get("target_recall", 0.0)),
            -float(appendix_summary.get("gt_changed_edge_recall", 0.0)),
        ),
    }
    best_key = min(scores.items(), key=lambda pair: pair[1])[0]
    return METHOD_LABELS[best_key]


def write_special_scene_summary_csv(
    *,
    reports: dict[str, dict[str, Any]],
    main_method: str,
    appendix_method: str,
    output_path: str,
) -> list[dict[str, Any]]:
    rows = []
    for scope, scope_name in SPECIAL_SCOPE_SPECS:
        sparse_summary = get_scope_summary(reports["sparse"], scope, scope_name)
        if not sparse_summary:
            continue
        main_summary = get_scope_summary(reports[main_method], scope, scope_name)
        appendix_summary = get_scope_summary(reports[appendix_method], scope, scope_name)
        row = summary_row(
            scope=scope,
            scope_name=scope_name,
            sparse_summary=sparse_summary,
            main_method=main_method,
            main_summary=main_summary,
            appendix_method=appendix_method,
            appendix_summary=appendix_summary,
        )
        row["best_method"] = better_method_name(
            sparse_summary=sparse_summary,
            main_method=main_method,
            main_summary=main_summary,
            appendix_method=appendix_method,
            appendix_summary=appendix_summary,
        )
        rows.append(row)

    fieldnames = [
        "scope",
        "scope_name",
        "count",
        "baseline_method",
        "baseline_dense_weight_mae",
        "baseline_MAE_on_changed_edges",
        "baseline_MAE_on_target_edges",
        "baseline_gt_changed_edge_recall",
        "baseline_target_recall",
        "main_method",
        "main_dense_weight_mae",
        "main_MAE_on_changed_edges",
        "main_MAE_on_target_edges",
        "main_gt_changed_edge_recall",
        "main_target_recall",
        "main_dense_mae_delta_vs_sparse",
        "main_changed_mae_delta_vs_sparse",
        "main_target_mae_delta_vs_sparse",
        "main_gt_changed_recall_delta_vs_sparse",
        "main_target_recall_delta_vs_sparse",
        "appendix_method",
        "appendix_dense_weight_mae",
        "appendix_MAE_on_changed_edges",
        "appendix_MAE_on_target_edges",
        "appendix_gt_changed_edge_recall",
        "appendix_target_recall",
        "appendix_dense_mae_delta_vs_sparse",
        "appendix_changed_mae_delta_vs_sparse",
        "appendix_target_mae_delta_vs_sparse",
        "appendix_gt_changed_recall_delta_vs_sparse",
        "appendix_target_recall_delta_vs_sparse",
        "best_method",
    ]
    write_csv_rows(output_path, fieldnames, rows)
    return rows


def clear_value_judgement(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    dense_gain = float(baseline.get("dense_weight_mae", 0.0)) - float(candidate.get("dense_weight_mae", 0.0))
    changed_gain = float(baseline.get("MAE_on_changed_edges", 0.0)) - float(candidate.get("MAE_on_changed_edges", 0.0))
    target_gain = float(baseline.get("MAE_on_target_edges", 0.0)) - float(candidate.get("MAE_on_target_edges", 0.0))
    changed_recall_delta = float(candidate.get("gt_changed_edge_recall", 0.0)) - float(baseline.get("gt_changed_edge_recall", 0.0))
    target_recall_delta = float(candidate.get("target_recall", 0.0)) - float(baseline.get("target_recall", 0.0))

    gain_threshold_hit = (
        dense_gain >= max(0.002, float(baseline.get("dense_weight_mae", 0.0)) * 0.05)
        or changed_gain >= max(0.01, float(baseline.get("MAE_on_changed_edges", 0.0)) * 0.05)
        or target_gain >= max(0.005, float(baseline.get("MAE_on_target_edges", 0.0)) * 0.05)
    )
    if gain_threshold_hit and changed_recall_delta >= -1.0 and target_recall_delta >= -1.0:
        return "有明确价值"
    if dense_gain > 0.0 or changed_gain > 0.0 or target_gain > 0.0:
        return "有边际价值，但不够稳"
    return "没有明确价值"


def scene_benefit_score(baseline: dict[str, Any], candidate: dict[str, Any]) -> float:
    return (
        (float(baseline.get("dense_weight_mae", 0.0)) - float(candidate.get("dense_weight_mae", 0.0))) * 10.0
        + (float(baseline.get("MAE_on_changed_edges", 0.0)) - float(candidate.get("MAE_on_changed_edges", 0.0))) * 3.0
        + (float(candidate.get("gt_changed_edge_recall", 0.0)) - float(baseline.get("gt_changed_edge_recall", 0.0))) * 0.1
        + (float(candidate.get("target_recall", 0.0)) - float(baseline.get("target_recall", 0.0))) * 0.1
    )


def paper_ready_decision(
    *,
    sparse_report: dict[str, Any],
    main_report: dict[str, Any],
) -> tuple[bool, str]:
    overall_sparse = sparse_report["overall"]
    overall_main = main_report["overall"]
    better_overall = (
        float(overall_main.get("dense_weight_mae", 1e9)) <= float(overall_sparse.get("dense_weight_mae", 1e9))
        and float(overall_main.get("MAE_on_changed_edges", 1e9)) <= float(overall_sparse.get("MAE_on_changed_edges", 1e9))
        and float(overall_main.get("target_recall", 0.0)) >= float(overall_sparse.get("target_recall", 0.0)) - 0.1
        and float(overall_main.get("gt_changed_edge_recall", 0.0)) >= float(overall_sparse.get("gt_changed_edge_recall", 0.0)) - 0.1
    )
    if better_overall:
        return True, "整体口径已不弱于 sparse 基线，可进入论文主表。"
    return False, "整体口径仍弱于 sparse 基线，更适合附录或特殊场景分析。"


def render_checkpoint_meta(path: str) -> list[str]:
    meta = describe_gat_checkpoint(path)
    summary = meta.get("training_summary", {})
    return [
        f"- path: `{path}`",
        f"- exists/aligned: `{meta.get('exists')}` / `{meta.get('aligned')}`",
        f"- mode: `{meta.get('mode')}`",
        f"- training best_epoch/best_eval_mae: `{summary.get('best_epoch', 'n/a')}` / `{summary.get('best_eval_mae', 'n/a')}`",
        f"- train/eval samples: `{summary.get('train_samples', 'n/a')}` / `{summary.get('eval_samples', 'n/a')}`",
        f"- upstream: `{summary.get('base_model_path', 'n/a')}`",
    ]


def render_scope_table_rows(
    *,
    reports: dict[str, dict[str, Any]],
    main_method: str,
    appendix_method: str,
    scope_specs: list[tuple[str, str]],
) -> list[str]:
    rows = [
        "| Scope | Sparse dense MAE | Main dense MAE | Main Δ | Appendix dense MAE | Appendix Δ |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for scope, scope_name in scope_specs:
        sparse_summary = get_scope_summary(reports["sparse"], scope, scope_name)
        if not sparse_summary:
            continue
        main_summary = get_scope_summary(reports[main_method], scope, scope_name)
        appendix_summary = get_scope_summary(reports[appendix_method], scope, scope_name)
        rows.append(
            "| "
            + f"{scope_name} | "
            + f"{sparse_summary.get('dense_weight_mae', '')} | "
            + f"{main_summary.get('dense_weight_mae', '')} | "
            + f"{round(float(main_summary.get('dense_weight_mae', 0.0)) - float(sparse_summary.get('dense_weight_mae', 0.0)), 4)} | "
            + f"{appendix_summary.get('dense_weight_mae', '')} | "
            + f"{round(float(appendix_summary.get('dense_weight_mae', 0.0)) - float(sparse_summary.get('dense_weight_mae', 0.0)), 4)} |"
        )
    return rows


def write_release_notes(
    *,
    output_path: str,
    model_path: str,
    dataset_root: str,
    reports: dict[str, dict[str, Any]],
    main_method: str,
    appendix_method: str,
    always_path: str,
    gated_path: str,
    ready_for_paper: bool,
    ready_reason: str,
    recommended_method_name: str,
) -> None:
    lines = [
        "# GAT-v2 Release Notes",
        "",
        "## Run Context",
        f"- generated_at_utc: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`",
        f"- frozen_sparse_upstream: `{model_path}`",
        f"- dataset_root: `{dataset_root}`",
        "",
        "## Checkpoints",
        "### Always-GAT-v2",
        *render_checkpoint_meta(always_path),
        "",
        "### Gated-GAT-v2",
        *render_checkpoint_meta(gated_path),
        "",
        "## Mainline Selection",
        f"- main_table_method: `{METHOD_LABELS[main_method]}`",
        f"- appendix_ablation_method: `{METHOD_LABELS[appendix_method]}`",
        "- selection_rule: `lower overall strict MAE first, then lower key-slice worst-case MAE; gated 不单独美化。`",
        "",
        "## Key Slices",
        *render_scope_table_rows(
            reports=reports,
            main_method=main_method,
            appendix_method=appendix_method,
            scope_specs=MAIN_SCOPE_SPECS,
        ),
        "",
        "## Release Decision",
        f"- GAT-v2 ready_for_paper_main_table: `{'yes' if ready_for_paper else 'no'}`",
        f"- decision_reason: {ready_reason}",
        f"- recommended_paper_method_name: `{recommended_method_name}`",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_special_scene_notes(
    *,
    output_path: str,
    reports: dict[str, dict[str, Any]],
    main_method: str,
    ready_for_paper: bool,
) -> None:
    sparse_report = reports["sparse"]
    main_report = reports[main_method]

    propagation_sparse = sparse_report["by_scene_type_overall"].get("propagation_range", {})
    propagation_main = main_report["by_scene_type_overall"].get("propagation_range", {})
    compound_sparse = sparse_report["by_scene_type_overall"].get("compound_disaster", {})
    compound_main = main_report["by_scene_type_overall"].get("compound_disaster", {})

    propagation_verdict = clear_value_judgement(propagation_sparse, propagation_main)
    compound_verdict = clear_value_judgement(compound_sparse, compound_main)

    best_scene = ""
    best_score = 0.0
    for scene_type in SPECIAL_SCENE_TYPES:
        sparse_summary = sparse_report["by_scene_type_overall"].get(scene_type, {})
        main_summary = main_report["by_scene_type_overall"].get(scene_type, {})
        if not sparse_summary or not main_summary:
            continue
        score = scene_benefit_score(sparse_summary, main_summary)
        if score > best_score:
            best_scene = scene_type
            best_score = score

    if best_score <= 0.0:
        conditional_benefit_answer = "当前没有一个场景能稳定体现 GAT 的条件性收益。"
    else:
        conditional_benefit_answer = f"`{best_scene}` 最接近条件性收益场景，但收益仍需保守表述。"

    if ready_for_paper:
        emphasis_answer = "可以写入主文，但仍应把特殊场景收益与整体结果同时报告。"
    elif best_score > 0.0:
        emphasis_answer = "建议。如果论文仍保留 GAT，应强调“特殊场景/条件性收益”，不要写成“全局稳定提升”。"
    else:
        emphasis_answer = "不建议把当前版本写成“全局稳定提升”。若保留 GAT，只能作为探索性附录，不宜在主文强推。"

    lines = [
        "# GAT-v2 Special Scene Notes",
        "",
        f"- propagation_range 是否有明确价值: {propagation_verdict}。"
        + f" sparse/main dense MAE=`{propagation_sparse.get('dense_weight_mae', 'n/a')}`/`{propagation_main.get('dense_weight_mae', 'n/a')}`，"
        + f" changed-edge MAE=`{propagation_sparse.get('MAE_on_changed_edges', 'n/a')}`/`{propagation_main.get('MAE_on_changed_edges', 'n/a')}`。",
        f"- compound_disaster 是否有明确价值: {compound_verdict}。"
        + f" sparse/main dense MAE=`{compound_sparse.get('dense_weight_mae', 'n/a')}`/`{compound_main.get('dense_weight_mae', 'n/a')}`，"
        + f" changed-edge MAE=`{compound_sparse.get('MAE_on_changed_edges', 'n/a')}`/`{compound_main.get('MAE_on_changed_edges', 'n/a')}`。",
        f"- 哪个场景最能体现 GAT 的条件性收益: {conditional_benefit_answer}",
        f"- 是否建议论文主文强调“特殊场景更有价值”而不是“全局稳定提升”: {emphasis_answer}",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_SPARSE_UPSTREAM)
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/dataset_sparse_v2")
    parser.add_argument("--always-gat", default=DEFAULT_ALWAYS_GAT_PATH)
    parser.add_argument("--gated-gat", default=DEFAULT_GATED_GAT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", default="/root/autodl-tmp/results/gat_v2_validation.json")
    parser.add_argument("--output-summary-csv", default="/root/autodl-tmp/results/gat_v2_main_summary.csv")
    parser.add_argument("--output-detail-csv", default="/root/autodl-tmp/results/gat_v2_main_details.csv")
    parser.add_argument("--output-release-notes", default="/root/autodl-tmp/results/gat_v2_release_notes.md")
    parser.add_argument("--output-special-summary-csv", default="/root/autodl-tmp/results/gat_v2_special_scene_summary.csv")
    parser.add_argument("--output-special-notes", default="/root/autodl-tmp/results/gat_v2_special_scene_notes.md")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path, device)
    edge_ids = live_edge_ids()

    split_details_by_method = {
        "sparse": {},
        "always_gat_v2": {},
        "gated_gat_v2": {},
    }
    try:
        for split in ("val_normal", "val_special", "val_anti_truncation"):
            rows = load_rows(os.path.join(args.dataset_root, split), limit=args.limit)
            split_result = evaluate_rows(
                model=model,
                tokenizer=tokenizer,
                device=device,
                rows=rows,
                split=split,
                edge_ids=edge_ids,
                always_path=args.always_gat,
                gated_path=args.gated_gat,
            )
            for method_name, details in split_result.items():
                split_details_by_method[method_name][split] = details
    finally:
        unload_model(model)

    reports = {
        method_name: build_report(
            model_path=args.model_path,
            dataset_root=args.dataset_root,
            split_details=split_details,
        )
        for method_name, split_details in split_details_by_method.items()
    }
    main_method, appendix_method = pick_mainline(reports)

    main_summary_rows = write_main_summary_csv(
        reports=reports,
        main_method=main_method,
        appendix_method=appendix_method,
        output_path=args.output_summary_csv,
    )
    write_main_detail_csv(
        split_details_by_method=split_details_by_method,
        main_method=main_method,
        appendix_method=appendix_method,
        output_path=args.output_detail_csv,
    )
    special_summary_rows = write_special_scene_summary_csv(
        reports=reports,
        main_method=main_method,
        appendix_method=appendix_method,
        output_path=args.output_special_summary_csv,
    )

    ready_for_paper, ready_reason = paper_ready_decision(
        sparse_report=reports["sparse"],
        main_report=reports[main_method],
    )
    recommended_method_name = METHOD_LABELS[main_method] if ready_for_paper else METHOD_LABELS["sparse"]

    write_release_notes(
        output_path=args.output_release_notes,
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        reports=reports,
        main_method=main_method,
        appendix_method=appendix_method,
        always_path=args.always_gat,
        gated_path=args.gated_gat,
        ready_for_paper=ready_for_paper,
        ready_reason=ready_reason,
        recommended_method_name=recommended_method_name,
    )
    write_special_scene_notes(
        output_path=args.output_special_notes,
        reports=reports,
        main_method=main_method,
        ready_for_paper=ready_for_paper,
    )

    final_report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_sparse_upstream": args.model_path,
        "dataset_root": args.dataset_root,
        "always_gat_path": args.always_gat,
        "gated_gat_path": args.gated_gat,
        "main_method": {
            "key": main_method,
            "label": METHOD_LABELS[main_method],
        },
        "appendix_method": {
            "key": appendix_method,
            "label": METHOD_LABELS[appendix_method],
        },
        "paper_decision": {
            "gat_v2_ready_for_main_table": ready_for_paper,
            "reason": ready_reason,
            "recommended_method_name": recommended_method_name,
        },
        "reports": reports,
        "main_summary_rows": main_summary_rows,
        "special_summary_rows": special_summary_rows,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "frozen_sparse_upstream": args.model_path,
                "main_method": METHOD_LABELS[main_method],
                "appendix_method": METHOD_LABELS[appendix_method],
                "gat_v2_ready_for_paper_main_table": ready_for_paper,
                "recommended_method_name": recommended_method_name,
                "summary_csv": args.output_summary_csv,
                "detail_csv": args.output_detail_csv,
                "release_notes": args.output_release_notes,
                "special_scene_summary_csv": args.output_special_summary_csv,
                "special_scene_notes": args.output_special_notes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
