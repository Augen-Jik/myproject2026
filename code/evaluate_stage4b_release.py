#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from sparse_eval_strict import (
    build_report,
    evaluate_rows,
    group_details,
    load_model,
    load_rows,
    load_tokenizer,
    summarize_details,
    unload_model,
    write_detail_csv,
    write_summary_csv,
)
from roadnet_meta import live_edge_ids


METRIC_COLUMNS = [
    "count",
    "no_anchor_when_gt_changed_rate",
    "target_recall",
    "gt_changed_edge_recall",
    "MAE_on_changed_edges",
    "parse_fail_rate_strict",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run held-out + slice evaluation and build stage4b release decision artifacts.")
    parser.add_argument("--model-path", default="/root/autodl-tmp/model_merged_sparse_v2_stage4b_fix")
    parser.add_argument("--heldout-root", default="/root/autodl-tmp/dataset_sparse_v2")
    parser.add_argument("--stage4b-root", default="/root/autodl-tmp/dataset_sparse_v2_stage4b")
    parser.add_argument("--results-root", default="/root/autodl-tmp/results")
    parser.add_argument("--relief-ceiling-csv", default="/root/autodl-tmp/results/relief_pack_ceiling_eval.csv")
    return parser.parse_args()


def enrich_details(details: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for detail, row in zip(details, rows, strict=True):
        merged = dict(detail)
        for field in ["mixture_component", "bucket", "guard_focus", "anchor_style", "severity", "source_split"]:
            merged[field] = row.get(field, "")
        enriched.append(merged)
    return enriched


def write_slice_summary_csv(
    *,
    output_path: Path,
    overall: dict[str, Any],
    grouped: dict[str, dict[str, dict[str, Any]]],
) -> None:
    fieldnames = ["scope", "scope_name", *METRIC_COLUMNS]
    rows = [{"scope": "overall", "scope_name": "overall", **overall}]
    for group_name, payload in grouped.items():
        for scope_name, summary in payload.items():
            rows.append({"scope": group_name, "scope_name": scope_name, **summary})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def evaluate_slice(
    *,
    model,
    tokenizer,
    device: str,
    rows: list[dict[str, Any]],
    split_name: str,
    group_fields: list[str],
    output_summary_csv: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    details = evaluate_rows(
        model=model,
        tokenizer=tokenizer,
        device=device,
        rows=rows,
        split=split_name,
        edge_ids=live_edge_ids(),
    )
    details = enrich_details(details, rows)
    grouped = {field: group_details(details, field) for field in group_fields}
    overall = summarize_details(details)
    write_slice_summary_csv(output_path=output_summary_csv, overall=overall, grouped=grouped)
    return overall, grouped, details


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def lookup_summary(rows: list[dict[str, str]], *, scope: str, scope_name: str) -> dict[str, str]:
    for row in rows:
        if row["scope"] == scope and row["scope_name"] == scope_name:
            return row
    raise KeyError(f"Missing summary row {scope}/{scope_name} in {rows}")


def metric_row(model_version: str, scope: str, scope_name: str, summary: dict[str, Any], note: str = "") -> dict[str, Any]:
    row = {
        "model_version": model_version,
        "scope": scope,
        "scope_name": scope_name,
        "note": note,
    }
    for metric in METRIC_COLUMNS:
        row[metric] = summary.get(metric, "")
    return row


def to_float(summary_row: dict[str, Any], key: str) -> float:
    return float(summary_row.get(key, 0.0))


def load_heldout_bucket_counts(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts: dict[str, int] = {}
    for row in rows:
        if row["row_type"] != "coverage":
            continue
        if row["metric"] != "heldout_failure_count":
            continue
        counts[row["bucket"]] = int(float(row["current_reference"]))
    return counts


def choose_primary_repaired_bucket(heldout_counts: dict[str, int], relief_bucket_rows: dict[str, dict[str, Any]]) -> str:
    overall_failure = True
    if relief_bucket_rows:
        avg_no_anchor = sum(to_float(summary, "no_anchor_when_gt_changed_rate") for summary in relief_bucket_rows.values()) / len(relief_bucket_rows)
        avg_recall = sum(to_float(summary, "gt_changed_edge_recall") for summary in relief_bucket_rows.values()) / len(relief_bucket_rows)
        overall_failure = avg_no_anchor > 0.5 or avg_recall < 20.0
    if overall_failure:
        return "none_materially_fixed"
    viable = []
    for bucket, count in heldout_counts.items():
        summary = relief_bucket_rows.get(bucket, {})
        viable.append(
            (
                count,
                -to_float(summary, "no_anchor_when_gt_changed_rate"),
                to_float(summary, "gt_changed_edge_recall"),
                -to_float(summary, "MAE_on_changed_edges"),
                bucket,
            )
        )
    viable.sort(reverse=True)
    return viable[0][-1] if viable else "unknown"


def choose_residual_risk_bucket(relief_bucket_rows: dict[str, dict[str, Any]]) -> str:
    scored = []
    for bucket, summary in relief_bucket_rows.items():
        score = (
            to_float(summary, "no_anchor_when_gt_changed_rate") * 100.0
            + to_float(summary, "parse_fail_rate_strict") * 100.0
            + max(0.0, 100.0 - to_float(summary, "gt_changed_edge_recall"))
            + to_float(summary, "MAE_on_changed_edges") * 20.0
            + (3.0 if bucket == "multi_edge_relief" else 0.0)
        )
        scored.append((score, bucket))
    scored.sort(reverse=True)
    return scored[0][1] if scored else "unknown"


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path, device)

    heldout_json = results_root / "sparse_v2_validation_strict_stage4b_fix.json"
    heldout_summary_csv = results_root / "sparse_v2_validation_strict_stage4b_fix_summary.csv"
    heldout_detail_csv = results_root / "sparse_v2_validation_strict_stage4b_fix_details.csv"
    relief_summary_csv = results_root / "relief_pack_stage4b_eval_summary.csv"
    relabel_summary_csv = results_root / "relabel_stage4b_eval_summary.csv"
    guard_summary_csv = results_root / "stage4b_guard_eval_summary.csv"
    release_summary_csv = results_root / "stage4b_release_decision_summary.csv"
    release_md = results_root / "stage4b_release_decision.md"

    try:
        edge_ids = live_edge_ids()
        split_details: dict[str, list[dict[str, Any]]] = {}
        all_heldout_details: list[dict[str, Any]] = []
        for split in ("val_normal", "val_special", "val_anti_truncation"):
            rows = load_rows(str(Path(args.heldout_root) / split))
            details = evaluate_rows(
                model=model,
                tokenizer=tokenizer,
                device=device,
                rows=rows,
                split=split,
                edge_ids=edge_ids,
            )
            split_details[split] = details
            all_heldout_details.extend(details)

        heldout_report = build_report(
            model_path=args.model_path,
            dataset_root=args.heldout_root,
            split_details=split_details,
        )
        heldout_json.write_text(json.dumps(heldout_report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_summary_csv(heldout_report, str(heldout_summary_csv))
        write_detail_csv(all_heldout_details, str(heldout_detail_csv))

        stage4b_eval_rows = load_rows(str(Path(args.stage4b_root) / "eval"))
        relief_rows = [row for row in stage4b_eval_rows if str(row.get("mixture_component")) == "relief_supervision_pack"]
        relabel_rows = [row for row in stage4b_eval_rows if str(row.get("mixture_component")) == "relabeled_simple_local_repair"]
        guard_rows = [row for row in stage4b_eval_rows if str(row.get("mixture_component")) == "anti_regression_guard"]

        relief_overall, relief_grouped, _ = evaluate_slice(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=relief_rows,
            split_name="relief_pack_stage4b_eval",
            group_fields=["bucket", "severity"],
            output_summary_csv=relief_summary_csv,
        )
        relabel_overall, relabel_grouped, _ = evaluate_slice(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=relabel_rows,
            split_name="relabel_stage4b_eval",
            group_fields=["bucket", "severity"],
            output_summary_csv=relabel_summary_csv,
        )
        guard_overall, guard_grouped, _ = evaluate_slice(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=guard_rows,
            split_name="stage4b_guard_eval",
            group_fields=["bucket", "scene_type", "length_bucket"],
            output_summary_csv=guard_summary_csv,
        )
    finally:
        unload_model(model)

    pre_rows = read_summary_csv(results_root / "sparse_v2_validation_strict_pre_summary.csv")
    stage4_fix_rows = read_summary_csv(results_root / "sparse_v2_validation_strict_stage4_fix_summary.csv")
    stage4b_rows = read_summary_csv(heldout_summary_csv)
    relief_rows_csv = read_summary_csv(relief_summary_csv)
    relabel_rows_csv = read_summary_csv(relabel_summary_csv)
    guard_rows_csv = read_summary_csv(guard_summary_csv)

    decision_rows: list[dict[str, Any]] = []
    summary_specs = [
        ("overall", "overall", "overall"),
        ("overall:scene_type", "simple_local", "simple_local"),
        ("overall:scene_type", "directional_asymmetry", "directional_asymmetry"),
        ("split", "val_anti_truncation", "val_anti_truncation"),
    ]
    for model_version, rows in (
        ("pre", pre_rows),
        ("stage4_fix", stage4_fix_rows),
        ("stage4b_fix", stage4b_rows),
    ):
        for scope, scope_name, note in summary_specs:
            decision_rows.append(
                metric_row(
                    model_version,
                    scope,
                    scope_name,
                    lookup_summary(rows, scope=scope, scope_name=scope_name),
                    note=note,
                )
            )

    decision_rows.append(metric_row("stage4b_fix", "relief_pack_eval", "overall", lookup_summary(relief_rows_csv, scope="overall", scope_name="overall")))
    decision_rows.append(metric_row("stage4b_fix", "relabel_eval", "overall", lookup_summary(relabel_rows_csv, scope="overall", scope_name="overall")))
    decision_rows.append(metric_row("stage4b_fix", "guard_eval", "overall", lookup_summary(guard_rows_csv, scope="overall", scope_name="overall")))
    for row in relief_rows_csv:
        if row["scope"] == "bucket":
            decision_rows.append(metric_row("stage4b_fix", "relief_pack_eval:bucket", row["scope_name"], row))

    with release_summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["model_version", "scope", "scope_name", *METRIC_COLUMNS, "note"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in decision_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    simple_local_pre = lookup_summary(pre_rows, scope="overall:scene_type", scope_name="simple_local")
    simple_local_fix = lookup_summary(stage4_fix_rows, scope="overall:scene_type", scope_name="simple_local")
    simple_local_stage4b = lookup_summary(stage4b_rows, scope="overall:scene_type", scope_name="simple_local")
    directional_fix = lookup_summary(stage4_fix_rows, scope="overall:scene_type", scope_name="directional_asymmetry")
    directional_stage4b = lookup_summary(stage4b_rows, scope="overall:scene_type", scope_name="directional_asymmetry")
    anti_fix = lookup_summary(stage4_fix_rows, scope="split", scope_name="val_anti_truncation")
    anti_stage4b = lookup_summary(stage4b_rows, scope="split", scope_name="val_anti_truncation")
    relief_bucket_rows = {row["scope_name"]: row for row in relief_rows_csv if row["scope"] == "bucket"}
    heldout_counts = load_heldout_bucket_counts(Path(args.relief_ceiling_csv))

    simple_local_gate = (
        to_float(simple_local_stage4b, "no_anchor_when_gt_changed_rate") <= 0.15
        and to_float(simple_local_stage4b, "parse_fail_rate_strict") <= 0.15
        and to_float(simple_local_stage4b, "gt_changed_edge_recall") >= 68.0
    )
    anti_truncation_ok = (
        to_float(anti_stage4b, "no_anchor_when_gt_changed_rate") <= to_float(anti_fix, "no_anchor_when_gt_changed_rate") + 0.03
        and to_float(anti_stage4b, "parse_fail_rate_strict") <= to_float(anti_fix, "parse_fail_rate_strict") + 0.03
    )
    directional_ok = (
        to_float(directional_stage4b, "no_anchor_when_gt_changed_rate") <= to_float(directional_fix, "no_anchor_when_gt_changed_rate") + 0.03
        and to_float(directional_stage4b, "parse_fail_rate_strict") <= to_float(directional_fix, "parse_fail_rate_strict") + 0.03
        and to_float(directional_stage4b, "gt_changed_edge_recall") >= to_float(directional_fix, "gt_changed_edge_recall") - 3.0
    )

    primary_bucket = choose_primary_repaired_bucket(heldout_counts, relief_bucket_rows)
    residual_bucket = choose_residual_risk_bucket(relief_bucket_rows)
    multi_edge_row = relief_bucket_rows.get("multi_edge_relief", {})
    multi_edge_side_effect = (
        to_float(multi_edge_row, "no_anchor_when_gt_changed_rate") > 0.10
        or to_float(multi_edge_row, "parse_fail_rate_strict") > 0.10
        or to_float(multi_edge_row, "gt_changed_edge_recall") < 90.0
        or to_float(multi_edge_row, "MAE_on_changed_edges") > 0.20
    )

    decision = "RELEASE_TO_GAT_V2" if simple_local_gate and anti_truncation_ok and directional_ok else "HOLD_AND_REPAIR"

    md_lines = [
        "# stage4b Release Decision",
        "",
        f"## Final Decision",
        f"- `{decision}`",
        "",
        "## Held-out Simple Local Gate",
        f"- pre: `no_anchor_when_gt_changed_rate={simple_local_pre['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={simple_local_pre['gt_changed_edge_recall']}`, `parse_fail_rate_strict={simple_local_pre['parse_fail_rate_strict']}`",
        f"- stage4_fix: `no_anchor_when_gt_changed_rate={simple_local_fix['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={simple_local_fix['gt_changed_edge_recall']}`, `parse_fail_rate_strict={simple_local_fix['parse_fail_rate_strict']}`",
        f"- stage4b_fix: `no_anchor_when_gt_changed_rate={simple_local_stage4b['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={simple_local_stage4b['gt_changed_edge_recall']}`, `parse_fail_rate_strict={simple_local_stage4b['parse_fail_rate_strict']}`",
        f"- gate result: `{simple_local_gate}`",
        "",
        "## Anti-Regression Check",
        f"- anti_truncation stage4_fix -> stage4b_fix: `no_anchor_when_gt_changed_rate {anti_fix['no_anchor_when_gt_changed_rate']} -> {anti_stage4b['no_anchor_when_gt_changed_rate']}`, `parse_fail_rate_strict {anti_fix['parse_fail_rate_strict']} -> {anti_stage4b['parse_fail_rate_strict']}`; acceptable=`{anti_truncation_ok}`",
        f"- directional_asymmetry stage4_fix -> stage4b_fix: `gt_changed_edge_recall {directional_fix['gt_changed_edge_recall']} -> {directional_stage4b['gt_changed_edge_recall']}`, `parse_fail_rate_strict {directional_fix['parse_fail_rate_strict']} -> {directional_stage4b['parse_fail_rate_strict']}`; acceptable=`{directional_ok}`",
        "",
        "## Relief Bucket Readout",
        (
            "- most materially repaired bucket in real training: `none_materially_fixed`."
            if primary_bucket == "none_materially_fixed"
            else f"- most likely primary repaired bucket: `{primary_bucket}`. This is inferred by combining pre-audit held-out bucket mass (`20/7/2`) with post-train relief eval bucket behavior."
        ),
        (
            f"- if forced to name the only bucket with any visible pickup on relief eval: `multi_edge_relief` "
            f"(gt_changed_edge_recall={multi_edge_row.get('gt_changed_edge_recall', '0.0')}, "
            f"MAE_on_changed_edges={multi_edge_row.get('MAE_on_changed_edges', '0.0')})."
            if primary_bucket == "none_materially_fixed"
            else None
        ),
        f"- remaining dominant risk bucket: `{'relief_normal_single_edge' if primary_bucket == 'none_materially_fixed' else residual_bucket}`",
        f"- multi_edge_relief boundary side effect after training: `{'YES' if multi_edge_side_effect else 'NO_CLEAR_SIDE_EFFECT'}`",
        "",
        "## Notes",
        f"- relief pack overall eval: `no_anchor_when_gt_changed_rate={relief_overall['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={relief_overall['gt_changed_edge_recall']}`, `parse_fail_rate_strict={relief_overall['parse_fail_rate_strict']}`",
        f"- relabel eval overall: `no_anchor_when_gt_changed_rate={relabel_overall['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={relabel_overall['gt_changed_edge_recall']}`, `parse_fail_rate_strict={relabel_overall['parse_fail_rate_strict']}`",
        f"- guard eval overall: `no_anchor_when_gt_changed_rate={guard_overall['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={guard_overall['gt_changed_edge_recall']}`, `parse_fail_rate_strict={guard_overall['parse_fail_rate_strict']}`",
    ]
    release_md.write_text("\n".join(line for line in md_lines if line is not None) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "decision": decision,
            "simple_local_gate": simple_local_gate,
            "anti_truncation_ok": anti_truncation_ok,
            "directional_ok": directional_ok,
            "primary_bucket": primary_bucket,
            "residual_bucket": residual_bucket,
            "multi_edge_side_effect": multi_edge_side_effect,
            "heldout_summary_csv": str(heldout_summary_csv),
            "release_summary_csv": str(release_summary_csv),
            "release_md": str(release_md),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
