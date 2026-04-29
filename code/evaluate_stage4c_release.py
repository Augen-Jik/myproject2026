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

RELIEF_COMPONENTS = {
    "singleedge_relief_core",
    "short_range_relief_support",
    "multi_edge_relief_tail",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage4c strict/shadow/relief/guard evaluation and build release artifacts.")
    parser.add_argument("--model-path", default="/root/autodl-tmp/model_merged_sparse_v2_stage4c_fix")
    parser.add_argument("--lora-root", default="/root/autodl-tmp/model_lora_sparse_v2_stage4c_fix")
    parser.add_argument("--heldout-root", default="/root/autodl-tmp/dataset_sparse_v2")
    parser.add_argument("--stage4c-root", default="/root/autodl-tmp/dataset_sparse_v2_stage4c")
    parser.add_argument("--results-root", default="/root/autodl-tmp/results")
    parser.add_argument("--diff-json", default="/root/autodl-tmp/results/stage4c_diff_29.json")
    parser.add_argument("--baseline-summary-csv", default="/root/autodl-tmp/results/sparse_v2_validation_strict_stage4_fix_summary.csv")
    parser.add_argument("--stage4b-summary-csv", default="/root/autodl-tmp/results/sparse_v2_validation_strict_stage4b_fix_summary.csv")
    parser.add_argument("--guard-baseline-csv", default="/root/autodl-tmp/results/stage4b_guard_eval_summary.csv")
    return parser.parse_args()


def enrich_details(details: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for detail, row in zip(details, rows, strict=True):
        merged = dict(detail)
        for field in [
            "mixture_component",
            "bucket",
            "guard_focus",
            "anchor_style",
            "severity",
            "source_split",
            "source_type",
            "edge_profile",
            "risk_flag",
            "phrase_variant",
        ]:
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
    raise KeyError(f"Missing summary row {scope}/{scope_name}")


def to_float(summary_row: dict[str, Any], key: str) -> float:
    return float(summary_row.get(key, 0.0))


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


def latest_checkpoint_state(lora_root: Path) -> dict[str, Any]:
    checkpoints = sorted(
        [path for path in lora_root.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found under {lora_root}")
    state_path = checkpoints[-1] / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing trainer_state.json: {state_path}")
    data = json.loads(state_path.read_text(encoding="utf-8"))
    data["_trainer_state_path"] = str(state_path)
    return data


def extract_training_readout(trainer_state: dict[str, Any]) -> dict[str, Any]:
    history = trainer_state.get("log_history", []) or []
    loss_entries = [item for item in history if "loss" in item]
    eval_entries = [item for item in history if "eval_loss" in item]
    lr_entries = [item for item in history if "learning_rate" in item]

    final_train_loss = float(loss_entries[-1]["loss"]) if loss_entries else None
    final_eval_loss = float(eval_entries[-1]["eval_loss"]) if eval_entries else None
    final_lr = float(lr_entries[-1]["learning_rate"]) if lr_entries else None
    max_steps = int(trainer_state.get("max_steps", 0) or 0)
    global_step = int(trainer_state.get("global_step", 0) or 0)
    learning_rates = [float(item["learning_rate"]) for item in lr_entries if "learning_rate" in item]
    stable = all(
        str(item.get("loss", "")).lower() not in {"nan", "inf", "-inf"}
        for item in loss_entries
    ) and all(
        str(item.get("eval_loss", "")).lower() not in {"nan", "inf", "-inf"}
        for item in eval_entries
    )
    lr_curve_too_short = global_step < 24
    return {
        "trainer_state_path": trainer_state["_trainer_state_path"],
        "actual_global_step": global_step,
        "max_steps": max_steps,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "final_learning_rate": final_lr,
        "first_learning_rate": learning_rates[0] if learning_rates else None,
        "min_learning_rate_logged": min(learning_rates) if learning_rates else None,
        "stable": stable,
        "learning_rate_curve_too_short": lr_curve_too_short,
    }


def bool_text(value: bool) -> str:
    return "是" if value else "否"


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    heldout_json = results_root / "sparse_v2_validation_strict_stage4c_fix.json"
    heldout_summary_csv = results_root / "sparse_v2_validation_strict_stage4c_fix_summary.csv"
    heldout_detail_csv = results_root / "sparse_v2_validation_strict_stage4c_fix_details.csv"
    shadow_summary_csv = results_root / "stage4c_shadow_eval_summary.csv"
    relief_summary_csv = results_root / "relief_pack_stage4c_eval_summary.csv"
    guard_summary_csv = results_root / "stage4c_guard_eval_summary.csv"
    release_md = results_root / "stage4c_release_decision.md"
    release_summary_csv = results_root / "stage4c_release_decision_summary.csv"

    trainer_state = latest_checkpoint_state(Path(args.lora_root))
    training_readout = extract_training_readout(trainer_state)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path, device)

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

        stage4c_eval_rows = load_rows(str(Path(args.stage4c_root) / "eval"))
        relief_rows = [row for row in stage4c_eval_rows if str(row.get("mixture_component", "")) in RELIEF_COMPONENTS]
        guard_rows = [row for row in stage4c_eval_rows if str(row.get("mixture_component", "")) == "guard_minimal"]

        shadow_overall, shadow_grouped, _ = evaluate_slice(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=stage4c_eval_rows,
            split_name="stage4c_shadow_eval",
            group_fields=["mixture_component", "bucket", "edge_profile", "source_type"],
            output_summary_csv=shadow_summary_csv,
        )
        relief_overall, relief_grouped, _ = evaluate_slice(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=relief_rows,
            split_name="relief_pack_stage4c_eval",
            group_fields=["mixture_component", "bucket", "edge_profile", "phrase_variant"],
            output_summary_csv=relief_summary_csv,
        )
        guard_overall, guard_grouped, _ = evaluate_slice(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=guard_rows,
            split_name="stage4c_guard_eval",
            group_fields=["bucket", "scene_type", "length_bucket"],
            output_summary_csv=guard_summary_csv,
        )
    finally:
        unload_model(model)

    base_rows = read_summary_csv(Path(args.baseline_summary_csv))
    stage4b_rows = read_summary_csv(Path(args.stage4b_summary_csv))
    guard_baseline_rows = read_summary_csv(Path(args.guard_baseline_csv))
    diff_payload = json.loads(Path(args.diff_json).read_text(encoding="utf-8"))

    simple_local_fix = lookup_summary(base_rows, scope="overall:scene_type", scope_name="simple_local")
    simple_local_stage4c = lookup_summary(read_summary_csv(heldout_summary_csv), scope="overall:scene_type", scope_name="simple_local")
    directional_fix = lookup_summary(base_rows, scope="overall:scene_type", scope_name="directional_asymmetry")
    directional_stage4c = lookup_summary(read_summary_csv(heldout_summary_csv), scope="overall:scene_type", scope_name="directional_asymmetry")
    anti_fix = lookup_summary(base_rows, scope="split", scope_name="val_anti_truncation")
    anti_stage4c = lookup_summary(read_summary_csv(heldout_summary_csv), scope="split", scope_name="val_anti_truncation")
    stage4b_simple_local = lookup_summary(stage4b_rows, scope="overall:scene_type", scope_name="simple_local")
    stage4b_guard_overall = lookup_summary(guard_baseline_rows, scope="overall", scope_name="overall")

    simple_local_rate = to_float(simple_local_stage4c, "no_anchor_when_gt_changed_rate")
    noticeably_lower_than_02762 = simple_local_rate <= 0.2262
    reaches_015 = simple_local_rate <= 0.15

    anti_truncation_ok = (
        to_float(anti_stage4c, "no_anchor_when_gt_changed_rate") <= to_float(anti_fix, "no_anchor_when_gt_changed_rate") + 0.03
        and to_float(anti_stage4c, "parse_fail_rate_strict") <= to_float(anti_fix, "parse_fail_rate_strict") + 0.03
    )
    directional_ok = (
        to_float(directional_stage4c, "no_anchor_when_gt_changed_rate") <= to_float(directional_fix, "no_anchor_when_gt_changed_rate") + 0.03
        and to_float(directional_stage4c, "parse_fail_rate_strict") <= to_float(directional_fix, "parse_fail_rate_strict") + 0.03
        and to_float(directional_stage4c, "gt_changed_edge_recall") >= to_float(directional_fix, "gt_changed_edge_recall") - 3.0
    )

    changed_subtype_breakdown = diff_payload.get("stats", {}).get("changed_subtype_breakdown", {})
    single_edge_changed = int(changed_subtype_breakdown.get("relief_normal_single_edge", 0))
    other_changed = sum(
        int(value)
        for key, value in changed_subtype_breakdown.items()
        if key != "relief_normal_single_edge"
    )
    single_edge_dominates = single_edge_changed > other_changed and single_edge_changed > 0
    moved_29 = int(diff_payload.get("completely_unchanged_count", diff_payload.get("stats", {}).get("completely_unchanged_count", 29))) < 29
    no_anchor_to_anchor_count = int(diff_payload.get("no_anchor_to_anchor_count", diff_payload.get("stats", {}).get("no_anchor_to_anchor_count", 0)))

    guard_stable = (
        anti_truncation_ok
        and directional_ok
        and to_float(guard_overall, "parse_fail_rate_strict") <= max(0.05, to_float(stage4b_guard_overall, "parse_fail_rate_strict") + 0.03)
    )

    decision = (
        "RELEASE_TO_GAT_V2"
        if moved_29 and reaches_015 and guard_stable and training_readout["stable"] and not training_readout["learning_rate_curve_too_short"]
        else "HOLD_AND_REPAIR"
    )

    decision_rows: list[dict[str, Any]] = []
    for model_version, rows in (
        ("stage4_fix", base_rows),
        ("stage4b_fix", stage4b_rows),
        ("stage4c_fix", read_summary_csv(heldout_summary_csv)),
    ):
        decision_rows.append(metric_row(model_version, "overall:scene_type", "simple_local", lookup_summary(rows, scope="overall:scene_type", scope_name="simple_local")))
        decision_rows.append(metric_row(model_version, "overall:scene_type", "directional_asymmetry", lookup_summary(rows, scope="overall:scene_type", scope_name="directional_asymmetry")))
        decision_rows.append(metric_row(model_version, "split", "val_anti_truncation", lookup_summary(rows, scope="split", scope_name="val_anti_truncation")))
    decision_rows.append(metric_row("stage4c_fix", "shadow_eval", "overall", shadow_overall))
    decision_rows.append(metric_row("stage4c_fix", "relief_eval", "overall", relief_overall))
    decision_rows.append(metric_row("stage4c_fix", "guard_eval", "overall", guard_overall))

    with release_summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["model_version", "scope", "scope_name", *METRIC_COLUMNS, "note"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in decision_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    md_lines = [
        "# stage4c Release Decision",
        "",
        "## Required Answers",
        f"- 29 条是否终于“动了”：`{bool_text(moved_29)}`（completely_unchanged_count=`{diff_payload.get('completely_unchanged_count', diff_payload.get('stats', {}).get('completely_unchanged_count', ''))}` / 29）",
        f"- 其中多少条是 no_anchor_to_anchor：`{no_anchor_to_anchor_count}`",
        f"- single_edge 是否开始主导改善：`{bool_text(single_edge_dominates)}`（single_edge_changed=`{single_edge_changed}`，other_changed=`{other_changed}`）",
        f"- simple_local no_anchor_when_gt_changed_rate 是否明显低于 0.2762：`{bool_text(noticeably_lower_than_02762)}`（stage4c=`{simple_local_rate:.4f}`）",
        f"- 是否达到 <= 0.15：`{bool_text(reaches_015)}`",
        f"- anti_truncation / directional_asymmetry 是否仍稳定：`{bool_text(anti_truncation_ok and directional_ok)}`",
        "",
        "## Training Readout",
        f"- actual training steps: `{training_readout['actual_global_step']}` / planned `{training_readout['max_steps']}`",
        f"- final train loss: `{training_readout['final_train_loss']}`",
        f"- eval loss: `{training_readout['final_eval_loss']}`",
        f"- 是否稳定: `{training_readout['stable']}`",
        f"- 学习率曲线是否仍过短: `{training_readout['learning_rate_curve_too_short']}`",
        f"- trainer_state: `{training_readout['trainer_state_path']}`",
        "",
        "## Strict Gate",
        f"- stage4_fix simple_local: `no_anchor_when_gt_changed_rate={simple_local_fix['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={simple_local_fix['gt_changed_edge_recall']}`, `parse_fail_rate_strict={simple_local_fix['parse_fail_rate_strict']}`",
        f"- stage4b_fix simple_local: `no_anchor_when_gt_changed_rate={stage4b_simple_local['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={stage4b_simple_local['gt_changed_edge_recall']}`, `parse_fail_rate_strict={stage4b_simple_local['parse_fail_rate_strict']}`",
        f"- stage4c_fix simple_local: `no_anchor_when_gt_changed_rate={simple_local_stage4c['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={simple_local_stage4c['gt_changed_edge_recall']}`, `parse_fail_rate_strict={simple_local_stage4c['parse_fail_rate_strict']}`",
        "",
        "## Slice Readout",
        f"- shadow eval overall: `no_anchor_when_gt_changed_rate={shadow_overall['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={shadow_overall['gt_changed_edge_recall']}`, `parse_fail_rate_strict={shadow_overall['parse_fail_rate_strict']}`",
        f"- relief eval overall: `no_anchor_when_gt_changed_rate={relief_overall['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={relief_overall['gt_changed_edge_recall']}`, `parse_fail_rate_strict={relief_overall['parse_fail_rate_strict']}`",
        f"- guard eval overall: `no_anchor_when_gt_changed_rate={guard_overall['no_anchor_when_gt_changed_rate']}`, `gt_changed_edge_recall={guard_overall['gt_changed_edge_recall']}`, `parse_fail_rate_strict={guard_overall['parse_fail_rate_strict']}`",
        "",
        "## Final Decision",
        f"- `{decision}`",
    ]
    release_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "decision": decision,
                "moved_29": moved_29,
                "no_anchor_to_anchor_count": no_anchor_to_anchor_count,
                "single_edge_dominates": single_edge_dominates,
                "simple_local_no_anchor_when_gt_changed_rate": simple_local_rate,
                "reaches_015": reaches_015,
                "anti_truncation_ok": anti_truncation_ok,
                "directional_ok": directional_ok,
                "training_stable": training_readout["stable"],
                "learning_rate_curve_too_short": training_readout["learning_rate_curve_too_short"],
                "heldout_summary_csv": str(heldout_summary_csv),
                "shadow_summary_csv": str(shadow_summary_csv),
                "relief_summary_csv": str(relief_summary_csv),
                "guard_summary_csv": str(guard_summary_csv),
                "release_md": str(release_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
