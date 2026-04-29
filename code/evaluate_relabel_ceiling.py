#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


CODE_DIR = Path("/root/autodl-tmp/code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from roadnet_meta import live_edge_ids  # noqa: E402
from sparse_eval_strict import dense_prediction, score_prediction, summarize_details  # noqa: E402
from sparse_utils import parse_sparse_output_bundle  # noqa: E402


METRICS = [
    "no_anchor_when_gt_changed_rate",
    "target_recall",
    "gt_changed_edge_recall",
    "MAE_on_changed_edges",
    "parse_fail_rate_strict",
]
TARGET_BUCKET = "anomaly_plus_normal_side_event"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate relabel oracle ceiling without training.")
    parser.add_argument(
        "--relabel-examples-json",
        default="/root/autodl-tmp/dataset_sparse_v2_stage4b_relabel/relabel_examples.json",
    )
    parser.add_argument(
        "--relabel-mapping-csv",
        default="/root/autodl-tmp/dataset_sparse_v2_stage4b_relabel/relabel_mapping.csv",
    )
    parser.add_argument(
        "--strict-details-csv",
        default="/root/autodl-tmp/results/sparse_v2_validation_strict_stage4_fix_details.csv",
    )
    parser.add_argument(
        "--stage4b-train-parquet",
        default="/root/autodl-tmp/dataset_sparse_v2_stage4b_relabel/train/data.parquet",
    )
    parser.add_argument(
        "--stage4b-eval-parquet",
        default="/root/autodl-tmp/dataset_sparse_v2_stage4b_relabel/eval/data.parquet",
    )
    parser.add_argument(
        "--policy-md",
        default="/root/autodl-tmp/results/simple_local_v2_label_policy.md",
    )
    parser.add_argument(
        "--output-csv",
        default="/root/autodl-tmp/results/relabel_ceiling_eval.csv",
    )
    parser.add_argument(
        "--output-notes",
        default="/root/autodl-tmp/results/relabel_ceiling_notes.md",
    )
    return parser.parse_args()


def clean_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.replace("\\n", "\n").strip()


def safe_json_loads(raw: str):
    if not isinstance(raw, str) or not raw:
        return []
    return json.loads(raw)


def parse_label_body(label: str) -> list[str]:
    lines = [line.strip() for line in clean_text(label).splitlines() if line.strip()]
    body = []
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


def label_anchor_count(label: str) -> int:
    return sum(1 for line in parse_label_body(label) if line.startswith("ANCHOR|"))


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def load_stage4b_rows(train_path: Path, eval_path: Path) -> pd.DataFrame:
    frames = []
    for parquet_path in [train_path, eval_path]:
        table = pq.read_table(parquet_path)
        df = pd.DataFrame(table.to_pydict())
        df["split_index"] = range(len(df))
        df["sample_id"] = df["split"].astype(str) + ":" + df["split_index"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def evaluate_label_column(df: pd.DataFrame, label_col: str, edge_ids: list[str]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_generation = clean_text(row[label_col])
        bundle = parse_sparse_output_bundle(raw_generation, scene_type=row["scene_type"])
        pred_weights = dense_prediction(bundle, edge_ids)
        details.append(
            score_prediction(
                row=row.to_dict(),
                bundle=bundle,
                pred_weights=pred_weights,
                edge_ids=edge_ids,
                sample_id=row["sample_id"],
                split=row["split"],
                split_index=int(row["split_index"]),
                raw_generation=raw_generation,
                infer_time_s=0.0,
            )
        )
    return details


def build_summary_from_details_csv(details_df: pd.DataFrame) -> dict[str, float]:
    simple = details_df[details_df["scene_type"].astype(str) == "simple_local"].copy()
    if simple.empty:
        raise ValueError("No simple_local rows found in current strict details CSV.")

    changed_samples = simple[simple["gt_changed_edge_count"] > 0]
    target_edge_total = float(simple["gt_target_edge_count"].sum())
    changed_edge_total = float(simple["gt_changed_edge_count"].sum())

    def weighted_metric(value_col: str, weight_col: str) -> float:
        weights = pd.to_numeric(simple[weight_col], errors="coerce").fillna(0)
        values = pd.to_numeric(simple[value_col], errors="coerce")
        valid = values.notna() & weights.notna()
        if not valid.any() or float(weights[valid].sum()) <= 0:
            return math.nan
        return float((values[valid] * weights[valid]).sum() / weights[valid].sum())

    return {
        "no_anchor_when_gt_changed_rate": round(float(changed_samples["no_anchor"].astype(bool).mean()), 4) if len(changed_samples) else 0.0,
        "target_recall": round(weighted_metric("target_recall", "gt_target_edge_count"), 1) if target_edge_total else 0.0,
        "gt_changed_edge_recall": round(weighted_metric("gt_changed_edge_recall", "gt_changed_edge_count"), 1) if changed_edge_total else 0.0,
        "MAE_on_changed_edges": weighted_metric("MAE_on_changed_edges", "gt_changed_edge_count"),
        "parse_fail_rate_strict": round(float(simple["strict_parse_fail"].astype(bool).mean()), 4),
        "count": int(len(simple)),
    }


def load_current_stage4_fix_reference(details_csv: Path) -> tuple[dict[str, float], str]:
    details_df = pd.read_csv(details_csv)
    summary = build_summary_from_details_csv(details_df)
    mae_missing = pd.to_numeric(details_df[details_df["scene_type"] == "simple_local"]["MAE_on_changed_edges"], errors="coerce").notna().sum() == 0
    fallback_note = ""
    if mae_missing:
        fallback_json = details_csv.with_name(details_csv.name.replace("_details.csv", ".json"))
        if not fallback_json.exists():
            fallback_json = Path("/root/autodl-tmp/results/sparse_v2_validation_strict_stage4_fix.json")
        if fallback_json.exists():
            data = json.loads(fallback_json.read_text(encoding="utf-8"))
            simple_summary = data.get("by_scene_type_overall", {}).get("simple_local", {})
            if "MAE_on_changed_edges" in simple_summary:
                summary["MAE_on_changed_edges"] = float(simple_summary["MAE_on_changed_edges"])
                fallback_note = f"MAE_on_changed_edges fallback loaded from {fallback_json}"
    if math.isnan(summary["MAE_on_changed_edges"]):
        raise ValueError("Unable to recover simple_local MAE_on_changed_edges from current stage4_fix artifacts.")
    return summary, fallback_note


def metric_direction(metric: str) -> str:
    return "lower_is_better" if metric in {"no_anchor_when_gt_changed_rate", "MAE_on_changed_edges", "parse_fail_rate_strict"} else "higher_is_better"


def relative_gain(current: float, new: float) -> float:
    if current == 0:
        return 0.0 if new == 0 else math.inf
    return (new - current) / abs(current) * 100.0


def metric_rows(
    *,
    scope: str,
    comparison_mode: str,
    alignment_status: str,
    baseline_source: str,
    ceiling_source: str,
    current_summary: dict[str, float],
    ceiling_summary: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        current = float(current_summary.get(metric, 0.0))
        ceiling = float(ceiling_summary.get(metric, 0.0))
        gain = ceiling - current
        rows.append(
            {
                "scope": scope,
                "comparison_mode": comparison_mode,
                "alignment_status": alignment_status,
                "baseline_source": baseline_source,
                "ceiling_source": ceiling_source,
                "metric_direction": metric_direction(metric),
                "metric": metric,
                "current_stage4_fix": round(current, 4),
                "relabel_oracle_ceiling": round(ceiling, 4),
                "absolute_gain": round(gain, 4),
                "relative_gain_pct": round(relative_gain(current, ceiling), 4) if math.isfinite(relative_gain(current, ceiling)) else "inf",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    relabel_examples_json = Path(args.relabel_examples_json)
    relabel_mapping_csv = Path(args.relabel_mapping_csv)
    strict_details_csv = Path(args.strict_details_csv)
    stage4b_train_parquet = Path(args.stage4b_train_parquet)
    stage4b_eval_parquet = Path(args.stage4b_eval_parquet)
    policy_md = Path(args.policy_md)
    output_csv = Path(args.output_csv)
    output_notes = Path(args.output_notes)

    mapping_df = pd.read_csv(relabel_mapping_csv)
    examples_payload = json.loads(relabel_examples_json.read_text(encoding="utf-8"))
    policy_text = policy_md.read_text(encoding="utf-8") if policy_md.exists() else ""
    strict_details_df = pd.read_csv(strict_details_csv)
    current_reference_summary, mae_fallback_note = load_current_stage4_fix_reference(strict_details_csv)

    if "strict-aligned labeling" not in policy_text:
        raise ValueError("Policy markdown does not contain the expected strict-aligned labeling policy.")

    stage4b_df = load_stage4b_rows(stage4b_train_parquet, stage4b_eval_parquet)
    stage4b_simple = stage4b_df[stage4b_df["scene_type"] == "simple_local"].copy()

    if mapping_df["sample_id"].duplicated().any():
        raise ValueError("relabel_mapping.csv contains duplicated sample_id values.")
    if not set(mapping_df["sample_id"]) == set(stage4b_simple["sample_id"]):
        missing_in_mapping = sorted(set(stage4b_simple["sample_id"]) - set(mapping_df["sample_id"]))
        missing_in_dataset = sorted(set(mapping_df["sample_id"]) - set(stage4b_simple["sample_id"]))
        raise ValueError(
            f"Stage4b/mapping sample_id mismatch. missing_in_mapping={missing_in_mapping[:5]}, missing_in_dataset={missing_in_dataset[:5]}"
        )

    merged = stage4b_simple.merge(
        mapping_df[
            [
                "sample_id",
                "old_label",
                "new_label",
                "relabel_bucket",
                "changed",
                "old_anchor_count",
                "new_anchor_count",
                "relabel_reason",
            ]
        ],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    if not (merged["response_text"].map(clean_text) == merged["new_label"].map(clean_text)).all():
        mismatch = merged[merged["response_text"].map(clean_text) != merged["new_label"].map(clean_text)]["sample_id"].tolist()
        raise ValueError(f"Stage4b dataset response_text does not match mapping new_label for sample_ids: {mismatch[:5]}")

    changed_rows = merged[merged["changed"].astype(bool)].copy()
    unchanged_rows = merged[~merged["changed"].astype(bool)].copy()
    changed_bucket_counts = changed_rows["relabel_bucket"].value_counts().to_dict()

    details_simple_ids = set(strict_details_df[strict_details_df["scene_type"] == "simple_local"]["sample_id"].astype(str))
    stage4b_ids = set(merged["sample_id"].astype(str))
    aligned_ids = details_simple_ids & stage4b_ids
    alignment_status = "aligned" if aligned_ids else "sample_id_mismatch"
    alignment_report = {
        "strict_details_simple_local_count": int(len(details_simple_ids)),
        "stage4b_simple_local_count": int(len(stage4b_ids)),
        "aligned_sample_id_overlap": int(len(aligned_ids)),
        "alignment_status": alignment_status,
        "message": (
            "Current strict details sample_id space does not align with stage4b relabel sample_id space; "
            "use held-out stage4_fix metrics only as an unpaired reference."
            if alignment_status == "sample_id_mismatch"
            else "Sample IDs align."
        ),
    }

    edge_ids = live_edge_ids()
    old_details = evaluate_label_column(merged, "old_label", edge_ids)
    new_details = evaluate_label_column(merged, "new_label", edge_ids)

    old_by_id = {item["sample_id"]: item for item in old_details}
    new_by_id = {item["sample_id"]: item for item in new_details}
    if set(old_by_id) != set(new_by_id):
        raise ValueError("old_label oracle and new_label oracle details do not align by sample_id.")

    changed_ids = set(changed_rows["sample_id"])
    unchanged_ids = set(unchanged_rows["sample_id"])

    old_summary_overall = summarize_details(old_details)
    new_summary_overall = summarize_details(new_details)
    old_summary_changed = summarize_details([old_by_id[sample_id] for sample_id in changed_ids])
    new_summary_changed = summarize_details([new_by_id[sample_id] for sample_id in changed_ids])
    old_summary_unchanged = summarize_details([old_by_id[sample_id] for sample_id in unchanged_ids])
    new_summary_unchanged = summarize_details([new_by_id[sample_id] for sample_id in unchanged_ids])

    rows: list[dict[str, Any]] = []
    rows.extend(
        metric_rows(
            scope="A.simple_local_overall_stage4b_aligned",
            comparison_mode="paired_old_label_oracle_vs_new_label_oracle",
            alignment_status="aligned",
            baseline_source="stage4b_old_label_oracle",
            ceiling_source="stage4b_new_label_oracle",
            current_summary=old_summary_overall,
            ceiling_summary=new_summary_overall,
        )
    )
    rows.extend(
        metric_rows(
            scope="B.simple_local_relabeled_subset_stage4b_aligned",
            comparison_mode="paired_old_label_oracle_vs_new_label_oracle",
            alignment_status="aligned",
            baseline_source="stage4b_old_label_oracle",
            ceiling_source="stage4b_new_label_oracle",
            current_summary=old_summary_changed,
            ceiling_summary=new_summary_changed,
        )
    )
    rows.extend(
        metric_rows(
            scope="C.simple_local_unchanged_subset_stage4b_aligned",
            comparison_mode="paired_old_label_oracle_vs_new_label_oracle",
            alignment_status="aligned",
            baseline_source="stage4b_old_label_oracle",
            ceiling_source="stage4b_new_label_oracle",
            current_summary=old_summary_unchanged,
            ceiling_summary=new_summary_unchanged,
        )
    )
    rows.extend(
        metric_rows(
            scope="R.simple_local_overall_unpaired_reference",
            comparison_mode="heldout_stage4_fix_reference_vs_stage4b_new_label_oracle",
            alignment_status=alignment_status,
            baseline_source="sparse_v2_validation_strict_stage4_fix",
            ceiling_source="stage4b_new_label_oracle",
            current_summary=current_reference_summary,
            ceiling_summary=new_summary_overall,
        )
    )
    eval_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_csv, index=False, encoding="utf-8")

    anchor_examples_count = len(examples_payload.get("examples_by_bucket", {}).get(TARGET_BUCKET, []))
    decision = "HOLD_FOR_POLICY_REVIEW"

    notes_rows = [["Scope", "Metric", "Current", "Oracle", "Gain"]]
    for scope in [
        "A.simple_local_overall_stage4b_aligned",
        "B.simple_local_relabeled_subset_stage4b_aligned",
        "C.simple_local_unchanged_subset_stage4b_aligned",
        "R.simple_local_overall_unpaired_reference",
    ]:
        subset = eval_df[eval_df["scope"] == scope]
        for metric in METRICS:
            row = subset[subset["metric"] == metric].iloc[0]
            notes_rows.append(
                [
                    scope,
                    metric,
                    str(row["current_stage4_fix"]),
                    str(row["relabel_oracle_ceiling"]),
                    str(row["absolute_gain"]),
                ]
            )

    heldout_no_anchor = float(current_reference_summary["no_anchor_when_gt_changed_rate"])
    heldout_changed_recall = float(current_reference_summary["gt_changed_edge_recall"])
    heldout_mae = float(current_reference_summary["MAE_on_changed_edges"])
    stage4b_overall_no_anchor = float(new_summary_overall["no_anchor_when_gt_changed_rate"])
    stage4b_overall_changed_recall = float(new_summary_overall["gt_changed_edge_recall"])
    stage4b_changed_subset_recall = float(new_summary_changed["gt_changed_edge_recall"])
    stage4b_changed_subset_mae = float(new_summary_changed["MAE_on_changed_edges"])

    notes = [
        "# Relabel Ceiling Evaluation",
        "",
        "## Alignment Report",
        f"- {alignment_report['message']}",
        f"- strict details simple_local sample count: `{alignment_report['strict_details_simple_local_count']}`",
        f"- stage4b simple_local sample count: `{alignment_report['stage4b_simple_local_count']}`",
        f"- aligned sample_id overlap: `{alignment_report['aligned_sample_id_overlap']}`",
        f"- MAE fallback note: `{mae_fallback_note or 'not needed'}`",
        "",
        "## How To Read The Tables",
        "- `A/B/C` rows are the sample-aligned ceiling estimate on the stage4b relabel dataset: old label as oracle baseline vs new label as oracle ceiling.",
        "- `R` rows use held-out `sparse_v2_validation_strict_stage4_fix` simple_local metrics as an unpaired reference because the sample_id spaces do not align.",
        "- Therefore, `A/B/C` are the trustworthy causal estimate for label-only improvement on the current stage4b data; `R` is only a directional comparison to current validation behavior.",
        "",
        "## Metric Table",
        markdown_table(notes_rows),
        "",
        "## Stage4b Coverage Facts",
        f"- stage4b simple_local total: `{len(merged)}`",
        f"- relabeled sample count: `{len(changed_rows)}`",
        f"- unchanged sample count: `{len(unchanged_rows)}`",
        f"- changed bucket counts: `{json.dumps(changed_bucket_counts, ensure_ascii=False)}`",
        f"- relabel_examples.json currently carries `{anchor_examples_count}` exemplar entries for `{TARGET_BUCKET}`",
        "",
        "## Required Answers",
        f"- 如果只改标签、不改模型，simple_local 的 `no_anchor_when_gt_changed_rate` 理论上最多能降到多少：",
        f"  1. 在当前 stage4b 的样本对齐 ceiling 里，old-label oracle 本来就是 `0.0`，new-label oracle 仍是 `0.0`。这说明本轮 relabel 对 `no_anchor` 主瓶颈没有直接作用，因为当前 stage4 里被改写的样本全部不是 `no_anchor` 桶。",
        f"  2. 相比之下，held-out `stage4_fix` simple_local 当前是 `{heldout_no_anchor:.4f}`。由于本轮 stage4b relabel 不包含 relief-only `no_anchor` 样本，不能把这个 held-out 指标现实地外推到 `0.0`。",
        f"- `gt_changed_edge_recall` 理论上最多能升到多少：",
        f"  1. 在 stage4b 样本对齐 ceiling 里，simple_local overall 可从 `{old_summary_overall['gt_changed_edge_recall']:.1f}` 升到 `{stage4b_overall_changed_recall:.1f}`。",
        f"  2. 在实际被改写的 `anomaly_plus_normal_side_event` 子集里，可从 `{old_summary_changed['gt_changed_edge_recall']:.1f}` 直接升到 `{stage4b_changed_subset_recall:.1f}`。",
        f"  3. held-out `stage4_fix` simple_local 当前是 `{heldout_changed_recall:.1f}`，但这与 stage4b oracle 仍是非配对参考。",
        f"- 当前 `0.2762` 是否有现实希望压到 `0.15` 以下：当前 `0.2762` 指的是 held-out `no_anchor_when_gt_changed_rate`。基于本轮 stage4b relabel 的覆盖事实，答案是 `没有充分现实依据`。因为这次改写的 100 条样本全部属于 `{TARGET_BUCKET}`，而 held-out 主失败模式是 relief-only `no_anchor`，本轮 supervision 没有直接覆盖这部分。",
        f"- 这次 relabel 是否足以支撑进入 stage4b 训练：`暂时不够`。它足以证明 `{TARGET_BUCKET}` 这一个 bucket 的 strict target 并不过强，且标签修复能把该 bucket 的 oracle recall 拉到 100；但它还不足以解决 held-out simple_local 的主失败模式。",
        f"- 如果值得进入 stage4b，最该训练的样本子集是什么：如果只是验证单一 bucket 的可学性，最该训练的是这 100 条 `{TARGET_BUCKET}` relabel 样本；但若目标是修复 held-out simple_local overall，则还必须先补齐 `relief_normal_single_edge / relief_normal_short_range / multi_edge_relief` 这三类监督。",
        f"- 如果还不值得，瓶颈是“标签修正覆盖仍不足”还是“strict 目标仍然过强”：瓶颈是 `标签修正覆盖仍不足`，不是 strict 目标过强。证据是：在已覆盖的 relabeled 子集上，new-label oracle 的 `gt_changed_edge_recall = {stage4b_changed_subset_recall:.1f}`、`MAE_on_changed_edges = {stage4b_changed_subset_mae:.4f}`，说明 strict 目标本身是可达的。",
        "",
        "## Bucket-Specific Judgment",
        f"- 由于这次 stage4 中实际出现并被改写的 gap 全部来自 `{TARGET_BUCKET}`，所以本轮 ceiling 改善几乎全部局限在这一个 bucket：overall `gt_changed_edge_recall` 的提升，本质上来自这 100 条样本从单锚点到双锚点的修正。",
        "- 其余 `relief_normal_single_edge / relief_normal_short_range / multi_edge_relief` 没有出现在当前 stage4 中，因此不能把它们没有改善误判为“relabel 无效”；正确结论应是：`当前 stage4b 还没有包含这些 gap 类型，所以本轮 ceiling 无法检验它们。`",
        "",
        "## Sanity Checks",
        f"- mapping sample_id unique: `{not mapping_df['sample_id'].duplicated().any()}`",
        f"- stage4b simple_local sample_id set matches mapping: `{set(mapping_df['sample_id']) == set(stage4b_simple['sample_id'])}`",
        f"- stage4b response_text equals mapping new_label: `{(merged['response_text'].map(clean_text) == merged['new_label'].map(clean_text)).all()}`",
        f"- old/new oracle sample_id sets align: `{set(old_by_id) == set(new_by_id)}`",
        "",
        "## Decision",
        f"- `{decision}`",
        "- 理由：当前 ceiling 证明确实存在可观的 `changed_edge_recall` 修复空间，但这次 stage4b relabel 只覆盖了 anomaly+normal side-event bucket，无法现实地推动 held-out `no_anchor_when_gt_changed_rate` 从 0.2762 压到 0.15 以下。因此进入训练前，先补 relief-only 三类监督更稳妥。",
        "",
        "## Output Files",
        f"- CSV: `{output_csv}`",
        f"- Notes: `{output_notes}`",
    ]

    output_notes.parent.mkdir(parents=True, exist_ok=True)
    output_notes.write_text("\n".join(notes), encoding="utf-8")

    if alignment_status == "sample_id_mismatch":
        print("ALIGNMENT_REPORT: current strict details sample_id space does not match stage4b relabel sample_id space; using unpaired reference rows plus aligned stage4b oracle rows.")
    print(
        "Ceiling summary: "
        f"stage4b overall changed-edge recall {old_summary_overall['gt_changed_edge_recall']:.1f} -> {new_summary_overall['gt_changed_edge_recall']:.1f}; "
        f"relabeled subset {old_summary_changed['gt_changed_edge_recall']:.1f} -> {new_summary_changed['gt_changed_edge_recall']:.1f}."
    )
    print(
        "Decision: HOLD_FOR_POLICY_REVIEW "
        f"(held-out no_anchor_when_gt_changed_rate={heldout_no_anchor:.4f} is not directly addressed by the current stage4b relabel coverage)."
    )


if __name__ == "__main__":
    main()
