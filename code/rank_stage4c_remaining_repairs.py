#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


RECOMMEND_DO_LAST = "DO_ONE_LAST_MICRO_PATCH"
RECOMMEND_FREEZE = "FREEZE_STAGE4_FIX_AND_STOP_PATCHING"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank stage4c remaining repairs into high-value, risky, and low-ROI buckets.")
    parser.add_argument(
        "--hard-cases-json",
        default="/root/autodl-tmp/results/stage4c_remaining_hard_cases.json",
    )
    parser.add_argument(
        "--postmortem-buckets-csv",
        default="/root/autodl-tmp/results/stage4c_postmortem_buckets.csv",
    )
    parser.add_argument(
        "--policy-md",
        default="/root/autodl-tmp/results/simple_local_v2_label_policy.md",
    )
    parser.add_argument(
        "--output-md",
        default="/root/autodl-tmp/results/stage4c_repair_priority.md",
    )
    parser.add_argument(
        "--output-csv",
        default="/root/autodl-tmp/results/stage4c_repair_priority.csv",
    )
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        if pd.isna(value):
            return 0.0
    except TypeError:
        pass
    return float(value)


def load_simple_local_changed_sample_count(buckets_df: pd.DataFrame, counts: dict[str, Any]) -> int:
    row = buckets_df[buckets_df["bucket_name"] == "val_normal:scene_type/simple_local"]
    if row.empty:
        raise KeyError("Missing val_normal:scene_type/simple_local in stage4c_postmortem_buckets.csv")
    no_anchor_rate = as_float(row.iloc[0]["post_no_anchor_when_gt_changed_rate"])
    remaining_no_anchor = int(counts["remaining_no_anchor_failures"])
    if no_anchor_rate <= 0:
        raise ValueError("post_no_anchor_when_gt_changed_rate must be > 0 to infer changed-sample denominator")
    return int(round(remaining_no_anchor / no_anchor_rate))


def validate_policy(policy_text: str) -> dict[str, bool]:
    return {
        "main_gap_mentions_changtong_shunchang": "真正需要补锚的主要是 `畅通/顺畅` relief 事件" in policy_text,
        "multi_edge_must_cover_full_range": "multi-edge relief 是否必须覆盖完整范围：`是`" in policy_text,
        "normal_and_relief_follow_same_edge_rule": "正常/保持正常通行" in policy_text and "edge-based" in policy_text,
    }


def build_priority_rows(
    *,
    hard_cases: dict[str, Any],
    changed_sample_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    single_df = pd.DataFrame(hard_cases["still_failed_single_edge_cases"])
    range_df = pd.DataFrame(hard_cases["range_boundary_reference_cases"])
    false_positive_df = pd.DataFrame(hard_cases["new_false_positive_normal_only_anchors"])
    counts = hard_cases["counts"]
    remaining_failures = int(counts["remaining_simple_local_failures"])

    def add_row(
        *,
        rank: int,
        bucket_name: str,
        repair_class: str,
        count: int,
        source_pool: str,
        failure_shape: str,
        patchable_with_micro_patch: bool,
        side_effect_risk: str,
        reason: str,
        stop_reason: str,
        phrases: str = "",
        structures: str = "",
        edge_profile: str = "",
        guardrail: str = "",
        missing_edge_units: float = 0.0,
    ) -> dict[str, Any]:
        strict_fail_pp = 100.0 * count / max(changed_sample_count, 1)
        share_of_remaining = 100.0 * count / max(remaining_failures, 1)
        return {
            "priority_rank": rank,
            "bucket_name": bucket_name,
            "repair_class": repair_class,
            "count": int(count),
            "source_pool": source_pool,
            "failure_shape": failure_shape,
            "edge_profile": edge_profile,
            "phrases": phrases,
            "structures": structures,
            "share_of_remaining_failures_pct": round(share_of_remaining, 2),
            "strict_fail_case_pp_upper_bound_on_simple_local": round(strict_fail_pp, 2),
            "missing_edge_units_upper_bound": round(float(missing_edge_units), 2),
            "patchable_with_micro_patch": patchable_with_micro_patch,
            "side_effect_risk": side_effect_risk,
            "guardrail": guardrail,
            "reason": reason,
            "stop_reason": stop_reason,
        }

    high_value_single = single_df[single_df["structure_tag"] != "planner_long"].copy()
    high_value_count = int(len(high_value_single))
    high_value_phrase_counts = (
        high_value_single["phrase_group"].value_counts().sort_index().to_dict()
        if not high_value_single.empty else {}
    )
    high_value_structure_counts = (
        high_value_single["structure_tag"].value_counts().sort_index().to_dict()
        if not high_value_single.empty else {}
    )

    planner_tail = single_df[single_df["structure_tag"] == "planner_long"].copy()

    risky_range_2edge = range_df[range_df["gt_changed_edge_count"] == 2].copy()
    low_roi_range_3plus = range_df[range_df["gt_changed_edge_count"] >= 3].copy()

    short_range_no_anchor_count = int(counts["remaining_no_anchor_failures"]) - int(len(single_df)) - 2
    multi_edge_no_anchor_count = 2
    if short_range_no_anchor_count < 0:
        short_range_no_anchor_count = 0

    rows = [
        add_row(
            rank=1,
            bucket_name="single_edge_changed_relief__short_plain_or_prefixed",
            repair_class="high_value_patch",
            count=high_value_count,
            source_pool="still_failed_single_edge_cases",
            failure_shape="single-edge relief still outputs NO_ANCHOR",
            edge_profile="single_edge",
            phrases="; ".join(f"{key}={value}" for key, value in high_value_phrase_counts.items()),
            structures="; ".join(f"{key}={value}" for key, value in high_value_structure_counts.items()),
            patchable_with_micro_patch=True,
            side_effect_risk="medium",
            guardrail="If patching, freeze all LEVEL=正常 positives and add explicit hard negatives for `交通基本正常` / `保持正常通行`.",
            reason="This is the only large, clean, concentrated bucket: all failures are single-edge, same semantic target (`LEVEL=畅通`), no mixed anomaly context, and they still fail by simple non-firing rather than range reasoning.",
            stop_reason="Stop immediately if this bucket does not move after one micro patch, because stage4c already proved broader patch SFT has sharply diminishing returns.",
            missing_edge_units=float(high_value_count),
        ),
        add_row(
            rank=2,
            bucket_name="anomaly_plus_normal_side_event__2edge_range_boundary",
            repair_class="risky_patch",
            count=int(len(risky_range_2edge)),
            source_pool="range_boundary_reference_cases",
            failure_shape="anchor appears but misses one side-event edge/range",
            edge_profile="short_range",
            phrases="; ".join(f"{key}={value}" for key, value in risky_range_2edge['phrase_group'].value_counts().sort_index().to_dict().items()),
            structures="; ".join(f"{key}={value}" for key, value in risky_range_2edge['structure_tag'].value_counts().sort_index().to_dict().items()),
            patchable_with_micro_patch=False,
            side_effect_risk="high",
            guardrail="Would need dual-anchor supervision plus strong boundary guards; not safe for a tiny patch.",
            reason="These failures are not anchor-on/off anymore; they are boundary-completion misses inside mixed abnormal+relief samples, so a tiny patch could easily over-expand ranges or duplicate anchors.",
            stop_reason="Do not spend the last micro patch here unless the goal changes from safe closure to aggressive last-minute score chasing.",
            missing_edge_units=float(risky_range_2edge["gt_changed_edge_count"].mul(1 - risky_range_2edge["post_gt_changed_edge_recall"] / 100).sum()),
        ),
        add_row(
            rank=3,
            bucket_name="relief_normal_short_range__no_anchor",
            repair_class="low_roi_patch",
            count=short_range_no_anchor_count,
            source_pool="inferred_from_postmortem_counts",
            failure_shape="short-range relief still outputs NO_ANCHOR",
            edge_profile="short_range",
            phrases="mostly 畅通无阻 / 顺畅",
            structures="not preserved in hard_cases json",
            patchable_with_micro_patch=False,
            side_effect_risk="high",
            guardrail="Needs full contiguous range coverage, not just anchor firing.",
            reason="Although the count is not tiny, stage4c moved `0/7` on this bucket after a much larger patch, which makes another micro patch low-confidence and range-boundary sensitive.",
            stop_reason="Directly stop patching this bucket in the last round.",
            missing_edge_units=2.0 * short_range_no_anchor_count,
        ),
        add_row(
            rank=4,
            bucket_name="anomaly_plus_normal_side_event__3plus_edge_range_boundary",
            repair_class="low_roi_patch",
            count=int(len(low_roi_range_3plus)),
            source_pool="range_boundary_reference_cases",
            failure_shape="mixed anomaly+normal with 3+ changed edges, anchor present but incomplete",
            edge_profile="multi_edge",
            phrases="; ".join(f"{key}={value}" for key, value in low_roi_range_3plus['phrase_group'].value_counts().sort_index().to_dict().items()),
            structures="; ".join(f"{key}={value}" for key, value in low_roi_range_3plus['structure_tag'].value_counts().sort_index().to_dict().items()),
            patchable_with_micro_patch=False,
            side_effect_risk="very_high",
            guardrail="Would require broader structured supervision, not a final tiny patch.",
            reason="This is the costliest tail: more edges, more mixed context, and boundary ambiguity dominates. A tiny patch is unlikely to generalize safely here.",
            stop_reason="Stop patching this bucket in the final round.",
            missing_edge_units=float(low_roi_range_3plus["gt_changed_edge_count"].mul(1 - low_roi_range_3plus["post_gt_changed_edge_recall"] / 100).sum()),
        ),
        add_row(
            rank=5,
            bucket_name="single_edge_changed_relief__planner_long_tail",
            repair_class="low_roi_patch",
            count=int(len(planner_tail)),
            source_pool="still_failed_single_edge_cases",
            failure_shape="single-edge relief NO_ANCHOR in planner_long wrapper",
            edge_profile="single_edge",
            phrases="; ".join(f"{key}={value}" for key, value in planner_tail['phrase_group'].value_counts().sort_index().to_dict().items()),
            structures="planner_long",
            patchable_with_micro_patch=False,
            side_effect_risk="medium",
            guardrail="Handle only as an evaluation note, not as a dedicated patch target.",
            reason="This tail is too small to justify a dedicated micro patch once short/prefixed single-edge is isolated.",
            stop_reason="Do not dedicate the last patch to this single long wrapper sample.",
            missing_edge_units=float(len(planner_tail)),
        ),
        add_row(
            rank=6,
            bucket_name="multi_edge_relief__no_anchor",
            repair_class="low_roi_patch",
            count=multi_edge_no_anchor_count,
            source_pool="inferred_from_postmortem_counts",
            failure_shape="multi-edge relief still outputs NO_ANCHOR",
            edge_profile="multi_edge",
            phrases="mostly 畅通无阻 / 顺畅",
            structures="not preserved in hard_cases json",
            patchable_with_micro_patch=False,
            side_effect_risk="very_high",
            guardrail="Requires range-complete coverage and is not a last-mile lexical fix.",
            reason="Count is tiny and the supervision cost is structurally high; not worth the final patch budget.",
            stop_reason="Directly stop patching this bucket.",
            missing_edge_units=3.0 * multi_edge_no_anchor_count,
        ),
    ]

    collateral = {
        "false_positive_normal_only_anchor_count": int(len(false_positive_df)),
        "false_positive_phrase_counts": false_positive_df["phrase_group"].value_counts().sort_index().to_dict(),
        "false_positive_structure_counts": false_positive_df["structure_tag"].value_counts().sort_index().to_dict(),
        "high_value_bucket_count": high_value_count,
    }
    return pd.DataFrame(rows), collateral


def choose_recommendation(priority_df: pd.DataFrame, collateral: dict[str, Any], policy_flags: dict[str, bool]) -> tuple[str, str]:
    top = priority_df.iloc[0]
    top_count = int(top["count"])
    top_pp = as_float(top["strict_fail_case_pp_upper_bound_on_simple_local"])
    false_positive_count = int(collateral["false_positive_normal_only_anchor_count"])
    policy_supports_relief_focus = bool(policy_flags["main_gap_mentions_changtong_shunchang"])

    if (
        top["repair_class"] == "high_value_patch"
        and top_count >= 12
        and top_pp >= 10.0
        and top_count > false_positive_count
        and policy_supports_relief_focus
    ):
        rationale = (
            "There is exactly one concentrated bucket still worth a final micro patch: "
            "single-edge `LEVEL=畅通` no-anchor failures in short/plain or prefixed wrappers. "
            "Everything else is already in range-boundary or low-ROI territory."
        )
        return RECOMMEND_DO_LAST, rationale

    rationale = (
        "No remaining bucket is both large enough and clean enough to justify another safe micro patch; "
        "the remaining errors are already dominated by range-boundary or long-tail cases."
    )
    return RECOMMEND_FREEZE, rationale


def build_markdown(
    *,
    priority_df: pd.DataFrame,
    hard_cases: dict[str, Any],
    changed_sample_count: int,
    recommendation: str,
    recommendation_rationale: str,
    policy_flags: dict[str, bool],
    collateral: dict[str, Any],
) -> str:
    top = priority_df.iloc[0]
    stop_buckets = priority_df[priority_df["repair_class"] == "low_roi_patch"]["bucket_name"].tolist()
    risky_buckets = priority_df[priority_df["repair_class"] == "risky_patch"]["bucket_name"].tolist()
    counts = hard_cases["counts"]

    lines = [
        "# stage4c Remaining Repair Priority",
        "",
        "## Final Recommendation",
        f"- `{recommendation}`",
        f"- Rationale: {recommendation_rationale}",
        "",
        "## What To Fix If Only One Last Micro Patch Is Allowed",
        f"- Best bucket: `{top['bucket_name']}`",
        f"- Why this one: it covers `{int(top['count'])}` cases, or `{top['share_of_remaining_failures_pct']:.2f}%` of all remaining hard cases, with an upper bound of `{top['strict_fail_case_pp_upper_bound_on_simple_local']:.2f}` percentage points on held-out `simple_local` strict-fail cases if fully fixed.",
        f"- Scope discipline: keep it strictly on `single-edge + LEVEL=畅通 + phrase in {{畅通无阻, 车流顺畅/顺畅}} + structure in {{short_plain, prefixed_short}}`, and add explicit hard negatives for `交通基本正常` / `保持正常通行`.",
        "",
        "## Buckets That Should Stop Now",
        f"- Low ROI buckets: {', '.join(f'`{name}`' for name in stop_buckets)}.",
        f"- Risky buckets that are not suitable for a tiny last patch: {', '.join(f'`{name}`' for name in risky_buckets)}.",
        "",
        "## Paper-Closure Readout",
        f"- Current state: remaining hard cases=`{counts['remaining_simple_local_failures']}`, remaining single-edge=`{counts['remaining_single_edge_failures']}`, remaining no-anchor=`{counts['remaining_no_anchor_failures']}`, remaining anchor-but-range-incomplete=`{counts['remaining_anchor_but_incomplete_range']}`.",
        f"- Upstream baselines are already stable enough for paper closure on everything except the one concentrated single-edge relief bucket. If that bucket does not move immediately after one micro patch, the next action should be to freeze and write up.",
        "",
        "## Why The Ranking Looks Like This",
        f"- Policy support for relief-focused last patch: `main_gap_mentions_changtong_shunchang={policy_flags['main_gap_mentions_changtong_shunchang']}`, `multi_edge_must_cover_full_range={policy_flags['multi_edge_must_cover_full_range']}`.",
        f"- Collateral risk already visible: normal-only false-positive anchors=`{collateral['false_positive_normal_only_anchor_count']}` with phrase mix `{collateral['false_positive_phrase_counts']}`.",
        f"- changed-edge denominator inferred from postmortem buckets: `{changed_sample_count}` held-out `val_normal/simple_local` changed samples.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    hard_cases = read_json(args.hard_cases_json)
    buckets_df = pd.read_csv(args.postmortem_buckets_csv)
    policy_text = Path(args.policy_md).read_text(encoding="utf-8")

    policy_flags = validate_policy(policy_text)
    changed_sample_count = load_simple_local_changed_sample_count(buckets_df, hard_cases["counts"])
    priority_df, collateral = build_priority_rows(
        hard_cases=hard_cases,
        changed_sample_count=changed_sample_count,
    )
    recommendation, recommendation_rationale = choose_recommendation(priority_df, collateral, policy_flags)

    priority_df["final_recommendation"] = recommendation
    priority_df["generated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    priority_df.to_csv(args.output_csv, index=False, encoding="utf-8")

    md_text = build_markdown(
        priority_df=priority_df,
        hard_cases=hard_cases,
        changed_sample_count=changed_sample_count,
        recommendation=recommendation,
        recommendation_rationale=recommendation_rationale,
        policy_flags=policy_flags,
        collateral=collateral,
    )
    Path(args.output_md).write_text(md_text, encoding="utf-8")

    print(recommendation)


if __name__ == "__main__":
    main()
