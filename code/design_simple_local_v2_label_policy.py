#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


NEUTRAL_LEVELS = {"正常", "畅通"}
SUBTYPE_QUOTA = {
    "anomaly_plus_normal_side_event": 4,
    "relief_normal_single_edge": 3,
    "relief_normal_short_range": 3,
    "multi_edge_relief": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design simple_local_v2 label policy from semantic-gap audit.")
    parser.add_argument(
        "--audit-csv",
        default="/root/autodl-tmp/results/simple_local_semantic_gap_audit.csv",
    )
    parser.add_argument(
        "--dataset-root",
        default="/root/autodl-tmp/dataset_sparse_v2",
    )
    parser.add_argument(
        "--output-md",
        default="/root/autodl-tmp/results/simple_local_v2_label_policy.md",
    )
    return parser.parse_args()


def safe_json_loads(raw: str):
    if not isinstance(raw, str) or not raw:
        return []
    return json.loads(raw)


def clean_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.replace("\\n", "\n").strip()


def event_is_changed(event: dict) -> bool:
    try:
        weight = float(event.get("weight", 2.0))
    except (TypeError, ValueError):
        weight = 2.0
    return abs(weight - 2.0) > 0.05


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    head = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([head, sep, *body])


def render_label(events: list[dict], policy_name: str) -> str:
    active_changed = [event for event in events if event.get("active", True) and event_is_changed(event)]
    if not active_changed:
        anchor_count = 0
        body = ["NO_ANCHOR"]
    else:
        anchor_count = len(active_changed)
        body = [
            f"ANCHOR|ROAD={event.get('road','')}|DIR={event.get('dir','')}|RANGE={event.get('range','')}|LEVEL={event.get('level','')}"
            for event in active_changed
        ]

    think = [
        "<think>",
        "scene=simple_local",
        f"anchor_count={anchor_count}",
        f"{policy_name}=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。",
        "</think>",
    ]
    return "\n".join([*think, *body])


def load_dataset_rows(dataset_root: Path, sample_ids: list[str]) -> pd.DataFrame:
    splits = sorted({sample_id.split(":")[0] for sample_id in sample_ids})
    frames = []
    for split in splits:
        path = dataset_root / split / "data.parquet"
        table = pq.read_table(
            path,
            columns=["split", "constraint_text", "response_text", "event_records_json"],
        )
        data = table.to_pydict()
        frame = pd.DataFrame(data)
        frame["split_index"] = range(len(frame))
        frame["sample_id"] = frame["split"].astype(str) + ":" + frame["split_index"].astype(str)
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    return all_rows[all_rows["sample_id"].isin(sample_ids)].copy()


def sample_reason(subtype: str) -> str:
    return {
        "anomaly_plus_normal_side_event": "旧标签只保留异常主事件，漏掉了同样属于 changed edges 的正常/畅通 side event；v2 需要双锚定。",
        "relief_normal_single_edge": "旧标签把单边 relief 事件写成 NO_ANCHOR，但 strict changed-edge 评估把该边计入召回分母；v2 需要显式锚定。",
        "relief_normal_short_range": "旧标签省略了连续 2-edge 的 relief 范围；v2 用一个覆盖完整范围的 relief ANCHOR 对齐 strict changed-edge 目标。",
        "multi_edge_relief": "旧标签把 3+ edge 的 relief 整段省略；v2 必须显式覆盖完整范围，否则 changed-edge recall 会系统性损失。",
    }[subtype]


def main() -> None:
    args = parse_args()
    audit_csv = Path(args.audit_csv)
    dataset_root = Path(args.dataset_root)
    output_md = Path(args.output_md)

    audit_df = pd.read_csv(audit_csv)
    audit_df = audit_df[audit_df["scene_type"] == "simple_local"].copy()
    strict_fail_df = audit_df[audit_df["strict_fail"].astype(bool)].copy()

    bucket_counts = strict_fail_df["semantic_gap_subtype"].value_counts()
    total = len(strict_fail_df)
    label_gap = int((strict_fail_df["audit_bucket"] == "label_semantic_gap").sum())
    unanchored_level_counts = strict_fail_df["unanchored_changed_levels"].fillna("").str.split("|").explode()
    unanchored_level_counts = unanchored_level_counts[unanchored_level_counts != ""].value_counts()

    selected_ids: list[str] = []
    for subtype, quota in SUBTYPE_QUOTA.items():
        selected_ids.extend(strict_fail_df[strict_fail_df["semantic_gap_subtype"] == subtype]["sample_id"].head(quota).tolist())
    dataset_rows = load_dataset_rows(dataset_root, selected_ids)
    merged_examples = strict_fail_df.merge(
        dataset_rows[["sample_id", "constraint_text", "response_text", "event_records_json"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    example_blocks = []
    for subtype, quota in SUBTYPE_QUOTA.items():
        subset = merged_examples[merged_examples["semantic_gap_subtype"] == subtype].head(quota)
        for _, row in subset.iterrows():
            events = safe_json_loads(row["event_records_json"])
            new_label = render_label(events, "simple_local_v2")
            block = [
                f"### `{row['sample_id']}` | {subtype}",
                "",
                f"- 原文本：{clean_text(row['constraint_text'])}",
                "- 旧标签：",
                "```text",
                clean_text(row["response_text"]),
                "```",
                "- 新标签：",
                "```text",
                new_label,
                "```",
                f"- 修改原因：{sample_reason(subtype)}",
                "",
            ]
            example_blocks.extend(block)

    subtype_table = [["Subtype", "Count", "Share of strict fail"]]
    for subtype in [
        "anomaly_plus_normal_side_event",
        "relief_normal_single_edge",
        "relief_normal_short_range",
        "multi_edge_relief",
    ]:
        count = int(bucket_counts.get(subtype, 0))
        subtype_table.append([subtype, str(count), f"{count / total:.1%}" if total else "0.0%"])

    unanchored_level_table = [["Level", "Count"]]
    for level, count in unanchored_level_counts.items():
        unanchored_level_table.append([str(level), str(int(count))])

    policy_table = [
        ["方案", "标签规则", "与当前 strict 的关系", "优点", "风险"],
        [
            "A. strict-aligned labeling",
            "输出所有 active changed-event anchors，包含异常主事件、畅通/顺畅/恢复 side event，以及 relief-only 范围。",
            "完全对齐，不改指标。",
            "最直接消除 semantic gap，训练/评估目标一致。",
            "标签更密，模型输出会比旧版更“啰嗦”。",
        ],
        [
            "B. anchor-semantic labeling",
            "继续只输出更强异常主事件；normal/relief 可省略。",
            "必须重写 strict 指标。",
            "保持输出更稀疏，更接近“异常锚点”直觉。",
            "如果不改评估，当前 strict fail 会原样保留；论文叙事也更容易被质疑为改口径。",
        ],
    ]

    lines = [
        "# simple_local_v2 Label Policy",
        "",
        "## Audit Facts",
        f"- 已确认 `simple_local` strict fail 共 `{total}` 条，其中 `label_semantic_gap = {label_gap}/{total} = {label_gap / total:.1%}`。",
        "- 已确认 `generation_or_parse_issue = 0`，说明当前 simple_local 的主问题不是生成质量，而是标签语义与 strict changed-edge 评估目标错位。",
        markdown_table(subtype_table),
        "",
        "## Current Problem Statement",
        "- 当前 simple_local 旧标签策略实质上等于“只锚定更强异常”。这会系统性漏掉 `畅通/顺畅` relief 事件，以及 `anomaly + normal` 样本里的正常 side event。",
        "- strict 评估却把这些 active changed edges 一并计入 changed-edge recall，因此模型即使完全复现旧标签，也仍然会被判 strict fail。",
        "",
        "## Two Candidate Policies",
        markdown_table(policy_table),
        "",
        "## A. strict-aligned labeling",
        "",
        "### Core Rule",
        "- 新主线规则：`simple_local_v2` 不再定义为“只输出异常语义锚点”，而是“输出所有 active changed-event anchors”。",
        "- 也就是说，只要某个事件对应到 active changed edges，且文本给出了可定位的 `ROAD + DIR + RANGE + LEVEL`，就必须显式输出 `ANCHOR`。",
        "",
        "### Lexical Normalization",
        "- `畅通无阻`、`车流顺畅`、`通行顺畅` 统一归到 `LEVEL=畅通`。",
        "- `保持正常通行`、`交通基本正常`、`恢复正常` 统一归到 `LEVEL=正常`。",
        "- 是否输出 ANCHOR 取决于它是否对应 `active changed edges`，而不是只看词面是否“轻”。",
        "",
        "### Required Answers",
        "- “畅通 / 正常 / 顺畅 / 保持正常通行”何时必须显式输出 ANCHOR：",
        "  1. 当这类描述对应一个显式、当前生效、可定位到 `ROAD + DIR + RANGE` 的局部事件时，如果它映射到至少一条 changed edge，就必须输出 `ANCHOR`。",
        "  2. 若它只是默认背景状态、泛化描述，或不对应任何 changed edge，则不需要输出。",
        "  3. 对于 simple_local 当前已知 gap，真正需要补锚的主要是 `畅通/顺畅` relief 事件；`正常/保持正常通行` 也遵循同一条 edge-based 规则，而不是因为词面较轻就天然省略。",
        "- anomaly_plus_normal 双事件样本是否要求同时锚定异常主事件和正常 side event：`是`。只要 side event 也是 active changed event，就必须和异常主事件一起输出。",
        "- multi-edge relief 是否必须覆盖完整范围：`是`。如果文本给的是连续区间，就必须用一个连续范围锚点覆盖完整区间；如果文本本身是非连续片段，则拆成多个锚点，但它们的并集必须覆盖全部 changed edges。",
        "",
        "### Labeling Rules in Practice",
        "- relief-only 单边样本：旧版 `NO_ANCHOR` 改为 1 个 `ANCHOR`。",
        "- relief-only 短范围样本：旧版 `NO_ANCHOR` 改为 1 个覆盖完整连续范围的 `ANCHOR`。",
        "- relief-only 多边样本：旧版 `NO_ANCHOR` 改为 1 个全范围 `ANCHOR`，必要时拆段但不得漏边。",
        "- anomaly_plus_normal：至少 2 个 `ANCHOR`，异常主事件和 normal/relief side event 都要保留。",
        "",
        "## B. anchor-semantic labeling",
        "",
        "### Core Rule",
        "- 保持旧哲学：只输出更强异常主事件，normal / relief side event 可省略，relief-only 样本允许 `NO_ANCHOR`。",
        "- 这套方案能保住“锚点=异常摘要”的简洁性，但它不再和当前 strict changed-edge 指标同义。",
        "",
        "### If We Keep This Scheme, How strict Must Change",
        "- `no_anchor_when_gt_changed` 必须改写为 `no_anchor_when_anchorworthy_gt_changed`：只有当样本中存在异常主事件时，`NO_ANCHOR` 才算失败。",
        "- `gt_changed_edge_recall` 必须改成 `anchorworthy_edge_recall`：分母只保留异常主事件对应的边，不再把 relief-only 边、normal side event 边算进召回分母。",
        "- `target_recall` 建议同步解释为“异常主事件 target recall”，避免和 normal side event 的漏锚混在一起。",
        "- 额外新增一个次级指标，例如 `side_event_consistency` 或 `relief_event_capture_rate`，用于单独分析 normal/relief 是否被捕获，但不再作为主 strict fail 条件。",
        "- 如果采用本方案，论文里必须明确承认：simple_local 的主任务定义已从“changed-edge coverage”改成“异常主事件提取”，否则容易被认为是为了分数而改口径。",
        "",
        "## Recommended Mainline",
        "- 推荐主线：`A. strict-aligned labeling`。",
        "- 为什么它最适合当前 simple_local 指标修复：因为审计已经证明 `51/51` strict fail 都是标签缺口，而不是模型生成问题。最短路径不是调模型，而是把 supervision 直接改成与 strict changed-edge 目标同义。",
        "- 为什么它更适合后续论文与答辩表述：",
        "  1. 叙事更干净。我们不是为了提高分数去改指标，而是发现训练标签和评测定义错位，随后修正标签使两者一致。",
        "  2. 可比性更强。修复前后仍然在同一个 strict 指标下比较，不需要额外解释“为什么换了口径”。",
        "  3. 更容易做 relabel ceiling eval。只要重新标注 simple_local_v2，就能直接测“仅修标签、不改模型结构”带来的上限提升。",
        "",
        "## Recommendation for Next Step",
        "- 当前 simple_local 的主线动作应是：`改标签`，不是先改评估。",
        "- 只有在产品定义明确坚持“锚点只服务异常摘要”时，才应该转向 `anchor-semantic labeling + metric rewrite`。",
        "- 基于当前审计结果，`relabel ceiling eval` 值得立刻排到高优先级，因为它能直接验证 simple_local strict 分数里有多少上限来自标签修复。",
        "",
        "## Unanchored Changed-Edge Levels Observed In Audit",
        markdown_table(unanchored_level_table),
        "",
        "## Before / After Examples",
        "- 下面示例全部按推荐主线 `strict-aligned labeling` 给出新标签。",
        "",
        *example_blocks,
        "## Final Conclusion",
        "- 当前 simple_local 应该优先 `改标签`；只有在坚持“只锚定更强异常”的产品哲学下，才应该同步 `改评估`。",
        "- 后续值得进入 `relabel ceiling eval`，而且优先级高，因为已确认 strict fail 的主瓶颈不是生成，而是 supervision 口径。",
    ]

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")

    print("主线结论：改标签（strict-aligned labeling），不是先改评估。")
    print("后续建议：值得进入 relabel ceiling eval，优先级高。")
    print(f"output_md={output_md}")


if __name__ == "__main__":
    main()
