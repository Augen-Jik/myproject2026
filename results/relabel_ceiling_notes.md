# Relabel Ceiling Evaluation

## Alignment Report
- Current strict details sample_id space does not align with stage4b relabel sample_id space; use held-out stage4_fix metrics only as an unpaired reference.
- strict details simple_local sample count: `139`
- stage4b simple_local sample count: `508`
- aligned sample_id overlap: `0`
- MAE fallback note: `MAE_on_changed_edges fallback loaded from /root/autodl-tmp/results/sparse_v2_validation_strict_stage4_fix.json`

## How To Read The Tables
- `A/B/C` rows are the sample-aligned ceiling estimate on the stage4b relabel dataset: old label as oracle baseline vs new label as oracle ceiling.
- `R` rows use held-out `sparse_v2_validation_strict_stage4_fix` simple_local metrics as an unpaired reference because the sample_id spaces do not align.
- Therefore, `A/B/C` are the trustworthy causal estimate for label-only improvement on the current stage4b data; `R` is only a directional comparison to current validation behavior.

## Metric Table
| Scope | Metric | Current | Oracle | Gain |
| --- | --- | --- | --- | --- |
| A.simple_local_overall_stage4b_aligned | no_anchor_when_gt_changed_rate | 0.0 | 0.0 | 0.0 |
| A.simple_local_overall_stage4b_aligned | target_recall | 100.0 | 100.0 | 0.0 |
| A.simple_local_overall_stage4b_aligned | gt_changed_edge_recall | 85.6 | 100.0 | 14.4 |
| A.simple_local_overall_stage4b_aligned | MAE_on_changed_edges | 0.1154 | 0.0 | -0.1154 |
| A.simple_local_overall_stage4b_aligned | parse_fail_rate_strict | 0.0 | 0.0 | 0.0 |
| B.simple_local_relabeled_subset_stage4b_aligned | no_anchor_when_gt_changed_rate | 0.0 | 0.0 | 0.0 |
| B.simple_local_relabeled_subset_stage4b_aligned | target_recall | 100.0 | 100.0 | 0.0 |
| B.simple_local_relabeled_subset_stage4b_aligned | gt_changed_edge_recall | 55.8 | 100.0 | 44.2 |
| B.simple_local_relabeled_subset_stage4b_aligned | MAE_on_changed_edges | 0.354 | 0.0 | -0.354 |
| B.simple_local_relabeled_subset_stage4b_aligned | parse_fail_rate_strict | 0.0 | 0.0 | 0.0 |
| C.simple_local_unchanged_subset_stage4b_aligned | no_anchor_when_gt_changed_rate | 0.0 | 0.0 | 0.0 |
| C.simple_local_unchanged_subset_stage4b_aligned | target_recall | 100.0 | 100.0 | 0.0 |
| C.simple_local_unchanged_subset_stage4b_aligned | gt_changed_edge_recall | 100.0 | 100.0 | 0.0 |
| C.simple_local_unchanged_subset_stage4b_aligned | MAE_on_changed_edges | 0.0 | 0.0 | 0.0 |
| C.simple_local_unchanged_subset_stage4b_aligned | parse_fail_rate_strict | 0.0 | 0.0 | 0.0 |
| R.simple_local_overall_unpaired_reference | no_anchor_when_gt_changed_rate | 0.2762 | 0.0 | -0.2762 |
| R.simple_local_overall_unpaired_reference | target_recall | 100.0 | 100.0 | 0.0 |
| R.simple_local_overall_unpaired_reference | gt_changed_edge_recall | 63.7 | 100.0 | 36.3 |
| R.simple_local_overall_unpaired_reference | MAE_on_changed_edges | 0.2901 | 0.0 | -0.2901 |
| R.simple_local_overall_unpaired_reference | parse_fail_rate_strict | 0.2086 | 0.0 | -0.2086 |

## Stage4b Coverage Facts
- stage4b simple_local total: `508`
- relabeled sample count: `100`
- unchanged sample count: `408`
- changed bucket counts: `{"anomaly_plus_normal_side_event": 100}`
- relabel_examples.json currently carries `5` exemplar entries for `anomaly_plus_normal_side_event`

## Required Answers
- 如果只改标签、不改模型，simple_local 的 `no_anchor_when_gt_changed_rate` 理论上最多能降到多少：
  1. 在当前 stage4b 的样本对齐 ceiling 里，old-label oracle 本来就是 `0.0`，new-label oracle 仍是 `0.0`。这说明本轮 relabel 对 `no_anchor` 主瓶颈没有直接作用，因为当前 stage4 里被改写的样本全部不是 `no_anchor` 桶。
  2. 相比之下，held-out `stage4_fix` simple_local 当前是 `0.2762`。由于本轮 stage4b relabel 不包含 relief-only `no_anchor` 样本，不能把这个 held-out 指标现实地外推到 `0.0`。
- `gt_changed_edge_recall` 理论上最多能升到多少：
  1. 在 stage4b 样本对齐 ceiling 里，simple_local overall 可从 `85.6` 升到 `100.0`。
  2. 在实际被改写的 `anomaly_plus_normal_side_event` 子集里，可从 `55.8` 直接升到 `100.0`。
  3. held-out `stage4_fix` simple_local 当前是 `63.7`，但这与 stage4b oracle 仍是非配对参考。
- 当前 `0.2762` 是否有现实希望压到 `0.15` 以下：当前 `0.2762` 指的是 held-out `no_anchor_when_gt_changed_rate`。基于本轮 stage4b relabel 的覆盖事实，答案是 `没有充分现实依据`。因为这次改写的 100 条样本全部属于 `anomaly_plus_normal_side_event`，而 held-out 主失败模式是 relief-only `no_anchor`，本轮 supervision 没有直接覆盖这部分。
- 这次 relabel 是否足以支撑进入 stage4b 训练：`暂时不够`。它足以证明 `anomaly_plus_normal_side_event` 这一个 bucket 的 strict target 并不过强，且标签修复能把该 bucket 的 oracle recall 拉到 100；但它还不足以解决 held-out simple_local 的主失败模式。
- 如果值得进入 stage4b，最该训练的样本子集是什么：如果只是验证单一 bucket 的可学性，最该训练的是这 100 条 `anomaly_plus_normal_side_event` relabel 样本；但若目标是修复 held-out simple_local overall，则还必须先补齐 `relief_normal_single_edge / relief_normal_short_range / multi_edge_relief` 这三类监督。
- 如果还不值得，瓶颈是“标签修正覆盖仍不足”还是“strict 目标仍然过强”：瓶颈是 `标签修正覆盖仍不足`，不是 strict 目标过强。证据是：在已覆盖的 relabeled 子集上，new-label oracle 的 `gt_changed_edge_recall = 100.0`、`MAE_on_changed_edges = 0.0000`，说明 strict 目标本身是可达的。

## Bucket-Specific Judgment
- 由于这次 stage4 中实际出现并被改写的 gap 全部来自 `anomaly_plus_normal_side_event`，所以本轮 ceiling 改善几乎全部局限在这一个 bucket：overall `gt_changed_edge_recall` 的提升，本质上来自这 100 条样本从单锚点到双锚点的修正。
- 其余 `relief_normal_single_edge / relief_normal_short_range / multi_edge_relief` 没有出现在当前 stage4 中，因此不能把它们没有改善误判为“relabel 无效”；正确结论应是：`当前 stage4b 还没有包含这些 gap 类型，所以本轮 ceiling 无法检验它们。`

## Sanity Checks
- mapping sample_id unique: `True`
- stage4b simple_local sample_id set matches mapping: `True`
- stage4b response_text equals mapping new_label: `True`
- old/new oracle sample_id sets align: `True`

## Decision
- `HOLD_FOR_POLICY_REVIEW`
- 理由：当前 ceiling 证明确实存在可观的 `changed_edge_recall` 修复空间，但这次 stage4b relabel 只覆盖了 anomaly+normal side-event bucket，无法现实地推动 held-out `no_anchor_when_gt_changed_rate` 从 0.2762 压到 0.15 以下。因此进入训练前，先补 relief-only 三类监督更稳妥。

## Output Files
- CSV: `/root/autodl-tmp/results/relabel_ceiling_eval.csv`
- Notes: `/root/autodl-tmp/results/relabel_ceiling_notes.md`