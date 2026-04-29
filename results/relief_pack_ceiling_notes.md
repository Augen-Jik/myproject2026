# Relief Pack Ceiling Evaluation

## Read Method
- 所有 parquet 均显式使用 `pyarrow.parquet.read_table(...)` 读取，并只在读入后转成 Python dict / pandas DataFrame；没有依赖 `pandas.read_parquet(...)`。

## Evaluation Setup
- `A.relief_pack_overall`：在 relief pack 自身上做 oracle ceiling sanity check，确认标签与 strict 目标是否一致。
- `B.<bucket>`：分别检查每个 relief bucket 在 pack 自身上的 oracle label adequacy。
- `C.heldout_alignment_reference`：不把 pack 和 held-out 伪装成 sample-level 配对；主证据明确限定为 `failure-mode alignment + bucket coverage + oracle label adequacy`。
- 另外，为了验证 strict 目标本身是否可达，我们只在 held-out 自身内部做 policy-A oracle rewrite：这说明标签上限，不等于训练后真实收益。

## Metric Table
| Scope | Bucket | Metric | Current | Oracle | Gain |
| --- | --- | --- | --- | --- | --- |
| A.relief_pack_overall | overall | no_anchor_when_gt_changed_rate | 0.0 | 0.0 | 0.0 |
| A.relief_pack_overall | overall | target_recall | 0.0 | 0.0 | 0.0 |
| A.relief_pack_overall | overall | gt_changed_edge_recall | 100.0 | 100.0 | 0.0 |
| A.relief_pack_overall | overall | MAE_on_changed_edges | 0.0 | 0.0 | 0.0 |
| A.relief_pack_overall | overall | parse_fail_rate_strict | 0.0 | 0.0 | 0.0 |
| C.heldout_alignment_reference | overall | no_anchor_when_gt_changed_rate | 0.2762 | 0.0 | -0.2762 |
| C.heldout_alignment_reference | overall | target_recall | 100.0 | 100.0 | 0.0 |
| C.heldout_alignment_reference | overall | gt_changed_edge_recall | 63.7 | 87.1 | 23.4 |
| C.heldout_alignment_reference | overall | MAE_on_changed_edges | 0.2901 | 0.1029 | -0.1872 |
| C.heldout_alignment_reference | overall | parse_fail_rate_strict | 0.2086 | 0.0 | -0.2086 |

## Held-out Bucket Table
| Bucket | Metric | Current | Oracle | Gain |
| --- | --- | --- | --- | --- |
| relief_normal_single_edge | no_anchor_when_gt_changed_rate | 1.0 | 0.0 | -1.0 |
| relief_normal_single_edge | target_recall | 0.0 | 0.0 | 0.0 |
| relief_normal_single_edge | gt_changed_edge_recall | 0.0 | 100.0 | 100.0 |
| relief_normal_single_edge | MAE_on_changed_edges | 0.8 | 0.0 | -0.8 |
| relief_normal_single_edge | parse_fail_rate_strict | 1.0 | 0.0 | -1.0 |
| relief_normal_short_range | no_anchor_when_gt_changed_rate | 1.0 | 0.0 | -1.0 |
| relief_normal_short_range | target_recall | 0.0 | 0.0 | 0.0 |
| relief_normal_short_range | gt_changed_edge_recall | 0.0 | 100.0 | 100.0 |
| relief_normal_short_range | MAE_on_changed_edges | 0.8 | 0.0 | -0.8 |
| relief_normal_short_range | parse_fail_rate_strict | 1.0 | 0.0 | -1.0 |
| multi_edge_relief | no_anchor_when_gt_changed_rate | 1.0 | 0.0 | -1.0 |
| multi_edge_relief | target_recall | 0.0 | 0.0 | 0.0 |
| multi_edge_relief | gt_changed_edge_recall | 0.0 | 100.0 | 100.0 |
| multi_edge_relief | MAE_on_changed_edges | 0.8 | 0.0 | -0.8 |
| multi_edge_relief | parse_fail_rate_strict | 1.0 | 0.0 | -1.0 |

## Coverage Table
| Bucket | Held-out Fail | Pack Train | Pack Eval | Direct(total) | Exact Range(total) |
| --- | --- | --- | --- | --- | --- |
| relief_normal_single_edge | 20 | 16 | 8 | 8 | 12 |
| relief_normal_short_range | 7 | 16 | 8 | 7 | 7 |
| multi_edge_relief | 2 | 16 | 8 | 2 | 2 |

## Sanity Checks
- relief pack bucket set == target buckets: `True`
- relief pack oracle `gt_changed_edge_recall`: `100.0`
- relief pack oracle `MAE_on_changed_edges`: `0.0000`
- relief pack oracle `parse_fail_rate_strict`: `0.0000`
- held-out strict details simple_local count: `139`
- failure summary simple_local count: `139`
- failure summary no_anchor count: `29`
- strict-details current no_anchor changed count: `29`
- 无额外 warning。

## Required Answers
- 这三类新 supervision 是否正中当前 held-out 主瓶颈：`是`。当前 held-out `simple_local` 的剩余主失败模式就是这三类 relief bucket，对应 `29/29` 个 `no_anchor_when_gt_changed` 失败样本。
- 若将其加入后续 stage4b，`simple_local no_anchor_when_gt_changed_rate` 是否“有现实依据”向 `0.15` 以下推进：`有`。
  现实依据来自三层证据而不是伪造配对：
  1. `failure-mode alignment`：当前 `29` 个 no-anchor changed 失败全部落在这三类 bucket。
  2. `oracle label adequacy`：这三类 held-out 失败样本在 policy-A oracle 下可从 `no_anchor_when_gt_changed_rate 1.0000 -> 0.0000`，`gt_changed_edge_recall 0.0 -> 100.0`，`MAE_on_changed_edges 0.8000 -> 0.0000`。
  3. `bucket coverage`：pack 全量对 held-out 失败样本有 `17/29` direct overlap、`21/29` exact ROAD+DIR+RANGE overlap，其中 `short_range` 与 `multi_edge` 已做到 exact coverage 全覆盖。
- 为什么说 `0.15` 以下有现实依据：当前 changed 样本总数是 `105`，当前 no-anchor changed 样本是 `29`。要把该指标压到 `0.15` 以下，至少只需修掉 `14` 个失败样本；而当前被 relief pack 对准的失败样本有 `29` 个，量级上明显足够。
- 哪个 bucket 对压 no_anchor 最关键：`relief_normal_single_edge`。因为它在 held-out 当前剩余瓶颈中占比最高，为 `20/29 = 69.0%`。
- 哪个 bucket 即使补了也仍然风险最高：`multi_edge_relief`。原因不是数量最大，而是连续 3+ edge 的范围边界最容易截短或偏移，而且该 bucket 的历史唯一语义最少。
- `multi_edge_relief` 的边界歧义是否会显著影响后续训练收益：`不会显著阻断 overall 收益，但会成为最需要监控的尾部风险`。当前 held-out 里该 bucket 只有 `2` 条失败，所以它不太可能主导 overall 结果；但它最容易让模型在区间边界上丢边，因此对单桶收益波动最敏感。
- 当前更适合：`GO_TO_STAGE4B_WITH_RELIEF_PACK`。

## Decision Rationale
- relief pack 自身 oracle 已经与 strict 对齐：overall `gt_changed_edge_recall=100.0`、`MAE_on_changed_edges=0.0000`。
- held-out overall `simple_local` 在 bucket-targeted oracle projection 下，`no_anchor_when_gt_changed_rate` 可从 `0.2762` 降到 `0.0000`，`gt_changed_edge_recall` 可从 `63.7` 升到 `87.1`。
- 因此，当前瓶颈更像是“缺少这三类 supervision”，而不是 strict 目标过强。

## Recommended Stage4b Composition
- 最推荐的 stage4b 训练集组成：
- 训练主集：保留现有 `dataset_sparse_v2_stage4b_relabel/train/data.parquet` 作为 anomaly+normal strict-aligned 主集，再追加 `dataset_simple_local_relief_pack/train/data.parquet` 这 48 条 relief 补充监督。
- 评估集：继续保留现有 `dataset_sparse_v2_stage4b_relabel/eval/data.parquet`，并额外单独监控 `dataset_simple_local_relief_pack/eval/data.parquet` 这 24 条 relief eval。
- 采样建议：适度提高 `relief_normal_single_edge` 的 batch 暴露频次，因为它对压 `no_anchor` 最关键；`multi_edge_relief` 保持较小但稳定的曝光，用于约束边界行为。

## Output Files
- CSV: `/root/autodl-tmp/results/relief_pack_ceiling_eval.csv`
- Notes: `/root/autodl-tmp/results/relief_pack_ceiling_notes.md`