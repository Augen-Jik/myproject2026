# Final Patch Negative Results

## stage4b
- 失败原因：训练预算过短，真实 `Trainer` 更新只有 `9` steps，target signal 太散，没形成可见参数位移。
- 直接证据：`diff_29` 完全不动，`completely_unchanged_count = 29 / 29`，`no_anchor_to_anchor_count = 0 / 29`。
- held-out 结果也没有净收益：`stage4_fix` 与 `stage4b_fix` 的 simple_local 指标完全相同，都是 `no_anchor_when_gt_changed_rate = 0.2762`、`gt_changed_edge_recall = 63.7`、`parse_fail_rate_strict = 0.2086`。
- 结论：`stage4b` 证明“轻量 patch + 分散 mixture”不足以推动主失败桶。

## stage4c
- 失败原因：虽然把训练预算抬到 `32` steps，也把 supervision 更集中到 single-edge relief，但迁移到 held-out 自然分布的收益仍然极小。
- 最关键数字：`stage4c_diff_29.json` 显示只动了 `1 / 29`，即 `completely_unchanged_count = 28 / 29`、`no_anchor_to_anchor_count = 1 / 29`。
- 同时新增 `8` 条 normal-only 假阳性：`stage4c_remaining_hard_cases.json` 中 `new_false_positive_normal_only_anchors = 8`。
- 即使 strict simple_local 有轻微改善，仍只到 `no_anchor_when_gt_changed_rate = 0.2667`，离放行阈值仍明显过远。
- 结论：`stage4c` 证明“更强 patch SFT”已经出现边际收益急剧递减，而且开始引入正常样本误触发。

## stage4d_micro
- 失败原因：最后一轮 micro patch 虽然在专门 eval mixture 上看起来学到了目标桶，但没有迁移到真正剩余 hard cases，且 hard negative guard 完全失守。
- 最关键数字：
- `stage4d_micro_target_diff.json`：`0 / 18 no_anchor -> anchor`，`completely_unchanged_count = 18 / 18`。
- `stage4d_micro_hard_negative_guard_summary.csv`：`8 / 8` hard negative 都出现假阳性，`false_positive_anchor_rate = 1.0`。
- held-out simple_local overall 也回退到 `stage4_fix` 水平：`no_anchor_when_gt_changed_rate = 0.2762`、`gt_changed_edge_recall = 63.7`、`parse_fail_rate_strict = 0.2086`。
- 结论：`stage4d_micro` 证明“只打唯一高 ROI 桶”的最后一轮 patch 也没能在自然 held-out 分布上产生稳定可放行收益。

## Key Numbers
- `stage4b`: `diff_29` 完全不动，`29 / 29` unchanged。
- `stage4c`: 只动 `1 / 29`，且新增 `8` 条 normal-only 假阳性。
- `stage4d_micro`: `0 / 18 no_anchor -> anchor`，`8 / 8` hard negative 假阳性。

## Final Conclusion
- patch SFT 在该 held-out 自然分布上的收益已接近耗尽。
- 正式冻结主实验 sparse 上游基线为 `/root/autodl-tmp/model_merged_sparse_v2_stage4_fix`。
- `stage4b_fix`、`stage4c_fix`、`stage4d_micro` 仅保留为 `exploratory_only` 负结果材料，不再进入默认主链路。
