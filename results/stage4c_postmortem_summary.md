# stage4c HOLD_AND_REPAIR Postmortem

## Final Judgment
- Action verdict: `DIMINISHING_RETURNS_REACHED`
- Root-cause shape: `迁移失败 > 范围边界失败`
- Final one-liner: `收益已接近耗尽`

## Executive Readout
- held-out `simple_local` 只出现边际改善：`no_anchor_when_gt_changed_rate 0.2762 -> 0.2667`，`gt_changed_edge_recall 63.7 -> 64.3`，`parse_fail_rate_strict 0.2086 -> 0.2014`。
- targeted `diff_29` 只动了 `1/29`：`relief_normal_single_edge` 仅 `1/20` 改善，`relief_normal_short_range` `7/7` 完全没动，`multi_edge_relief` `2/2` 完全没动。
- 真正落到 held-out target 的净收益只有 `+1` 条 single-edge 修复，但 normal-only 假阳性新锚点新增了 `8` 条。
- `shadow_eval / relief_eval` 明显学会了：relief overall `gt_changed_edge_recall=100.0`、`no_anchor_when_gt_changed_rate=0.0`、`parse_fail_rate_strict=0.0`；shadow overall `gt_changed_edge_recall=91.3`。

## A. Net Improvements Vs stage4_fix
- summary 层面真正相关的净提升集中在 `val_normal/simple_local`，尤其是 `short`：`no_anchor_when_gt_changed_rate 0.2353 -> 0.2206`，`gt_changed_edge_recall 62.7 -> 63.6`，`parse_fail_rate_strict 0.1905 -> 0.1786`。`medium` 只有 recall 小幅增加：`68.8 -> 70.1`。
- targeted semantic-gap 层面，只有 `relief_normal_single_edge` 有净提升，而且只是 `1/20`；具体只多救回了一个 `车流顺畅` 单边样本。

## B. Completely Unchanged Buckets
- `diff_29` 里 `relief_normal_short_range` `7/7` 完全没动，`multi_edge_relief` `2/2` 完全没动；即便是 `relief_normal_single_edge` 也还有 `19/20` 没动。
- held-out summary 里 `val_normal:length_bucket/long` 与 `val_normal:length_bucket/noisy_long` 指标完全不变：`long recall 80.0 -> 80.0`，`noisy_long recall 55.6 -> 55.6`；single-edge 的 `prefixed_short` 和 `planner_long` 结构也没有任何净改善。

## C. no_anchor -> anchor But Still Wrong Range
- targeted `diff_29` 里这一类是 `0`。
- 整个 held-out `val_normal/simple_local/gt_changed>0` 里这一类也是 `0`；`no_anchor -> anchor` 一共出现 `9` 次，但其中只有 `1` 次是真修复，其余 `8` 次都是 `gt_changed=0` 的 normal-only 假阳性，不是“范围边界差一点”的 recoveries。

## D. shadow_eval / relief_eval Learned But Held-out Did Not Transfer
- `relief_eval` 已经把 `relief_normal_single_edge` 学满：bucket `count=34`，`gt_changed_edge_recall=100.0`，`no_anchor_when_gt_changed_rate=0.0`。
- phrase 细分同样学满：`畅通无阻` eval `10/10`，`顺畅` eval `14/14`，都是 `100` recall / `0` parse fail；但 held-out single-edge 里 `畅通无阻` 仍是 `11/11` 失败，`车流顺畅` 仍是 `8/9` 失败。
- `shadow_eval` 的 `anomaly_plus_normal_side_event` 也不是完全不会，bucket `gt_changed_edge_recall=68.8`、`no_anchor_when_gt_changed_rate=0.0`；但 held-out 剩余 hard cases 里 mixed normal+abnormal 还有 `22` 条，且主要表现为 anchor 出来了但范围没补全。

## E. Guard Stability
- guard 可以判定为稳定：held-out `directional_asymmetry` recall `+3.6`，`anti_truncation` recall `+0.5`，no-anchor 与 strict parse fail 都没有变差。
- dedicated guard eval 也是 `0` no_anchor / `0` strict parse fail；其中 `directional_asymmetry_stable` `100.0` recall，`anti_truncation_canary` `80.0` recall，`hard_negative_no_anchor` 仅 `50.0` recall 但样本数只有 `1`。

## Remaining Hard Cases
- stage4c 之后 `val_normal/simple_local` 还剩 `50` 条 changed-edge 失败，其中 `28` 条还是纯 `no_anchor`，`22` 条变成了 `anchor_but_incomplete_range`。
- single-edge 剩余失败仍是主桶：`19` 条，phrase 只剩两类：`畅通无阻=11`、`车流顺畅=8`；结构上 `short_plain=11`、`prefixed_short=7`、`planner_long=1`。
- 弱语气 relief 不是当前主难点：remaining hard cases 里 `weak_relief=0`。

## Interpretation
- 这轮 patch SFT 不是完全没学到，而是“in-mixture 学会了，但 held-out 迁移非常弱”。从根因上看更像 `迁移失败`，不是 `no_anchor -> anchor 但范围差一点` 的边界问题。
- 但从继续推进 patch SFT 的行动价值看，`32` steps、`59.5%` target share、relief eval 全会之后，held-out 仍只多修 `1` 条 single-edge 且引入 `8` 条 normal-only 假阳性，所以更接近 `收益已接近耗尽`。

## Source Note
- release decision 仍是 `HOLD_AND_REPAIR`。
