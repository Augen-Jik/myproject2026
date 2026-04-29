# stage4c Release Decision

## Required Answers
- 29 条是否终于“动了”：`是`（completely_unchanged_count=`28` / 29）
- 其中多少条是 no_anchor_to_anchor：`1`
- single_edge 是否开始主导改善：`是`（single_edge_changed=`1`，other_changed=`0`）
- simple_local no_anchor_when_gt_changed_rate 是否明显低于 0.2762：`否`（stage4c=`0.2667`）
- 是否达到 <= 0.15：`否`
- anti_truncation / directional_asymmetry 是否仍稳定：`是`

## Training Readout
- actual training steps: `32` / planned `32`
- final train loss: `0.3067091703414917`
- eval loss: `None`
- 是否稳定: `True`
- 学习率曲线是否仍过短: `False`
- trainer_state: `/root/autodl-tmp/model_lora_sparse_v2_stage4c_fix/checkpoint-32/trainer_state.json`

## Strict Gate
- stage4_fix simple_local: `no_anchor_when_gt_changed_rate=0.2762`, `gt_changed_edge_recall=63.7`, `parse_fail_rate_strict=0.2086`
- stage4b_fix simple_local: `no_anchor_when_gt_changed_rate=0.2762`, `gt_changed_edge_recall=63.7`, `parse_fail_rate_strict=0.2086`
- stage4c_fix simple_local: `no_anchor_when_gt_changed_rate=0.2667`, `gt_changed_edge_recall=64.3`, `parse_fail_rate_strict=0.2014`

## Slice Readout
- shadow eval overall: `no_anchor_when_gt_changed_rate=0.0`, `gt_changed_edge_recall=91.3`, `parse_fail_rate_strict=0.0`
- relief eval overall: `no_anchor_when_gt_changed_rate=0.0`, `gt_changed_edge_recall=100.0`, `parse_fail_rate_strict=0.0`
- guard eval overall: `no_anchor_when_gt_changed_rate=0.0`, `gt_changed_edge_recall=76.9`, `parse_fail_rate_strict=0.0`

## Final Decision
- `HOLD_AND_REPAIR`
