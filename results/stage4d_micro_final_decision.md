# stage4d_micro Final Decision

## Required Answers
- 目标桶是否终于明显动了：`否`（sample_count=`18`，no_anchor_to_anchor=`0`，completely_unchanged=`18`）
- no_anchor_to_anchor 有多少条：`0`
- normal-only 假阳性是否少于 stage4c：`否`（stage4d=`8`，stage4c=`8`）
- simple_local overall 是否有可观净收益：`否`（no_anchor_when_gt_changed_rate: stage4c=`0.2667` -> stage4d=`0.2762`；gt_changed_edge_recall: stage4c=`64.3` -> stage4d=`63.7`；parse_fail_rate_strict: stage4c=`0.2014` -> stage4d=`0.2086`）
- 是否达到放行条件：`否`
- 若仍未达到，是否正式停止所有后续 patch：`是`

## Supporting Readout
- target_bucket_moved_any: `False`
- anchor_but_still_wrong_count: `0`
- stage4_fix simple_local: `no_anchor_when_gt_changed_rate=0.2762`, `gt_changed_edge_recall=63.7`, `parse_fail_rate_strict=0.2086`
- stage4c simple_local: `no_anchor_when_gt_changed_rate=0.2667`, `gt_changed_edge_recall=64.3`, `parse_fail_rate_strict=0.2014`
- stage4d simple_local: `no_anchor_when_gt_changed_rate=0.2762`, `gt_changed_edge_recall=63.7`, `parse_fail_rate_strict=0.2086`
- target bucket eval overall: `no_anchor_when_gt_changed_rate=0.0556`, `gt_changed_edge_recall=94.4`, `parse_fail_rate_strict=0.0556`
- hard negative guard overall: `false_positive_anchor_count=8`, `false_positive_anchor_rate=1.0`, `exact_no_anchor_rate=0.0`
- canary overall: `no_anchor_when_gt_changed_rate=0.0`, `gt_changed_edge_recall=66.7`, `parse_fail_rate_strict=0.0`
- training steps: `27` / `27`
- final train loss: `0.16193881630897522`
- final eval loss: `None`
- trainer_state: `/root/autodl-tmp/model_lora_sparse_v2_stage4d_micro/checkpoint-27/trainer_state.json`

## Final Decision
- `FREEZE_STAGE4_FIX_AND_STOP_PATCHING`
