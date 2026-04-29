# stage4b Release Decision

## Final Decision
- `HOLD_AND_REPAIR`

## Held-out Simple Local Gate
- pre: `no_anchor_when_gt_changed_rate=0.3619`, `gt_changed_edge_recall=56.7`, `parse_fail_rate_strict=0.2734`
- stage4_fix: `no_anchor_when_gt_changed_rate=0.2762`, `gt_changed_edge_recall=63.7`, `parse_fail_rate_strict=0.2086`
- stage4b_fix: `no_anchor_when_gt_changed_rate=0.2762`, `gt_changed_edge_recall=63.7`, `parse_fail_rate_strict=0.2086`
- gate result: `False`

## Anti-Regression Check
- anti_truncation stage4_fix -> stage4b_fix: `no_anchor_when_gt_changed_rate 0.0 -> 0.0`, `parse_fail_rate_strict 0.0 -> 0.0`; acceptable=`True`
- directional_asymmetry stage4_fix -> stage4b_fix: `gt_changed_edge_recall 71.8 -> 71.8`, `parse_fail_rate_strict 0.0 -> 0.0`; acceptable=`True`

## Relief Bucket Readout
- most materially repaired bucket in real training: `none_materially_fixed`
- if forced to name the only bucket with any visible pickup on relief eval: `multi_edge_relief` (`gt_changed_edge_recall=12.5`, `MAE_on_changed_edges=0.7`)
- remaining dominant risk bucket: `relief_normal_single_edge`
- multi_edge_relief boundary side effect after training: `YES`

## Notes
- relief pack overall eval: `no_anchor_when_gt_changed_rate=0.9583`, `gt_changed_edge_recall=6.2`, `parse_fail_rate_strict=0.9583`
- relabel eval overall: `no_anchor_when_gt_changed_rate=0.0`, `gt_changed_edge_recall=56.9`, `parse_fail_rate_strict=0.0`
- guard eval overall: `no_anchor_when_gt_changed_rate=0.0`, `gt_changed_edge_recall=80.6`, `parse_fail_rate_strict=0.0`
