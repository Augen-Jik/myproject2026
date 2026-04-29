# GAT-v2 Release Notes

## Run Context
- generated_at_utc: `2026-04-09T06:29:38Z`
- frozen_sparse_upstream: `/root/autodl-tmp/model_merged_sparse_v2_stage4_fix`
- dataset_root: `/root/autodl-tmp/dataset_sparse_v2`

## Checkpoints
### Always-GAT-v2
- path: `/root/autodl-tmp/gat_model_v2_always.pt`
- exists/aligned: `True` / `True`
- mode: `always`
- training best_epoch/best_eval_mae: `11` / `0.2637`
- train/eval samples: `1200` / `240`
- upstream: `/root/autodl-tmp/model_merged_sparse_v2_stage4_fix`

### Gated-GAT-v2
- path: `/root/autodl-tmp/gat_model_v2_gated.pt`
- exists/aligned: `True` / `True`
- mode: `gated`
- training best_epoch/best_eval_mae: `11` / `0.2637`
- train/eval samples: `1200` / `240`
- upstream: `/root/autodl-tmp/model_merged_sparse_v2_stage4_fix`

## Mainline Selection
- main_table_method: `Sparse-LoRA-v2 + Gated-GAT-v2`
- appendix_ablation_method: `Sparse-LoRA-v2 + Always-GAT-v2`
- selection_rule: `lower overall strict MAE first, then lower key-slice worst-case MAE; gated 不单独美化。`

## Key Slices
| Scope | Sparse dense MAE | Main dense MAE | Main Δ | Appendix dense MAE | Appendix Δ |
| --- | --- | --- | --- | --- | --- |
| overall | 0.0073 | 0.2012 | 0.1939 | 0.2282 | 0.2209 |
| simple_local | 0.0037 | 0.2021 | 0.1984 | 0.2152 | 0.2115 |
| directional_asymmetry | 0.0064 | 0.2024 | 0.196 | 0.2218 | 0.2154 |
| propagation_range | 0.0 | 0.1902 | 0.1902 | 0.2424 | 0.2424 |
| compound_disaster | 0.0043 | 0.1912 | 0.1869 | 0.2417 | 0.2374 |
| val_anti_truncation | 0.0112 | 0.2027 | 0.1915 | 0.2391 | 0.2279 |

## Release Decision
- GAT-v2 ready_for_paper_main_table: `no`
- decision_reason: 整体口径仍弱于 sparse 基线，更适合附录或特殊场景分析。
- recommended_paper_method_name: `Sparse-LoRA-v2`
