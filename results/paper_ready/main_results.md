# Main Results

## Table 1. Main Results

Rows are 6-scenario averages under the unified SUMO evaluation protocol.

| Method | Travel Time (s) | Constraint Rate (%) | model_infer_time_s | route_solve_time_s | Planning Time (s) | Signal Ratio (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Sparse-LoRA-v2 | 511.15 | 68.23 | 1.47 | 0.000080 | 1.470080 | 9.38 |
| Rule-A* | 513.55 | 61.57 | 0.00 | 0.000070 | 0.000070 | 91.70 |
| Qwen-LoRA | 591.95 | 47.42 | 15.86 | 0.000050 | 15.860050 | 34.72 |
| R1-LoRA | 785.65 | 42.73 | 31.49 | 0.000070 | 31.490070 | 51.73 |

`Planning Time (s) = model_infer_time_s + route_solve_time_s`.

## GAT-v2 Positioning

- `GAT-v2` 已从论文主结果表移除，降级为附录 / exploratory analysis。
- 主推荐方法名锁定为 `Sparse-LoRA-v2`。
- 若附录保留 GAT，推荐名称为 `Sparse-LoRA-v2 + Gated-GAT-v2`。
- `Always-GAT-v2` 仅保留为附录 / 消融。
- 详细结论见 `results/gat_v2_final_positioning.md` 与 `results/gat_v2_appendix_only_table.csv`。
