# Delta Analysis

`Travel Time (s)` deltas follow the unified SUMO protocol. `route_solve_time_s` deltas are negligible (`10^-5` to `10^-4` s) and omitted here.

| Scenario | Base Method | Enhanced Method | ΔTravel Time (s) | ΔConstraint Rate (%) | Δmodel_infer_time_s | GAT Helpful? |
| --- | --- | --- | --- | --- | --- | --- |
| 场景A 黄河路单向拥堵 | Sparse-LoRA | Sparse-LoRA+GAT | 0.00 | 0.00 | 0.02 | No |
| 场景B 花园路封闭绕行 | Sparse-LoRA | Sparse-LoRA+GAT | 0.00 | 0.00 | 0.02 | No |
| 场景C 多路段复合拥堵 | Sparse-LoRA | Sparse-LoRA+GAT | 0.00 | 0.00 | 0.02 | No |
| 场景D 中央核心节点封锁（强迫大范围绕行） | Sparse-LoRA | Sparse-LoRA+GAT | -46.10 | 0.00 | -0.02 | Yes |
| 场景E 不对称方向拥堵（测试模型方向提取能力） | Sparse-LoRA | Sparse-LoRA+GAT | 0.00 | 0.00 | -0.01 | No |
| 场景F 大规模复合灾害（极限压力测试） | Sparse-LoRA | Sparse-LoRA+GAT | 0.00 | 0.00 | 0.00 | No |
| 场景A 黄河路单向拥堵 | Rule-A* | Sparse-LoRA+GAT | 0.00 | 0.00 | 1.17 | No |
| 场景B 花园路封闭绕行 | Rule-A* | Sparse-LoRA+GAT | 0.00 | 0.00 | 1.16 | No |
| 场景C 多路段复合拥堵 | Rule-A* | Sparse-LoRA+GAT | 0.00 | 0.00 | 1.52 | No |
| 场景D 中央核心节点封锁（强迫大范围绕行） | Rule-A* | Sparse-LoRA+GAT | 0.00 | 40.00 | 1.86 | Yes |
| 场景E 不对称方向拥堵（测试模型方向提取能力） | Rule-A* | Sparse-LoRA+GAT | -60.50 | 0.00 | 1.35 | Yes |
| 场景F 大规模复合灾害（极限压力测试） | Rule-A* | Sparse-LoRA+GAT | 0.00 | 0.00 | 1.83 | No |
