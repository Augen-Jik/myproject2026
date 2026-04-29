# GAT Case Analysis

以下结论均基于 clean run 自动汇总；其中 `travel_time_delta` 指协议化 `Travel Time (s)` 的差值。涉及原因解释时均属于“推测/可能原因”。

## GAT 有帮助的场景

- scenario: 场景D 中央核心节点封锁（强迫大范围绕行）
  base_method: Sparse-LoRA
  enhanced_method: Sparse-LoRA+GAT
  travel_time_delta: -46.10s
  constraint_delta: 0.00%
  inferred_reason: 推测：图补全补上了稀疏异常边之间的拓扑传播信息，因此在 SUMO 闭环 `Travel Time (s)` 上减少了绕行损失。

## GAT 无明显收益或持平的场景

- scenario: 场景A 黄河路单向拥堵
  base_method: Sparse-LoRA
  enhanced_method: Sparse-LoRA+GAT
  travel_time_delta: 0.00s
  constraint_delta: 0.00%
  inferred_reason: 推测：稀疏输出本身已足够指示关键异常边，额外 GAT 平滑未显著改善路径选择，或仅带来轻微代价扰动。
- scenario: 场景B 花园路封闭绕行
  base_method: Sparse-LoRA
  enhanced_method: Sparse-LoRA+GAT
  travel_time_delta: 0.00s
  constraint_delta: 0.00%
  inferred_reason: 推测：稀疏输出本身已足够指示关键异常边，额外 GAT 平滑未显著改善路径选择，或仅带来轻微代价扰动。
- scenario: 场景C 多路段复合拥堵
  base_method: Sparse-LoRA
  enhanced_method: Sparse-LoRA+GAT
  travel_time_delta: 0.00s
  constraint_delta: 0.00%
  inferred_reason: 推测：稀疏输出本身已足够指示关键异常边，额外 GAT 平滑未显著改善路径选择，或仅带来轻微代价扰动。
- scenario: 场景E 不对称方向拥堵（测试模型方向提取能力）
  base_method: Sparse-LoRA
  enhanced_method: Sparse-LoRA+GAT
  travel_time_delta: 0.00s
  constraint_delta: 0.00%
  inferred_reason: 推测：稀疏输出本身已足够指示关键异常边，额外 GAT 平滑未显著改善路径选择，或仅带来轻微代价扰动。
- scenario: 场景F 大规模复合灾害（极限压力测试）
  base_method: Sparse-LoRA
  enhanced_method: Sparse-LoRA+GAT
  travel_time_delta: 0.00s
  constraint_delta: 0.00%
  inferred_reason: 推测：稀疏输出本身已足够指示关键异常边，额外 GAT 平滑未显著改善路径选择，或仅带来轻微代价扰动。
