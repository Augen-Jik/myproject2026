# Table2 Final Preview

> Frozen for writing: `LoRA+GAT` is retained only as a legacy dense+GAT control, while `No-SpatioTemporal` isolates the removal of structured spatiotemporal prompt fields.

## Mean Rows

| Variant | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) | Parsed Edge Ratio (%) |
|---|---:|---:|---:|---:|---:|
| Qwen-LoRA | 591.95 | 15.858387 | 47.42 | 34.72 | 50.00 |
| LoRA+GAT | 574.42 | 15.785057 | 47.42 | 64.25 | 50.00 |
| No-SpatioTemporal | 542.33 | 1.173277 | 50.98 | 9.55 | 10.25 |
| Sparse-LoRA | 511.15 | 1.475079 | 68.23 | 9.38 | 9.38 |
| Sparse-LoRA+GAT | 503.47 | 1.481741 | 68.23 | 61.45 | 9.38 |

## No-SpatioTemporal Scene Rows

| Scenario | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) | Parsed Edge Ratio (%) | Status |
|---|---:|---:|---:|---:|---:|---|
| 场景A 黄河路单向拥堵 | 623.9 | 0.994914 | 50.0 | 8.3 | 8.3 | ok |
| 场景B 花园路封闭绕行 | 708.4 | 0.771432 | 100.0 | 12.5 | 12.5 | ok |
| 场景C 多路段复合拥堵 | 544.6 | 1.053821 | 71.4 | 27.1 | 27.1 | ok |
| 场景D 中央核心节点封锁（强迫大范围绕行） | 536.5 | 1.067341 | 26.7 | 0.0 | 4.2 | ok |
| 场景E 不对称方向拥堵（测试模型方向提取能力） | 394.1 | 1.374574 | 36.4 | 9.4 | 9.4 | ok |
| 场景F 大规模复合灾害（极限压力测试） | 446.5 | 1.777582 | 21.4 | 0.0 | 0.0 | parse_failed |
