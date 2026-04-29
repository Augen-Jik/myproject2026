# No-SpatioTemporal Notes

## Sparse-LoRA Main Prompt Chain

- `code/inference.py` uses `build_sparse_task_prompt()` for the normal Sparse-LoRA path.
- `build_sparse_task_prompt()` delegates to `code/sparse_utils.py::build_sparse_task_prompt()` which injects `SCENE_TYPE`, `SCHEMA`, `NARRATIVE`, and per-event structured fields such as `ROAD`, `DIR`, `RANGE`, `LEVEL`, `TIME`, `PROPAGATION`, and `CONFLICT`.
- Sparse inference then stays on the same chain: prompt -> `infer_sparse()` -> `parse_sparse_output_bundle()` -> A* path planning -> SUMO evaluation.

## No-SpatioTemporal Change

- This ablation switches only `prompt_mode` from `spatiotemporal_enhanced_prompt` to `baseline_sparse_prompt`.
- The same model weights are reused from the Sparse-LoRA mainline model path.
- The same sparse parser, path planner, no-GAT setting, and evaluation metrics are reused unchanged.

## Scene Results

| Scenario | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) | Parsed Edge Ratio (%) | Status |
|---|---:|---:|---:|---:|---:|---|
| 场景A 黄河路单向拥堵 | 623.9 | 0.994914 | 50.0 | 8.3 | 8.3 | ok |
| 场景B 花园路封闭绕行 | 708.4 | 0.771432 | 100.0 | 12.5 | 12.5 | ok |
| 场景C 多路段复合拥堵 | 544.6 | 1.053821 | 71.4 | 27.1 | 27.1 | ok |
| 场景D 中央核心节点封锁（强迫大范围绕行） | 536.5 | 1.067341 | 26.7 | 0.0 | 4.2 | ok |
| 场景E 不对称方向拥堵（测试模型方向提取能力） | 394.1 | 1.374574 | 36.4 | 9.4 | 9.4 | ok |
| 场景F 大规模复合灾害（极限压力测试） | 446.5 | 1.777582 | 21.4 | 0.0 | 0.0 | parse_failed |

## Mean Row

| Variant | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) | Parsed Edge Ratio (%) |
|---|---:|---:|---:|---:|---:|
| No-SpatioTemporal | 542.333333 | 1.173277 | 50.983333 | 9.550000 | 10.250000 |
