# UI Mainline Alignment Notes

## Scope

本轮仅调整 Streamlit 主界面的展示逻辑与说明文案，不改核心算法逻辑。

主改动文件：

- `/root/autodl-tmp/code/app_fixed.py`

## 已完成的主界面对齐

- 默认主方法已锁定为 `Sparse-LoRA-v2`。
- 默认上游模型路径已锁定为 `/root/autodl-tmp/model_merged_sparse_v2_stage4_fix`。
- 主界面标题、副标题与顶部说明区已统一到最终论文口径。
- 主界面新增主链路说明：
  - `stage4_fix -> planner / SUMO / visualization`
  - `Planning Time (s) = 模型推理 + 路径求解`
  - `Travel Time (s) = SUMO 实际通行时间`

## 模型与方法展示调整

- 旧的 starred 方法名不再作为主界面默认展示名。
- 主界面默认只显式暴露 `Sparse-LoRA-v2`。
- baseline 模型入口被收纳到“高级选项（baseline / exploratory only）”。
- GAT 入口仍保留，但已明确标注：
  - `exploratory only, not mainline`
- patch SFT 分支未作为主界面默认切换项暴露，并在高级选项中明确标注：
  - `negative results / exploratory only`
- PPO 在主界面中被明确定位为：
  - `baseline only`

## 对比与附录区调整

- “方法对比”标签页已改为“附录/探索性对比”定位。
- 对比页按钮在 `Demo Mode` 下默认隐藏；仅 `Experiment Mode` 显示探索性运行入口。
- 对比页图表与表格已把主时间口径统一为：
  - `Travel Time (s)`
  - `Planning Time (s)`
- GAT 行不再作为默认主方法高亮，而是按 exploratory / appendix-only 口径展示。
- SmallNet 区块已改为附录反证实验定位，并在 `Demo Mode` 下隐藏运行按钮。

## 兼容性说明

- 历史结果文件中的旧方法名与旧字段仍可被读取，但主界面会优先映射为当前最终展示口径。
- 本轮没有删除 exploratory 路径，只是把它们从默认主链路中降级并隐藏。

## 当前判断

就主界面展示层而言，默认入口已经与最终论文口径对齐：

- `Sparse-LoRA-v2` = 主方法
- `stage4_fix` = 默认上游
- `GAT` = exploratory / appendix-only
- `patch SFT` = negative results / exploratory-only
- `PPO` = baseline only

剩余注意点：

- `compare/run_compare.py` 与部分历史对比结果文件仍属于旧实验产物，因此主界面已把它们降级为附录/探索性入口，而不是默认答辩主链路。
