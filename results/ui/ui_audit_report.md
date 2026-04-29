# UI Audit Report

## Scope

本轮只审计界面与展示层，不改核心算法逻辑。已检查的主入口与支撑模块包括：

- `code/app_fixed.py`
- `code/config.py`
- `code/scenarios.py`
- `code/viz.py`
- `code/sim_eval_protocol.py`
- `compare/run_compare.py`
- `code/smallnet_baseline.py`
- `code/ui.py`（作为遗留备用 UI 模块单独核查）

审计时采用的最终结论口径基线为：

- `results/paper_ready/final_method_notes.md`
- `results/gat_v2_release_notes.md`
- `results/final_sparse_freeze_notice.md`
- `results/paper_ready/simulation_setup_section.md`

## 总结论

结论：**当前 UI 整体还不适合直接用于答辩展示。**

更准确地说：

- 主页面的“单次规划主链路”已经基本对齐最终结论。
- 但“综合对比实验页”和“SmallNet 反证页”仍会读取或重新生成旧口径结果。
- 因此，只要答辩现场展示到对比区、GAT 区或 SmallNet 区，就仍有较高概率把老师带回旧方法名、旧时间字段、旧主方法排序逻辑。

如果必须今天就演示，建议**只使用单次规划主链路**，不要展示：

- GAT 开关
- 运行对比实验按钮
- 运行 SmallNet 对比实验按钮
- 当前综合对比表、GAT 高亮表和 SmallNet 结果表

## 哪些界面模块已经和最终结论一致

- `code/config.py:18` 默认模型路径已经冻结到 `/root/autodl-tmp/model_merged_sparse_v2_stage4_fix`，没有把 `stage4b_fix` / `stage4c_fix` / `stage4d_micro` 当默认值。
- `code/app_fixed.py:2249-2258` 主 sidebar 的默认主模型实际指向 `stage4_fix`，没有暴露 patch 分支 checkpoint 作为默认选择。
- `code/app_fixed.py:2702-2768` 已把 `planning_time_s`、`model_infer_time_s`、`route_solve_time_s`、`travel_time_s` 分开记录，运行时主链路口径与最终协议一致。
- `code/app_fixed.py:2718-2720` 明确写出 scene profile 才是环境侧控制源，模型只影响边代价与路径选择，这与最终论文口径一致。
- `code/app_fixed.py:2795-2811` 单次运行页面中，`通行时间` 与 `规划时间` 是分开展示的，没有直接混写成一个指标。
- `code/sim_eval_protocol.py:15-22` 与 `code/sim_eval_protocol.py:101-132` 对旧字段做了统一映射，并把 `true_travel_time` 明确降级到 `analytical_travel_time_s_legacy`，没有把它当 SUMO 实测行程时间。
- `code/scenarios.py` 的 scene presets 仍以 scene type / scene profile 为中心，没有发现 patch SFT 主链路入口。
- 活跃入口里没有发现 `stage4b_fix` / `stage4c_fix` / `stage4d_micro` 作为可选 UI 主模型；这点与 `results/final_sparse_freeze_notice.md:3-18` 一致。

## 哪些界面模块仍需更新

| 模块 | 当前状态 | 主要问题 | 风险判断 |
| --- | --- | --- | --- |
| `code/app_fixed.py` 主页面方法名与对比页 | 部分一致 | 单次运行主链路对齐，但方法名、GAT 高亮、对比页时间标签仍是旧口径 | 高 |
| `compare/run_compare.py` | 不一致 | UI 按钮会重新生成旧 sparse 路径、旧方法名、旧时间字段 | 极高 |
| `code/smallnet_baseline.py` | 不一致 | SmallNet 按钮会重新生成旧 sparse 路径与旧方法名 | 高 |
| 当前被 UI 自动选中的 `compare_results.json` | 不一致 | 当前被选中的文件仍是 `Sparse-LoRA★★ / Sparse-LoRA+GAT★★★` 与 `time_s / sumo_time` | 高 |
| `code/ui.py` | 遗留未接线 | 不是当前主入口，但保留了完全不同的一套算法名与泛化指标 | 低到中 |
| `code/viz.py` | 基本一致 | 主要负责底图与路线渲染，本轮未发现决定性口径冲突 | 低 |

## 关键不一致项

### 1. 主方法名还没有完全冻结成 `Sparse-LoRA-v2`

- 最终结论要求论文口径锁定为 `Sparse-LoRA-v2`，并拒绝把带 `GAT` 的名字设为默认主方法，见 `results/paper_ready/final_method_notes.md:3-6,22-26`。
- 但 `code/app_fixed.py:98-113` 仍把方法全集定义为 `Qwen-LoRA★ / LoRA+GAT★★ / Sparse-LoRA★★ / Sparse-LoRA+GAT★★★`。
- `code/app_fixed.py:2242-2252` 主 sidebar 虽然已出现 `Qwen-Sparse-LoRA-v2`，但这仍不是最终论文口径要求的精确主方法名 `Sparse-LoRA-v2`。
- `code/smallnet_baseline.py:68-71` 仍使用 `Sparse-LoRA★★`。
- `compare/run_compare.py:632-637,769-793` 仍输出 `Sparse-LoRA★★` 与 `Sparse-LoRA+GAT★★★`，并把后者写成“本文终极级联方法”。

判断：**主方法名在单模型入口处部分更新，但在对比展示链路中仍明显残留旧口径。**

### 2. 默认模型路径本身基本干净，但对比/SmallNet 按钮仍回到旧 sparse 路径

- `code/config.py:18` 默认是 `model_merged_sparse_v2_stage4_fix`，这点正确。
- `code/app_fixed.py:2251-2258` 主页面默认也指向 `model_merged_sparse_v2_stage4_fix`，这点正确。
- 但 `compare/run_compare.py:508-510` 仍从 `/root/autodl-tmp/model_merged_sparse` 加载 Sparse-LoRA。
- `code/smallnet_baseline.py:68-71` 也仍从 `/root/autodl-tmp/model_merged_sparse` 加载 Sparse-LoRA。
- 根据 `results/final_sparse_freeze_notice.md:3-18`，演示入口与默认主实验都应冻结到 `stage4_fix`。

判断：**主页面默认路径已对齐；但两个按钮背后的生成脚本还在使用旧 sparse 模型路径。**

### 3. GAT 仍被明显展示成“主方法族/默认增强”

- 最终结论要求 GAT 仅保留为附录或 exploratory analysis，见 `results/paper_ready/final_method_notes.md:22-26,33-36` 与 `results/gat_v2_release_notes.md:25-40`。
- `code/app_fixed.py:143-146` 的 `_IS_OURS` 会把任何包含 `Sparse-LoRA` 或 `LoRA+GAT` 的方法都当成“本文方法”高亮。
- `code/app_fixed.py:2280-2305` 在 sidebar 中以一级区块呈现 GAT 开关、GAT alpha、GAT 模型路径，虽然默认 `value=False`，但视觉权重仍很高。
- `code/app_fixed.py:3050-3079` 的综合对比表把 GAT 级联方法涂成绿色，并在图例中写成“绿底：GAT级联类方法”。
- 当前 UI 自动选中的对比文件仍包含 `LoRA+GAT★★` 与 `Sparse-LoRA+GAT★★★`，因此对比页会持续把 GAT 显示在主方法集里。
- `compare/run_compare.py:771-793` 仍把 `Sparse-LoRA+GAT★★★` 标为“本文终极级联方法”。

判断：**GAT 不是默认开关，但仍被 UI 作为主方法族高亮展示，和最终结论不一致。**

### 4. patch SFT 没有作为 active UI 入口出现，但配套实验页仍未完全切回最终 frozen sparse 主线

- 正面情况：本轮未在 `code/app_fixed.py` 主 sidebar 中发现 `stage4b_fix` / `stage4c_fix` / `stage4d_micro` 的可选入口。
- 这与 `results/final_sparse_freeze_notice.md:12-18` 和 `results/final_patch_negative_results.md:29-32` 基本一致。
- 但问题在于，`compare/run_compare.py` 与 `code/smallnet_baseline.py` 并没有切回最终 frozen sparse 主线，而是继续用旧 sparse 路径。
- 也就是说，patch SFT 虽然没有被当作 UI 主链路继续卖点化，但辅助展示页依然没有完全贴合“冻结后的最终主线”。

判断：**没有发现 patch SFT 被直接当成当前 UI 主卖点；但辅助展示链路仍未完全回到最终 freeze 口径。**

### 5. `Planning Time` / `Travel Time` 仍在综合对比页发生混写

- `code/app_fixed.py:1535-1549` 中，对比页内部的 `time` 字段来自 `planning_time_s`；如果没有该字段，就回退到 legacy `time_s`。
- 但 `code/app_fixed.py:1754-1756`、`code/app_fixed.py:1883-1886`、`code/app_fixed.py:3005-3045` 却把这个 `time` 展示成“推理速度 / 推理时间 / 推理耗时”。
- 这会把“规划总时间”错误表述成“推理时间”。
- 同时，对比页又把 travel side 全部写成 `SUMO时间` / `SUMO 行程时间`，而不是最终锁定的 `Travel Time (s)`。
- 最终协议要求统一口径是 `Travel Time (s)`、`model_infer_time_s`、`route_solve_time_s`、`Planning Time (s)`，见 `results/paper_ready/final_method_notes.md:33-38`。

判断：**单次运行页时间口径是对的，但综合对比页把 planning side 又重新写成 inference side，属于当前最明显的展示层混写问题。**

### 6. `true_travel_time` 没有在 active UI 中被误当成 SUMO travel time

- 本轮在 `code/app_fixed.py`、`code/config.py`、`code/viz.py`、`code/scenarios.py`、`code/ui.py` 中未发现 `true_travel_time` 被直接展示成 SUMO 行程时间。
- `code/sim_eval_protocol.py:15-22,122-132` 也已把 `true_travel_time` 映射为 `analytical_travel_time_s_legacy`。

判断：**这一项在 active UI 主链路中目前基本一致。**

### 7. old metric names 没有直接出现在主页面标题里，但仍通过兼容链路和结果生成器渗透到对比页

- `code/app_fixed.py:1500-1549` 明确兼容 `plan_ms / time_s / sumo_time`。
- `compare/run_compare.py:817-818,857-860` 继续生成 `time_s / sumo_time / planning_ms`。
- 当前被选中的对比结果文件就是旧字段结构，因此对比页虽然不会把字段名原样打出来，但底层仍在依赖旧字段。

判断：**old metric names 在展示文案层“半隐藏”，但在数据来源层仍是 active input。**

## 哪些展示元素会误导答辩老师

- `code/app_fixed.py:98-113` 的方法全集仍保留星号方法名，会让老师以为当前项目主结论仍是旧命名版本。
- `code/app_fixed.py:2225` 的副标题仍是 `Qwen/DeepSeek-R1 SFT`，会把叙述中心拉回“多模型 SFT 家族”，而不是冻结后的 `Sparse-LoRA-v2` 主线。
- `code/app_fixed.py:2280-2305` 的 GAT 一级侧边栏会让老师误以为 GAT 是当前默认增强或推荐主方法。
- `code/app_fixed.py:1754-1756`、`code/app_fixed.py:1886`、`code/app_fixed.py:3005-3045` 把 planning side 说成“推理时间/推理耗时”，会直接混淆 `Planning Time` 与 `model_infer_time_s`。
- `code/app_fixed.py:3052-3079` 的绿色高亮和图例会把 GAT 行 visually 提升为“更高级的本文方法”。
- `compare/run_compare.py` 与 `smallnet_baseline.py` 会在点击按钮后生成旧 sparse 路径和旧标签结果，使 live demo 重新偏离最终结论。

## 哪些按钮/开关应该隐藏、降级或标注 `exploratory_only`

- `code/app_fixed.py:2280-2305` 的 GAT 区块：建议默认隐藏，或在 defense mode 下明确标注 `exploratory_only / appendix only`。
- `code/app_fixed.py:2954-2969` 触发的“运行对比实验”按钮：在 `compare/run_compare.py` 更新前应隐藏，至少要标注 `legacy compare runner`，否则会重生成旧口径结果。
- `code/app_fixed.py:3102-3118` 触发的“运行SmallNet对比实验”按钮：在 `code/smallnet_baseline.py` 更新前应隐藏，至少要标注 `legacy smallnet runner`。
- `code/app_fixed.py:3050-3079` 的 GAT 绿色高亮与图例：应移除，或仅在 appendix/exploratory mode 中出现。
- 对比页中的 GAT 方法行：应默认折叠到 appendix/exploratory_only 分组，而不是放在“完整方法集”默认视图里。

## 对 `code/ui.py` 的判断

- 本轮 repo-wide grep 没有发现其他模块导入 `code/ui.py`；它看起来不是当前主入口。
- 但该文件内仍保留了另一套完全不同的产品叙事，例如 `城市交通信号灯智能调度系统`、`LLM-R1-SFT`、`平均延迟/成功率/运行时间` 等泛化指标。
- 因为它当前未接到 `app_fixed.py` 主链路，所以不是当前最高风险项。
- 但如果后续有人误把它重新接回 Streamlit 主入口，会立即引入新的答辩口径偏移。

判断：**建议标注 deprecated，或在代码层明确归档。**

## 当前 UI 是否适合直接答辩展示

**不适合直接整页答辩展示。**

原因不是主链路逻辑错误，而是展示层仍有三类高风险残留：

- 对比页和 SmallNet 页会读取或生成旧方法名、旧 sparse 路径、旧 timing 字段。
- GAT 仍被视觉高亮为主方法族，而最终结论要求其退到 appendix / exploratory。
- 对比页把 planning side 重新包装成 inference side，容易在答辩中被老师追问时间定义是否一致。

## 最优先要修的 5 个点

1. 先修 `compare/run_compare.py`：把 Sparse 默认路径切到 `model_merged_sparse_v2_stage4_fix`，并改成输出 `Sparse-LoRA-v2` 与协议化时间字段；否则“运行对比实验”按钮会持续产出旧口径结果。
2. 再修 `code/app_fixed.py` 对比页的时间展示：把当前“推理耗时/推理速度”改成 `Planning Time (s)`，必要时另列 `model_infer_time_s` 与 `route_solve_time_s`，杜绝 planning / inference 混写。
3. 把 `code/app_fixed.py` 中所有 paper-facing 方法名统一冻结到 `Sparse-LoRA-v2`，并移除 `★/★★/★★★` 旧标签；GAT 如果保留，只能用 appendix/exploratory 命名。
4. 降级 GAT 展示：隐藏 sidebar GAT 区块，移除对比表中的绿色高亮与“GAT级联类方法”图例，避免老师误判 GAT 仍是当前默认主方法。
5. 修 `code/smallnet_baseline.py` 与 `results/smallnet_results.json`：把 SmallNet 路径、方法名与 frozen sparse 主线同步到 `stage4_fix / Sparse-LoRA-v2`；若本轮来不及，就直接隐藏 SmallNet 按钮和当前结果区。

## 一句话结论

**主链路 demo 已接近可答辩，但整页 UI 还没有完成最终口径收口；最危险的不是算法，而是“对比/辅助实验页会把旧口径重新带回现场”。**
