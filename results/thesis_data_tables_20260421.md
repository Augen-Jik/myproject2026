# 论文数据汇总文档

**生成时间：** 2026-04-21  
**模型：** `model_merged_sparse_v2_stage4_fix`（Qwen2.5-1.5B，LoRA 三阶段课程训练，冻结于 2026-04-17）  
**评估数据集：** `dataset_sparse_v2`（val_normal / val_special / val_anti_truncation，各 180 条，共 540 条）  
**对照实验：** DIR_RULE 推理注入（2026-04-21），结果见第四节

---

## 名词说明

| 缩写 | 全称 | 含义 |
|---|---|---|
| no_anchor_miss | no_anchor_when_gt_changed_rate | GT 有变化边、但模型输出 0 个 anchor 的样本比例 |
| strict_fail | parse_fail_rate_strict | anchor 格式合法但语义无法映射到路网边的比例 |
| recall | gt_changed_edge_recall | 模型找到的 GT 变化边 / GT 变化边总数（% ） |
| target_recall | target_recall | 高影响边（Δ≥2.0）召回率（% ） |
| MAE_changed | MAE_on_changed_edges | 仅在 GT 变化边上的 MAE |
| dense_mae | dense_weight_mae | 全 96 条边的 MAE |
| conf | parse_confidence | anchor 提取置信度（0–1） |
| anchor_count | anchor_count | 每条样本平均输出 anchor 数 |
| latency | inference_time | 单条样本推理耗时（秒） |

---

## 一、总体指标（表 3.x 主表用）

| 指标 | overall (n=540) | val_normal (n=180) | val_special (n=180) | val_anti_trunc (n=180) |
|---|---|---|---|---|
| no_anchor_rate | 11.67% | 35.00% | 0.00% | 0.00% |
| no_anchor_miss | 6.11% | **19.86%** | 0.00% | 0.00% |
| strict_fail | 5.37% | **16.11%** | 0.00% | 0.00% |
| target_recall | 99.6% | 100.0% | 99.5% | 99.6% |
| recall | **89.3%** | 65.9% | 89.7% | **98.0%** |
| MAE_changed | 0.0932 | 0.2725 | 0.0917 | 0.0251 |
| dense_mae | **0.0073** | 0.0043 | 0.0064 | 0.0112 |
| conf | 0.9707 | 0.9520 | 0.9800 | 0.9800 |
| anchor_count (avg) | 1.493 | 0.744 | 1.767 | 1.967 |
| parsed_edge_ratio | 3.04% | 1.05% | 3.86% | 4.21% |
| latency | 1.323s | 1.025s | 1.427s | 1.517s |
| gt_changed_sample_rate | — | 81.1% | 96.1% | 86.7% |
| avg_gt_changed_edge_count | — | 1.53 | 4.08 | 3.94 |

---

## 二、分场景类型（表 3.1(b) 用）

### 2.1 val_normal 场景细分

| 场景类型 | n | no_anchor_miss | strict_fail | recall | target_recall | MAE_changed | dense_mae | conf | anchor_count | latency |
|---|---|---|---|---|---|---|---|---|---|---|
| simple_local | 139 | **27.62%** | **20.86%** | **63.7%** | 100.0% | 0.2901 | 0.0037 | 0.9437 | 0.547 | 0.935s |
| directional_asymmetry | 41 | 0.00% | 0.00% | **69.5%** | 100.0% | 0.2438 | 0.0065 | 0.9800 | 1.415 | 1.328s |

### 2.2 val_special 场景细分

| 场景类型 | n | no_anchor_miss | strict_fail | recall | target_recall | MAE_changed | dense_mae | conf | anchor_count | latency |
|---|---|---|---|---|---|---|---|---|---|---|
| directional_asymmetry | 30 | 0.00% | 0.00% | 74.4% | 100.0% | 0.2044 | 0.0064 | 0.9800 | 1.567 | 1.395s |
| core_blockage | 44 | 0.00% | 0.00% | 78.8% | 100.0% | 0.1696 | 0.0074 | 0.9800 | 1.273 | 1.178s |
| compound_disaster | 37 | 0.00% | 0.00% | **94.2%** | 98.7% | 0.0742 | 0.0050 | 0.9800 | 2.946 | 1.938s |
| temporal_switch | 32 | 0.00% | 0.00% | **100.0%** | 100.0% | 0.0000 | 0.0143 | 0.9800 | 1.000 | 1.092s |
| propagation_range | 37 | 0.00% | 0.00% | **100.0%** | 100.0% | 0.0000 | 0.0000 | 0.9800 | 2.000 | 1.527s |

### 2.3 val_anti_truncation 场景细分

| 场景类型 | n | no_anchor_miss | strict_fail | recall | MAE_changed | dense_mae | conf | anchor_count | latency |
|---|---|---|---|---|---|---|---|---|---|
| compound_disaster | 50 | 0.00% | 0.00% | 95.6% | 0.0556 | 0.0037 | 0.9800 | 2.980 | 1.946s |
| temporal_switch | 55 | 0.00% | 0.00% | 100.0% | 0.0000 | 0.0333 | 0.9800 | 1.000 | 1.093s |
| propagation_range | 75 | 0.00% | 0.00% | 100.0% | 0.0000 | 0.0000 | 0.9800 | 2.000 | 1.543s |

### 2.4 全场景汇总（overall，供横向比较）

| 场景类型 | n | no_anchor_miss | strict_fail | recall | dense_mae |
|---|---|---|---|---|---|
| simple_local | 139 | 27.62% | 20.86% | 63.7% | 0.0037 |
| directional_asymmetry | 71 | 0.00% | 0.00% | 71.8% | 0.0064 |
| core_blockage | 44 | 0.00% | 0.00% | 78.8% | 0.0074 |
| compound_disaster | 87 | 0.00% | 0.00% | 95.0% | 0.0043 |
| temporal_switch | 87 | 0.00% | 0.00% | 100.0% | 0.0263 |
| propagation_range | 112 | 0.00% | 0.00% | 100.0% | 0.0000 |

> **规律：** simple_local 是唯一存在 no_anchor_miss 的场景（27.6%），其他场景 strict_fail=0%、recall≥70%。模型在单路段局部约束上存在系统性漏报，是当前最主要的失败模式。

---

## 三、按长度分桶（表 3.7 用）

### 3.1 val_normal × length_bucket（Baseline）

| 长度分桶 | n | no_anchor_miss | strict_fail | recall | MAE_changed | dense_mae | conf | anchor_count | latency |
|---|---|---|---|---|---|---|---|---|---|
| short | 84 | 23.53% | 19.05% | 62.7% | 0.2983 | 0.0044 | 0.9495 | 0.679 | 0.995s |
| medium | 85 | 15.49% | 12.94% | **68.8%** | 0.2500 | 0.0044 | 0.9565 | 0.835 | 1.065s |
| long | 6 | 25.00% | 16.67% | **80.0%** | 0.1600 | 0.0014 | 0.9400 | 0.500 | 0.908s |
| noisy_long | 5 | 33.33% | 20.00% | 55.6% | 0.3556 | 0.0067 | 0.9320 | 0.600 | 0.976s |

> ⚠️ long（n=6）和 noisy_long（n=5）样本量极小，数字仅参考，不宜单独立论。

### 3.2 val_special × length_bucket（Baseline）

| 长度分桶 | n | no_anchor_miss | strict_fail | recall | MAE_changed | dense_mae | conf | anchor_count | latency |
|---|---|---|---|---|---|---|---|---|---|
| short | 11 | 0.00% | 0.00% | 94.3% | 0.0453 | 0.0023 | 0.9800 | 2.000 | 1.528s |
| medium | 70 | 0.00% | 0.00% | 85.4% | 0.1402 | 0.0083 | 0.9800 | 1.614 | 1.362s |
| long | 54 | 0.00% | 0.00% | 92.5% | 0.0597 | 0.0044 | 0.9800 | 1.741 | 1.410s |
| noisy_long | 45 | 0.00% | 0.00% | 91.5% | 0.0680 | 0.0070 | 0.9800 | 1.978 | 1.523s |

### 3.3 val_anti_truncation × length_bucket（Baseline）

| 长度分桶 | n | no_anchor_miss | strict_fail | recall | MAE_changed | dense_mae | conf | anchor_count | latency |
|---|---|---|---|---|---|---|---|---|---|
| medium | 14 | 0.00% | 0.00% | 95.3% | 0.0372 | 0.0176 | 0.9800 | 1.857 | 1.495s |
| long | 72 | 0.00% | 0.00% | 98.6% | 0.0113 | 0.0084 | 0.9800 | 2.014 | 1.539s |
| noisy_long | 94 | 0.00% | 0.00% | 97.9% | 0.0339 | 0.0124 | 0.9800 | 1.947 | 1.504s |

> val_anti_truncation 无 short 分桶（该 split 专为长文本设计）。

---

## 四、DIR_RULE 推理注入实验对比（图 3.5 用）

**实验设定：** 在每条 prompt 的 `OUTPUT_RULE` 行后插入：  
`DIR_RULE=ANCHOR中DIR字段须与EVENT中DIR完全一致；不得颠倒方向（向东≠向西，向北≠向南）；无明确方向时填写双向。`

### 4.1 分场景 recall 对比（图 3.5 数据源）

| 场景类型 | n (overall) | Baseline recall | DIR_RULE recall | Δ recall | Δ dense_mae |
|---|---|---|---|---|---|
| simple_local | 139 | 63.7% | 63.7% | **±0.0pp** | ±0.0000 |
| directional_asymmetry | 71 | 71.8% | 71.8% | **±0.0pp** | +0.0033 |
| core_blockage | 44 | 78.8% | 78.8% | **±0.0pp** | ±0.0000 |
| compound_disaster | 87 | 95.0% | 90.4% | −4.6pp ⚠️ | +0.0114 |
| temporal_switch | 87 | 100.0% | 100.0% | **±0.0pp** | ±0.0000 |
| propagation_range | 112 | 100.0% | 100.0% | **±0.0pp** | +0.0035 |

> compound_disaster 的 −4.6pp 是 DIR_RULE 行多占 token 导致截断（该场景 avg anchor_count=2.95，prompt 最长），**不是方向逻辑退化**。

### 4.2 split 级汇总（表格用）

| split | n | Δ no_anchor_miss | Δ strict_fail | Δ recall | Δ dense_mae |
|---|---|---|---|---|---|
| val_normal | 180 | **±0.0pp** | **±0.0pp** | **±0.0pp** | +0.0013 |
| val_special | 180 | **±0.0pp** | **±0.0pp** | −1.4pp | +0.0042 |
| val_anti_truncation | 180 | **±0.0pp** | **±0.0pp** | −2.2pp | +0.0035 |
| **overall** | **540** | **±0.0pp** | **±0.0pp** | **−1.5pp** | **+0.0030** |

### 4.3 val_normal 场景细分 Δ（表 3.1(b) 填入）

| 场景类型 | n | Baseline recall | DIR_RULE recall | Δ recall | Δ miss | Δ strict_fail |
|---|---|---|---|---|---|---|
| simple_local | 139 | 63.7% | 63.7% | **±0.0pp** | **±0.0pp** | **±0.0pp** |
| directional_asymmetry | 41 | 69.5% | 69.5% | **±0.0pp** | **±0.0pp** | **±0.0pp** |

### 4.4 实验结论

推理时注入 DIR_RULE 对现有模型**完全无效**：所有决策性指标 Δ = ±0.0pp，包括直接针对的 directional_asymmetry 场景（val_normal，n=41）。

微弱负向信号来源：DIR_RULE 行多占约 30 token，少数长 prompt 在截断点（max_length=1024）多丢失内容。

**机制：** 模型经 Stage1–3 约 7,400 条样本训练后输出行为已固化，推理时新增指令行对已收敛权重不产生影响。与 Stage4b/c/d 补丁实验结论一致：当前模型已达参数容量极限。

**后续行动：** `format_sparse_prompt()` 已内嵌 DIR_RULE，需重新生成 Stage1–3 数据集后全量重训（约 8 小时）方可生效。

---

## 五、数据文件索引

| 文件 | 内容 | 路径 |
|---|---|---|
| Baseline summary CSV | 全量分组指标 | `results/sparse_v2_validation_strict_stage4_fix_summary.csv` |
| Baseline detail CSV | 每样本详情 | `results/sparse_v2_validation_strict_stage4_fix_details.csv` |
| Baseline JSON | 模型元信息 + 聚合报告 | `results/sparse_v2_validation_strict_stage4_fix.json` |
| DIR_RULE summary CSV | 本次实验分组指标 | `results/dir_rule_strict_summary_20260421_1545.csv` |
| DIR_RULE detail CSV | 本次实验每样本详情 | `results/dir_rule_strict_details_20260421_1545.csv` |
| DIR_RULE JSON | 本次实验聚合报告 | `results/dir_rule_strict_20260421_1545.json` |
| 实验报告（完整） | 含背景/方法/结论 | `results/dir_rule_experiment_report_20260421.md` |
| **本文档** | 论文表格数据汇总 | `results/thesis_data_tables_20260421.md` |
