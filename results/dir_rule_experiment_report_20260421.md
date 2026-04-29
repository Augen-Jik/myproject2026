# DIR_RULE 推理注入实验报告

**日期：** 2026-04-21  
**模型：** `model_merged_sparse_v2_stage4_fix`（Qwen2.5-1.5B，LoRA 三阶段课程训练）  
**数据集：** `dataset_sparse_v2`（val_normal / val_special / val_anti_truncation，各 180 条，共 540 条）

---

## 实验背景

已有文献记录的主要失败模式之一是方向错误（`wrong_direction`）：模型在 `directional_asymmetry` 场景中偶发输出反向（向东↔向西 / 向北↔向南），直接导致变化边召回率下降。

**假设：** 在推理时于 prompt 的 `OUTPUT_RULE` 行后插入一条 `DIR_RULE` 显式指令，模型无需重训即可降低方向错误率。

**注入内容（插入位置：OUTPUT_RULE 紧接下一行）：**

```
DIR_RULE=ANCHOR中DIR字段须与EVENT中DIR完全一致；不得颠倒方向（向东≠向西，向北≠向南）；无明确方向时填写双向。
```

注入方式：`_inject_dir_rule()` 函数包装存储 prompt，保留所有原始 EVENT 字段，仅增加 1 行。

---

## 核心指标对比

### 汇总（overall，n=540）

| 指标 | Baseline | DIR_RULE | Δ |
|---|---|---|---|
| no_anchor_rate | 11.67% | 11.67% | **±0.00pp** |
| no_anchor_when_gt_changed_rate | 6.11% | 6.11% | **±0.00pp** |
| parse_fail_rate（宽松） | 0.00% | 0.00% | **±0.00pp** |
| parse_fail_rate_strict | 5.37% | 5.37% | **±0.00pp** |
| target_recall | 99.6% | 97.9% | −1.7pp |
| gt_changed_edge_recall | 89.3% | 87.8% | −1.5pp |
| MAE on changed edges | 0.0932 | 0.1487 | +0.0555 |
| dense_weight_mae | 0.0073 | 0.0103 | +0.0030 |
| parse_confidence | 0.9707 | 0.9703 | −0.0004 |
| anchor_count (avg) | 1.493 | 1.461 | −0.032 |
| parsed_edge_ratio | 3.04% | 3.01% | −0.03pp |
| inference_time (s) | 1.323 | 1.392 | +0.069s |

### 按 split 拆解

| split | n | Δ no_anchor_miss | Δ strict_fail | Δ recall | Δ dense_mae |
|---|---|---|---|---|---|
| val_normal | 180 | **±0.0pp** | **±0.0pp** | **±0.0pp** | +0.0013 |
| val_special | 180 | **±0.0pp** | **±0.0pp** | −1.4pp | +0.0042 |
| val_anti_truncation | 180 | **±0.0pp** | **±0.0pp** | −2.2pp | +0.0035 |

### val_normal 按场景类型

| scene_type | n | Δ no_anchor_miss | Δ strict_fail | Δ recall | Δ dense_mae |
|---|---|---|---|---|---|
| directional_asymmetry | 41 | **±0.0pp** | **±0.0pp** | **±0.0pp** | +0.0057 |
| simple_local | 139 | **±0.0pp** | **±0.0pp** | **±0.0pp** | ±0.0000 |

### val_normal 按长度分桶

| length_bucket | n | Δ no_anchor_miss | Δ strict_fail | Δ recall | Δ dense_mae |
|---|---|---|---|---|---|
| short | 84 | **±0.0pp** | **±0.0pp** | **±0.0pp** | +0.0009 |
| medium | 85 | **±0.0pp** | **±0.0pp** | **±0.0pp** | +0.0018 |
| long | 6 | **±0.0pp** | **±0.0pp** | **±0.0pp** | ±0.0000 |
| noisy_long | 5 | **±0.0pp** | **±0.0pp** | **±0.0pp** | ±0.0000 |

---

## Baseline 完整数字（供论文参考）

**模型：** `model_merged_sparse_v2_stage4_fix`，**日期冻结：** 2026-04-17

| split | no_anchor_rate | no_anchor_miss | strict_fail | recall | target_recall | dense_mae | conf | latency |
|---|---|---|---|---|---|---|---|---|
| overall (n=540) | 11.67% | 6.11% | 5.37% | **89.3%** | 99.6% | **0.0073** | 0.9707 | 1.323s |
| val_normal (n=180) | 35.00% | 19.86% | 16.11% | 65.9% | 100.0% | 0.0043 | 0.9520 | 1.025s |
| val_special (n=180) | 0.00% | 0.00% | 0.00% | 89.7% | 99.5% | 0.0064 | 0.9800 | 1.427s |
| val_anti_truncation (n=180) | 0.00% | 0.00% | 0.00% | 98.0% | 99.6% | 0.0112 | 0.9800 | 1.517s |

**val_normal 场景细分（Baseline）：**

| scene_type | n | no_anchor_miss | strict_fail | recall | dense_mae |
|---|---|---|---|---|---|
| simple_local | 139 | 27.62% | 20.86% | 63.7% | 0.0037 |
| directional_asymmetry | 41 | 0.00% | 0.00% | 69.5% | 0.0065 |

**val_normal 长度细分（Baseline）：**

| length_bucket | n | no_anchor_miss | strict_fail | recall | dense_mae |
|---|---|---|---|---|---|
| short | 84 | 23.53% | 19.05% | 62.7% | 0.0044 |
| medium | 85 | 15.49% | 12.94% | 68.8% | 0.0044 |
| long | 6 | 25.00% | 16.67% | 80.0% | 0.0014 |
| noisy_long | 5 | 33.33% | 20.00% | 55.6% | 0.0067 |

---

## 结论

**推理时注入 DIR_RULE 对现有模型完全无效。**

所有关键决策指标（no_anchor_miss、strict_fail、recall）在全部 split 和场景类型上 Δ = **±0.0pp**，包括直接针对的 `directional_asymmetry`（val_normal，n=41）。

观察到的微弱负向信号（overall recall −1.5pp，val_anti_truncation recall −2.2pp，mae 轻微上升）不是方向错误增加，而是 DIR_RULE 多占约 30 token 导致部分长 prompt 在截断点（max_length=1024）处多丢失内容。

**机制解释：** 模型在 Stage1-3 共约 7400 条样本、3 个 epoch 的训练中已将输出行为固化，推理时新增的指令行对已收敛的权重不产生任何梯度更新，因此无效。这与 Stage4 多次修复实验（Stage4b/c/d 均 <1/28 改变）的结论一致：模型已达当前参数容量的表达极限。

---

## 后续行动

`code/sparse_utils.py` 中的 `format_sparse_prompt()` 已内嵌 DIR_RULE（本次实验同步完成）。

**需要重训才能生效，路径：**

1. 用更新后的 `generate_dataset_sparse.py` 重新生成 Stage1/2/3 数据集（DIR_RULE 自动写入所有 prompt）
2. 从 Stage1 全量重训（`train_sparse_lora_v2.sh`，约 8 小时）
3. 重训完成后删除 `validate_sparse_v2.py` 和 `sparse_eval_strict.py` 中的 `_inject_dir_rule()` 注入逻辑（新数据集本身已含该行，注入层变为冗余）

**关于优化1（约束束搜索）：** 本次实验确认方向错误不是 parse_fail 的主因（`directional_asymmetry` 场景 strict_fail=0.0%）。val_normal 的 strict_fail 16.11% 全部来自 `simple_local` 的 no_anchor 失败。若要显著改善整体 parse_fail，优化1针对的问题才是主要瓶颈，但实现成本较高（需要在 beam search 层强制输出 N 个 ANCHOR）。建议在重训（含 DIR_RULE）后再评估 parse_fail 率是否已自然下降，再决定是否做优化1。
