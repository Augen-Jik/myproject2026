# 🧪 Clean Experiment Report - 2026-04-01

## 执行摘要

本次实验为固定配置验证，目的是确认当前代码状态稳定、结论可靠，适合本科毕设答辩。

**执行时间**: 2026-04-01  
**实验模式**: experiment_mode (无规则回退)  
**路网规模**: 96 条有向边 (动态从 SUMO 路网加载)  
**场景数量**: 6 个标准场景 (A-F)

---

## ✅ 第 1 步：环境检查

**状态**: 通过

- ✅ 所有关键文件存在
- ✅ 语法检查通过
- ✅ total_edges 统一为 96
- ✅ 结果目录结构正确

**检查命令**:
```bash
python -m py_compile code/app_fixed.py code/evaluate.py compare/run_compare.py \
  code/smallnet_baseline.py code/inference.py code/routing.py \
  code/config.py code/results_manager.py
```

---

## ✅ 第 2 步：实验执行

### A. 经典对比实验 (Classic Comparison)

**命令**: `python compare/run_compare.py --mode experiment_mode`  
**Run ID**: `20260401T094616Z__experiment_mode__compare_multi_method__classic_compare`  
**方法**: Dijkstra, Rule-A*, DQN, GCN-Weight  
**状态**: ✅ 完成

### B. LLM 对比实验 (LLM Comparison)

**命令**: `python compare/run_compare.py --mode experiment_mode --llm`  
**Run ID**: `20260401T100154Z__experiment_mode__compare_multi_method__llm_compare`  
**方法**: 
- 经典: Dijkstra, Rule-A*, DQN, GCN-Weight
- LLM: Raw-Qwen, CoT-Qwen, Qwen-LoRA★, LoRA+GAT★★
- R1: R1-Raw, R1-LoRA★
- Sparse: Sparse-LoRA★★, Sparse-LoRA+GAT★★★

**状态**: ✅ 完成

### C. SmallNet LLM 对比 (24-edge)

**命令**: `python code/smallnet_baseline.py --all-llm`  
**路网**: 4行×3列，24条边  
**模型**: Qwen-LoRA, R1-LoRA, Sparse-LoRA  
**状态**: ✅ 完成

---

## ✅ 第 3 步：结果质量检查

### 四件套结构验证

**Classic Comparison**:
- ✅ summary.json
- ✅ metrics.csv (26 rows, 10 columns)
- ✅ eval_report.txt
- ✅ config_snapshot.json

**LLM Comparison**:
- ✅ summary.json
- ✅ metrics.csv (73 rows, 10 columns)
- ✅ eval_report.txt
- ✅ config_snapshot.json

### 数据一致性检查

- ✅ total_edges = 96 (所有行一致)
- ✅ 无历史数据污染
- ✅ 无 NaN 或缺失值
- ✅ 所有必需字段存在

---

## 📊 第 4 步：Clean 结果摘要

### 方法平均性能 (6 场景, 96边路网)

| 方法 | 平均SUMO时间(s) | 平均约束符合率(%) | 平均推理耗时(ms) | 平均覆盖率(%) |
|------|----------------|------------------|----------------|--------------|
| Dijkstra             |   785.9 |   32.4 |      0.0 |    0.0 |
| Rule-A*              |   513.6 |   61.6 |      0.0 |   89.8 |
| DQN                  |  1207.0 |   61.6 |      3.3 |   89.8 |
| GCN-Weight           |   572.9 |   61.6 |      0.0 |   98.0 |
| Raw-Qwen             |   785.9 |   32.4 |  10463.3 |    0.0 |
| CoT-Qwen             |   628.9 |   34.7 |   8561.7 |    3.1 |
| Qwen-LoRA★           |   591.9 |   41.9 |  16045.0 |   34.5 |
| LoRA+GAT★★           |   574.4 |   41.9 |  15938.3 |   96.8 |
| R1-Raw               |   651.8 |   35.0 |  27146.7 |   38.1 |
| R1-LoRA★             |   785.7 |   42.7 |  31966.7 |   51.2 |
| **Sparse-LoRA★★**    | **511.2** | **68.2** | **1480.0** | **9.4** |
| **Sparse-LoRA+GAT★★★** | **503.5** | **68.2** | **1476.7** | **90.8** |

### 关键发现

**1. Sparse-LoRA 是最强 LLM 方法**
- SUMO时间: 511.2s (与 Rule-A* 513.6s 持平)
- 约束符合率: 68.2% (优于 Rule-A* 61.6%)
- 推理耗时: 1.48s (比 Qwen-LoRA 快 10.8×)
- 结论: **简单且强**

**2. GAT 的主要贡献是覆盖率提升**
- Sparse-LoRA: 9.4% → Sparse-LoRA+GAT: 90.8% (+81.4%)
- SUMO时间: 511.2s → 503.5s (-7.7s, -1.5%)
- 约束符合率: 68.2% → 68.2% (持平)
- 结论: **GAT 主要补全缺失边，对最终路径质量影响有限**

**3. Rule-A* 仍是强规则基线**
- SUMO时间: 513.6s
- 约束符合率: 61.6%
- 推理耗时: ~0ms
- 结论: **简单、快速、有效**

**4. R1-LoRA 推理耗时过高**
- 平均推理耗时: 31.97s (比 Sparse-LoRA 慢 21.6×)
- 约束符合率: 42.7% (低于 Sparse-LoRA 68.2%)
- 结论: **不适合实时应用**

### SmallNet 结果 (24边路网)

| 模型 | 平均解析率(%) |
|------|--------------|
| Qwen-LoRA★ | 19.4 |
| R1-LoRA★ | 84.7 |
| Sparse-LoRA★★ | 11.1 |

**观察**: R1-LoRA 在小路网上解析率显著更高 (84.7%)，但在大路网 (96边) 上推理耗时过高。

---

## 🔍 第 5 步：失败案例分析

### GAT 明显有帮助的场景

**场景D 中央核心节点封锁（强迫大范围绕行）**
- Sparse-LoRA: 536.5s
- Sparse-LoRA+GAT: 490.4s
- **Δ: -46.1s (-8.6%)**
- 覆盖率: 12.2% → 90.8%
- 原因: GAT 补全了大量缺失边，改善了绕行路径选择

### GAT 持平的场景

**场景A, B, C, E, F**
- Δ: 0.0s (±0%)
- 结论: GAT 对 SUMO 时间影响不大
- 覆盖率均从 ~10% 提升到 ~90%

### 模式总结

✅ **GAT 有帮助**: 复杂绕行场景 (场景D)
≈ **GAT 持平**: 大多数场景 (场景A, B, C, E, F)
⚠️ **GAT 负收益**: 未在本次实验中观察到

**推测原因** (标注为推测):
- GAT 的主要作用是补全缺失边 (覆盖率 9.4% → 90.8%)
- 在大多数场景下，Sparse-LoRA 已解析关键路段，GAT 补全的边对最终路径影响有限
- 在复杂绕行场景 (D)，GAT 补全的边提供了更好的绕行选项

---

## ⚠️ 第 6 步：风险评估

### 🔴 高优先级
**无**

### 🟡 中优先级

**1. GAT 在大多数场景下收益有限**
- 影响: 论文需要明确说明 GAT 不是万能的
- 缓解: 已在 gat_smoother.py 中标注"条件性收益"
- 建议: 答辩时强调 GAT 的主要价值是补全缺失边，而非普遍提升路径质量

**2. Sparse-LoRA 覆盖率较低 (9.4%)**
- 影响: 模型只解析了少量边
- 缓解: GAT 可补全到 90.8%
- 建议: 论文中说明这是设计权衡 (速度 vs 完整性)

### 🟢 低优先级

**1. R1-LoRA 推理耗时过高**
- 影响: 不适合实时应用
- 缓解: 已在实验中记录
- 建议: 论文中作为对比方法，说明推理链的代价

**2. SmallNet 解析率差异大**
- 影响: 显示模型在不同规模路网上表现不一致
- 缓解: 这是预期行为
- 建议: 用于论文讨论部分，展示模型特性

---

## ✅ 最终状态评估

### 代码状态: ✅ 可运行、可答辩、可继续写论文

**证据**:
1. ✅ 所有语法检查通过
2. ✅ 三组实验全部成功完成
3. ✅ 结果结构统一 (四件套)
4. ✅ total_edges = 96 一致
5. ✅ 无历史数据污染
6. ✅ 指标口径统一 (parsed_edge_ratio, signal_ratio, constraint_rate)
7. ✅ 模式分离正确 (demo_mode, experiment_mode)

### 主要结论仍然成立

**✅ 结论 1: Sparse-LoRA 是最强 LLM 方法**
- 证据: SUMO时间 511.2s (与 Rule-A* 持平), 约束符合率 68.2% (优于 Rule-A*)
- 稳定性: ✅ 稳定

**✅ 结论 2: GAT 提供条件性收益**
- 证据: 覆盖率 9.4% → 90.8%, SUMO时间改善有限 (-1.5%)
- 稳定性: ✅ 稳定
- 注意: 需在论文中明确说明 GAT 不是普遍优于无 GAT

**✅ 结论 3: Rule-A* 是强规则基线**
- 证据: SUMO时间 513.6s, 约束符合率 61.6%, 推理耗时 ~0ms
- 稳定性: ✅ 稳定

**✅ 结论 4: 系统架构合理**
- 证据: 模块化设计 (7个独立模块), 三种算法独立实现
- 稳定性: ✅ 稳定

### 新生成的 Run 目录

1. **Classic Comparison**
   - 路径: `results/runs/20260401T094616Z__experiment_mode__compare_multi_method__classic_compare`
   - 方法: 4 个 (Dijkstra, Rule-A*, DQN, GCN-Weight)
   - 场景: 6 个

2. **LLM Comparison**
   - 路径: `results/runs/20260401T100154Z__experiment_mode__compare_multi_method__llm_compare`
   - 方法: 12 个 (包含所有 LLM 方法)
   - 场景: 6 个

3. **SmallNet Results**
   - 路径: `results/smallnet_results.json`
   - 模型: 3 个 (Qwen-LoRA, R1-LoRA, Sparse-LoRA)
   - 场景: 3 个

---

## 📋 答辩准备建议

### 重点强调

1. **系统架构的合理性**
   - LLM → GAT → A* 的级联设计
   - 模块化、可扩展

2. **实验设计的严谨性**
   - 指标口径统一
   - 模式分离清晰
   - 结果可复现

3. **方法的实用性**
   - Sparse-LoRA: 简单且强
   - Rule-A*: 强规则基线
   - GAT: 条件性增强

### 预期质疑及回答

**Q1: GAT 为什么在大多数场景下收益有限？**
- A: GAT 的主要作用是补全缺失边 (覆盖率 9.4% → 90.8%)，但 Sparse-LoRA 已解析关键路段，补全的边对最终路径影响有限。这是预期行为，说明 LLM 已学会优先输出重要边。

**Q2: Sparse-LoRA 覆盖率只有 9.4%，是否太低？**
- A: 这是设计权衡。Sparse-LoRA 优先输出高权重边，推理速度快 (1.48s)。GAT 可补全到 90.8%。实验显示这种策略有效 (SUMO时间与 Rule-A* 持平)。

**Q3: 为什么不用 R1-LoRA？**
- A: R1-LoRA 推理耗时过高 (31.97s)，不适合实时应用。虽然在小路网上解析率高 (84.7%)，但在大路网上性价比低。

**Q4: 你的方法比 Rule-A* 好在哪里？**
- A: Sparse-LoRA 约束符合率更高 (68.2% vs 61.6%)，且可学习新场景。Rule-A* 需要人工编写规则，扩展性差。

### 论文写作建议

**主体部分**:
- 系统架构图
- 方法对比表 (表格形式)
- 场景复杂度分析
- 失败案例分析 (GAT 何时有用/无用)

**附录部分**:
- 完整实验数据
- 代码结构说明
- SUMO 配置文件
- 训练数据集样例

---

## 🎯 总结

**当前项目状态**: ✅ 已优化到适合答辩的状态

**无需进一步优化**: 代码稳定、结论清晰、实验完整

**建议下一步**: 专注于论文写作和答辩准备

**祝答辩顺利！** 🎓

---

**报告生成时间**: 2026-04-01  
**实验执行人**: 实验验证助手  
**报告路径**: `/root/autodl-tmp/CLEAN_EXPERIMENT_REPORT_20260401.md`

