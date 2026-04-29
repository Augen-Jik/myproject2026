# myproject2026

面向自然语言动态交通约束的稀疏语义解析与路径规划实验系统。

本项目对应论文《面向自然语言动态交通约束的稀疏语义解析与路径规划方法研究》，提出以**结构化时空增强输入 + 稀疏语义锚点输出 + 经典搜索 + SUMO 闭环验证**为主线的混合式框架，在 96 条有向边的郑州金水区路网上与 Dijkstra、Rule-A\*、DQN、GCN-Weight 等基线方法进行完成状态感知的闭环对比。

---

## 目录

- [核心贡献](#核心贡献)
- [主要实验结论](#主要实验结论)
- [项目结构](#项目结构)
- [已上传内容与未上传大文件](#已上传内容与未上传大文件)
- [环境依赖](#环境依赖)
- [快速启动](#快速启动)
- [数据生成](#数据生成)
- [LoRA 训练（Sparse-LoRA-v2 三阶段课程训练）](#lora-训练sparse-lora-v2-三阶段课程训练)
- [运行对比实验](#运行对比实验)
- [严格评估与消融实验](#严格评估与消融实验)
- [SUMO 闭环验证](#sumo-闭环验证)
- [可视化界面](#可视化界面)

---

## 核心贡献

**贡献 1 — 结构化时空增强输入机制**

将自然语言交通描述拆解为 `ROAD | DIR | RANGE | LEVEL | TIME | PROPAGATION | CONFLICT` 七维结构化字段，使时空依赖从隐式语言知识变为显式任务输入。消融实验表明，该机制使约束遵守率提升 **17.25 个百分点**（68.23% → 50.98%，去除后下降）。

**贡献 2 — 稀疏语义锚点输出范式**

将 LLM 从全图边权生成器（96 条边全量输出）重构为局部异常锚点提取器，仅输出偏离正常状态的关键异常边。在抗截断测试中，`noisy_long` 场景截断率从 **100%（全量输出）降至 0%**。

**贡献 3 — 严格评估协议与 SUMO 闭环验证框架**

引入 `changed-edge recall`、`NO_ANCHOR 漏报率`、`parse_fail_rate` 等边级指标替代传统平均 MAE，并在固定场景、固定种子的 SUMO 微观仿真协议下进行闭环验证，区分成功到达与评估窗口截断样本。

---

## 主要实验结论

以下结论基于 **Live SUMO 6 场景完成状态感知协议**（seeds 101–505）：

| 方法 | 行程时间 (s) | 约束率 (%) | 完成率 (%) | 截断次数 |
|---|---:|---:|---:|---:|
| Dijkstra | 1276.12 | 23.91 | 83.33 | 1 |
| Rule-A\* | 1257.12 | 69.78 | **100.00** | 0 |
| DQN | 1796.57 | 69.78 | 50.00 | 3 |
| GCN-Weight | 1471.90 | 57.30 | 83.33 | 1 |
| **Sparse-LoRA-v2（本文）** | **1257.33** | 67.25 | 83.33 | 1 |
| Sparse-LoRA+GAT（扩展） | 1257.33 | 67.25 | 83.33 | 1 |

> **结论说明**：all-run 口径下 Sparse-LoRA-v2 行程时间（1257.33 s）与 Rule-A\*（1257.12 s）基本持平，但完成率低于 Rule-A\*（83.33% vs 100%）。在相同 5 个成功到达场景上 arrived-only 均值为 1028.78 s，Rule-A\* 对应为 1077.94 s（约降低 4.6%），但不能作为整体优于规则基线的证据。本文方法的主要价值在于**可学习的语义解析接口**和**抗截断输出稳定性**。

> Rule-A\* 是最强的非学习基线，速度快、稳定性高、完成率 100%，是本文方法在完成率上尚未超越的对象。

**消融结果（独立消融场景协议，不可与主表直接比较）**：

| 变体 | 行程时间 (s) | 约束遵守率 (%) |
|---|---:|---:|
| No-SpatioTemporal（去除结构化字段） | 542.33 | 50.98 |
| Sparse-LoRA-v2 完整 | 511.15 | 68.23 |

**抗截断结果**：

| 方法 | short | noisy\_long |
|---|---:|---:|
| Qwen-LoRA（全量输出） | 0% | 100% |
| R1-LoRA（全量输出） | 0% | 100% |
| Sparse-LoRA-v2 | 0% | **0%** |

---

## 项目结构

```
myproject2026/
├── code/                          # 核心模块
│   ├── app_fixed.py               # Streamlit UI 主入口
│   ├── baselines.py               # 经典算法与基线实现（Dijkstra / Rule-A* / GCN-Weight）
│   ├── scenarios.py               # 6 类实验场景定义与参数
│   ├── sim_eval_protocol.py       # SUMO 闭环评估协议
│   ├── sumo_runner.py             # SUMO 仿真调用接口
│   ├── routing.py                 # A* / Dijkstra 路由求解
│   ├── config.py                  # 全局配置
│   ├── inference.py               # LLM 推理与锚点解析
│   ├── sparse_eval_strict.py      # 严格边级评估（changed-edge recall 等）
│   ├── sparse_utils.py            # 稀疏锚点解析工具
│   ├── gat_smoother.py            # GAT 图补全模块（探索性扩展）
│   ├── eval_metrics.py            # 约束遵守率计算
│   ├── traffic_rules.py           # Rule-A* 关键词-路阻系数映射
│   ├── congestion_propagation.py  # 拥堵传播建模
│   └── ...
├── compare/                       # 多方法对比实验脚本
│   ├── run_live_scene_profile_main_table.py  # 主表对比（最终冻结协议）
│   ├── run_no_spatiotemporal_ablation.py     # 时空增强输入消融
│   ├── run_spatiotemporal_prompt_ablation.py
│   └── ...
├── rl_baseline/                   # 强化学习基线
│   ├── run_dqn_fixed_baseline.py
│   └── run_ppo_quick_baseline.py
├── SUMO/                          # SUMO 仿真环境
│   ├── config/my_config.sumocfg
│   ├── net/my_net.net.xml         # 96 条有向边路网
│   ├── net/urban_edges.xml
│   ├── net/urban_nodes.xml
│   └── routes.rou.xml
├── dataset_sparse_v2/             # Sparse-LoRA-v2 主训练数据（三阶段 + 验证集）
│   ├── stage1/train, eval
│   ├── stage2/train, eval
│   ├── stage3/train, eval
│   ├── train/, eval/              # 全量合并数据
│   ├── val_normal/                # 常规验证集（含 simple_local、directional_asymmetry）
│   ├── val_special/               # 复杂场景验证集
│   ├── val_anti_truncation/       # 抗截断验证集
│   └── summary.json
├── dataset_sparse_v2_stage4/      # Stage4 定向修复数据
├── dataset_sparse_v2_stage4b/     # Stage4b 迭代数据（负结果，见附录 A5）
├── dataset_sparse_v2_stage4c/
├── dataset_sparse_v2_stage4d_micro/
├── dataset_simple_local_relief_pack/  # simple_local 标注修复数据（Policy A，待重训）
├── model_lora_sparse_v2_stage4_fix/   # 主方法最终采用的 LoRA adapter（71 MB）
├── model_lora_sparse_v2_stage1/       # Stage1 adapter
├── model_lora_sparse_v2_stage2/       # Stage2 adapter
├── model_lora_sparse_v2_stage4/       # Stage4 adapter
├── model_lora_sparse_v2_stage4b_fix/  # Stage4b adapter（负结果）
├── model_lora_sparse_v2_stage4c_fix/  # Stage4c adapter（负结果）
├── model_lora_sparse_v2_stage4d_micro/ # Stage4d adapter（负结果）
├── model_lora_sparse/             # 早期 Sparse-LoRA（全量输出）adapter
├── model_lora_qwen/               # 早期 Qwen CoT LoRA adapter
├── model_lora_r1/                 # 早期 R1 LoRA adapter
├── Qwen2.5-1.5B-Instruct/        # 基座模型配置（权重不含，见下文）
├── DeepSeek-R1-1.5B/             # 基座模型配置（权重不含，见下文）
├── results/
│   ├── final_frozen_20260424_completion_aware/  # 最终冻结实验结果
│   │   ├── method_comparison_final.csv
│   │   ├── completion_aware_stats.csv / .md
│   │   ├── ablation_study.csv
│   │   ├── table1_method_comparison_final.tex
│   │   └── FREEZE_NOTICE.md
│   ├── dir_rule_experiment_report_20260421.md   # DIR_RULE 推理注入负结果
│   ├── final_patch_negative_results.md          # Stage4 patch 负结果汇总
│   └── paper_ready/                             # 论文稿件
├── run_sft_lora.py                # 通用 LoRA 微调入口
├── generate_dataset_sparse.py     # Sparse 数据族生成脚本
├── generate_dataset_cot.py        # CoT 数据族生成脚本（早期探索）
├── generate_stage4_sparse_v2.py   # Stage4 定向修复数据生成
├── validate_sparse_v2.py          # 稀疏输出宽松验证
├── validate_sparse_v2_strict.py   # 稀疏输出严格边级验证
├── train_gat_v2.py / validate_gat_v2.py  # GAT 补全训练与验证
├── merge_lora.py                  # LoRA 权重合并
├── sft_config_lora_sparse_v2_stage1.yaml  # 训练配置
├── sft_config_lora_sparse_v2_stage2.yaml
├── sft_config_lora_sparse_v2_stage3.yaml
├── sft_config_lora_sparse_v2_stage4_fix.yaml
├── start_viz.sh                   # Streamlit 服务启动脚本
├── CLEAN_EXPERIMENT_REPORT_20260401.md  # 阶段性实验报告
└── .gitignore
```

---

## 已上传内容与未上传大文件

**已上传（仓库内）**

- 全部源代码（`code/`、`compare/`、`rl_baseline/`）
- 数据集 Parquet 文件（`dataset_sparse_v2/` 各阶段训练/验证分片）
- LoRA adapter 权重（每个 ~71 MB，**主方法 `model_lora_sparse_v2_stage4_fix/adapter_model.safetensors`**）
- 训练配置文件（`sft_config_*.yaml`）
- SUMO 路网与仿真配置（`SUMO/`）
- 最终冻结实验结果与论文相关输出（`results/`）
- 基座模型的配置、tokenizer 等轻量文件（不含模型权重）

**未上传（超出 GitHub 限制或为本地缓存）**

| 类型 | 说明 |
|---|---|
| `Qwen2.5-1.5B-Instruct/model.safetensors` | ~3.1 GB，需从 Hugging Face 下载 |
| `DeepSeek-R1-1.5B/model.safetensors` | ~3.1 GB，需从 Hugging Face 下载 |
| `model_merged_*/model.safetensors` | 合并后完整模型，单文件 >3 GB |
| `checkpoint-*/optimizer.pt` 等 | 训练中间状态，体积大且不用于推理 |
| `.cache/`、`logs/`、`tmp/` | 运行缓存与日志 |

> 如需完整推理，只需下载对应基座模型权重，LoRA adapter 已包含在仓库内。

---

## 环境依赖

项目无固定 `requirements.txt`，常用依赖如下：

```bash
# 路径规划与仿真
pip install networkx sumolib traci

# 数据处理与可视化
pip install pandas numpy pyarrow plotly streamlit folium streamlit-folium

# 模型训练与推理
pip install torch transformers datasets peft pyyaml trl

# 可选：加速训练
pip install accelerate bitsandbytes
```

配置 SUMO 环境变量（需系统安装 SUMO ≥ 1.18）：

```bash
export SUMO_HOME=/usr/share/sumo   # 根据实际安装路径调整
```

LoRA 训练需要 GPU（建议 24 GB+ 显存）。推理仅需约 4 GB 显存（Qwen2.5-1.5B + LoRA）。

---

## 快速启动

**下载基座模型权重**（选其一）：

```bash
# Qwen2.5-1.5B-Instruct（主方法使用）
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir Qwen2.5-1.5B-Instruct

# DeepSeek-R1-Distill-Qwen-1.5B（早期探索用）
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --local-dir DeepSeek-R1-1.5B
```

**快速验证主方法（stage4_fix）**：

```bash
# 严格边级指标验证（n=540，changed-edge recall / NO_ANCHOR 漏报率 等）
python validate_sparse_v2_strict.py \
    --adapter model_lora_sparse_v2_stage4_fix \
    --base_model Qwen2.5-1.5B-Instruct

# 宽松验证（parse_fail_rate / dense MAE）
python validate_sparse_v2.py \
    --adapter model_lora_sparse_v2_stage4_fix \
    --base_model Qwen2.5-1.5B-Instruct
```

---

## 数据生成

生成 Sparse-LoRA-v2 主训练数据（三阶段渐进式覆盖，约 7400 条）：

```bash
python generate_dataset_sparse.py
```

生成 Stage4 定向修复数据（约 800–1000 条，用于补充 Stage3 遗留失败模式）：

```bash
python generate_stage4_sparse_v2.py
```

早期 CoT 全量输出数据（约 5000 条，已被稀疏格式替代，仅供参考）：

```bash
python generate_dataset_cot.py
```

---

## LoRA 训练（Sparse-LoRA-v2 三阶段课程训练）

主方法采用课程学习策略，LoRA 权重在阶段间链式继承（Stage1 → Stage2 → Stage3 → Stage4_fix）。

| 阶段 | 数据规模 | 场景分布 | 学习率 | 输出目录 |
|---|---|---|---|---|
| Stage1 | 2000/260 | simple_local + directional_asymmetry | 5e-5 | `model_lora_sparse_v2_stage1/` |
| Stage2 | 4200/420 | 全 6 种场景均衡 | 4e-5 | `model_lora_sparse_v2_stage2/` |
| Stage3 | 1200/180 | 复杂场景为主 | 3.5e-5 | *(继承入 stage4)* |
| Stage4_fix | ~800 | 定向修复 | — | `model_lora_sparse_v2_stage4_fix/` |

```bash
# Stage1
python run_sft_lora.py --config sft_config_lora_sparse_v2_stage1.yaml

# Stage2（从 stage1 继承）
python run_sft_lora.py --config sft_config_lora_sparse_v2_stage2.yaml

# Stage3（从 stage2 继承）
python run_sft_lora.py --config sft_config_lora_sparse_v2_stage3.yaml

# Stage4_fix（从 stage3 继承，最终采用模型）
python run_sft_lora.py --config sft_config_lora_sparse_v2_stage4_fix.yaml
```

LoRA 超参数：`r=16, alpha=32, dropout=0.05`，目标模块覆盖 `q/k/v/o_proj` 及 `gate/up/down_proj`，可训练参数量约为基座模型的 **2.4%**。

早期探索模型（已被稀疏格式替代，不纳入主表）：

```bash
python run_sft_lora.py --config sft_config_lora_qwen.yaml    # CoT 全量输出
python run_sft_lora.py --config sft_config_lora_r1.yaml      # R1 LoRA
```

---

## 运行对比实验

**主表：Live SUMO 6 场景完成状态感知协议**（最终冻结实验）：

```bash
python compare/run_live_scene_profile_main_table.py
```

结果保存至 `results/final_frozen_20260424_completion_aware/`。

**单方法运行**：

```bash
# 经典算法对比（Dijkstra / Rule-A* / GCN-Weight）
python compare/run_compare.py

# 加入 LLM 方法
python compare/run_compare.py --llm

# DQN 强化学习基线（简化 5×5 环境，结果不可与主表直接比较）
python rl_baseline/run_dqn_fixed_baseline.py
```

---

## 严格评估与消融实验

**严格边级评估**（changed-edge recall、NO_ANCHOR 漏报率、parse_fail_rate）：

```bash
python validate_sparse_v2_strict.py
```

**时空增强输入消融**（No-SpatioTemporal vs 完整模型）：

```bash
python compare/run_no_spatiotemporal_ablation.py
# 或
python compare/run_spatiotemporal_prompt_ablation.py
```

**抗截断实验**（short / noisy_long 压力档对比）：

```bash
python code/truncation_experiment.py
```

**GAT 图补全验证**（探索性扩展，行程时间无改善，signal_ratio 提升）：

```bash
python train_gat_v2.py
python validate_gat_v2.py
python compare/run_gated_gat_ablation.py
```

---

## SUMO 闭环验证

实验路网：郑州金水区核心区域，**96 条有向边**，权重 0–10（0=畅通，10=严重拥堵）。

6 个主实验场景（seeds 101–505）：

| 场景 | 种子 | 说明 |
|---|---|---|
| normal_baseline | 101 | 正常通行 |
| simple_local | 111 | 单一局部拥堵 |
| directional_asymmetry | 202 | 方向不对称约束 |
| core_blockage | 303 | 核心节点封闭 |
| propagation_range | 404 | 拥堵传播外溢 |
| compound_disaster | 505 | 多事件复合 |

直接运行仿真：

```bash
cd SUMO
sumo-gui -c config/my_config.sumocfg   # 可视化
sumo -c config/my_config.sumocfg       # 无 GUI
```

通过 Python 接口驱动（推荐，用于评估协议）：

```bash
python code/sim_eval_protocol.py --scene compound_disaster --seed 505
```

---

## 可视化界面

启动 Streamlit 可视化服务：

```bash
bash start_viz.sh start --port 8504
```

查看运行状态 / 日志 / 停止：

```bash
bash start_viz.sh status --port 8504
bash start_viz.sh logs --port 8504 -n 100
bash start_viz.sh stop --port 8504
```

主入口：`code/app_fixed.py`，提供路网展示、场景配置、路径对比和实验指标面板。

---

## 关键文件索引

| 文件 | 说明 |
|---|---|
| `model_lora_sparse_v2_stage4_fix/adapter_model.safetensors` | **主方法最终采用的 LoRA adapter** |
| `results/final_frozen_20260424_completion_aware/` | 最终冻结实验数据（主表、消融、runtime 审计） |
| `results/final_frozen_20260424_completion_aware/table1_method_comparison_final.tex` | 论文主表 LaTeX 源码 |
| `CLEAN_EXPERIMENT_REPORT_20260401.md` | 阶段性实验报告与关键决策记录 |
| `results/dir_rule_experiment_report_20260421.md` | DIR_RULE 推理注入负结果报告 |
| `results/final_patch_negative_results.md` | Stage4 四轮 patch 负结果汇总（附录 A5） |
| `code/traffic_rules.py` | Rule-A\* 关键词—路阻系数映射表 |
| `code/sparse_eval_strict.py` | 严格边级评估实现 |
| `code/scenarios.py` | 6 类场景参数定义 |

---

## 仓库说明

- 所有实验结果已冻结于 `results/final_frozen_20260424_completion_aware/`，对应论文主表日期 2026-04-24。
- Stage4 四轮定向修复实验（stage4b/4c/4d_micro）均未形成稳定收益，相关 adapter 和负结果报告保留在仓库中供复现参考。
- `model_lora_qwen`（Qwen CoT 全量输出）适配器已上传，但因推理延迟高（8.60 s/query）和约束率低（34.7%）不作为主方法。
- `model_lora_r1` 适配器已上传，但因推理延迟过高（31.97 s/query）不满足实时部署需求。
- LoRA adapter（~71 MB 每个）通过 Git 直接追踪；基座模型权重（~3 GB）不包含在仓库内，需自行下载。
