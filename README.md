# myproject2026

基于大语言模型、LoRA 微调、GAT 补全和 SUMO 仿真的城市交通路径规划实验系统。项目围绕灾害、拥堵、局部封锁、方向不对称等交通场景，比较 Dijkstra、Rule-A*、DQN、GCN-Weight、Qwen/R1 原始模型、LoRA 模型、Sparse-LoRA 以及 Sparse-LoRA+GAT 等方法在路径质量、约束符合率、覆盖率和推理耗时上的表现。

## 项目概述

本项目用于本科毕设/实验验证场景，主要目标是：

- 构建 96 条有向边的 SUMO 城市路网实验环境。
- 生成交通路径规划监督数据和稀疏边权数据。
- 使用 Qwen2.5 与 DeepSeek-R1 Distill Qwen 进行 LoRA/Sparse-LoRA 微调。
- 使用 GAT 对稀疏输出进行边权补全，提高可覆盖路段比例。
- 在多种交通异常场景下对比传统算法、强化学习基线和 LLM 方法。
- 提供 Streamlit 可视化界面，用于展示路网、场景、路径和实验指标。

## 已上传内容

仓库包含项目运行和复现实验所需的主要代码、配置、数据摘要、实验结果和轻量模型文件：

- `code/`：核心交通仿真、路径规划、场景构造、指标计算、UI 和推理逻辑。
- `compare/`：多方法对比实验脚本，包括经典基线、LLM 基线和消融实验。
- `rl_baseline/`：DQN/PPO 强化学习基线脚本。
- `SUMO/`：SUMO 路网、路由和仿真配置文件。
- `dataset*/`：训练、验证和不同阶段的监督数据。
- `results/`：实验结果、表格、报告、可视化 HTML 和论文相关输出。
- `model_lora*/`：LoRA adapter、tokenizer 配置和 smoke test 结果。
- `sft_config_*.yaml`：不同模型和阶段的 LoRA/Sparse-LoRA 训练配置。
- `run_sft_lora.py`：通用 LoRA 微调脚本。
- `merge_lora.py`：LoRA 合并脚本。
- `generate_dataset_*.py`、`generate_stage4_sparse_v2.py`：数据生成脚本。
- `validate_*.py`、`audit_*.py`：验证和一致性审计脚本。
- `start_viz.sh`：Streamlit 可视化服务启动脚本。
- `CLEAN_EXPERIMENT_REPORT_20260401.md`：干净实验报告和关键结论。

## 未上传的大文件

为了确保 GitHub 普通仓库可以正常 push，本仓库没有上传以下运行缓存和超大训练产物：

- `.cache/`、`.autodl/`、`.Trash-0/`、`tmp/` 等本地缓存和回收站文件。
- `logs/` 和运行日志。
- `checkpoint-*` 训练检查点。
- `optimizer.pt`、`scheduler.pt` 等优化器状态。
- `DeepSeek-R1-1.5B/model.safetensors`、`Qwen2.5-1.5B-Instruct/model.safetensors`。
- `model_merged_*/model.safetensors` 等 3GB 级合并模型权重。

这些文件体积从数百 MB 到数十 GB 不等，普通 GitHub 仓库会拒收超过 100MB 的单文件。如果需要长期保存完整模型权重，建议使用 Hugging Face Hub、Git LFS 或对象存储。

## 环境依赖

项目没有固定的 `requirements.txt`，运行时需要根据任务安装对应依赖。常用依赖包括：

```bash
pip install streamlit folium streamlit-folium plotly pandas numpy networkx
pip install torch transformers datasets peft pyyaml
pip install sumolib traci
```

如果要运行 SUMO 仿真，需要系统安装 SUMO，并配置：

```bash
export SUMO_HOME=/path/to/sumo
```

如果要进行 LoRA 训练，需要 GPU 环境。`run_sft_lora.py` 会检查 CUDA，并优先使用 BF16 或 FP16。

## 快速启动可视化

启动 Streamlit UI：

```bash
bash start_viz.sh start --port 8504 --mode demo_mode
```

查看状态：

```bash
bash start_viz.sh status --port 8504
```

查看日志：

```bash
bash start_viz.sh logs --port 8504 -n 100
```

停止服务：

```bash
bash start_viz.sh stop --port 8504
```

主要入口应用为：

```text
code/app_fixed.py
```

## 运行对比实验

经典方法对比：

```bash
python compare/run_compare.py --mode experiment_mode
```

加入 LLM 方法对比：

```bash
python compare/run_compare.py --mode experiment_mode --llm
```

SmallNet LLM 对比：

```bash
python code/smallnet_baseline.py --all-llm
```

稀疏输出验证：

```bash
python validate_sparse_v2.py
python validate_sparse_v2_strict.py
```

GAT v2 验证：

```bash
python train_gat_v2.py
python validate_gat_v2.py
```

## LoRA 训练

Qwen LoRA：

```bash
python run_sft_lora.py --config sft_config_lora_qwen.yaml
```

DeepSeek-R1 Distill Qwen LoRA：

```bash
python run_sft_lora.py --config sft_config_lora_r1.yaml
```

Sparse-LoRA：

```bash
python run_sft_lora.py --config sft_config_lora_sparse.yaml
```

Sparse-LoRA v2 分阶段训练：

```bash
python run_sft_lora.py --config sft_config_lora_sparse_v2_stage1.yaml
python run_sft_lora.py --config sft_config_lora_sparse_v2_stage2.yaml
python run_sft_lora.py --config sft_config_lora_sparse_v2_stage4.yaml
```

## 主要实验结论

根据 `CLEAN_EXPERIMENT_REPORT_20260401.md`：

- Sparse-LoRA 是表现最强的 LLM 方法之一，平均 SUMO 时间接近 Rule-A*，约束符合率更高。
- Sparse-LoRA+GAT 显著提升覆盖率，尤其在复杂绕行场景中更有价值。
- Rule-A* 仍然是非常强的传统规则基线，速度快且稳定。
- R1-LoRA 在部分小路网解析率较高，但大路网推理耗时明显偏高。
- GAT 的主要贡献是补全缺失边，在复杂中央封锁场景中能明显改善路径质量。

## 仓库状态说明

当前 Git 提交只包含可以进入普通 GitHub 仓库的有用文件。超大模型权重和训练中间状态已通过 `.gitignore` 排除，避免 push 失败。

如果后续需要恢复完整实验环境，需要额外准备：

- Qwen2.5-1.5B-Instruct 基座模型权重。
- DeepSeek-R1-Distill-Qwen-1.5B 基座模型权重。
- 已合并模型的 `model.safetensors`。
- 必要的训练 checkpoint 或 optimizer state。
