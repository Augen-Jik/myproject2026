# stage4d_micro Expected Budget

## Goal
- 本轮是最后一轮 `micro patch`，不再追求 broad coverage，只追求唯一高 ROI 桶 `single_edge_changed_relief__short_plain_or_prefixed` 发生 `no_anchor -> anchor` 位移。
- 同时必须压住 normal-only 假阳性，因此 `hard_negative_guard` 继续保留为训练集的 `25%`，只覆盖 `交通基本正常` / `保持正常通行`。
- 若本轮仍不放行，则按既定策略冻结 `stage4_fix` 并停止 patch。

## Config Snapshot
| Item | Value |
| --- | --- |
| config | `/root/autodl-tmp/sft_config_lora_sparse_v2_stage4d_micro.yaml` |
| init adapter | `/root/autodl-tmp/model_lora_sparse_v2_stage4_fix` |
| output_dir | `/root/autodl-tmp/model_lora_sparse_v2_stage4d_micro` |
| train dataset | `/root/autodl-tmp/dataset_sparse_v2_stage4d_micro/train/data.parquet` |
| eval dataset | `/root/autodl-tmp/dataset_sparse_v2_stage4d_micro/eval/data.parquet` |
| max_seq_length | `1024` |
| learning_rate | `2.0e-5` |
| scheduler | `cosine` |
| warmup_ratio | `0.05` |

## Trainer Step Math
- `train_rows = 100`
- `batch = 4`
- `grad_accum = 3`
- `effective_global_batch = 12`
- `len_dataloader = ceil(100 / 4) = 25` micro-batches / epoch
- `updates_per_epoch = ceil(25 / 3) = 9`
- `planned_epochs = 3.0`
- `expected_max_steps = ceil(3.0 * 9) = 27`
- `expected_epoch_reached = 27 / 9 = 3.0`

## Exposure Budget
- 当前数据集是高度定向的 micro patch mixture：`micro_patch_positive_core=65`、`hard_negative_guard=25`、`minimal_stability_canary=10`。
- 由于 `expected_epoch_reached = 3.0`，单条训练样本的平均理论曝光次数约为 `3.0x`。

| Bucket | Train Rows | Train Share | Expected Example-Sees | Avg Exposure Per Sample |
| --- | --- | --- | --- | --- |
| `micro_patch_positive_core` | `65` | `65%` | `65 * 3.0 = 195` | `3.0x` |
| `hard_negative_guard` | `25` | `25%` | `25 * 3.0 = 75` | `3.0x` |
| `minimal_stability_canary` | `10` | `10%` | `10 * 3.0 = 30` | `3.0x` |

## Required Readout
- `train_rows`: `100`
- `batch`: `4`
- `grad_accum`: `3`
- `updates_per_epoch`: `9`
- `planned_epochs`: `3.0`
- `expected_max_steps`: `27`
- `target phrase 样本平均曝光次数`: `3.0x`
- `hard_negative_guard 平均曝光次数`: `3.0x`

## Why This Fits The Micro-Patch Goal
- 相比 `stage4c`，本轮更窄：train rows 从 `153` 收缩到 `100`，只保留唯一目标桶正例、两类 hard negative 和最小 canary。
- 相比 `stage4c`，本轮不是更大：计划 `27` 个 optimizer steps，低于 `stage4c` 的 `32`，因此预算更克制。
- 相比 `stage4b`，本轮有效更新明显更强：`27` 步远高于 `9` 步，且 `65%` 的样本都直接服务于目标桶位移。
- `cosine` + `warmup_ratio=0.05` 能避免在短窗口里过快衰减；同时保留 `25%` 的 hard negative，继续对冲 normal-only 假阳性风险。

## Final Readout
- `当前配置是否满足 20~32 steps`：`是，expected_max_steps = 27`
- `target bucket 平均曝光是否足够`：`是，约 3.0 次，且目标桶占 train 的 65%`
- `是否具备进入最终训练的条件`：`是，具备`
