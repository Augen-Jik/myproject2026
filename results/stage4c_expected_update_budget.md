# stage4c Expected Update Budget

## Goal
- 本轮目标不是“轻微补丁”，而是让 `relief_normal_single_edge` 主桶真正发生可见参数位移。
- 已知 stage4b 的核心问题是：`29/29 held-out 主失败样本完全未变`，本轮必须把真实有效更新步数显著抬高。
- 当前策略：使用更集中的 `stage4c_singleedge_fix_v2` mixture，并把训练预算提升到约 `32` 个 optimizer steps。

## Config Snapshot
| Item | Value |
| --- | --- |
| config | `/root/autodl-tmp/sft_config_lora_sparse_v2_stage4c_fix.yaml` |
| init adapter | `/root/autodl-tmp/model_lora_sparse_v2_stage4_fix` |
| output_dir | `/root/autodl-tmp/model_lora_sparse_v2_stage4c_fix` |
| train dataset | `/root/autodl-tmp/dataset_sparse_v2_stage4c/train/data.parquet` |
| eval dataset | `/root/autodl-tmp/dataset_sparse_v2_stage4c/eval/data.parquet` |
| max_seq_length | `1024` |
| learning_rate | `2.0e-5` |
| scheduler | `cosine` |
| warmup_ratio | `0.0625` |

## Trainer Step Math
- `train_rows = 153`
- `batch = 4`
- `grad_accum = 4`
- `effective_global_batch = 16`
- `len_dataloader = ceil(153 / 4) = 39` micro-batches / epoch
- `updates_per_epoch = ceil(39 / 4) = 10`
- `planned_epochs = 3.2`
- `expected_max_steps = ceil(3.2 * 10) = 32`
- `expected_epoch_reached ≈ 32 / 10 = 3.2`

## Why This Is Stronger Than stage4b
| Metric | stage4b | stage4c plan | Delta |
| --- | --- | --- | --- |
| train rows | `168` | `153` | 更小但更集中 |
| target bucket share | `9.5%` (`relief_normal_single_edge`) | `59.48%` (`singleedge_relief_core`) | `6.26x` 更高 |
| Trainer max_steps | `9` | `32` | `3.56x` 更多更新 |
| scheduler pressure | `9` 步内快速线性衰减 | `32` 步 + `cosine` | 明显更缓 |

- stage4b 的问题不是单纯“数据量小”，而是 `9` 步更新把 target signal 冲得太散。
- 本轮虽然 train rows 稍小，但 `singleedge_relief_core` 已占 `59.48%`，而且总更新步数被抬到 `32`，因此主桶会得到显著更高的累计 exposure。

## Mixture Exposure Budget
- 下面 exposure 按 `expected_epoch_reached = 3.2` 估算，含义是每个 component 在整个 run 中的理论 example-sees。
- 对单个 component 内的单条样本来说，平均预计曝光次数约等于 `3.2` 次；若该 component 内部有重复加权样本，则被加权样本会高于平均值。

| Mixture Component | Train Rows | Train Share | Expected Example-Sees | Avg Exposure Per Sample |
| --- | --- | --- | --- | --- |
| `singleedge_relief_core` | `91` | `59.48%` | `91 * 3.2 = 291.2` | `3.2x` |
| `short_range_relief_support` | `23` | `15.03%` | `23 * 3.2 = 73.6` | `3.2x` |
| `relabeled_anomaly_plus_normal_repair` | `16` | `10.46%` | `16 * 3.2 = 51.2` | `3.2x` |
| `guard_minimal` | `16` | `10.46%` | `16 * 3.2 = 51.2` | `3.2x` |
| `multi_edge_relief_tail` | `7` | `4.58%` | `7 * 3.2 = 22.4` | `3.2x` |

## Main-Bucket Interpretation
- `singleedge_relief_core` 的单条样本平均预计曝光次数约为 `3.2` 次，已经达到“至少 3 次左右”的目标。
- 与 stage4b 审计里的目标 bucket 相比，这轮不只是“每条样本从 `0.857x` 提高到 `3.2x`”，更重要的是 bucket 总 share 从 `9.5%` 抬到了 `59.48%`。
- 若按 audit 口径粗算，stage4b 的 `relief_normal_single_edge` 总 example-sees 约为 `13.7`；本轮 `singleedge_relief_core` 约为 `291.2`，target signal 总量约是前者的 `21.3x`。

## Scheduler Rationale
- 当前训练脚本不支持 sample weighting / oversampling，因此本轮主策略不是额外 sampler，而是：
  1. 直接使用已经把 `singleedge_relief_core` 抬到 `59.48%` 的新数据集。
  2. 把真实 `Trainer` 更新步数从 `9` 提高到 `32`。
  3. 把调度改为 `cosine`，并把 warmup 压到 `0.0625`，避免像 stage4b 那样在极短训练窗口里过快衰减。

## Final Readout
- `明显 stronger than stage4b`：`是`
- `single-edge 平均曝光次数是否达到至少 3 次左右`：`是，约 3.2 次`
- `是否具备进入训练的条件`：`是，具备`
