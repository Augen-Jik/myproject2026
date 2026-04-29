# stage4c Training Signal Audit

## Inputs
| Item | Resolved Path |
| --- | --- |
| Config | /root/autodl-tmp/sft_config_lora_sparse_v2_stage4b_fix.yaml |
| Dataset Summary | /root/autodl-tmp/dataset_sparse_v2_stage4b/summary.json |
| Diff JSON | /root/autodl-tmp/results/stage4b_diff_29.json |
| Trainer State | /root/autodl-tmp/model_lora_sparse_v2_stage4b_fix/checkpoint-9/trainer_state.json |
| Matched Log Files | no dedicated stage4b log file found in /root/autodl-tmp/logs |

## Core Finding
- stage4b 之所以最终只有 `9` 个 optimizer/update steps，是因为当前配置是 `train_rows=168`、`per_device_batch=4`、`gradient_accumulation_steps=4`、`num_train_epochs=0.8`，而 Transformers 5.3.0 的 `Trainer` 用的是 `num_update_steps_per_epoch = ceil(len_dataloader / grad_accum)`。
- 具体推导：`len_dataloader = ceil(168 / 4) = 42` micro-batches；`num_update_steps_per_epoch = ceil(42 / 4) = 11`；`max_steps = ceil(0.8 * 11) = 9`。
- `run_sft_lora.py` 里打印总步数的公式是 `len(train_dataset) // (BS*ACCUM)`，会给出 `10 * 0.8 = 8.0` 这类“脚本估算值”；但真实训练步数以 `Trainer` 为准，最终 checkpoint 证实 `max_steps=9`。
- 这 9 步只处理了约 `9 * 16 = 144` 个样本曝光，相当于 `0.8571` 个 epoch；也就是说本次训练理论上仍有约 `24.0` 条训练样本根本没被看到。

## Training Signal Budget
| Metric | Value |
| --- | --- |
| train rows | 168 |
| eval rows | 72 |
| per-device batch | 4 |
| gradient accumulation | 4 |
| effective global batch | 16 |
| micro-batches / epoch | 42 |
| Trainer updates / epoch | 11 |
| requested epochs | 0.8 |
| Trainer max_steps | 9 |
| actual global_step | 9 |
| actual epoch reached | 0.8571 |
| script printed total_steps | 8.0 |
| latest logged loss | 1.5815 |
| latest logged lr | 0.00000375 |

## Mixture Exposure
| Mixture Component | Train Rows | Train Share | Expected Example-Sees In This Run |
| --- | --- | --- | --- |
| anti_regression_guard | 48 | 28.6% | 41.1 |
| relabeled_simple_local_repair | 72 | 42.9% | 61.7 |
| relief_supervision_pack | 48 | 28.6% | 41.1 |

- 一个完整 epoch 中，每个 mixture component 都只会被完整看一遍：relabel `72`、relief `48`、guard `48`。
- 但当前 run 只有 `0.8571` epoch，所以实际理论曝光约为：relabel `61.7`、relief `41.1`、guard `41.1`。
- 按每个 optimizer step 的期望组成估算，当前配置下每一步平均只有 `1.52` 个 `relief_normal_single_edge` 样本，却有 `4.57` 个 guard 样本。

## Target Bucket Signal
- `relief_normal_single_edge` 在 stage4b train 中只有 `16` 条，占全部训练样本的 `9.5%`。
- 由于这次训练不到 1 个 epoch，`relief_normal_single_edge` 在整个训练过程中理论上只被学习了约 `13.7` 次 example-sees；换成单个训练样本视角，就是“平均每条只被看到 `0.857` 次，且 `14.3%` 概率一次都没看到”。
- `hard_negative_no_anchor` guard 有 `10` 条，本次 run 理论曝光约 `8.6` 次，已经接近 single-edge 总曝光量的三分之二。

## Held-out Alignment
| Held-out Subtype | Held-out Failures | Train Bucket Rows | Exact Active-Event Signature Overlap |
| --- | --- | --- | --- |
| relief_normal_single_edge | 20 | 16 | 3 |
| relief_normal_short_range | 7 | 16 | 0 |
| multi_edge_relief | 2 | 16 | 0 |

- `stage4b_diff_29.json` 显示 held-out 主失败 `29/29` 在 pre/post 完全没变，其中 `relief_normal_single_edge` 就占 `20/29`。
- 即使只看 `relief_normal_single_edge`，train 里也只有 `16` 条，而对 held-out `20` 条主失败的 exact active-event signature overlap 只有 `3` 条。这说明 stage4b 不只是更新太少，target coverage 也不够贴脸。

## Why Guard Feels Strong Here
- guard 总量是 `48` 条，占 `28.6%`；它和 relief 总量相等，但却是 `relief_normal_single_edge` 的 `3x`。
- 在只有 9 个 update 的情况下，guard 理论曝光约 `41.1` 次，而 `relief_normal_single_edge` 只有 `13.7` 次。
- 因此 guard 在绝对比例上不算压倒一切，但对一个“只想修 single-edge no-anchor”的微型 patch 来说，当前 guard 比例确实偏强，容易把稀缺更新预算冲淡。

## Trainer Evidence
- step 2: loss=2.0849, lr=0.00001500, epoch=0.1905
- step 4: loss=1.9401, lr=0.00001125, epoch=0.3810
- step 6: loss=1.5283, lr=0.00000750, epoch=0.5714
- step 8: loss=1.5815, lr=0.00000375, epoch=0.7619
- 从 `trainer_state.json` 看，训练是稳定的，没有 NaN/Inf；但 schedule 只有 9 步，学习率从 `1.5e-5` 很快衰减到 `3.75e-6`，参数更新窗口非常短。

## Stage4c Recommendation
| Question | Recommendation | Rationale |
| --- | --- | --- |
| 最小有效更新步数区间 | 24-40 steps | 9 steps 完全不动；24 steps 才开始进入有效区，32-40 steps 更接近当前 mix 下每个 single-edge 样本 3-4 次平均暴露。 |
| 是否提高 epoch | 是 | 若沿用当前 168-row mix，Trainer 实际是 11 updates/epoch；要到 32-40 steps，需要约 2.9-3.6 epochs。 |
| 是否提高 target bucket 权重 | 是 | `relief_normal_single_edge` 当前只占 9.5%，约 1.52 例/update，建议提高到至少 40-60% 的 exposure。 |
| 是否缩小 guard 比例 | 是 | guard 当前占 28.6%，是 single-edge 的 3.0 倍，建议压到 10-20%，只留 anti-regression canary。 |
| 是否拆成更小更集中的 single-edge patch 训练 | 是 | 当前 stage4b 把 9 次 update 分散到 relabel/relief/guard；stage4c_singleedge_fix 更适合让 single-edge 成为主信号，再附少量 guard。 |

- 如果继续沿用当前 168-row stage4b mixture，不改比例，那么更现实的目标是直接把训练提高到 `32-40 optimizer steps`。对应 epoch 约为 `2.9-3.6`。
- 如果 stage4c 改成更集中的 `singleedge_fix` patch，建议把 `relief_normal_single_edge` 提到 40-60% 的 exposure，并把 guard 压到 10-20%，这样 `24-40 optimizer steps` 就足以形成可见参数位移。
- 以当前 mix 粗算：要让 `relief_normal_single_edge` 达到每条平均 2/3/4 次暴露，大约需要 `21` / `32` / `42` 个 optimizer steps。

## Final Answer
- 当前失败判断：`两者都有，但以更新太弱为主。`
- 更细一点说：`更新太弱` 是主因，因为 9 步更新不足以让任何 held-out 失败样本发生输出层面的位移；但 `数据覆盖不足` 也真实存在，尤其 `relief_normal_single_edge` 对 held-out 20 条主失败只有 3 条 exact signature train overlap。
