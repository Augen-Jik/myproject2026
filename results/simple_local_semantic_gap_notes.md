# Simple Local Semantic Gap Audit

## Scope
- Source details: `/root/autodl-tmp/results/sparse_v2_validation_strict_stage4_fix_details.csv`
- Source examples: `/root/autodl-tmp/results/simple_local_failure_examples.json`
- Dataset root resolved from examples meta: `/root/autodl-tmp/dataset_sparse_v2`
- Analyzed samples: `scene_type == simple_local` and strict fail, total `51`

## Audit Split
| Bucket | Count | Share of strict fail |
| --- | --- | --- |
| label_semantic_gap | 51 | 100.0% |
| generation_or_parse_issue | 0 | 0.0% |
| mixed | 0 | 0.0% |

## Semantic Gap Subtypes
| Type | Samples | Share |
| --- | --- | --- |
| anomaly_plus_normal_side_event | 22 | 43.1% |
| relief_normal_single_edge | 20 | 39.2% |
| relief_normal_short_range | 7 | 13.7% |
| multi_edge_relief | 2 | 3.9% |

## Unanchored Changed-Edge Levels
| Level | Unanchored Events | Approx Unanchored Edges |
| --- | --- | --- |
| 畅通 | 51 | 62 |

## Required Answers
- strict fail 中有多少比例其实是 label semantic gap：`51/51`，即 `100.0%`。当前审计里，generation/parse 问题样本数为 `0`，说明 strict fail 几乎全部来自标签语义覆盖缺口，而不是模型没学到标签格式。
- 哪些 changed-edge 类型在 GT 标签里长期未被显式锚定：
  1. 纯 relief/normal changed edges，尤其 `正常` / `畅通` 的单边样本、2-edge 短范围样本、3+ edge 多边样本。对应 subtype 主要是 `relief_normal_single_edge`、`relief_normal_short_range`、`multi_edge_relief`。
  2. `anomaly_plus_normal_side_event` 类型里，GT 标签通常只锚定异常主事件，把同时存在的正常/缓解 side event 留空，导致 changed-edge recall 丢失。
  3. 从未显式锚定的 level 以 `畅通` 为主；这些 level 在 strict 评估里仍对应 changed edges，所以会持续被判 miss。
- 继续用当前标签做 SFT 的收益为什么会受限：
  1. 当 `pred_text == gt_label` 仍然被 strict 判错时，继续喂同一套标签只会强化当前的漏锚策略，而不会提升 strict recall。
  2. 训练目标与评估目标不一致。标签在 `simple_local` 中默认忽略 relief/normal changed edges 或 anomaly 的正常 side event，但 strict 指标把这些边都算作必须覆盖的 changed edges。
  3. 结果上会出现学习饱和：模型越来越稳定地复现标签语义，但 strict 分数卡在同一个上限，尤其是 `no_anchor` 和 `changed_edge_miss` 这两类系统性失败。

## Representative Semantic-Gap Samples
- `val_normal:0` | anomaly_plus_normal_side_event | changed_edge_miss | input: 常态或局部短时事件：经一路红专路至政七街段向南中度拥堵缓行；政七街经八路至花园路段向东畅通无阻 | GT: <think>
scene=simple_local
anchor_count=1
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
ANCHOR|ROAD=经一路|DIR=向南|RANGE=红专路至政七街|LEVEL=中度拥堵
- `val_normal:1` | relief_normal_single_edge | no_anchor | input: 常态或局部短时事件：经三路红专路至政七街段向南畅通无阻 | GT: <think>
scene=simple_local
anchor_count=0
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
NO_ANCHOR
- `val_normal:3` | relief_normal_single_edge | no_anchor | input: 经一路农业路至红专路段向南车流顺畅 | GT: <think>
scene=simple_local
anchor_count=0
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
NO_ANCHOR
- `val_normal:5` | relief_normal_short_range | no_anchor | input: 常态或局部短时事件：花园路政七街至纬五路段向北畅通无阻 | GT: <think>
scene=simple_local
anchor_count=0
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
NO_ANCHOR
- `val_normal:6` | relief_normal_short_range | no_anchor | input: 红专路经八路至未来路段向东畅通无阻 | GT: <think>
scene=simple_local
anchor_count=0
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
NO_ANCHOR

## Output Files
- CSV: `/root/autodl-tmp/results/simple_local_semantic_gap_audit.csv`
- Notes: `/root/autodl-tmp/results/simple_local_semantic_gap_notes.md`