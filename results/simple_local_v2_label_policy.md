# simple_local_v2 Label Policy

## Audit Facts
- 已确认 `simple_local` strict fail 共 `51` 条，其中 `label_semantic_gap = 51/51 = 100.0%`。
- 已确认 `generation_or_parse_issue = 0`，说明当前 simple_local 的主问题不是生成质量，而是标签语义与 strict changed-edge 评估目标错位。
| Subtype | Count | Share of strict fail |
| --- | --- | --- |
| anomaly_plus_normal_side_event | 22 | 43.1% |
| relief_normal_single_edge | 20 | 39.2% |
| relief_normal_short_range | 7 | 13.7% |
| multi_edge_relief | 2 | 3.9% |

## Current Problem Statement
- 当前 simple_local 旧标签策略实质上等于“只锚定更强异常”。这会系统性漏掉 `畅通/顺畅` relief 事件，以及 `anomaly + normal` 样本里的正常 side event。
- strict 评估却把这些 active changed edges 一并计入 changed-edge recall，因此模型即使完全复现旧标签，也仍然会被判 strict fail。

## Two Candidate Policies
| 方案 | 标签规则 | 与当前 strict 的关系 | 优点 | 风险 |
| --- | --- | --- | --- | --- |
| A. strict-aligned labeling | 输出所有 active changed-event anchors，包含异常主事件、畅通/顺畅/恢复 side event，以及 relief-only 范围。 | 完全对齐，不改指标。 | 最直接消除 semantic gap，训练/评估目标一致。 | 标签更密，模型输出会比旧版更“啰嗦”。 |
| B. anchor-semantic labeling | 继续只输出更强异常主事件；normal/relief 可省略。 | 必须重写 strict 指标。 | 保持输出更稀疏，更接近“异常锚点”直觉。 | 如果不改评估，当前 strict fail 会原样保留；论文叙事也更容易被质疑为改口径。 |

## A. strict-aligned labeling

### Core Rule
- 新主线规则：`simple_local_v2` 不再定义为“只输出异常语义锚点”，而是“输出所有 active changed-event anchors”。
- 也就是说，只要某个事件对应到 active changed edges，且文本给出了可定位的 `ROAD + DIR + RANGE + LEVEL`，就必须显式输出 `ANCHOR`。

### Lexical Normalization
- `畅通无阻`、`车流顺畅`、`通行顺畅` 统一归到 `LEVEL=畅通`。
- `保持正常通行`、`交通基本正常`、`恢复正常` 统一归到 `LEVEL=正常`。
- 是否输出 ANCHOR 取决于它是否对应 `active changed edges`，而不是只看词面是否“轻”。

### Required Answers
- “畅通 / 正常 / 顺畅 / 保持正常通行”何时必须显式输出 ANCHOR：
  1. 当这类描述对应一个显式、当前生效、可定位到 `ROAD + DIR + RANGE` 的局部事件时，如果它映射到至少一条 changed edge，就必须输出 `ANCHOR`。
  2. 若它只是默认背景状态、泛化描述，或不对应任何 changed edge，则不需要输出。
  3. 对于 simple_local 当前已知 gap，真正需要补锚的主要是 `畅通/顺畅` relief 事件；`正常/保持正常通行` 也遵循同一条 edge-based 规则，而不是因为词面较轻就天然省略。
- anomaly_plus_normal 双事件样本是否要求同时锚定异常主事件和正常 side event：`是`。只要 side event 也是 active changed event，就必须和异常主事件一起输出。
- multi-edge relief 是否必须覆盖完整范围：`是`。如果文本给的是连续区间，就必须用一个连续范围锚点覆盖完整区间；如果文本本身是非连续片段，则拆成多个锚点，但它们的并集必须覆盖全部 changed edges。

### Labeling Rules in Practice
- relief-only 单边样本：旧版 `NO_ANCHOR` 改为 1 个 `ANCHOR`。
- relief-only 短范围样本：旧版 `NO_ANCHOR` 改为 1 个覆盖完整连续范围的 `ANCHOR`。
- relief-only 多边样本：旧版 `NO_ANCHOR` 改为 1 个全范围 `ANCHOR`，必要时拆段但不得漏边。
- anomaly_plus_normal：至少 2 个 `ANCHOR`，异常主事件和 normal/relief side event 都要保留。

## B. anchor-semantic labeling

### Core Rule
- 保持旧哲学：只输出更强异常主事件，normal / relief side event 可省略，relief-only 样本允许 `NO_ANCHOR`。
- 这套方案能保住“锚点=异常摘要”的简洁性，但它不再和当前 strict changed-edge 指标同义。

### If We Keep This Scheme, How strict Must Change
- `no_anchor_when_gt_changed` 必须改写为 `no_anchor_when_anchorworthy_gt_changed`：只有当样本中存在异常主事件时，`NO_ANCHOR` 才算失败。
- `gt_changed_edge_recall` 必须改成 `anchorworthy_edge_recall`：分母只保留异常主事件对应的边，不再把 relief-only 边、normal side event 边算进召回分母。
- `target_recall` 建议同步解释为“异常主事件 target recall”，避免和 normal side event 的漏锚混在一起。
- 额外新增一个次级指标，例如 `side_event_consistency` 或 `relief_event_capture_rate`，用于单独分析 normal/relief 是否被捕获，但不再作为主 strict fail 条件。
- 如果采用本方案，论文里必须明确承认：simple_local 的主任务定义已从“changed-edge coverage”改成“异常主事件提取”，否则容易被认为是为了分数而改口径。

## Recommended Mainline
- 推荐主线：`A. strict-aligned labeling`。
- 为什么它最适合当前 simple_local 指标修复：因为审计已经证明 `51/51` strict fail 都是标签缺口，而不是模型生成问题。最短路径不是调模型，而是把 supervision 直接改成与 strict changed-edge 目标同义。
- 为什么它更适合后续论文与答辩表述：
  1. 叙事更干净。我们不是为了提高分数去改指标，而是发现训练标签和评测定义错位，随后修正标签使两者一致。
  2. 可比性更强。修复前后仍然在同一个 strict 指标下比较，不需要额外解释“为什么换了口径”。
  3. 更容易做 relabel ceiling eval。只要重新标注 simple_local_v2，就能直接测“仅修标签、不改模型结构”带来的上限提升。

## Recommendation for Next Step
- 当前 simple_local 的主线动作应是：`改标签`，不是先改评估。
- 只有在产品定义明确坚持“锚点只服务异常摘要”时，才应该转向 `anchor-semantic labeling + metric rewrite`。
- 基于当前审计结果，`relabel ceiling eval` 值得立刻排到高优先级，因为它能直接验证 simple_local strict 分数里有多少上限来自标签修复。

## Unanchored Changed-Edge Levels Observed In Audit
| Level | Count |
| --- | --- |
| 畅通 | 51 |

## Before / After Examples
- 下面示例全部按推荐主线 `strict-aligned labeling` 给出新标签。

### `val_normal:0` | anomaly_plus_normal_side_event

- 原文本：常态或局部短时事件：经一路红专路至政七街段向南中度拥堵缓行；政七街经八路至花园路段向东畅通无阻
- 旧标签：
```text
<think>
scene=simple_local
anchor_count=1
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
ANCHOR|ROAD=经一路|DIR=向南|RANGE=红专路至政七街|LEVEL=中度拥堵
```
- 新标签：
```text
<think>
scene=simple_local
anchor_count=2
simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。
</think>
ANCHOR|ROAD=经一路|DIR=向南|RANGE=红专路至政七街|LEVEL=中度拥堵
ANCHOR|ROAD=政七街|DIR=向东|RANGE=经八路至花园路|LEVEL=畅通
```
- 修改原因：旧标签只保留异常主事件，漏掉了同样属于 changed edges 的正常/畅通 side event；v2 需要双锚定。

### `val_normal:26` | anomaly_plus_normal_side_event

- 原文本：常态或局部短时事件：政七街经三路至经六路段向东排队长度明显增加；经一路红专路至政七街段向北畅通无阻
- 旧标签：
```text
<think>
scene=simple_local
anchor_count=1
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
ANCHOR|ROAD=政七街|DIR=向东|RANGE=经三路至经六路|LEVEL=严重拥堵
```
- 新标签：
```text
<think>
scene=simple_local
anchor_count=2
simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。
</think>
ANCHOR|ROAD=政七街|DIR=向东|RANGE=经三路至经六路|LEVEL=严重拥堵
ANCHOR|ROAD=经一路|DIR=向北|RANGE=红专路至政七街|LEVEL=畅通
```
- 修改原因：旧标签只保留异常主事件，漏掉了同样属于 changed edges 的正常/畅通 side event；v2 需要双锚定。

### `val_normal:43` | anomaly_plus_normal_side_event

- 原文本：常态或局部短时事件：纬五路经六路至花园路段向东排队通行效率下降；经一路农业路至红专路段向南畅通无阻
- 旧标签：
```text
<think>
scene=simple_local
anchor_count=1
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
ANCHOR|ROAD=纬五路|DIR=向东|RANGE=经六路至花园路|LEVEL=中度拥堵
```
- 新标签：
```text
<think>
scene=simple_local
anchor_count=2
simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。
</think>
ANCHOR|ROAD=纬五路|DIR=向东|RANGE=经六路至花园路|LEVEL=中度拥堵
ANCHOR|ROAD=经一路|DIR=向南|RANGE=农业路至红专路|LEVEL=畅通
```
- 修改原因：旧标签只保留异常主事件，漏掉了同样属于 changed edges 的正常/畅通 side event；v2 需要双锚定。

### `val_normal:53` | anomaly_plus_normal_side_event

- 原文本：常态或局部短时事件：经一路红专路至政七街段向南追尾事故严重拥堵；政七街经三路至经六路段向东车流顺畅
- 旧标签：
```text
<think>
scene=simple_local
anchor_count=1
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
ANCHOR|ROAD=经一路|DIR=向南|RANGE=红专路至政七街|LEVEL=严重拥堵
```
- 新标签：
```text
<think>
scene=simple_local
anchor_count=2
simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。
</think>
ANCHOR|ROAD=经一路|DIR=向南|RANGE=红专路至政七街|LEVEL=严重拥堵
ANCHOR|ROAD=政七街|DIR=向东|RANGE=经三路至经六路|LEVEL=畅通
```
- 修改原因：旧标签只保留异常主事件，漏掉了同样属于 changed edges 的正常/畅通 side event；v2 需要双锚定。

### `val_normal:1` | relief_normal_single_edge

- 原文本：常态或局部短时事件：经三路红专路至政七街段向南畅通无阻
- 旧标签：
```text
<think>
scene=simple_local
anchor_count=0
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
NO_ANCHOR
```
- 新标签：
```text
<think>
scene=simple_local
anchor_count=1
simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。
</think>
ANCHOR|ROAD=经三路|DIR=向南|RANGE=红专路至政七街|LEVEL=畅通
```
- 修改原因：旧标签把单边 relief 事件写成 NO_ANCHOR，但 strict changed-edge 评估把该边计入召回分母；v2 需要显式锚定。

### `val_normal:3` | relief_normal_single_edge

- 原文本：经一路农业路至红专路段向南车流顺畅
- 旧标签：
```text
<think>
scene=simple_local
anchor_count=0
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
NO_ANCHOR
```
- 新标签：
```text
<think>
scene=simple_local
anchor_count=1
simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。
</think>
ANCHOR|ROAD=经一路|DIR=向南|RANGE=农业路至红专路|LEVEL=畅通
```
- 修改原因：旧标签把单边 relief 事件写成 NO_ANCHOR，但 strict changed-edge 评估把该边计入召回分母；v2 需要显式锚定。

### `val_normal:20` | relief_normal_single_edge

- 原文本：农业路经三路至经六路段向西车流顺畅
- 旧标签：
```text
<think>
scene=simple_local
anchor_count=0
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
NO_ANCHOR
```
- 新标签：
```text
<think>
scene=simple_local
anchor_count=1
simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。
</think>
ANCHOR|ROAD=农业路|DIR=向西|RANGE=经三路至经六路|LEVEL=畅通
```
- 修改原因：旧标签把单边 relief 事件写成 NO_ANCHOR，但 strict changed-edge 评估把该边计入召回分母；v2 需要显式锚定。

### `val_normal:5` | relief_normal_short_range

- 原文本：常态或局部短时事件：花园路政七街至纬五路段向北畅通无阻
- 旧标签：
```text
<think>
scene=simple_local
anchor_count=0
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
NO_ANCHOR
```
- 新标签：
```text
<think>
scene=simple_local
anchor_count=1
simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。
</think>
ANCHOR|ROAD=花园路|DIR=向北|RANGE=政七街至纬五路|LEVEL=畅通
```
- 修改原因：旧标签省略了连续 2-edge 的 relief 范围；v2 用一个覆盖完整范围的 relief ANCHOR 对齐 strict changed-edge 目标。

### `val_normal:6` | relief_normal_short_range

- 原文本：红专路经八路至未来路段向东畅通无阻
- 旧标签：
```text
<think>
scene=simple_local
anchor_count=0
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
NO_ANCHOR
```
- 新标签：
```text
<think>
scene=simple_local
anchor_count=1
simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。
</think>
ANCHOR|ROAD=红专路|DIR=向东|RANGE=经八路至未来路|LEVEL=畅通
```
- 修改原因：旧标签省略了连续 2-edge 的 relief 范围；v2 用一个覆盖完整范围的 relief ANCHOR 对齐 strict changed-edge 目标。

### `val_normal:13` | relief_normal_short_range

- 原文本：常态或局部短时事件：经六路农业路至政七街段向南畅通无阻
- 旧标签：
```text
<think>
scene=simple_local
anchor_count=0
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
NO_ANCHOR
```
- 新标签：
```text
<think>
scene=simple_local
anchor_count=1
simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。
</think>
ANCHOR|ROAD=经六路|DIR=向南|RANGE=农业路至政七街|LEVEL=畅通
```
- 修改原因：旧标签省略了连续 2-edge 的 relief 范围；v2 用一个覆盖完整范围的 relief ANCHOR 对齐 strict changed-edge 目标。

### `val_normal:18` | multi_edge_relief

- 原文本：黄河路经六路至未来路段向西畅通无阻
- 旧标签：
```text
<think>
scene=simple_local
anchor_count=0
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
NO_ANCHOR
```
- 新标签：
```text
<think>
scene=simple_local
anchor_count=1
simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。
</think>
ANCHOR|ROAD=黄河路|DIR=向西|RANGE=经六路至未来路|LEVEL=畅通
```
- 修改原因：旧标签把 3+ edge 的 relief 整段省略；v2 必须显式覆盖完整范围，否则 changed-edge recall 会系统性损失。

### `val_normal:172` | multi_edge_relief

- 原文本：当前需要根据常态或局部短时事件进行路径规划。 请优先按当前生效约束理解，不要被已解除或背景信息误导。 纬五路经一路至经八路段向西畅通无阻 气象部门提示傍晚仍有小雨，但目前并未形成新增积水。 导航播报中还提到相邻商圈有出库车流，但不直接改变本段主判断。 部分播报会提到周边商圈或停车场信息，但主判断仍以道路方向、区间、时段和传播范围为核心。
- 旧标签：
```text
<think>
scene=simple_local
anchor_count=0
仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。
</think>
NO_ANCHOR
```
- 新标签：
```text
<think>
scene=simple_local
anchor_count=1
simple_local_v2=输出所有 active changed-event anchors，含异常与恢复/畅通 side event。
</think>
ANCHOR|ROAD=纬五路|DIR=向西|RANGE=经一路至经八路|LEVEL=畅通
```
- 修改原因：旧标签把 3+ edge 的 relief 整段省略；v2 必须显式覆盖完整范围，否则 changed-edge recall 会系统性损失。

## Final Conclusion
- 当前 simple_local 应该优先 `改标签`；只有在坚持“只锚定更强异常”的产品哲学下，才应该同步 `改评估`。
- 后续值得进入 `relabel ceiling eval`，而且优先级高，因为已确认 strict fail 的主瓶颈不是生成，而是 supervision 口径。