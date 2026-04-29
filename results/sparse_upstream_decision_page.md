# Sparse Upstream Decision Page

## 1. 当前最稳可用基线是谁
- 当前最稳可用上游基线仍然是 `stage4_fix`。
- 结论：`是，仍建议用 stage4_fix 作为上游基线。`
- 依据很简单：
  - `stage4_fix` 是目前最后一个已经稳定通过主干 guard 的版本。
  - strict summary 中，`stage4_fix` 的 held-out `simple_local` 虽然还不完美，但已经把 pre 压到 `no_anchor_when_gt_changed_rate=0.2762`、`gt_changed_edge_recall=63.7`、`parse_fail_rate_strict=0.2086`。
  - 同时它在 `directional_asymmetry` 和 `anti_truncation` 上是稳定的：`directional_asymmetry gt_changed_edge_recall=71.8`、`parse_fail_rate_strict=0.0`；`val_anti_truncation gt_changed_edge_recall=98.0`、`parse_fail_rate_strict=0.0`。
- 因为 `stage4b` 和 `stage4c` 都没有放行，所以在导师沟通和后续实验收口里，默认上游基线仍应写成 `stage4_fix`，不是 `stage4b_fix`、也不是 `stage4c_fix`。

## 2. 为什么 stage4b / stage4c 没有放行
- `stage4b` 没放行的真实原因：方向判断是对的，但训练没真正打进去。
  - `relief_pack_ceiling_notes.md` 已经说明 relief supervision 在理论上正中瓶颈，oracle ceiling 也显示有现实依据把 `simple_local` 往下压。
  - 但 `stage4b_release_decision.md` 里实际结果是 held-out `simple_local` 完全没动：`stage4_fix` 和 `stage4b_fix` 都是 `no_anchor_when_gt_changed_rate=0.2762`、`gt_changed_edge_recall=63.7`、`parse_fail_rate_strict=0.2086`。
  - 所以 `stage4b` 的问题不是方向错，而是“训练信号太弱，没形成可见参数位移”。
- `stage4c` 没放行的真实原因：slice 内学会了，但 held-out 没迁移。
  - `stage4c_release_decision.md` 里只有很小幅改善：`simple_local no_anchor_when_gt_changed_rate 0.2762 -> 0.2667`，`gt_changed_edge_recall 63.7 -> 64.3`，仍远没到 `<=0.15`。
  - `stage4c_postmortem_summary.md` 给出的核心证据更直接：targeted `diff_29` 只动了 `1/29`，其中 `relief_normal_single_edge` 仅 `1/20` 改善，`relief_normal_short_range` `7/7` 完全没动，`multi_edge_relief` `2/2` 完全没动。
  - 同时还引入了 `8` 条 normal-only 假阳性锚点，所以不能把 `stage4c` 当成安全上游。
- 最简洁总结：
  - `stage4b`：`训练太弱，几乎没动。`
  - `stage4c`：`学会了 slice，但 held-out 迁移失败，而且开始出现副作用。`

## 3. patch SFT 还有没有继续投入价值
- broad patch SFT 的 stop/go 判断已经非常清楚：
  - `stage4c_postmortem_summary.md` 的总判断是 `DIMINISHING_RETURNS_REACHED`。
  - 它的含义不是“完全不能再动”，而是“广义 patch SFT 继续加码已经不划算”。
- 但 `stage4c_repair_priority.md` 又给了一个更细的 go/no-go：
  - 结论是 `DO_ONE_LAST_MICRO_PATCH`。
  - 仅剩一个值得再投的窄桶：`single_edge_changed_relief__short_plain_or_prefixed`。
  - 这 1 个桶覆盖 `18` 条剩余 hard cases，占剩余失败的 `36.00%`，理论上对应 held-out `simple_local` strict-fail case 的上限收益约 `17.14` 个百分点。
  - 其余 bucket 都已经被归到 `risky_patch` 或 `low_roi_patch`，不值得继续扩 patch 面。
- 所以对 patch SFT 的准确表述应该是：
  - `大方向上已经接近 stop。`
  - `但还保留一个最后的、严格限域的 micro patch 窗口。`

## 4. 下一阶段建议
- 建议走第一条：`继续最后一轮 micro patch`。
- 但这个建议有明确边界，不是“继续 patch 体系化推进”：
  - patch 范围只允许锁定在 `single-edge + LEVEL=畅通 + phrase in {畅通无阻, 车流顺畅/顺畅} + structure in {short_plain, prefixed_short}`。
  - 必须同时加 `交通基本正常 / 保持正常通行` 的硬负例 guard，避免继续放大 normal-only 假阳性。
  - 如果这最后一轮 micro patch 仍然不能让该桶出现明确 held-out 位移，就应立即停止 patch，冻结 `stage4_fix`，转向 `GAT 特殊场景、PPO baseline、论文答辩收口`。
- 对导师沟通时可以用一句话概括：
  - `当前推荐是：基线仍用 stage4_fix，不放行 stage4b/stage4c；patch SFT 只再做最后一个极小、极窄的 single-edge relief micro patch，成则升级，不成就彻底停止。`

## Final Decision
- `CONTINUE_ONE_LAST_PATCH`
