# UI Defense Mode Notes

## 本轮目标

为 Streamlit 主界面增加一个专门面向现场展示的 `Defense Mode`，把答辩时真正需要讲的内容单独收敛出来，避免信息过载。

## 已完成改动

在 `/root/autodl-tmp/code/app_fixed.py` 中新增了：

- `Defense Mode` 开关
- `Show Debug Details` 按钮

## Defense Mode 打开后默认保留的内容

主界面默认只保留以下答辩关键结果：

- 场景名称
- 主方法名：`Sparse-LoRA-v2`
- 关键 sparse 结果 / 异常锚点摘要
- 最终路径
- `Planning Time (s)`
- `Travel Time (s)`
- 场景环境摘要
- 地图展示

界面结构也被收缩为答辩视图：

- `🛡️ 答辩总览`
- `⏱️ 时间指标 & 场景环境`

只有点击 `Show Debug Details` 后，才会额外展开调试细节页。

## Defense Mode 下默认隐藏 / 收起的内容

以下内容在默认答辩模式下已隐藏或折叠：

- exploratory model 切换
- patch / GAT 试验入口
- 手动权重调试入口
- 原始输出 / 调试日志
- 规划历史
- 低层算法切换
- legacy / 兼容字段直接展示
- 附录对比与高级分析页

## Show Debug Details 的行为

点击 `Show Debug Details` 后，会恢复调试可见性，包括：

- 侧边栏中的低层调试项
- 原始输出
- route 明细
- 场景环境细表
- Debug tab 中的附录 / 高级分析内容

这样可以保证：

- 默认答辩视图足够干净
- 需要时仍然能快速切回更深的细节层

## 结果

当前已经形成了一套可现场展示的答辩模式：

- 平时用完整模式做分析
- 现场用 `Defense Mode`
- 被追问时点击 `Show Debug Details`

## 备注

`Defense Mode` 不会改变核心模型逻辑，只改变展示层组织方式。
