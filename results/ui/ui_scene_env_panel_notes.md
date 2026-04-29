# UI Scene / Env Panel Notes

## 本轮目标

把场景配置和环境参数整理成一个清晰的 UI 面板，让答辩时可以直接说明：

- 哪些由 `scene_profile` 固定控制
- 哪些由模型负责

## 已完成改动

在 `/root/autodl-tmp/code/app_fixed.py` 的侧边栏新增了 `🧭 场景与环境配置` 折叠面板，当前会实时展示：

- `scene_profile`
- `scene_type`
- `speed_profile`
- `tls_profile_name`
- `reroute_enabled / reroute_period`
- `simulation_seed`
- `warmup_seconds`
- `evaluation window`
- `incident / blocked / lane reduction` 摘要

同时在面板顶部显式加入说明：

> 这些环境参数由场景配置固定控制，不直接受模型理解影响；模型只影响边权估计和路径选择。

## 刷新与防误读

本轮还增加了请求签名保护：

- 当前输入会生成 `current_request_signature`
- 每次成功运行后会记录 `last_result_signature`
- 如果老师切换了场景、起终点、约束文本、算法或相关输入，但还没有重新运行：
  - 原始输出面板会隐藏旧结果
  - 导出按钮会临时禁用
  - 面板会提示“上一轮缓存结果已停止作为当前场景展示”

这样可以避免把上一个场景的缓存结果误认为是当前场景结果。

## 答辩层面的价值

现在 UI 上的分工已经更清楚：

- 模型层：
  - 边权估计
  - 路径选择

- 环境层：
  - TLS
  - reroute policy
  - seed
  - warmup / evaluation window
  - incident / blocked / lane reduction
  - speed profile

老师看侧边栏时，可以直接把“固定环境”与“模型决策”区分开，不需要再从结果卡片里反向推断。
