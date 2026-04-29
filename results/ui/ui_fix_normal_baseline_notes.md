# UI Fix: normal_baseline Entry Alignment

## 原问题

- `scenarios.py` 的 `APP_SCENARIO_PRESETS` 里没有显式暴露 `normal_baseline`，所以 UI 里无法直接选到 baseline 预设。
- `scenarios.py` 的 `SCENE_TYPES` 不包含 `normal_baseline`，即使后续补了 preset，如果仍经过 `normalize_scene_type(...)`，也会被兜底回 `simple_local`。
- `app_fixed.py` 在 `sel == "— 自定义 —"` 时直接走 `classify_scene_type(text=traffic_constraint)`；而旧逻辑对“正常通行/整体通畅”没有 baseline 分支，默认会落到 `simple_local`。
- 结果写回时，`results["scene_type"]` 之前优先用了 `_timing_info.get("scene_type", _scene_type)`，这会让显式选择的 baseline 在运行后仍有机会被模型/解析侧口径带回别的类型。

## 这次如何修

- 在 [scenarios.py](/root/autodl-tmp/code/scenarios.py) 中把 `normal_baseline` 加入 `SCENE_TYPES` 和 `SCENE_BUCKETS`。
- 在 [scenarios.py](/root/autodl-tmp/code/scenarios.py) 中新增显式预设：
  `场景N normal_baseline 正常通行`
- 给 `normal_baseline` 增加最小 UI 模板文本：
  `当前路网整体通行正常，无明显拥堵或事故；各主干道保持基线交通状态。`
- 在 [scenarios.py](/root/autodl-tmp/code/scenarios.py) 的 `classify_scene_type(...)` 中新增 baseline 识别词，只在出现“整体通行正常 / 无明显拥堵 / 无事故 / 基线交通状态”等全局正常描述时返回 `normal_baseline`。
- 在 [app_fixed.py](/root/autodl-tmp/code/app_fixed.py) 中新增 `Scene Profile Binding`：
  `Auto Detect`
  `Manual Scene Profile`
- 当用户显式选择 preset 时，`scene_profile` 和 `scene_type` 会被直接锁定为 preset 对应值，不再按文本重判。
- 当用户在自定义模式下切到 `Manual Scene Profile` 时，`scene_profile` 和 `scene_type` 同样会被锁定为手动选择值，不再按文本重判。
- 在 [app_fixed.py](/root/autodl-tmp/code/app_fixed.py) 中，运行前会把当前有效场景写回 `path_engine._last_scene_type` / `_last_scene_profile_name`，避免后续解析链路把显式 baseline 又带回 `simple_local`。
- 在 [app_fixed.py](/root/autodl-tmp/code/app_fixed.py) 中，结果记录改为以 `scene_profile.scene_type` 作为最终 `results["scene_type"]`，并额外保留 `detected_scene_type` 作为调试字段；`scene_profile_config` 继续直接来自 `scene_profile_to_dict(scene_profile)`。

## 如何避免 normal_baseline 被误判到 simple_local

- UI 现在提供了显式 baseline preset，用户可以直接进入 `normal_baseline`。
- 一旦用户显式选了 baseline preset，`scene_profile` 会强制绑定到 `normal_baseline`，不会再因为文本内容或解析结果回落到 `simple_local`。
- 自定义模式下如果不想依赖文本推断，可以切到 `Manual Scene Profile` 并手动锁定 `normal_baseline`。
- 自定义文本的 `Auto Detect` 仍保留，但现在对“整体正常/无明显拥堵/无事故”会优先识别成 `normal_baseline`，不再无条件兜底到 `simple_local`。

## 是否影响其他场景自动识别

- 不会影响现有 preset 的主链路：preset 仍按显式配置进入，只是现在多了 baseline preset。
- 不会改变 `simple_local / directional_asymmetry / core_blockage / propagation_range / compound_disaster / temporal_switch` 的既有显式 preset 绑定逻辑。
- 自由文本自动识别主规则仍保留；本次只新增了 baseline 识别分支。
- 已额外验证：典型 `simple_local` 文本
  `黄河路经一路至经三路段向东因追尾事故严重拥堵；花园路向北畅通无阻。`
  仍识别为 `simple_local`。

## 验证

- `python -m py_compile /root/autodl-tmp/code/scenarios.py /root/autodl-tmp/code/app_fixed.py`
  通过
- `AppTest` 验证显式 preset `场景N normal_baseline 正常通行`
  运行前环境面板显示 `scene_profile=normal_baseline`、`scene_type=normal_baseline`、`seed=101`、`warmup=120 s`
  运行后 `results["scene_profile"] = normal_baseline`
  运行后 `results["scene_type"] = normal_baseline`
  运行后 `results["scene_profile_config"]["scene_name"] = normal_baseline`
  `Defense Mode` 下 `场景名称 = normal_baseline`
- `AppTest` 验证自定义文本 `当前路网整体通行正常，无明显拥堵或事故；各主干道保持基线交通状态。`
  在 `Auto Detect` 下也会进入 `normal_baseline`
