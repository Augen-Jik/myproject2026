# UI Map Dual View Notes

## 本轮目标

把地图展示区整理成稳定的主备双链路：

- 默认主展示：`Real Map (Folium)`
- 备用展示：`Fallback Flat Map (Plotly)`

## 已完成改动

### 1. 新增统一的双视图切换逻辑

在 `/root/autodl-tmp/code/app_fixed.py` 中新增：

- `_render_dual_map_view(...)`

当前以下两个区域都使用统一切换方式：

- 路网总览（Tab5）
- 规划路径可视化（Tab1）

界面上现在有清晰切换项：

- `Real Map (Folium)`
- `Fallback Flat Map (Plotly)`

### 2. 默认主展示仍然是 Folium 真实底图

Folium 现在仍是默认推荐主视图，因为它更适合答辩时解释：

- 真实道路参考
- SUMO 边
- 规划路径
- 起终点
- 图层控制

### 3. Plotly 作为稳定备用平面图保留

Plotly 平面图保留为备用链路，适合以下情况：

- 浏览器对 iframe / HTML 兼容性较差
- Folium 加载慢
- 现场网络/前端环境不稳定

切换到 Plotly 时不会重复跑模型、planner 或 SUMO，只切换展示层。

### 4. 增加加载与回退提示

现在在 Folium 视图下会明确提示：

- 正在加载 `Real Map (Folium)`
- 若 HTML / iframe 加载慢、空白或浏览器不兼容，可切换到 `Fallback Flat Map (Plotly)`

如果 Folium 渲染抛错，会自动提示并回退到 Plotly。

### 5. 地图信息层补强

本轮还补强了地图内容：

- Folium 侧：
  - 路网
  - 关键路段图层
  - 规划路径图层
  - 起终点 Marker
  - tooltip / popup
  - `LayerControl`

- Plotly 侧：
  - 路网底图
  - 路径
  - 起终点
  - 关键路段高亮

### 6. 支持基于缓存结果切换地图视图

之前 UI 在 `run_button=False` 时不会复用结果区，导致切换显示控件后结果页可能空掉。

本轮增加了缓存结果恢复逻辑：

- 若当前输入上下文未变化，则直接复用已有 `session_state["results"]`
- 地图切换只刷新视图，不重复计算核心模型逻辑

这使得答辩现场可以更放心地在 Folium 与 Plotly 之间切换。

## 结论

当前 UI 已具备主备双链路展示能力：

- 主链路：`Real Map (Folium)`
- 备链路：`Fallback Flat Map (Plotly)`

## 答辩建议

默认推荐用于答辩的是：

- `Real Map (Folium)`

原因：

- 更直观
- 更贴近真实道路形态
- 图层控制更利于解释“路网 / 关键路段 / 路径 / 起终点”

若现场浏览器表现不稳定，再切换到：

- `Fallback Flat Map (Plotly)`
