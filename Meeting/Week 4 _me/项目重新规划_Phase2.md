# Group Project Phase 2 — 重新规划：探索不同物理场景

> **目标**: 将项目重心从"哪种控制策略最好"转向"探索不同物理场景下的温度控制"——窗户、门、房间形状、多房间等。展示团队对各种实际可能性的思考深度。
>
> **总工时**: ~80 小时（6人 × 2.5周 × ~5.5h/人/周）| **文献调研+设计**: 15h | **代码开发**: 30h | **实验运行+出图**: 15h | **报告+PPT**: 20h
>
> **截止日期**: 2026年3月9日（周一）1pm 提交报告+PPT | 3月16日（周一）答辩

---

## 核心思路转变

### 老师要什么

| 老师不在意的 | 老师在意的 |
|---|---|
| 哪种控制策略"最好" | 你们**考虑了哪些不同的物理场景** |
| 复杂的评价方程和排名 | 每种场景下温度分布**有什么不同**，为什么 |
| 优化到极致的参数 | 你们的思考过程：为什么选这些场景、结果是否符合物理直觉 |

### 新的项目叙事线

```
简单房间（基线）→ 加窗户 → 加门 → 变形状 → 多房间 → 综合讨论
     ↓              ↓         ↓        ↓          ↓           ↓
  已完成          热损失    通风换热   几何效应   耦合传热    工程启示
```

### 已有代码可复用的部分

| 模块 | 路径 | 状态 |
|------|------|------|
| 2D 有限差分求解器 | `models/pde_2d_model.py` | 直接复用，需扩展 BC 接口 |
| 1D PDE 模型 | `models/pde_1d_model.py` | 直接复用 |
| Bang-Bang/Hysteresis 控制器 | `controllers/bang_bang.py` | 直接复用（统一用这一种即可） |
| 绘图工具 | `utils/plotting.py` | 复用+扩展 |
| 参数管理 | `utils/parameters.py` | 复用+扩展 |

### 需要新增的代码

| 新模块 | 说明 |
|--------|------|
| `scenarios/` 目录 | 每个场景一个 .py 配置文件 |
| `models/pde_2d_model.py` 扩展 | 支持分段式边界条件（同一墙面不同区段用不同 h 值）|
| `models/pde_2d_model.py` 扩展 | 支持 L 形/矩形等非正方形域的 mask |
| `models/multi_room.py`（可选） | 双房间耦合模型 |
| `experiments/exp_scenarios.py` | 统一的场景实验运行器 |
| `results/scenario_gallery.py` | 最终出图脚本（一键生成所有图） |

---

## 一、文献调研 + 场景设计 (15 小时)

### 1.1 边界条件建模参考 (4 小时)

> 目标：搞清楚窗户、门、不同墙面在数学上怎么表达

| 资源 | 链接 | 阅读重点 | 时间 |
|------|------|----------|------|
| Robin BC 的数学推导 | https://en.wikipedia.org/wiki/Robin_boundary_condition | `a·u + b·(du/dn) = g` 的标准形式 | 30min |
| The Robin BC for Heat Transfer (Royal Society) | https://royalsocietypublishing.org/doi/10.1098/rspa.2023.0850 | Robin 系数 h 的物理含义；h→∞ 为 Dirichlet，h→0 为 Neumann | 1h |
| MDPI: 隔热房间冷却模拟 | https://www.mdpi.com/1996-1073/11/11/3205 | 墙壁、窗户、门用不同热传导系数建模 | 1h |
| 墙壁和窗户的热传递系数 | https://mepacademy.com/heat-transfer-thru-walls-and-windows/ | 实际工程中墙 vs 窗的 U-value 差多少 | 30min |
| Heat Loss Calculations (PDF) | https://www.cedengineering.com/userfiles/M05-003%20-%20Heat%20Loss%20Calculations%20and%20Principles%20-%20US.pdf | 复合墙体、窗户的热阻计算 | 1h |

**产出**: 一张汇总表——各种边界类型（保温墙、外墙、单层窗、双层窗、开门、关门）对应的 Robin 系数 h 值范围

### 1.2 房间几何与多房间参考 (3 小时)

| 资源 | 链接 | 阅读重点 | 时间 |
|------|------|----------|------|
| Swarthmore: 热系统数学模型 | https://lpsa.swarthmore.edu/Systems/Thermal/SysThermalModel.html | 双房间 ODE 模型：R-C 类比，耦合方程 | 1h |
| MDPI: 辐射器房间瞬态加热 | https://www.mdpi.com/2075-5309/8/11/163 | 多墙面（内墙+外墙+辐射器）耦合模型 | 1h |
| UTM: L 形域的有限元求解 (PDF) | https://science.utm.my/procscimath/wp-content/uploads/sites/605/2023/10/45-55.-FATANAH-BINTI-HAMZAH-A19SC0080.pdf | 非矩形域的网格处理 | 1h |

### 1.3 场景清单设计 (4 小时)

> 这是最关键的环节。目标：设计 **6-8 个**有代表性的物理场景，覆盖老师提到的各种可能性

**推荐场景清单**（6个核心 + 2个进阶）：

| # | 场景 | 物理含义 | 数学建模变化 | 预期亮点 |
|---|------|----------|-------------|----------|
| S1 | **基线：方形房间** | 5×5m，四面保温墙+一面外墙，无窗无门 | 已有的 RNNN | 对照组 |
| S2 | **加窗户** | 外墙上有一扇窗（高热损区域） | 外墙 Robin BC 中，窗户区段 h 值更大（如 h_window=5×h_wall） | 窗户附近冷区、温度场不对称 |
| S3 | **大窗 vs 小窗 vs 双层窗** | 窗户大小和隔热性能影响 | 改变窗户占墙面比例和 h_window 值 | 定量展示窗户隔热的价值 |
| S4 | **开门通风** | 房间有门，门打开时冷空气进入 | 门所在墙段的 BC 在 Neumann（关门）和 Dirichlet(T_a)（开门）之间切换 | 温度骤降和恢复过程 |
| S5 | **长窄房间** | 7.5×2.5m（等面积矩形） | 改变域的长宽比 | 热量传播距离更远，均匀度差 |
| S6 | **L 形房间** | 非凸域，存在热死角 | mask 掉右上角 | 拐角处温度盲区 |
| S7 | **双房间（进阶）** | 两个房间通过门洞相连，只有一个加热器 | 两个域通过共享边界耦合，或用大域+中间隔墙 | 远端房间能否被加热 |
| S8 | **综合场景（进阶）** | L 形房间 + 窗户 + 门 | 组合 S2+S4+S6 | 最接近现实的复杂情况 |

**设计原则**：
- 每个场景只改变**一个变量**（窗户/门/形状），和 S1 对比
- S7、S8 是组合场景，展示思考深度
- 每个场景用 **同一种控制策略**（Hysteresis Bang-Bang），因为重点是物理场景而非控制

**产出**: 每个场景的数学定义文档（域形状、BC 定义、参数值）

### 1.4 实验设计矩阵 (4 小时)

> 每个场景下要跑什么实验、生成什么图

**每个场景统一输出**：

| 输出 | 说明 |
|------|------|
| 终态温度场热力图 | 2D 颜色图，标注加热器和恒温器位置 |
| 温度时间曲线 | 恒温器读数 + 房间均温 vs 时间 |
| 加热器位置敏感性 | 扫描 3-5 个加热器位置，看最优放置 |
| 恒温器位置敏感性 | 扫描 3-5 个恒温器位置 |
| 能耗和舒适度 | 简单的 RMSE + 能耗（不做复杂加权评分） |

**对比图**（跨场景）：

| 对比 | 包含场景 | 回答的问题 |
|------|----------|------------|
| 窗户效应 | S1 vs S2 vs S3 | 窗户如何影响温度分布？双层窗改善多少？ |
| 门的影响 | S1 vs S4 | 开门通风的温度恢复过程 |
| 几何效应 | S1 vs S5 vs S6 | 房间形状对控制难度的影响 |
| 综合对比 | S1-S8 | 哪个因素影响最大？ |

---

## 二、代码开发 (30 小时)

### 2.1 扩展 2D PDE 求解器 (10 小时) — 编程组

> 核心修改：让现有求解器支持分段式边界条件和非矩形域

**任务清单**：

#### Task A: 分段式 Robin BC (4h)
- 修改 `models/pde_2d_model.py`，支持同一条边上不同区段使用不同 h 值
- 接口设计：`boundary_segments = [{"wall": "south", "start": 0, "end": 2, "h": 0.5}, {"wall": "south", "start": 2, "end": 3, "h": 2.5}, ...]`
- 这样窗户就是外墙上一段 h 值更大的区间

#### Task B: 时变边界条件 (3h)
- 支持 BC 参数随时间变化：`h(t)` 或在特定时刻切换 BC 类型
- 门的模型：t < t_open 为 Neumann，t_open < t < t_close 为 Dirichlet(T_a)，t > t_close 恢复 Neumann

#### Task C: 域形状 mask (3h)
- L 形域：在正方形网格上 mask 掉右上角，mask 区域不参与计算
- 长窄矩形：改变 Lx/Ly 比例
- 已有代码可能已部分支持，检查 `pde_2d_model.py` 现状

### 2.2 场景配置系统 (6 小时) — 编程组

#### Task D: scenarios/ 目录 (4h)
```python
# scenarios/s1_baseline.py
SCENARIO = {
    "name": "S1: Baseline Square Room",
    "domain": {"Lx": 5, "Ly": 5, "shape": "rectangle"},
    "boundaries": {
        "south": [{"start": 0, "end": 5, "type": "robin", "h": 0.5}],  # 外墙
        "north": [{"start": 0, "end": 5, "type": "neumann"}],           # 保温墙
        "west":  [{"start": 0, "end": 5, "type": "neumann"}],
        "east":  [{"start": 0, "end": 5, "type": "neumann"}],
    },
    "heater": {"x": 2.5, "y": 0.5, "type": "gaussian", "sigma": 0.5},
    "thermostat": {"x": 2.5, "y": 2.5},
    "params": {"T_a": 5, "T_set": 20, "T_init": 15, "alpha": 0.01},
}
```

```python
# scenarios/s2_with_window.py — 在南墙(外墙)中间加窗
SCENARIO = {
    ...
    "boundaries": {
        "south": [
            {"start": 0, "end": 1.5, "type": "robin", "h": 0.5},    # 外墙
            {"start": 1.5, "end": 3.5, "type": "robin", "h": 2.5},  # 窗户 (h 是墙的5倍)
            {"start": 3.5, "end": 5, "type": "robin", "h": 0.5},    # 外墙
        ],
        ...
    },
}
```

#### Task E: 统一实验运行器 (2h)
```python
# experiments/exp_scenarios.py
# 对每个场景自动运行：
# 1. 默认配置的温度场模拟
# 2. 加热器位置扫描（3-5个位置）
# 3. 恒温器位置扫描（3-5个位置）
# 4. 输出标准化图表到 results/scenario_XX/
```

### 2.3 可视化 (6 小时) — 编程组

#### Task F: 标准化出图函数 (3h)
每个场景输出以下标准图：
- `plot_temperature_field(scenario, T, t)` — 2D 热力图，高亮窗户/门区域
- `plot_time_series(scenario, T_mean, T_sensor, u)` — 温度+控制信号时间曲线
- `plot_position_sensitivity(scenario, results)` — 加热器/恒温器位置扫描

#### Task G: 跨场景对比图 (3h)
- 多场景温度场并排（如 1×4 面板：基线/加窗/长窄/L形）
- 多场景指标柱状图（能耗、RMSE 并排对比）
- 每种因素（窗户/门/形状）单独一组对比图

### 2.4 多房间模型（进阶，可选）(8 小时) — 建模组+编程组

#### Task H: 双房间耦合 (8h)
- 方案一（推荐）：用一个大的 2D 域（如 10×5m），中间放一堵"内墙"（高热阻区域），门洞处热阻为 0
- 方案二：两个独立域通过门洞边界条件耦合
- 如时间不够可降为选做

---

## 三、实验运行 + 结果分析 (15 小时)

### 3.1 逐场景运行 (8 小时)

| 场景 | 运行内容 | 预估时间 |
|------|----------|----------|
| S1 基线 | 默认配置 + 加热器扫描 + 恒温器扫描 | 1h |
| S2 加窗户 | 同上 + 额外：窗户位置在 3 个不同位置 | 1.5h |
| S3 窗户参数 | 大窗/小窗/双层窗 3 组对比 | 1h |
| S4 开门 | 门开 5min/10min/20min 的温度响应 | 1h |
| S5 长窄房间 | 默认配置 + 位置扫描 | 1h |
| S6 L 形 | 默认配置 + 位置扫描 + 死角分析 | 1.5h |
| S7 双房间 | 门开/门关两种状态 | 可选 1h |

### 3.2 跨场景对比分析 (4 小时)

- 生成所有对比图
- 整理关键发现（每个场景 2-3 句话结论）
- 建立"因素影响排名"：哪个物理因素对温度分布影响最大

### 3.3 结果验证 (3 小时)

- 检查物理合理性：窗户附近是否确实更冷？L 形死角是否温度低？
- 能量守恒检查
- 与文献定性结论对比

---

## 四、报告 + PPT (20 小时)

### 4.1 报告结构 (≤25 页)

| 章节 | 页数 | 内容 | 负责 |
|------|------|------|------|
| Executive Summary | 1 | 项目目标、方法、关键发现 | 写作组 |
| 1. Introduction | 2 | 问题背景、项目目标、报告结构 | 写作组 |
| 2. Mathematical Model | 3 | 热方程、控制策略(Bang-Bang)、边界条件建模（**重点：如何用 Robin BC 建模窗户/门/墙**）| 建模组 |
| 3. Numerical Methods | 2 | 有限差分、Method of Lines、网格设计、非矩形域处理 | 建模组 |
| 4. Scenario Design | 2 | 为什么选这些场景、物理意义、参数选择的依据 | 全组 |
| 5. Results | 8 | **核心章节**：每个场景的结果 + 跨场景对比，大量图表 | 全组 |
| 6. Discussion | 3 | 关键发现总结、物理解释、实际工程启示、局限性 | 写作组 |
| 7. Conclusions | 1 | 主要结论、未来工作 | 写作组 |
| References | 1 | 参考文献 | 写作组 |
| Appendix | 2 | 代码结构说明、额外图表 | 编程组 |

### 4.2 PPT (10 分钟)

| 幻灯片 | 内容 | 时间 |
|--------|------|------|
| 1 | Title + Team | 15s |
| 2 | 问题：房间供暖——为什么需要数学建模？ | 30s |
| 3 | 数学模型：热方程 + Bang-Bang 控制 | 1min |
| 4 | 边界条件的多样性：墙、窗、门 → Robin BC 参数化 | 1min |
| 5-6 | 场景展示 1：窗户效应（S1 vs S2 vs S3 温度场对比） | 1.5min |
| 7 | 场景展示 2：开门通风（S4 温度恢复动画/快照） | 1min |
| 8-9 | 场景展示 3：房间形状（S5 长窄 + S6 L形 对比） | 1.5min |
| 10 | 场景展示 4：多房间/综合（S7 或 S8） | 1min |
| 11 | 跨场景综合对比：哪个因素影响最大？ | 1min |
| 12 | 结论 + 实际启示 | 30s |
| 13 | Q&A 备用页（额外图表、参数细节） | - |

### 4.3 时间分配

| 任务 | 时间 | 负责 |
|------|------|------|
| 报告初稿 | 10h | 写作组为主，建模组写 §2-3 |
| PPT 制作 | 4h | 写作组 |
| 报告校对+修改 | 4h | 全组 |
| 答辩彩排 | 2h | 全组（至少 2 次） |

---

## 五、每周时间表

### Week 1: 2月19日(周三) — 2月25日(周二) | 重点：调研 + 代码开发

| 日期 | 建模组 (2人) | 编程组 (2人) | 写作组 (2人) |
|------|-------------|-------------|-------------|
| **周三 2/19** | 阅读 Robin BC 文献 (1.1) | 检查现有代码，评估需要修改的范围 | 阅读场景设计参考文献 |
| **周四 2/20** | 设计 S1-S6 的数学定义 | **Task A**: 分段式 Robin BC 实现 | 阅读多房间建模文献 (1.2) |
| **周五 2/21** | 审核代码实现的数学正确性 | **Task A** 完成 + **Task B** 开始 | 场景清单最终定稿 (1.3) |
| **周六 2/22** | 完成实验设计矩阵 (1.4) | **Task B** 完成 + **Task C** 域 mask | 开始搜集报告参考文献 |
| **周日 2/23** | 编写 scenario 配置 (Task D) | **Task C** 完成 + **Task D** 配合 | 写报告 §1 Introduction 初稿 |
| **周一 2/24** | 测试+验证 S1-S3 | **Task E**: 实验运行器 | 写报告 §2 Mathematical Model |
| **周二 2/25** | 测试+验证 S4-S6 | **Task F**: 标准化出图函数 | 写报告 §3 Numerical Methods |

**Week 1 里程碑**：
- [ ] S1-S6 的代码全部能跑通
- [ ] 每个场景至少有一张温度场热力图验证物理合理性
- [ ] 报告 §1-§3 初稿完成

---

### Week 2: 2月26日(周三) — 3月4日(周二) | 重点：实验运行 + 出图 + 报告

| 日期 | 建模组 (2人) | 编程组 (2人) | 写作组 (2人) |
|------|-------------|-------------|-------------|
| **周三 2/26** | 运行 S1-S3 全量实验 | **Task G**: 跨场景对比图函数 | 写报告 §4 Scenario Design |
| **周四 2/27** | 运行 S4-S6 全量实验 | S7 双房间代码（如有余力）| 整理 S1-S3 结果到报告 §5 |
| **周五 2/28** | 跨场景对比分析 | 调试出图、修复 bug | 整理 S4-S6 结果到报告 §5 |
| **周六 3/1** | 结果验证：物理合理性检查 | 生成所有最终图表 | 写报告 §5 跨场景对比部分 |
| **周日 3/2** | S7/S8 进阶场景（如有余力）| 最终图表打磨 | 写报告 §6 Discussion |
| **周一 3/3** | 审核全部图表和数据 | 代码整理、注释 | 写报告 §7 Conclusions + References |
| **周二 3/4** | 审核报告中的数学内容 | 准备代码附录 | 报告全文初稿完成 |

**Week 2 里程碑**：
- [ ] 所有场景实验完成，图表生成
- [ ] 跨场景对比分析完成
- [ ] 报告全文初稿完成（可能粗糙但完整）

---

### Week 3: 3月5日(周三) — 3月9日(周日) | 重点：打磨 + 提交

| 日期 | 全组 |
|------|------|
| **周三 3/5** | 全组审读报告，逐章修改。PPT 制作开始 |
| **周四 3/6** | 报告定稿修改。PPT 完成初稿 |
| **周五 3/7** | 报告最终校对。PPT 修改。第一次答辩彩排 |
| **周六 3/8** | 最终修改。第二次答辩彩排。准备 Q&A 备用材料 |
| **周日 3/9** | 上午最终检查。**1pm 前提交** |

**Week 3 里程碑**：
- [ ] 报告终稿 ≤25 页
- [ ] PPT 完成（10 分钟）
- [ ] 至少 2 次答辩彩排
- [ ] Moodle 提交

---

## 六、补充资源

### 数学与数值方法

| 资源 | 链接 | 用途 |
|------|------|------|
| LeVeque: Finite Difference Methods (教材) | https://faculty.washington.edu/rjl/fdmbook/ | FDM 参考书，含 Neumann/Robin BC 处理 |
| ULB MOOC: Python PDE 求解 | https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/01_Introduction/01_00_Preface.html | 热方程有限差分实现 |
| ULB MOOC: 2D 问题与迭代法 | https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/05_IterativeMethods/05_01_Iteration_and_2D.html | 2D 有限差分实现 |
| MDPI: 教育用 2D 热传导 Python 代码 | https://www.mdpi.com/2076-3417/14/16/7159 | 2D 不规则形状的 FDM Python 实现 |

### 物理参数参考

| 资源 | 链接 | 用途 |
|------|------|------|
| 墙壁/窗户 U-value 实用参考 | https://mepacademy.com/heat-transfer-thru-walls-and-windows/ | 确定不同材质的热传递系数 |
| NIST: 建筑热传递基础 (PDF) | https://nvlpubs.nist.gov/nistpubs/jres/82/jresv82n2p97_a1b.pdf | 复合墙体热阻叠加 |
| BC Hydro: 建筑围护结构热分析 (PDF) | https://www.bchydro.com/content/dam/BCHydro/customer-portal/documents/power-smart/builders-developers/final-mh-bc-part-1-envelope-guide.pdf | 工程参数值 |

### 可视化与交互工具

| 资源 | 链接 | 用途 |
|------|------|------|
| VisualPDE: 热方程交互模拟 | https://visualpde.com/basic-pdes/heat-equation.html | 快速验证 BC 效果，支持 Robin BC |
| VisualPDE: 非齐次热方程 | https://visualpde.com/basic-pdes/inhomogeneous-heat-equation.html | 带热源的热方程 |
| Wolfram: 房间加热 PDE 教程 | https://reference.wolfram.com/language/PDEModels/tutorial/SystemPhysics/ModelCollection/RoomHeating.html | 完整的房间加热建模示例 |

### GitHub 代码参考

| 仓库 | 链接 | 亮点 |
|------|------|------|
| heatrapy (Python 热传递库) | https://github.com/djsilva99/heatrapy | **支持运行时改变 BC**，适合模拟开门 |
| 2D Heat Equation FDM | https://github.com/SbElolen/2D-Heat-Conduction-Simulation-using-Finite-Difference-Method | 可定制 BC 的 2D FDM |
| 2D Heat + Wave (矩形+圆形域) | https://github.com/leo-aa88/heat-equation-2d | 非矩形域处理参考 |
| ULB PDE MOOC 代码 | https://github.com/aquaULB/solving_pde_mooc | 完整 Jupyter 实现 |

### 已有项目文献

| 文献 | 位置 | 与新方向的关系 |
|------|------|-------------|
| Astrom & Murray — Feedback Systems | `Reference/` | Bang-Bang/Hysteresis 控制参考 |
| Goebel et al. — Hybrid Dynamical Systems | `Reference/` | Zeno 效应分析保留 |
| Strikwerda — FDM | `Reference/` | 有限差分方法参考 |

---

## 七、团队分工

| 角色 | 人数 | Week 1 重点 | Week 2 重点 | Week 3 重点 |
|------|------|-------------|-------------|-------------|
| **建模组** | 2 | 文献调研、场景数学定义、审核代码 | 运行实验、分析结果、验证物理合理性 | 审核报告数学内容、答辩准备 |
| **编程组** | 2 | 扩展 2D 求解器、场景配置、出图函数 | 调试、最终出图、代码整理 | 代码附录、答辩准备 |
| **写作组** | 2 | 文献搜集、场景清单、报告 §1-3 | 报告 §4-7、图表整理 | PPT、报告打磨、答辩彩排 |

---

## 八、检查标准

### Week 1 结束检查
- [ ] 能解释 Robin BC 的 h 参数如何区分墙壁、窗户、门
- [ ] 6 个核心场景的代码全部能运行
- [ ] 每个场景有至少 1 张温度场热力图，物理上合理
- [ ] 报告 §1-§3 初稿完成

### Week 2 结束检查
- [ ] 所有场景的完整实验结果出齐
- [ ] 跨场景对比图生成，能清楚回答"窗户/门/形状分别怎么影响温度"
- [ ] 报告全文初稿完成（粗糙但完整）
- [ ] 有至少 3 个关键发现能在答辩中讲述

### 最终提交检查
- [ ] 报告 ≤25 页，包含所有必要章节
- [ ] PPT 10 分钟，每张幻灯片有清晰要点
- [ ] 能回答以下 Q&A 问题：
  - "为什么选这些场景？" → 覆盖老师提到的所有类型
  - "窗户对温度分布影响有多大？" → 定量数据
  - "L 形房间的控制难点在哪？" → 死角分析数据
  - "恒温器应该放在哪？" → 位置敏感性结果
  - "你们的模型有什么局限？" → 2D 简化、忽略对流等
- [ ] 至少 2 次答辩彩排完成
- [ ] Moodle 提交前全组确认
