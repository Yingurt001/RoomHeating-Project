# 项目 2 (iii) — 房间供暖系统的有效控制

## 课程信息

- **模块**: MATH3060/1 — 应用数学方向，数学小组项目
- **大学**: University of Nottingham, School of Mathematical Sciences
- **学期**: 2026 年春季
- **GitHub**: https://github.com/Yingurt001/RoomHeating-Project

---

## 项目目标

> 对于一个房间供暖系统，**什么控制策略最有效**？**恒温器放在哪里最合理**？

### 核心交付物
- **学术报告**（≤25 页）+ **PPT** + **可复现的 Python 代码仓库**

### 故事线

```
Bang-Bang（开关控制）→ PID（连续反馈）→ LQR（最优控制）→ Pontryagin（解析最优）
        ↓                    ↓                ↓                    ↓
      基线/对照          工程改进          数学最优          理论极限
```

空间维度递进：
```
ODE（整间房一个温度）→ 1D PDE（沿一条线的温度分布）→ 2D PDE（整个房间温度场）
```

最终产出：**控制策略 × 空间模型**的对比矩阵，用量化指标评判。

---

## 数学模型

### 1. Newton 冷却定律（ODE，仅时间维度）

```
dT/dt = -k(T(t) - T_a) + u(t)
```

- `T(t)`: 室温
- `T_a`: 室外温度
- `u(t)`: 加热器输入
- `k`: 冷却常数

### 2. 热方程（PDE，时空维度）

```
∂T/∂t = α∇²T + S(x, y, t)
```

- `T(x, y, t)`: 温度场
- `α`: 热扩散率
- `S(x, y, t)`: 空间分布的热源

---

## 控制策略

| # | 策略 | 数学核心 | 定位 |
|---|------|----------|------|
| 1 | **Bang-Bang + 滞回** | 混合动力系统、Zeno 效应 | 基线对照 |
| 2 | **PID 控制** | `u = Kp·e + Ki·∫e + Kd·de/dt`，闭环稳定性 | 工程标准 |
| 3 | **LQR 最优控制** | Riccati 方程，`J = ∫[Q·x² + R·u²]dt` | 数学最优 |
| 4 | **Pontryagin 极小值原理** | Hamilton 量、伴随方程、两点边值问题 | 理论极限 |

### 文献支撑

- **Bang-Bang**: Goebel, Sanfelice & Teel (2012). *Hybrid Dynamical Systems*; Zhang et al. (2001). Zeno hybrid systems.
- **PID**: Astrom & Murray (2021). *Feedback Systems*, Ch.11; Blasco et al. (2012). PID control of HVAC.
- **LQR**: Anderson & Moore (1990/2007). *Optimal Control: Linear Quadratic Methods*; Astrom & Murray (2021), Ch.7.
- **Pontryagin**: Liberzon (2012). *Calculus of Variations and Optimal Control Theory*; Kirk (2004). *Optimal Control Theory*.

---

## 评价指标体系

所有策略用统一指标对比：

| 指标            | 公式                             | 含义    |
| ------------- | ------------------------------ | ----- |
| **能耗**        | `E = ∫u(t)dt`                  | 总能量消耗 |
| **温度偏差 RMSE** | `√(1/T·∫(T(t)-T_set)²dt)`      | 舒适度   |
| **最大超调**      | `max(T(t) - T_set)`            | 峰值偏差  |
| **稳定时间**      | 首次进入 ±0.5°C 并保持的时刻             | 响应速度  |
| **切换次数**      | 加热器开关次数                        | 设备磨损  |
| **统一代价**      | `J = ∫[Q·(T-T_set)² + R·u²]dt` | 综合权衡  |

**亮点图表**：Pareto 前沿图（X 轴=能耗，Y 轴=RMSE）。

---

## 项目结构

```
RoomHeating-Project/
├── Code/
│   ├── models/                    # 物理模型（与控制器解耦）
│   │   ├── ode_model.py          # ODE: Newton 冷却定律
│   │   ├── pde_1d_model.py       # 1D PDE: 有限差分 + Method of Lines
│   │   └── pde_2d_model.py       # 2D PDE: 矩形房间温度场
│   ├── controllers/               # 控制策略（与模型解耦）
│   │   ├── bang_bang.py           # Bang-Bang + 滞回
│   │   ├── pid.py                # PID 控制
│   │   ├── lqr.py                # LQR 最优控制
│   │   └── pontryagin.py         # Pontryagin 极小值原理
│   ├── experiments/               # 实验脚本
│   │   ├── exp_ode_comparison.py  # ODE 下各策略对比
│   │   ├── exp_1d_placement.py    # 1D 恒温器位置实验
│   │   └── exp_2d_placement.py    # 2D 位置实验
│   ├── utils/
│   │   ├── parameters.py          # 统一物理参数
│   │   ├── metrics.py             # 统一评价指标
│   │   └── plotting.py            # 统一绘图工具
│   ├── tests/
│   │   ├── test_models.py         # 模型测试
│   │   └── test_controllers.py    # 控制器测试
│   ├── results/                   # 实验输出图表
│   └── phase1_ode/                # Phase 1 原始代码（保留）
├── Reference/                     # 参考文献
├── Meeting/                       # 会议记录
├── .gitignore
├── requirements.txt
└── Claude.md                      # 本文档
```

**核心设计原则**：模型与控制器解耦 — 任何控制器可搭配任何模型运行。

---

## 技术栈

- **语言**: Python 3
- **核心库**: NumPy, SciPy (`solve_ivp`, `solve_bvp`, `solve_continuous_are`), Matplotlib
- **测试**: pytest
- **版本控制**: Git + GitHub

---

## 团队结构（6 人）

| 小队 | 角色 | 主要职责 |
|------|------|----------|
| **A — 建模** (2人) | 数学建模 | 推导方程、边界条件、稳定性分析、最优控制理论 |
| **B — 编程** (2人) | 数值实现 | Python 代码、数值方法、参数扫描、可视化 |
| **C — 研究与写作** (2人) | 调研 + 写作 | 文献调研、报告撰写、PPT 制作 |

> 所有人都需要理解整体模型。分队是为了效率，不是隔离。

---

## 执行计划

### 第 0 步：基础设施 ✅
- 项目结构重组（模型/控制器分离）
- `.gitignore` + `requirements.txt`
- 测试框架

### 第 1 步：重构 Phase 1 ✅
- `models/ode_model.py`: 纯物理模型
- `controllers/bang_bang.py`: Bang-Bang 控制逻辑

### 第 2 步：PID 控制器 ✅
- 离散 PID 控制器类（含 anti-windup）
- Ziegler-Nichols 调参
- 与 ODE 模型集成

### 第 3 步：LQR 控制器 ✅
- 状态空间建模 `dx/dt = Ax + Bu`
- 求解 Riccati 方程 → 最优增益 K
- Q/R 参数扫描 → Pareto 前沿

### 第 4 步：1D PDE 模型 ✅
- 有限差分网格 + Robin 边界条件
- Method of Lines → solve_ivp
- 恒温器位置参数化

### 第 5 步：Pontryagin 最优控制 ✅
- Hamilton 量构造 + 伴随方程
- 两点边值问题 (solve_bvp)
- 最优控制轨迹可视化

### 第 6 步：2D PDE 模型 ✅
- 2D 有限差分网格
- 加热器/恒温器位置参数化
- 温度场可视化

### 第 7 步：综合对比
- 运行全量实验矩阵
- 生成所有对比图表 + Pareto 前沿
- 整理关键发现

---

## 验证命令

```bash
# 运行所有测试
cd Code && python -m pytest tests/ -v

# 运行 ODE 策略对比实验
cd Code && python experiments/exp_ode_comparison.py

# 运行 1D 恒温器位置实验
cd Code && python experiments/exp_1d_placement.py

# 运行 2D 位置实验
cd Code && python experiments/exp_2d_placement.py
```

---

## 截止日期

| 事项 | 日期 |
|------|------|
| 报告 + PPT 提交 (Moodle) | **2026年3月9日 (周一) 下午1点** |
| 答辩 (10分钟 + 10分钟 Q&A) | **2026年3月16日 (周一)** |

---

## 参考文献

详见 `Reference/references.md`，核心文献包括：

1. Boyce, DiPrima & Meade (2021) — ODE 基础
2. Strogatz (2024) — 非线性动力学
3. **Astrom & Murray (2021) — 反馈控制、PID** [免费在线]
4. Goebel, Sanfelice & Teel (2012) — 混合动力系统
5. Zhang et al. (2001) — Zeno 效应
6. Lygeros et al. (2003) — 混合自动机
7. Strikwerda (2004) — 有限差分方法
8. Blasco et al. (2012) — HVAC 系统 PID 控制
9. **Anderson & Moore (1990) — LQR 最优控制**
10. **Liberzon (2012) — 变分法与最优控制 (Pontryagin)**
11. **Kirk (2004) — 最优控制理论导论 (Pontryagin)**
