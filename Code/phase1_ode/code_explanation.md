# ODE 求解器代码逐行解释

## Code Explanation for `ode_model.py`

本文档对 `ode_model.py` 中的每个函数进行逐行解释，说明求解逻辑和数值方法。

---

## 0. 为什么需要特殊的求解策略？

我们的 ODE 是：

$$\frac{dT}{dt} = -k(T - T_a) + u(T)$$

其中 $u(T)$ 是一个**不连续函数**（开/关切换）。这意味着 ODE 的右端函数在 $T = T_{set}$ 处有跳跃。

**标准 ODE solver**（如 RK45）假设右端函数是连续的。如果我们直接把不连续的 $u(T)$ 扔给 solver，它会在切换点附近：
- 精度急剧下降
- 时间步长被迫缩小到极小
- 可能产生虚假振荡

**解决方案：分段积分（Piecewise Integration）**

在每一段内，heater 状态不变，ODE 是光滑的：
- Heater ON：$dT/dt = -k(T - T_a) + U_{max}$ — 一阶线性 ODE，光滑
- Heater OFF：$dT/dt = -k(T - T_a)$ — 一阶线性 ODE，光滑

我们用 **event detection** 精确定位切换时刻，然后在切换点断开，切换状态，继续下一段积分。

---

## 1. `solve_bang_bang()` — 无滞回 Bang-Bang 求解器

### 函数签名

```python
def solve_bang_bang(T0=T_INITIAL, t_end=T_END, k=K_COOL, T_a=T_AMBIENT,
                    T_set=T_SET, U_max=U_MAX, max_switches=500):
```

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `T0` | 初始温度 | 10°C |
| `t_end` | 仿真终止时间 | 120 min |
| `k` | 冷却常数 | 0.1 /min |
| `T_a` | 室外温度 | 5°C |
| `T_set` | 恒温器设定温度 | 20°C |
| `U_max` | 加热器最大功率 | 15 |
| `max_switches` | 最大切换次数限制（防 Zeno） | 500 |

### 初始化

```python
t_all = [0.0]       # 时间序列，初始 t=0
T_all = [T0]        # 温度序列，初始 T=T0
heater_all = [1 if T0 < T_set else 0]  # 初始 heater 状态

t_current = 0.0     # 当前积分起点时间
T_current = T0      # 当前积分起点温度
heater_on = T0 < T_set  # 初始状态：如果 T0 < T_set，heater 开
n_switches = 0      # 已发生的切换次数
```

**逻辑**：三个列表 `t_all`, `T_all`, `heater_all` 用于收集所有分段的结果，最后拼接成完整的时间序列。

### 主循环：分段积分

```python
while t_current < t_end and n_switches < max_switches:
```

循环在两种情况下终止：
1. 时间到达 `t_end`（正常结束）
2. 切换次数达到 `max_switches`（Zeno 保护）

### 每段内：定义 ODE 和事件

**Heater ON 的情况：**

```python
if heater_on:
    def rhs(t, T):
        return [-k * (T[0] - T_a) + U_max]   # ODE右端：加热+散热

    def event_off(t, T):
        return T[0] - T_set                    # 当 T = T_set 时此值为零
    event_off.terminal = True                   # 过零时终止积分
    event_off.direction = 1                     # 只在向上穿越时触发
```

- `rhs`：当前段的 ODE 右端函数。`T` 是一个数组（`solve_ivp` 的要求），`T[0]` 是温度值。
- `event_off`：**事件函数**。`solve_ivp` 在积分过程中持续监控此函数的值。当值从负变正（`direction=1`，即温度从低于 $T_{set}$ 升到高于 $T_{set}$），solver 精确定位过零时刻并停止积分。
- `terminal=True`：事件触发后**立即停止**本段积分。

**Heater OFF 的情况：**

```python
else:
    def rhs(t, T):
        return [-k * (T[0] - T_a)]             # ODE右端：只有散热

    def event_on(t, T):
        return T[0] - T_set                     # 当 T = T_set 时此值为零
    event_on.terminal = True
    event_on.direction = -1                     # 只在向下穿越时触发
```

- `direction=-1`：只在从正变负时触发（温度从高于 $T_{set}$ 降到低于 $T_{set}$）。

### 调用 `solve_ivp`

```python
sol = solve_ivp(
    rhs,                    # ODE 右端函数
    [t_current, t_end],     # 积分区间：从当前时间到结束
    [T_current],            # 初始值：当前温度
    events=events,          # 事件列表
    max_step=DT,            # 最大时间步长 (0.01 min)
    dense_output=True       # 生成连续插值解
)
```

**`solve_ivp` 内部做了什么？**

1. 使用 **RK45**（Runge-Kutta 4(5) 阶方法，即 Dormand-Prince 方法）进行数值积分
2. 自适应步长控制：自动选择时间步长以满足误差容限
3. `max_step=0.01`：限制最大步长，确保输出足够平滑
4. **Event detection**：每一步之后检查事件函数是否变号。如果变号了，使用**二分法 + 插值**精确定位过零时刻（精度约 $10^{-12}$）
5. `terminal=True` 时，在事件点立即停止，返回结果

**`sol` 返回对象包含：**
- `sol.t`：时间点数组
- `sol.y`：对应的温度值数组（shape = `[1, n_points]`）
- `sol.t_events[0]`：事件触发的精确时刻（如果有）

### 收集结果

```python
# Store results (skip first point to avoid duplicates)
t_all.extend(sol.t[1:].tolist())
T_all.extend(sol.y[0, 1:].tolist())
heater_all.extend([int(heater_on)] * (len(sol.t) - 1))
```

- 跳过 `sol.t[0]`（即 `t_current`），因为上一段末尾已经存了这个点
- 记录本段所有点的 heater 状态（本段内不变）

### 更新状态并切换

```python
t_current = sol.t[-1]      # 更新时间到本段末尾
T_current = sol.y[0, -1]   # 更新温度到本段末尾

if sol.t_events[0].size > 0:   # 如果事件触发了
    heater_on = not heater_on   # 翻转 heater 状态
    n_switches += 1             # 切换计数 +1
```

然后回到 `while` 循环开头，用新状态继续积分。

### 返回值

```python
return np.array(t_all), np.array(T_all), np.array(heater_all)
```

三个数组：时间、温度、heater 状态（0/1），长度相同。

---

## 2. `solve_bang_bang_hysteresis()` — 带滞回 Bang-Bang 求解器

与 `solve_bang_bang()` 逻辑**完全相同**，唯一区别是**切换阈值不同**：

| | 无滞回 | 有滞回 |
|--|--------|--------|
| ON → OFF 的触发条件 | $T$ 升到 $T_{set}$ | $T$ 升到 $T_{set} + \delta$ |
| OFF → ON 的触发条件 | $T$ 降到 $T_{set}$ | $T$ 降到 $T_{set} - \delta$ |

```python
T_low = T_set - delta    # = 19.5°C  (OFF→ON 的阈值)
T_high = T_set + delta   # = 20.5°C  (ON→OFF 的阈值)
```

**为什么滞回能消除 Zeno？**

无滞回时：`T_on = T_off = T_set`，温度到达后两个方向的事件函数共享同一个零点，导致无限次切换。

有滞回时：`T_on = 19.5 < T_off = 20.5`，heater 打开后温度必须从 19.5 升到 20.5（跨越 1°C），这需要有限时间。反之亦然。每次切换之间有**最小时间间隔**，所以有限时间内只有有限次切换。

---

## 3. `steady_state_temperature()` — 稳态分析

```python
def steady_state_temperature(k=K_COOL, T_a=T_AMBIENT, U_max=U_MAX):
    return T_a + U_max / k
```

推导：设 $dT/dt = 0$（稳态），heater 常开：

$$0 = -k(T_{ss} - T_a) + U_{max} \implies T_{ss} = T_a + \frac{U_{max}}{k}$$

默认参数：$T_{ss} = 5 + 15/0.1 = 155°C$

这个值的意义：**加热器在理论上能把房间加热到 155°C**（如果不关）。因为 $T_{ss} \gg T_{set} = 20°C$，说明加热器功率充足，恒温器能有效控制温度。

如果 $T_{ss} < T_{set}$，说明加热器太弱，永远达不到设定温度。

---

## 4. `oscillation_period_estimate()` — 振荡周期解析公式

```python
def oscillation_period_estimate(k, T_a, T_set, U_max, delta):
    T_low = T_set - delta       # 19.5°C
    T_high = T_set + delta      # 20.5°C
    T_ss = T_a + U_max / k      # 155°C

    if T_ss <= T_high:
        return float('inf')     # 加热器太弱，永远到不了上阈值

    t_heat = (1/k) * ln((T_ss - T_low) / (T_ss - T_high))
    t_cool = (1/k) * ln((T_high - T_a) / (T_low - T_a))

    return t_heat + t_cool
```

### 推导过程

**加热阶段**（从 $T_{low}$ 到 $T_{high}$）：

ODE：$dT/dt = -k(T - T_a) + U_{max} = -k(T - T_{ss})$

这是标准的一阶线性 ODE，解为：

$$T(t) = T_{ss} + (T_{low} - T_{ss})\,e^{-kt}$$

当 $T(t_{heat}) = T_{high}$ 时：

$$T_{high} = T_{ss} + (T_{low} - T_{ss})\,e^{-k\,t_{heat}}$$

$$e^{-k\,t_{heat}} = \frac{T_{high} - T_{ss}}{T_{low} - T_{ss}} = \frac{T_{ss} - T_{high}}{T_{ss} - T_{low}} \cdot (-1/(-1))$$

$$t_{heat} = \frac{1}{k}\ln\left(\frac{T_{ss} - T_{low}}{T_{ss} - T_{high}}\right)$$

**冷却阶段**（从 $T_{high}$ 到 $T_{low}$）：

ODE：$dT/dt = -k(T - T_a)$

解为：$T(t) = T_a + (T_{high} - T_a)\,e^{-kt}$

当 $T(t_{cool}) = T_{low}$ 时：

$$t_{cool} = \frac{1}{k}\ln\left(\frac{T_{high} - T_a}{T_{low} - T_a}\right)$$

**默认参数下的数值**：

$$t_{heat} = \frac{1}{0.1}\ln\left(\frac{155 - 19.5}{155 - 20.5}\right) = 10\ln\left(\frac{135.5}{134.5}\right) = 10 \times 0.00742 = 0.074 \text{ min}$$

$$t_{cool} = \frac{1}{0.1}\ln\left(\frac{20.5 - 5}{19.5 - 5}\right) = 10\ln\left(\frac{15.5}{14.5}\right) = 10 \times 0.0668 = 0.668 \text{ min}$$

$$P = 0.074 + 0.668 = 0.742 \text{ min}$$

**注意**：加热阶段（0.074 min）远短于冷却阶段（0.668 min），因为 $U_{max}$ 很大（加热器很强），但冷却仅靠自然散热。

---

## 5. `plot_results()` — 可视化

```python
def plot_results(t, T, heater, title, T_set, T_a, delta=None, save_path=None):
```

创建一个**双子图**（upper + lower）：

**上图**（height ratio = 3）：
- 蓝色实线：$T(t)$ 温度曲线
- 红色虚线：$T_{set}$ 设定温度
- 青色点线：$T_a$ 室外温度
- 橙色区域：滞回带 $[T_{set}-\delta, T_{set}+\delta]$（如果有）

**下图**（height ratio = 1）：
- 红色填充区域：heater ON 的时段
- `step='post'`：使用阶梯图，每个数据点之后保持值不变直到下一个点

---

## 6. `plot_parameter_sensitivity()` — 参数敏感性图

```python
def plot_parameter_sensitivity(param_name, param_values, results_list, T_set, save_path):
```

把多组不同参数的 $T(t)$ 曲线画在同一张图上，用于比较某个参数变化对系统行为的影响。

---

## 7. `__main__` — 主程序（5 组实验）

| 实验 | 变化的参数 | 目的 |
|------|-----------|------|
| Exp 1 | 无滞回 | 演示 Zeno 效应 |
| Exp 2 | $\delta=0.5$ | 展示滞回如何消除 Zeno |
| Exp 3 | $k \in \{0.05, 0.1, 0.15, 0.2, 0.3\}$ | 隔热质量对切换频率的影响 |
| Exp 4 | $U_{max} \in \{5, 10, 15, 20, 30\}$ | 加热器功率的影响 |
| Exp 5 | $\delta \in \{0, 0.25, 0.5, 1, 2\}$ | 滞回带宽度的舒适度-效率 trade-off |

---

## 数值方法总结（向老师讲解用）

| 方面 | 方法 | 原因 |
|------|------|------|
| ODE 求解器 | `solve_ivp` (RK45, Dormand-Prince) | 自适应步长，4(5) 阶精度，工业标准 |
| 不连续处理 | 分段积分 + event detection | 避免 solver 在切换点失精度 |
| 事件定位 | 二分法 + Hermite 插值 | 精确到 $\sim 10^{-12}$ 的切换时刻 |
| Zeno 保护 | `max_switches=500` | 防止无滞回时无限循环 |
| 步长控制 | `max_step=0.01` min | 保证输出分辨率 + 不错过快速变化 |

### English summary for supervisor

> We solve the hybrid ODE by **piecewise integration**: within each segment the heater state is fixed, so the ODE is smooth and solved by the Dormand-Prince RK4(5) method (`scipy.integrate.solve_ivp`). Switching times are located precisely using **event detection** — a root-finding procedure applied to the event function $g(T) = T - T_{threshold}$ at each integration step. When the event triggers (`terminal=True`), the solver stops, the heater state is toggled, and integration restarts. A `max_switches` safeguard prevents infinite loops in the Zeno regime (no hysteresis). The analytical oscillation period formula is verified against numerical results to validate the implementation.
