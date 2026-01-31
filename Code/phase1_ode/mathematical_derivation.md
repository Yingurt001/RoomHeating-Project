# Phase 1: Mathematical Derivation — ODE Model

## 1. Problem Setup

Consider a room with:
- A single heater (heat source), controlled by a thermostat
- The room loses heat to the outside environment through its walls
- We model the room temperature as spatially uniform (well-mixed assumption)

**Variables:**
- $T(t)$: room temperature at time $t$ (°C)
- $T_a$: ambient outside temperature (°C), assumed constant
- $u(t)$: heat input rate from the heater (°C/min)
- $k$: cooling constant (1/min), characterises heat loss rate

## 2. Newton's Law of Cooling

The rate of heat loss from the room to the outside is proportional to the temperature difference:

$$\text{Heat loss rate} = k \cdot (T(t) - T_a)$$

This gives Newton's Law of Cooling:

$$\frac{dT}{dt} = -k(T(t) - T_a)$$

**Physical meaning of $k$:**
- Large $k$: poor insulation, room cools quickly
- Small $k$: good insulation, room retains heat
- $k$ depends on wall material, thickness, surface area, room volume
- More precisely: $k = \frac{hA}{mc}$, where $h$ = heat transfer coefficient, $A$ = wall surface area, $m$ = air mass, $c$ = specific heat capacity

## 3. Adding the Heater

When the heater provides heat at rate $u(t)$, the full ODE becomes:

$$\boxed{\frac{dT}{dt} = -k(T(t) - T_a) + u(t)}$$

This can be rewritten as:

$$\frac{dT}{dt} = -kT + (kT_a + u(t))$$

which is a first-order linear ODE with a forcing term.

## 4. Thermostat Control Strategies

### 4.1 Constant Heating (Open-Loop)

If $u(t) = U_{max}$ (heater always on):

$$\frac{dT}{dt} = -k(T - T_a) + U_{max}$$

**Steady state** ($dT/dt = 0$):

$$T_{ss} = T_a + \frac{U_{max}}{k}$$

**Analytical solution** (with initial condition $T(0) = T_0$):

$$T(t) = T_{ss} + (T_0 - T_{ss}) e^{-kt}$$

The room exponentially approaches $T_{ss}$ with time constant $\tau = 1/k$.

> **Example:** $T_a = 5°C$, $U_{max} = 15$, $k = 0.1$ gives $T_{ss} = 5 + 150 = 155°C$.
> This is unrealistically hot — in practice, the heater shuts off before reaching this.
> This is why we need thermostat control.

### 4.2 Bang-Bang Control (Closed-Loop, No Hysteresis)

The simplest feedback control: the thermostat measures $T(t)$ and switches the heater ON/OFF:

$$u(T) = \begin{cases} U_{max} & \text{if } T < T_{set} \\ 0 & \text{if } T \geq T_{set} \end{cases}$$

This creates a **discontinuous** right-hand side in the ODE. The system is a **hybrid dynamical system** — it combines continuous dynamics (temperature evolution) with discrete events (heater switching).

**Behaviour:**
1. Starting from $T_0 < T_{set}$: heater ON, temperature rises
2. $T$ reaches $T_{set}$: heater switches OFF
3. Temperature falls due to cooling: $dT/dt = -k(T - T_a) < 0$
4. $T$ drops below $T_{set}$: heater switches ON again
5. Cycle repeats → oscillation around $T_{set}$

**Problem — Zeno effect:**
At the exact switching point $T = T_{set}$, the system may exhibit infinitely rapid switching. This is the **Zeno phenomenon** in hybrid systems theory. In the pure mathematical model (no hysteresis), the system reaches $T_{set}$ and then:
- With heater ON: $dT/dt = -k(T_{set} - T_a) + U_{max} > 0$ (temperature would rise above $T_{set}$)
- With heater OFF: $dT/dt = -k(T_{set} - T_a) < 0$ (temperature would fall below $T_{set}$)

This creates a **sliding mode** along $T = T_{set}$. In practice, the system oscillates with infinitesimally small amplitude — the Zeno effect.

### 4.3 Bang-Bang Control with Hysteresis

To prevent chattering, introduce a **dead band** $[T_{set} - \delta, T_{set} + \delta]$:

$$\text{Heater} \begin{cases} \text{turns ON} & \text{when } T \text{ falls below } T_{set} - \delta \\ \text{turns OFF} & \text{when } T \text{ rises above } T_{set} + \delta \\ \text{stays in current state} & \text{when } T_{set} - \delta \leq T \leq T_{set} + \delta \end{cases}$$

This ensures a minimum ON/OFF duration and eliminates Zeno behaviour.

## 5. Analytical Solution for Oscillation Period (with Hysteresis)

Let $T_{low} = T_{set} - \delta$, $T_{high} = T_{set} + \delta$, $T_{ss} = T_a + U_{max}/k$.

### Heating Phase (Heater ON, from $T_{low}$ to $T_{high}$)

$$\frac{dT}{dt} = -k(T - T_a) + U_{max} = -k(T - T_{ss})$$

Solution: $T(t) = T_{ss} + (T_{low} - T_{ss}) e^{-kt}$

Time to reach $T_{high}$:

$$\boxed{t_{heat} = \frac{1}{k} \ln\left(\frac{T_{ss} - T_{low}}{T_{ss} - T_{high}}\right)}$$

This requires $T_{ss} > T_{high}$, i.e., the heater must be powerful enough to heat the room above the upper threshold.

### Cooling Phase (Heater OFF, from $T_{high}$ to $T_{low}$)

$$\frac{dT}{dt} = -k(T - T_a)$$

Solution: $T(t) = T_a + (T_{high} - T_a) e^{-kt}$

Time to reach $T_{low}$:

$$\boxed{t_{cool} = \frac{1}{k} \ln\left(\frac{T_{high} - T_a}{T_{low} - T_a}\right)}$$

### Oscillation Period

$$\boxed{P = t_{heat} + t_{cool} = \frac{1}{k}\left[\ln\left(\frac{T_{ss} - T_{low}}{T_{ss} - T_{high}}\right) + \ln\left(\frac{T_{high} - T_a}{T_{low} - T_a}\right)\right]}$$

### Energy Consumption (per cycle)

The heater is ON for duration $t_{heat}$ per cycle, so:

$$\text{Energy per cycle} = U_{max} \cdot t_{heat}$$

$$\text{Duty cycle} = \frac{t_{heat}}{P}$$

## 6. Equilibrium and Stability Analysis

### Heater always ON

The equilibrium $T_{ss} = T_a + U_{max}/k$ is **globally asymptotically stable** (eigenvalue = $-k < 0$).

### Heater always OFF

The equilibrium $T = T_a$ is **globally asymptotically stable** (same reasoning).

### Bang-Bang system

The system has no classical equilibrium — instead it exhibits a **limit cycle** (periodic oscillation) around $T_{set}$. This limit cycle is:
- **Stable**: perturbations decay and the system returns to the same oscillation
- **Unique** (for given parameters)
- **Amplitude** = $2\delta$ (determined by hysteresis band)
- **Period** = $P$ as derived above

## 7. Numerical Considerations

- The ODE is **stiff near switching points** — use event detection in `solve_ivp`
- Without hysteresis ($\delta = 0$), numerical solvers will produce very rapid oscillations depending on step size — this is the numerical manifestation of the Zeno effect
- With hysteresis ($\delta > 0$), the problem is well-posed and easy to solve numerically

## 8. Parameter Values and Physical Justification

| Parameter | Symbol | Value | Justification |
|-----------|--------|-------|---------------|
| Outside temperature | $T_a$ | 5°C | UK winter average |
| Initial room temp | $T_0$ | 10°C | Unheated room in winter |
| Set-point | $T_{set}$ | 20°C | Comfortable room temperature |
| Heater power | $U_{max}$ | 15 °C/min | Normalised; typical radiator can heat a room ~1°C/min in practice, but our $k$ is also normalised |
| Cooling constant | $k$ | 0.1 /min | Time constant $\tau = 10$ min for noticeable cooling |
| Hysteresis band | $\delta$ | 0.5°C | Typical thermostat dead band |

> **Note:** These are simplified/normalised parameters. In a real model, one would use:
> $k = hA/(mc)$ with $h \approx 5-25$ W/(m²·K), $A$ = wall area, $m$ = air mass, $c = 1005$ J/(kg·K).
> For a 4m × 4m × 3m room: $m \approx 58$ kg, $A \approx 96$ m², giving $k \approx 0.008-0.04$ /s.
